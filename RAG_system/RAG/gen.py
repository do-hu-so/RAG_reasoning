"""
Module kết nối LLM cho hệ thống RAG.

Module này chịu trách nhiệm:
- Gọi các model của Groq qua API tương thích OpenAI (free tier rộng)
- Tự chuyển sang model kế tiếp khi một model hết quota / lỗi (FallbackLLM);
  Groq tính hạn mức riêng cho từng model nên nhiều model = nhiều quota hơn
- Hỗ trợ output JSON (JSON object mode) và mức suy luận (reasoning_effort)
- Đọc cấu hình prompt từ file YAML

Author: Physics Problem Solving System Team
Version: 4.0.0
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import yaml

from .settings import (
    DEFAULT_LLM_MODELS,
    DEFAULT_PROMPT_FILE,
    DEFAULT_THINKING_LEVEL,
)

logger = logging.getLogger(__name__)

REQUIRED_PROMPTS = (
    "query_analysis_system",
    "query_analysis_user",
    "answer_system",
    "answer_user",
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Mức suy luận -> reasoning_effort của các model gpt-oss
REASONING_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


class LLMClient(Protocol):
    """
    Interface tối thiểu mà pipeline cần ở một LLM.

    Có thể thay bằng LLM khác (hoặc LLM giả lập khi test)
    miễn là cài đặt hàm generate với chữ ký dưới đây.
    """

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        ...


def load_prompts(path: str | Path = DEFAULT_PROMPT_FILE) -> Dict[str, str]:
    """
    Đọc các prompt từ file YAML.

    Args:
        path: Đường dẫn file YAML chứa prompt

    Returns:
        Dict tên prompt -> nội dung

    Raises:
        FileNotFoundError: Khi không tìm thấy file
        ValueError: Khi file thiếu prompt bắt buộc
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        prompts: Dict[str, str] = yaml.safe_load(file) or {}

    missing = [name for name in REQUIRED_PROMPTS if not prompts.get(name)]
    if missing:
        raise ValueError(f"File prompt {path} thiếu các mục: {', '.join(missing)}")

    return prompts


def fill_template(template: str, **values: str) -> str:
    """
    Điền giá trị vào các chỗ {ten} trong template.

    Không dùng str.format để prompt có thể chứa dấu ngoặc nhọn tuỳ ý.
    """
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


class ProviderError(Exception):
    """
    Lỗi trả về từ API của nhà cung cấp LLM / embedding, kèm mã HTTP (code).
    """

    def __init__(self, provider: str, code: int, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(f"{provider} lỗi {code}: {message}")
        self.provider = provider
        self.code = code
        self.retry_after = retry_after


def is_rate_limit_error(error: Exception) -> bool:
    """
    Lỗi do hết quota / gọi quá nhanh (HTTP 429).
    """
    text = str(error)
    return getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


class OpenAICompatibleClient:
    """
    Client cho API chat completions tương thích OpenAI (mặc định Groq).

    Output JSON dùng JSON object mode và mô tả schema trong system prompt.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        http_client: Any = None,
    ) -> None:
        """
        Args:
            name: Tên hiển thị của nhà cung cấp (vd "groq")
            base_url: URL gốc của API, ví dụ https://api.groq.com/openai/v1
            api_key: API key
            model: Tên model
            thinking_level: Mức suy luận mặc định (chỉ áp dụng cho model gpt-oss)
            max_tokens: Số token tối đa của câu trả lời (giữ nhỏ để không vượt giới hạn token/phút)
            timeout: Thời gian chờ mỗi request (giây)
            http_client: httpx.Client có sẵn (dùng khi test)
        """
        import httpx

        self.name = name
        self.model = model
        self.thinking_level = thinking_level
        self.max_tokens = max_tokens
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._http = http_client or httpx.Client(timeout=timeout)
        self._reasoning_supported = "gpt-oss" in model
        self._local = threading.local()

    @property
    def last_usage(self) -> Optional[Dict[str, int]]:
        """
        Số token của lần gọi gần nhất trong luồng hiện tại (prompt_tokens, completion_tokens).
        """
        return getattr(self._local, "last_usage", None)

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self._http.post(self._url, headers=self._headers, json=body)
        if response.status_code >= 400:
            retry_after = None
            try:
                retry_after = float(response.headers.get("retry-after", ""))
            except ValueError:
                pass
            raise ProviderError(f"{self.name}/{self.model}", response.status_code, response.text[:500], retry_after)
        return response.json()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        system_text = system or ""
        if json_schema:
            system_text += (
                "\n\nChỉ trả về DUY NHẤT một object JSON hợp lệ (không kèm chữ nào khác), "
                "đúng JSON schema sau:\n" + json.dumps(json_schema, ensure_ascii=False)
            )

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text.strip()},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
        }
        if json_schema:
            body["response_format"] = {"type": "json_object"}

        effort = REASONING_EFFORT.get((thinking_level or self.thinking_level or "").lower())
        if effort and self._reasoning_supported:
            body["reasoning_effort"] = effort

        try:
            data = self._post(body)
        except ProviderError as error:
            # Model không nhận reasoning_effort -> gọi lại không kèm tham số này
            if error.code == 400 and "reasoning" in str(error).lower() and "reasoning_effort" in body:
                self._reasoning_supported = False
                body.pop("reasoning_effort")
                data = self._post(body)
            else:
                raise

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError(self.name, 502, f"Phản hồi không đúng định dạng: {str(data)[:300]}") from None
        if not text:
            raise ProviderError(self.name, 502, "Model không trả về nội dung")

        usage = data.get("usage") or {}
        self._local.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }
        return text


class FallbackLLM:
    """
    Gọi lần lượt nhiều model: model đầu lỗi (hết quota, lỗi mạng, ...) thì tự chuyển
    sang model kế tiếp.

    - Model vừa hết quota được "nghỉ" theo header retry-after (hoặc 60 giây).
    - Mọi model đều đang chạm giới hạn theo phút (chỉ cần chờ ngắn) thì chờ rồi thử lại
      một lần, thay vì báo lỗi ngay.
    - Ghi lại model đã trả lời (last_used, riêng cho từng luồng xử lý) để hiển thị.
    """

    def __init__(
        self,
        providers: Sequence[Tuple[str, LLMClient]],
        cooldown_seconds: float = 60.0,
        max_wait_seconds: float = 20.0,
    ) -> None:
        if not providers:
            raise ValueError("FallbackLLM cần ít nhất một model")
        self.providers = list(providers)
        self.cooldown_seconds = cooldown_seconds
        self.max_wait_seconds = max_wait_seconds
        self._resting_until: Dict[str, float] = {}
        self._local = threading.local()

    @property
    def names(self) -> List[str]:
        return [name for name, _ in self.providers]

    @property
    def last_used(self) -> Optional[str]:
        """
        Model đã trả lời lần gọi gần nhất trong luồng hiện tại.
        """
        return getattr(self._local, "last_used", None)

    @property
    def last_usage(self) -> Optional[Dict[str, int]]:
        """
        Số token của lần gọi gần nhất trong luồng hiện tại.
        """
        return getattr(self._local, "last_usage", None)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            now = time.time()
            # Ưu tiên model không trong thời gian nghỉ, giữ nguyên thứ tự cấu hình
            ordered = sorted(self.providers, key=lambda item: self._resting_until.get(item[0], 0) > now)
            waits: List[Optional[float]] = []

            for name, client in ordered:
                try:
                    text = client.generate(prompt, system=system, json_schema=json_schema, thinking_level=thinking_level)
                    self._local.last_used = name
                    self._local.last_usage = getattr(client, "last_usage", None)
                    return text
                except Exception as error:
                    last_error = error
                    if is_rate_limit_error(error):
                        rest = getattr(error, "retry_after", None) or self.cooldown_seconds
                        self._resting_until[name] = time.time() + rest
                        waits.append(rest)
                    else:
                        waits.append(None)
                    logger.warning("LLM %s lỗi (%s), chuyển sang model kế tiếp.", name, error)

            # Tất cả đều chỉ bị giới hạn theo phút -> chờ model sớm nhất rồi thử lại một lần
            shortest = min((wait for wait in waits if wait is not None), default=None)
            if attempt == 0 and None not in waits and shortest is not None and shortest <= self.max_wait_seconds:
                time.sleep(shortest)
                continue
            break
        raise last_error


def parse_models(models: Union[str, Sequence[str], None]) -> List[str]:
    """
    "a, b" hoặc ["a", "b"] -> ["a", "b"] (bỏ trống, bỏ trùng, giữ thứ tự).
    """
    if models is None:
        models = DEFAULT_LLM_MODELS
    items = models.split(",") if isinstance(models, str) else list(models)
    result: List[str] = []
    for item in items:
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result


def create_llm(
    models: Union[str, Sequence[str], None] = None,
    thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
    api_key: Optional[str] = None,
) -> LLMClient:
    """
    Tạo LLM dùng Groq.

    Args:
        models: Danh sách model theo thứ tự ưu tiên (mặc định biến GROQ_MODELS hoặc DEFAULT_LLM_MODELS)
        thinking_level: Mức suy luận mặc định
        api_key: API key; mặc định đọc GROQ_API_KEY

    Returns:
        FallbackLLM gồm các model theo thứ tự (kể cả khi chỉ có một model,
        để vẫn có chờ-rồi-thử-lại khi chạm giới hạn theo phút)

    Raises:
        ValueError: Khi chưa có GROQ_API_KEY
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Chưa có GROQ_API_KEY. Lấy key tại https://console.groq.com/keys rồi đặt vào file .env "
            "(hoặc Environment Variables trên Vercel)."
        )

    names = parse_models(models if models is not None else os.getenv("GROQ_MODELS"))
    providers: List[Tuple[str, LLMClient]] = [
        (f"groq/{model}", OpenAICompatibleClient("groq", GROQ_BASE_URL, api_key, model, thinking_level=thinking_level))
        for model in names
    ]
    return FallbackLLM(providers)


def describe_llm(llm: Any) -> List[str]:
    """
    Tên các model của một LLM (để hiển thị), ví dụ ["groq/openai/gpt-oss-120b", "groq/llama-3.3-70b-versatile"].
    """
    if isinstance(llm, FallbackLLM):
        return llm.names
    return [f"{getattr(llm, 'name', 'llm')}/{getattr(llm, 'model', '')}"]
