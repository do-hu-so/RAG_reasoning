"""
Module kết nối LLM cho hệ thống RAG.

Module này chịu trách nhiệm:
- Khởi tạo client Gemini bằng SDK google-genai (bật thinking theo thinking_level)
- Kết nối các API tương thích OpenAI (Groq, OpenRouter, ...) có free tier rộng hơn
- Tự chuyển sang nhà cung cấp kế tiếp khi một nhà cung cấp hết quota / lỗi (FallbackLLM)
- Hỗ trợ output JSON theo schema (structured output)
- Đọc cấu hình prompt từ file YAML

Author: Physics Problem Solving System Team
Version: 3.0.0
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import yaml

from .settings import (
    DEFAULT_LLM_MODEL,
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


class LLMClient(Protocol):
    """
    Interface tối thiểu mà pipeline cần ở một LLM.

    Có thể thay Gemini bằng LLM khác (hoặc LLM giả lập khi test)
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


class GeminiClient:
    """
    Client gọi Google Gemini qua SDK google-genai.

    Class này chịu trách nhiệm:
    - Đọc API key từ biến môi trường GEMINI_API_KEY (hoặc GOOGLE_API_KEY)
    - Gọi model với system prompt, thinking_level và JSON schema tuỳ chọn
    - Tự động thử lại khi gặp lỗi tạm thời (429 / 5xx)
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None,
        thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
        max_output_tokens: int = 8192,
        retry_attempts: int = 3,
    ) -> None:
        """
        Khởi tạo GeminiClient.

        Args:
            model: Tên model Gemini
            api_key: API key; mặc định đọc từ GEMINI_API_KEY hoặc GOOGLE_API_KEY
            thinking_level: Mức suy luận mặc định (minimal | low | medium | high | off)
            max_output_tokens: Số token tối đa của câu trả lời
            retry_attempts: Số lần gọi tối đa khi gặp lỗi tạm thời
        """
        from google import genai
        from google.genai import types

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Chưa có API key cho Gemini. Tạo file .env ở thư mục gốc project "
                "với dòng GEMINI_API_KEY=<key của bạn> (xem .env.example)."
            )

        self.model: str = model
        self.thinking_level: Optional[str] = thinking_level
        self.max_output_tokens: int = max_output_tokens
        self._types = types
        self._thinking_supported: bool = True
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=retry_attempts)
            ),
        )

    def _thinking_config(self, thinking_level: Optional[str]):
        level = (thinking_level or self.thinking_level or "").strip().lower()
        if not self._thinking_supported or level in ("", "off", "none"):
            return None
        return self._types.ThinkingConfig(thinking_level=level.upper())

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        """
        Gọi model và trả về nội dung text.

        Args:
            prompt: Nội dung gửi cho model
            system: System prompt
            json_schema: JSON schema bắt buộc output phải tuân theo (tuỳ chọn)
            thinking_level: Ghi đè mức suy luận cho lần gọi này

        Returns:
            Text model trả về (chuỗi JSON nếu có json_schema)

        Raises:
            RuntimeError: Khi model không trả về nội dung (bị chặn, hết token...)
        """
        from google.genai import errors

        def build_config():
            return self._types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self.max_output_tokens,
                thinking_config=self._thinking_config(thinking_level),
                response_mime_type="application/json" if json_schema else None,
                response_json_schema=json_schema,
            )

        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=build_config()
            )
        except errors.ClientError as error:
            # Model cũ (ví dụ dòng 2.5) không nhận thinking_level -> gọi lại không kèm cấu hình suy luận
            if error.code == 400 and "thinking" in str(error).lower() and self._thinking_supported:
                logger.warning("Model %s không hỗ trợ thinking_level, gọi lại không kèm cấu hình suy luận.", self.model)
                self._thinking_supported = False
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=build_config()
                )
            else:
                raise

        text = response.text
        if not text:
            candidates = response.candidates or []
            reason = candidates[0].finish_reason if candidates else response.prompt_feedback
            raise RuntimeError(f"Gemini không trả về nội dung (lý do: {reason})")

        return text


class ProviderError(Exception):
    """
    Lỗi trả về từ API của nhà cung cấp LLM / embedding, kèm mã HTTP (code).
    """

    def __init__(self, provider: str, code: int, message: str) -> None:
        super().__init__(f"{provider} lỗi {code}: {message}")
        self.provider = provider
        self.code = code


def is_rate_limit_error(error: Exception) -> bool:
    """
    Lỗi do hết quota / gọi quá nhanh (HTTP 429).
    """
    text = str(error)
    return getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


# Nhà cung cấp tương thích OpenAI có sẵn cấu hình: tên -> (base URL, biến API key, model mặc định)
OPENAI_COMPATIBLE_PRESETS: Dict[str, Tuple[str, str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
}

# Mức suy luận -> reasoning_effort của các model gpt-oss
REASONING_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


class OpenAICompatibleClient:
    """
    Client cho các API chat completions tương thích OpenAI (Groq, OpenRouter, Cloudflare, ...).

    Output JSON dùng JSON object mode (hỗ trợ trên hầu hết model) và mô tả schema
    trong system prompt.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
        max_tokens: int = 8192,
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
            max_tokens: Số token tối đa của câu trả lời
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

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self._http.post(self._url, headers=self._headers, json=body)
        if response.status_code >= 400:
            raise ProviderError(self.name, response.status_code, response.text[:500])
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
        return text


class FallbackLLM:
    """
    Gọi lần lượt nhiều nhà cung cấp LLM: nhà cung cấp đầu lỗi (hết quota, lỗi mạng, ...)
    thì tự chuyển sang nhà cung cấp kế tiếp.

    Nhà cung cấp vừa hết quota được "nghỉ" một lúc (cooldown) để không tốn request vô ích.
    """

    def __init__(self, providers: Sequence[Tuple[str, LLMClient]], cooldown_seconds: float = 60.0) -> None:
        if not providers:
            raise ValueError("FallbackLLM cần ít nhất một nhà cung cấp")
        self.providers = list(providers)
        self.cooldown_seconds = cooldown_seconds
        self._resting_until: Dict[str, float] = {}

    @property
    def names(self) -> List[str]:
        return [name for name, _ in self.providers]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        now = time.time()
        # Ưu tiên nhà cung cấp không trong thời gian nghỉ, giữ nguyên thứ tự cấu hình
        ordered = sorted(self.providers, key=lambda item: self._resting_until.get(item[0], 0) > now)

        last_error: Optional[Exception] = None
        for name, client in ordered:
            try:
                return client.generate(prompt, system=system, json_schema=json_schema, thinking_level=thinking_level)
            except Exception as error:
                last_error = error
                if is_rate_limit_error(error):
                    self._resting_until[name] = time.time() + self.cooldown_seconds
                logger.warning("LLM %s lỗi (%s), chuyển sang nhà cung cấp kế tiếp.", name, error)
        raise last_error


def create_llm(
    gemini_model: str = DEFAULT_LLM_MODEL,
    thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
) -> LLMClient:
    """
    Tạo LLM theo biến môi trường LLM_PROVIDERS (mặc định "gemini,groq,openai").

    Chỉ dùng các nhà cung cấp đã có API key. Có từ 2 nhà cung cấp trở lên thì
    trả về FallbackLLM để tự chuyển khi một bên hết quota.

    - gemini: GEMINI_API_KEY (hoặc GOOGLE_API_KEY), model GEMINI_MODEL
    - groq:   GROQ_API_KEY, model GROQ_MODEL (mặc định openai/gpt-oss-120b)
    - openai: API tương thích OpenAI bất kỳ (OpenRouter, Cloudflare, ...):
              OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_MODEL

    Raises:
        ValueError: Khi không có nhà cung cấp nào được cấu hình
    """
    order = [
        name.strip().lower()
        for name in (os.getenv("LLM_PROVIDERS") or "gemini,groq,openai").split(",")
        if name.strip()
    ]
    providers: List[Tuple[str, LLMClient]] = []

    for name in order:
        if name == "gemini":
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                providers.append((f"gemini/{gemini_model}", GeminiClient(model=gemini_model, thinking_level=thinking_level)))
        elif name in OPENAI_COMPATIBLE_PRESETS:
            base_url, key_var, default_model = OPENAI_COMPATIBLE_PRESETS[name]
            if os.getenv(key_var):
                model = os.getenv(f"{name.upper()}_MODEL") or default_model
                providers.append((f"{name}/{model}", OpenAICompatibleClient(
                    name, base_url, os.environ[key_var], model, thinking_level=thinking_level)))
        elif name == "openai":
            base_url = os.getenv("OPENAI_COMPAT_BASE_URL")
            api_key = os.getenv("OPENAI_COMPAT_API_KEY")
            model = os.getenv("OPENAI_COMPAT_MODEL")
            if base_url and api_key and model:
                providers.append((f"openai/{model}", OpenAICompatibleClient(
                    "openai", base_url, api_key, model, thinking_level=thinking_level)))
        else:
            logger.warning("Bỏ qua nhà cung cấp LLM không hỗ trợ: %s", name)

    if not providers:
        raise ValueError(
            "Chưa có API key cho LLM. Đặt GEMINI_API_KEY và/hoặc GROQ_API_KEY "
            "trong file .env (hoặc Environment Variables trên Vercel)."
        )
    if len(providers) == 1:
        return providers[0][1]
    return FallbackLLM(providers)


def describe_llm(llm: Any) -> List[str]:
    """
    Tên các nhà cung cấp của một LLM (để hiển thị), ví dụ ["gemini/gemini-3.5-flash", "groq/openai/gpt-oss-120b"].
    """
    if isinstance(llm, FallbackLLM):
        return llm.names
    name = getattr(llm, "name", "gemini")
    return [f"{name}/{getattr(llm, 'model', '')}"]
