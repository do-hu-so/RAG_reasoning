"""
Module kết nối LLM (Google Gemini) cho hệ thống RAG.

Module này chịu trách nhiệm:
- Khởi tạo client Gemini bằng SDK google-genai
- Bật chế độ suy luận (thinking) của model theo thinking_level
- Hỗ trợ output JSON theo schema (structured output)
- Đọc cấu hình prompt từ file YAML

Author: Physics Problem Solving System Team
Version: 2.0.0
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

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
