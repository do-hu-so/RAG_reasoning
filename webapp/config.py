"""
Module cấu hình cho bản web.

Mọi giá trị có thể ghi đè bằng biến môi trường (Vercel: Settings -> Environment
Variables; local: file .env ở gốc project).

Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from RAG.settings import (
    DEFAULT_ANALYSIS_THINKING_LEVEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_THINKING_LEVEL,
    PROJECT_ROOT,
    RAG_SYSTEM_DIR,
)

WEB_PROMPT_FILE = RAG_SYSTEM_DIR / "config_prompt_web.yaml"

# Thư mục / tiền tố lưu file upload: local là thư mục dataset_web/, trên Vercel là Blob "dataset_web/"
DATASET_WEB_NAME = "dataset_web"
LOCAL_DATASET_DIR = PROJECT_ROOT / DATASET_WEB_NAME

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass
class WebSettings:
    """
    Cấu hình của bản web.
    """

    llm_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", DEFAULT_LLM_MODEL))
    thinking_level: str = field(default_factory=lambda: _env_str("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL))
    analysis_thinking_level: str = DEFAULT_ANALYSIS_THINKING_LEVEL

    embedding_model: str = field(default_factory=lambda: _env_str("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    embedding_dimensions: int = field(default_factory=lambda: _env_int("GEMINI_EMBEDDING_DIMENSIONS", 768))

    prompt_file: Path = WEB_PROMPT_FILE
    local_dataset_dir: Path = LOCAL_DATASET_DIR
    blob_access: str = field(default_factory=lambda: _env_str("BLOB_ACCESS", "private"))

    chunk_size: int = field(default_factory=lambda: _env_int("WEB_CHUNK_SIZE", 800))
    overlap: int = field(default_factory=lambda: _env_int("WEB_CHUNK_OVERLAP", 150))
    top_n: int = field(default_factory=lambda: _env_int("WEB_TOP_N", 6))
    history_turns: int = 4

    # Vercel giới hạn body của request ở 4.5 MB
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 4))
    max_documents_per_conversation: int = 10
    max_chunks_per_document: int = 1500

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
