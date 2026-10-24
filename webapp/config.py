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
    DEFAULT_LLM_MODELS,
    DEFAULT_THINKING_LEVEL,
    PROJECT_ROOT,
    RAG_SYSTEM_DIR,
)

WEB_PROMPT_FILE = RAG_SYSTEM_DIR / "config_prompt_web.yaml"

# Thư mục / tiền tố lưu file upload: local là thư mục dataset_web/, trên Vercel là Blob "dataset_web/"
DATASET_WEB_NAME = "dataset_web"
LOCAL_DATASET_DIR = PROJECT_ROOT / DATASET_WEB_NAME

DEFAULT_EMBEDDING_MODEL = "@cf/baai/bge-m3"
# Model dùng chung cho cả 4 phương pháp ở chế độ so sánh
DEFAULT_COMPARE_MODEL = "openai/gpt-oss-20b"


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

    # Các model Groq theo thứ tự ưu tiên (hết quota model đầu thì dùng model sau)
    llm_models: str = field(default_factory=lambda: _env_str("GROQ_MODELS", DEFAULT_LLM_MODELS))
    thinking_level: str = field(default_factory=lambda: _env_str("GROQ_REASONING_EFFORT", DEFAULT_THINKING_LEVEL))
    analysis_thinking_level: str = DEFAULT_ANALYSIS_THINKING_LEVEL

    # Chế độ so sánh: cả 4 phương pháp dùng cùng một model (mặc định gpt-oss-20b)
    compare_model: str = field(default_factory=lambda: _env_str("COMPARE_MODEL", DEFAULT_COMPARE_MODEL))
    # Chạm giới hạn token/phút thì chờ Groq cấp lại hạn mức, tối đa bấy nhiêu giây mỗi lần (không đổi model)
    compare_max_wait: int = field(default_factory=lambda: _env_int("COMPARE_MAX_WAIT", 65))
    # Thời gian tối đa của một lượt so sánh (Vercel Hobby dừng hàm sau 300 giây)
    compare_time_budget: int = field(default_factory=lambda: _env_int("COMPARE_TIME_BUDGET", 240))

    # auto: Cloudflare nếu có key, không thì chỉ tìm từ khoá; hoặc cloudflare | none
    embedding_provider: str = field(default_factory=lambda: _env_str("WEB_EMBEDDING_PROVIDER", "auto"))
    embedding_model: str = field(default_factory=lambda: _env_str("CLOUDFLARE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))

    prompt_file: Path = WEB_PROMPT_FILE
    local_dataset_dir: Path = LOCAL_DATASET_DIR
    blob_access: str = field(default_factory=lambda: _env_str("BLOB_ACCESS", "private"))

    chunk_size: int = field(default_factory=lambda: _env_int("WEB_CHUNK_SIZE", 800))
    overlap: int = field(default_factory=lambda: _env_int("WEB_CHUNK_OVERLAP", 150))
    top_n: int = field(default_factory=lambda: _env_int("WEB_TOP_N", 6))
    # Phương pháp 03: số đoạn ứng viên lấy ra trước khi xếp hạng lại còn top_n
    rerank_candidates: int = field(default_factory=lambda: _env_int("WEB_RERANK_CANDIDATES", 15))
    # Phương pháp 04 có xếp hạng lại trước khi suy luận không (mặc định không: đúng nghĩa "RAG + Reasoning")
    reasoning_rerank: bool = field(
        default_factory=lambda: (os.getenv("WEB_REASONING_RERANK") or "").strip().lower() in ("1", "true", "yes")
    )
    history_turns: int = 4

    # Vercel giới hạn body của request ở 4.5 MB
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 4))
    max_documents_per_conversation: int = 10
    max_chunks_per_document: int = 1500

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
