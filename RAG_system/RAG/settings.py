"""
Module cấu hình tập trung cho hệ thống RAG.

Module này chịu trách nhiệm:
- Khai báo đường dẫn mặc định (dữ liệu, index, file prompt)
- Khai báo model embedding / reranker / LLM và tham số tìm kiếm
- Cho phép ghi đè mọi giá trị bằng biến môi trường hoặc file .env ở gốc project

Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SYSTEM_DIR = PROJECT_ROOT / "RAG_system"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"
DEFAULT_INDEX_DIR = RAG_SYSTEM_DIR / "faiss_index"
DEFAULT_PROMPT_FILE = RAG_SYSTEM_DIR / "config_prompt_system.yaml"

# Model embedding / reranker đa ngôn ngữ, hỗ trợ tốt tiếng Việt
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Model Gemini và mức suy luận (thinking_level: minimal | low | medium | high | off)
DEFAULT_LLM_MODEL = "gemini-3.5-flash"
DEFAULT_THINKING_LEVEL = "medium"
DEFAULT_ANALYSIS_THINKING_LEVEL = "low"


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass
class RAGSettings:
    """
    Tập hợp cấu hình của pipeline RAG.

    Giá trị mặc định đọc từ biến môi trường tại thời điểm khởi tạo,
    có thể truyền tham số trực tiếp để ghi đè.
    """

    data_dir: Path = field(default_factory=lambda: Path(_env_str("RAG_DATA_DIR", str(DEFAULT_DATA_DIR))))
    index_dir: Path = field(default_factory=lambda: Path(_env_str("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR))))
    prompt_file: Path = DEFAULT_PROMPT_FILE

    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    reranker_model: str = field(default_factory=lambda: _env_str("RERANKER_MODEL", DEFAULT_RERANKER_MODEL))
    llm_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", DEFAULT_LLM_MODEL))
    thinking_level: str = field(default_factory=lambda: _env_str("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL))
    analysis_thinking_level: str = DEFAULT_ANALYSIS_THINKING_LEVEL

    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 500))
    overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 100))
    # Tổng số đoạn đưa vào reranker (reranker chạy CPU khoảng 0.5 giây / đoạn)
    initial_k: int = field(default_factory=lambda: _env_int("RAG_INITIAL_K", 10))
    top_n: int = field(default_factory=lambda: _env_int("RAG_TOP_N", 5))

    # Số lượt hỏi–đáp gần nhất dùng để hiểu câu hỏi nối tiếp
    history_turns: int = 4
