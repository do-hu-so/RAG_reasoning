"""
Package web: API hỏi–đáp tài liệu PDF (RAG + reasoning) chạy trên Vercel.

Dùng lại lõi suy luận trong RAG_system/RAG (reasoning, gen) và phần đọc / chia
chunk PDF trong RAG_system/read_file.

Version: 1.0.0
"""

import sys
from pathlib import Path

# Thêm đường dẫn RAG_system vào sys.path để import RAG và read_file
RAG_SYSTEM_PATH = Path(__file__).resolve().parents[1] / "RAG_system"
if str(RAG_SYSTEM_PATH) not in sys.path:
    sys.path.insert(0, str(RAG_SYSTEM_PATH))
