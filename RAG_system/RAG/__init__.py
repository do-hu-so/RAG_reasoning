"""
Module RAG (Retrieval-Augmented Generation) hỏi–đáp tài liệu có suy luận.

Module này chứa các thành phần chính của hệ thống RAG:
- settings: Cấu hình tập trung (model, đường dẫn, tham số)
- ingest: Nạp dữ liệu từ PDF / .md / .txt vào FAISS (bản dòng lệnh)
- reranking: Tìm kiếm nhiều truy vấn và sắp xếp lại kết quả
- reasoning: Phân tích câu hỏi, suy luận có trích dẫn, kiểm chứng câu trả lời
- gen: Kết nối LLM (Groq) và đọc prompt
- pipeline: Điều phối toàn bộ quá trình hỏi–đáp (bản dòng lệnh)
- add_path: Quản lý đường dẫn

Author: Physics Problem Solving System Team
Version: 2.1.0
"""

# Import theo nhu cầu: bản web chỉ dùng reasoning / gen, không cần tải
# FAISS, langchain-community, sentence-transformers của bản dòng lệnh.
__all__ = ["PDF_Ingestor", "build_vector_db_from_pdf"]


def __getattr__(name):
    if name in __all__:
        from . import ingest
        return getattr(ingest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
