"""
Module nạp dữ liệu từ PDF vào vector database.

Module này chịu trách nhiệm:
- Quét và đọc toàn bộ file PDF (và file .md / .txt bổ sung) trong thư mục dữ liệu
- Chia nhỏ nội dung thành các chunk
- Tạo embedding cho từng chunk
- Xây dựng, lưu FAISS vector store và tự build lại khi dữ liệu / cấu hình thay đổi

Author: Physics Problem Solving System Team
Version: 3.0.0
"""

from .add_path import add
add()

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from read_file import Read_File_PDF, Read_File_Text

from .settings import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
)

# Giữ tên cũ để code bên ngoài vẫn import được
DEFAULT_PDF_DIR = DEFAULT_DATA_DIR

SUPPORTED_SUFFIXES = (".pdf", ".md", ".txt")
INDEX_INFO_FILE = "index_info.json"

# Tăng số này khi đổi cách chunk để index cũ tự động bị build lại
CHUNKER_VERSION = 2


class PDF_Ingestor:
    """
    Class nạp dữ liệu từ file PDF vào vector database.

    Class này chịu trách nhiệm:
    - Tìm tất cả file PDF / .md / .txt trong thư mục
    - Đọc và chia nhỏ nội dung
    - Chuyển nội dung thành Document
    - Tạo embedding và lưu vào FAISS
    - Ghi lại "dấu vân tay" dữ liệu để biết khi nào index đã cũ
    """

    def __init__(
        self,
        pdf_dir : str | Path = DEFAULT_PDF_DIR,
        index_dir : str | Path = DEFAULT_INDEX_DIR,
        embedding_model : str = DEFAULT_EMBEDDING_MODEL,
        chunk_size : int = 500,
        overlap : int = 100,
        embeddings : Optional[Embeddings] = None
    ) -> None:
        """
        Khởi tạo PDF_Ingestor.

        Args:
            pdf_dir: Thư mục chứa các file nguồn (PDF, .md, .txt)
            index_dir: Thư mục lưu FAISS index sau khi build
            embedding_model: Tên model dùng để tạo embedding
            chunk_size: Số ký tự tối đa trong mỗi chunk
            overlap: Số ký tự chồng lấp giữa các chunk
            embeddings: Object embedding có sẵn (tuỳ chọn, ví dụ khi test)
        """
        self.pdf_dir : Path = Path(pdf_dir)
        self.index_dir : Path = Path(index_dir)
        self.embedding_model : str = embedding_model
        self.chunk_size : int = chunk_size
        self.overlap : int = overlap
        self._embeddings : Optional[Embeddings] = embeddings

    def list_source_files(self) -> List[Path]:
        """
        Liệt kê toàn bộ file nguồn trong thư mục dữ liệu.

        Returns:
            List đường dẫn các file PDF / .md / .txt
        """
        if not self.pdf_dir.exists():
            raise FileNotFoundError(
                f"Thư mục chứa dữ liệu không tồn tại: {self.pdf_dir}"
            )

        return sorted(
            path for path in self.pdf_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    def _read_file(self, path: Path) -> pd.DataFrame:
        reader_class = Read_File_PDF if path.suffix.lower() == ".pdf" else Read_File_Text
        return reader_class(str(path), chunk_size=self.chunk_size, overlap=self.overlap).Read

    def load_documents(self) -> List[Document]:
        """
        Đọc toàn bộ file nguồn và chuyển thành danh sách Document.

        Mỗi chunk nội dung trở thành một Document, kèm metadata
        về file nguồn, trang bắt đầu / kết thúc và chỉ số chunk.

        Returns:
            List các Document sẵn sàng để embedding
        """
        source_files = self.list_source_files()

        if not source_files:
            raise FileNotFoundError(
                f"Không tìm thấy file PDF / .md / .txt nào trong: {self.pdf_dir}"
            )

        documents : List[Document] = []

        for path in source_files:
            for row in self._read_file(path).to_dict(orient="records"):
                documents.append(
                    Document(
                        page_content=row["content"],
                        metadata={
                            "source": path.name,
                            "page_number": _optional_int(row.get("page_number")),
                            "page_end": _optional_int(row.get("page_end")),
                            "chunk_id": int(row["chunk_id"])
                        }
                    )
                )

        return documents

    def get_embeddings(self) -> Embeddings:
        """
        Khởi tạo (một lần) model embedding.

        Returns:
            Đối tượng Embeddings dùng cho FAISS
        """
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def fingerprint(self) -> Dict[str, Any]:
        """
        Tạo "dấu vân tay" của dữ liệu và cấu hình hiện tại.

        Returns:
            Dict mô tả model, tham số chunk và mã băm từng file nguồn
        """
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunker_version": CHUNKER_VERSION,
            "files": {
                path.relative_to(self.pdf_dir).as_posix(): hashlib.sha1(path.read_bytes()).hexdigest()
                for path in self.list_source_files()
            }
        }

    def is_index_fresh(self) -> bool:
        """
        Kiểm tra index đã lưu có khớp với dữ liệu và cấu hình hiện tại không.
        """
        info_path = self.index_dir / INDEX_INFO_FILE
        if not (self.index_dir / "index.faiss").exists() or not info_path.exists():
            return False

        try:
            saved = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        return saved == self.fingerprint()

    def build_index(self, save : bool = True) -> FAISS:
        """
        Xây dựng FAISS index từ toàn bộ file nguồn.

        Args:
            save: Có lưu index xuống đĩa hay không

        Returns:
            FAISS vector store đã được build
        """
        documents = self.load_documents()

        vector_db = FAISS.from_documents(
            documents,
            self.get_embeddings()
        )

        if save:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            vector_db.save_local(str(self.index_dir))
            (self.index_dir / INDEX_INFO_FILE).write_text(
                json.dumps(self.fingerprint(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        return vector_db

    def load_index(self, auto_rebuild : bool = True) -> FAISS:
        """
        Nạp FAISS index đã lưu; build lại nếu chưa có hoặc đã cũ.

        Args:
            auto_rebuild: Tự build lại khi index không khớp dữ liệu / cấu hình

        Returns:
            FAISS vector store
        """
        if self.is_index_fresh():
            return FAISS.load_local(
                str(self.index_dir),
                self.get_embeddings(),
                allow_dangerous_deserialization=True
            )

        if not auto_rebuild:
            raise RuntimeError(
                f"Index tại {self.index_dir} chưa có hoặc không khớp dữ liệu hiện tại. "
                "Chạy `python run.py` để build lại."
            )

        print("[RAG] Index chưa có hoặc dữ liệu / cấu hình đã thay đổi -> đang build lại index...")
        return self.build_index()


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None or pd.isna(value) else int(value)


def build_vector_db_from_pdf(
    pdf_dir : str | Path = DEFAULT_PDF_DIR,
    index_dir : str | Path = DEFAULT_INDEX_DIR,
    embedding_model : str = DEFAULT_EMBEDDING_MODEL,
    chunk_size : int = 500,
    overlap : int = 100
) -> FAISS:
    """
    Hàm tiện ích để build vector database từ thư mục dữ liệu.

    Args:
        pdf_dir: Thư mục chứa file nguồn
        index_dir: Thư mục lưu index
        embedding_model: Model embedding
        chunk_size: Kích thước chunk
        overlap: Độ chồng lấp

    Returns:
        FAISS vector store
    """
    ingestor = PDF_Ingestor(
        pdf_dir=pdf_dir,
        index_dir=index_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap
    )
    return ingestor.build_index()


if __name__ == "__main__":
    # Chạy: python -m RAG_system.RAG.ingest
    vector_db = build_vector_db_from_pdf()
    print(f"Đã nạp {vector_db.index.ntotal} chunks từ thư mục dữ liệu.")
