"""
Module đọc file PDF / văn bản cho hệ thống RAG.

Module này chịu trách nhiệm:
- Đọc file PDF và trích xuất text theo từng trang
- Chia nhỏ text thành chunks (tách theo câu / gạch đầu dòng, có overlap)
- Ghi nhận trang bắt đầu / kết thúc của mỗi chunk để trích dẫn nguồn
- Chuẩn bị dữ liệu cho vector database

Author: Physics Problem Solving System Team
Version: 2.0.0
"""

import logging
import re
from pathlib import Path
from typing import BinaryIO, List, Dict, Tuple, Any, Optional

import pandas as pd
from pypdf import PdfReader

# PDF nguồn có bảng xref lỗi nhẹ; pypdf tự sửa được nhưng in rất nhiều cảnh báo
logging.getLogger("pypdf").setLevel(logging.ERROR)


def read_pdf_pages(source: str | Path | BinaryIO) -> List[Dict[str, Any]]:
    """
    Trích xuất text theo từng trang từ đường dẫn hoặc stream PDF (ví dụ io.BytesIO).

    Returns:
        List các dictionary chứa số trang và nội dung (bỏ qua trang không có chữ)
    """
    reader = PdfReader(source)
    pages_data = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:  # Chỉ thêm trang có nội dung
            pages_data.append({
                'page_number': page_num,
                'content': text
            })

    return pages_data


class PDFReader:
    """
    Class đọc và xử lý file PDF.

    Class này chịu trách nhiệm:
    - Đọc nội dung từ file PDF
    - Trích xuất text từ từng trang
    - Kết hợp và làm sạch text
    """

    def __init__(self, pdf_path: str) -> None:
        """
        Khởi tạo PDFReader với đường dẫn file PDF.

        Args:
            pdf_path: Đường dẫn đến file PDF
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")

    def extract_text(self) -> str:
        """
        Trích xuất toàn bộ text từ file PDF.

        Returns:
            String chứa toàn bộ text từ PDF
        """
        return "\n".join(page["content"] for page in self.extract_text_by_pages())

    def extract_text_by_pages(self) -> List[Dict[str, Any]]:
        """
        Trích xuất text theo từng trang.

        Returns:
            List các dictionary chứa số trang và nội dung
        """
        try:
            return read_pdf_pages(str(self.pdf_path))
        except Exception as e:
            raise Exception(f"Lỗi khi đọc PDF theo trang: {str(e)}") from e


class TextChunker:
    """
    Class chia nhỏ text thành các chunks.

    Class này chịu trách nhiệm:
    - Tách text thành các đơn vị nhỏ (câu, gạch đầu dòng, đoạn văn)
    - Gom các đơn vị thành chunk không vượt quá chunk_size ký tự
    - Lặp lại `overlap` ký tự cuối của chunk trước ở đầu chunk sau
    - Cho phép chunk nối qua ranh giới trang (câu bị ngắt giữa 2 trang vẫn liền mạch)
    """

    # Tách sau dấu kết thúc câu, tại dòng trống, hoặc ngay trước gạch đầu dòng
    _UNIT_SPLIT = re.compile(r'(?<=[.!?…])\s+|\n\s*\n|\s+(?=[•・])')

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        """
        Khởi tạo TextChunker với kích thước chunk.

        Args:
            chunk_size: Số ký tự tối đa trong mỗi chunk
            overlap: Số ký tự chồng lấp giữa các chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap phải nằm trong khoảng [0, chunk_size)")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_by_sentences(self, text: str) -> List[str]:
        """
        Chia text thành các đơn vị (câu / gạch đầu dòng / đoạn văn).

        Args:
            text: Text cần chia

        Returns:
            List các đơn vị đã chuẩn hoá khoảng trắng
        """
        units = []
        for part in self._UNIT_SPLIT.split(text):
            part = re.sub(r'\s+', ' ', part).strip()
            # Bỏ các mảnh chỉ có ký hiệu (ví dụ một dấu "・" đứng riêng)
            if re.search(r'\w', part):
                units.append(part)
        return units

    def create_chunks(self, text: str) -> List[str]:
        """
        Tạo chunks từ một đoạn text với overlap.

        Args:
            text: Text cần chia thành chunks

        Returns:
            List các chunks
        """
        units = [(unit, None) for unit in self.split_by_sentences(text)]
        return [chunk['content'] for chunk in self._pack(units)]

    def create_chunks_from_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tạo chunks từ danh sách trang, chunk được phép nối qua ranh giới trang.

        Args:
            pages: List dict gồm 'page_number' và 'content'

        Returns:
            List dict gồm 'content', 'page_number' (trang bắt đầu), 'page_end' (trang kết thúc)
        """
        units = [
            (unit, page['page_number'])
            for page in pages
            for unit in self.split_by_sentences(page['content'])
        ]
        return self._pack(units)

    def _pack(self, units: List[Tuple[str, Optional[int]]]) -> List[Dict[str, Any]]:
        """
        Gom các đơn vị thành chunk, giữ overlap và theo dõi số trang.
        """
        chunks: List[Dict[str, Any]] = []
        current, start, end = "", None, None

        for unit, page in units:
            for piece in self._split_long(unit):
                if current and len(current) + 1 + len(piece) > self.chunk_size:
                    chunks.append({'content': current, 'page_number': start, 'page_end': end})

                    # Bắt đầu chunk mới bằng phần đuôi của chunk vừa đóng
                    tail = self._tail(current)
                    if tail and len(tail) + 1 + len(piece) <= self.chunk_size:
                        current, start = tail, end
                    else:
                        current = ""

                if not current:
                    start = page
                current = f"{current} {piece}" if current else piece
                end = page

        if current:
            chunks.append({'content': current, 'page_number': start, 'page_end': end})

        return chunks

    def _split_long(self, unit: str) -> List[str]:
        """
        Cắt một đơn vị dài hơn chunk_size thành nhiều mảnh theo ranh giới từ.
        """
        if len(unit) <= self.chunk_size:
            return [unit]

        pieces, current = [], ""
        for word in unit.split():
            # Từ đơn lẻ quá dài (hiếm gặp) thì cắt cứng
            while len(word) > self.chunk_size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(word[:self.chunk_size])
                word = word[self.chunk_size:]

            if current and len(current) + 1 + len(word) > self.chunk_size:
                pieces.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word

        if current:
            pieces.append(current)
        return pieces

    def _tail(self, text: str) -> str:
        """
        Lấy tối đa `overlap` ký tự cuối của text, bắt đầu tại đầu một từ.
        """
        if self.overlap == 0 or len(text) <= self.overlap:
            return text if self.overlap else ""

        tail = text[-self.overlap:]
        if not text[-self.overlap - 1].isspace():
            # Bỏ phần từ bị cắt dở ở đầu
            space = tail.find(" ")
            tail = tail[space + 1:] if space != -1 else ""
        return tail.strip()


class PDFToDataset:
    """
    Class chuyển đổi PDF thành dataset cho RAG.

    Class này chịu trách nhiệm:
    - Kết hợp PDFReader và TextChunker
    - Tạo DataFrame từ PDF
    - Chuẩn bị dữ liệu cho embedding
    """

    def __init__(self, pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> None:
        """
        Khởi tạo PDFToDataset.

        Args:
            pdf_path: Đường dẫn file PDF
            chunk_size: Kích thước chunk
            overlap: Độ chồng lấp giữa chunks
        """
        self.pdf_reader = PDFReader(pdf_path)
        self.text_chunker = TextChunker(chunk_size, overlap)

    def create_dataset(self) -> pd.DataFrame:
        """
        Tạo dataset từ PDF.

        Returns:
            pandas DataFrame với columns: chunk_id, content, page_number, page_end
        """
        pages_data = self.pdf_reader.extract_text_by_pages()
        chunks = self.text_chunker.create_chunks_from_pages(pages_data)

        return pd.DataFrame(
            [{'chunk_id': chunk_id, **chunk} for chunk_id, chunk in enumerate(chunks)],
            columns=['chunk_id', 'content', 'page_number', 'page_end']
        )


def process_pdf_to_dataframe(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> pd.DataFrame:
    """
    Hàm tiện ích để xử lý PDF thành DataFrame.

    Args:
        pdf_path: Đường dẫn file PDF
        chunk_size: Kích thước chunk
        overlap: Độ chồng lấp

    Returns:
        pandas DataFrame chứa chunks từ PDF
    """
    processor = PDFToDataset(pdf_path, chunk_size, overlap)
    return processor.create_dataset()


def process_text_to_dataframe(text_path: str, chunk_size: int = 500, overlap: int = 100) -> pd.DataFrame:
    """
    Hàm tiện ích để xử lý file văn bản (.md / .txt) thành DataFrame.

    File văn bản không có khái niệm trang nên page_number / page_end là None.

    Args:
        text_path: Đường dẫn file văn bản (UTF-8)
        chunk_size: Kích thước chunk
        overlap: Độ chồng lấp

    Returns:
        pandas DataFrame với columns: chunk_id, content, page_number, page_end
    """
    text = Path(text_path).read_text(encoding="utf-8")
    chunks = TextChunker(chunk_size, overlap).create_chunks(text)

    return pd.DataFrame(
        [
            {'chunk_id': chunk_id, 'content': chunk, 'page_number': None, 'page_end': None}
            for chunk_id, chunk in enumerate(chunks)
        ],
        columns=['chunk_id', 'content', 'page_number', 'page_end']
    )
