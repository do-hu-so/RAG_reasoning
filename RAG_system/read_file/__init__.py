"""
Module đọc file cho hệ thống.

Module này chịu trách nhiệm:
- Đọc file PDF và trích xuất nội dung
- Quản lý đường dẫn file
- Cung cấp interface để truy cập dữ liệu từ file
- Xử lý encoding và format dữ liệu

Author: Physics Problem Solving System Team
Version: 2.0.0
"""

import sys
import os

# thêm path thủ công 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import List, Dict, Any

from .read_pdf import (
    PDFReader,
    TextChunker,
    PDFToDataset,
    process_pdf_to_dataframe,
    process_text_to_dataframe,
    read_pdf_pages,
)


class Get_Path:
    """
    Class cơ sở để quản lý đường dẫn file.
    
    Class này chịu trách nhiệm:
    - Lưu trữ đường dẫn file
    - Kiểm tra sự tồn tại của file
    - Cung cấp interface cơ sở cho các class đọc file
    """
    
    def __init__(self, path : str) -> None:
        """
        Khởi tạo Get_Path với đường dẫn file.
        
        Args:
            path: Đường dẫn đầu vào
        """
        self.path = path

    def exists(self) -> bool:
        """
        Kiểm tra file có tồn tại hay không.

        Returns:
            True nếu file tồn tại, ngược lại False
        """
        return os.path.isfile(self.path)


class Read_File_PDF(Get_Path):
    """
    Class đọc file PDF.

    Class này chịu trách nhiệm:
    - Đọc và trích xuất text từ file PDF
    - Chia nhỏ nội dung thành các đoạn (chunk) phục vụ embedding
    - Trả về DataFrame sẵn sàng cho vector database
    - Kế thừa từ Get_Path để quản lý đường dẫn
    """

    def __init__(self, path : str, chunk_size : int = 500, overlap : int = 100) -> None:
        """
        Khởi tạo Read_File_PDF với đường dẫn file PDF.

        Args:
            path: Đường dẫn đến file pdf
            chunk_size: Số ký tự tối đa trong mỗi chunk
            overlap: Số ký tự chồng lấp giữa các chunk
        """
        super().__init__(path)
        self.chunk_size : int = chunk_size
        self.overlap : int = overlap

    @property
    def Read(self) -> pd.DataFrame:
        """
        Đọc file PDF và trả về DataFrame các chunk.

        Returns:
            pandas.DataFrame với các cột: chunk_id, content, page_number, page_end
        """
        return process_pdf_to_dataframe(
            self.path,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

    @property
    def Read_Pages(self) -> List[Dict[str, Any]]:
        """
        Đọc file PDF và trả về nội dung theo từng trang.

        Returns:
            List các dict chứa page_number và content
        """
        return PDFReader(self.path).extract_text_by_pages()

    @property
    def Read_Text(self) -> str:
        """
        Đọc file PDF và trả về toàn bộ text thô.

        Returns:
            String chứa toàn bộ nội dung PDF
        """
        return PDFReader(self.path).extract_text()


class Read_File_Text(Get_Path):
    """
    Class đọc file văn bản (.md / .txt).

    Dùng cho tài liệu bổ sung, ví dụ nội dung bảng / hình trong PDF
    đã được chép lại thành chữ (PDF chỉ chứa chúng dưới dạng ảnh).
    """

    def __init__(self, path : str, chunk_size : int = 500, overlap : int = 100) -> None:
        """
        Khởi tạo Read_File_Text với đường dẫn file văn bản.

        Args:
            path: Đường dẫn đến file .md / .txt (UTF-8)
            chunk_size: Số ký tự tối đa trong mỗi chunk
            overlap: Số ký tự chồng lấp giữa các chunk
        """
        super().__init__(path)
        self.chunk_size : int = chunk_size
        self.overlap : int = overlap

    @property
    def Read(self) -> pd.DataFrame:
        """
        Đọc file văn bản và trả về DataFrame các chunk.

        Returns:
            pandas.DataFrame với các cột: chunk_id, content, page_number, page_end
        """
        return process_text_to_dataframe(
            self.path,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
