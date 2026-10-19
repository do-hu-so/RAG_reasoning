"""
Module tìm kiếm từ khoá (BM25) cho bản web.

Không gọi API nên không tốn quota: upload luôn dùng được ngay cả khi nhà cung cấp
embedding hết quota. Khi có embedding, kết quả BM25 được kết hợp với tìm kiếm
ngữ nghĩa (hybrid search).

Tiếng Việt được tách theo âm tiết, kèm cặp âm tiết liền nhau (bigram) để bắt
các từ ghép như "nhiên liệu", "xe tuần tra".

Version: 1.0.0
"""

import math
import re
import unicodedata
from collections import Counter
from typing import List, Sequence

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Tách văn bản thành âm tiết (chữ thường, chuẩn hoá Unicode) và cặp âm tiết liền nhau.
    """
    words = TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text).lower())
    return words + [f"{first}_{second}" for first, second in zip(words, words[1:])]


class BM25:
    """
    Chấm điểm BM25 cho một tập đoạn văn đã tách từ.
    """

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            documents: Mỗi phần tử là danh sách token của một đoạn văn
            k1, b: Tham số chuẩn của BM25
        """
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokens) for tokens in documents]
        self.lengths = [len(tokens) for tokens in documents]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        count = len(self.term_freqs)
        doc_freq = Counter(term for freqs in self.term_freqs for term in freqs)
        self.idf = {
            term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        """
        Điểm BM25 của từng đoạn văn với câu truy vấn (0 nếu không có từ chung).
        """
        terms = [term for term in set(query_tokens) if term in self.idf]
        results = []
        for freqs, length in zip(self.term_freqs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length) if self.avg_length else self.k1
            score = 0.0
            for term in terms:
                freq = freqs.get(term)
                if freq:
                    score += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
            results.append(score)
        return results
