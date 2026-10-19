"""
Module tạo embedding bằng Gemini Embedding API cho bản web.

Bản web không chạy được model embedding cục bộ (bge-m3 + torch quá lớn so với
giới hạn function của Vercel), nên dùng API của Gemini (đa ngôn ngữ, hỗ trợ tiếng Việt).

Version: 1.0.0
"""

import os
from typing import List, Optional, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """
    Interface tạo embedding mà service cần (có thể thay bằng embedder giả khi test).
    """

    model: str

    def embed_documents(self, texts: Sequence[str], title: Optional[str] = None) -> np.ndarray:
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Chuẩn hoá mỗi vector về độ dài 1 để tích vô hướng = cosine similarity.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class GeminiEmbedder:
    """
    Tạo embedding bằng Gemini (mặc định gemini-embedding-2).

    - gemini-embedding-2: phân biệt tài liệu / câu hỏi bằng tiền tố trong văn bản
      ("title: ... | text: ..." và "task: search result | query: ...")
    - gemini-embedding-001: phân biệt bằng task_type RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
    """

    def __init__(
        self,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
        api_key: Optional[str] = None,
        batch_size: int = 50,
    ) -> None:
        from google import genai
        from google.genai import types

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chưa có GEMINI_API_KEY để tạo embedding.")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3)),
        )

    @property
    def _uses_prefix(self) -> bool:
        return not self.model.startswith(("gemini-embedding-001", "text-embedding"))

    def _config(self, task_type: str):
        if self._uses_prefix:
            return self._types.EmbedContentConfig(output_dimensionality=self.dimensions)
        return self._types.EmbedContentConfig(output_dimensionality=self.dimensions, task_type=task_type)

    def _embed(self, texts: List[str], task_type: str) -> np.ndarray:
        config = self._config(task_type)
        vectors: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # Bọc mỗi đoạn trong một Content để nhận một vector cho mỗi đoạn
            contents = [self._types.Content(parts=[self._types.Part(text=text)]) for text in batch]
            response = self._client.models.embed_content(model=self.model, contents=contents, config=config)
            embeddings = response.embeddings or []

            if len(embeddings) != len(batch):
                # Phòng trường hợp model gộp nhiều input thành một vector: gọi từng đoạn
                embeddings = [
                    self._client.models.embed_content(model=self.model, contents=text, config=config).embeddings[0]
                    for text in batch
                ]
            vectors.extend(embedding.values for embedding in embeddings)

        return normalize_rows(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str], title: Optional[str] = None) -> np.ndarray:
        """
        Tạo embedding cho các đoạn tài liệu.

        Args:
            texts: Nội dung các đoạn
            title: Tên tài liệu (giúp embedding hiểu ngữ cảnh)

        Returns:
            Ma trận float32 (số đoạn x số chiều), mỗi dòng đã chuẩn hoá
        """
        if self._uses_prefix:
            texts = [f"title: {title or 'none'} | text: {text}" for text in texts]
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """
        Tạo embedding cho các câu truy vấn.
        """
        if self._uses_prefix:
            texts = [f"task: search result | query: {text}" for text in texts]
        return self._embed(list(texts), "RETRIEVAL_QUERY")
