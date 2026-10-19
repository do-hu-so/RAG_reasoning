"""
Module tạo embedding qua API cho bản web.

Bản web không chạy được model embedding cục bộ (bge-m3 + torch quá lớn so với
giới hạn function của Vercel), nên dùng API:
- CloudflareEmbedder: @cf/baai/bge-m3 trên Cloudflare Workers AI (free tier rộng, đa ngôn ngữ)
- GeminiEmbedder: gemini-embedding-2 (free tier hẹp)
Không có embedder nào thì service chỉ dùng tìm kiếm từ khoá (BM25).

Version: 2.0.0
"""

import logging
import os
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np

from RAG.gen import ProviderError

logger = logging.getLogger(__name__)


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


class CloudflareEmbedder:
    """
    Tạo embedding bằng Cloudflare Workers AI (mặc định @cf/baai/bge-m3, 1024 chiều).

    bge-m3 là model đa ngôn ngữ (cùng model với bản dòng lệnh), không cần tiền tố
    phân biệt tài liệu / câu hỏi. Free tier của Workers AI tính theo "neuron" mỗi ngày.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model: str = "@cf/baai/bge-m3",
        batch_size: int = 50,
        timeout: float = 60.0,
        http_client: Any = None,
    ) -> None:
        import httpx

        account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        if not account_id or not api_token:
            raise ValueError("Chưa có CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN để tạo embedding.")

        self.model = model
        self.batch_size = batch_size
        self._url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self._http = http_client or httpx.Client(timeout=timeout)

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self._http.post(self._url, headers=self._headers, json={"text": batch, "truncate_inputs": True})
            if response.status_code >= 400:
                raise ProviderError("cloudflare", response.status_code, response.text[:500])

            result = (response.json() or {}).get("result") or {}
            data = result.get("data") or result.get("response")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ProviderError("cloudflare", 502, f"Phản hồi embedding không đúng định dạng: {str(result)[:200]}")
            vectors.extend(data)

        return normalize_rows(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str], title: Optional[str] = None) -> np.ndarray:
        return self._embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts))


def create_embedder(provider: str, gemini_model: str, dimensions: int) -> Optional[Embedder]:
    """
    Chọn nhà cung cấp embedding.

    Args:
        provider: "auto" (Cloudflare nếu có key, rồi đến Gemini), "cloudflare", "gemini"
                  hoặc "none" (chỉ tìm kiếm từ khoá)

    Returns:
        Embedder, hoặc None khi không dùng embedding
    """
    provider = (provider or "auto").strip().lower()
    has_cloudflare = bool(os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if provider == "none":
        return None
    if provider == "cloudflare" or (provider == "auto" and has_cloudflare):
        return CloudflareEmbedder(model=os.getenv("CLOUDFLARE_EMBEDDING_MODEL") or "@cf/baai/bge-m3")
    if provider == "gemini" or (provider == "auto" and has_gemini):
        return GeminiEmbedder(model=gemini_model, dimensions=dimensions)
    if provider != "auto":
        logger.warning("Nhà cung cấp embedding không hỗ trợ: %s, chỉ dùng tìm kiếm từ khoá.", provider)
    return None
