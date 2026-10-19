"""
Module tạo embedding qua API cho bản web.

Bản web không chạy được model embedding cục bộ (bge-m3 + torch quá lớn so với
giới hạn function của Vercel), nên dùng Cloudflare Workers AI (@cf/baai/bge-m3,
đa ngôn ngữ, free tier rộng). Không cấu hình Cloudflare thì service chỉ dùng
tìm kiếm từ khoá (BM25).

Version: 3.0.0
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
                raise ProviderError("Cloudflare", response.status_code, response.text[:500])

            result = (response.json() or {}).get("result") or {}
            data = result.get("data") or result.get("response")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ProviderError("Cloudflare", 502, f"Phản hồi embedding không đúng định dạng: {str(result)[:200]}")
            vectors.extend(data)

        return normalize_rows(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str], title: Optional[str] = None) -> np.ndarray:
        return self._embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts))


def create_embedder(provider: str, model: str = "@cf/baai/bge-m3") -> Optional[Embedder]:
    """
    Chọn nhà cung cấp embedding.

    Args:
        provider: "auto" (Cloudflare nếu có key), "cloudflare" hoặc "none" (chỉ tìm kiếm từ khoá)
        model: Model embedding của Cloudflare Workers AI

    Returns:
        Embedder, hoặc None khi chỉ dùng tìm kiếm từ khoá
    """
    provider = (provider or "auto").strip().lower()
    has_cloudflare = bool(os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN"))

    if provider == "cloudflare" or (provider == "auto" and has_cloudflare):
        return CloudflareEmbedder(model=model)
    if provider not in ("auto", "none"):
        logger.warning("Nhà cung cấp embedding không hỗ trợ: %s, chỉ dùng tìm kiếm từ khoá.", provider)
    return None
