"""
Module reranking cho hệ thống RAG.

Module này chịu trách nhiệm:
- Lấy kết quả ban đầu từ vector database (hỗ trợ nhiều truy vấn con)
- Sắp xếp lại kết quả bằng cross-encoder để cải thiện độ chính xác
- Kết hợp similarity search và reranking
- Trả về kết quả có độ liên quan cao nhất

Author: Physics Problem Solving System Team
Version: 2.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .settings import DEFAULT_RERANKER_MODEL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker dùng sentence-transformers CrossEncoder.

    Có cùng interface compute_score(pairs) với FlagEmbedding.FlagReranker,
    nên có thể truyền một trong hai vào Reranking.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, max_length: int = 512) -> None:
        """
        Args:
            model_name: Tên model reranker trên HuggingFace
            max_length: Số token tối đa của mỗi cặp (câu hỏi, đoạn văn)
        """
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def compute_score(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        """
        Tính điểm liên quan cho từng cặp [câu hỏi, đoạn văn].
        """
        scores = self._model.predict([tuple(pair) for pair in pairs])
        return np.atleast_1d(scores).astype(float).tolist()


def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoderReranker:
    """
    Khởi tạo reranker mặc định.
    """
    return CrossEncoderReranker(model_name)


def _document_key(document: Document) -> Tuple[Any, ...]:
    metadata = document.metadata
    if metadata.get("chunk_id") is not None:
        return (metadata.get("source"), metadata.get("chunk_id"))
    return (document.page_content,)


class Reranking:
    """
    Class thực hiện reranking cho kết quả tìm kiếm.

    Class này chịu trách nhiệm:
    - Lấy kết quả ban đầu từ vector database
    - Sắp xếp lại kết quả bằng reranker
    - Kết hợp similarity search và reranking
    - Trả về kết quả có độ liên quan cao nhất
    """

    def __init__(self, user_query: str) -> None:
        """
        Initialize Reranking với user query

        Args:
            user_query: Câu hỏi của người dùng (dùng để rerank)
        """
        self.__user_query: str = user_query

    def get_initial_results(
        self,
        vectordb: VectorStore,
        k: int = 10,
        queries: Optional[Sequence[str]] = None,
    ) -> List[Document]:
        """
        Lấy kết quả ban đầu từ vector database.

        Khi có nhiều truy vấn con, mỗi truy vấn lấy k kết quả, gộp lại,
        bỏ trùng và sắp xếp theo khoảng cách nhỏ nhất.

        Args:
            vectordb: Vector database (FAISS, Chroma, ...)
            k: Số lượng documents cần retrieve cho mỗi truy vấn
            queries: Các truy vấn tìm kiếm; mặc định là chính user_query

        Returns:
            List các documents
        """
        best: Dict[Tuple[Any, ...], Tuple[Document, float]] = {}

        for query in queries or [self.__user_query]:
            for document, distance in vectordb.similarity_search_with_score(query, k=k):
                key = _document_key(document)
                if key not in best or distance < best[key][1]:
                    best[key] = (document, float(distance))

        return [document for document, _ in sorted(best.values(), key=lambda item: item[1])]

    def rerank_results(
        self,
        reranker: Any,
        initial_results: List[Document],
        top_n: int = 3
    ) -> List[Tuple[Document, Optional[float]]]:
        """
        Rerank các kết quả bằng reranker.

        Args:
            reranker: Object có hàm compute_score(pairs); None để giữ thứ tự vector search
            initial_results: Danh sách documents ban đầu
            top_n: Số lượng kết quả top cần giữ lại

        Returns:
            List các tuples (document, score) đã được sắp xếp
        """
        if not initial_results:
            return []

        if reranker is None:
            return [(document, None) for document in initial_results[:top_n]]

        pairs: List[List[str]] = [
            [self.__user_query, doc.page_content]
            for doc in initial_results
        ]
        scores = np.atleast_1d(reranker.compute_score(pairs)).astype(float).tolist()

        ranked_results = sorted(
            zip(initial_results, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return ranked_results

    def search_with_reranking(
        self,
        VectorDB: VectorStore,
        reranker: Any,
        initial_k: int = 10,
        top_n: int = 3,
        return_scores: bool = True,
        queries: Optional[Sequence[str]] = None,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[Document, Optional[float]]] | List[Document]:
        """
        Thực hiện search với reranking

        Quy trình:
        1. Lấy kết quả ban đầu từ vector database (cho từng truy vấn)
        2. Sắp xếp lại kết quả bằng reranker
        3. Trả về top_n kết quả có điểm cao nhất

        Args:
            VectorDB: Vector database
            reranker: Reranker (CrossEncoderReranker / FlagReranker) hoặc None
            initial_k: Số documents ban đầu để retrieve cho mỗi truy vấn
            top_n: Số documents sau reranking
            return_scores: Có trả về scores hay không
            queries: Các truy vấn tìm kiếm; mặc định là chính user_query
            max_candidates: Giới hạn tổng số documents đưa vào reranker (reranker chạy CPU khá chậm)

        Returns:
            List documents hoặc list tuples (document, score)
        """
        # Bước 1: Retrieve
        initial_results = self.get_initial_results(vectordb=VectorDB, k=initial_k, queries=queries)
        if max_candidates is not None:
            initial_results = initial_results[:max_candidates]

        # Bước 2: Rerank
        ranked_results = self.rerank_results(reranker, initial_results, top_n=top_n)

        if return_scores:
            return ranked_results
        else:
            return [doc for doc, _ in ranked_results]


def get_information(
    user_query: str,
    VectorDB: VectorStore,
    reranking: Any,
    initial_k: int = 15,
    top_n: int = 5
) -> str:
    """
    Hàm tiện ích lấy nội dung liên quan từ vector database dưới dạng một chuỗi text.

    Args:
        user_query: Câu hỏi của người dùng
        VectorDB: Vector database
        reranking: Reranker hoặc None
        initial_k: Số documents ban đầu để retrieve
        top_n: Số documents sau reranking

    Returns:
        String chứa nội dung liên quan được kết hợp
    """
    documents = Reranking(user_query).search_with_reranking(
        VectorDB,
        reranking,
        initial_k=initial_k,
        top_n=top_n,
        return_scores=False
    )
    return "\n".join(document.page_content for document in documents)
