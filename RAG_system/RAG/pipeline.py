"""
Module pipeline hỏi–đáp hoàn chỉnh của hệ thống RAG.

Quy trình cho mỗi câu hỏi:
1. QueryAnalyzer   : phân loại + viết lại câu hỏi (dựa vào lịch sử) + tách truy vấn con
2. Retrieval       : tìm kiếm FAISS cho từng truy vấn, gộp kết quả
3. Reranking       : cross-encoder chọn ra top_n đoạn liên quan nhất
4. Reasoning       : model trích bằng chứng -> suy luận từng bước -> kết luận có trích dẫn
5. Grounding check : kiểm chứng trích dẫn với tài liệu, gắn cảnh báo nếu cần

Version: 1.0.0
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .gen import GeminiClient, LLMClient, load_prompts
from .ingest import PDF_Ingestor
from .reasoning import (
    OUT_OF_SCOPE_MESSAGE,
    QueryAnalysis,
    QueryAnalyzer,
    ReasonedAnswer,
    ReasoningAnswerer,
)
from .reranking import Reranking, load_reranker
from .settings import RAGSettings


@dataclass
class RAGResponse:
    """
    Toàn bộ kết quả của một lượt hỏi–đáp (để hiển thị hoặc debug).
    """

    question: str
    analysis: QueryAnalysis
    result: ReasonedAnswer
    documents: List[Tuple[Document, Optional[float]]] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.result.answer


class RAGPipeline:
    """
    Class điều phối toàn bộ quá trình hỏi–đáp.

    Class này chịu trách nhiệm:
    - Khởi tạo LLM, vector database, reranker (có thể truyền sẵn để test / tái sử dụng)
    - Thực hiện các bước phân tích -> tìm kiếm -> rerank -> suy luận -> kiểm chứng
    - Lưu lịch sử hội thoại ngắn để hiểu câu hỏi nối tiếp
    """

    def __init__(
        self,
        settings: Optional[RAGSettings] = None,
        llm: Optional[LLMClient] = None,
        vector_db: Optional[VectorStore] = None,
        reranker: Any = None,
        use_reranker: bool = True,
        use_query_analysis: bool = True,
        prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            settings: Cấu hình; mặc định đọc từ biến môi trường / .env
            llm: LLM có hàm generate; mặc định GeminiClient
            vector_db: Vector database có sẵn; mặc định nạp (hoặc build) FAISS index
            reranker: Reranker có sẵn; mặc định tải model theo settings
            use_reranker: Tắt để chạy nhanh hơn trên máy yếu (chỉ dùng vector search)
            use_query_analysis: Tắt để bỏ bước phân tích câu hỏi (ít gọi LLM hơn)
            prompts: Prompt có sẵn; mặc định đọc từ file YAML trong settings
        """
        self.settings = settings or RAGSettings()

        # Khởi tạo LLM trước để báo lỗi thiếu API key ngay, trước khi tải các model lớn
        self.llm: LLMClient = llm or GeminiClient(
            model=self.settings.llm_model,
            thinking_level=self.settings.thinking_level,
        )
        self.prompts = prompts or load_prompts(self.settings.prompt_file)

        self.vector_db = vector_db or PDF_Ingestor(
            pdf_dir=self.settings.data_dir,
            index_dir=self.settings.index_dir,
            embedding_model=self.settings.embedding_model,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.overlap,
        ).load_index()

        if reranker is None and use_reranker:
            reranker = load_reranker(self.settings.reranker_model)
        self.reranker = reranker

        self.analyzer = (
            QueryAnalyzer(self.llm, self.prompts, thinking_level=self.settings.analysis_thinking_level)
            if use_query_analysis else None
        )
        self.answerer = ReasoningAnswerer(self.llm, self.prompts)
        self.history: List[Tuple[str, str]] = []

    def reset(self) -> None:
        """
        Xoá lịch sử hội thoại.
        """
        self.history.clear()

    def retrieve(self, analysis: QueryAnalysis) -> List[Tuple[Document, Optional[float]]]:
        """
        Tìm và rerank các đoạn tài liệu cho câu hỏi đã phân tích.

        Tổng số ứng viên đưa vào reranker giữ ở mức initial_k,
        chia đều cho các truy vấn con.
        """
        queries = analysis.sub_queries or [analysis.standalone_question]
        per_query_k = max(self.settings.top_n, math.ceil(self.settings.initial_k / len(queries)))

        return Reranking(analysis.standalone_question).search_with_reranking(
            self.vector_db,
            self.reranker,
            initial_k=per_query_k,
            top_n=self.settings.top_n,
            return_scores=True,
            queries=queries,
            max_candidates=max(self.settings.initial_k, self.settings.top_n),
        )

    def ask(self, question: str) -> RAGResponse:
        """
        Trả lời một câu hỏi.

        Args:
            question: Câu hỏi của người dùng

        Returns:
            RAGResponse gồm câu trả lời, suy luận, bằng chứng và các đoạn tài liệu đã dùng
        """
        question = question.strip()
        history = self.history[-self.settings.history_turns:]

        if self.analyzer is not None:
            analysis = self.analyzer.analyze(question, history)
        else:
            analysis = QueryAnalysis("document", question, [question])

        if analysis.category != "document":
            reply = analysis.direct_reply or OUT_OF_SCOPE_MESSAGE
            response = RAGResponse(
                question=question,
                analysis=analysis,
                result=ReasonedAnswer(answer=reply, answerable=False, confidence="cao"),
            )
        else:
            documents = self.retrieve(analysis)
            result = self.answerer.answer(
                analysis.standalone_question,
                [document for document, _ in documents],
            )
            response = RAGResponse(question=question, analysis=analysis, result=result, documents=documents)

        self.history.append((question, response.answer))
        return response
