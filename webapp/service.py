"""
Module xử lý nghiệp vụ của bản web.

Module này chịu trách nhiệm:
- Nhận PDF upload -> đọc text -> chia chunk -> embedding -> lưu vào "dataset web"
- Quản lý tài liệu theo từng cuộc trò chuyện
- Hỏi–đáp: phân tích câu hỏi -> tìm kiếm cosine -> suy luận có trích dẫn -> kiểm chứng,
  trả về từng bước (status) để giao diện hiển thị tiến trình
- Tóm tắt tài liệu và gợi ý câu hỏi ngay sau khi upload

Version: 1.0.0
"""

import io
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document

from RAG.gen import GeminiClient, LLMClient, fill_template, load_prompts
from RAG.reasoning import (
    QueryAnalysis,
    QueryAnalyzer,
    ReasonedAnswer,
    ReasoningAnswerer,
    describe_source,
    parse_json,
)
from read_file import TextChunker, read_pdf_pages

from .config import WebSettings
from .embeddings import Embedder, GeminiEmbedder
from .storage import Storage, create_storage

logger = logging.getLogger(__name__)

CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
DOCUMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")

META_FILE = "meta.json"
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
OVERVIEW_FILE = "overview.json"

NO_DOCUMENT_MESSAGE = (
    "Cuộc trò chuyện này chưa có tài liệu nào. Hãy bấm biểu tượng 📎 (hoặc kéo thả) "
    "để tải file PDF lên, sau đó đặt câu hỏi nhé."
)
OUT_OF_SCOPE_MESSAGE = "Mình chỉ trả lời câu hỏi dựa trên các tài liệu PDF bạn đã tải lên trong cuộc trò chuyện này."

OVERVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "questions"],
}


class ServiceError(Exception):
    """
    Lỗi nghiệp vụ kèm mã HTTP để API trả về cho người dùng.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class StoredDocument:
    """
    Tài liệu đã nạp vào bộ nhớ để tìm kiếm.
    """

    meta: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    vectors: np.ndarray


class StageTimer:
    """
    Đo thời gian từng bước xử lý của một câu hỏi (giây).
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.timings: Dict[str, float] = {}
        self._stage: Optional[str] = None
        self._stage_started = self.started

    def elapsed(self) -> float:
        return round(time.perf_counter() - self.started, 3)

    def start(self, stage: str) -> None:
        self._close()
        self._stage = stage
        self._stage_started = time.perf_counter()

    def finish(self) -> Dict[str, float]:
        self._close()
        self.timings["total"] = self.elapsed()
        return dict(self.timings)

    def _close(self) -> None:
        if self._stage:
            self.timings[self._stage] = round(time.perf_counter() - self._stage_started, 3)
            self._stage = None


def describe_error(error: Exception) -> str:
    """
    Chuyển lỗi kỹ thuật thành thông báo dễ hiểu cho người dùng.
    """
    if isinstance(error, ServiceError):
        return error.message

    code = getattr(error, "code", None)
    text = str(error)
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        return "Đã vượt giới hạn gọi Gemini API (quota / rate limit). Vui lòng thử lại sau ít phút."
    if code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
        return "GEMINI_API_KEY không hợp lệ hoặc không có quyền. Kiểm tra lại biến môi trường trên server."
    if code == 404:
        return "Model Gemini không tồn tại hoặc đã ngừng hỗ trợ. Đổi GEMINI_MODEL / GEMINI_EMBEDDING_MODEL rồi deploy lại."
    if code is not None and code >= 500:
        return "Dịch vụ Gemini đang gặp sự cố tạm thời. Vui lòng thử lại."
    return f"Đã xảy ra lỗi: {text}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(filename: str) -> str:
    name = PurePath(filename.replace("\\", "/")).name
    name = re.sub(r'[\x00-\x1f/:*?"<>|#%]', "_", name).strip(" .")[:120]
    if not name.lower().endswith(".pdf"):
        name = f"{name or 'document'}.pdf"
    return name


class WebRAGService:
    """
    Service hỏi–đáp tài liệu PDF cho bản web.

    LLM và embedder được khởi tạo khi cần lần đầu, để các API không dùng đến
    Gemini (liệt kê / xoá tài liệu, health) vẫn chạy khi chưa cấu hình API key.
    """

    def __init__(
        self,
        settings: WebSettings,
        storage: Storage,
        embedder: Optional[Embedder] = None,
        llm: Optional[LLMClient] = None,
        prompts: Optional[Dict[str, str]] = None,
        cache_size: int = 32,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self._embedder = embedder
        self._llm = llm
        self.prompts = prompts or load_prompts(settings.prompt_file)
        self._cache: "OrderedDict[str, StoredDocument]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lazy clients
    @property
    def embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                try:
                    self._embedder = GeminiEmbedder(
                        model=self.settings.embedding_model,
                        dimensions=self.settings.embedding_dimensions,
                    )
                except ValueError as error:
                    raise ServiceError(503, "Server chưa cấu hình GEMINI_API_KEY.") from error
            return self._embedder

    @property
    def llm(self) -> LLMClient:
        with self._lock:
            if self._llm is None:
                try:
                    self._llm = GeminiClient(
                        model=self.settings.llm_model,
                        thinking_level=self.settings.thinking_level,
                    )
                except ValueError as error:
                    raise ServiceError(503, "Server chưa cấu hình GEMINI_API_KEY.") from error
            return self._llm

    # ------------------------------------------------------------------ validation
    @staticmethod
    def check_conversation_id(conversation_id: str) -> None:
        if not CONVERSATION_ID_PATTERN.match(conversation_id or ""):
            raise ServiceError(400, "Mã cuộc trò chuyện không hợp lệ.")

    @staticmethod
    def check_document_id(document_id: str) -> None:
        if not DOCUMENT_ID_PATTERN.match(document_id or ""):
            raise ServiceError(400, "Mã tài liệu không hợp lệ.")

    # ------------------------------------------------------------------ documents
    def list_documents(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Liệt kê tài liệu đã xử lý xong của một cuộc trò chuyện.
        """
        self.check_conversation_id(conversation_id)
        metas = []
        for key in self.storage.list(f"{conversation_id}/"):
            if key.endswith(f"/{META_FILE}"):
                try:
                    metas.append(json.loads(self.storage.get(key)))
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
        return sorted(metas, key=lambda meta: meta.get("created_at", ""))

    def _get_meta(self, conversation_id: str, document_id: str) -> Dict[str, Any]:
        self.check_conversation_id(conversation_id)
        self.check_document_id(document_id)
        try:
            return json.loads(self.storage.get(f"{conversation_id}/{document_id}/{META_FILE}"))
        except FileNotFoundError:
            raise ServiceError(404, "Không tìm thấy tài liệu.") from None

    def add_document(self, conversation_id: str, filename: str, data: bytes) -> Dict[str, Any]:
        """
        Xử lý một file PDF upload: đọc, chia chunk, embedding và lưu lại.

        Returns:
            Thông tin tài liệu (meta)
        """
        self.check_conversation_id(conversation_id)

        if not filename.lower().endswith(".pdf"):
            raise ServiceError(415, "Chỉ hỗ trợ file PDF.")
        if len(data) > self.settings.max_upload_bytes:
            raise ServiceError(413, f"File vượt quá {self.settings.max_upload_mb} MB.")
        if not data.startswith(b"%PDF"):
            raise ServiceError(415, "File không phải PDF hợp lệ.")
        if len(self.list_documents(conversation_id)) >= self.settings.max_documents_per_conversation:
            raise ServiceError(
                409, f"Mỗi cuộc trò chuyện tối đa {self.settings.max_documents_per_conversation} tài liệu."
            )

        try:
            pages = read_pdf_pages(io.BytesIO(data))
        except Exception as error:
            raise ServiceError(422, f"Không đọc được PDF (file hỏng hoặc bị đặt mật khẩu): {error}") from error

        if not pages:
            raise ServiceError(
                422, "PDF không có lớp chữ (có thể là bản scan dạng ảnh). Hãy OCR file trước khi tải lên."
            )

        chunks = TextChunker(self.settings.chunk_size, self.settings.overlap).create_chunks_from_pages(pages)
        if len(chunks) > self.settings.max_chunks_per_document:
            raise ServiceError(413, "Tài liệu quá dài, hãy tách thành nhiều file nhỏ hơn.")

        stored_name = _safe_filename(filename)
        vectors = self.embedder.embed_documents(
            [chunk["content"] for chunk in chunks],
            title=PurePath(stored_name).stem,
        )

        document_id = uuid.uuid4().hex[:12]
        prefix = f"{conversation_id}/{document_id}"
        meta = {
            "id": document_id,
            "filename": filename,
            "stored_name": stored_name,
            "size": len(data),
            "pages": max(page["page_number"] for page in pages),
            "text_pages": len(pages),
            "chunks": len(chunks),
            "embedding_model": self.embedder.model,
            "dimensions": int(vectors.shape[1]),
            "created_at": _now(),
        }

        buffer = io.BytesIO()
        np.save(buffer, vectors.astype(np.float32), allow_pickle=False)

        self.storage.put(f"{prefix}/{stored_name}", data, "application/pdf")
        self.storage.put(f"{prefix}/{CHUNKS_FILE}", json.dumps(chunks, ensure_ascii=False).encode("utf-8"), "application/json")
        self.storage.put(f"{prefix}/{EMBEDDINGS_FILE}", buffer.getvalue(), "application/octet-stream")
        # Ghi meta cuối cùng: có meta nghĩa là tài liệu đã xử lý xong
        self.storage.put(f"{prefix}/{META_FILE}", json.dumps(meta, ensure_ascii=False).encode("utf-8"), "application/json")

        self._remember(prefix, StoredDocument(meta, chunks, vectors))
        return meta

    def get_document_file(self, conversation_id: str, document_id: str) -> Tuple[str, bytes]:
        """
        Lấy file PDF gốc.

        Returns:
            (tên file, nội dung)
        """
        meta = self._get_meta(conversation_id, document_id)
        try:
            data = self.storage.get(f"{conversation_id}/{document_id}/{meta['stored_name']}")
        except FileNotFoundError:
            raise ServiceError(404, "Không tìm thấy file PDF.") from None
        return meta["filename"], data

    def delete_document(self, conversation_id: str, document_id: str) -> None:
        self.check_conversation_id(conversation_id)
        self.check_document_id(document_id)
        prefix = f"{conversation_id}/{document_id}/"
        self.storage.delete(self.storage.list(prefix))
        self._forget(prefix)

    def delete_conversation(self, conversation_id: str) -> None:
        self.check_conversation_id(conversation_id)
        prefix = f"{conversation_id}/"
        self.storage.delete(self.storage.list(prefix))
        self._forget(prefix)

    def overview(self, conversation_id: str, document_id: str) -> Dict[str, Any]:
        """
        Tóm tắt tài liệu và gợi ý câu hỏi (lưu lại để không gọi lại model).
        """
        meta = self._get_meta(conversation_id, document_id)
        key = f"{conversation_id}/{document_id}/{OVERVIEW_FILE}"
        try:
            return json.loads(self.storage.get(key))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        document = self._load_document(meta, conversation_id)
        excerpt, length = [], 0
        for chunk in document.chunks:
            if length > 6000:
                break
            excerpt.append(chunk["content"])
            length += len(chunk["content"])

        raw = self.llm.generate(
            fill_template(self.prompts["overview_user"], filename=meta["filename"], content="\n\n".join(excerpt)),
            system=self.prompts["overview_system"],
            json_schema=OVERVIEW_SCHEMA,
            thinking_level=self.settings.analysis_thinking_level,
        )
        try:
            data = parse_json(raw)
        except ValueError:
            data = {}

        overview = {
            "summary": str(data.get("summary") or "").strip(),
            "questions": [str(q).strip() for q in data.get("questions") or [] if str(q).strip()][:3],
        }
        self.storage.put(key, json.dumps(overview, ensure_ascii=False).encode("utf-8"), "application/json")
        return overview

    # ------------------------------------------------------------------ cache
    def _remember(self, prefix: str, document: StoredDocument) -> None:
        with self._lock:
            self._cache[prefix] = document
            self._cache.move_to_end(prefix)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _forget(self, prefix: str) -> None:
        with self._lock:
            for key in [key for key in self._cache if key.startswith(prefix.rstrip("/"))]:
                del self._cache[key]

    def _load_document(self, meta: Dict[str, Any], conversation_id: str) -> StoredDocument:
        prefix = f"{conversation_id}/{meta['id']}"
        with self._lock:
            if prefix in self._cache:
                self._cache.move_to_end(prefix)
                return self._cache[prefix]

        chunks = json.loads(self.storage.get(f"{prefix}/{CHUNKS_FILE}"))
        vectors = np.load(io.BytesIO(self.storage.get(f"{prefix}/{EMBEDDINGS_FILE}")), allow_pickle=False)
        document = StoredDocument(meta, chunks, vectors)
        self._remember(prefix, document)
        return document

    # ------------------------------------------------------------------ question answering
    def _retrieve(
        self,
        documents: Sequence[StoredDocument],
        queries: Sequence[str],
    ) -> List[Tuple[Document, float]]:
        query_vectors = self.embedder.embed_queries(list(queries))

        candidates: List[Tuple[float, StoredDocument, int]] = []
        for document in documents:
            if document.vectors.shape[1] != query_vectors.shape[1]:
                raise ServiceError(
                    409,
                    f'Tài liệu "{document.meta["filename"]}" được embedding bằng cấu hình khác. '
                    "Hãy xoá và tải lên lại.",
                )
            # Điểm của mỗi đoạn = cosine cao nhất với một trong các truy vấn
            scores = (document.vectors @ query_vectors.T).max(axis=1)
            for index in np.argsort(-scores)[: self.settings.top_n]:
                candidates.append((float(scores[index]), document, int(index)))

        candidates.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, document, index in candidates[: self.settings.top_n]:
            chunk = document.chunks[index]
            results.append((
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": document.meta["filename"],
                        "page_number": chunk.get("page_number"),
                        "page_end": chunk.get("page_end"),
                        "chunk_id": index,
                        "doc_id": document.meta["id"],
                    },
                ),
                score,
            ))
        return results

    @staticmethod
    def _result_event(
        question: str,
        analysis: QueryAnalysis,
        result: ReasonedAnswer,
        timings: Dict[str, float],
        hits: Sequence[Tuple[Document, float]] = (),
    ) -> Dict[str, Any]:
        return {
            "type": "result",
            "timings": timings,
            "question": question,
            "category": analysis.category,
            "standalone_question": analysis.standalone_question,
            "sub_queries": analysis.sub_queries,
            "answer": result.answer,
            "answerable": result.answerable,
            "confidence": result.confidence,
            "reasoning_steps": result.reasoning_steps,
            "warnings": result.warnings,
            "evidence": [asdict(item) for item in result.evidence],
            "sources": [
                {
                    "index": index,
                    "doc_id": document.metadata["doc_id"],
                    "filename": document.metadata["source"],
                    "label": describe_source(document),
                    "page_number": document.metadata["page_number"],
                    "page_end": document.metadata["page_end"],
                    "score": round(score, 4),
                    "content": document.page_content,
                }
                for index, (document, score) in enumerate(hits, start=1)
            ],
        }

    def ask(
        self,
        conversation_id: str,
        question: str,
        history: Sequence[Tuple[str, str]] = (),
    ) -> Iterator[Dict[str, Any]]:
        """
        Trả lời câu hỏi, lần lượt sinh ra các sự kiện:
        {"type": "status", "elapsed": ...} khi bắt đầu mỗi bước và cuối cùng
        {"type": "result", "timings": {...}, ...} kèm thời gian từng bước (giây).
        """
        self.check_conversation_id(conversation_id)
        question = question.strip()
        if not question:
            raise ServiceError(400, "Câu hỏi đang trống.")

        timer = StageTimer()

        def status(stage: str, message: str) -> Dict[str, Any]:
            timer.start(stage)
            return {"type": "status", "stage": stage, "message": message, "elapsed": timer.elapsed()}

        metas = self.list_documents(conversation_id)
        history = list(history)[-self.settings.history_turns:]

        yield status("analyzing", "Đang phân tích câu hỏi")
        analyzer = QueryAnalyzer(self.llm, self.prompts, thinking_level=self.settings.analysis_thinking_level)
        names = "\n".join(f"- {meta['filename']}" for meta in metas) or "(chưa có tài liệu nào)"
        analysis = analyzer.analyze(question, history, documents=names)

        if analysis.category != "document":
            reply = analysis.direct_reply or OUT_OF_SCOPE_MESSAGE
            answer = ReasonedAnswer(answer=reply, answerable=False, confidence="cao")
            yield self._result_event(question, analysis, answer, timer.finish())
            return

        if not metas:
            answer = ReasonedAnswer(answer=NO_DOCUMENT_MESSAGE, answerable=False)
            yield self._result_event(question, analysis, answer, timer.finish())
            return

        yield status("retrieving", f"Đang tìm trong {len(metas)} tài liệu")
        documents = [self._load_document(meta, conversation_id) for meta in metas]
        hits = self._retrieve(documents, analysis.sub_queries or [analysis.standalone_question])

        yield status("reasoning", f"Đang suy luận trên {len(hits)} đoạn trích")
        answerer = ReasoningAnswerer(self.llm, self.prompts)
        result = answerer.answer(analysis.standalone_question, [document for document, _ in hits])

        yield self._result_event(question, analysis, result, timer.finish(), hits)


_service: Optional[WebRAGService] = None
_service_lock = threading.Lock()


def get_service() -> WebRAGService:
    """
    Trả về service dùng chung (khởi tạo lần đầu khi có request).
    """
    global _service
    with _service_lock:
        if _service is None:
            settings = WebSettings()
            try:
                storage = create_storage(settings)
            except RuntimeError as error:
                raise ServiceError(503, str(error)) from error
            _service = WebRAGService(settings, storage)
        return _service


def set_service(service: Optional[WebRAGService]) -> None:
    """
    Thay service dùng chung (dùng khi test).
    """
    global _service
    with _service_lock:
        _service = service
