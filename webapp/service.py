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
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document

from RAG.gen import (
    FallbackLLM,
    LLMClient,
    create_llm,
    fill_template,
    is_rate_limit_error,
    load_prompts,
    model_of,
    redact_secrets,
)
from RAG.reasoning import (
    QueryAnalysis,
    QueryAnalyzer,
    ReasonedAnswer,
    ReasoningAnswerer,
    describe_source,
    format_context,
    format_history,
    parse_json,
)
from read_file import TextChunker, read_pdf_pages

from .config import WebSettings
from .embeddings import Embedder, create_embedder
from .keyword_search import BM25, tokenize
from .reranking import LLMReranker
from .storage import Storage, create_storage

logger = logging.getLogger(__name__)

CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
DOCUMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")

META_FILE = "meta.json"
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"

# Hybrid search: gộp thứ hạng BM25 và embedding bằng Reciprocal Rank Fusion
RRF_K = 60
CANDIDATES_PER_RANKING = 30

# Giá trị mặc định của tham số embedder: tự chọn theo biến môi trường
AUTO = object()

# Embedding hết quota thì tạm ngừng thử bổ sung embedding cho tài liệu cũ (giây)
EMBEDDING_REST_SECONDS = 15 * 60
OVERVIEW_FILE = "overview.json"

# 4 phương pháp hỏi–đáp để so sánh / đánh giá
MODES: Dict[str, str] = {
    "llm": "LLM thuần, không RAG",
    "rag": "LLM + RAG thuần",
    "rerank": "LLM + RAG + Reranking",
    "reasoning": "LLM + RAG + Reasoning",
}
COMPARE_ORDER = ("llm", "rag", "rerank", "reasoning")
# Chế độ so sánh dùng một model cố định: số lần tối đa chờ hạn mức Groq rồi thử lại cho mỗi lần gọi
COMPARE_MAX_RETRIES = 4
STAGE_LABELS: Dict[str, str] = {
    "analyzing": "Phân tích câu hỏi",
    "retrieving": "Tìm đoạn liên quan",
    "reranking": "Xếp hạng lại (rerank)",
    "generating": "Sinh câu trả lời",
    "reasoning": "Suy luận & kiểm chứng",
}

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
    vectors: Optional[np.ndarray]  # None: tài liệu chỉ tìm được bằng từ khoá
    tokens: Optional[List[List[str]]] = None  # token BM25 của từng đoạn (tính khi cần)

    def keyword_tokens(self) -> List[List[str]]:
        if self.tokens is None:
            self.tokens = [tokenize(chunk["content"]) for chunk in self.chunks]
        return self.tokens


class StageTimer:
    """
    Đo thời gian từng bước xử lý của một câu hỏi (giây).

    Thời gian chờ Groq cấp lại hạn mức (wait_source: tổng số giây đã chờ) không tính
    vào từng bước và tổng, mà ghi riêng ở "rate_limit_wait".
    """

    def __init__(self, wait_source: Optional[Callable[[], float]] = None) -> None:
        self.started = time.perf_counter()
        self.timings: Dict[str, float] = {}
        self._stage: Optional[str] = None
        self._stage_started = self.started
        self._wait_source = wait_source or (lambda: 0.0)
        self._stage_waited = 0.0

    def elapsed(self) -> float:
        return round(time.perf_counter() - self.started - self._wait_source(), 3)

    def start(self, stage: str) -> None:
        self._close()
        self._stage = stage
        self._stage_started = time.perf_counter()
        self._stage_waited = self._wait_source()

    def finish(self) -> Dict[str, float]:
        self._close()
        self.timings["total"] = self.elapsed()
        waited = self._wait_source()
        if waited:
            self.timings["rate_limit_wait"] = round(waited, 3)
        return dict(self.timings)

    def _close(self) -> None:
        if self._stage:
            waited = self._wait_source() - self._stage_waited
            self.timings[self._stage] = round(time.perf_counter() - self._stage_started - waited, 3)
            self._stage = None


class UsageRecorder:
    """
    Bọc LLM để đếm số lần gọi, số token, model đã trả lời và thời gian chờ hạn mức
    trong một lần chạy (chỉ số dùng để so sánh / đánh giá các phương pháp).
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.models: List[str] = []
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.waited = 0.0

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = None,
    ) -> str:
        try:
            text = self.llm.generate(prompt, system=system, json_schema=json_schema, thinking_level=thinking_level)
        finally:
            # Kể cả khi lỗi (vd. reranking lỗi rồi dùng thứ tự cũ): thời gian chờ vẫn đã trôi qua
            self.waited += float(getattr(self.llm, "last_wait", 0.0) or 0.0)
        self.calls += 1
        model = getattr(self.llm, "last_used", None)
        if model and model not in self.models:
            self.models.append(model)
        usage = getattr(self.llm, "last_usage", None) or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return text

    def summary(self) -> Dict[str, int]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def describe_error(error: Exception) -> str:
    """
    Chuyển lỗi kỹ thuật thành thông báo dễ hiểu cho người dùng.
    """
    if isinstance(error, ServiceError):
        return error.message

    code = getattr(error, "code", None)
    text = str(error)
    provider = getattr(error, "provider", None) or "Groq"
    if code == 429 or "rate limit" in text.lower():
        return (
            "Groq đã hết quota (hoặc gọi quá nhanh) cho tất cả model đang dùng. "
            "Vui lòng thử lại sau ít phút; hạn mức theo ngày của Groq đặt lại lúc 0h UTC (7h sáng giờ Việt Nam)."
        )
    if code == 413:
        return "Câu hỏi kèm tài liệu quá dài so với giới hạn token/phút của Groq. Hãy hỏi ngắn gọn hơn."
    if code in (401, 403):
        return f"API key của {provider} không hợp lệ hoặc không có quyền. Kiểm tra lại biến môi trường trên server."
    if code == 404:
        return f"Model của {provider} không tồn tại hoặc đã ngừng hỗ trợ. Kiểm tra biến GROQ_MODELS rồi deploy lại."
    if code is not None and code >= 500:
        return f"Dịch vụ {provider} đang gặp sự cố tạm thời. Vui lòng thử lại."
    return f"Đã xảy ra lỗi: {redact_secrets(text)}"


def describe_embedding_error(error: Exception) -> str:
    """
    Thông báo khi tạo embedding (Cloudflare) lỗi.
    """
    code = getattr(error, "code", None)
    provider = getattr(error, "provider", None) or "Cloudflare"
    if code == 429:
        return (
            f"{provider} đã hết quota embedding trong ngày; "
            "đặt WEB_EMBEDDING_PROVIDER=none nếu chỉ muốn dùng tìm kiếm từ khoá"
        )
    if code in (401, 403):
        return f"{provider} từ chối truy cập: kiểm tra CLOUDFLARE_ACCOUNT_ID và quyền Workers AI của CLOUDFLARE_API_TOKEN"
    if code is not None and code >= 500:
        return f"dịch vụ embedding của {provider} đang gặp sự cố tạm thời"
    return describe_error(error).rstrip(".")


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
    LLM (liệt kê / xoá tài liệu, health) vẫn chạy khi chưa cấu hình API key.
    """

    def __init__(
        self,
        settings: WebSettings,
        storage: Storage,
        embedder: Any = AUTO,
        llm: Optional[LLMClient] = None,
        prompts: Optional[Dict[str, str]] = None,
        cache_size: int = 32,
    ) -> None:
        """
        Args:
            embedder: Embedder có sẵn; None = chỉ tìm kiếm từ khoá; mặc định chọn theo biến môi trường
            llm: LLM có sẵn; mặc định các model Groq theo biến môi trường GROQ_MODELS
        """
        self.settings = settings
        self.storage = storage
        self._embedder = None if embedder is AUTO else embedder
        self._embedder_ready = embedder is not AUTO
        self._embedding_rest_until = 0.0
        # Lỗi cấu hình embedding (vd. token sai định dạng) để hiển thị trên giao diện
        self.embedding_error: Optional[str] = None
        self._llm = llm
        self.prompts = prompts or load_prompts(settings.prompt_file)
        self._cache: "OrderedDict[str, StoredDocument]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lazy clients
    @property
    def embedder(self) -> Optional[Embedder]:
        """
        Embedder theo cấu hình; None nghĩa là chỉ dùng tìm kiếm từ khoá.
        """
        with self._lock:
            if not self._embedder_ready:
                try:
                    self._embedder = create_embedder(
                        self.settings.embedding_provider,
                        self.settings.embedding_model,
                    )
                except ValueError as error:
                    logger.warning("Không tạo được embedder (%s), chỉ dùng tìm kiếm từ khoá.", redact_secrets(error))
                    self._embedder = None
                    self.embedding_error = redact_secrets(error)
                self._embedder_ready = True
            return self._embedder

    @property
    def llm(self) -> LLMClient:
        with self._lock:
            if self._llm is None:
                try:
                    self._llm = create_llm(
                        models=self.settings.llm_models,
                        thinking_level=self.settings.thinking_level,
                    )
                except ValueError as error:
                    raise ServiceError(503, str(error)) from error
            return self._llm

    def compare_llm(self, deadline: Optional[float] = None) -> LLMClient:
        """
        LLM cho chế độ so sánh: cả 4 phương pháp dùng cùng một model (settings.compare_model;
        chuỗi rỗng thì model đầu tiên của GROQ_MODELS) để so sánh công bằng. Key này hết
        hạn mức thì dùng key Groq khác (vẫn model đó); mọi key đều chạm giới hạn theo
        phút thì chờ Groq cấp lại hạn mức thay vì chuyển sang model khác.

        Args:
            deadline: Mốc thời gian (time.time()) không được chờ vượt quá
        """
        base = self.llm
        if not isinstance(base, FallbackLLM):
            return base  # LLM tự cung cấp (vd. khi test): dùng nguyên

        model = self.settings.compare_model.strip()
        if model.startswith("groq/"):
            model = model[len("groq/"):]
        target = f"groq/{model}" if model else model_of(base.names[0])
        providers = [(name, client) for name, client in base.providers if model_of(name) == target]
        if not providers:
            # Model không nằm trong GROQ_MODELS: tạo riêng (với mọi key)
            try:
                providers = create_llm(models=[model], thinking_level=self.settings.thinking_level).providers
            except ValueError as error:
                raise ServiceError(503, str(error)) from error
        return FallbackLLM(
            providers,
            max_wait_seconds=self.settings.compare_max_wait,
            max_retries=COMPARE_MAX_RETRIES,
            deadline=deadline,
        )

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

        # Embedding là tuỳ chọn: lỗi (hết quota, mất mạng...) thì tài liệu vẫn dùng được
        # với tìm kiếm từ khoá BM25, không làm hỏng cả lần upload
        embedder = self.embedder
        vectors: Optional[np.ndarray] = None
        warning: Optional[str] = None
        if embedder is not None:
            try:
                vectors = embedder.embed_documents(
                    [chunk["content"] for chunk in chunks],
                    title=PurePath(stored_name).stem,
                )
            except Exception as error:
                logger.warning("Không tạo được embedding cho %s: %s", filename, redact_secrets(error))
                warning = (
                    f"Chưa tạo được embedding: {describe_embedding_error(error)}. "
                    "Tài liệu vẫn hỏi được bằng tìm kiếm từ khoá."
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
            "search": "hybrid" if vectors is not None else "keyword",
            "embedding_model": embedder.model if vectors is not None else None,
            "dimensions": int(vectors.shape[1]) if vectors is not None else None,
            "warning": warning,
            "created_at": _now(),
        }

        self.storage.put(f"{prefix}/{stored_name}", data, "application/pdf")
        self.storage.put(f"{prefix}/{CHUNKS_FILE}", json.dumps(chunks, ensure_ascii=False).encode("utf-8"), "application/json")
        if vectors is not None:
            buffer = io.BytesIO()
            np.save(buffer, vectors.astype(np.float32), allow_pickle=False)
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
        vectors = None
        if meta.get("embedding_model"):
            vectors = np.load(io.BytesIO(self.storage.get(f"{prefix}/{EMBEDDINGS_FILE}")), allow_pickle=False)
        document = StoredDocument(meta, chunks, vectors)
        self._remember(prefix, document)
        return document

    def _embed_missing(self, documents: Sequence[StoredDocument], conversation_id: str) -> None:
        """
        Bổ sung embedding cho tài liệu chưa có (hoặc embedding bằng model khác) khi
        embedder dùng được, để tài liệu upload lúc hết quota được nâng lên tìm kiếm hybrid.
        """
        embedder = self.embedder
        if embedder is None or time.time() < self._embedding_rest_until:
            return

        for document in documents:
            if document.vectors is not None and document.meta.get("embedding_model") == embedder.model:
                continue
            try:
                vectors = embedder.embed_documents(
                    [chunk["content"] for chunk in document.chunks],
                    title=PurePath(document.meta["stored_name"]).stem,
                )
            except Exception as error:
                logger.warning("Chưa bổ sung được embedding cho %s: %s", document.meta["filename"], redact_secrets(error))
                if is_rate_limit_error(error):
                    self._embedding_rest_until = time.time() + EMBEDDING_REST_SECONDS
                return

            prefix = f"{conversation_id}/{document.meta['id']}"
            buffer = io.BytesIO()
            np.save(buffer, vectors.astype(np.float32), allow_pickle=False)
            self.storage.put(f"{prefix}/{EMBEDDINGS_FILE}", buffer.getvalue(), "application/octet-stream")

            document.vectors = vectors
            document.meta.update(
                search="hybrid",
                embedding_model=embedder.model,
                dimensions=int(vectors.shape[1]),
                warning=None,
            )
            self.storage.put(
                f"{prefix}/{META_FILE}",
                json.dumps(document.meta, ensure_ascii=False).encode("utf-8"),
                "application/json",
            )

    # ------------------------------------------------------------------ question answering
    def _retrieve(
        self,
        documents: Sequence[StoredDocument],
        queries: Sequence[str],
        limit: Optional[int] = None,
    ) -> Tuple[List[Tuple[Document, float]], List[str], bool]:
        """
        Hybrid search: gộp thứ hạng BM25 (từ khoá) và cosine (embedding) bằng
        Reciprocal Rank Fusion, cho mỗi truy vấn con.

        Args:
            limit: Số đoạn cần lấy (mặc định settings.top_n)

        Returns:
            (các đoạn kèm độ liên quan tương đối 0-1, cảnh báo, có dùng tìm kiếm ngữ nghĩa không)
        """
        limit = limit or self.settings.top_n
        pool = [(document, index) for document in documents for index in range(len(document.chunks))]
        fused: Dict[int, float] = defaultdict(float)
        warnings: List[str] = []
        used_dense = False

        def add_ranking(scores: np.ndarray) -> None:
            for rank, position in enumerate(np.argsort(-scores)[:CANDIDATES_PER_RANKING]):
                if scores[position] <= 0:
                    break
                fused[int(position)] += 1.0 / (RRF_K + rank + 1)

        # 1) Từ khoá (BM25): luôn chạy, không tốn quota
        bm25 = BM25([tokens for document in documents for tokens in document.keyword_tokens()])
        for query in queries:
            add_ranking(np.asarray(bm25.scores(tokenize(query))))

        # 2) Ngữ nghĩa (embedding): chỉ với tài liệu đã có vector cùng model với embedder hiện tại
        embedder = self.embedder
        dense = [
            document for document in documents
            if embedder is not None and document.vectors is not None
            and document.meta.get("embedding_model") == embedder.model
        ]
        if dense:
            try:
                query_vectors = embedder.embed_queries(list(queries))
            except Exception as error:
                logger.warning("Không tạo được embedding cho câu hỏi: %s", redact_secrets(error))
                query_vectors = None
                warnings.append(
                    f"Tìm kiếm ngữ nghĩa tạm thời không dùng được ({describe_embedding_error(error)}); "
                    "câu trả lời dựa trên tìm kiếm từ khoá."
                )
            if query_vectors is not None:
                used_dense = True
                dense_ids = {id(document) for document in dense}
                cosine = np.zeros((len(pool), len(queries)), dtype=np.float32)
                offset = 0
                for document in documents:
                    count = len(document.chunks)
                    if id(document) in dense_ids and document.vectors.shape[1] == query_vectors.shape[1]:
                        cosine[offset:offset + count] = document.vectors @ query_vectors.T
                    offset += count
                for column in range(cosine.shape[1]):
                    add_ranking(cosine[:, column])

        if not fused:
            # Không đoạn nào khớp: dùng các đoạn đầu tài liệu (hữu ích cho câu hỏi kiểu "tóm tắt")
            ranked = [(position, 0.0) for position in range(min(limit, len(pool)))]
        else:
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]

        best = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        results = [self._hit(*pool[position], score / best) for position, score in ranked]
        return results, warnings, used_dense

    @staticmethod
    def _hit(document: StoredDocument, index: int, score: float) -> Tuple[Document, float]:
        """
        Đoạn thứ index của một tài liệu, kèm metadata nguồn để trích dẫn.
        """
        chunk = document.chunks[index]
        return (
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
            float(score),
        )

    def _semantic_search(
        self,
        documents: Sequence[StoredDocument],
        question: str,
        limit: int,
    ) -> Tuple[List[Tuple[Document, float]], List[str], str]:
        """
        Tìm kiếm của RAG thuần (02, 03): so độ tương đồng cosine giữa embedding của
        nguyên văn câu hỏi và từng đoạn, lấy `limit` đoạn cao nhất. Không viết lại câu hỏi,
        không kết hợp từ khoá. Chưa dùng được embedding thì xếp theo từ khoá (BM25).

        Returns:
            (các đoạn kèm điểm, cảnh báo, cách tìm: "ngữ nghĩa" hoặc "từ khoá")
        """
        embedder = self.embedder
        dense = [
            document for document in documents
            if embedder is not None and document.vectors is not None
            and document.meta.get("embedding_model") == embedder.model
        ]
        warnings: List[str] = []
        if dense:
            try:
                query = embedder.embed_queries([question])[0]
            except Exception as error:
                logger.warning("Không tạo được embedding cho câu hỏi: %s", redact_secrets(error))
                warnings.append(
                    f"Tìm kiếm ngữ nghĩa tạm thời không dùng được ({describe_embedding_error(error)}); "
                    "đã tìm theo từ khoá."
                )
            else:
                if len(dense) < len(documents):
                    warnings.append("Có tài liệu chưa có embedding nên chưa tìm được trong tài liệu đó.")
                scored = [
                    (float(score), document, index)
                    for document in dense if document.vectors.shape[1] == query.shape[0]
                    for index, score in enumerate(document.vectors @ query)
                ]
                scored.sort(key=lambda item: item[0], reverse=True)
                return [self._hit(document, index, score) for score, document, index in scored[:limit]], warnings, "ngữ nghĩa"

        pool = [(document, index) for document in documents for index in range(len(document.chunks))]
        scores = np.asarray(BM25([tokens for document in documents for tokens in document.keyword_tokens()]).scores(tokenize(question)))
        order = [int(position) for position in np.argsort(-scores, kind="stable")[:limit]]
        best = scores[order[0]] if order and scores[order[0]] > 0 else 1.0
        return [self._hit(*pool[position], scores[position] / best) for position in order], warnings, "từ khoá"

    def _rerank(
        self,
        question: str,
        hits: List[Tuple[Document, float]],
        llm: LLMClient,
    ) -> Tuple[List[Tuple[Document, float]], List[str]]:
        """
        Xếp hạng lại các đoạn ứng viên bằng LLM, giữ top_n đoạn điểm cao nhất.

        Ghi vào metadata mỗi đoạn: rerank_score (0-10) và initial_rank (thứ hạng trước khi rerank).
        Lỗi thì giữ thứ tự tìm kiếm ban đầu và trả về cảnh báo.
        """
        top_n = self.settings.top_n
        if not hits:
            return hits, []
        reranker = LLMReranker(llm, self.prompts, thinking_level=self.settings.analysis_thinking_level)
        try:
            scores = reranker.score(question, [document.page_content for document, _ in hits])
        except Exception as error:
            logger.warning("Reranking lỗi: %s", redact_secrets(error))
            return hits[:top_n], [f"Không xếp hạng lại được ({describe_error(error)}); dùng thứ tự tìm kiếm ban đầu."]

        order = sorted(range(len(hits)), key=lambda position: (-scores[position], position))[:top_n]
        reranked = []
        for position in order:
            document, fused = hits[position]
            document.metadata["rerank_score"] = round(scores[position], 2)
            document.metadata["initial_rank"] = position + 1
            reranked.append((document, fused))
        return reranked, []

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
                    "rerank_score": document.metadata.get("rerank_score"),
                    "initial_rank": document.metadata.get("initial_rank"),
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
        mode: str = "reasoning",
    ) -> Iterator[Dict[str, Any]]:
        """
        Trả lời câu hỏi theo một phương pháp, hoặc chạy lần lượt cả 4 để so sánh.

        Args:
            mode: "llm" | "rag" | "rerank" | "reasoning" | "compare"

        Mỗi phương pháp sinh ra các sự kiện (đều có trường "mode"):
        {"type": "start", "stages": [...]}, {"type": "status", "elapsed": ...} cho từng bước,
        rồi {"type": "result", "timings": {...}, "usage": {...}, "models": [...], ...}.
        Chế độ "compare" dùng cùng một model cho cả 4 phương pháp (báo trước bằng
        {"type": "compare", "model": ...}) và kết thúc bằng {"type": "done"}; phương pháp nào
        lỗi thì gửi {"type": "error", "mode": ...} và các phương pháp còn lại vẫn chạy tiếp.
        """
        self.check_conversation_id(conversation_id)
        if mode != "compare" and mode not in MODES:
            raise ServiceError(400, "Phương pháp không hợp lệ.")
        question = question.strip()
        if not question:
            raise ServiceError(400, "Câu hỏi đang trống.")

        metas = self.list_documents(conversation_id)
        history = list(history)[-self.settings.history_turns:]

        if mode != "compare":
            yield from self._run_mode(mode, conversation_id, question, history, metas)
            return

        deadline = time.time() + self.settings.compare_time_budget
        llm = self.compare_llm(deadline)
        pinned = model_of(llm.names[0]) if isinstance(llm, FallbackLLM) else None
        if pinned:
            yield {"type": "compare", "model": pinned}

        for item in COMPARE_ORDER:
            if time.time() > deadline:
                yield {
                    "type": "error",
                    "mode": item,
                    "message": "Đã bỏ qua vì sắp hết thời gian xử lý tối đa của server. Hãy hỏi riêng phương pháp này.",
                }
                continue
            try:
                yield from self._run_mode(item, conversation_id, question, history, metas, llm=llm)
            except Exception as error:
                if not isinstance(error, ServiceError):
                    logger.exception("Lỗi khi chạy phương pháp %s", item)
                message = describe_error(error)
                if pinned and is_rate_limit_error(error):
                    message = (
                        f"Model {pinned} đã chạm giới hạn của Groq và phải chờ quá lâu để so sánh trên cùng một model. "
                        "Thử lại sau ít phút, hoặc hỏi riêng phương pháp này (tự chuyển model khi hết quota)."
                    )
                yield {"type": "error", "mode": item, "message": message}
        yield {"type": "done"}

    def _run_mode(
        self,
        mode: str,
        conversation_id: str,
        question: str,
        history: Sequence[Tuple[str, str]],
        metas: List[Dict[str, Any]],
        llm: Optional[LLMClient] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Chạy một phương pháp:
        - llm:       hỏi thẳng LLM, không dùng tài liệu
        - rag:       tìm top_n đoạn tương đồng nhất (embedding) với nguyên văn câu hỏi -> prompt RAG cơ bản
        - rerank:    như rag nhưng lấy rerank_candidates đoạn -> LLM chấm điểm lại -> top_n -> prompt RAG cơ bản
        - reasoning: phân tích, tách truy vấn -> tìm hybrid (từ khoá + ngữ nghĩa) theo từng truy vấn
                     -> suy luận có trích dẫn -> kiểm chứng

        Args:
            llm: LLM dùng cho lần chạy này (mặc định self.llm, tự chuyển model khi hết quota)
        """
        llm = UsageRecorder(llm or self.llm)
        timer = StageTimer(lambda: llm.waited)
        use_rerank = mode == "rerank" or (mode == "reasoning" and self.settings.reasoning_rerank)

        stages = {
            "llm": ["generating"],
            "rag": ["retrieving", "generating"],
            "rerank": ["retrieving", "reranking", "generating"],
            "reasoning": ["analyzing", "retrieving"] + (["reranking"] if use_rerank else []) + ["reasoning"],
        }[mode]
        yield {
            "type": "start",
            "mode": mode,
            "title": MODES[mode],
            "stages": [{"key": key, "label": STAGE_LABELS[key]} for key in stages],
        }

        def status(stage: str, message: str) -> Dict[str, Any]:
            timer.start(stage)
            return {"type": "status", "mode": mode, "stage": stage, "message": message, "elapsed": timer.elapsed()}

        def finish(result: ReasonedAnswer, analysis: QueryAnalysis, hits=(), pipeline=(), retrieved=None, search=None) -> Dict[str, Any]:
            event = self._result_event(question, analysis, result, timer.finish(), hits)
            event.update(
                mode=mode,
                title=MODES[mode],
                models=list(llm.models),
                usage=llm.summary(),
                pipeline=list(pipeline),
                retrieved=retrieved,  # số đoạn tìm được trước khi xếp hạng lại
                search=search,  # cách tìm: "ngữ nghĩa" | "từ khoá" | "từ khoá + ngữ nghĩa"
            )
            return event

        # 01 — LLM thuần: không tìm tài liệu
        if mode == "llm":
            yield status("generating", "Đang hỏi LLM (không dùng tài liệu)")
            answer = llm.generate(
                fill_template(self.prompts["direct_user"], history=format_history(history), question=question),
                system=self.prompts["direct_system"],
                thinking_level=self.settings.thinking_level,
            )
            yield finish(
                ReasonedAnswer(answer=answer.strip(), answerable=True, confidence=""),
                QueryAnalysis("document", question, []),
                pipeline=["Không đọc tài liệu", "LLM trả lời bằng hiểu biết sẵn có"],
            )
            return

        pipeline: List[str] = []
        if mode == "reasoning":
            yield status("analyzing", "Đang phân tích câu hỏi")
            analyzer = QueryAnalyzer(llm, self.prompts, thinking_level=self.settings.analysis_thinking_level)
            names = "\n".join(f"- {meta['filename']}" for meta in metas) or "(chưa có tài liệu nào)"
            analysis = analyzer.analyze(question, history, documents=names)
            pipeline.append(f"Phân tích câu hỏi, tách {len(analysis.sub_queries) or 1} truy vấn tìm kiếm")

            if analysis.category != "document":
                reply = analysis.direct_reply or OUT_OF_SCOPE_MESSAGE
                yield finish(ReasonedAnswer(answer=reply, answerable=False, confidence="cao"), analysis, pipeline=pipeline)
                return
            queries = analysis.sub_queries or [analysis.standalone_question]
            search_question = analysis.standalone_question
        else:
            # 02, 03: tìm theo nguyên văn câu hỏi, không viết lại / tách truy vấn
            analysis = QueryAnalysis("document", question, [question])
            queries = [question]
            search_question = question

        if not metas:
            yield finish(ReasonedAnswer(answer=NO_DOCUMENT_MESSAGE, answerable=False, confidence=""), analysis, pipeline=pipeline)
            return

        yield status("retrieving", f"Đang tìm trong {len(metas)} tài liệu")
        documents = [self._load_document(meta, conversation_id) for meta in metas]
        self._embed_missing(documents, conversation_id)
        limit = self.settings.rerank_candidates if use_rerank else self.settings.top_n
        if mode == "reasoning":
            # 04: tìm theo các truy vấn LLM đã tách, kết hợp từ khoá + ngữ nghĩa (hybrid)
            hits, warnings, used_dense = self._retrieve(documents, queries, limit=limit)
            search = "từ khoá + ngữ nghĩa" if used_dense else "từ khoá"
            pipeline.append(f"Tìm {len(hits)} đoạn theo {len(queries)} truy vấn ({search})")
        else:
            # 02, 03: RAG thuần, chỉ so độ tương đồng với nguyên văn câu hỏi
            hits, warnings, search = self._semantic_search(documents, question, limit)
            target = "ứng viên" if use_rerank else "tương đồng nhất với câu hỏi"
            pipeline.append(f"Tìm {len(hits)} đoạn {target} ({search})")
        retrieved = len(hits)

        if use_rerank:
            yield status("reranking", f"Đang xếp hạng lại {len(hits)} đoạn")
            hits, rerank_warnings = self._rerank(search_question, hits, llm)
            warnings += rerank_warnings
            pipeline.append(f"LLM đọc và chấm điểm lại từng đoạn, giữ {len(hits)} đoạn tốt nhất")

        if mode == "reasoning":
            yield status("reasoning", f"Đang suy luận trên {len(hits)} đoạn trích")
            result = ReasoningAnswerer(llm, self.prompts).answer(search_question, [document for document, _ in hits])
            result.warnings = warnings + result.warnings
            pipeline += ["Suy luận từng bước, trích nguyên văn làm bằng chứng", "Kiểm chứng trích dẫn với tài liệu"]
        else:
            yield status("generating", f"Đang tạo câu trả lời từ {len(hits)} đoạn")
            answer = llm.generate(
                fill_template(
                    self.prompts["plain_rag_user"],
                    context=format_context([document for document, _ in hits]),
                    question=question,
                ),
                system=self.prompts["plain_rag_system"],
                thinking_level=self.settings.thinking_level,
            )
            result = ReasonedAnswer(answer=answer.strip(), answerable=True, confidence="", warnings=warnings)
            pipeline.append(f"LLM trả lời từ {len(hits)} đoạn")

        yield finish(result, analysis, hits, pipeline, retrieved, search)


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
