"""
Web app hỏi–đáp tài liệu PDF bằng RAG + reasoning.

- Vercel tự nhận file này làm entrypoint (biến `app` là FastAPI)
- Giao diện tĩnh nằm trong public/ (Vercel phục vụ qua CDN)

Chạy trên máy:
    uvicorn app:app --reload        # mở http://127.0.0.1:8000

Version: 1.0.0
"""

import json
import logging
import os
from pathlib import Path
from typing import Iterator, List
from urllib.parse import quote

import webapp  # noqa: F401  (thêm RAG_system vào sys.path)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from RAG.gen import create_llm, describe_llm
from webapp.config import WebSettings
from webapp.service import ServiceError, describe_error, get_service

logger = logging.getLogger("rag_web")

PUBLIC_DIR = Path(__file__).resolve().parent / "public"

app = FastAPI(title="RAG Reasoning", docs_url="/api/docs", openapi_url="/api/openapi.json")


class HistoryTurn(BaseModel):
    question: str = Field(max_length=4000)
    answer: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: List[HistoryTurn] = Field(default_factory=list, max_length=20)


@app.exception_handler(ServiceError)
def handle_service_error(request, error: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": error.message})


@app.get("/api/health")
def health() -> dict:
    settings = WebSettings()
    try:
        llm_providers = describe_llm(create_llm(models=settings.llm_models))
    except ValueError:
        llm_providers = []
    status = {
        # Thứ tự dự phòng: hết quota ở nhà cung cấp đầu thì chuyển sang nhà cung cấp sau
        "llm_providers": llm_providers,
        "llm_configured": bool(llm_providers),
        "embedding_model": None,
        "max_upload_mb": settings.max_upload_mb,
        "max_documents": settings.max_documents_per_conversation,
    }
    try:
        service = get_service()
        status["storage"] = service.storage.name
        embedder = service.embedder
        status["embedding_model"] = embedder.model if embedder is not None else None
    except ServiceError as error:
        status["storage"] = None
        status["storage_error"] = error.message
    return status


@app.get("/api/conversations/{conversation_id}/documents")
def list_documents(conversation_id: str) -> List[dict]:
    return get_service().list_documents(conversation_id)


@app.post("/api/conversations/{conversation_id}/documents")
def upload_document(conversation_id: str, file: UploadFile = File(...)) -> dict:
    service = get_service()
    data = file.file.read(service.settings.max_upload_bytes + 1)
    try:
        return service.add_document(conversation_id, file.filename or "document.pdf", data)
    except ServiceError:
        raise
    except Exception as error:
        logger.exception("Lỗi khi xử lý file upload")
        raise HTTPException(status_code=502, detail=describe_error(error)) from error


@app.get("/api/conversations/{conversation_id}/documents/{document_id}/file")
def download_document(conversation_id: str, document_id: str) -> Response:
    filename, data = get_service().get_document_file(conversation_id, document_id)
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"},
    )


@app.get("/api/conversations/{conversation_id}/documents/{document_id}/overview")
def document_overview(conversation_id: str, document_id: str) -> dict:
    try:
        return get_service().overview(conversation_id, document_id)
    except ServiceError:
        raise
    except Exception as error:
        logger.exception("Lỗi khi tóm tắt tài liệu")
        raise HTTPException(status_code=502, detail=describe_error(error)) from error


@app.delete("/api/conversations/{conversation_id}/documents/{document_id}")
def delete_document(conversation_id: str, document_id: str) -> dict:
    get_service().delete_document(conversation_id, document_id)
    return {"deleted": document_id}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    get_service().delete_conversation(conversation_id)
    return {"deleted": conversation_id}


@app.post("/api/conversations/{conversation_id}/ask")
def ask(conversation_id: str, request: AskRequest) -> StreamingResponse:
    """
    Trả lời câu hỏi dạng NDJSON: mỗi dòng là một sự kiện status / result / error.
    """
    service = get_service()
    service.check_conversation_id(conversation_id)
    history = [(turn.question, turn.answer) for turn in request.history]

    def events() -> Iterator[str]:
        try:
            for event in service.ask(conversation_id, request.question, history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as error:
            if not isinstance(error, ServiceError):
                logger.exception("Lỗi khi trả lời câu hỏi")
            yield json.dumps({"type": "error", "message": describe_error(error)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if os.getenv("VERCEL"):
    # Trên Vercel, CDN phục vụ public/ (kể cả "/"); route này chỉ là phương án dự phòng
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/index.html")
else:
    # Chạy trên máy: tự phục vụ giao diện trong public/
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
