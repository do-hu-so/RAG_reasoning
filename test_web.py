"""
Test offline cho bản web: không cần API key, không cần Vercel.

Dùng embedder giả (bag-of-words), LLM giả và LocalStorage trong thư mục tạm để kiểm tra:
1. Upload PDF -> đọc -> chunk -> embedding -> lưu "dataset web"
2. Liệt kê / tải / xoá tài liệu, xoá cuộc trò chuyện
3. Hỏi–đáp dạng stream (status -> result), lịch sử hội thoại, câu chào hỏi
4. Các lỗi: file không phải PDF, quá lớn, PDF không có chữ, mã không hợp lệ
5. GeminiEmbedder: định dạng tiền tố, tách từng vector, chuẩn hoá

Chạy: python test_web.py

Version: 1.0.0
"""

import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows mặc định không in được tiếng Việt
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import app
from RAG.gen import load_prompts
from RAG.reasoning import ANSWER_SCHEMA, QUERY_ANALYSIS_SCHEMA
from webapp.config import WEB_PROMPT_FILE, WebSettings
from webapp.embeddings import GeminiEmbedder
from webapp.service import NO_DOCUMENT_MESSAGE, OVERVIEW_SCHEMA, WebRAGService, set_service
from webapp.storage import LocalStorage

PDF_PATH = PROJECT_ROOT / "dataset" / "Đề thi Procon_Tiếng Việt.pdf"
CONVERSATION = "conv_test_0001"


class FakeEmbedder:
    """Embedding bag-of-words: đoạn có nhiều từ chung với câu hỏi thì gần nhau."""

    model = "fake-embedding"

    def __init__(self):
        self.query_calls = 0

    def _embed(self, texts):
        matrix = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                matrix[row, zlib.crc32(word.encode("utf-8")) % 256] += 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def embed_documents(self, texts, title=None):
        return self._embed(texts)

    def embed_queries(self, texts):
        self.query_calls += 1
        return self._embed(texts)


class FakeLLM:
    """LLM giả: trả lời theo schema được yêu cầu, trích nguyên văn đoạn [1] trong prompt."""

    def __init__(self, category="document", fail=False):
        self.category = category
        self.fail = fail
        self.calls = []

    def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
        self.calls.append({"prompt": prompt, "schema": json_schema})
        if self.fail:
            raise RuntimeError("model lỗi giả lập")

        if json_schema is QUERY_ANALYSIS_SCHEMA:
            question = prompt.rsplit("Câu hỏi mới:", 1)[1].strip()
            return json.dumps({
                "category": self.category,
                "standalone_question": question,
                "sub_queries": [question],
                "direct_reply": "Xin chào!" if self.category != "document" else "",
            })

        if json_schema is ANSWER_SCHEMA:
            first_passage = prompt.split("[1] (", 1)[1].split("\n", 1)[1].split("\n\n[2]", 1)[0]
            quote = " ".join(first_passage.split()[:8])
            return json.dumps({
                "evidence": [{"source_id": 1, "quote": quote}],
                "reasoning_steps": ["Theo [1] ..."],
                "answerable": True,
                "answer": "Câu trả lời dựa trên tài liệu [1].",
                "confidence": "cao",
            })

        if json_schema is OVERVIEW_SCHEMA:
            return json.dumps({"summary": "Đề thi PROCON phần HEXUDON.", "questions": ["Câu 1?", "Câu 2?", "Câu 3?"]})

        raise AssertionError("schema không mong đợi")


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def read_events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


class WebTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = WebSettings(local_dataset_dir=self.tmp)
        self.embedder = FakeEmbedder()
        self.llm = FakeLLM()
        self.service = WebRAGService(
            self.settings,
            LocalStorage(self.tmp),
            embedder=self.embedder,
            llm=self.llm,
            prompts=load_prompts(WEB_PROMPT_FILE),
        )
        set_service(self.service)
        self.client = TestClient(app)

    def tearDown(self):
        set_service(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, data=None, filename="de_thi.pdf", conversation=CONVERSATION):
        data = PDF_PATH.read_bytes() if data is None else data
        return self.client.post(
            f"/api/conversations/{conversation}/documents",
            files={"file": (filename, data, "application/pdf")},
        )

    def ask(self, question, history=(), conversation=CONVERSATION):
        response = self.client.post(
            f"/api/conversations/{conversation}/ask",
            json={"question": question, "history": [{"question": q, "answer": a} for q, a in history]},
        )
        self.assertEqual(response.status_code, 200)
        return read_events(response)


class TestDocuments(WebTestCase):

    def test_upload_list_download_overview_delete(self):
        response = self.upload(filename="Đề thi Procon.pdf")
        self.assertEqual(response.status_code, 200, response.text)
        meta = response.json()
        self.assertEqual(meta["pages"], 7)
        self.assertGreater(meta["chunks"], 5)
        self.assertEqual(meta["filename"], "Đề thi Procon.pdf")

        # File gốc + chunks + embeddings + meta được lưu trong "dataset web"
        stored = sorted(p.name for p in (self.tmp / CONVERSATION / meta["id"]).iterdir())
        self.assertEqual(stored, ["chunks.json", "embeddings.npy", "meta.json", "Đề thi Procon.pdf"])

        listed = self.client.get(f"/api/conversations/{CONVERSATION}/documents").json()
        self.assertEqual([doc["id"] for doc in listed], [meta["id"]])

        file_response = self.client.get(f"/api/conversations/{CONVERSATION}/documents/{meta['id']}/file")
        self.assertEqual(file_response.content, PDF_PATH.read_bytes())
        self.assertEqual(file_response.headers["content-type"], "application/pdf")

        overview = self.client.get(f"/api/conversations/{CONVERSATION}/documents/{meta['id']}/overview").json()
        self.assertEqual(len(overview["questions"]), 3)
        calls = len(self.llm.calls)
        self.client.get(f"/api/conversations/{CONVERSATION}/documents/{meta['id']}/overview")
        self.assertEqual(len(self.llm.calls), calls)  # lần 2 đọc từ overview.json

        self.assertEqual(self.client.delete(f"/api/conversations/{CONVERSATION}/documents/{meta['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/conversations/{CONVERSATION}/documents").json(), [])
        self.assertFalse((self.tmp / CONVERSATION).exists())

    def test_rejections(self):
        self.assertEqual(self.upload(filename="ghi_chu.txt").status_code, 415)
        self.assertEqual(self.upload(data=b"not a pdf").status_code, 415)

        response = self.upload(data=blank_pdf())
        self.assertEqual(response.status_code, 422)
        self.assertIn("không có lớp chữ", response.json()["detail"])

        self.settings.max_upload_mb = 1  # PDF mẫu nặng 1.33 MB
        self.assertEqual(self.upload().status_code, 413)

        self.assertEqual(self.upload(conversation="bad id!").status_code, 400)
        self.assertEqual(self.client.get(f"/api/conversations/{CONVERSATION}/documents/abcdefabcdef/file").status_code, 404)
        self.assertEqual(self.client.get(f"/api/conversations/{CONVERSATION}/documents/../file").status_code, 404)

    def test_delete_conversation(self):
        self.upload()
        self.upload(filename="ban_sao.pdf")
        self.assertEqual(len(self.client.get(f"/api/conversations/{CONVERSATION}/documents").json()), 2)
        self.client.delete(f"/api/conversations/{CONVERSATION}")
        self.assertEqual(self.client.get(f"/api/conversations/{CONVERSATION}/documents").json(), [])

    def test_local_storage_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            LocalStorage(self.tmp).put("../ngoai.txt", b"x", "text/plain")


class TestAsk(WebTestCase):

    def test_ask_streams_status_then_grounded_result(self):
        self.upload()
        events = self.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")

        statuses = [event for event in events if event["type"] == "status"]
        self.assertEqual([event["stage"] for event in statuses], ["analyzing", "retrieving", "reasoning"])
        elapsed = [event["elapsed"] for event in statuses]
        self.assertEqual(elapsed, sorted(elapsed))  # mốc thời gian tăng dần

        result = events[-1]
        self.assertEqual(result["type"], "result")
        timings = result["timings"]
        self.assertEqual(set(timings), {"analyzing", "retrieving", "reasoning", "total"})
        self.assertGreaterEqual(timings["total"], timings["analyzing"] + timings["retrieving"] + timings["reasoning"] - 0.01)
        self.assertGreaterEqual(timings["total"], elapsed[-1])
        self.assertIsNotNone(result["sources"][0]["page_end"])
        self.assertEqual(result["category"], "document")
        self.assertLessEqual(len(result["sources"]), self.settings.top_n)
        self.assertTrue(result["evidence"][0]["verified"])
        self.assertEqual(result["confidence"], "cao")
        self.assertIn("nhiên liệu", result["sources"][0]["content"])
        self.assertTrue(result["sources"][0]["label"].startswith("de_thi.pdf, trang"))

    def test_history_and_document_names_reach_analyzer(self):
        self.upload(filename="luat_thi.pdf")
        self.ask("Còn xe tiếp nhiên liệu thì sao?", history=[("Xe tuần tra làm gì?", "Thu thập udon.")])
        analysis_prompt = self.llm.calls[0]["prompt"]
        self.assertIn("Xe tuần tra làm gì?", analysis_prompt)
        self.assertIn("- luat_thi.pdf", analysis_prompt)

    def test_ask_without_documents(self):
        events = self.ask("Tài liệu nói gì?")
        self.assertEqual(events[-1]["answer"], NO_DOCUMENT_MESSAGE)
        self.assertEqual(self.embedder.query_calls, 0)

    def test_chitchat_skips_retrieval(self):
        self.upload()
        self.llm.category = "chitchat"
        events = self.ask("Xin chào")
        self.assertEqual(events[-1]["answer"], "Xin chào!")
        self.assertEqual(events[-1]["sources"], [])
        self.assertEqual(set(events[-1]["timings"]), {"analyzing", "total"})
        self.assertEqual(self.embedder.query_calls, 0)

    def test_errors_become_error_event(self):
        self.llm.fail = True
        events = self.ask("Câu hỏi bất kỳ")
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("model lỗi giả lập", events[-1]["message"])

        response = self.client.post(f"/api/conversations/{CONVERSATION}/ask", json={"question": ""})
        self.assertEqual(response.status_code, 422)


class TestAppShell(WebTestCase):

    def test_health_and_static_frontend(self):
        health = self.client.get("/api/health").json()
        self.assertEqual(health["storage"], "local")
        self.assertIn("gemini_configured", health)

        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("RAG Reasoning", page.text)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)


class TestStorageSelection(unittest.TestCase):

    def test_choose_storage_by_environment(self):
        from unittest import mock
        from webapp.storage import BlobStorage, create_storage

        clean = {k: v for k, v in os.environ.items() if "BLOB" not in k.upper() and k not in ("VERCEL", "VERCEL_ENV")}
        settings = WebSettings(local_dataset_dir=Path(tempfile.mkdtemp()))

        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertIsInstance(create_storage(settings), LocalStorage)

        with mock.patch.dict(os.environ, dict(clean, VERCEL="1", VERCEL_ENV="production", BLOB_WEBHOOK_PUBLIC_KEY="x"), clear=True):
            with self.assertRaises(RuntimeError) as caught:
                create_storage(settings)
            message = str(caught.exception)
            self.assertIn("production", message)
            self.assertIn("Redeploy", message)
            self.assertIn("BLOB_WEBHOOK_PUBLIC_KEY", message)  # chỉ tên biến
            self.assertNotIn("=x", message)

        for name in ("BLOB_STORE_ID", "BLOB_READ_WRITE_TOKEN", "VERCEL_BLOB_READ_WRITE_TOKEN"):
            with mock.patch.dict(os.environ, dict(clean, VERCEL="1", **{name: "store_abc"}), clear=True):
                self.assertIsInstance(create_storage(settings), BlobStorage)


class TestGeminiEmbedder(unittest.TestCase):

    def make(self, model, aggregate=False):
        embedder = GeminiEmbedder(model=model, dimensions=4, api_key="khong-goi-mang")
        calls = []

        def embed_content(model, contents, config):
            calls.append({"contents": contents, "config": config})
            items = contents if isinstance(contents, list) else [contents]
            count = 1 if aggregate and isinstance(contents, list) else len(items)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0, 0.0, 0.0]) for _ in range(count)])

        embedder._client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
        return embedder, calls

    def test_prefix_format_and_normalization(self):
        embedder, calls = self.make("gemini-embedding-2")
        vectors = embedder.embed_documents(["nội dung A", "nội dung B"], title="De thi")
        self.assertEqual(vectors.shape, (2, 4))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
        self.assertEqual(calls[0]["contents"][0].parts[0].text, "title: De thi | text: nội dung A")
        self.assertIsNone(calls[0]["config"].task_type)

        embedder.embed_queries(["câu hỏi"])
        self.assertEqual(calls[1]["contents"][0].parts[0].text, "task: search result | query: câu hỏi")

    def test_task_type_for_embedding_001_and_aggregate_fallback(self):
        embedder, calls = self.make("gemini-embedding-001")
        embedder.embed_queries(["q"])
        self.assertEqual(calls[0]["config"].task_type, "RETRIEVAL_QUERY")
        self.assertEqual(calls[0]["contents"][0].parts[0].text, "q")

        embedder, calls = self.make("gemini-embedding-2", aggregate=True)
        vectors = embedder.embed_documents(["a", "b", "c"])
        self.assertEqual(vectors.shape, (3, 4))
        self.assertEqual(len(calls), 1 + 3)  # 1 lần gộp + gọi lại từng đoạn


if __name__ == "__main__":
    unittest.main(verbosity=2)
