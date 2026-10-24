"""
Test offline cho bản web: không cần API key, không cần Vercel.

Dùng embedder giả (bag-of-words), LLM giả và LocalStorage trong thư mục tạm để kiểm tra:
1. Upload PDF -> đọc -> chunk -> embedding -> lưu "dataset web"
2. Liệt kê / tải / xoá tài liệu, xoá cuộc trò chuyện
3. Hỏi–đáp dạng stream (status -> result), lịch sử hội thoại, câu chào hỏi
4. Các lỗi: file không phải PDF, quá lớn, PDF không có chữ, mã không hợp lệ
5. Groq (nhiều model dự phòng), Cloudflare embedding, BM25

Chạy: python test_web.py

Version: 1.0.0
"""

import io
import json
import os
import re
import shutil
import time
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows mặc định không in được tiếng Việt
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")

import httpx
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import app
from RAG.gen import (
    FallbackLLM,
    OpenAICompatibleClient,
    ProviderError,
    create_llm,
    describe_llm,
    load_prompts,
)
from RAG.reasoning import ANSWER_SCHEMA, QUERY_ANALYSIS_SCHEMA
from webapp.config import WEB_PROMPT_FILE, WebSettings
from webapp.embeddings import CloudflareEmbedder, create_embedder
from webapp.keyword_search import BM25, tokenize
from webapp.reranking import RERANK_SCHEMA, LLMReranker
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

    def __init__(self, category="document", fail=False, fail_direct=False):
        self.category = category
        self.fail = fail
        self.fail_direct = fail_direct
        self.calls = []
        self.last_used = "groq/model-gia"
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 20}

    def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
        self.calls.append({"prompt": prompt, "schema": json_schema, "system": system})
        if self.fail:
            raise RuntimeError("model lỗi giả lập")

        if json_schema is None:
            if "Ngữ cảnh:" in prompt:  # RAG thuần (02 / 03)
                return "Trả lời RAG thuần từ ngữ cảnh."
            if self.fail_direct:
                raise RuntimeError("LLM thuần lỗi giả lập")
            return "Trả lời trực tiếp, không dùng tài liệu."  # LLM thuần (01)

        if json_schema is RERANK_SCHEMA:
            # Chấm điểm cao nhất cho đoạn nói về "hết nhiên liệu"
            passages = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.M)
            return json.dumps({"scores": [
                {"id": int(number), "score": 9 if "hết nhiên liệu" in text else 2} for number, text in passages
            ]})

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


class RateLimitedLLM(FakeLLM):
    """Model chạm giới hạn token/phút ở vài lần gọi đầu (chờ retry_after giây là gọi lại được)."""

    def __init__(self, limited_calls=1, retry_after=0.2):
        super().__init__()
        self.limited_calls = limited_calls
        self.retry_after = retry_after

    def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
        if self.limited_calls:
            self.limited_calls -= 1
            raise ProviderError("groq", 429, "rate limit", retry_after=self.retry_after)
        return super().generate(prompt, system, json_schema, thinking_level)


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

    def ask(self, question, history=(), conversation=CONVERSATION, mode="reasoning"):
        response = self.client.post(
            f"/api/conversations/{conversation}/ask",
            json={"question": question, "history": [{"question": q, "answer": a} for q, a in history], "mode": mode},
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


class TestModes(WebTestCase):
    """4 phương pháp 01–04 và chế độ so sánh."""

    QUESTION = "Xe tuần tra hết nhiên liệu thì phải làm gì?"

    def result_of(self, events, mode):
        return [event for event in events if event["type"] == "result" and event["mode"] == mode][0]

    def test_llm_only_does_not_use_documents(self):
        self.upload()
        events = self.ask(self.QUESTION, mode="llm")
        self.assertEqual(events[0], {"type": "start", "mode": "llm", "title": "LLM thuần, không RAG",
                                     "stages": [{"key": "generating", "label": "Sinh câu trả lời"}]})
        result = events[-1]
        self.assertEqual(result["answer"], "Trả lời trực tiếp, không dùng tài liệu.")
        self.assertEqual(result["sources"], [])
        self.assertEqual(len(self.llm.calls), 1)
        self.assertNotIn("Ngữ cảnh", self.llm.calls[0]["prompt"])
        self.assertEqual(self.embedder.query_calls, 0)  # không tìm kiếm
        self.assertEqual(result["usage"], {"llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 20})
        self.assertEqual(result["models"], ["groq/model-gia"])
        self.assertEqual(set(result["timings"]), {"generating", "total"})

    def test_plain_rag_uses_raw_question_without_analysis(self):
        self.upload()
        result = self.ask(self.QUESTION, mode="rag")[-1]
        self.assertEqual(result["mode"], "rag")
        self.assertEqual(result["answer"], "Trả lời RAG thuần từ ngữ cảnh.")
        self.assertEqual(len(result["sources"]), self.settings.top_n)
        self.assertEqual(len(self.llm.calls), 1)  # không có bước phân tích câu hỏi
        self.assertIn("Ngữ cảnh:", self.llm.calls[0]["prompt"])
        self.assertEqual(result["sub_queries"], [self.QUESTION])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["confidence"], "")
        self.assertIsNone(result["sources"][0]["rerank_score"])
        self.assertEqual(set(result["timings"]), {"retrieving", "generating", "total"})

    def test_rerank_reorders_candidates(self):
        self.upload()
        events = self.ask(self.QUESTION, mode="rerank")
        stages = [event["stage"] for event in events if event["type"] == "status"]
        self.assertEqual(stages, ["retrieving", "reranking", "generating"])

        result = events[-1]
        self.assertEqual(len(result["sources"]), self.settings.top_n)
        top = result["sources"][0]
        self.assertIn("hết nhiên liệu", top["content"])
        self.assertEqual(top["rerank_score"], 9.0)
        self.assertGreaterEqual(top["initial_rank"], 1)
        scores = [source["rerank_score"] for source in result["sources"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Đã chấm điểm nhiều ứng viên hơn số đoạn giữ lại
        rerank_prompt = [call for call in self.llm.calls if call["schema"] is RERANK_SCHEMA][0]["prompt"]
        self.assertEqual(len(re.findall(r"^\[\d+\] ", rerank_prompt, re.M)), self.settings.rerank_candidates)
        self.assertEqual(result["usage"]["llm_calls"], 2)

    def test_reasoning_mode_reports_usage(self):
        self.upload()
        result = self.ask(self.QUESTION, mode="reasoning")[-1]
        self.assertEqual(result["mode"], "reasoning")
        self.assertEqual(result["usage"]["llm_calls"], 2)  # phân tích + suy luận
        self.assertTrue(result["evidence"][0]["verified"])
        self.assertIn("Kiểm chứng trích dẫn với tài liệu", result["pipeline"])

    def test_compare_runs_all_modes_in_order(self):
        self.upload()
        events = self.ask(self.QUESTION, mode="compare")
        starts = [event["mode"] for event in events if event["type"] == "start"]
        self.assertEqual(starts, ["llm", "rag", "rerank", "reasoning"])
        self.assertEqual(events[-1], {"type": "done"})
        for mode in starts:
            result = self.result_of(events, mode)
            self.assertTrue(result["answer"])
            self.assertIn("total", result["timings"])
        self.assertEqual(self.result_of(events, "llm")["sources"], [])

    def test_compare_continues_after_one_mode_fails(self):
        self.upload()
        self.llm.fail_direct = True
        events = self.ask(self.QUESTION, mode="compare")
        errors = [event for event in events if event["type"] == "error"]
        self.assertEqual([event["mode"] for event in errors], ["llm"])
        self.assertIn("LLM thuần lỗi giả lập", errors[0]["message"])
        self.assertEqual([event["mode"] for event in events if event["type"] == "result"], ["rag", "rerank", "reasoning"])
        self.assertEqual(events[-1]["type"], "done")

    def use_llm(self, llm, **settings):
        settings.setdefault("compare_model", "")  # model đầu tiên của llm
        self.settings = WebSettings(local_dataset_dir=self.tmp, **settings)
        self.service = WebRAGService(self.settings, LocalStorage(self.tmp), embedder=self.embedder,
                                     llm=llm, prompts=load_prompts(WEB_PROMPT_FILE))
        set_service(self.service)

    def test_compare_pins_one_model_and_waits_for_quota(self):
        main, backup = RateLimitedLLM(limited_calls=1), FakeLLM()
        self.use_llm(FallbackLLM([("groq/main", main), ("groq/backup", backup)]))
        self.upload()
        events = self.ask(self.QUESTION, mode="compare")
        self.assertEqual(events[0], {"type": "compare", "model": "groq/main"})

        results = [event for event in events if event["type"] == "result"]
        self.assertEqual([result["mode"] for result in results], ["llm", "rag", "rerank", "reasoning"])
        self.assertTrue(all(result["models"] == ["groq/main"] for result in results))
        self.assertEqual(backup.calls, [])  # chờ hạn mức thay vì chuyển model

        # Thời gian chờ ghi riêng, không cộng vào thời gian xử lý
        timings = results[0]["timings"]
        self.assertGreaterEqual(timings["rate_limit_wait"], 0.2)
        self.assertLess(timings["total"], 0.2)
        self.assertLess(timings["generating"], 0.2)
        self.assertTrue(all("rate_limit_wait" not in result["timings"] for result in results[1:]))

    def test_single_mode_still_switches_model(self):
        main, backup = RateLimitedLLM(limited_calls=1), FakeLLM()
        self.use_llm(FallbackLLM([("groq/main", main), ("groq/backup", backup)]))
        result = self.ask(self.QUESTION, mode="llm")[-1]
        self.assertEqual(result["models"], ["groq/backup"])
        self.assertNotIn("rate_limit_wait", result["timings"])

    def test_compare_model_setting(self):
        main, backup = FakeLLM(), FakeLLM()
        self.use_llm(FallbackLLM([("groq/main", main), ("groq/backup", backup)]), compare_model="groq/backup")
        events = self.ask(self.QUESTION, mode="compare")
        self.assertEqual(events[0]["model"], "groq/backup")
        self.assertEqual(main.calls, [])

    def test_compare_switches_key_but_keeps_model(self):
        key1, key2, other = RateLimitedLLM(limited_calls=99, retry_after=600), FakeLLM(), FakeLLM()
        self.use_llm(FallbackLLM([("groq/openai/gpt-oss-20b", key1), ("groq/openai/gpt-oss-20b#2", key2),
                                  ("groq/llama", other)]), compare_model="openai/gpt-oss-20b")
        self.upload()
        events = self.ask(self.QUESTION, mode="compare")
        self.assertEqual(events[0], {"type": "compare", "model": "groq/openai/gpt-oss-20b"})
        results = [event for event in events if event["type"] == "result"]
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["models"] == ["groq/openai/gpt-oss-20b#2"] for result in results))
        self.assertEqual(other.calls, [])  # không đổi sang model khác

    def test_compare_reports_pinned_model_quota_error(self):
        main = RateLimitedLLM(limited_calls=99, retry_after=600)  # hết hạn mức lâu: không chờ
        self.use_llm(FallbackLLM([("groq/main", main), ("groq/backup", FakeLLM())]))
        self.upload()
        events = self.ask(self.QUESTION, mode="compare")
        errors = [event for event in events if event["type"] == "error"]
        self.assertEqual([event["mode"] for event in errors], ["llm", "rag", "rerank", "reasoning"])
        self.assertIn("groq/main", errors[0]["message"])

    def test_rag_modes_without_documents(self):
        for mode in ("rag", "rerank"):
            self.assertIn("chưa có tài liệu", self.ask(self.QUESTION, mode=mode)[-1]["answer"])
        self.assertEqual(self.llm.calls, [])

    def test_invalid_mode_rejected(self):
        response = self.client.post(f"/api/conversations/{CONVERSATION}/ask", json={"question": "hỏi", "mode": "abc"})
        self.assertEqual(response.status_code, 422)

    def test_llm_reranker_parsing(self):
        class Reply:
            def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
                return '{"scores": [{"id": 2, "score": 15}, {"id": 1, "score": "x"}, {"id": 9, "score": 5}]}'
        scores = LLMReranker(Reply(), load_prompts(WEB_PROMPT_FILE)).score("q", ["a", "b", "c"])
        self.assertEqual(scores, [0.0, 10.0, 0.0])  # kẹp về 0-10, bỏ id sai / điểm sai


class TestAppShell(WebTestCase):

    def test_health_and_static_frontend(self):
        health = self.client.get("/api/health").json()
        self.assertEqual(health["storage"], "local")
        self.assertIn("llm_configured", health)
        self.assertEqual(health["embedding_model"], "fake-embedding")

        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("RAG Reasoning", page.text)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)


class FailingEmbedder(FakeEmbedder):
    """Embedder giả lập hết quota (429) khi tạo vector cho tài liệu và / hoặc câu hỏi."""

    model = "fake-embedding"

    def __init__(self, fail_documents=True, fail_queries=True):
        super().__init__()
        self.fail_documents = fail_documents
        self.fail_queries = fail_queries
        self.document_calls = 0

    def embed_documents(self, texts, title=None):
        self.document_calls += 1
        if self.fail_documents:
            raise ProviderError("Cloudflare", 429, "quota")
        return super().embed_documents(texts, title)

    def embed_queries(self, texts):
        if self.fail_queries:
            raise ProviderError("Cloudflare", 429, "quota")
        return super().embed_queries(texts)


class TestKeywordSearch(WebTestCase):

    def use_embedder(self, embedder):
        self.service = WebRAGService(self.settings, LocalStorage(self.tmp), embedder=embedder,
                                     llm=self.llm, prompts=load_prompts(WEB_PROMPT_FILE))
        set_service(self.service)

    def test_tokenize_and_bm25_ranking(self):
        self.assertEqual(tokenize("Xe TUẦN tra"), ["xe", "tuần", "tra", "xe_tuần", "tuần_tra"])
        bm25 = BM25([tokenize("xe tuần tra hết nhiên liệu"), tokenize("bản đồ lục giác"), tokenize("nhiên liệu")])
        scores = bm25.scores(tokenize("hết nhiên liệu"))
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 0)
        self.assertEqual(scores[1], 0)

    def test_keyword_only_without_embedder(self):
        self.use_embedder(None)
        meta = self.upload().json()
        self.assertEqual(meta["search"], "keyword")
        self.assertIsNone(meta["warning"])
        self.assertFalse((self.tmp / CONVERSATION / meta["id"] / "embeddings.npy").exists())

        result = self.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")[-1]
        self.assertIn("hết nhiên liệu", result["sources"][0]["content"])
        self.assertEqual(result["sources"][0]["score"], 1.0)
        self.assertEqual(result["warnings"], [])

    def test_upload_survives_embedding_quota(self):
        self.use_embedder(FailingEmbedder())
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        meta = response.json()
        self.assertEqual(meta["search"], "keyword")
        self.assertIn("Cloudflare đã hết quota embedding", meta["warning"])
        self.assertIn("WEB_EMBEDDING_PROVIDER=none", meta["warning"])
        result = self.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")[-1]
        self.assertEqual(result["type"], "result")
        self.assertTrue(result["sources"])

    def test_keyword_document_upgraded_when_embedding_available(self):
        self.use_embedder(None)
        meta = self.upload().json()
        self.assertEqual(meta["search"], "keyword")

        embedder = FakeEmbedder()
        self.use_embedder(embedder)
        result = self.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")[-1]
        self.assertEqual(result["type"], "result")
        self.assertEqual(embedder.query_calls, 1)  # đã dùng tìm kiếm ngữ nghĩa

        upgraded = self.client.get(f"/api/conversations/{CONVERSATION}/documents").json()[0]
        self.assertEqual((upgraded["search"], upgraded["embedding_model"], upgraded["warning"]),
                         ("hybrid", "fake-embedding", None))
        self.assertTrue((self.tmp / CONVERSATION / meta["id"] / "embeddings.npy").exists())

    def test_embedding_rests_after_quota_error(self):
        embedder = FailingEmbedder()
        self.use_embedder(embedder)
        self.upload()                                   # lần 1: upload
        self.ask("Xe tuần tra hết nhiên liệu thì sao?")  # lần 2: thử bổ sung -> 429 -> tạm ngừng
        self.ask("Đội thắng được xác định thế nào?")     # không thử lại trong thời gian nghỉ
        self.assertEqual(embedder.document_calls, 2)

    def test_query_embedding_quota_falls_back_to_keywords(self):
        embedder = FailingEmbedder(fail_documents=False, fail_queries=True)
        self.use_embedder(embedder)
        self.assertEqual(self.upload().json()["search"], "hybrid")
        result = self.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")[-1]
        self.assertTrue(result["sources"])
        self.assertTrue(any("Tìm kiếm ngữ nghĩa tạm thời" in w for w in result["warnings"]))

    def test_documents_reembedded_with_current_model(self):
        self.upload()  # embedding bằng "fake-embedding"
        other = FakeEmbedder()
        other.model = "model-khac"
        self.use_embedder(other)
        result = self.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")[-1]
        self.assertEqual(result["type"], "result")
        listed = self.client.get(f"/api/conversations/{CONVERSATION}/documents").json()
        self.assertEqual(listed[0]["embedding_model"], "model-khac")
        self.assertEqual(other.query_calls, 1)


def mock_http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestProviders(unittest.TestCase):

    def test_fallback_llm_switches_and_rests_provider(self):
        calls = []

        class Provider:
            def __init__(self, name, error=None):
                self.name, self.error = name, error

            def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
                calls.append(self.name)
                if self.error:
                    raise self.error
                return f"từ {self.name}"

        llm = FallbackLLM([("gpt-oss-120b", Provider("gpt-oss-120b", ProviderError("groq", 429, "quota"))),
                           ("llama-70b", Provider("llama-70b"))])
        self.assertEqual(llm.generate("hỏi"), "từ llama-70b")
        self.assertEqual(llm.generate("hỏi lần 2"), "từ llama-70b")
        self.assertEqual(calls, ["gpt-oss-120b", "llama-70b", "llama-70b"])  # model hết quota đang nghỉ nên được thử sau

        broken = FallbackLLM([("a", Provider("a", ProviderError("a", 500, "x"))),
                              ("b", Provider("b", ProviderError("b", 429, "y")))])
        with self.assertRaises(ProviderError) as caught:
            broken.generate("hỏi")
        self.assertEqual(caught.exception.code, 429)

    def test_openai_compatible_client_request_and_errors(self):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append((str(request.url), request.headers["authorization"], body))
            if len(seen) == 1 and "reasoning_effort" in body:
                return httpx.Response(400, text="reasoning_effort is not supported")
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

        client = OpenAICompatibleClient("groq", "https://api.groq.com/openai/v1", "k123", "openai/gpt-oss-120b",
                                        thinking_level="high", http_client=mock_http(handler))
        text = client.generate("câu hỏi", system="Hệ thống", json_schema={"type": "object"})
        self.assertEqual(text, '{"ok": true}')

        url, auth, body = seen[0]
        self.assertEqual(url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(auth, "Bearer k123")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertIn('"type": "object"', body["messages"][0]["content"])
        self.assertNotIn("reasoning_effort", seen[1][2])  # gọi lại không kèm tham số bị từ chối

        limited = OpenAICompatibleClient("groq", "https://x/v1", "k", "llama", http_client=mock_http(
            lambda request: httpx.Response(429, text="rate limit reached")))
        with self.assertRaises(ProviderError) as caught:
            limited.generate("hỏi")
        self.assertEqual(caught.exception.code, 429)

    def test_create_llm_groq_only(self):
        from unittest import mock
        base = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY", "GROQ_MODELS")}

        with mock.patch.dict(os.environ, base, clear=True):
            with self.assertRaises(ValueError) as caught:
                create_llm()
            self.assertIn("GROQ_API_KEY", str(caught.exception))

        with mock.patch.dict(os.environ, dict(base, GROQ_API_KEY="g"), clear=True):
            # Mặc định: nhiều model Groq dự phòng cho nhau (hạn mức tính riêng từng model)
            self.assertEqual(describe_llm(create_llm()), [
                "groq/openai/gpt-oss-120b", "groq/llama-3.3-70b-versatile",
                "groq/openai/gpt-oss-20b", "groq/llama-3.1-8b-instant",
            ])
            self.assertEqual(describe_llm(create_llm(models="llama-3.3-70b-versatile")), ["groq/llama-3.3-70b-versatile"])

        with mock.patch.dict(os.environ, dict(base, GROQ_API_KEY="g", GROQ_MODELS=" a , b, a ,"), clear=True):
            self.assertEqual(describe_llm(create_llm()), ["groq/a", "groq/b"])

    def test_create_llm_with_several_keys(self):
        from unittest import mock
        base = {k: v for k, v in os.environ.items() if not k.startswith("GROQ_")}

        # Key thứ hai: cùng model thử lần lượt từng key, rồi mới sang model sau; key trùng / trống bị bỏ
        keys = dict(base, GROQ_API_KEY="gsk_one", GROQ_API_KEY_2="gsk_two", GROQ_API_KEY_3="gsk_one", GROQ_API_KEY_4=" ")
        with mock.patch.dict(os.environ, keys, clear=True):
            llm = create_llm(models="a, b")
            self.assertEqual(llm.names, ["groq/a", "groq/a#2", "groq/b", "groq/b#2"])
            self.assertEqual([client._headers["Authorization"] for _, client in llm.providers],
                             ["Bearer gsk_one", "Bearer gsk_two"] * 2)
            self.assertEqual(describe_llm(llm), ["groq/a", "groq/b"])

        # Chỉ có key thứ hai cũng chạy được
        with mock.patch.dict(os.environ, dict(base, GROQ_API_KEY_2="gsk_two"), clear=True):
            self.assertEqual(create_llm(models="a").names, ["groq/a"])

        # Key sai định dạng: báo tên biến, không lộ giá trị
        with mock.patch.dict(os.environ, dict(base, GROQ_API_KEY="gsk_one", GROQ_API_KEY_2="gsk sai"), clear=True):
            with self.assertRaises(ValueError) as caught:
                create_llm(models="a")
            self.assertIn("GROQ_API_KEY_2", str(caught.exception))
            self.assertNotIn("gsk sai", str(caught.exception))

        with mock.patch.dict(os.environ, dict(base, GROQ_API_KEY_2="gsk_secret_value_2"), clear=True):
            from RAG.gen import redact_secrets
            self.assertNotIn("gsk_secret_value_2", redact_secrets("lỗi với key gsk_secret_value_2"))

    def test_fallback_waits_when_every_model_is_briefly_limited(self):
        attempts = []

        class Flaky:
            def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
                attempts.append(time.time())
                if len(attempts) == 1:
                    raise ProviderError("groq", 429, "rate limit", retry_after=0.05)
                return "ok"

        llm = FallbackLLM([("groq/only", Flaky())])
        self.assertEqual(llm.generate("hỏi"), "ok")
        self.assertEqual(len(attempts), 2)
        self.assertGreaterEqual(attempts[1] - attempts[0], 0.04)
        self.assertEqual(llm.last_used, "groq/only")
        self.assertGreaterEqual(llm.last_wait, 0.05)

    def test_pinned_model_waits_several_times_within_limits(self):
        class Limited:
            def __init__(self, failures):
                self.failures = failures

            def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
                if self.failures:
                    self.failures -= 1
                    raise ProviderError("groq", 429, "rate limit", retry_after=0.02)
                return "ok"

        llm = FallbackLLM([("groq/only", Limited(3))], max_retries=4)
        self.assertEqual(llm.generate("hỏi"), "ok")
        self.assertGreaterEqual(llm.last_wait, 0.06)
        self.assertEqual(llm.generate("hỏi lần 2"), "ok")
        self.assertEqual(llm.last_wait, 0.0)

        # Cần chờ lâu hơn max_wait_seconds, hoặc chờ sẽ vượt deadline: báo lỗi ngay
        too_long = FallbackLLM([("groq/only", Limited(1))], max_wait_seconds=0.01, max_retries=4)
        with self.assertRaises(ProviderError):
            too_long.generate("hỏi")
        late = FallbackLLM([("groq/only", Limited(1))], max_retries=4, deadline=time.time())
        with self.assertRaises(ProviderError):
            late.generate("hỏi")
        self.assertEqual(late.last_wait, 0.0)

    def test_rate_limit_rest_uses_retry_after(self):
        client = OpenAICompatibleClient("groq", "https://x/v1", "k", "m1", http_client=mock_http(
            lambda request: httpx.Response(429, headers={"retry-after": "7"}, text="rate limit")))
        with self.assertRaises(ProviderError) as caught:
            client.generate("hỏi")
        self.assertEqual(caught.exception.retry_after, 7.0)

        backup = OpenAICompatibleClient("groq", "https://x/v1", "k", "m2", http_client=mock_http(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})))
        llm = FallbackLLM([("groq/m1", client), ("groq/m2", backup)], cooldown_seconds=999)
        self.assertEqual(llm.generate("hỏi"), "ok")
        rest = llm._resting_until["groq/m1"] - time.time()
        self.assertTrue(5 < rest <= 7.5)

    def test_cloudflare_embedder(self):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append((str(request.url), body))
            return httpx.Response(200, json={"success": True, "result": {
                "shape": [len(body["text"]), 3], "data": [[3.0, 4.0, 0.0] for _ in body["text"]]}})

        embedder = CloudflareEmbedder(account_id="0123456789abcdef0123456789abcdef", api_token="tok", batch_size=2, http_client=mock_http(handler))
        vectors = embedder.embed_documents(["a", "b", "c"])
        self.assertEqual(vectors.shape, (3, 3))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
        self.assertEqual(seen[0][0], "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/run/@cf/baai/bge-m3")
        self.assertEqual([len(body["text"]) for _, body in seen], [2, 1])
        self.assertTrue(seen[0][1]["truncate_inputs"])

        broken = CloudflareEmbedder(account_id="0123456789abcdef0123456789abcdef", api_token="tok", http_client=mock_http(
            lambda request: httpx.Response(200, json={"result": {"data": []}})))
        with self.assertRaises(ProviderError):
            broken.embed_queries(["q"])

    def test_create_embedder_selection(self):
        from unittest import mock
        base = {k: v for k, v in os.environ.items() if k not in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")}
        with mock.patch.dict(os.environ, base, clear=True):
            self.assertIsNone(create_embedder("auto"))  # không có Cloudflare -> chỉ tìm từ khoá
            with self.assertRaises(ValueError):
                create_embedder("cloudflare")
        with mock.patch.dict(os.environ, dict(base, CLOUDFLARE_ACCOUNT_ID="0123456789abcdef0123456789abcdef", CLOUDFLARE_API_TOKEN="t"), clear=True):
            self.assertIsInstance(create_embedder("auto"), CloudflareEmbedder)
            self.assertIsNone(create_embedder("none"))


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


class TestSecrets(unittest.TestCase):
    """Key / token dán sai định dạng và việc che bí mật trong thông báo lỗi."""

    TOKEN = "cfut_FAKE0123456789abcdefTOKEN"
    CURL = ('curl "https://api.cloudflare.com/client/v4/user/tokens/verify" \\\n'
            ' -H "Authorization: Bearer cfut_FAKE0123456789abcdefTOKEN"')
    ACCOUNT = "0123456789abcdef0123456789abcdef"

    def test_clean_secret_extracts_token(self):
        from RAG.gen import clean_secret
        for value in (self.TOKEN, f"  {self.TOKEN}\n", f'"{self.TOKEN}"', f"Bearer {self.TOKEN}", self.CURL):
            self.assertEqual(clean_secret(value), self.TOKEN)

    def test_cloudflare_embedder_accepts_pasted_curl_and_dashboard_url(self):
        seen = []

        def handler(request):
            seen.append((str(request.url), request.headers["authorization"]))
            return httpx.Response(200, json={"result": {"data": [[1.0, 0.0]]}})

        embedder = CloudflareEmbedder(account_id=f"https://dash.cloudflare.com/{self.ACCOUNT}/home",
                                      api_token=self.CURL, http_client=mock_http(handler))
        embedder.embed_queries(["q"])
        self.assertEqual(seen[0][0], f"https://api.cloudflare.com/client/v4/accounts/{self.ACCOUNT}/ai/run/@cf/baai/bge-m3")
        self.assertEqual(seen[0][1], f"Bearer {self.TOKEN}")

    def test_invalid_values_rejected_without_leaking(self):
        with self.assertRaises(ValueError) as caught:
            CloudflareEmbedder(account_id=self.ACCOUNT, api_token="token có khoảng trắng")
        self.assertIn("CLOUDFLARE_API_TOKEN không đúng định dạng", str(caught.exception))
        self.assertNotIn("khoảng trắng", str(caught.exception).split(":")[0])
        with self.assertRaises(ValueError) as caught:
            CloudflareEmbedder(account_id="không-phải-id", api_token=self.TOKEN)
        self.assertIn("CLOUDFLARE_ACCOUNT_ID", str(caught.exception))
        self.assertNotIn(self.TOKEN, str(caught.exception))

    def test_groq_key_cleaned(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": f"Bearer gsk_{self.ACCOUNT}"}):
            llm = create_llm(models="m")
            self.assertEqual(llm.providers[0][1]._headers["Authorization"], f"Bearer gsk_{self.ACCOUNT}")
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk abc def"}):
            with self.assertRaises(ValueError):
                create_llm(models="m")

    def test_error_messages_redact_secrets(self):
        from unittest import mock
        from RAG.gen import redact_secrets
        from webapp.service import describe_error
        with mock.patch.dict(os.environ, {"CLOUDFLARE_API_TOKEN": self.CURL}):
            leaked = f"Illegal header value b'Bearer {self.TOKEN}' / {self.TOKEN}"
            self.assertNotIn(self.TOKEN, redact_secrets(leaked))
            self.assertNotIn(self.TOKEN, describe_error(ValueError(leaked)))

    def test_health_reports_embedding_config_error(self):
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        service = WebRAGService(WebSettings(local_dataset_dir=tmp), LocalStorage(tmp), llm=FakeLLM(),
                                prompts=load_prompts(WEB_PROMPT_FILE))
        set_service(service)
        try:
            with mock.patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": self.ACCOUNT, "CLOUDFLARE_API_TOKEN": "sai định dạng"}):
                health = TestClient(app).get("/api/health").json()
            self.assertIsNone(health["embedding_model"])
            self.assertIn("CLOUDFLARE_API_TOKEN không đúng định dạng", health["embedding_error"])
        finally:
            set_service(None)
            shutil.rmtree(tmp, ignore_errors=True)


class TestHistoryPrompt(unittest.TestCase):

    def test_long_history_is_clipped(self):
        from RAG.reasoning import format_history
        text = format_history([("hỏi " * 10, "trả lời rất dài " * 200)], max_chars=100)
        self.assertLess(len(text), 300)
        self.assertTrue(text.endswith("…"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
