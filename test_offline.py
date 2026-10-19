"""
Test offline cho hệ thống RAG: không cần API key, không cần tải model.

Dùng embedding giả (DeterministicFakeEmbedding) và LLM giả (FakeLLM) để kiểm tra:
1. Chia chunk (overlap, nối qua trang, gạch đầu dòng, câu quá dài)
2. Build / nạp index và tự phát hiện index cũ
3. Tìm kiếm nhiều truy vấn + rerank
4. Phân tích câu hỏi, trả lời có suy luận, kiểm chứng trích dẫn
5. Pipeline hoàn chỉnh kèm lịch sử hội thoại

Chạy: python test_offline.py

Version: 1.0.0
"""

import json
import shutil
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "RAG_system"))

# Console Windows mặc định không in được tiếng Việt
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from read_file import TextChunker
from RAG.gen import load_prompts
from RAG.ingest import PDF_Ingestor
from RAG.pipeline import RAGPipeline
from RAG.reasoning import (
    GroundingChecker,
    QueryAnalyzer,
    ReasoningAnswerer,
    parse_json,
)
from RAG.reranking import Reranking
from RAG.settings import RAGSettings

PDF_PATH = PROJECT_ROOT / "dataset" / "Đề thi Procon_Tiếng Việt.pdf"

DOCS = [
    Document(
        page_content="Núi: thời gian di chuyển 3 bước, tiêu thụ 2 nhiên liệu.",
        metadata={"source": "bang1.md", "page_number": None, "page_end": None, "chunk_id": 0},
    ),
    Document(
        page_content="Nếu xe tuần tra hết nhiên liệu, nó phải đợi cho đến khi được tiếp nhiên liệu.",
        metadata={"source": "de.pdf", "page_number": 5, "page_end": 5, "chunk_id": 1},
    ),
    Document(
        page_content="Đội thu thập được nhiều loại udon khác nhau nhất trong một trận đấu là đội chiến thắng.",
        metadata={"source": "de.pdf", "page_number": 6, "page_end": 7, "chunk_id": 2},
    ),
]


class FakeLLM:
    """LLM giả: trả lần lượt các câu trả lời định sẵn và ghi lại các lần gọi."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, system=None, json_schema=None, thinking_level=None):
        self.calls.append({"prompt": prompt, "system": system, "schema": json_schema})
        return self.responses.pop(0)


class KeywordReranker:
    """Reranker giả: điểm = số từ của câu hỏi xuất hiện trong đoạn văn."""

    def compute_score(self, pairs):
        return [sum(word in passage.lower() for word in query.lower().split()) for query, passage in pairs]


def make_vector_db():
    return FAISS.from_documents(DOCS, DeterministicFakeEmbedding(size=32))


class TestChunker(unittest.TestCase):

    def test_chunks_respect_size_and_overlap(self):
        chunker = TextChunker(chunk_size=120, overlap=30)
        text = " ".join(f"Câu số {i} nói về luật di chuyển của xe tuần tra." for i in range(20))
        chunks = chunker.create_chunks(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        for previous, current in zip(chunks, chunks[1:]):
            # Đầu chunk sau lặp lại một phần đuôi (<= overlap ký tự) của chunk trước
            overlap_words = current.split()[:3]
            self.assertIn(" ".join(overlap_words), previous[-60:])

    def test_chunk_joins_sentence_across_pages(self):
        chunker = TextChunker(chunk_size=500, overlap=50)
        pages = [
            {"page_number": 1, "content": "Mỗi ô sẽ tiếp giáp với các"},
            {"page_number": 2, "content": "ô khác theo sáu hướng. Bản đồ rộng tối đa 32 ô."},
        ]
        chunks = chunker.create_chunks_from_pages(pages)

        self.assertEqual(len(chunks), 1)
        self.assertIn("tiếp giáp với các ô khác theo sáu hướng", chunks[0]["content"])
        self.assertEqual((chunks[0]["page_number"], chunks[0]["page_end"]), (1, 2))

    def test_split_bullets_and_long_sentence(self):
        chunker = TextChunker(chunk_size=50, overlap=0)
        units = chunker.split_by_sentences("Bản đồ • Mỗi ô có địa hình • Ao không đi vào được ・")
        self.assertEqual(units, ["Bản đồ", "• Mỗi ô có địa hình", "• Ao không đi vào được"])

        long_sentence = "từ " * 100
        chunks = chunker.create_chunks(long_sentence)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), long_sentence.split())


class TestIngest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_dir = self.tmp / "data"
        self.data_dir.mkdir()
        shutil.copy(PDF_PATH, self.data_dir / PDF_PATH.name)
        (self.data_dir / "bo_sung.md").write_text(
            "## Bảng 1\n\nĐồng bằng: 2 bước, 1 nhiên liệu. Núi: 3 bước, 2 nhiên liệu.", encoding="utf-8"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_ingestor(self, chunk_size=500):
        return PDF_Ingestor(
            pdf_dir=self.data_dir,
            index_dir=self.tmp / "index",
            chunk_size=chunk_size,
            overlap=100,
            embeddings=DeterministicFakeEmbedding(size=32),
        )

    def test_load_documents_reads_pdf_and_markdown(self):
        documents = self.make_ingestor().load_documents()
        sources = {document.metadata["source"] for document in documents}

        self.assertEqual(sources, {PDF_PATH.name, "bo_sung.md"})
        pdf_docs = [d for d in documents if d.metadata["source"] == PDF_PATH.name]
        self.assertTrue(all(1 <= d.metadata["page_number"] <= 7 for d in pdf_docs))
        self.assertTrue(any("HEXUDON" in d.page_content for d in pdf_docs))
        md_doc = next(d for d in documents if d.metadata["source"] == "bo_sung.md")
        self.assertIsNone(md_doc.metadata["page_number"])

    def test_index_freshness_and_auto_rebuild(self):
        ingestor = self.make_ingestor()
        self.assertFalse(ingestor.is_index_fresh())

        vector_db = ingestor.build_index(save=True)
        self.assertTrue(ingestor.is_index_fresh())
        self.assertEqual(ingestor.load_index().index.ntotal, vector_db.index.ntotal)

        # Đổi tham số chunk -> index cũ không còn khớp
        self.assertFalse(self.make_ingestor(chunk_size=300).is_index_fresh())

        # Sửa file nguồn -> index cũ không còn khớp, load_index tự build lại
        (self.data_dir / "bo_sung.md").write_text("Nội dung mới.", encoding="utf-8")
        self.assertFalse(ingestor.is_index_fresh())
        with self.assertRaises(RuntimeError):
            ingestor.load_index(auto_rebuild=False)
        ingestor.load_index()
        self.assertTrue(ingestor.is_index_fresh())


class TestReranking(unittest.TestCase):

    def test_multi_query_dedup_and_rerank(self):
        vector_db = make_vector_db()
        searcher = Reranking("xe tuần tra hết nhiên liệu")

        initial = searcher.get_initial_results(vector_db, k=3, queries=["nhiên liệu", "xe tuần tra"])
        self.assertEqual(len(initial), len(DOCS))  # 2 truy vấn x 3 kết quả, đã bỏ trùng

        ranked = searcher.search_with_reranking(vector_db, KeywordReranker(), initial_k=3, top_n=1)
        self.assertIn("hết nhiên liệu", ranked[0][0].page_content)

        without_reranker = searcher.search_with_reranking(vector_db, None, initial_k=3, top_n=2)
        self.assertEqual(len(without_reranker), 2)
        self.assertIsNone(without_reranker[0][1])

        # Giới hạn số ứng viên đưa vào reranker
        scored_pairs = []
        class CountingReranker(KeywordReranker):
            def compute_score(self, pairs):
                scored_pairs.extend(pairs)
                return super().compute_score(pairs)
        searcher.search_with_reranking(
            vector_db, CountingReranker(), initial_k=3, top_n=1,
            queries=["nhiên liệu", "xe tuần tra"], max_candidates=2,
        )
        self.assertEqual(len(scored_pairs), 2)


class TestReasoning(unittest.TestCase):

    def setUp(self):
        self.prompts = load_prompts()

    def test_parse_json_handles_code_fence(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('Kết quả: {"a": 2} xong'), {"a": 2})
        with self.assertRaises(ValueError):
            parse_json("không có json")

    def test_quote_matching(self):
        checker = GroundingChecker()
        passage = DOCS[1].page_content
        self.assertTrue(checker.quote_in_passage("nó phải đợi cho đến khi được tiếp nhiên liệu", passage))
        self.assertTrue(checker.quote_in_passage("  NẾU xe tuần tra   hết nhiên liệu ", passage))
        self.assertTrue(checker.quote_in_passage(unicodedata.normalize("NFD", "hết nhiên liệu"), passage))
        self.assertTrue(checker.quote_in_passage("Nếu xe tuần tra ... được tiếp nhiên liệu", passage))
        self.assertFalse(checker.quote_in_passage("xe tuần tra được tiếp nhiên liệu miễn phí tại trạm xăng", passage))

    def test_answer_verified(self):
        llm = FakeLLM([json.dumps({
            "evidence": [{"source_id": 1, "quote": "Núi: thời gian di chuyển 3 bước, tiêu thụ 2 nhiên liệu"}],
            "reasoning_steps": ["Theo [1], rời ô núi mất 3 bước và 2 nhiên liệu."],
            "answerable": True,
            "answer": "Đi qua núi mất 3 bước và 2 nhiên liệu [1].",
            "confidence": "cao",
        })])
        result = ReasoningAnswerer(llm, self.prompts).answer("Đi núi tốn gì?", DOCS)

        self.assertTrue(result.evidence[0].verified)
        self.assertEqual(result.evidence[0].source, "bang1.md")
        self.assertEqual(result.confidence, "cao")
        self.assertEqual(result.warnings, [])
        # Prompt gửi cho model có đánh số tài liệu và câu hỏi
        self.assertIn("[2] (de.pdf, trang 5)", llm.calls[0]["prompt"])
        self.assertIn("[3] (de.pdf, trang 6-7)", llm.calls[0]["prompt"])
        self.assertIn("Đi núi tốn gì?", llm.calls[0]["prompt"])
        self.assertIsNotNone(llm.calls[0]["schema"])

    def test_answer_with_fabricated_quote_is_flagged(self):
        llm = FakeLLM([json.dumps({
            "evidence": [{"source_id": 2, "quote": "Xe tuần tra được nạp nhiên liệu miễn phí ở trạm xăng"}],
            "reasoning_steps": ["..."],
            "answerable": True,
            "answer": "Xe được nạp ở trạm xăng [2][9].",
            "confidence": "cao",
        })])
        result = ReasoningAnswerer(llm, self.prompts).answer("Nạp nhiên liệu ở đâu?", DOCS)

        self.assertFalse(result.evidence[0].verified)
        self.assertEqual(result.confidence, "thấp")
        self.assertTrue(any("[9]" in warning for warning in result.warnings))
        self.assertTrue(any("Không có trích dẫn nào khớp" in warning for warning in result.warnings))

    def test_answer_fallbacks(self):
        llm = FakeLLM(["Câu trả lời dạng text thường"])
        result = ReasoningAnswerer(llm, self.prompts).answer("?", DOCS)
        self.assertEqual(result.answer, "Câu trả lời dạng text thường")
        self.assertTrue(result.warnings)

        result = ReasoningAnswerer(FakeLLM([]), self.prompts).answer("?", [])
        self.assertFalse(result.answerable)

    def test_query_analyzer(self):
        llm = FakeLLM([
            json.dumps({
                "category": "document",
                "standalone_question": "Xe tiếp nhiên liệu có tiêu thụ nhiên liệu không?",
                "sub_queries": ["xe tiếp nhiên liệu nhiên liệu", "Xe tiếp nhiên liệu có tiêu thụ nhiên liệu không?"],
                "direct_reply": "",
            }),
            json.dumps({"category": "lạ", "standalone_question": "", "sub_queries": [], "direct_reply": ""}),
            "không phải json",
        ])
        analyzer = QueryAnalyzer(llm, self.prompts)

        analysis = analyzer.analyze("còn xe tiếp nhiên liệu thì sao?", [("Xe tuần tra tốn nhiên liệu không?", "Có.")])
        self.assertEqual(analysis.standalone_question, "Xe tiếp nhiên liệu có tiêu thụ nhiên liệu không?")
        self.assertEqual(len(analysis.sub_queries), 2)  # câu hỏi độc lập luôn đứng đầu, bỏ trùng
        self.assertIn("Xe tuần tra tốn nhiên liệu không?", llm.calls[0]["prompt"])

        analysis = analyzer.analyze("câu hỏi")
        self.assertEqual((analysis.category, analysis.standalone_question), ("document", "câu hỏi"))

        analysis = analyzer.analyze("câu hỏi 2")
        self.assertEqual(analysis.sub_queries, ["câu hỏi 2"])


class TestPipeline(unittest.TestCase):

    def test_ask_and_history(self):
        llm = FakeLLM([
            # Lượt 1: câu hỏi về luật
            json.dumps({"category": "document", "standalone_question": "Xe tuần tra hết nhiên liệu thì sao?",
                        "sub_queries": ["xe tuần tra hết nhiên liệu"], "direct_reply": ""}),
            json.dumps({"evidence": [{"source_id": 1, "quote": "nó phải đợi cho đến khi được tiếp nhiên liệu"}],
                        "reasoning_steps": ["Theo [1] xe phải đợi."], "answerable": True,
                        "answer": "Xe phải đợi được tiếp nhiên liệu [1].", "confidence": "cao"}),
            # Lượt 2: chào hỏi -> không tìm kiếm, không gọi bước trả lời
            json.dumps({"category": "chitchat", "standalone_question": "cảm ơn",
                        "sub_queries": [], "direct_reply": "Rất vui được giúp bạn!"}),
        ])
        pipeline = RAGPipeline(
            settings=RAGSettings(top_n=2, initial_k=3),
            llm=llm,
            vector_db=make_vector_db(),
            reranker=KeywordReranker(),
        )

        response = pipeline.ask("Xe tuần tra hết nhiên liệu thì sao?")
        self.assertIn("hết nhiên liệu", response.documents[0][0].page_content)
        self.assertTrue(response.result.evidence[0].verified)
        self.assertEqual(len(response.documents), 2)

        response = pipeline.ask("cảm ơn")
        self.assertEqual(response.answer, "Rất vui được giúp bạn!")
        self.assertEqual(response.documents, [])
        self.assertEqual(len(llm.calls), 3)

        # Lịch sử lượt 1 được đưa vào prompt phân tích của lượt 2
        self.assertIn("Xe phải đợi được tiếp nhiên liệu", llm.calls[2]["prompt"])
        self.assertEqual(len(pipeline.history), 2)
        pipeline.reset()
        self.assertEqual(pipeline.history, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
