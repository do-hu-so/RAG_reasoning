"""
Module xếp hạng lại (reranking) cho bản web.

Vercel không chạy được cross-encoder (bge-reranker-v2-m3 của bản dòng lệnh), và
reranker duy nhất của Cloudflare (bge-reranker-base) chủ yếu cho tiếng Anh / Trung.
Vì vậy bản web dùng chính LLM (Groq) để chấm điểm mức độ liên quan của từng đoạn
với câu hỏi (LLM reranking), rồi giữ các đoạn điểm cao nhất.

Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Sequence

from RAG.gen import LLMClient, fill_template
from RAG.reasoning import parse_json

RERANK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["id", "score"],
            },
        },
    },
    "required": ["scores"],
}


class LLMReranker:
    """
    Chấm điểm 0-10 mức độ liên quan của từng đoạn với câu hỏi bằng LLM.

    Mỗi đoạn được cắt còn max_chars ký tự để cả danh sách vừa một lần gọi
    (Groq free tier giới hạn token mỗi phút).
    """

    name = "LLM (Groq)"

    def __init__(
        self,
        llm: LLMClient,
        prompts: Dict[str, str],
        thinking_level: Optional[str] = "low",
        max_chars: int = 400,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.thinking_level = thinking_level
        self.max_chars = max_chars

    def _clip(self, text: str) -> str:
        text = " ".join(text.split())
        return text if len(text) <= self.max_chars else text[: self.max_chars].rstrip() + "…"

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """
        Điểm 0-10 cho từng đoạn (đoạn model không chấm được tính 0).

        Raises:
            ValueError: Khi output của model không phải JSON
        """
        listing = "\n\n".join(f"[{index}] {self._clip(text)}" for index, text in enumerate(passages, start=1))
        raw = self.llm.generate(
            fill_template(self.prompts["rerank_user"], question=query, passages=listing),
            system=self.prompts["rerank_system"],
            json_schema=RERANK_SCHEMA,
            thinking_level=self.thinking_level,
        )
        data = parse_json(raw)

        scores = [0.0] * len(passages)
        for item in data.get("scores") or []:
            try:
                position = int(item["id"]) - 1
                value = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= position < len(passages):
                scores[position] = max(0.0, min(10.0, value))
        return scores
