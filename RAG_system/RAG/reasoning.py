"""
Module suy luận (reasoning) cho hệ thống RAG.

Module này chịu trách nhiệm:
- QueryAnalyzer: hiểu câu hỏi trước khi tìm kiếm
    + phân loại (câu hỏi về cuộc thi / chào hỏi / ngoài phạm vi)
    + viết lại câu hỏi nối tiếp thành câu hỏi độc lập dựa trên lịch sử hội thoại
    + tách câu hỏi phức tạp thành các truy vấn con để tìm kiếm
- ReasoningAnswerer: trả lời có suy luận từng bước
    + trích bằng chứng nguyên văn từ tài liệu
    + suy luận / tính toán từng bước dựa trên bằng chứng
    + kết luận kèm trích dẫn [n], tự đánh giá có đủ thông tin hay không
- GroundingChecker: kiểm chứng tự động (không cần gọi LLM)
    + mỗi trích dẫn có thực sự nằm trong đoạn tài liệu được dẫn hay không
    + hạ độ tin cậy và cảnh báo khi câu trả lời không bám vào tài liệu

Version: 1.0.0
"""

import difflib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from .gen import LLMClient, fill_template

logger = logging.getLogger(__name__)

CATEGORIES = ("document", "chitchat", "out_of_scope")
CONFIDENCE_LEVELS = ("cao", "trung bình", "thấp")

NOT_FOUND_MESSAGE = "Mình không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này."
OUT_OF_SCOPE_MESSAGE = (
    "Mình chỉ trả lời các câu hỏi về luật thi PROCON (phần thi HEXUDON) "
    "dựa trên tài liệu đề thi được cung cấp."
)

# JSON schema bắt buộc model tuân theo. Thứ tự thuộc tính là thứ tự model sinh ra:
# bằng chứng -> suy luận -> kết luận, để model "nghĩ trước, trả lời sau".
QUERY_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "standalone_question": {"type": "string"},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "direct_reply": {"type": "string"},
    },
    "required": ["category", "standalone_question", "sub_queries", "direct_reply"],
}

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["source_id", "quote"],
            },
        },
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["cao", "trung_binh", "thap"]},
    },
    "required": ["evidence", "reasoning_steps", "answerable", "answer", "confidence"],
}


@dataclass
class QueryAnalysis:
    """
    Kết quả phân tích câu hỏi.
    """

    category: str
    standalone_question: str
    sub_queries: List[str]
    direct_reply: str = ""


@dataclass
class Evidence:
    """
    Một bằng chứng trích từ tài liệu.
    """

    source_id: int
    quote: str
    verified: bool = False
    source: str = ""
    page: Optional[int] = None


@dataclass
class ReasonedAnswer:
    """
    Câu trả lời kèm quá trình suy luận và kết quả kiểm chứng.
    """

    answer: str
    answerable: bool
    reasoning_steps: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    confidence: str = "thấp"
    warnings: List[str] = field(default_factory=list)


def parse_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON từ output của model, chịu được code fence ```json ... ```.

    Raises:
        ValueError: Khi không tìm thấy object JSON hợp lệ
    """
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise ValueError("Output của model không chứa JSON")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON không hợp lệ: {error}") from error

    if not isinstance(data, dict):
        raise ValueError("Output JSON của model không phải object")
    return data


def format_history(history: Sequence[Tuple[str, str]], max_chars: int = 600) -> str:
    """
    Định dạng lịch sử hội thoại để đưa vào prompt.

    Mỗi câu hỏi / câu trả lời chỉ giữ tối đa max_chars ký tự: đủ để hiểu câu hỏi
    nối tiếp mà không tốn nhiều token (Groq free tier giới hạn token mỗi phút).
    """
    if not history:
        return "(chưa có)"

    def clip(text: str) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"

    return "\n".join(f"Người dùng: {clip(question)}\nTrợ lý: {clip(answer)}" for question, answer in history)


def describe_source(document: Document) -> str:
    """
    Mô tả nguồn của một đoạn tài liệu, ví dụ "Đề thi.pdf, trang 3".
    """
    metadata = document.metadata
    source = metadata.get("source", "không rõ nguồn")
    start, end = metadata.get("page_number"), metadata.get("page_end")
    if start is None:
        return source
    if end is not None and end != start:
        return f"{source}, trang {start}-{end}"
    return f"{source}, trang {start}"


def format_context(documents: Sequence[Document]) -> str:
    """
    Đánh số các đoạn tài liệu [1], [2], ... để model trích dẫn.
    """
    return "\n\n".join(
        f"[{index}] ({describe_source(document)})\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )


def _normalize_confidence(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip().lower().replace("_", " ")
    mapping = {
        "cao": "cao", "high": "cao",
        "trung binh": "trung bình", "trung bình": "trung bình", "medium": "trung bình",
        "thap": "thấp", "thấp": "thấp", "low": "thấp",
    }
    return mapping.get(text, "thấp")


class QueryAnalyzer:
    """
    Class phân tích câu hỏi trước khi tìm kiếm.

    Class này chịu trách nhiệm:
    - Phân loại câu hỏi để không tìm kiếm vô ích với câu chào hỏi / ngoài phạm vi
    - Viết lại câu hỏi nối tiếp ("còn xe tiếp nhiên liệu thì sao?") thành câu hỏi độc lập
    - Tách câu hỏi nhiều ý thành các truy vấn con cho bước tìm kiếm
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts: Dict[str, str],
        max_sub_queries: int = 3,
        thinking_level: Optional[str] = "low",
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.max_sub_queries = max_sub_queries
        self.thinking_level = thinking_level

    def analyze(
        self,
        question: str,
        history: Sequence[Tuple[str, str]] = (),
        **context: str,
    ) -> QueryAnalysis:
        """
        Phân tích câu hỏi.

        Args:
            question: Câu hỏi mới của người dùng
            history: Các cặp (câu hỏi, câu trả lời) gần nhất
            context: Giá trị bổ sung cho prompt, ví dụ documents="..." điền vào {documents}

        Returns:
            QueryAnalysis; nếu model trả về sai định dạng thì dùng nguyên câu hỏi gốc
        """
        prompt = fill_template(
            self.prompts["query_analysis_user"],
            history=format_history(history),
            question=question,
            **context,
        )
        raw = self.llm.generate(
            prompt,
            system=self.prompts["query_analysis_system"],
            json_schema=QUERY_ANALYSIS_SCHEMA,
            thinking_level=self.thinking_level,
        )

        try:
            data = parse_json(raw)
        except ValueError as error:
            logger.warning("Không phân tích được câu hỏi (%s), dùng câu hỏi gốc.", error)
            return QueryAnalysis("document", question, [question])

        category = str(data.get("category", "")).strip().lower()
        if category not in CATEGORIES:
            category = "document"

        standalone = str(data.get("standalone_question") or "").strip() or question

        # Luôn tìm theo câu hỏi độc lập, cộng thêm các truy vấn con (bỏ trùng)
        queries: List[str] = []
        for query in [standalone, *(data.get("sub_queries") or [])]:
            query = str(query).strip()
            if query and query not in queries:
                queries.append(query)

        return QueryAnalysis(
            category=category,
            standalone_question=standalone,
            sub_queries=queries[: self.max_sub_queries + 1],
            direct_reply=str(data.get("direct_reply") or "").strip(),
        )


class GroundingChecker:
    """
    Class kiểm chứng câu trả lời có bám vào tài liệu hay không.

    Class này chịu trách nhiệm:
    - So khớp từng trích dẫn với đoạn tài liệu được dẫn (chuẩn hoá Unicode, khoảng trắng)
    - Phát hiện trích dẫn [n] không tồn tại trong câu trả lời
    - Hạ độ tin cậy và thêm cảnh báo khi bằng chứng không khớp
    """

    def __init__(self, min_match_ratio: float = 0.85) -> None:
        """
        Args:
            min_match_ratio: Tỉ lệ tối thiểu của trích dẫn phải khớp liên tục với tài liệu
        """
        self.min_match_ratio = min_match_ratio

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text).lower()
        text = re.sub(r"[“”„\"«»]", '"', text)
        text = re.sub(r"[‘’']", "'", text)
        return re.sub(r"\s+", " ", text).strip(" .\"'")

    def quote_in_passage(self, quote: str, passage: str) -> bool:
        """
        Kiểm tra trích dẫn có nằm trong đoạn tài liệu hay không.

        Trích dẫn có thể lược bớt bằng "..." / "…"; khi đó mỗi phần phải khớp.
        """
        passage_norm = self._normalize(passage)
        segments = [self._normalize(part) for part in re.split(r"\.\.\.|…", quote)]
        segments = [segment for segment in segments if len(segment) >= 5]
        if not segments:
            return False

        for segment in segments:
            if segment in passage_norm:
                continue
            matcher = difflib.SequenceMatcher(None, passage_norm, segment, autojunk=False)
            match = matcher.find_longest_match(0, len(passage_norm), 0, len(segment))
            if match.size / len(segment) < self.min_match_ratio:
                return False
        return True

    def check(self, result: ReasonedAnswer, documents: Sequence[Document]) -> ReasonedAnswer:
        """
        Kiểm chứng và cập nhật kết quả (verified, confidence, warnings).
        """
        for item in result.evidence:
            if 1 <= item.source_id <= len(documents):
                document = documents[item.source_id - 1]
                item.source = describe_source(document)
                item.page = document.metadata.get("page_number")
                item.verified = self.quote_in_passage(item.quote, document.page_content)
            else:
                item.verified = False

        cited = {int(number) for number in re.findall(r"\[(\d+)\]", result.answer)}
        invalid = sorted(number for number in cited if not 1 <= number <= len(documents))
        if invalid:
            result.warnings.append(
                "Câu trả lời trích dẫn nguồn không tồn tại: " + ", ".join(f"[{n}]" for n in invalid)
            )

        if not result.answerable:
            return result

        verified = sum(item.verified for item in result.evidence)
        if verified == 0:
            result.confidence = "thấp"
            result.warnings.append(
                "Không có trích dẫn nào khớp với tài liệu — câu trả lời có thể chưa chính xác, hãy đối chiếu lại đề."
            )
        elif verified < len(result.evidence):
            result.warnings.append(
                f"Có {len(result.evidence) - verified}/{len(result.evidence)} trích dẫn không khớp nguyên văn với tài liệu."
            )
            if result.confidence == "cao":
                result.confidence = "trung bình"

        return result


class ReasoningAnswerer:
    """
    Class trả lời câu hỏi bằng suy luận có căn cứ.

    Class này chịu trách nhiệm:
    - Đưa các đoạn tài liệu đã đánh số cho model
    - Buộc model trả về bằng chứng -> các bước suy luận -> kết luận (JSON)
    - Gọi GroundingChecker để kiểm chứng kết quả
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts: Dict[str, str],
        checker: Optional[GroundingChecker] = None,
        thinking_level: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.checker = checker or GroundingChecker()
        self.thinking_level = thinking_level

    def answer(self, question: str, documents: Sequence[Document]) -> ReasonedAnswer:
        """
        Trả lời câu hỏi dựa trên các đoạn tài liệu.

        Args:
            question: Câu hỏi (đã được viết lại thành câu hỏi độc lập)
            documents: Các đoạn tài liệu liên quan, theo thứ tự ưu tiên

        Returns:
            ReasonedAnswer đã được kiểm chứng
        """
        if not documents:
            return ReasonedAnswer(answer=NOT_FOUND_MESSAGE, answerable=False)

        prompt = fill_template(
            self.prompts["answer_user"],
            context=format_context(documents),
            question=question,
        )
        raw = self.llm.generate(
            prompt,
            system=self.prompts["answer_system"],
            json_schema=ANSWER_SCHEMA,
            thinking_level=self.thinking_level,
        )

        try:
            data = parse_json(raw)
        except ValueError:
            return ReasonedAnswer(
                answer=raw.strip(),
                answerable=True,
                warnings=["Model không trả về đúng định dạng suy luận; câu trả lời chưa được kiểm chứng."],
            )

        evidence = []
        for item in data.get("evidence") or []:
            try:
                evidence.append(Evidence(source_id=int(item["source_id"]), quote=str(item["quote"])))
            except (KeyError, TypeError, ValueError):
                continue

        result = ReasonedAnswer(
            answer=str(data.get("answer") or "").strip() or NOT_FOUND_MESSAGE,
            answerable=bool(data.get("answerable", True)),
            reasoning_steps=[str(step).strip() for step in data.get("reasoning_steps") or [] if str(step).strip()],
            evidence=evidence,
            confidence=_normalize_confidence(data.get("confidence")),
        )
        return self.checker.check(result, documents)
