"""
Chatbot hỏi–đáp luật thi PROCON bằng RAG + reasoning.

Cách chạy:
    python chat.py                                   # hội thoại nhiều lượt
    python chat.py -q "Xe tuần tra hết nhiên liệu thì sao?"
    python chat.py --show-context                    # in thêm các đoạn tài liệu đã dùng
    python chat.py --no-rerank                       # bỏ reranker cho máy yếu

Trong chế độ hội thoại: gõ /reset để xoá lịch sử, /exit để thoát.

Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path

# Thêm đường dẫn RAG_system vào sys.path
RAG_SYSTEM_PATH = Path(__file__).resolve().parent / "RAG_system"
sys.path.insert(0, str(RAG_SYSTEM_PATH))

# Console Windows mặc định không in được tiếng Việt
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")

from RAG.pipeline import RAGPipeline, RAGResponse
from RAG.reasoning import describe_source
from RAG.settings import RAGSettings


def print_response(response: RAGResponse, show_reasoning: bool, show_context: bool) -> None:
    """In câu trả lời, quá trình suy luận và nguồn."""
    result = response.result
    analysis = response.analysis

    if analysis.standalone_question != response.question:
        print(f"  (hiểu câu hỏi là: {analysis.standalone_question})")

    print(f"\nTrả lời: {result.answer}")

    if analysis.category != "document":
        return

    print(f"Độ tin cậy: {result.confidence}")

    if show_reasoning and result.reasoning_steps:
        print("\nSuy luận:")
        for index, step in enumerate(result.reasoning_steps, start=1):
            print(f"  {index}. {step}")

    if result.evidence:
        print("\nBằng chứng:")
        for item in result.evidence:
            mark = "✓" if item.verified else "✗ không khớp tài liệu"
            print(f"  [{item.source_id}] {item.source or '?'} {mark}\n      \"{item.quote}\"")

    for warning in result.warnings:
        print(f"\n⚠ {warning}")

    if show_context and response.documents:
        print(f"\nTruy vấn tìm kiếm: {analysis.sub_queries}")
        print("Các đoạn tài liệu đã dùng:")
        for index, (document, score) in enumerate(response.documents, start=1):
            score_text = f"{score:.3f}" if score is not None else "-"
            print(f"  [{index}] ({describe_source(document)}, điểm rerank {score_text})")
            print(f"      {document.page_content[:300]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot hỏi–đáp luật thi PROCON (RAG + reasoning)")
    parser.add_argument("-q", "--question", help="Hỏi một câu rồi thoát")
    parser.add_argument("--show-context", action="store_true", help="In các đoạn tài liệu đã dùng")
    parser.add_argument("--hide-reasoning", action="store_true", help="Không in các bước suy luận")
    parser.add_argument("--no-rerank", action="store_true", help="Không dùng reranker (nhanh hơn, kém chính xác hơn)")
    parser.add_argument("--no-analysis", action="store_true", help="Bỏ bước phân tích câu hỏi")
    args = parser.parse_args()

    print("Đang khởi tạo hệ thống (lần đầu có thể mất vài phút để tải model)...")
    try:
        pipeline = RAGPipeline(
            settings=RAGSettings(),
            use_reranker=not args.no_rerank,
            use_query_analysis=not args.no_analysis,
        )
    except ValueError as error:
        print(f"Lỗi cấu hình: {error}")
        sys.exit(1)

    def ask(question: str) -> None:
        try:
            response = pipeline.ask(question)
        except Exception as error:
            print(f"\nLỗi khi trả lời: {error}")
            return
        print_response(response, show_reasoning=not args.hide_reasoning, show_context=args.show_context)

    if args.question:
        ask(args.question)
        return

    print("Sẵn sàng! Hỏi về luật thi PROCON (gõ /reset để xoá lịch sử, /exit để thoát).")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if question.lower() == "/reset":
            pipeline.reset()
            print("Đã xoá lịch sử hội thoại.")
            continue

        ask(question)


if __name__ == "__main__":
    main()
