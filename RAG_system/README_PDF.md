# Hướng dẫn dùng API của RAG_system trong code

Tài liệu này dành cho việc gọi các thành phần của hệ thống từ code Python. Cách cài đặt và chạy chatbot xem `README.md` ở thư mục gốc.

Mọi ví dụ giả định đã thêm `RAG_system` vào `sys.path` (như `chat.py` / `run.py`):

```python
import sys
sys.path.insert(0, "RAG_system")
```

---

## 1. Hỏi–đáp hoàn chỉnh (khuyên dùng)

```python
from RAG.pipeline import RAGPipeline

pipeline = RAGPipeline()          # đọc cấu hình từ .env, tự nạp / build index
response = pipeline.ask("Xe tuần tra hết nhiên liệu thì phải làm gì?")

print(response.answer)                    # câu trả lời có trích dẫn [n]
print(response.result.confidence)         # "cao" | "trung bình" | "thấp"
print(response.result.reasoning_steps)    # các bước suy luận
for item in response.result.evidence:     # bằng chứng + kết quả kiểm chứng
    print(item.source_id, item.verified, item.source, item.quote)
print(response.result.warnings)
print(response.analysis.standalone_question, response.analysis.sub_queries)
for document, score in response.documents:  # các đoạn tài liệu đã dùng
    print(score, document.metadata)

pipeline.ask("Còn xe tiếp nhiên liệu thì sao?")  # câu hỏi nối tiếp dùng lịch sử
pipeline.reset()                                # xoá lịch sử
```

Các tham số hay dùng của `RAGPipeline`:

| Tham số | Ý nghĩa |
|---|---|
| `settings=RAGSettings(top_n=3, thinking_level="high")` | Ghi đè cấu hình |
| `use_reranker=False` | Bỏ reranker (nhanh hơn trên CPU) |
| `use_query_analysis=False` | Bỏ bước phân tích câu hỏi |
| `llm=...`, `vector_db=...`, `reranker=...` | Truyền thành phần có sẵn (ví dụ LLM giả khi test, xem `test_offline.py`) |

Muốn dùng LLM khác Groq: viết class có hàm `generate(prompt, system=None, json_schema=None, thinking_level=None) -> str` rồi truyền vào `llm=`.

---

## 2. Đọc PDF / văn bản

```python
from read_file import Read_File_PDF, Read_File_Text

df = Read_File_PDF("dataset/Đề thi Procon_Tiếng Việt.pdf", chunk_size=500, overlap=100).Read
print(df.columns)   # ['chunk_id', 'content', 'page_number', 'page_end']

pages = Read_File_PDF("dataset/...pdf").Read_Pages   # List[{'page_number', 'content'}]
text = Read_File_PDF("dataset/...pdf").Read_Text     # toàn bộ text

df_md = Read_File_Text("dataset/bo_sung_bang_hinh_de_thi.md").Read   # page_number = None
```

Một chunk có thể nối qua hai trang (khi câu bị ngắt giữa trang). `page_number` là trang bắt đầu, `page_end` là trang kết thúc.

---

## 3. Build / nạp vector database

```python
from RAG.ingest import PDF_Ingestor

ingestor = PDF_Ingestor()                 # mặc định: dataset/ -> RAG_system/faiss_index/, model BAAI/bge-m3
vector_db = ingestor.load_index()         # nạp index; tự build lại nếu dữ liệu / cấu hình đã đổi
vector_db = ingestor.build_index()        # ép build lại
ingestor.is_index_fresh()                 # index có khớp dữ liệu hiện tại không

results = vector_db.similarity_search("số bước đi qua ô núi", k=5)
```

---

## 4. Tìm kiếm + rerank

```python
from RAG.reranking import Reranking, load_reranker, get_information

reranker = load_reranker()                # BAAI/bge-reranker-v2-m3 (sentence-transformers CrossEncoder)
ranked = Reranking("Xe tuần tra hết nhiên liệu thì sao?").search_with_reranking(
    VectorDB=vector_db,
    reranker=reranker,                    # None = giữ thứ tự vector search
    initial_k=5,                          # số kết quả cho mỗi truy vấn
    top_n=5,
    queries=["xe tuần tra hết nhiên liệu", "tiếp nhiên liệu"],   # tuỳ chọn: nhiều truy vấn con
    max_candidates=10,                    # tuỳ chọn: giới hạn số đoạn đưa vào reranker
)
for document, score in ranked:
    print(score, document.page_content[:100])

context = get_information("câu hỏi", vector_db, reranker, initial_k=10, top_n=5)   # gộp thành 1 chuỗi
```

`reranker` cũng nhận `FlagEmbedding.FlagReranker` (cùng hàm `compute_score`).

---

## 5. Từng bước reasoning

```python
from RAG.gen import create_llm, load_prompts
from RAG.reasoning import QueryAnalyzer, ReasoningAnswerer, GroundingChecker

llm = create_llm()                        # các model Groq trong GROQ_MODELS, key GROQ_API_KEY
prompts = load_prompts()                  # RAG_system/config_prompt_system.yaml

analysis = QueryAnalyzer(llm, prompts).analyze("còn xe tiếp nhiên liệu thì sao?",
                                               history=[("Xe tuần tra tốn nhiên liệu không?", "Có ...")])
result = ReasoningAnswerer(llm, prompts).answer(analysis.standalone_question, documents)

GroundingChecker().quote_in_passage("trích dẫn", "đoạn tài liệu")   # kiểm tra một trích dẫn
```

Sửa nội dung prompt trong `config_prompt_system.yaml`; các chỗ `{history}`, `{question}`, `{context}` được điền tự động.

---

## 6. Lưu ý

1. Thêm / sửa / xoá file trong `dataset/` thì index tự build lại ở lần chạy sau (hoặc chạy `python run.py`).
2. Đổi `EMBEDDING_MODEL`, `RAG_CHUNK_SIZE` hoặc `RAG_CHUNK_OVERLAP` cũng khiến index tự build lại.
3. `faiss_index/` được tạo tự động; xoá thư mục này cũng an toàn.
