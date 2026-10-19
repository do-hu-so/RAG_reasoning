# RAG Reasoning

Hỏi–đáp tài liệu PDF bằng tiếng Việt theo kiến trúc **RAG có suy luận (reasoning)**: mỗi câu trả lời đi kèm bằng chứng trích nguyên văn, các bước suy luận, trích dẫn `[n]` về đúng đoạn / trang, và được **kiểm chứng tự động** với tài liệu.

Project có hai cách dùng:

| | Bản web (`app.py`) | Bản dòng lệnh (`chat.py`) |
|---|---|---|
| Giao diện | Chat kiểu ChatGPT, upload PDF ngay trong khung chat | Terminal |
| Tài liệu | PDF người dùng tải lên, riêng cho từng cuộc trò chuyện | Thư mục `dataset/` (đề thi PROCON) |
| Embedding | Gemini API (`gemini-embedding-2`) | `BAAI/bge-m3` chạy trên máy + reranker `bge-reranker-v2-m3` |
| Lưu trữ | `dataset_web/` trên máy, Vercel Blob khi deploy | FAISS index trong `RAG_system/faiss_index/` |
| Deploy | Vercel | Chạy trên máy |

Cả hai dùng chung lõi suy luận trong `RAG_system/RAG/reasoning.py`:

```text
Câu hỏi
  │ 1. Phân tích câu hỏi: phân loại, viết lại câu hỏi nối tiếp, tách truy vấn con
  ▼
  │ 2. Tìm các đoạn liên quan (embedding + cosine / FAISS + rerank)
  ▼
  │ 3. Suy luận (Gemini, thinking): bằng chứng nguyên văn -> các bước suy luận -> kết luận có trích dẫn [n]
  ▼
  │ 4. Kiểm chứng: trích dẫn có thật sự nằm trong tài liệu không -> độ tin cậy + cảnh báo
  ▼
Câu trả lời
```

---

## 1. Bản web

### Tính năng

- Giao diện chat giống ChatGPT, có thanh bên chứa các cuộc trò chuyện, chế độ sáng / tối, dùng tốt trên điện thoại.
- Bấm 📎 hoặc kéo thả PDF vào khung chat. Hệ thống đọc file, chia đoạn và tạo embedding; xong là hỏi được ngay. Có thể gõ câu hỏi trong lúc file đang xử lý, câu hỏi sẽ tự gửi khi file sẵn sàng.
- Sau khi upload, bot tóm tắt tài liệu và gợi ý 3 câu hỏi (bấm vào để hỏi luôn).
- **Thời gian phản hồi realtime:** đồng hồ chạy theo từng 0,1 giây trong lúc chờ, kèm thời gian của từng bước (phân tích câu hỏi → tìm đoạn liên quan → suy luận & kiểm chứng). Khi xong, câu trả lời hiện tổng thời gian; khối suy luận có bảng thời gian từng bước do server đo, cộng thêm phần mạng & hiển thị.
- Khối "Đã suy luận trong N giây" (mở ra để xem): câu hỏi được hiểu thế nào, truy vấn tìm kiếm, các bước suy luận, bằng chứng ✓ / ✗, thời gian xử lý.
- **Hộp "Trích dẫn văn bản gốc" cạnh câu trả lời:**
  - Màn hình rộng (≥ 1200px): hộp tự mở thành cột bên phải. Màn hình nhỏ: bấm "Văn bản gốc" để mở hộp nổi hoặc toàn màn hình.
  - Hộp hiện nguyên văn từng đoạn nguồn và **tô vàng đúng câu được trích làm bằng chứng**; đoạn được câu trả lời dùng có viền xanh.
  - Mỗi đoạn có nút mở PDF tại đúng trang.
  - Bấm chip `[n]` trong câu trả lời hoặc chip nguồn bên dưới sẽ cuộn tới đoạn tương ứng.
- Mỗi cuộc trò chuyện chỉ trả lời dựa trên các PDF đã tải lên trong cuộc trò chuyện đó. Panel "tài liệu" cho phép mở hoặc xoá từng file.

### Chạy trên máy

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env      # rồi điền GEMINI_API_KEY
uvicorn app:app --reload
```

Mở http://127.0.0.1:8000. File upload được lưu trong thư mục `dataset_web/`.

### Deploy lên Vercel

1. Vào https://vercel.com/new, chọn **Import Git Repository**, chọn repo này. Vercel tự nhận FastAPI qua `app.py` và cài `requirements.txt`. Bấm **Deploy**.
2. Trong project trên Vercel, mở tab **Storage**, tạo một **Blob** store, chọn chế độ **Private** rồi bấm **Connect** với project. Vercel sẽ tự thêm thông tin đăng nhập Blob vào project.
3. Mở **Settings → Environment Variables** và thêm `GEMINI_API_KEY`. Có thể thêm các biến tuỳ chọn ở mục 3.
4. Mở **Deployments**, chọn **Redeploy** để áp dụng biến môi trường.
5. Mở trang web. Góc dưới thanh bên sẽ hiện "Lưu trữ: Vercel Blob". Nếu chưa kết nối Blob hoặc thiếu API key, trang sẽ hiện thông báo màu vàng hướng dẫn cách sửa.

Nếu bạn tạo Blob store ở chế độ **Public**, cần đặt thêm biến `BLOB_ACCESS=public`.

### Dữ liệu được lưu như thế nào

```text
dataset_web/                        (thư mục trên máy, hoặc tiền tố dataset_web/ trong Vercel Blob)
└── <conversation_id>/
    └── <document_id>/
        ├── <tên file gốc>.pdf      file PDF đã upload
        ├── chunks.json             các đoạn văn + số trang
        ├── embeddings.npy          vector embedding của từng đoạn
        ├── meta.json               thông tin tài liệu (ghi sau cùng, đánh dấu đã xử lý xong)
        └── overview.json           tóm tắt + câu hỏi gợi ý
```

Lịch sử chat lưu trong trình duyệt (localStorage). Xoá một cuộc trò chuyện trên giao diện sẽ xoá luôn các file của nó trên server.

### Giới hạn

- **Mỗi PDF tối đa 4 MB**, vì Vercel giới hạn body của request ở 4.5 MB.
- **Tối đa 10 PDF** cho mỗi cuộc trò chuyện.
- **PDF scan dạng ảnh** (không có lớp chữ) sẽ bị từ chối; cần OCR trước khi tải lên.
- **Mỗi request chạy tối đa 300 giây** ở gói Hobby của Vercel, đủ cho tài liệu vài trăm trang.
- **Web để công khai:** ai có link cũng upload được file và dùng quota Gemini của bạn. Nếu cần hạn chế, bật **Deployment Protection** trong Settings của Vercel.

---

## 2. Bản dòng lệnh (đề thi PROCON)

Dùng model chạy trên máy nên chính xác hơn khi không có mạng, nhưng nặng (~4.5 GB model, cần torch).

```powershell
python -m pip install -r RAG_system\requirements.txt
python run.py      # build index FAISS từ dataset/ + thử tìm kiếm
python chat.py     # chat nhiều lượt; thêm --show-context để xem đoạn tài liệu đã dùng
```

| Lệnh | Tác dụng |
|---|---|
| `python chat.py -q "Xe tuần tra hết nhiên liệu thì sao?"` | Hỏi một câu rồi thoát |
| `--show-context` | In truy vấn con và các đoạn tài liệu đã dùng |
| `--hide-reasoning` | Ẩn các bước suy luận |
| `--no-rerank` | Bỏ reranker (nhanh hơn nhiều trên CPU) |
| `--no-analysis` | Bỏ bước phân tích câu hỏi |

Thư mục `dataset/` gồm file `Đề thi Procon_Tiếng Việt.pdf` và file `bo_sung_bang_hinh_de_thi.md`. File `.md` chép lại Bảng 1 và Hình 1–4 thành chữ, vì trong PDF chúng là ảnh. Index tự build lại khi file nguồn hoặc cấu hình thay đổi.

---

## 3. Cấu hình

Đặt trong `.env` khi chạy trên máy, hoặc trong Environment Variables của Vercel. Xem mẫu ở `.env.example`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GEMINI_API_KEY` | (bắt buộc) | API key Gemini, lấy tại https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model trả lời |
| `GEMINI_THINKING_LEVEL` | `medium` | Mức suy luận: `minimal`, `low`, `medium`, `high`, `off` |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Model embedding của bản web |
| `GEMINI_EMBEDDING_DIMENSIONS` | `768` | Số chiều vector của bản web |
| `BLOB_ACCESS` | `private` | Chế độ của Vercel Blob store |
| `MAX_UPLOAD_MB` | `4` | Dung lượng tối đa mỗi PDF |
| `WEB_CHUNK_SIZE` / `WEB_CHUNK_OVERLAP` / `WEB_TOP_N` | `800` / `150` / `6` | Chia đoạn và số đoạn đưa cho Gemini (bản web) |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` | Model của bản dòng lệnh |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` / `RAG_INITIAL_K` / `RAG_TOP_N` | `500` / `100` / `10` / `5` | Tham số của bản dòng lệnh |

Đổi model embedding hoặc số chiều vector của bản web thì các PDF đã upload trước đó phải tải lại.

---

## 4. Phần reasoning hoạt động thế nào

Mã nguồn: `RAG_system/RAG/reasoning.py`. Prompt: `RAG_system/config_prompt_web.yaml` (bản web) và `RAG_system/config_prompt_system.yaml` (bản dòng lệnh).

1. **Phân tích câu hỏi (`QueryAnalyzer`):**
   - Câu chào hỏi hoặc câu ngoài phạm vi được trả lời ngay, không tìm kiếm.
   - Câu hỏi nối tiếp được viết lại thành câu đầy đủ dựa trên 4 lượt hội thoại gần nhất.
   - Câu hỏi nhiều ý được tách thành tối đa 3 truy vấn con.
2. **Trả lời có suy luận (`ReasoningAnswerer`):** Gemini bắt buộc trả JSON theo thứ tự `evidence` (trích nguyên văn) → `reasoning_steps` → `answerable` → `answer` (có trích dẫn [n]) → `confidence`.
   - Nếu tài liệu không đề cập, bot nói rõ là không có thay vì bịa.
3. **Kiểm chứng (`GroundingChecker`):** so khớp từng trích dẫn với đoạn tài liệu được dẫn. Việc so khớp đã chuẩn hoá Unicode tiếng Việt và khoảng trắng, và cho phép lược bằng "...".
   - Trích dẫn bịa hoặc số [n] không tồn tại sẽ bị cảnh báo, và độ tin cậy bị hạ xuống.

---

## 5. Kiểm thử

```powershell
python test_web.py        # 12 test API web: upload, hỏi–đáp stream, lỗi, embedder (không cần API key)
python test_offline.py    # 13 test lõi RAG + bản dòng lệnh (không cần API key / model)
python test_simple.py     # kiểm tra nhanh file PDF và cấu trúc project
```

---

## 6. Xử lý lỗi thường gặp

- **Thanh vàng "Server chưa cấu hình GEMINI_API_KEY":** thêm biến môi trường rồi deploy lại (hoặc tạo `.env` khi chạy trên máy).
- **"Chưa kết nối Vercel Blob store":** làm bước 2 ở mục Deploy, rồi Redeploy.
- **Lỗi 404 / model không tồn tại:** Google thường xuyên ngừng model cũ. Đổi `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL` sang model đang được hỗ trợ (https://ai.google.dev/gemini-api/docs/models).
- **"Đã vượt giới hạn gọi Gemini API":** hết quota hoặc gọi quá nhanh; chờ vài phút hoặc nâng gói.
- **File quá 4 MB:** nén PDF hoặc tách thành nhiều file.

---

## 7. Cấu trúc thư mục

```text
RAG_reasoning/
├── app.py                          # API FastAPI của bản web (entrypoint Vercel)
├── public/                         # Giao diện chat (HTML/CSS/JS, không cần build)
│   ├── index.html
│   └── assets/ (app.js, styles.css, favicon.svg)
├── webapp/                         # Backend web
│   ├── config.py                   # Cấu hình bản web
│   ├── storage.py                  # Lưu trữ: thư mục dataset_web/ hoặc Vercel Blob
│   ├── embeddings.py               # Gemini Embedding API
│   └── service.py                  # Upload -> chunk -> embedding; hỏi–đáp có suy luận
├── RAG_system/
│   ├── RAG/                        # Lõi RAG: reasoning, gen (Gemini), settings, pipeline / ingest / reranking (CLI)
│   ├── read_file/                  # Đọc PDF, chia chunk
│   ├── config_prompt_web.yaml      # Prompt bản web
│   ├── config_prompt_system.yaml   # Prompt bản dòng lệnh
│   └── requirements.txt            # Thư viện bản dòng lệnh (torch, sentence-transformers, FAISS...)
├── dataset/                        # Tài liệu của bản dòng lệnh
├── chat.py / run.py                # Bản dòng lệnh
├── test_web.py / test_offline.py / test_simple.py
├── requirements.txt                # Thư viện bản web (Vercel cài file này)
├── .vercelignore
└── .env.example
```
