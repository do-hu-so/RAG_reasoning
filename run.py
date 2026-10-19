"""
File build index và chạy thử tìm kiếm của hệ thống RAG.

Script này thực hiện:
1. Kiểm tra thư mục dataset và các file nguồn (PDF, .md, .txt)
2. Nạp dữ liệu vào vector database (build lại index)
3. Test tìm kiếm với vài câu hỏi mẫu về luật thi
4. Hiển thị kết quả

Author: Physics Problem Solving System Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Thêm đường dẫn RAG_system vào sys.path
RAG_SYSTEM_PATH = Path(__file__).resolve().parent / "RAG_system"
sys.path.insert(0, str(RAG_SYSTEM_PATH))

# Console Windows mặc định không in được tiếng Việt
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")

from RAG.ingest import PDF_Ingestor
from RAG.settings import RAGSettings

TEST_QUERIES = [
    "Đi qua ô núi tốn bao nhiêu bước và bao nhiêu nhiên liệu?",
    "Xe tuần tra hết nhiên liệu thì phải làm gì?",
    "Đội chiến thắng được xác định như thế nào?",
]


def main():
    """Hàm chính test hệ thống RAG."""
    
    print("=" * 80)
    print("TEST HỆ THỐNG RAG ĐỌC PDF")
    print("=" * 80)

    settings = RAGSettings()
    dataset_dir = settings.data_dir

    print(f"\n1. Kiểm tra thư mục dataset: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"   ❌ Thư mục dataset không tồn tại!")
        return

    print(f"   ✓ Thư mục tồn tại")

    # Khởi tạo Ingestor
    ingestor = PDF_Ingestor(
        pdf_dir=dataset_dir,
        index_dir=settings.index_dir,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        overlap=settings.overlap
    )

    # Liệt kê file nguồn
    source_files = ingestor.list_source_files()
    print(f"\n2. Tìm thấy {len(source_files)} file nguồn:")

    if not source_files:
        print("   ❌ Không tìm thấy file PDF / .md / .txt nào!")
        return

    for source_file in source_files:
        size_mb = source_file.stat().st_size / (1024 * 1024)
        print(f"   - {source_file.name} ({size_mb:.2f} MB)")

    print(f"\n3. Cấu hình: embedding = {settings.embedding_model}, "
          f"chunk_size = {settings.chunk_size}, overlap = {settings.overlap}")

    
    # Đọc và xử lý file nguồn
    print(f"\n4. Đọc và xử lý các file nguồn...")

    try:
        documents = ingestor.load_documents()
        print(f"   ✓ Đã xử lý {len(documents)} documents")
        
        if documents:
            print(f"\n   Ví dụ document đầu tiên:")
            first_doc = documents[0]
            print(f"   - Source: {first_doc.metadata.get('source', 'N/A')}")
            print(f"   - Page: {first_doc.metadata.get('page_number', 'N/A')}")
            print(f"   - Content preview: {first_doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"   ❌ Lỗi khi xử lý PDF: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Build vector database
    print(f"\n5. Xây dựng vector database...")
    
    try:
        vector_db = ingestor.build_index(save=True)
        print(f"   ✓ Vector database đã được xây dựng và lưu ({vector_db.index.ntotal} chunks)")
        print(f"   ✓ Index được lưu tại: {settings.index_dir}")
        
    except Exception as e:
        print(f"   ❌ Lỗi khi build vector database: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test tìm kiếm
    print(f"\n6. Test tìm kiếm với các câu hỏi mẫu...")

    try:
        for test_query in TEST_QUERIES:
            print(f"\n   Câu hỏi: '{test_query}'")
            results = vector_db.similarity_search(test_query, k=3)

            for i, doc in enumerate(results, 1):
                print(f"   Kết quả {i}: {doc.metadata.get('source', 'N/A')}, "
                      f"trang {doc.metadata.get('page_number') or '-'}")
                print(f"      {doc.page_content[:200]}...")

    except Exception as e:
        print(f"   ❌ Lỗi khi tìm kiếm: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("✓ TEST HOÀN TẤT THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
