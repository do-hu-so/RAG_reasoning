"""
Script test đơn giản: Chỉ test đọc PDF, không cần dependencies phức tạp.

Test:
1. Kiểm tra file PDF có tồn tại không
2. Đọc PDF bằng pypdf (nếu có)
3. Hiển thị thông tin cơ bản

Author: Physics Problem Solving System Team
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Console Windows mặc định không in được tiếng Việt
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")


def test_pdf_exists():
    """Test 1: Kiểm tra file PDF."""
    print("=" * 80)
    print("TEST 1: KIỂM TRA FILE PDF")
    print("=" * 80)

    dataset_dir = PROJECT_ROOT / "dataset"
    
    if not dataset_dir.exists():
        print(f"❌ Thư mục 'dataset' không tồn tại!")
        return False
    
    print(f"✓ Thư mục 'dataset' tồn tại")
    
    pdf_files = list(dataset_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ Không có file PDF nào trong thư mục!")
        return False
    
    print(f"✓ Tìm thấy {len(pdf_files)} file PDF:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  - {pdf.name} ({size_mb:.2f} MB)")
    
    return True


def test_read_pdf_basic():
    """Test 2: Đọc PDF cơ bản."""
    print("\n" + "=" * 80)
    print("TEST 2: ĐỌC NỘI DUNG PDF")
    print("=" * 80)
    
    try:
        import pypdf
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        print("✓ pypdf đã được cài đặt")
    except ImportError:
        print("❌ pypdf chưa được cài đặt")
        print("   Chạy: pip install pypdf")
        return False

    dataset_dir = PROJECT_ROOT / "dataset"
    pdf_files = list(dataset_dir.glob("*.pdf"))
    
    if not pdf_files:
        return False
    
    # Đọc file đầu tiên
    pdf_path = pdf_files[0]
    print(f"\nĐọc file: {pdf_path.name}")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            num_pages = len(reader.pages)
            
            print(f"✓ Số trang: {num_pages}")
            
            # Đọc trang đầu tiên
            if num_pages > 0:
                first_page = reader.pages[0]
                text = first_page.extract_text()
                
                print(f"\nNội dung trang đầu tiên (200 ký tự):")
                print("-" * 80)
                print(text[:200])
                print("-" * 80)
                
                print(f"\n✓ Đọc PDF thành công!")
                return True
                
    except Exception as e:
        print(f"❌ Lỗi khi đọc PDF: {e}")
        return False
    
    return False


def test_module_structure():
    """Test 3: Kiểm tra cấu trúc module RAG_system."""
    print("\n" + "=" * 80)
    print("TEST 3: KIỂM TRA CẤU TRÚC MODULE")
    print("=" * 80)
    
    rag_system = PROJECT_ROOT / "RAG_system"
    
    if not rag_system.exists():
        print("❌ Thư mục RAG_system không tồn tại!")
        return False
    
    print("✓ Thư mục RAG_system tồn tại")
    
    # Kiểm tra các file quan trọng
    important_files = [
        "read_file/__init__.py",
        "read_file/read_pdf.py",
        "RAG/__init__.py",
        "RAG/settings.py",
        "RAG/ingest.py",
        "RAG/reranking.py",
        "RAG/gen.py",
        "RAG/reasoning.py",
        "RAG/pipeline.py",
        "config_prompt_system.yaml",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in important_files:
        full_path = rag_system / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path} (không tồn tại)")
            all_exist = False
    
    return all_exist


def main():
    """Chạy tất cả các test."""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "TEST HỆ THỐNG RAG - PHIÊN BẢN ĐƠN GIẢN")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1
    results.append(("Kiểm tra file PDF", test_pdf_exists()))
    
    # Test 2
    results.append(("Đọc nội dung PDF", test_read_pdf_basic()))
    
    # Test 3
    results.append(("Kiểm tra cấu trúc module", test_module_structure()))
    
    # Kết quả tổng hợp
    print("\n" + "=" * 80)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 80)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("\n" + "=" * 80)
    print(f"Tổng: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n[OK] TAT CA CAC TEST DEU PASS!")
        print("\nBước tiếp theo:")
        print("1. Cài đặt dependencies: pip install -r RAG_system/requirements.txt")
        print("2. Chạy test offline: python test_offline.py")
        print("3. Build index và test tìm kiếm: python run.py")
        print("4. Chat với bot: python chat.py")
    else:
        print("\n[!] Một số test chưa pass. Vui lòng kiểm tra lại.")
    
    print()


if __name__ == "__main__":
    main()
