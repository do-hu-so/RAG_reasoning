"""
Module lưu trữ file cho bản web ("dataset web").

- LocalStorage: lưu vào thư mục dataset_web/ khi chạy trên máy
- BlobStorage: lưu vào Vercel Blob (tiền tố dataset_web/) khi deploy trên Vercel,
  vì function của Vercel không có ổ đĩa lưu lâu dài

Khoá (key) của mọi file có dạng "<conversation_id>/<document_id>/<tên file>".

Version: 1.0.0
"""

import os
from pathlib import Path
from typing import List, Protocol, Sequence

from .config import DATASET_WEB_NAME, WebSettings


class Storage(Protocol):
    """
    Interface lưu trữ tối thiểu mà service cần.
    """

    name: str

    def put(self, key: str, data: bytes, content_type: str) -> None:
        ...

    def get(self, key: str) -> bytes:
        """Raises FileNotFoundError khi không có file."""
        ...

    def list(self, prefix: str) -> List[str]:
        ...

    def delete(self, keys: Sequence[str]) -> None:
        ...


class LocalStorage:
    """
    Lưu file trong một thư mục trên máy (mặc định dataset_web/ ở gốc project).
    """

    name = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Đường dẫn không hợp lệ: {key}")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list(self, prefix: str) -> List[str]:
        if not self.root.exists():
            return []
        keys = (path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        return sorted(key for key in keys if key.startswith(prefix))

    def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            path = self._path(key)
            path.unlink(missing_ok=True)
            # Xoá các thư mục rỗng còn lại
            for parent in path.parents:
                if parent == self.root or not parent.exists() or any(parent.iterdir()):
                    break
                parent.rmdir()


class BlobStorage:
    """
    Lưu file trong Vercel Blob qua SDK chính thức (package vercel-blob).

    Thông tin đăng nhập do Vercel tự cung cấp khi store được kết nối với project
    (BLOB_STORE_ID + OIDC) hoặc lấy từ BLOB_READ_WRITE_TOKEN.
    """

    name = "vercel-blob"

    def __init__(self, access: str = "private", prefix: str = f"{DATASET_WEB_NAME}/") -> None:
        """
        Args:
            access: "private" hoặc "public", phải khớp chế độ của Blob store
            prefix: Thư mục gốc trong store
        """
        from vercel.blob import sync as blob

        if access not in ("private", "public"):
            raise ValueError("BLOB_ACCESS phải là 'private' hoặc 'public'")
        self._blob = blob
        self.access = access
        self.prefix = prefix

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._blob.put_blob(
            self.prefix + key,
            data,
            access=self.access,
            content_type=content_type,
            allow_overwrite=True,
        )

    def get(self, key: str) -> bytes:
        # BlobNotFoundError là lớp con của FileNotFoundError
        return self._blob.get_blob(self.prefix + key, access=self.access).body

    def list(self, prefix: str) -> List[str]:
        return [
            item.pathname[len(self.prefix):]
            for item in self._blob.query_blobs(prefix=self.prefix + prefix)
        ]

    def delete(self, keys: Sequence[str]) -> None:
        pathnames = [self.prefix + key for key in keys]
        for start in range(0, len(pathnames), 1000):
            self._blob.delete_blob(pathnames[start:start + 1000])


# Các biến mà SDK Vercel Blob dùng để xác thực (OIDC + store id, hoặc token đọc–ghi)
BLOB_CREDENTIAL_VARS = ("BLOB_STORE_ID", "BLOB_READ_WRITE_TOKEN", "VERCEL_BLOB_READ_WRITE_TOKEN")


def create_storage(settings: WebSettings) -> Storage:
    """
    Chọn nơi lưu trữ theo môi trường.

    Raises:
        RuntimeError: Khi chạy trên Vercel mà bản deploy chưa có thông tin Blob store
    """
    if any(os.getenv(name) for name in BLOB_CREDENTIAL_VARS):
        return BlobStorage(access=settings.blob_access)

    if os.getenv("VERCEL"):
        environment = os.getenv("VERCEL_ENV", "không rõ")
        # Chỉ liệt kê TÊN biến (không lộ giá trị) để dễ chẩn đoán
        related = sorted(name for name in os.environ if "BLOB" in name.upper())
        seen = f" Biến liên quan đến Blob đang có: {', '.join(related)}." if related else ""
        raise RuntimeError(
            f"Bản deploy này (môi trường {environment}) chưa có thông tin Vercel Blob (BLOB_STORE_ID). "
            "Nếu bạn đã Connect Blob store: biến môi trường chỉ áp dụng cho bản deploy mới, hãy vào "
            f"Deployments và bấm Redeploy; đồng thời kiểm tra store đã được Connect cho môi trường {environment}. "
            "Nếu chưa có store: vào tab Storage, tạo Blob store rồi Connect với project này." + seen
        )

    return LocalStorage(settings.local_dataset_dir)
