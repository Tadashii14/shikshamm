"""Upload helpers.

Reads multipart uploads in bounded chunks so an oversized file cannot be pulled
entirely into memory. Endpoints that accept PDFs/notes use this instead of a
single ``await file.read()`` (which would happily load an arbitrarily large
upload and risk exhausting memory on a small deployment).
"""
from __future__ import annotations

from fastapi import UploadFile

# 15 MB is generous for a lecture PDF/notes file while still bounding memory.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


async def read_limited(file: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read ``file`` up to ``limit`` bytes.

    Raises ``ValueError`` if the upload exceeds the limit; callers translate
    that into a 413 response.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ValueError(f"Uploaded file is too large (max {limit // (1024 * 1024)} MB)")
        chunks.append(chunk)
    return b"".join(chunks)
