r"""Focused regression tests for the completion-pass changes.

Covers upload-size enforcement, attendance statistics, and face proxy honesty.

Run with a temp DB, e.g.:
  set DATABASE_URL=sqlite:///./_audit_tmp.db
  .venv\Scripts\python.exe -m pytest tests/test_audit_targeted.py -q

No real attendai.db is touched; no face-recognition stack is required.
"""
import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db import init_db
from app.services.uploads import read_limited

init_db()  # a bare TestClient does not fire startup events; create tables (temp DB)
client = TestClient(app)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _png(n=64):
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * n)


class _FakeUpload:
    """Minimal UploadFile-like object exposing async read(size)."""
    def __init__(self, data=b"", size=None):
        if size is not None:
            self._left = size
            self._buf = None
        else:
            self._buf = io.BytesIO(data)
            self._left = None

    async def read(self, size=-1):
        if self._buf is not None:
            return self._buf.read(size)
        if self._left <= 0:
            return b""
        chunk = b"a" * (min(size, self._left) if size and size > 0 else self._left)
        self._left -= len(chunk)
        return chunk


# --------------------------------------------------------------------------
# 1. Upload size enforcement
# --------------------------------------------------------------------------
class TestUploadLimit:
    def test_read_limited_empty_file(self):
        assert _run(read_limited(_FakeUpload(b""), limit=1024)) == b""

    def test_read_limited_boundary(self):
        # exactly at the limit -> allowed
        assert _run(read_limited(_FakeUpload(b"a" * 1024), limit=1024)) == b"a" * 1024
        # one byte over -> rejected while reading
        with pytest.raises(ValueError):
            _run(read_limited(_FakeUpload(b"a" * 1025), limit=1024))

    def test_read_limited_enforced_before_full_load(self):
        # 5 MB payload under a 1 KB limit must raise without holding it all
        with pytest.raises(ValueError):
            _run(read_limited(_FakeUpload(size=5 * 1024 * 1024), limit=1024))

    def test_oversized_upload_returns_413(self, monkeypatch):
        import app.routers.advanced as adv

        async def tiny_read(file, limit=None):
            raise ValueError("Uploaded file is too large (max 15 MB)")

        monkeypatch.setattr(adv, "read_limited", tiny_read)
        r = client.post(
            "/api/advanced/pdf/extract-text",
            files={"file": ("big.pdf", io.BytesIO(b"%PDF-" + b"x" * 2048), "application/pdf")},
        )
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_valid_upload_still_works(self):
        r = client.post(
            "/notes/upload",
            data={"user_id": "1"},
            files={"file": ("n.txt", b"The cell is the basic unit of life. " * 5, "text/plain")},
        )
        assert r.status_code == 200
        assert r.json().get("note_id") is not None

    def test_advanced_endpoints_use_capped_reader(self):
        src = io.open("app/routers/advanced.py", encoding="utf-8").read()
        assert "await file.read()" not in src
        assert src.count("await _read_upload(file)") >= 5


# --------------------------------------------------------------------------
# 2. Student attendance statistics
# --------------------------------------------------------------------------
class TestAttendanceStats:
    def _setup(self, tag, total, attended):
        client.post("/admin/classes", params={"name": f"Stat {tag}", "code": f"ST{tag}"})
        r = client.post(
            "/admin/students/create",
            params={"email": f"stat{tag}@t.com", "full_name": f"Stat {tag}", "admission_number": f"STAT{tag}"},
        )
        assert r.status_code == 200
        uid = r.json()["id"]
        for i in range(total):
            code = f"S{tag}{i}"
            rs = client.post("/admin/sessions/start", params={"class_code": f"ST{tag}", "code": code, "subject": "M"})
            assert rs.status_code == 200
            if i < attended:
                rm = client.post("/admin/attendance/manual", json={"session_code": code, "user_id": uid, "status": "present"})
                assert rm.status_code == 200
        return uid

    def test_denominator_is_sessions_not_rows(self):
        """2 of 3 sessions must NOT report 100% (the old bug)."""
        uid = self._setup("A", total=3, attended=2)
        j = client.get(f"/stats/student/attendance?user_id={uid}").json()
        assert j["present"] == 2
        assert j["total_sessions"] >= 3
        expected = round(j["present"] / j["total_sessions"] * 100.0, 2)
        assert j["attendance_pct"] == expected
        assert j["attendance_pct"] < 100.0

    def test_all_attended(self):
        uid = self._setup("B", total=3, attended=3)
        j = client.get(f"/stats/student/attendance?user_id={uid}").json()
        assert j["present"] == 3
        assert j["attendance_pct"] == round(j["present"] / j["total_sessions"] * 100.0, 2)

    def test_no_attendance_is_zero(self):
        uid = self._setup("C", total=2, attended=0)
        j = client.get(f"/stats/student/attendance?user_id={uid}").json()
        assert j["present"] == 0
        assert j["attendance_pct"] == 0.0

    def test_unknown_user_404(self):
        assert client.get("/stats/student/attendance?user_id=999999").status_code == 404


# --------------------------------------------------------------------------
# 3. Face proxy-mode honesty
# --------------------------------------------------------------------------
class TestFaceProxyHonesty:
    def _active(self, tag):
        client.post("/admin/classes", params={"name": f"F {tag}", "code": f"FC{tag}"})
        r = client.post(
            "/admin/students/create",
            params={"email": f"f{tag}@t.com", "full_name": f"F {tag}", "admission_number": f"FADM{tag}"},
        )
        uid = r.json()["id"]
        client.post("/admin/sessions/start", params={"class_code": f"FC{tag}", "code": f"FS{tag}", "subject": "M"})
        return uid

    def test_verify_by_admission_proxy_is_honest(self):
        self._active("P")
        r = client.post(
            "/face/verify-by-admission",
            data={"admission_number": "FADMP", "code": "FSP"},
            files={"file": ("a.png", _png(), "image/png")},
        )
        assert r.status_code == 200
        j = r.json()
        assert j.get("proxy_mode") is True
        assert j.get("similarity") is None
        assert "not face-verified" in j["message"].lower()

    def test_verify_and_mark_proxy_is_honest(self):
        uid = self._active("Q")
        r = client.post(
            "/face/verify-and-mark",
            data={"code": "FSQ", "user_id": uid},
            files={"file": ("a.png", _png(), "image/png")},
        )
        assert r.status_code == 200
        j = r.json()
        assert j.get("proxy_mode") is True
        assert j.get("similarity") is None
        assert "not face-verified" in j["message"].lower()

    def test_ui_branches_on_proxy_mode(self):
        html = io.open("app/templates/student.html", encoding="utf-8").read()
        assert "result.proxy_mode" in html
        assert "not face-verified" in html

    def test_demo_login_and_admin_camera_are_honest(self):
        login = client.get("/login").text
        admin = client.get("/admin").text
        assert "any email and password work" in login
        assert "not production authentication" in login
        assert "super-secret-key" not in login + admin
        assert "Camera preview active. Demo mode does not verify identity" in admin
        assert "Camera preview only. Demo attendance must be marked manually." in admin
        assert "Image captured, but identity matching is unavailable" in admin
        assert "not identity recognition" in admin
        assert "manual attendance" in admin.lower()
        assert "X-Admin-Key" not in admin


# --------------------------------------------------------------------------
# 4. Face image-upload size enforcement (app/routers/face.py)
# --------------------------------------------------------------------------
class TestFaceUploadLimit:
    """The five /face/* image endpoints must read with the 15 MB cap and return
    HTTP 413 on oversize, without loading the whole oversized file into memory."""

    def test_no_unbounded_reads_remain(self):
        src = io.open("app/routers/face.py", encoding="utf-8").read()
        assert "await file.read()" not in src
        assert src.count("await _read_face_image(file)") == 5

    def _oversized_enroll(self):
        # 16 MB of bytes -> exceeds the 15 MB cap while reading incrementally.
        big = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024))
        return client.post(
            "/face/enroll",
            data={"user_id": "1"},
            files={"file": ("big.png", big, "image/png")},
        )

    def test_oversized_face_upload_returns_413(self):
        r = self._oversized_enroll()
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_valid_face_upload_still_works(self):
        # A normal (tiny) image is accepted; proxy mode -> honest response.
        r = client.post(
            "/face/enroll",
            data={"user_id": "1"},
            files={"file": ("a.png", _png(), "image/png")},
        )
        assert r.status_code == 200
        j = r.json()
        assert j.get("proxy_mode") is True  # model absent -> honest proxy, not "enrolled" by real match

    def test_empty_image_does_not_crash(self):
        r = client.post(
            "/face/enroll",
            data={"user_id": "1"},
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        # empty bytes -> no embedding -> proxy mode still returns 200 (no 500)
        assert r.status_code == 200
        assert r.json().get("proxy_mode") is True

    def test_invalid_image_bytes_do_not_crash(self):
        r = client.post(
            "/face/enroll",
            data={"user_id": "1"},
            files={"file": ("junk.png", io.BytesIO(b"not-an-image" * 100), "image/png")},
        )
        assert r.status_code == 200
        assert r.json().get("proxy_mode") is True

    def test_oversized_verify_by_admission_returns_413(self, monkeypatch):
        # verify-by-admission has a proxy fast-path (default) that returns 200
        # WITHOUT reading the file, so no memory risk there. The size cap applies
        # on the strict path, which DOES read the image. Force strict mode so the
        # oversized upload reaches _read_face_image() and must yield 413.
        import app.routers.face as face_mod
        monkeypatch.setattr(face_mod, "is_face_strict", lambda: True)
        client.post("/admin/classes", params={"name": "FB", "code": "FBC"})
        client.post("/admin/students/create", params={"email": "fb@t.com", "full_name": "FB", "admission_number": "FBADM"})
        client.post("/admin/sessions/start", params={"class_code": "FBC", "code": "FBS", "subject": "M"})
        big = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024))
        r = client.post(
            "/face/verify-by-admission",
            data={"admission_number": "FBADM", "code": "FBS"},
            files={"file": ("big.png", big, "image/png")},
        )
        assert r.status_code == 413

    def test_proxy_fastpath_does_not_read_oversized(self):
        # Default (non-strict) proxy path marks attendance by admission number
        # WITHOUT reading the image, so an oversized file cannot exhaust memory
        # there; it returns the honest proxy response (not a face verification).
        client.post("/admin/classes", params={"name": "FB2", "code": "FBC2"})
        client.post("/admin/students/create", params={"email": "fb2@t.com", "full_name": "FB2", "admission_number": "FBADM2"})
        client.post("/admin/sessions/start", params={"class_code": "FBC2", "code": "FBS2", "subject": "M"})
        big = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024))
        r = client.post(
            "/face/verify-by-admission",
            data={"admission_number": "FBADM2", "code": "FBS2"},
            files={"file": ("big.png", big, "image/png")},
        )
        assert r.status_code == 200
        j = r.json()
        assert j.get("proxy_mode") is True
        assert j.get("similarity") is None



