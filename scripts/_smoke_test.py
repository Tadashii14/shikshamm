"""Smoke test: exercise every route of the app locally (proxy mode defaults)."""
import io
import sys

sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from app.main import app
from app.db import init_db

init_db()  # ensure tables exist (lifespan not run outside context manager)
client = TestClient(app)
results = []


def check(name, resp, expect=200):
    ok = resp.status_code == expect
    results.append((ok, name, resp.status_code, expect))
    preview = ""
    try:
        j = resp.json()
        s = str(j)
        preview = s[:90]
    except Exception:
        preview = resp.text[:90]
    print(f"{'PASS' if ok else 'FAIL'} {name} -> {resp.status_code} (want {expect}) {preview}")


# ---- Pages ----
for path in ["/", "/login", "/admin", "/student", "/quiz", "/flashcards", "/timetable"]:
    check(f"GET {path}", client.get(path))

# ---- Admin: class, students, session ----
check("POST /admin/classes", client.post("/admin/classes", params={"name": "Smoke Class", "code": "SMOKE", "created_by_user_id": 1}))
check("POST /admin/students/create", client.post("/admin/students/create", params={"email": "smoke1@test.com", "full_name": "Smoke One", "admission_number": "SMK001"}))
check("POST /admin/students/create", client.post("/admin/students/create", params={"email": "smoke2@test.com", "full_name": "Smoke Two", "admission_number": "SMK002"}))
check("POST /admin/sessions/start", client.post("/admin/sessions/start", params={"class_code": "SMOKE", "code": "SMKC1", "subject": "Math"}))
check("GET /admin/sessions/active", client.get("/admin/sessions/active"))

# ---- Face: enroll + verify (proxy mode, no model) ----
png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 64)  # fake image bytes
check("POST /face/enroll-by-admission (proxy)", client.post("/face/enroll-by-admission", data={"admission_number": "SMK001"}, files={"file": ("a.png", png, "image/png")}))
png2 = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"1" * 64)
check("POST /face/verify-by-admission (proxy mark)", client.post("/face/verify-by-admission", data={"admission_number": "SMK001", "code": "SMKC1"}, files={"file": ("a.png", png2, "image/png")}))
check("POST /face/auto-identify-and-mark", client.post("/face/auto-identify-and-mark", data={"code": "SMKC1"}, files={"file": ("a.png", png2, "image/png")}))
check("POST /face/verify-and-mark", client.post("/face/verify-and-mark", data={"code": "SMKC1", "user_id": 2}, files={"file": ("a.png", png2, "image/png")}))

# ---- Admin: manual attendance (JSON body, like admin.html sends) ----
check("POST /admin/attendance/manual (JSON)", client.post("/admin/attendance/manual", json={"session_code": "SMKC1", "user_id": 2, "status": "present"}))
check("POST /admin/attendance/manual (dup)", client.post("/admin/attendance/manual", json={"session_code": "SMKC1", "user_id": 2, "status": "present"}))
check("GET /admin/students/list", client.get("/admin/students/list"))
check("GET /admin/attendance/stats/1", client.get("/admin/attendance/stats/1"))
check("POST /admin/sessions/stop", client.post("/admin/sessions/stop", params={"code": "SMKC1"}))

# ---- Advanced: quiz + flashcards (multipart PDF, like the pages send) ----
# pypdf-generated minimal PDF (blank page -> endpoint should answer 400 'meaningful text')
try:
    from pypdf import PdfWriter
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    w.write(buf)
    pdf_bytes = buf.getvalue()
    HAVE_PDF = True
except Exception:
    pdf_bytes = b"%PDF-1.4 fake"
    HAVE_PDF = False

r = client.post("/api/advanced/pdf/generate-quiz", files={"file": ("t.pdf", pdf_bytes, "application/pdf")}, data={"num_questions": "2", "difficulty": "easy"})
# blank PDF cannot yield >=100 chars text -> expect 400 from the endpoint (contract works)
check("POST /api/advanced/pdf/generate-quiz (contract)", r, expect=400)

r = client.post("/api/advanced/pdf/generate-flashcards", files={"file": ("t.pdf", pdf_bytes, "application/pdf")}, data={"cards_per_paragraph": "12", "fallback_cards_total": "16", "explain_min": "25", "explain_max": "55"})
check("POST /api/advanced/pdf/generate-flashcards (contract)", r, expect=400)

# Pure algorithms behind those endpoints (text path used in production)
from app.services.pdf_processor import generate_moderate_questions, generate_flashcards_from_text
qs = generate_moderate_questions("Photosynthesis is the process by which plants convert light energy into chemical energy. " * 8, 3)
print(("PASS" if isinstance(qs, list) and qs else "FAIL"), "generate_moderate_questions ->", len(qs) if isinstance(qs, list) else type(qs), str(qs)[:80])
results.append((bool(qs), "service generate_moderate_questions", 0, 0))
cards = generate_flashcards_from_text("Gravity is a force that attracts two bodies toward each other. " * 8, fallback_cards_total=2)
print(("PASS" if isinstance(cards, list) and cards else "FAIL"), "generate_flashcards_from_text ->", len(cards) if isinstance(cards, list) else type(cards), str(cards)[:80])
results.append((bool(cards), "service generate_flashcards_from_text", 0, 0))

# ---- Advanced: timetable (JSON body, like student.html/timetable.html send) ----
check("POST /api/advanced/timetable/generate", client.post("/api/advanced/timetable/generate", json={"subjects": ["Math", "Science"], "difficulty_levels": ["easy", "hard"], "study_hours_per_day": 4}))
check("POST /api/advanced/timetable/generate (form)", client.post("/api/advanced/timetable/generate", data={"subjects": "Math,Science", "study_hours_per_day": "3"}))

# ---- Health/monitoring ----
check("GET /health", client.get("/health"))
check("GET /metrics", client.get("/metrics"))

# ---- Notes (upload txt + flashcard gen) ----
check("POST /notes/upload (txt)", client.post("/notes/upload", data={"user_id": "1"}, files={"file": ("n.txt", b"Chapter one. The cell is the basic unit of life. " * 10, "text/plain")}))
check("POST /notes/quiz/generate", client.post("/notes/quiz/generate", data={"note_id": "1", "num_questions": "3"}))

fails = [r for r in results if not r[0]]
print(f"\n===== {len(results) - len(fails)}/{len(results)} passed =====")
if fails:
    for f in fails:
        print("FAIL:", f[1], "->", f[2], "want", f[3])
    sys.exit(1)
