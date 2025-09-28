# AttendAI

FastAPI app for attendance management with face recognition and smart curriculum features (notes → flashcards/quiz). Includes simple Admin and Student pages.

## Features
- Face-based attendance:
  - Admin camera auto-identifies students and marks attendance
  - Students enroll face once and verify during sessions
  - WebSocket notification to student when marked
- Admin: create classes, start/stop sessions, view active sessions and summaries
- Student: attendance stats, upload notes, generate flashcards and quizzes

## Requirements
- Python 3.10+
- SQLite (bundled)
- Packages in `requirements.txt`
- Env vars:
  - `OPENAI_API_KEY` for flashcards/quiz
  - `JWT_SECRET` (optional)
  - `ADMIN_KEY` (optional, protects admin auto-identify endpoint)

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/`

## Usage
- Admin page:
  1. Create students (email, full name)
  2. Create class
  3. Start session (subject optional)
  4. Start Camera → frames sent every 2s to auto-identify and mark
- Student page:
  - Enter user ID and session code; camera auto-starts
  - Enroll face once, then Verify & Mark during active session
  - On success, a popup appears via WebSocket
  - Upload notes → Generate flashcards or quiz

## Notes
- Matching threshold currently 0.35 (conservative). Tune in `app/routers/face.py`.
- If serving over HTTPS, WebSocket auto-selects `wss`.
- A minimal duplicate FastAPI scaffold exists under `attend-ai/app`; it's unused.

## Security
- Auth endpoints exist but are not enforced on admin/student actions. Add dependencies to protect endpoints if needed.

## License
MIT
