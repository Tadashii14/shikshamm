# AttendAI

FastAPI app for attendance management with face recognition and smart curriculum features (notes → flashcards/quiz). Includes simple Admin and Student pages.

## Features
- Face-based attendance:
  - Admin camera matches enrolled students and marks attendance during an active session
  - On localhost, an OpenCV camera feed is available when the browser cannot see a webcam
  - Students enroll face once and verify during sessions
  - WebSocket notification to student when marked
- Admin: create classes, start/stop sessions, view active sessions and summaries
- Student: attendance stats, upload notes, generate flashcards and quizzes

## Requirements
- Python 3.10+
- SQLite (bundled)
- Packages in `requirements.txt`
- Env vars:
  - `OPENAI_API_KEY` is optional; quiz and flashcard generation has a local fallback
  - `JWT_SECRET` (optional)
  - `ADMIN_KEY` (optional, protects admin auto-identify endpoint)

## Quickstart
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
$env:DATABASE_URL = "sqlite:///./attendai_demo.db"
pip install -r requirements.txt
pip install -r requirements-local.txt
./run_local.ps1
```
Open `http://127.0.0.1:8765/`. `run_local.ps1` uses the project `.venv`, the safe demo database by default, and strict face matching. Keep the server bound to `127.0.0.1` so the local camera feed is not exposed to the network.

## Usage
- Admin page:
  1. Create students (email, full name)
  2. Create class
  3. Start session (subject optional)
  4. Enroll each student's face once while centered and well lit
  5. Start a session and the camera; matching enrolled students are marked automatically
- Student page:
  - Enter admission number and session code; camera starts when the session is active
  - Enroll face once from the admin page, then verify during the active session
  - On success, a popup appears via WebSocket
  - Upload notes → Generate flashcards or quiz

## Notes
- InsightFace is used when installed. The local setup also has a CPU-only OpenCV LBP matcher so it works without downloading a large neural model. LBP matching is intended for a classroom demo, is less robust than InsightFace, and does not detect photo/video spoofing.
- Camera preview frames stay in memory and are not written to disk. Local camera endpoints reject non-loopback and cross-origin requests.
- `FACE_STRICT` defaults to `true`. Set `FACE_STRICT=false` only when you intentionally want the explicitly labeled demo attendance bypass.
- If serving over HTTPS, WebSocket auto-selects `wss`.
- A minimal duplicate FastAPI scaffold exists under `attend-ai/app`; it's unused.

## Security
- Auth endpoints exist but are not enforced on admin/student actions. Add dependencies to protect endpoints if needed.

## License
MIT
