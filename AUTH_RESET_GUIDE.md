# 🔐 Login — Demo Mode (no Firebase, no DB check)

**As of this update, the website login requires NO real credentials.**

## How login works now
1. Land on `/` (landing page!) → pick **"I am a Student"** (`/login?target=student`() or **"I am a Teacher/Admin"** (`/login?target=teacher`(.
2. Fill in **any** email/username and password → approved automatically.
3. You're redirected to:
   - Student → `/student` (Student Portal)
   - Teacher/Admin → `/admin` (Admin/Teacher Portal)

> The role is tracked **client-side in localStorage** — no database,, no Firebase.
> Direct visits to `/admin` without the teacher role in localStorage redirect back to `/login?target=teacher`.

## What about the backend DB?
The FastAPI backend (SQLite by default, or Postgres via `DATABASE_URL`) still stores
users / attendance / faces / notes — it is just **no longer used for the web login**.
Manage those DB users with:

```bash
python scripts/reset_local_credentials.py --list
python scripts/reset_local_credentials.py --create admin@example.com 'Admin@12345' admin 'System Admin'
python scripts/reset_local_credentials.py --set-password user@example.com 'NewPass@123'
python scripts/reset_local_credentials.py --reset-all --new-password 'Bulk@123'
```

## Re-enabling real auth later
If you want real password auth again: restore Firebase (login.html, admin.html, flashcards.html,, timetable.html) or wire the frontend
to the backend JWT `/auth/login` endpoints. All Firebase references were cleanly removed in this commit;
git history has them.

## Troubleshooting: passlib / bcrypt
`passlib 1.7.4` is incompatible with `bcrypt >= 4.1` — requirements now pin `bcrypt==4.0.1`.
Always run `pip install -r requirements.txt` after pulling.