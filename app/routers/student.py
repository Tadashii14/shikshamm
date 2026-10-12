from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from sqlalchemy import func

from app.db import get_session
from app.models import AttendanceSession, Attendance, User


router = APIRouter(prefix="/student", tags=["student"])


@router.get("/identity")
def resolve_identity(
    email: str | None = None,
    admission_number: str | None = None,
    session: Session = Depends(get_session),
):
    """Resolve a typed account name to its registered student record.

    The local demo login stores the entered account name in the browser. Only an
    exact student-roster match is returned; an unknown name never falls back to
    another student's attendance data.
    """
    if not email and not admission_number:
        raise HTTPException(status_code=400, detail="email or admission_number is required")

    user = None
    if email:
        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            # Preserve free-form input, but allow case/outer-space differences.
            user = session.exec(select(User).where(func.lower(User.email) == email.strip().lower())).first()
    if not user and admission_number:
        user = session.exec(select(User).where(User.admission_number == admission_number)).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="No student record matches this login. Ask your teacher/admin to register your e-mail and admission number.",
        )
    if user.role != "student":
        raise HTTPException(status_code=403, detail="This login belongs to a teacher/admin account, not a student account")

    return {
        "user_id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "admission_number": user.admission_number,
        "role": user.role,
    }


@router.post("/join")
def join_session(code: str, user_id: int, session: Session = Depends(get_session)):
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)).first()  # noqa: E712
    if not sess:
        raise HTTPException(status_code=404, detail="Active session not found")
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # naive: mark present once; ignore duplicates
    existing = session.exec(select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user_id)).first()
    if existing:
        return {"message": "Already marked present"}
    att = Attendance(session_id=sess.id, user_id=user_id, status="present")
    session.add(att)
    session.commit()
    session.refresh(att)
    return {"message": "Marked present", "attendance_id": att.id}


