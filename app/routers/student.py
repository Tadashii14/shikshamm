from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import AttendanceSession, Attendance, User


router = APIRouter(prefix="/student", tags=["student"])


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


