from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Attendance, AttendanceSession, ClassRoom, User


router = APIRouter(prefix="/stats", tags=["stats"]) 


@router.get("/student/attendance")
def student_attendance(user_id: int, days: int = 60, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    since = datetime.utcnow() - timedelta(days=days)
    attns: List[Attendance] = session.exec(
        select(Attendance)
        .where(Attendance.user_id == user_id)
        .where(Attendance.timestamp >= since)
        .order_by(Attendance.timestamp.desc())
    ).all()
    total = len(attns)
    present = sum(1 for a in attns if a.status == "present")
    pct = (present / total * 100.0) if total else 0.0
    return {
        "user_id": user_id,
        "window_days": days,
        "total_marked": total,
        "present": present,
        "attendance_pct": round(pct, 2),
        "recent": [
            {"session_id": a.session_id, "status": a.status, "timestamp": a.timestamp.isoformat()} for a in attns[:20]
        ],
    }


@router.get("/admin/class-attendance")
def class_attendance(class_code: str, session: Session = Depends(get_session)):
    cls = session.exec(select(ClassRoom).where(ClassRoom.code == class_code)).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    sessions = session.exec(select(AttendanceSession).where(AttendanceSession.class_id == cls.id).order_by(AttendanceSession.started_at.desc())).all()
    result = []
    for s in sessions[:20]:
        count = session.exec(select(Attendance).where(Attendance.session_id == s.id)).all()
        result.append({
            "session_code": s.code,
            "subject": s.subject,
            "is_active": s.is_active,
            "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            "present_count": len(count)
        })
    return {"class_code": class_code, "sessions": result}


