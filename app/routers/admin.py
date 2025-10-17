from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import ClassRoom, AttendanceSession, User, Attendance


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/classes")
def create_class(name: str, code: str, created_by_user_id: int, session: Session = Depends(get_session)):
    existing = session.exec(select(ClassRoom).where(ClassRoom.code == code)).first()
    if existing:
        return existing  # Return existing class instead of error
    cls = ClassRoom(name=name, code=code, created_by_user_id=created_by_user_id)
    session.add(cls)
    session.commit()
    session.refresh(cls)
    return cls


@router.post("/sessions/start")
def start_session(class_code: str, code: str, subject: str | None = None, session: Session = Depends(get_session)):
    cls = session.exec(select(ClassRoom).where(ClassRoom.code == class_code)).first()
    if not cls:
        # Auto-provision a default admin and class if missing so the UI works out-of-the-box
        admin_user = session.exec(select(User).where(User.role == "admin")).first()
        if not admin_user:
            admin_user = User(email="admin@local", full_name="Default Admin", role="admin", hashed_password="")
            session.add(admin_user)
            session.commit()
            session.refresh(admin_user)
        cls = ClassRoom(name="Auto Class", code=class_code, created_by_user_id=admin_user.id)
        session.add(cls)
        session.commit()
        session.refresh(cls)
    
    # Stop any existing active session for this class
    active_sessions = session.exec(select(AttendanceSession).where(AttendanceSession.class_id == cls.id, AttendanceSession.is_active == True)).all()  # noqa: E712
    for active_sess in active_sessions:
        active_sess.is_active = False
        active_sess.ended_at = datetime.utcnow()
        session.add(active_sess)
    
    # Start new session
    sess = AttendanceSession(class_id=cls.id, subject=subject, code=code, is_active=True, started_at=datetime.utcnow())
    session.add(sess)
    session.commit()
    session.refresh(sess)
    return sess


@router.post("/sessions/stop")
def stop_session(code: str, session: Session = Depends(get_session)):
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)).first()  # noqa: E712
    if not sess:
        raise HTTPException(status_code=404, detail="Active session not found")
    sess.is_active = False
    sess.ended_at = datetime.utcnow()
    session.add(sess)
    session.commit()
    session.refresh(sess)
    return sess
@router.post("/students/create")
def create_student(email: str, full_name: str, admission_number: str | None = None, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.email == email)).first()
    if existing:
        return existing
    
    # Check if admission number already exists
    if admission_number:
        existing_admission = session.exec(select(User).where(User.admission_number == admission_number)).first()
        if existing_admission:
            raise HTTPException(status_code=400, detail="Student with this admission number already exists")
    
    user = User(email=email, full_name=full_name, admission_number=admission_number, role="student", hashed_password="")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.get("/students/list")
def list_students(session: Session = Depends(get_session)):
    users = session.exec(select(User).where(User.role == "student")).all()
    return [{"id": u.id, "email": u.email, "full_name": u.full_name, "admission_number": u.admission_number} for u in users]


@router.post("/attendance/manual")
def mark_manual_attendance(user_id: int, session_code: str, status: str, session: Session = Depends(get_session)):
    # Find the session
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.code == session_code, AttendanceSession.is_active == True)).first()  # noqa: E712
    if not sess:
        raise HTTPException(status_code=404, detail="Active session not found")
    
    # Check if already marked
    existing = session.exec(select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user_id)).first()
    if existing:
        return {"message": "Already marked", "user_id": user_id}
    
    # Mark attendance
    att = Attendance(session_id=sess.id, user_id=user_id, status=status)
    session.add(att)
    session.commit()
    
    return {"message": "Marked present", "user_id": user_id}


@router.get("/attendance/stats/{user_id}")
def get_attendance_stats(user_id: int, session: Session = Depends(get_session)):
    """Get real-time attendance statistics for a user"""
    # Get all sessions the user has attended
    attended_sessions = session.exec(
        select(AttendanceSession)
        .join(Attendance)
        .where(Attendance.user_id == user_id, Attendance.status == "present")
    ).all()
    
    # Get all sessions (for total count)
    all_sessions = session.exec(select(AttendanceSession)).all()
    
    # Calculate statistics
    total_sessions = len(all_sessions)
    attended_count = len(attended_sessions)
    attendance_percentage = round((attended_count / total_sessions * 100), 1) if total_sessions > 0 else 0
    
    # Get last attendance
    last_attendance = session.exec(
        select(Attendance)
        .join(AttendanceSession)
        .where(Attendance.user_id == user_id, Attendance.status == "present")
        .order_by(Attendance.created_at.desc())
        .limit(1)
    ).first()
    
    last_attendance_date = "Never"
    if last_attendance:
        last_attendance_date = last_attendance.created_at.strftime("%Y-%m-%d") if last_attendance.created_at else "Unknown"
    
    return {
        "total_sessions": total_sessions,
        "attended_sessions": attended_count,
        "attendance_percentage": f"{attendance_percentage}%",
        "last_attendance": last_attendance_date
    }


@router.get("/sessions/active")
def get_active_session(class_code: str | None = None, session: Session = Depends(get_session)):
    """Get active session(s) - returns single session for student auto-detection or list for admin"""
    if class_code:
        # Admin query - return list of active sessions for specific class
        cls = session.exec(select(ClassRoom).where(ClassRoom.code == class_code)).first()
        if not cls:
            return []
        query = select(AttendanceSession).where(AttendanceSession.is_active == True, AttendanceSession.class_id == cls.id)  # noqa: E712
        return session.exec(query).all()
    else:
        # Student query - return first active session
        active_session = session.exec(select(AttendanceSession).where(AttendanceSession.is_active == True).limit(1)).first()  # noqa: E712
        if active_session:
            return {"is_active": True, "code": active_session.code, "subject": active_session.subject}
        return {"is_active": False}


