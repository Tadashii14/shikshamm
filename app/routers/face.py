from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header, WebSocket
from sqlmodel import Session, select
import os
try:
    import numpy as np  # Optional on Vercel
except Exception:
    np = None  # type: ignore

from app.db import get_session
from app.models import User, AttendanceSession, Attendance
from app.services.face import compute_face_embedding, cosine_similarity

router = APIRouter(prefix="/face", tags=["face"])

# WebSocket clients storage
connected_clients = {}


async def notify_user(user_id: int, message: str):
    ws = connected_clients.get(str(user_id))
    if ws:
        try:
            await ws.send_json({"message": message})
        except Exception as e:
            print(f"WebSocket send failed for user {user_id}: {e}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    connected_clients[user_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except Exception:
        connected_clients.pop(user_id, None)


@router.post("/enroll")
async def enroll(user_id: int = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = await file.read()
        emb = compute_face_embedding(data)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user.face_embedding = emb.tolist()
        session.add(user)
        session.commit()
        return {"message": "enrolled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-and-mark")
async def verify_and_mark(code: str = Form(...), user_id: int = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = await file.read()
        emb = compute_face_embedding(data)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.get(User, user_id)
        if not user or not user.face_embedding:
            raise HTTPException(status_code=400, detail="User not enrolled")
        user_emb = (
            np.array(user.face_embedding, dtype=np.float32) if np is not None else list(map(float, user.face_embedding))
        )
        sim = cosine_similarity(emb, user_emb)
        if sim < 0.35:
            raise HTTPException(status_code=401, detail="Face mismatch")
        sess = session.exec(
            select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
        ).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Active session not found")
        existing = session.exec(
            select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user_id)
        ).first()
        if existing:
            return {"message": "Already marked present"}
        att = Attendance(session_id=sess.id, user_id=user_id, status="present")
        session.add(att)
        session.commit()
        await notify_user(user_id, f"✅ Attendance marked for session {sess.code}")
        return {"message": "Marked present", "similarity": sim}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-identify-and-mark")
async def auto_identify_and_mark(code: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session), x_admin_key: str | None = Header(default=None)):
    try:
        admin_key = os.getenv("ADMIN_KEY")
        if admin_key and (x_admin_key is None or x_admin_key != admin_key):
            raise HTTPException(status_code=403, detail="Forbidden")
        
        data = await file.read()
        emb = compute_face_embedding(data)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")
        sess = session.exec(
            select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
        ).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Active session not found")
        
        users = session.exec(select(User).where(User.face_embedding.is_not(None))).all()
        best_user_id = None
        best_sim = -1.0
        for u in users:
            try:
                uemb = (
                    np.array(u.face_embedding, dtype=np.float32) if np is not None else list(map(float, u.face_embedding))
                )
                s = cosine_similarity(emb, uemb)
                if s > best_sim:
                    best_sim = s
                    best_user_id = u.id
            except Exception:
                continue
        if best_user_id is None or best_sim < 0.35:
            raise HTTPException(status_code=401, detail="No matching enrolled user")
        
        existing = session.exec(
            select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == best_user_id)
        ).first()
        if existing:
            return {"message": "Already marked", "user_id": best_user_id, "similarity": best_sim}
        
        att = Attendance(session_id=sess.id, user_id=best_user_id, status="present")
        session.add(att)
        session.commit()
        await notify_user(best_user_id, f"✅ Attendance marked for session {sess.code}")
        return {"message": "Marked present", "user_id": best_user_id, "similarity": best_sim}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll-by-admission")
async def enroll_by_admission(admission_number: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = await file.read()
        emb = compute_face_embedding(data)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.exec(select(User).where(User.admission_number == admission_number)).first()
        if not user:
            raise HTTPException(status_code=404, detail="Student with this admission number not found")
        user.face_embedding = emb.tolist()
        session.add(user)
        session.commit()
        return {"message": "Face enrolled successfully", "admission_number": admission_number, "student_name": user.full_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-by-admission")
async def verify_by_admission(admission_number: str = Form(...), code: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = await file.read()
        emb = compute_face_embedding(data)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.exec(select(User).where(User.admission_number == admission_number)).first()
        if not user:
            raise HTTPException(status_code=404, detail="Student with this admission number not found")
        if user.face_embedding is None:
            raise HTTPException(status_code=400, detail="Student face not enrolled yet")
        user_emb = (
            np.array(user.face_embedding, dtype=np.float32) if np is not None else list(map(float, user.face_embedding))
        )
        sim = cosine_similarity(emb, user_emb)
        if sim < 0.35:
            raise HTTPException(status_code=401, detail="Face verification failed - similarity too low")
        sess = session.exec(
            select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
        ).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Active session not found")
        existing = session.exec(
            select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user.id)
        ).first()
        if existing:
            return {"message": "Already marked", "user_id": user.id, "similarity": sim, "student_name": user.full_name}
        att = Attendance(session_id=sess.id, user_id=user.id, status="present")
        session.add(att)
        session.commit()
        await notify_user(user.id, f"✅ Attendance marked for session {sess.code}")
        return {"message": "Marked present", "user_id": user.id, "similarity": sim, "student_name": user.full_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attendance")
async def get_attendance(user_id: int, session: Session = Depends(get_session)):
    attendances = session.exec(
        select(Attendance)
        .join(AttendanceSession)
        .where(Attendance.user_id == user_id)
        .order_by(AttendanceSession.started_at.desc())
        .limit(20)
    ).all()
    
    result = []
    for att in attendances:
        sess = session.get(AttendanceSession, att.session_id)
        result.append({
            "session_code": sess.code,
            "marked_at": att.created_at,
            "status": att.status
        })
    return result
