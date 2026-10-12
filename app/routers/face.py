from __future__ import annotations
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import os
import math
try:
    import numpy as np  # Optional on Vercel
except Exception:
    np = None  # type: ignore

from app.db import DATABASE_UNAVAILABLE_DETAIL, get_session, mark_database_unavailable
from app.models import User, AttendanceSession, Attendance
from app.services.face import compute_face_embedding, cosine_similarity, face_match_margin, face_match_threshold, get_face_backend
from app.services.uploads import read_limited
from app.notifications import manager, notify_user as send_notification

router = APIRouter(prefix="/face", tags=["face"]) 

AUTO_ATTENDANCE_FRAME_COUNT = 5
MIN_CONSISTENT_FACE_MATCHES = 3


async def _face_embeddings_from_frames(files: list[UploadFile]):
    if len(files) != AUTO_ATTENDANCE_FRAME_COUNT:
        raise HTTPException(status_code=400, detail="Capture five camera frames and try again")
    embeddings = []
    for frame in files:
        embeddings.append(compute_face_embedding(await _read_face_image(frame)))
    return [embedding for embedding in embeddings if embedding is not None]


def _rank_face_users(embedding, users: list[User]):
    ranked = []
    for user in users:
        try:
            user_embedding = (
                np.array(user.face_embedding, dtype=np.float32)
                if np is not None else list(map(float, user.face_embedding))
            )
            if len(embedding) != len(user_embedding) or any(
                not math.isfinite(float(value)) for value in user_embedding
            ):
                continue
            similarity = cosine_similarity(embedding, user_embedding)
            if math.isfinite(similarity):
                ranked.append((user.id, similarity))
        except Exception:
            continue
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _identify_consistent_user(embeddings, users: list[User]) -> tuple[int, float]:
    frame_matches = []
    for embedding in embeddings:
        ranked = _rank_face_users(embedding, users)
        if not ranked or ranked[0][1] < face_match_threshold(embedding):
            continue
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < face_match_margin(embedding):
            raise HTTPException(status_code=401, detail="Face match is inconclusive because enrolled profiles are too similar. Re-enroll this student's face in bright, front-facing light, or mark attendance manually")
        frame_matches.append(ranked[0])

    if len(frame_matches) < MIN_CONSISTENT_FACE_MATCHES:
        raise HTTPException(status_code=401, detail="Face could not be matched consistently; hold still and try again")
    if len({user_id for user_id, _score in frame_matches}) != 1:
        raise HTTPException(status_code=401, detail="Camera frames matched different students; ensure only one person is in view and try again")
    best_user_id = frame_matches[0][0]
    average_similarity = sum(score for _user_id, score in frame_matches) / len(frame_matches)
    return best_user_id, average_similarity

async def notify_user(user_id: int, message: str):
    await send_notification(user_id, message)


def is_face_strict() -> bool:
    """Require an actual enrolled face by default; opt into proxy mode explicitly."""
    return str(os.getenv("FACE_STRICT", "true")).strip().lower() not in {"0", "false", "no", "off"}


async def _read_face_image(file: UploadFile) -> bytes:
    """Read an uploaded face image with the same 15 MB cap used for PDFs/notes.
    Raises HTTP 413 while reading (incrementally) if the upload is too large;
    never loads the whole oversized file into memory."""
    try:
        return await read_limited(file)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(user_id, websocket)


@router.post("/enroll")
async def enroll(user_id: int = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = await _read_face_image(file)
        emb = compute_face_embedding(data)
        if emb is None and is_face_strict():
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if emb is not None:
            user.face_embedding = emb.tolist()
            session.add(user)
            session.commit()
            return {"message": "enrolled"}
        # Proxy mode: face model unavailable (cloud/free-tier) — accept enrollment
        return {"message": "Face enrolled successfully (proxy mode — face model not available on this deployment)", "proxy_mode": True}
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(status_code=409, detail="The attendance or enrollment record conflicts with existing data.") from None
    except SQLAlchemyError as e:
        mark_database_unavailable(e, "face route database operation")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Face processing failed") from None


@router.post("/verify-and-mark")
async def verify_and_mark(code: str = Form(...), user_id: int = Form(...), files: list[UploadFile] = File(..., alias="file"), session: Session = Depends(get_session)):
    try:
        # Fast path when strict mode is off
        if not is_face_strict():
            sess = session.exec(
                select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
            ).first()
            if not sess:
                raise HTTPException(status_code=404, detail="Active session not found")
            existing = session.exec(
                select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user_id)
            ).first()
            if existing:
                return {"message": "Attendance already marked (demo mode — not face-verified)", "user_id": user_id, "proxy_mode": True}
            att = Attendance(session_id=sess.id, user_id=user_id, status="present")
            session.add(att)
            session.commit()
            await notify_user(user_id, f"Attendance marked (demo mode) — session {sess.code}")
            return {"message": "Attendance marked (demo mode — not face-verified)", "user_id": user_id, "proxy_mode": True}

        embeddings = await _face_embeddings_from_frames(files)
        if len(embeddings) < MIN_CONSISTENT_FACE_MATCHES:
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.get(User, user_id)
        if not user or not user.face_embedding:
            raise HTTPException(status_code=400, detail="User not enrolled")
        user_emb = (
            np.array(user.face_embedding, dtype=np.float32) if np is not None else list(map(float, user.face_embedding))
        )
        accepted = []
        for emb in embeddings:
            similarity = cosine_similarity(emb, user_emb)
            if similarity >= face_match_threshold(emb):
                accepted.append(similarity)
        if len(accepted) < MIN_CONSISTENT_FACE_MATCHES:
            raise HTTPException(status_code=401, detail="Face mismatch")
        sim = sum(accepted) / len(accepted)
        sess = session.exec(
            select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
        ).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Active session not found")
        existing = session.exec(
            select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user_id)
        ).first()
        if existing:
            return {"message": "Attendance marked, face verified"}
        att = Attendance(session_id=sess.id, user_id=user_id, status="present")
        session.add(att)
        session.commit()
        await notify_user(user_id, f"✅ Attendance marked, face verified — session {sess.code}")
        return {"message": "Attendance marked, face verified", "similarity": sim}
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Attendance was already recorded or conflicts with existing data.") from None
    except SQLAlchemyError as e:
        mark_database_unavailable(e, "face route database operation")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Face processing failed") from None


@router.post("/auto-identify-and-mark")
async def auto_identify_and_mark(code: str = Form(...), files: list[UploadFile] = File(..., alias="file"), session: Session = Depends(get_session), x_admin_key: str | None = Header(default=None)):
    try:
        admin_key = os.getenv("ADMIN_KEY")
        if admin_key and (x_admin_key is None or x_admin_key != admin_key):
            raise HTTPException(status_code=403, detail="Forbidden")
        
        embeddings = await _face_embeddings_from_frames(files)
        if len(embeddings) < MIN_CONSISTENT_FACE_MATCHES:
            if get_face_backend() is None:
                return {
                    "message": "Face recognition backend is unavailable.",
                    "error_code": "face_backend_unavailable",
                }
            return {
                "message": "No face detected. Face the camera, move back until your full face is visible, and improve lighting or focus.",
                "error_code": "no_face_detected",
            }
        sess = session.exec(
            select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
        ).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Active session not found")
        
        users = session.exec(select(User).where(User.face_embedding.is_not(None))).all()
        if not users:
            raise HTTPException(status_code=401, detail="No matching enrolled user")
        best_user_id, best_sim = _identify_consistent_user(embeddings, users)
        
        existing = session.exec(
            select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == best_user_id)
        ).first()
        if existing:
            return {"message": "Already marked", "user_id": best_user_id, "similarity": best_sim}
        
        att = Attendance(session_id=sess.id, user_id=best_user_id, status="present")
        session.add(att)
        session.commit()
        await notify_user(best_user_id, f"✅ Attendance marked for session {sess.code}")
        student = session.get(User, best_user_id)
        return {
            "message": "Marked present",
            "user_id": best_user_id,
            "student_name": student.full_name if student else None,
            "similarity": best_sim,
        }
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(status_code=409, detail="The attendance or enrollment record conflicts with existing data.") from None
    except SQLAlchemyError as e:
        mark_database_unavailable(e, "face route database operation")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Face processing failed") from None


@router.post("/enroll-by-admission")
async def enroll_by_admission(admission_number: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = await _read_face_image(file)
        emb = compute_face_embedding(data)
        if emb is None and is_face_strict():
            raise HTTPException(status_code=400, detail="No face detected")
        user = session.exec(select(User).where(User.admission_number == admission_number)).first()
        if not user:
            raise HTTPException(status_code=404, detail="Student with this admission number not found")
        if emb is not None:
            user.face_embedding = emb.tolist()
            session.add(user)
            session.commit()
            return {"message": "Face enrolled successfully", "admission_number": admission_number, "student_name": user.full_name}
        # Proxy mode: face model unavailable (cloud/free-tier) — accept enrollment
        return {
            "message": "Face enrolled successfully (proxy mode — face model not available on this deployment)",
            "admission_number": admission_number,
            "student_name": user.full_name,
            "proxy_mode": True,
        }
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(status_code=409, detail="The attendance or enrollment record conflicts with existing data.") from None
    except SQLAlchemyError as e:
        mark_database_unavailable(e, "face route database operation")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Face processing failed") from None


@router.post("/verify-by-admission")
async def verify_by_admission(admission_number: str = Form(...), code: str = Form(...), files: list[UploadFile] = File(..., alias="file"), session: Session = Depends(get_session)):
    try:
        # Fast path when strict mode is off
        if not is_face_strict():
            user = session.exec(select(User).where(User.admission_number == admission_number)).first()
            if not user:
                raise HTTPException(status_code=404, detail="Student with this admission number not found")
            sess = session.exec(
                select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
            ).first()
            if not sess:
                raise HTTPException(status_code=404, detail="Active session not found")
            existing = session.exec(
                select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user.id)
            ).first()
            if existing:
                return {"message": "Attendance already marked (demo mode — not face-verified)", "user_id": user.id, "proxy_mode": True, "student_name": user.full_name}
            att = Attendance(session_id=sess.id, user_id=user.id, status="present")
            session.add(att)
            session.commit()
            await notify_user(user.id, f"Attendance marked (demo mode) — session {sess.code}")
            return {"message": "Attendance marked (demo mode — not face-verified)", "user_id": user.id, "proxy_mode": True, "student_name": user.full_name}

        embeddings = await _face_embeddings_from_frames(files)
        if len(embeddings) < MIN_CONSISTENT_FACE_MATCHES:
            # Graceful response to avoid frontend error toasts on cloud/free-tier
            return {"message": "No face detected", "similarity": 0.0}
        user = session.exec(select(User).where(User.admission_number == admission_number)).first()
        if not user:
            raise HTTPException(status_code=404, detail="Student with this admission number not found")
        if user.face_embedding is None:
            raise HTTPException(status_code=400, detail="Student face not enrolled yet")
        user_emb = (
            np.array(user.face_embedding, dtype=np.float32) if np is not None else list(map(float, user.face_embedding))
        )
        accepted = []
        for emb in embeddings:
            similarity = cosine_similarity(emb, user_emb)
            if similarity >= face_match_threshold(emb):
                accepted.append(similarity)
        if len(accepted) < MIN_CONSISTENT_FACE_MATCHES:
            raise HTTPException(status_code=401, detail="Face verification failed - similarity too low")
        sim = sum(accepted) / len(accepted)
        sess = session.exec(
            select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)
        ).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Active session not found")
        existing = session.exec(
            select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user.id)
        ).first()
        if existing:
            return {"message": "Attendance marked, face verified", "user_id": user.id, "similarity": sim, "student_name": user.full_name}
        att = Attendance(session_id=sess.id, user_id=user.id, status="present")
        session.add(att)
        session.commit()
        await notify_user(user.id, f"✅ Attendance marked, face verified — session {sess.code}")
        return {"message": "Attendance marked, face verified", "user_id": user.id, "similarity": sim, "student_name": user.full_name}
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Attendance was already recorded or conflicts with existing data.") from None
    except SQLAlchemyError as e:
        mark_database_unavailable(e, "face route database operation")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Face processing failed") from None


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
            "marked_at": att.timestamp,
            "status": att.status
        })
    return result
