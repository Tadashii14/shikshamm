from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select
from typing import List

from app.db import get_session
from app.models import Note, Flashcard
from app.services.llm import generate_flashcards_from_text, generate_quiz_from_text


router = APIRouter(prefix="/notes", tags=["notes"]) 


@router.post("/upload")
async def upload_note(user_id: int = Form(...), class_id: int | None = Form(None), file: UploadFile = File(...), session: Session = Depends(get_session)):
    content = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(content))
            for page in reader.pages[:20]:
                text += page.extract_text() or ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")
    else:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    note = Note(user_id=user_id, class_id=class_id, filename=file.filename, text=text)
    session.add(note)
    session.commit()
    session.refresh(note)
    return {"note_id": note.id, "chars": len(text)}


@router.post("/flashcards/generate")
def generate_flashcards(note_id: int = Form(...), max_cards: int = Form(12), session: Session = Depends(get_session)):
    note = session.get(Note, note_id)
    if not note or not note.text:
        raise HTTPException(status_code=404, detail="Note not found or empty")
    try:
        cards = generate_flashcards_from_text(note.text, max_cards=max_cards)
    except Exception as e:
        # fallback: create one summary card to avoid UI hanging
        cards = [{"front": "Summary", "back": "Flashcard generation unavailable: " + str(e)}]
    created: List[Flashcard] = []
    for c in cards:
        fc = Flashcard(note_id=note.id, front=c["front"], back=c["back"], score=0)
        session.add(fc)
        created.append(fc)
    session.commit()
    return {"created": len(created)}


@router.get("/flashcards/list")
def list_flashcards(note_id: int, session: Session = Depends(get_session)):
    fcs = session.exec(select(Flashcard).where(Flashcard.note_id == note_id)).all()
    return [{"id": f.id, "front": f.front, "back": f.back, "score": f.score} for f in fcs]


@router.post("/quiz/generate")
def generate_quiz(note_id: int = Form(...), num_questions: int = Form(8), session: Session = Depends(get_session)):
    note = session.get(Note, note_id)
    if not note or not note.text:
        raise HTTPException(status_code=404, detail="Note not found or empty")
    try:
        quiz = generate_quiz_from_text(note.text, num_questions=num_questions)
    except Exception as e:
        quiz = [{"question": "Quiz generation unavailable", "options": ["OK"], "answer": "OK", "error": str(e)}]
    return {"questions": quiz}

