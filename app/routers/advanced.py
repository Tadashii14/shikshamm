from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from sqlmodel import Session, select
from typing import List, Dict, Optional
import datetime
import json

from app.db import get_session
from app.services.pdf_processor import (
    extract_text_from_pdf, clean_text, split_into_paragraphs, split_into_sentences,
    summarize_main_points, generate_paragraph_flashcards, generate_global_fallback_flashcards,
    build_qa_index, answer_question, generate_moderate_questions, generate_flashcards_from_text
)
# Avoid importing heavy timetable integrations (pandas) at startup on serverless

router = APIRouter()


@router.post("/pdf/extract-text")
async def extract_pdf_text(file: UploadFile = File(...)):
    """Extract and clean text from PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        content = await file.read()
        import io
        text = extract_text_from_pdf(io.BytesIO(content))
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        cleaned_text = clean_text(text)
        paragraphs = split_into_paragraphs(cleaned_text)
        sentences = split_into_sentences(cleaned_text)
        
        return {
            "success": True,
            "text_length": len(cleaned_text),
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "preview": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/pdf/summarize")
async def summarize_pdf(
    file: UploadFile = File(...),
    sentence_count: int = Form(10)
):
    """Generate summary of PDF content"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        content = await file.read()
        import io
        text = extract_text_from_pdf(io.BytesIO(content))
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        cleaned_text = clean_text(text)
        sentences = split_into_sentences(cleaned_text)
        summary = summarize_main_points(sentences, sentence_count)
        
        return {
            "success": True,
            "summary": summary,
            "sentence_count": len(summary)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing PDF: {str(e)}")


@router.post("/pdf/generate-flashcards")
async def generate_flashcards(
    file: UploadFile = File(...),
    cards_per_paragraph: int = Form(12),
    fallback_cards_total: int = Form(16),
    explain_min: int = Form(25),
    explain_max: int = Form(55)
):
    """Generate flashcards from PDF content"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        content = await file.read()
        import io
        text = extract_text_from_pdf(io.BytesIO(content))
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        cleaned_text = clean_text(text)
        paragraphs = split_into_paragraphs(cleaned_text)
        sentences = split_into_sentences(cleaned_text)
        
        # Generate summary for banned sentences
        summary_sents = summarize_main_points(sentences, 12)
        
        # Generate flashcards using the proper repository algorithm
        cards = generate_flashcards_from_text(
            text, 
            cards_per_paragraph=cards_per_paragraph,
            fallback_cards_total=fallback_cards_total,
            explain_min=explain_min,
            explain_max=explain_max
        )
        
        return {
            "success": True,
            "cards": cards,
            "total_cards": len(cards)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")


@router.post("/pdf/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    num_questions: int = Form(10)
):
    """Generate quiz questions from PDF content"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        content = await file.read()
        import io
        text = extract_text_from_pdf(io.BytesIO(content))
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        cleaned_text = clean_text(text)
        questions = generate_moderate_questions(cleaned_text, num_questions)
        
        return {
            "success": True,
            "questions": questions,
            "total_questions": len(questions)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")


@router.post("/pdf/ask-question")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """Ask a question about PDF content"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        content = await file.read()
        import io
        text = extract_text_from_pdf(io.BytesIO(content))
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        cleaned_text = clean_text(text)
        paragraphs = split_into_paragraphs(cleaned_text)
        
        # Build QA index
        vec, mat = build_qa_index(paragraphs)
        if vec is None or mat is None:
            raise HTTPException(status_code=400, detail="Could not build question answering index")
        
        # Answer question
        result = answer_question(question, paragraphs, vec, mat, top_k=3)
        
        return {
            "success": True,
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@router.post("/timetable/generate")
async def generate_timetable(request: Request):
    """Generate Pomodoro-based study timetable."""
    # Accepts BOTH JSON body (student.html) and form-urlencoded (timetable.html).
    # Falls back to a pure-Python generator (no pandas/plotly) so it works on Vercel.
    def _as_list(v):
        import json
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                arr = json.loads(s)
                return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
        return [x.strip() for x in s.split(",") if x.strip()]

    try:
        ct = request.headers.get("content-type", "").lower()
        if "application/json" in ct:
            body = await request.json()
        else:
            form = await request.form()
            body = dict(form)

        subjects = _as_list(body.get("subjects"))
        if not subjects:
            raise HTTPException(status_code=400, detail="Subjects are required")

        difficulty = _as_list(body.get("difficulty_levels", body.get("difficulty")))
        while len(difficulty) < len(subjects):
            difficulty.append("medium")
        difficulty = difficulty[:len(subjects)]

        try:
            study_hours = int(body.get("study_hours_per_day", 4)) or 4
        except (TypeError, ValueError):
            study_hours = 4
        if study_hours < 1:
            study_hours = 4

        busy_hours = body.get("busy_hours")
        if isinstance(busy_hours, str):
            busy_str = busy_hours.strip()
            if busy_str:
                try:
                    import json
                    busy_hours = json.loads(busy_str)
                except Exception:
                    busy_hours = None
            else:
                busy_hours = None

        max_blocks = None
        mb = body.get("max_blocks_per_day")
        if mb is not None:
            try:
                max_blocks = int(mb)
            except (TypeError, ValueError):
                max_blocks = None

        result = None
        try:
            from app.services.timetable_repository import generate_timetable_from_repository
            result = generate_timetable_from_repository(subjects, difficulty, study_hours)
        except Exception:
            result = None

        if result is None or not result.get("schedule"):
            from app.services.timetable_pure import generate_timetable_pure
            result = generate_timetable_pure(
                subjects=subjects,
                difficulty_levels=difficulty,
                study_hours_per_day=study_hours,
                busy_hours=busy_hours,
                max_blocks_per_day=max_blocks,
            )

        result["success"] = True
        if "schedule" in result and "timetable" not in result:
            result["timetable"] = result["schedule"]
        metrics = result.get("metrics", {})
        if "daily_average" not in metrics:
            metrics["daily_average"] = metrics.get("daily_average_hours_with_breaks", 0)
        result["metrics"] = metrics
        return result
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating timetable: {str(e)}")
@router.get("/timetable/google-calendar/auth-url")
async def get_google_calendar_auth_url():
    """Get Google Calendar authentication URL"""
    try:
        from app.services.timetable import GoogleCalendarIntegrator
        integrator = GoogleCalendarIntegrator()
        auth_url = integrator.get_auth_url()
        
        if not auth_url:
            raise HTTPException(status_code=400, detail="Google Calendar not configured")
        
        return {
            "success": True,
            "auth_url": auth_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting auth URL: {str(e)}")


@router.post("/timetable/google-calendar/create-events")
async def create_calendar_events(
    timetable: str = Form(...),  # JSON string of timetable
    subjects: str = Form(...),  # JSON string of subjects
    timezone: str = Form("America/New_York"),
    start_date: str = Form(...)  # ISO date string
):
    """Create Google Calendar events for timetable"""
    try:
        # Parse JSON inputs
        timetable_dict = json.loads(timetable)
        subjects_dict = json.loads(subjects)
        
        # Parse start date
        start_date_obj = datetime.datetime.fromisoformat(start_date).date()
        
        # Create calendar events
        from app.services.timetable import GoogleCalendarIntegrator
        integrator = GoogleCalendarIntegrator()
        # Note: This would need proper authentication handling in production
        
        return {
            "success": True,
            "message": "Calendar events would be created here (requires proper authentication setup)"
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating calendar events: {str(e)}")

