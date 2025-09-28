from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column, JSON


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: str
    role: str = Field(description="admin or student")
    hashed_password: str
    admission_number: Optional[str] = Field(default=None, index=True, unique=True)
    face_embedding: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))


class ClassRoom(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    code: str = Field(index=True, unique=True)
    created_by_user_id: int = Field(foreign_key="user.id")


class AttendanceSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    class_id: int = Field(foreign_key="classroom.id")
    subject: Optional[str] = None
    code: str = Field(index=True)
    is_active: bool = True
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None


class Attendance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="attendancesession.id")
    user_id: int = Field(foreign_key="user.id")
    status: str = Field(default="present")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Note(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    class_id: Optional[int] = Field(default=None, foreign_key="classroom.id")
    filename: str
    text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Flashcard(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    note_id: int = Field(foreign_key="note.id")
    front: str
    back: str
    score: int = 0


