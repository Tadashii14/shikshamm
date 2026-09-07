from __future__ import annotations

from sqlmodel import SQLModel, create_engine, Session
import os
import sqlite3
import tempfile

from app.models import User  # make sure User has admission_number in models.py

# Prefer DATABASE_URL (e.g., Postgres on Render). Fallback to local SQLite.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendai.db")

# Normalize scheme and ensure psycopg (v3) driver for Postgres
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if DATABASE_URL.startswith("postgresql://") and "+" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Serverless (Vercel) has a read-only filesystem except /tmp. If we are using
# SQLite and the working directory isn't writable, fall back to /tmp.
_DB_FILE = "attendai.db"
if DATABASE_URL.startswith("sqlite"):
    try:
        probe = sqlite3.connect(_DB_FILE)
        probe.execute("SELECT 1")
        probe.close()
    except sqlite3.OperationalError:
        _DB_FILE = os.path.join(tempfile.gettempdir(), "attendai.db")
        DATABASE_URL = "sqlite:///" + _DB_FILE.replace(os.sep, "/")

# If using Render Postgres, ensure async drivers are not required by SQLModel
engine = create_engine(DATABASE_URL, echo=False)


def init_db() -> None:
    """
    Initializes the database tables..
    Also ensures 'admission_number' column exists in 'user' table..
    """
    try:
        # Create all tables if they don't exist
        SQLModel.metadata.create_all(engine)

        # Only run SQLite-specific migration when using SQLite
        if DATABASE_URL.startswith("sqlite"):
            conn = sqlite3.connect(_DB_FILE)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(user)")
            columns = [col[1] for col in cursor.fetchall()]
            if "admission_number" not in columns:
                cursor.execute("ALTER TABLE user ADD COLUMN admission_number TEXT;")
                print("Added 'admission_number' column to user table.")
            conn.commit()
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never crash startup (e.g., ephemeral serverless DB)
        print(f"init_db warning (continuing, {exc})")


def get_session():
    with Session(engine) as session:
        yield session