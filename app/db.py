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


def _writable_sqlite_url() -> str:
    """Return a SQLite URL in a writable location (CWD or /tmp on serverless)."""
    global _DB_FILE
    try:
        probe = sqlite3.connect(_DB_FILE)
        probe.execute("SELECT 1")
        probe.close()
        return "sqlite:///" + _DB_FILE.replace(os.sep, "/")
    except sqlite3.OperationalError:
        pass
    _DB_FILE = os.path.join(tempfile.gettempdir(), "attendai.db")
    return "sqlite:///" + _DB_FILE.replace(os.sep, "/")


def _can_connect(url: str) -> bool:
    """Quick connectivity probe so a dead/unreachable remote DB (e.g. expired
    Render Postgres) no longer takes the whole app down — we fall back to SQLite."""
    if not url.startswith(("postgresql", "mysql", "mssql")):
        return True
    try:
        connect_args = {"connect_timeout": 5} if url.startswith("postgresql") else {}
        test_engine = create_engine(url, connect_args=connect_args)
        with test_engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        test_engine.dispose()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: DATABASE_URL unreachable ({exc.__class__.__name__}: {exc})")
        return False


_DB_FILE = "attendai.db"
if DATABASE_URL.startswith("sqlite"):
    DATABASE_URL = _writable_sqlite_url()
elif not _can_connect(DATABASE_URL):
    # Remote DB is unreachable — degrade gracefully to SQLite so sessions,
    # attendance, notes and AI tools keep working instead of erroring on
    # every request.
    print("WARNING: falling back to SQLite (remote database unavailable)")
    DATABASE_URL = _writable_sqlite_url()

# If using Render Postgres, ensure async drivers are not required by SQLModel
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)


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