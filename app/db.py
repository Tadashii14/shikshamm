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


# Database availability tracking for honest /health reporting.
# DB_AVAILABLE becomes False ONLY when a REMOTE database was explicitly
# configured (DATABASE_URL points at Postgres/MySQL/MSSQL) but is unreachable.
# In that case we deliberately do NOT silently fall back to ephemeral SQLite,
# because that would let /health look healthy while attendance/notes are written
# to storage that is wiped on restart. Local dev (no DATABASE_URL) and tests
# (DATABASE_URL=sqlite:///...) always take the SQLite branch and stay available.
DB_AVAILABLE = True
DB_ERROR: str | None = None


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
if DATABASE_URL.startswith("sqlite://"):
    # An explicit file path is honored; only URLs without a file component
    # (e.g. sqlite://) fall back to a writable temp location so the app is safe
    # on ephemeral/serverless hosts.
    rest = DATABASE_URL[len("sqlite://"):]
    if rest.startswith("/"):
        rest = rest[1:]
    if not rest or rest == "/":
        # No file path given -> writable temp location (ephemeral-safe)
        _DB_FILE = os.path.join(tempfile.gettempdir(), "attendai.db")
        DATABASE_URL = "sqlite:///" + _DB_FILE.replace(os.sep, "/")
    else:
        # Keep the URL path relative (SQLite URIs with Windows drive letters
        # cannot open files reliably); _DB_FILE holds the absolute path so
        # sqlite3.connect() always finds the file.
        _DB_FILE = os.path.abspath(rest) if not os.path.isabs(rest) else rest
        DATABASE_URL = "sqlite:///" + rest
elif not _can_connect(DATABASE_URL):
    # A remote database was explicitly configured but is unreachable.
    # Do NOT silently switch to ephemeral SQLite: that would mask a
    # misconfigured/expired database and make /health report "healthy" while
    # data is being written to storage that is lost on the next restart.
    # Keep the configured engine (lazy — no connection until first use) so the
    # failure stays visible, and record it so /health reports the DB unhealthy.
    DB_AVAILABLE = False
    DB_ERROR = (
        "Configured database is unreachable; refusing to fall back to "
        "ephemeral SQLite to avoid silent data loss."
    )
    # Log clearly WITHOUT the connection string (which contains credentials).
    print(
        "ERROR: DATABASE_URL is set but the database is unreachable. "
        "The service will report the database as unhealthy on /health until "
        "the database is reachable. No connection string is logged."
    )

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
            try:
                # Database-level uniqueness for (session_id, user_id): new DBs get it via
                # the model constraint; SQLite needs an explicit index and it is idempotent.
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_attendance_session_user "
                    "ON attendance(session_id, user_id)"
                )
                conn.commit()
                print("Created unique index uq_attendance_session_user on attendance")
            except sqlite3.OperationalError as exc:
                print(f"WARNING: could not create unique index on attendance (duplicates may exist): {exc}")
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never crash startup (e.g., ephemeral serverless DB)
        print(f"init_db warning (continuing, {exc})")


def get_session():
    with Session(engine) as session:
        yield session