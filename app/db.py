from __future__ import annotations

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from fastapi import HTTPException
import os
import sqlite3
import tempfile

from app.models import User  # make sure User has admission_number in models.py

# Local development can use SQLite. Hosted environments must be configured
# with persistent storage explicitly; a local SQLite file is ephemeral there.
_HOSTED_RUNTIME = any(os.getenv(key) for key in (
    "ATTENDAI_HOSTED_RUNTIME", "VERCEL", "VERCEL_ENV", "VERCEL_URL", "RENDER", "RENDER_SERVICE_ID"
))
_configured_database_url = os.getenv("DATABASE_URL")
if _configured_database_url:
    DATABASE_URL = _configured_database_url
elif _HOSTED_RUNTIME:
    DATABASE_URL = ""
else:
    DATABASE_URL = "sqlite:///./attendai_demo.db"

DB_AVAILABLE = True
DB_ERROR: str | None = None
DATABASE_UNAVAILABLE_DETAIL = (
    "Database unavailable. Attendance and student data were not saved; "
    "check the configured database and schema."
)

if _HOSTED_RUNTIME and not DATABASE_URL:
    DB_AVAILABLE = False
    DB_ERROR = "DATABASE_URL is not configured; persistent storage is required on hosted deployments."
    print("ERROR: DATABASE_URL is not configured for this hosted runtime. SQLite fallback is disabled.")
elif _HOSTED_RUNTIME and DATABASE_URL.startswith("sqlite://"):
    DB_AVAILABLE = False
    DB_ERROR = "Hosted deployments require persistent database storage; SQLite is disabled."
    print("ERROR: hosted runtime configured with SQLite. Persistent database storage is required; no fallback is used.")

# Normalize scheme and ensure psycopg (v3) driver for Postgres
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if DATABASE_URL.startswith("postgresql://") and "+" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)


def _can_connect(url: str) -> bool:
    """Quickly check a configured remote database without exposing its DSN."""
    if not url.startswith(("postgresql", "mysql", "mssql")):
        return True
    test_engine = None
    try:
        connect_args = {"connect_timeout": 5} if url.startswith("postgresql") else {}
        test_engine = create_engine(url, connect_args=connect_args)
        with test_engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except Exception as exc:  # noqa: BLE001
        # Driver errors can include connection details. Log the exception type
        # only; never print a DSN or raw driver message.
        print(f"WARNING: DATABASE_URL connection probe failed ({exc.__class__.__name__}); details hidden.")
        return False
    finally:
        if test_engine is not None:
            test_engine.dispose()


def mark_database_unavailable(exc: Exception, context: str = "database operation") -> None:
    """Record a sanitized database failure for /health and API responses."""
    global DB_AVAILABLE, DB_ERROR
    DB_AVAILABLE = False
    if context == "connection probe":
        DB_ERROR = (
            "Configured database is unreachable; refusing to fall back to "
            "ephemeral SQLite to avoid silent data loss."
        )
    elif context == "initialization":
        DB_ERROR = "Database initialization failed; check database connectivity and schema."
    else:
        DB_ERROR = "Database operation failed; check database connectivity and schema."
    print(f"ERROR: {context} failed ({exc.__class__.__name__}); connection details hidden.")


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
elif DATABASE_URL and not _can_connect(DATABASE_URL):
    # A remote database was explicitly configured but is unreachable.
    # Do NOT silently switch to ephemeral SQLite: that would mask a
    # misconfigured database and make /health report "healthy" while
    # data is being written to storage that is lost on the next restart.
    # Keep the configured engine (lazy — no connection until first use) so the
    # failure stays visible, and record it so /health reports the DB unhealthy.
    DB_AVAILABLE = False
    DB_ERROR = (
        "Configured database is unreachable; refusing to fall back to "
        "ephemeral SQLite to avoid silent data loss."
    )

# If using Render Postgres, ensure async drivers are not required by SQLModel
engine = None
if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    except Exception as exc:  # noqa: BLE001 - keep startup and page routes available
        mark_database_unavailable(exc, "database engine setup")


def init_db() -> None:
    """
    Initializes the database tables..
    Also ensures 'admission_number' column exists in 'user' table..
    """
    global DB_AVAILABLE, DB_ERROR
    if not DB_AVAILABLE or engine is None:
        return
    try:
        # Create all tables if they don't exist
        SQLModel.metadata.create_all(engine)

        # Only run the SQLite-specific migration when using SQLite.
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
        DB_AVAILABLE = True
        DB_ERROR = None
    except Exception as exc:  # noqa: BLE001 - keep page routes available on DB failure
        mark_database_unavailable(exc, "initialization")


def get_session():
    if not DB_AVAILABLE or engine is None:
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL)

    with Session(engine) as session:
        try:
            yield session
        except IntegrityError:
            # Preserve endpoint-specific duplicate/conflict handling.
            raise
        except SQLAlchemyError as exc:
            mark_database_unavailable(exc)
            raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from None
