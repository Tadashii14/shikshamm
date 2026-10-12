"""Regression tests for production database-fallback safety.

Verifies:
- Local dev / tests (SQLite) stay available.
- A configured but unreachable remote database does NOT silently fall back to
  ephemeral SQLite, and marks the app DB as unavailable (app.db.DB_AVAILABLE).
- /health reports the database honestly when it is unavailable.
- Database error responses/logs never expose the connection string/credentials.

tests/conftest.py supplies a fresh temporary SQLite database for each run.

No real attendai.db is touched; no external database is contacted (the
unreachable-DB case uses a reserved local port that refuses instantly).
"""
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db import init_db
import app.db as db
from app.services.monitoring import health_checker

init_db()
client = TestClient(app)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable


class TestSqliteRemainsAvailable:
    def test_sqlite_url_is_available(self):
        # The test process is configured with a temp SQLite DB -> available.
        assert db.DATABASE_URL.startswith("sqlite")
        assert db.DB_AVAILABLE is True
        assert db.DB_ERROR is None

    def test_health_reports_database_healthy(self):
        status = health_checker.check_database()
        assert status["status"] == "healthy"


class TestUnreachableRemoteDoesNotFallBack:
    """Faithfully exercise app.db's import-time decision in a subprocess so we
    never mutate this test process's module state. Port 1 refuses instantly, so
    no external database is ever contacted."""

    def _import_db(self, env_url):
        code = (
            "import app.db as d; "
            "print('AVAILABLE', d.DB_AVAILABLE); "
            "print('IS_SQLITE', d.DATABASE_URL.startswith('sqlite'))"
        )
        env = dict(os.environ)
        for key in ("ATTENDAI_HOSTED_RUNTIME", "VERCEL", "VERCEL_ENV", "VERCEL_URL", "RENDER", "RENDER_SERVICE_ID"):
            env.pop(key, None)
        env["DATABASE_URL"] = env_url
        return subprocess.run(
            [PY, "-c", code], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=60,
        )

    def test_unreachable_postgres_marks_unavailable_no_sqlite_fallback(self):
        out = self._import_db("postgresql://user:supersecretpw@127.0.0.1:1/nope")
        combined = out.stdout + out.stderr
        assert "AVAILABLE False" in out.stdout
        assert "IS_SQLITE False" in out.stdout
        # startup must have logged a clear error and NOT leaked the password
        assert "WARNING: DATABASE_URL connection probe failed" in combined
        assert "supersecretpw" not in combined

    def test_empty_sqlite_url_stays_available(self):
        # `sqlite://` (no path) -> temp location, still available, never remote.
        out = self._import_db("sqlite://")
        assert "AVAILABLE True" in out.stdout
        assert "IS_SQLITE True" in out.stdout

    def test_vercel_without_database_url_fails_closed(self):
        env = dict(os.environ)
        env.pop("DATABASE_URL", None)
        for key in ("ATTENDAI_HOSTED_RUNTIME", "VERCEL", "VERCEL_ENV", "VERCEL_URL", "RENDER", "RENDER_SERVICE_ID"):
            env.pop(key, None)
        env["VERCEL"] = "1"
        out = subprocess.run(
            [PY, "-c", "import app.db as d; print('AVAILABLE', d.DB_AVAILABLE); print('URL', repr(d.DATABASE_URL))"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        assert out.returncode == 0
        assert "AVAILABLE False" in out.stdout
        assert "URL ''" in out.stdout
        assert "attendai.db" not in out.stdout

    def test_render_with_sqlite_url_fails_closed(self):
        env = dict(os.environ)
        for key in ("ATTENDAI_HOSTED_RUNTIME", "VERCEL", "VERCEL_ENV", "VERCEL_URL", "RENDER", "RENDER_SERVICE_ID"):
            env.pop(key, None)
        env["RENDER"] = "true"
        env["DATABASE_URL"] = "sqlite:///./attendai_demo.db"
        code = "import app.db as d; print('AVAILABLE', d.DB_AVAILABLE); print('URL', d.DATABASE_URL)"
        out = subprocess.run([PY, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0
        assert "AVAILABLE False" in out.stdout
        assert "Persistent database storage is required" in out.stdout + out.stderr

    def test_vercel_entrypoint_marks_runtime_before_importing_app(self):
        env = dict(os.environ)
        env.pop("DATABASE_URL", None)
        for key in ("ATTENDAI_HOSTED_RUNTIME", "VERCEL", "VERCEL_ENV", "VERCEL_URL", "RENDER", "RENDER_SERVICE_ID"):
            env.pop(key, None)
        code = "import api.index; import app.db as d; print('AVAILABLE', d.DB_AVAILABLE); print('URL', repr(d.DATABASE_URL))"
        out = subprocess.run(
            [PY, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        assert out.returncode == 0
        assert "AVAILABLE False" in out.stdout
        assert "URL ''" in out.stdout


class TestHealthHonestyAndNoSecretLeak:
    def test_health_unhealthy_when_db_unavailable(self, monkeypatch):
        monkeypatch.setattr(db, "DB_AVAILABLE", False)
        monkeypatch.setattr(
            db, "DB_ERROR",
            "Configured database is unreachable; refusing to fall back to "
            "ephemeral SQLite to avoid silent data loss.",
        )
        status = health_checker.check_database()
        assert status["status"] == "unhealthy"
        assert "unreachable" in status["message"].lower()

    def test_health_recovers_when_remote_db_becomes_available(self, monkeypatch):
        import app.services.monitoring as monitoring

        monkeypatch.setattr(db, "DB_AVAILABLE", False)
        monkeypatch.setattr(db, "DB_ERROR", "Configured database is unreachable")
        monkeypatch.setattr(db, "DATABASE_URL", "postgresql+psycopg://user:secret@db.example/demo")
        monkeypatch.setattr(db, "engine", object())
        monkeypatch.setattr(db, "_can_connect", lambda _url: True)

        def initialize_after_reconnect():
            db.DB_AVAILABLE = True
            db.DB_ERROR = None

        monkeypatch.setattr(db, "init_db", initialize_after_reconnect)

        class _Rows:
            def all(self):
                return []

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def exec(self, *_args, **_kwargs):
                return _Rows()

        monkeypatch.setattr(monitoring, "Session", lambda _engine: _Session())
        status = monitoring.HealthChecker().check_database()

        assert status["status"] == "healthy"
        assert db.DB_AVAILABLE is True
        assert db.DB_ERROR is None

    def test_db_error_message_has_no_credentials(self, monkeypatch):
        # Simulate a driver error whose text embeds a DSN with a password.
        monkeypatch.setattr(db, "DB_AVAILABLE", True)
        monkeypatch.setattr(db, "DB_ERROR", None)

        class _Sess:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def exec(self, *a, **k):
                raise RuntimeError(
                    "could not connect to postgresql://user:supersecretpw@host/db"
                )

        import app.services.monitoring as mon
        monkeypatch.setattr(mon, "Session", lambda _engine: _Sess())
        status = health_checker.check_database()
        assert status["status"] == "unhealthy"
        assert "supersecretpw" not in status["message"]
        assert status["message"] == "Database operation failed; check database connectivity and schema."

    def test_health_endpoint_still_returns_200(self):
        # Render/Vercel health checks only need HTTP 200; the body carries truth.
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "checks" in body and "database" in body["checks"]

    def test_runtime_database_failure_is_safe_and_returns_503(self, monkeypatch, capsys):
        secret_dsn = "postgresql://user:supersecretpw@host/database"
        monkeypatch.setattr(db, "DB_AVAILABLE", True)
        monkeypatch.setattr(db, "DB_ERROR", None)
        db.mark_database_unavailable(RuntimeError(f"could not connect to {secret_dsn}"))

        response = client.get("/admin/students/list")
        assert response.status_code == 503
        assert "supersecretpw" not in response.text

        health = client.get("/health")
        assert health.status_code == 200
        assert "supersecretpw" not in health.text
        assert "Database unavailable" in health.text

        captured = capsys.readouterr()
        assert "supersecretpw" not in captured.out + captured.err

    def test_init_failure_marks_database_unhealthy_without_logging_dsn(self, monkeypatch, capsys):
        monkeypatch.setattr(db, "DB_AVAILABLE", True)
        monkeypatch.setattr(db, "DB_ERROR", None)

        def fail_create_all(*_args, **_kwargs):
            raise RuntimeError("connection failed for postgresql://user:supersecretpw@host/database")

        monkeypatch.setattr(db.SQLModel.metadata, "create_all", fail_create_all)
        init_db()

        assert db.DB_AVAILABLE is False
        assert "supersecretpw" not in (db.DB_ERROR or "")
        captured = capsys.readouterr()
        assert "supersecretpw" not in captured.out + captured.err
