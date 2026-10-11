"""Regression tests for production database-fallback safety.

Verifies:
- Local dev / tests (SQLite) stay available.
- A configured but unreachable remote database does NOT silently fall back to
  ephemeral SQLite, and marks the app DB as unavailable (app.db.DB_AVAILABLE).
- /health reports the database honestly when it is unavailable.
- Database error responses/logs never expose the connection string/credentials.

Run with a temp DB, e.g.:
  set DATABASE_URL=sqlite:///./_dbfb_tmp.db
  .venv\\Scripts\\python.exe -m pytest tests/test_db_fallback.py -q

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
        assert "ERROR: DATABASE_URL is set but the database is unreachable" in combined
        assert "supersecretpw" not in combined

    def test_empty_sqlite_url_stays_available(self):
        # `sqlite://` (no path) -> temp location, still available, never remote.
        out = self._import_db("sqlite://")
        assert "AVAILABLE True" in out.stdout
        assert "IS_SQLITE True" in out.stdout


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

    def test_db_error_message_has_no_credentials(self, monkeypatch):
        # Simulate a driver error whose text embeds a DSN with a password.
        monkeypatch.setattr(db, "DB_AVAILABLE", True)

        class _Sess:
            def exec(self, *a, **k):
                raise RuntimeError(
                    "could not connect to postgresql://user:supersecretpw@host/db"
                )

            def close(self):
                pass

        import app.services.monitoring as mon
        monkeypatch.setattr(mon, "get_session", lambda: iter([_Sess()]))
        status = health_checker.check_database()
        assert status["status"] == "unhealthy"
        assert "supersecretpw" not in status["message"]
        assert status["message"] == "Database connection failed"

    def test_health_endpoint_still_returns_200(self):
        # Render/Vercel health checks only need HTTP 200; the body carries truth.
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "checks" in body and "database" in body["checks"]
