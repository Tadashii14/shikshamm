"""Run tests against a fresh SQLite database, never a developer's demo DB."""

import os
import tempfile
import uuid


_TEST_DATABASE_PATH = os.path.join(
    tempfile.gettempdir(), f"attendai-pytest-{uuid.uuid4().hex}.db"
)

# Tests must not accidentally connect to hosted services or append records to
# the developer's normal demo database. Individual subprocess tests set their
# own DATABASE_URL when exercising hosted configuration.
for _hosted_key in (
    "ATTENDAI_HOSTED_RUNTIME",
    "VERCEL",
    "VERCEL_ENV",
    "VERCEL_URL",
    "RENDER",
    "RENDER_SERVICE_ID",
):
    os.environ.pop(_hosted_key, None)

os.environ["DATABASE_URL"] = "sqlite:///" + _TEST_DATABASE_PATH.replace(os.sep, "/")


def pytest_sessionfinish(session, exitstatus):
    """Close and remove only the unique database created for this test run."""
    try:
        import app.db as db

        if db.engine is not None:
            db.engine.dispose()
    except Exception:
        pass

    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(_TEST_DATABASE_PATH + suffix)
        except FileNotFoundError:
            pass
        except OSError:
            # Windows may keep an SQLite sidecar open briefly as pytest exits.
            pass
