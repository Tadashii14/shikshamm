import os

# This entry point is only used by Vercel. Mark the runtime before importing
# app.db so a missing DATABASE_URL cannot quietly select local SQLite.
os.environ.setdefault("ATTENDAI_HOSTED_RUNTIME", "vercel")

from app.main import app as app


