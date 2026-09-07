"""
Reset / manage local backend user credentials (JWT-based auth).

IMPORTANT: The public website login uses FIREBASE (Firebase Authentication for
the password + Firestore for the role). This script only manages users that
live in the FastAPI backend's OWN database (SQLite by default, or Postgres via
the DATABASE_URL env var). The web UI does NOT log in against this DB - it is
used for attendance, face, notes, etc.

For the hosted Firebase login credentials see: AUTH_RESET_GUIDE.md

Examples:
    # List all backend users
    python scripts/reset_local_credentials.py --list

    # Reset ONE user's password
    python scripts/reset_local_credentials.py --set-password admin@example.com 'NewPass123!'

    # Reset ALL passwords to the same value (useful for fresh/demo setup)
    python scripts/reset_local_credentials.py --reset-all --new-password 'Admin@12345'

    # Create a new admin user (e.g. if you lost the only account)
    python scripts/reset_local_credentials.py --create admin@example.com 'Admin@12345' admin 'System Admin'

    # Point at a Postgres DB (like on Render) instead of the default SQLite
    set DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
    python scripts/reset_local_credentials.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the app package importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, select  # noqa: E402

from app.auth import get_password_hash  # noqa: E402
from app.db import engine, init_db  # noqa: E402
from app.models import User  # noqa: E402


def list_users(session: Session) -> None:
    users = session.exec(select(User)).all()
    if not users:
        print("No users found in the backend database.")
        return
    print(f"{'ID':<5} {'Email':<40} {'Role':<12} {'Name':<25} Admission#")
    print("-" * 100)
    for u in users:
        print(f"{u.id:<5} {u.email:<40} {(u.role or ''):<12} {(u.full_name or ''):<25} {u.admission_number or ''}")


def set_password(session: Session, email: str, password: str) -> None:
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        print(f"User '{email}' not found.")
        sys.exit(1)
    user.hashed_password = get_password_hash(password)
    session.add(user)
    session.commit()
    print(f"Password updated for {email}.")


def reset_all(session: Session, password: str) -> None:
    users = session.exec(select(User)).all()
    if not users:
        print("No users in the database to reset.")
        return
    new_hash = get_password_hash(password)
    for u in users:
        u.hashed_password = new_hash
        session.add(u)
    session.commit()
    print(f"Reset password for ALL {len(users)} user(s) to the supplied value.")


def create_user(session: Session, email: str, password: str, role: str, full_name: str) -> None:
    existing = session.exec(select(User).where(User.email == email)).first()
    if existing:
        print(f"User '{email}' already exists. Use --set-password to update it.")
        sys.exit(1)
    if role not in ("admin", "student"):
        print("Role must be 'admin' or 'student'.")
        sys.exit(1)
    user = User(
        email=email,
        full_name=full_name,
        role=role,
        hashed_password=get_password_hash(password),
    )
    session.add(user)
    session.commit()
    print(f"Created user: {email} (role={role}, name={full_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List all backend DB users")
    parser.add_argument("--set-password", nargs=2, metavar=("EMAIL", "NEW_PASSWORD"),
                        help="Reset a single user's password")
    parser.add_argument("--reset-all", action="store_true",
                        help="Reset every user's password")
    parser.add_argument("--new-password", metavar="PASSWORD",
                        help="Password to use with --reset-all")
    parser.add_argument("--create", nargs=4, metavar=("EMAIL", "PASSWORD", "ROLE", "FULL_NAME"),
                        help="Create a new user (role: admin or student)")
    args = parser.parse_args()

    if not (args.list or args.set_password or args.reset_all or args.create):
        parser.print_help()
        sys.exit(1)

    # Ensure tables exist (creates attendai.db if using SQLite).
    # Wrap in try/except: on synced drives (OneDrive) create_all can hit a
    # transient 'table already exists' race when tables were just created.
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 - keep going, tables may already exist
        print(f"init_db warning (continuing): {exc}")

    with Session(engine) as session:
        if args.list:
            list_users(session)
        if args.set_password:
            email, password = args.set_password
            set_password(session, email, password)
        if args.reset_all:
            if not args.new_password:
                print("--reset-all requires --new-password <value>")
                sys.exit(1)
            reset_all(session, args.new_password)
        if args.create:
            email, password, role, full_name = args.create
            create_user(session, email, password, role, full_name)


if __name__ == "__main__":
    main()