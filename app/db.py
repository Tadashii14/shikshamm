from sqlmodel import SQLModel, create_engine, Session
import sqlite3
from app.models import User  # make sure User has admission_number in models.py

DATABASE_URL = "sqlite:///./attendai.db"

engine = create_engine(DATABASE_URL, echo=False)


def init_db() -> None:
    """
    Initializes the database tables.
    Also ensures 'admission_number' column exists in 'user' table.
    """
    # Create all tables if they don't exist
    SQLModel.metadata.create_all(engine)

    # Ensure 'admission_number' column exists
    conn = sqlite3.connect("attendai.db")
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(user)")
    columns = [col[1] for col in cursor.fetchall()]
    if "admission_number" not in columns:
        cursor.execute("ALTER TABLE user ADD COLUMN admission_number TEXT;")
        print("Added 'admission_number' column to user table.")

    conn.commit()
    conn.close()


def get_session():
    with Session(engine) as session:
        yield session
