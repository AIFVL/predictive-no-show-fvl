from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "backend" / "appointments.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    medic_id = Column(String, index=True)
    patient_id = Column(String, index=True)
    hour = Column(Integer, nullable=False)
    day = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    appointment_type = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Ensure new columns added to models are present in existing sqlite DB
    # (SQLite's create_all does not ALTER existing tables.)
    with engine.begin() as conn:
        try:
            res = conn.execute(text("PRAGMA table_info('appointments')")).all()
            existing_cols = [row[1] for row in res]
            if 'medic_id' not in existing_cols:
                # add medic_id column (nullable) for existing rows
                conn.execute(text("ALTER TABLE appointments ADD COLUMN medic_id VARCHAR"))
        except Exception:
            # if PRAGMA fails (no table yet), ignore — create_all handled it
            pass
