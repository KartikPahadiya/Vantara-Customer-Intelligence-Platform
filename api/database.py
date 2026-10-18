"""Database layer (PRD Section 12.1/14).

Scores and segment assignments are persisted so the dashboard does not
recompute them on every page load. The connection string comes from the
``DATABASE_URL`` environment variable; it defaults to a local SQLite file
so the stack runs without Docker, while ``docker-compose.yml`` injects
the PostgreSQL service URL.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./vantara.db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One scored customer snapshot (batch-scoring job output)."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, index=True, nullable=True)
    churn_probability = Column(Float, nullable=False)
    churn_label = Column(Integer, nullable=False)
    clv_estimate = Column(Float, nullable=False)
    segment = Column(Integer, nullable=False)
    model_name = Column(String(64), nullable=False)
    scored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SegmentAssignment(Base):
    """Persisted segment assignment per customer."""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, index=True, unique=True, nullable=False)
    segment = Column(Integer, nullable=False)
    segment_label = Column(String(128), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(engine)
