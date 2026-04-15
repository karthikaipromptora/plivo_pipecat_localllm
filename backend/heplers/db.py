"""db.py — Neon PostgreSQL (asyncpg) connection pool and all DB operations.

Tables created on startup (idempotent):
  calls        — one row per outbound call
  transcripts  — one row per conversation turn (user / assistant)
  recordings   — one row per VoBiz recording file
"""

import os
import re
from datetime import datetime
from typing import Optional

import asyncpg
from loguru import logger

_pool: Optional[asyncpg.Pool] = None


# ── Init ──────────────────────────────────────────────────────────────────────


async def init_db() -> asyncpg.Pool:
    """Create connection pool and ensure tables exist. Called once at startup."""
    global _pool

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set in .env")

    # asyncpg doesn't support all postgres DSN params (e.g. channel_binding).
    # Strip query string and pass ssl='require' explicitly — works with Neon.
    clean_dsn = re.sub(r"\?.*$", "", dsn)

    _pool = await asyncpg.create_pool(
        clean_dsn,
        ssl="require",
        min_size=1,
        max_size=5,
        command_timeout=10,
    )

    await _create_tables()
    logger.info("DB: connected to Neon PostgreSQL, tables ready")
    return _pool


async def close_db():
    """Close pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("DB: pool closed")


async def _create_tables():
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id            SERIAL PRIMARY KEY,
                call_uuid     VARCHAR(255) UNIQUE NOT NULL,
                timestamp     TIMESTAMPTZ         NOT NULL,
                rider_name    VARCHAR(255),
                phone_number  VARCHAR(50),
                vehicle_model VARCHAR(100),
                end_date      VARCHAR(100),
                amount        VARCHAR(50),
                call_day      VARCHAR(10),
                language      VARCHAR(50),
                status        VARCHAR(50) DEFAULT 'initiated',
                sentiment     VARCHAR(20),
                takeaway      TEXT,
                callback      VARCHAR(3),
                created_at    TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                id          SERIAL PRIMARY KEY,
                call_uuid   VARCHAR(255) NOT NULL,
                role        VARCHAR(20)  NOT NULL,
                text        TEXT         NOT NULL,
                turn_index  INT          NOT NULL,
                created_at  TIMESTAMPTZ  DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS recordings (
                id            SERIAL PRIMARY KEY,
                call_uuid     VARCHAR(255) NOT NULL,
                recording_id  VARCHAR(255),
                recording_url TEXT,
                file_path     TEXT,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            );
        """)


# ── Calls ─────────────────────────────────────────────────────────────────────


async def insert_call(
    call_uuid: str,
    timestamp: datetime,
    rider_name: str,
    phone_number: str,
    vehicle_model: str,
    end_date: str,
    amount: str,
    call_day: str,
    language: str,
    status: str = "initiated",
):
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO calls
                (call_uuid, timestamp, rider_name, phone_number, vehicle_model,
                 end_date, amount, call_day, language, status)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT (call_uuid) DO NOTHING
            """,
            call_uuid, timestamp, rider_name, phone_number, vehicle_model,
            end_date, amount, call_day, language, status,
        )


async def get_calls() -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM calls ORDER BY timestamp DESC")
        return [_row(r) for r in rows]


async def get_call(call_uuid: str) -> Optional[dict]:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM calls WHERE call_uuid = $1", call_uuid
        )
        return _row(row) if row else None


async def update_call_analysis(call_uuid: str, sentiment: str, takeaway: str, callback: str):
    """Store sentiment + takeaway + callback after post-call LLM analysis."""
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE calls SET sentiment = $1, takeaway = $2, callback = $3 WHERE call_uuid = $4",
            sentiment, takeaway, callback, call_uuid,
        )


# ── Transcripts ───────────────────────────────────────────────────────────────


async def insert_transcript(call_uuid: str, turns: list[dict]):
    """Bulk-insert all turns for a call. turns = [{role, text}, ...]"""
    if not turns:
        return
    async with _pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO transcripts (call_uuid, role, text, turn_index)
            VALUES ($1, $2, $3, $4)
            """,
            [(call_uuid, t["role"], t["text"], i) for i, t in enumerate(turns)],
        )


async def get_transcript(call_uuid: str) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, text, turn_index FROM transcripts "
            "WHERE call_uuid = $1 ORDER BY turn_index",
            call_uuid,
        )
        return [_row(r) for r in rows]


# ── Recordings ────────────────────────────────────────────────────────────────


async def insert_recording(
    call_uuid: str,
    recording_id: str,
    recording_url: str,
    file_path: str,
):
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO recordings (call_uuid, recording_id, recording_url, file_path)
            VALUES ($1, $2, $3, $4)
            """,
            call_uuid, recording_id, recording_url, file_path,
        )


async def get_recording(call_uuid: str) -> Optional[dict]:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM recordings WHERE call_uuid = $1", call_uuid
        )
        return _row(row) if row else None


# ── Users ─────────────────────────────────────────────────────────────────────


async def create_user(email: str, password_hash: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ($1, $2)",
            email, password_hash,
        )


async def get_user_by_email(email: str) -> Optional[dict]:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1", email
        )
        return _row(row) if row else None


# ── Helper ────────────────────────────────────────────────────────────────────


def _row(record) -> dict:
    """Convert asyncpg Record to plain dict (JSON-serialisable)."""
    d = dict(record)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d
