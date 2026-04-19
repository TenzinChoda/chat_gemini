"""Orchestrator: RAG (Chroma + Ollama) → fallback, with SQLite logging."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from chatbot.rag_chain import get_rag_response
from config.settings import get_settings

DB_PATH = get_settings().conversations_db_path

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a clear answer for your query. \n"
    "Please contact us directly:\n"
    "📞 Call: 1300\n"
    "🌐 Website: bt.bt\n"
    "📍 Visit any BT service center"
)


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                intent TEXT NOT NULL,
                method TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                session_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _log_turn(
    session_id: str,
    user_message: str,
    bot_response: str,
    intent: str,
    method: str,
    confidence: float | None = None,
) -> None:
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO conversations
            (session_id, user_message, bot_response, intent, method, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_message,
                bot_response,
                intent,
                method,
                datetime.now(timezone.utc).isoformat(),
                confidence,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def generate_response(user_message: str, session_id: str = "default") -> dict:
    msg = (user_message or "").strip()

    if not msg:
        out = {
            "response": "Please type your question about Bhutan Telecom services.",
            "intent": "empty",
            "method": "fallback",
            "session_id": session_id,
        }
        _log_turn(session_id, user_message or "", out["response"], out["intent"], out["method"])
        return out

    try:
        rag = get_rag_response(msg, session_id)
    except Exception:
        rag = {"response": "", "source_services": [], "confidence": 0.0, "error": "rag_error"}

    text = (rag.get("response") or "").strip()
    conf = float(rag.get("confidence") or 0.0)
    if text and not rag.get("error"):
        out = {
            "response": text,
            "intent": "rag",
            "method": "rag",
            "session_id": session_id,
            "confidence": conf,
            "source_services": rag.get("source_services") or [],
        }
        _log_turn(session_id, msg, out["response"], out["intent"], out["method"], confidence=conf)
        return out

    out = {
        "response": FALLBACK_MESSAGE,
        "intent": "fallback",
        "method": "fallback",
        "session_id": session_id,
    }
    _log_turn(session_id, msg, out["response"], out["intent"], out["method"], confidence=0.0)
    return out


_init_db()
