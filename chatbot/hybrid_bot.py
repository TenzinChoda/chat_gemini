"""Orchestrator: rules → RAG → fallback, with SQLite logging."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from chatbot.rag_chain import get_rag_response
from chatbot.rule_engine import get_rule_response
from config.settings import get_settings

DB_PATH = get_settings().conversations_db_path

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a clear answer for your query. \n"
    "Please contact us directly:\n"
    "📞 Call: 1300\n"
    "🌐 Website: bt.bt\n"
    "📍 Visit any BT service center"
)

_GREETING_WORDS = frozenset(
    ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "namaste"]
)
_FAREWELL_WORDS = frozenset(
    ["bye", "goodbye", "see you", "thank you", "thanks", "later"]
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


def _is_greeting(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    return any(w in t for w in _GREETING_WORDS) and not any(
        x in t for x in ("sim", "data", "bill", "network", "problem", "issue")
    )


def _is_farewell(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(w in t for w in _FAREWELL_WORDS)


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

    if _is_farewell(msg):
        out = {
            "response": "Thank you for contacting BT. Have a great day! For urgent help, call 1300.",
            "intent": "farewell",
            "method": "greeting",
            "session_id": session_id,
        }
        _log_turn(session_id, msg, out["response"], out["intent"], out["method"])
        return out

    if _is_greeting(msg):
        out = {
            "response": (
                "Hello! I'm the BT Virtual Assistant. How can I help you today "
                "with mobile, internet, SIM, billing, or network?"
            ),
            "intent": "greeting",
            "method": "greeting",
            "session_id": session_id,
        }
        _log_turn(session_id, msg, out["response"], out["intent"], out["method"])
        return out

    rule_text, rule_intent = get_rule_response(msg)
    if rule_text and rule_intent:
        out = {
            "response": rule_text,
            "intent": rule_intent,
            "method": "rule",
            "session_id": session_id,
        }
        _log_turn(session_id, msg, out["response"], out["intent"], out["method"])
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
