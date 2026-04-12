"""FastAPI server for the BT hybrid chatbot."""

from __future__ import annotations

import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot.hybrid_bot import DB_PATH, generate_response
from config.settings import get_settings

_settings = get_settings()
_cors = _settings.cors_allow_origins.strip()
_cors_list = ["*"] if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()]

app = FastAPI(title="BT Virtual Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"


class FeedbackIn(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)


@app.get("/")
def root():
    return {
        "service": "BT Virtual Assistant API",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
        "feedback": "POST /feedback",
        "stats": "/stats",
    }


@app.get("/chat")
def chat_help():
    """Browser GET returns 405 by default; this explains how to call the chat API."""
    return {
        "info": "This URL only accepts POST (not a normal browser visit).",
        "hint": "Send JSON: {\"message\": \"...\", \"session_id\": \"...\"}",
        "example": {"message": "How do I check my data balance?", "session_id": "my-session"},
        "try_it": "http://127.0.0.1:8000/docs — or open frontend/index.html",
    }


@app.post("/chat")
def chat(body: ChatIn):
    return generate_response(body.message, session_id=body.session_id)


@app.post("/feedback")
def feedback(body: FeedbackIn):
    from datetime import datetime, timezone

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO feedback (session_id, rating, timestamp) VALUES (?, ?, ?)",
            (body.session_id, body.rating, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "running", "model": _settings.ollama_model}


@app.get("/stats")
def stats():
    conn = sqlite3.connect(DB_PATH)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT method, COUNT(*) FROM conversations GROUP BY method"
        ).fetchall()
        dist = {m: c for m, c in rows}
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM conversations WHERE method = 'rag' AND confidence IS NOT NULL"
        ).fetchone()[0]
        n_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        avg_rating = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0]
    finally:
        conn.close()
    return {
        "total_conversations": total,
        "method_distribution": dist,
        "average_confidence_rag": round(float(avg_conf or 0), 4),
        "feedback_count": n_feedback,
        "average_rating": round(float(avg_rating or 0), 4) if avg_rating else None,
    }
