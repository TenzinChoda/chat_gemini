"""Normalize and clean text for embedding and for duplicate detection."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

# Strip common BT website boilerplate so PDF/web duplicates match more often.
_BOILERPLATE_TAIL = re.compile(
    r"(?:\s*Copyright\s*©?\s*\d{4}\s*Bhutan Telecom Ltd\.?\s*All rights reserved\.\s*Designed by\s*)$",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_CALL = re.compile(r"^\s*call\s+", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_bt_boilerplate(text: str) -> str:
    """Remove trailing copyright/footer lines often repeated across scraped pages."""
    if not text:
        return ""
    s = text.strip()
    prev = None
    while prev != s:
        prev = s
        s = _BOILERPLATE_TAIL.sub("", s).strip()
    return s


def clean_text_for_embedding(text: str) -> str:
    """
    Light cleaning for indexed text: keep punctuation and casing where useful for meaning,
    unlike the old all-lowercase punctuation-stripping path.
    """
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS.sub("", s)
    s = _LEADING_CALL.sub("", s)
    s = strip_bt_boilerplate(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_for_dedup(text: str) -> str:
    """
    Canonical form for duplicate detection across files (case/punctuation insensitive).
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text).lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def content_fingerprint(text: str) -> str:
    """SHA256 hex of normalized dedup string."""
    n = normalize_for_dedup(text)
    return sha256(n.encode("utf-8")).hexdigest()


def legacy_clean_description(text: str) -> str:
    """Backward-compatible aggressive normalize (used only where old behaviour is required)."""
    return normalize_for_dedup(text)
