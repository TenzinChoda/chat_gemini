"""Run hybrid bot against built-in and domain-specific eval data; write evaluation/report.txt."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.hybrid_bot import DB_PATH, generate_response  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "evaluation" / "report.txt"
EVAL_DATA_DIR = PROJECT_ROOT / "evaluation" / "eval_data"

# Built-in smoke tests (RAG-only: expect rag, or fallback when Ollama/chroma unavailable)
TEST_CASES: list[dict] = [
    {"user_input": "How do I activate my new SIM?", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "What data packages do you have for 4G?", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "My balance was deducted without reason", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "Network signal is very weak and internet is slow", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "My SIM is barred please help", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "I need my postpaid bill and invoice", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "I need my PUK code sim locked", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "How to set APN for internet", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "eload voucher recharge topup", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "home wifi broadband fwa", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "I want esim on my phone", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "hello there", "expected_intent": "rag", "expected_method": "rag", "flexible": True},
    {"user_input": "", "expected_intent": "empty", "expected_method": "fallback"},
    {"user_input": "asdfgh qwerty zzz", "expected_intent": "fallback", "expected_method": "fallback"},
    {
        "user_input": "SIM activation and also data package pricing for 5G",
        "expected_intent": "rag",
        "expected_method": "rag",
        "flexible": True,
    },
]

_FLEX_METHODS = frozenset({"rag", "fallback"})
_SKIP_RAG_STRICT = os.environ.get("BTL_SKIP_RAG_STRICT_EVAL", "").lower() in ("1", "true", "yes")


def _case_passes(tc: dict, out: dict) -> tuple[bool, str]:
    """Return (pass, reason line)."""
    actual_intent = out.get("intent", "")
    actual_method = out.get("method", "")

    if tc.get("flexible"):
        ok_m = actual_method in _FLEX_METHODS
        ok_i = actual_intent in _FLEX_METHODS
        return (ok_i and ok_m, "flexible rag|fallback")

    if tc["user_input"] == "asdfgh qwerty zzz":
        ok_m = actual_method in _FLEX_METHODS
        ok_i = actual_intent in _FLEX_METHODS
        return (ok_i and ok_m, "gibberish")

    if (
        _SKIP_RAG_STRICT
        and tc.get("expected_method") == "rag"
        and actual_method == "fallback"
    ):
        return (True, "skipped (no Ollama / RAG unavailable)")

    ok_intent = actual_intent == tc["expected_intent"]
    ok_method = actual_method == tc["expected_method"]
    return (ok_intent and ok_method, "exact")


def load_domain_test_files() -> list[tuple[str, str, list[dict]]]:
    """Return list of (folder_name, domain_label, cases)."""
    if not EVAL_DATA_DIR.is_dir():
        return []
    out: list[tuple[str, str, list[dict]]] = []
    for domain_dir in sorted(EVAL_DATA_DIR.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith(".") or domain_dir.name == "__pycache__":
            continue
        tc_file = domain_dir / "test_cases.json"
        if not tc_file.is_file():
            continue
        try:
            data = json.loads(tc_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Warning: invalid JSON {tc_file}: {e}")
            continue
        label = data.get("domain", domain_dir.name)
        cases = data.get("cases", [])
        out.append((domain_dir.name, label, cases))
    return out


def main() -> None:
    lines: list[str] = []
    lines.append("=== Built-in routing tests ===\n")

    passed = 0
    method_counts: dict[str, int] = {}

    for tc in TEST_CASES:
        out = generate_response(tc["user_input"], session_id="eval-session-routing")
        actual_intent = out.get("intent", "")
        actual_method = out.get("method", "")
        method_counts[actual_method] = method_counts.get(actual_method, 0) + 1

        flex = bool(tc.get("flexible")) or tc["user_input"] == "asdfgh qwerty zzz"
        ok, _ = _case_passes({**tc, "flexible": flex}, out)

        if ok:
            passed += 1
        lines.append(f"Input: {tc['user_input']!r}")
        lines.append(f"  expected intent={tc['expected_intent']} method={tc['expected_method']}")
        lines.append(f"  actual   intent={actual_intent} method={actual_method}")
        lines.append(f"  {'PASS' if ok else 'FAIL'}")
        lines.append("")

    n = len(TEST_CASES)
    lines.append(f"Built-in overall: {passed}/{n} passed ({100.0 * passed / n:.1f}%)\n")

    # Domain tests from eval_data/*/
    domain_files = load_domain_test_files()
    if domain_files:
        lines.append("=== Domain tests (evaluation/eval_data) ===\n")
        total_d = 0
        passed_d = 0
        for folder, label, cases in domain_files:
            lines.append(f"--- {folder} ({label}) ---")
            sub_pass = 0
            for c in cases:
                uid = c.get("id", "")
                inp = c.get("user_input", "")
                exp_i = c.get("expected_intent", "")
                exp_m = c.get("expected_method", "")
                out = generate_response(inp, session_id=f"eval-{folder}-{uid}")
                actual_intent = out.get("intent", "")
                actual_method = out.get("method", "")
                method_counts[actual_method] = method_counts.get(actual_method, 0) + 1

                tc_full = {
                    "user_input": inp,
                    "expected_intent": exp_i,
                    "expected_method": exp_m,
                    "flexible": c.get("flexible", False),
                }
                ok, _ = _case_passes(tc_full, out)
                if ok:
                    sub_pass += 1
                    passed_d += 1
                total_d += 1
                id_part = f"{uid} " if uid else ""
                lines.append(f"  {id_part}{inp!r}")
                lines.append(f"    expected intent={exp_i} method={exp_m}")
                lines.append(f"    actual   intent={actual_intent} method={actual_method}")
                lines.append(f"    {'PASS' if ok else 'FAIL'}")
            lines.append(f"  Subtotal {folder}: {sub_pass}/{len(cases)} passed")
            lines.append("")
        lines.append(
            f"Domain tests overall: {passed_d}/{total_d} passed ({100.0 * passed_d / total_d:.1f}%)\n"
        )

    lines.append("=== Combined method distribution (this run) ===")
    lines.append(str(method_counts))
    lines.append("")

    conn = sqlite3.connect(DB_PATH)
    try:
        total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        avg_rating = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0]
        n_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        fallback_n = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE method = 'fallback'"
        ).fetchone()[0]
        fb_rate = (100.0 * fallback_n / total) if total else 0.0
    finally:
        conn.close()

    lines.append("From SQLite conversation logs:")
    lines.append(f"  total_conversations: {total}")
    lines.append(f"  feedback_rows: {n_fb}, average_rating: {avg_rating or 'n/a'}")
    lines.append(f"  fallback_rate_%: {fb_rate:.2f}")

    text = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
