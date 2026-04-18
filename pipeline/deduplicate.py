"""Cross-source duplicate removal: keep highest-priority source per content fingerprint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

from pipeline.text_normalize import content_fingerprint, normalize_for_dedup

if TYPE_CHECKING:
    pass

# Lower number = higher priority when the same informational content appears in multiple files.
ORIGIN_PRIORITY: dict[str, int] = {
    "bt_web": 0,  # bt_dataset.json (curated website pages)
    "structured_json": 1,  # all_csv_structured_data.json tables
    "excel_ticket": 2,
    "csv_row": 3,
    "pdf_page": 4,  # often overlaps website PDFs; drop if already seen
}


def _rank(doc: Document) -> tuple[int, str]:
    meta = doc.metadata or {}
    origin = str(meta.get("source_origin") or "unknown")
    pr = ORIGIN_PRIORITY.get(origin, 99)
    path = str(meta.get("source_path") or meta.get("source_file") or "")
    return (pr, path)


def deduplicate_documents(documents: list[Document]) -> tuple[list[Document], dict[str, int]]:
    """
    Drop documents whose content fingerprint was already seen from a higher-priority source.
    Returns (kept_docs, stats).
    """
    stats: dict[str, int] = {
        "input": len(documents),
        "dropped_duplicate": 0,
        "dropped_too_short": 0,
        "output": 0,
    }
    sorted_docs = sorted(documents, key=_rank)
    seen: set[str] = set()
    kept: list[Document] = []

    for doc in sorted_docs:
        raw = doc.page_content or ""
        if len(normalize_for_dedup(raw)) < 8:
            stats["dropped_too_short"] += 1
            continue
        fp = content_fingerprint(raw)
        if not fp:
            continue
        if fp in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(fp)
        meta = dict(doc.metadata or {})
        meta["content_fingerprint"] = fp[:16]
        kept.append(Document(page_content=doc.page_content, metadata=meta))

    stats["output"] = len(kept)
    return kept, stats
