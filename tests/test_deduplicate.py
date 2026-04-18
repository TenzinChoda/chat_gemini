"""Unit tests for cross-source deduplication."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.deduplicate import deduplicate_documents


def test_dedup_keeps_higher_priority_source() -> None:
    text = "Topic: VoLTE. Information: VoLTE uses 4G for clear voice calls."
    web = Document(
        page_content=text,
        metadata={"source_origin": "bt_web", "source_path": "json/bt_dataset.json"},
    )
    pdf = Document(
        page_content=text,
        metadata={"source_origin": "pdf_page", "source_path": "pdf/VoLTE-BTL.pdf"},
    )
    kept, stats = deduplicate_documents([pdf, web])
    assert stats["output"] == 1
    assert stats["dropped_duplicate"] == 1
    assert kept[0].metadata["source_origin"] == "bt_web"


def test_dedup_drops_identical_excel_rows() -> None:
    t = "Service: Mobile. Issue: customer reports no signal in thimphu"
    a = Document(
        page_content=t,
        metadata={"source_origin": "excel_ticket", "source_path": "excel/a.xlsx"},
    )
    b = Document(
        page_content=t,
        metadata={"source_origin": "excel_ticket", "source_path": "excel/b.xlsx"},
    )
    kept, stats = deduplicate_documents([a, b])
    assert stats["output"] == 1
    assert stats["dropped_duplicate"] == 1
