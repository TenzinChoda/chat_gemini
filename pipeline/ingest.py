"""
ETL: load Excel / JSON / CSV / PDF knowledge into ChromaDB with cleaning and deduplication.

Steps:
  1) Load raw documents from data/raw/
  2) Clean text (embedding-friendly, boilerplate stripped)
  3) Deduplicate across sources (website JSON > structured tables > tickets > CSV > PDF)
  4) Reset vector store and persist a new Chroma collection

Run from bt_chatbot/: python pipeline/ingest.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings
from pipeline.deduplicate import deduplicate_documents
from pipeline.text_normalize import clean_text_for_embedding, legacy_clean_description

_s = get_settings()
DATA_RAW = _s.data_raw_dir
DIR_JSON = _s.dir_json
DIR_EXCEL = _s.dir_excel
DIR_CSV = _s.dir_csv
DIR_PDF = _s.dir_pdf
CHROMA_DIR = _s.chroma_persist_dir
COLLECTION_NAME = _s.chroma_collection_name
EMBED_MODEL = _s.embedding_model
INGEST_RESET_CHROMA = _s.ingest_reset_chroma
MANIFEST_PATH = _s.project_root / "database" / "ingest_manifest.json"

_CATEGORY_TO_COMPLAINT: dict[str, str] = {
    "pricing": "Internet",
    "device_sales": "Mobile",
    "network": "Internet",
    "billing": "Mobile",
    "support": "Customercare/Info Desk",
    "info": "Enquery",
    "vas": "Mobile",
    "hosting": "IT services",
    "leased": "Internet",
    "mobile": "Mobile",
    "fixed": "Fixed Line",
}


def normalize_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        col_map[c] = key
    df = df.rename(columns=col_map)
    renames = {}
    if "ticket_id" not in df.columns:
        for cand in ("ticket id", "ticketid"):
            if cand in df.columns:
                renames[cand] = "ticket_id"
                break
    if "ticket_date" not in df.columns and "ticket_date_time" in df.columns:
        renames["ticket_date_time"] = "ticket_date"
    if "ticket_date" not in df.columns and "ticket date" in df.columns:
        renames["ticket date"] = "ticket_date"
    df = df.rename(columns=renames)
    return df


def complaint_from_category(cat: str | None) -> str:
    if not cat:
        return "Others"
    k = str(cat).lower().strip()
    return _CATEGORY_TO_COMPLAINT.get(k, "Others")


def row_dict_to_text(data: dict) -> str:
    parts = []
    for k, v in data.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if v == "":
            continue
        parts.append(f"{k}: {v}")
    return "; ".join(parts)


def _resolve_json(name: str) -> Path | None:
    p = DIR_JSON / name
    if p.is_file():
        return p
    legacy = DATA_RAW / name
    if legacy.is_file():
        return legacy
    return None


def _meta_base(
    *,
    source_origin: str,
    source_path: str,
    ticket_id: str,
    complaint_type: str,
    service: str,
    status: str,
    month: str,
    extra: dict | None = None,
) -> dict:
    m = {
        "source_origin": source_origin,
        "source_path": source_path,
        "ticket_id": ticket_id,
        "complaint_type": complaint_type,
        "service": service,
        "status": status,
        "month": month,
    }
    if extra:
        m.update(extra)
    return m


def load_excel_files() -> tuple[list[Document], dict[str, int]]:
    docs: list[Document] = []
    counts: dict[str, int] = {}
    paths: list[Path] = []
    if DIR_EXCEL.is_dir():
        paths.extend(sorted(DIR_EXCEL.glob("*.xlsx")))
    paths.extend(sorted(DATA_RAW.glob("*.xlsx")))
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique_paths.append(p)

    for xlsx in unique_paths:
        sheet_counts = 0
        xl = pd.ExcelFile(xlsx)
        rel = f"excel/{xlsx.name}"
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = normalize_excel_columns(df)
            if "ticket_id" not in df.columns or "description" not in df.columns:
                continue
            if "service" not in df.columns:
                df["service"] = ""
            if "complaint_type" not in df.columns:
                df["complaint_type"] = "Others"
            if "status" not in df.columns:
                df["status"] = "Close"
            if "ticket_date" not in df.columns:
                df["ticket_date"] = ""
            for _, row in df.iterrows():
                tid = str(row.get("ticket_id", "") or "").strip()
                if not tid.startswith("BTL"):
                    continue
                raw_desc = str(row.get("description", "") or "")
                desc = clean_text_for_embedding(raw_desc)
                if len(legacy_clean_description(desc)) < 3:
                    continue
                service = str(row.get("service", "") or "")[:500]
                month = str(row.get("ticket_date", "") or "")[:32]
                if not month:
                    month = sheet
                page = f"Service: {service}. Issue: {desc}"
                meta = _meta_base(
                    source_origin="excel_ticket",
                    source_path=rel,
                    ticket_id=tid,
                    complaint_type=str(row.get("complaint_type", "") or "Others"),
                    service=service,
                    status=str(row.get("status", "") or "Close"),
                    month=month,
                )
                docs.append(Document(page_content=page, metadata=meta))
                sheet_counts += 1
        counts[xlsx.name] = sheet_counts
    return docs, counts


def load_bt_dataset_json(path: Path) -> tuple[list[Document], int]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    rel = f"json/{path.name}"
    for i, item in enumerate(data):
        title = str(item.get("title", "") or "Website")
        content = str(item.get("content", "") or "")
        url = str(item.get("url", "") or "")
        body = clean_text_for_embedding(content)
        if len(legacy_clean_description(body)) < 3:
            continue
        tid = f"BTL-WEB-{i:06d}"
        service = title[:500]
        page = f"Topic: {service}. Information: {body}"
        meta = _meta_base(
            source_origin="bt_web",
            source_path=rel,
            ticket_id=tid,
            complaint_type="Enquery",
            service=service,
            status="Close",
            month="web",
            extra={"source_url": url[:2000]},
        )
        docs.append(Document(page_content=page, metadata=meta))
        n += 1
    return docs, n


def load_structured_tables_json(path: Path) -> tuple[list[Document], int]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    tables = root.get("tables", [])
    n = 0
    rel = f"json/{path.name}"
    for ti, table in enumerate(tables):
        tname = str(table.get("table_name", "table"))
        cat = table.get("category")
        complaint = complaint_from_category(cat)
        for row in table.get("rows", []):
            rd = row.get("data", {})
            raw_joined = row_dict_to_text(rd)
            text = clean_text_for_embedding(raw_joined)
            if len(legacy_clean_description(text)) < 3:
                continue
            ri = row.get("row_number", n)
            tid = f"BTL-TBL-{ti}-{ri}"
            service = tname[:500]
            page = f"Service: {service}. Issue: {text}"
            meta = _meta_base(
                source_origin="structured_json",
                source_path=rel,
                ticket_id=tid,
                complaint_type=complaint,
                service=service,
                status="Close",
                month="structured",
            )
            docs.append(Document(page_content=page, metadata=meta))
            n += 1
    return docs, n


def load_csv_files() -> tuple[list[Document], dict[str, int]]:
    docs: list[Document] = []
    counts: dict[str, int] = {}
    if not DIR_CSV.is_dir():
        return docs, counts

    for ci, csv_path in enumerate(sorted(DIR_CSV.glob("*.csv"))):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: could not read {csv_path.name}: {e}")
            counts[csv_path.name] = 0
            continue

        stem = csv_path.stem[:500]
        rel = f"csv/{csv_path.name}"
        added = 0
        for j, (_, row) in enumerate(df.iterrows()):
            d = row.to_dict()
            text = clean_text_for_embedding(row_dict_to_text(d))
            if len(legacy_clean_description(text)) < 3:
                continue
            tid = f"BTL-CSV-{ci}-{j}"
            page = f"Service: {stem}. Issue: {text}"
            meta = _meta_base(
                source_origin="csv_row",
                source_path=rel,
                ticket_id=tid,
                complaint_type="Others",
                service=stem,
                status="Close",
                month="csv",
                extra={"source_file": csv_path.name},
            )
            docs.append(Document(page_content=page, metadata=meta))
            added += 1
        counts[csv_path.name] = added

    return docs, counts


def load_pdf_directory(pdf_dir: Path) -> tuple[list[Document], dict[str, int]]:
    docs: list[Document] = []
    per_file: dict[str, int] = {}
    if not pdf_dir.is_dir():
        return docs, per_file

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    for fi, pdf_path in enumerate(pdf_paths):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as e:
            print(f"Warning: could not read {pdf_path.name}: {e}")
            per_file[pdf_path.name] = 0
            continue

        stem = pdf_path.stem[:500]
        rel = f"pdf/{pdf_path.name}"
        added = 0
        for pi, page in enumerate(pages):
            raw = page.page_content or ""
            body = clean_text_for_embedding(raw)
            if len(legacy_clean_description(body)) < 3:
                continue
            pnum = pi + 1
            tid = f"BTL-PDF-{fi:03d}-{pnum:04d}"
            page_text = f"Document: {stem} (page {pnum}). Content: {body}"
            meta = _meta_base(
                source_origin="pdf_page",
                source_path=rel,
                ticket_id=tid,
                complaint_type="Enquery",
                service=stem,
                status="Close",
                month="pdf",
                extra={"source_file": pdf_path.name, "page": pnum},
            )
            docs.append(Document(page_content=page_text, metadata=meta))
            added += 1
        per_file[pdf_path.name] = added

    return docs, per_file


def reset_vector_store_dir() -> None:
    if CHROMA_DIR.is_dir() and INGEST_RESET_CHROMA:
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for d in (DATA_RAW, DIR_JSON, DIR_EXCEL, DIR_CSV, DIR_PDF):
        d.mkdir(parents=True, exist_ok=True)

    all_docs: list[Document] = []
    file_counts: dict[str, int] = {}

    excel_docs, excel_counts = load_excel_files()
    for k, v in excel_counts.items():
        if v:
            file_counts[k] = v
            print(f"Loaded {v} tickets from excel/{k}")
    all_docs.extend(excel_docs)

    web_path = _resolve_json("bt_dataset.json")
    if web_path:
        d, c = load_bt_dataset_json(web_path)
        all_docs.extend(d)
        file_counts[web_path.name] = c
        print(f"Loaded {c} documents from json/{web_path.name}")

    struct_path = _resolve_json("all_csv_structured_data.json")
    if struct_path:
        d, c = load_structured_tables_json(struct_path)
        all_docs.extend(d)
        file_counts[struct_path.name] = c
        print(f"Loaded {c} row-documents from json/{struct_path.name}")

    csv_docs, csv_counts = load_csv_files()
    all_docs.extend(csv_docs)
    for k, v in sorted(csv_counts.items()):
        if v:
            file_counts[f"csv/{k}"] = v
            print(f"Loaded {v} rows from csv/{k}")

    pdf_docs_list, pdf_counts = load_pdf_directory(DIR_PDF)
    all_docs.extend(pdf_docs_list)
    total_pdf_pages = sum(pdf_counts.values())
    if DIR_PDF.is_dir() and total_pdf_pages:
        for fname, cnt in sorted(pdf_counts.items()):
            if cnt:
                print(f"Loaded {cnt} pages from pdf/{fname}")
        print(f"Loaded {total_pdf_pages} PDF page-documents from {DIR_PDF}")
    elif DIR_PDF.is_dir():
        print(f"No PDF pages loaded from {DIR_PDF} (add .pdf files or check extraction)")

    if not all_docs:
        print(
            "No documents ingested. Populate data/raw/json/, excel/, csv/, pdf/ "
            "(see docstring at top of pipeline/ingest.py)."
        )
        return

    print(f"Total raw documents: {len(all_docs)}")
    deduped, stats = deduplicate_documents(all_docs)
    print(
        "Deduplication: "
        f"input={stats['input']}, kept={stats['output']}, "
        f"dropped_duplicate={stats['dropped_duplicate']}, "
        f"dropped_too_short={stats['dropped_too_short']}"
    )

    reset_vector_store_dir()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=deduped,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Total documents indexed: {len(deduped)}")
    print(f"Chroma collection={COLLECTION_NAME!r} persisted to {CHROMA_DIR}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBED_MODEL,
        "chroma_dir": str(CHROMA_DIR),
        "file_counts": file_counts,
        "deduplication": stats,
        "indexed_documents": len(deduped),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
