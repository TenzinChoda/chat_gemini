"""Quick checks after ingest: manifest + Chroma collection document count."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings


def main() -> int:
    s = get_settings()
    manifest = s.project_root / "database" / "ingest_manifest.json"
    if not manifest.is_file():
        print(f"FAIL: manifest missing at {manifest} (run python pipeline/ingest.py)")
        return 1
    with open(manifest, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Manifest collection: {data.get('collection_name')}")
    print(f"Indexed documents (manifest): {data.get('indexed_documents')}")
    print(f"Dedup stats: {data.get('deduplication')}")

    try:
        import chromadb
    except ImportError:
        print("chromadb not importable; skipping collection count.")
        return 0

    client = chromadb.PersistentClient(path=str(s.chroma_persist_dir))
    try:
        col = client.get_collection(s.chroma_collection_name)
        n = col.count()
        print(f"Chroma collection {s.chroma_collection_name!r} count: {n}")
    except Exception as e:
        print(f"Could not read Chroma collection: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
