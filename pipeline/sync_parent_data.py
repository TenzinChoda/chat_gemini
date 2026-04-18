"""
Copy knowledge files from the parent project's data/ folder into data/raw/.

When this repo lives at .../Ganga_Data_new/bt_chatbot, source data is often maintained under
.../Ganga_Data_new/data/ (JSON exports and pdf_data/). This script syncs those into data/raw/json
and data/raw/pdf so ingest uses the latest copy.

Run from bt_chatbot/: python pipeline/sync_parent_data.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings

_JSON_NAMES = ("bt_dataset.json", "all_csv_structured_data.json")


def main() -> None:
    s = get_settings()
    parent_data = s.project_root.parent / "data"
    if not parent_data.is_dir():
        print(f"No sibling data folder found at {parent_data}. Nothing to sync.")
        return

    json_dir = s.dir_json
    pdf_out = s.dir_pdf
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in _JSON_NAMES:
        src = parent_data / name
        if src.is_file():
            dst = json_dir / name
            shutil.copy2(src, dst)
            print(f"Synced {src} -> {dst}")
            copied += 1

    pdf_src_dir = parent_data / "pdf_data"
    if pdf_src_dir.is_dir():
        for pdf in sorted(pdf_src_dir.glob("*.pdf")):
            shutil.copy2(pdf, pdf_out / pdf.name)
            print(f"Synced {pdf.name} -> {pdf_out / pdf.name}")
            copied += 1

    if copied == 0:
        print(f"No JSON or PDF files copied from {parent_data}.")
    else:
        print(f"Done. {copied} file(s) updated. Run: python pipeline/ingest.py")


if __name__ == "__main__":
    main()
