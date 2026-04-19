#!/usr/bin/env bash
# Rebuild Chroma from parent Ganga_Data_new/data (when present) and run verification.
set -euo pipefail
cd "$(dirname "$0")/.."
python pipeline/sync_parent_data.py
python pipeline/ingest.py
python pipeline/verify_kb.py
