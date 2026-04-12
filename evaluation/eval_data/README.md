# Domain evaluation data

Each subfolder is one **business domain** (aligned with BTL topics and indexed data: mobile, billing, network, leased line, etc.).

- **`test_cases.json`** — list of questions with expected routing (`expected_intent`, `expected_method`).
- Run **`python evaluation/evaluate.py`** from `bt_chatbot/` to execute built-in tests **and** all domain JSON files; see **`evaluation/report.txt`**.

**Optional fields per case**

- `"flexible": true` — for gibberish/noisy inputs: pass if `method` is `rag` or `fallback`.
- `"skip_if_no_ollama": true` — reserved for future use when tests require RAG only.

Add new domains by creating a folder + `test_cases.json` using the same schema as existing files.
