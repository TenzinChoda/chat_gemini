# BT Virtual Assistant

**A hybrid AI assistant for Bhutan Telecom Limited (BTL)** — it answers common customer questions using a mix of **instant keyword rules**, **retrieval from your own documents** (Excel, JSON, CSV, PDF), and a **local large language model** (Ollama). No paid cloud LLM API is required for the default setup.

---

## Run from GitHub (step-by-step)

Use this procedure after you **clone** the project from GitHub. You must run commands from the **`bt_chatbot`** directory — the folder that contains **`requirements.txt`**, **`api/`**, and **`frontend/`**.

If your repository has a parent folder (for example you clone into `my-project/` and the app is in `my-project/bt_chatbot/`), then **`cd bt_chatbot`** (or the correct path) before Step 2.

**Step 1 — Clone the repository**

```bash
git clone https://github.com/OmPrakash-Acharya99/Bt_chatbot.git
cd Bt_chatbot
```

Confirm you see **`requirements.txt`** in the current directory (`ls` on Linux/macOS, `dir` on Windows). This repository uses **`Bt_chatbot`** as the project root (there is no extra nested `bt_chatbot` folder after clone).

**Step 2 — Create a Python virtual environment**

```bash
python3 -m venv .venv
```

**Step 3 — Activate the virtual environment**

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Your prompt should show `(.venv)` when it is active.

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Install Ollama and download the chat model**

1. Install **[Ollama](https://ollama.com/)** for your operating system and start it (it usually runs in the background).
2. In a terminal:

```bash
ollama pull llama3
```

Ollama is required for answers. If Ollama is not running, the API will fall back to a generic contact message (1300 / bt.bt).

**Step 6 — Build the knowledge index**

This reads files under `data/raw/` and creates `database/chroma_db/`.

```bash
python pipeline/ingest.py
```

Run this command again whenever you add or change files under `data/raw/`.

**Step 7 — Start the FastAPI server**

Keep this terminal open.

```bash
uvicorn api.app:app --reload --port 8000
```

You should see that the server is listening on **http://127.0.0.1:8000** . Check [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) in a browser.

**Step 8 — Open the chat UI**

With the API running (Step 7), open the chat in your browser:

**[http://127.0.0.1:8000/ui/](http://127.0.0.1:8000/ui/)**

This is served by the same FastAPI process (no second terminal or port **8080** needed). You can also open `frontend/index.html` as a file or via another static server; the page will call the API on port **8000**.

**Step 9 — (Optional) Local configuration**

To override model names or paths without editing code, copy **`.env.example`** to **`.env`** and edit values. See [Configuration](#configuration).

---

## Who should read what?

| If you are… | Focus on |
|-------------|----------|
| **Anyone new to the repo** | [Run from GitHub (step-by-step)](#run-from-github-step-by-step) first |
| **Manager / product owner** | [How it works](#how-it-works-at-a-glance), [Limitations](#limitations--responsible-use), [Evaluation](#evaluation--quality-checks) |
| **Operations / support lead** | [Run from GitHub](#run-from-github-step-by-step), [Running the system](#running-the-system), [Data & knowledge base](#data--knowledge-base), [API overview](#api-overview) |
| **Developer / data engineer** | [Run from GitHub](#run-from-github-step-by-step), [Developer setup](#developer-setup), [Configuration](#configuration), [Repository layout](#repository-layout) |

---

## How it works (at a glance)

1. **RAG (retrieval-augmented generation)** — The system retrieves relevant passages from your indexed material (Chroma) and uses the local LLM to draft a natural, conversational reply grounded in that context.
2. **Fallback** — If retrieval or the model fails, or the answer is not supported by context, the user is directed to **1300**, **bt.bt**, or a service centre.

Conversation turns can be logged locally for review and statistics (see `database/`).

---

## Running the system

For a **complete procedure from `git clone`**, use [Run from GitHub (step-by-step)](#run-from-github-step-by-step). The section below is a shorter reference.

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** with the **`llama3`** model: `ollama pull llama3`

### One-time setup

```bash
cd bt_chatbot
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3
python pipeline/ingest.py          # builds the searchable knowledge index
```

### Start the backend (API)

```bash
source .venv/bin/activate
uvicorn api.app:app --reload --port 8000
```

- **Health check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)  
- **API docs (try `/chat`):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Open the chat UI

The chat screen is a **local web page**, not served by the API:

- **macOS:** `open frontend/index.html`  
- Or open **`frontend/index.html`** from Finder / File Explorer.

The page talks to **`http://localhost:8000`** — keep the API running while you chat.

---

## Data & knowledge base

| Folder | What to put here |
|--------|-------------------|
| `data/raw/json/` | Website export JSON, structured table JSON (e.g. `bt_dataset.json`, `all_csv_structured_data.json`) |
| `data/raw/excel/` | Call-report **`.xlsx`** files (BTL ticket format per your spec) |
| `data/raw/csv/` | **`.csv`** tables (each row becomes one searchable chunk) |
| `data/raw/pdf/` | **`.pdf`** files (each page becomes one chunk) |

**After adding or changing files**, rebuild the index:

```bash
python pipeline/ingest.py
```

The vector database under `database/chroma_db/` is **rebuilt** from everything still present under `data/raw/` (keep old sources if you want them to stay searchable).

---

## Developer setup

Same as [One-time setup](#one-time-setup). Run all commands from the **`bt_chatbot/`** directory unless noted.

---

## Configuration

- Defaults live in **`config/settings.py`**.
- Override with **environment variables** prefixed with **`BTL_`** (see **`.env.example`**). Copy to **`.env`** for local use.
- Production: set variables in your host or orchestrator; **do not** commit secrets.

Examples: `BTL_OLLAMA_MODEL`, `BTL_CHROMA_PERSIST_DIR`, `BTL_CORS_ALLOW_ORIGINS`.

---

## API overview

**Base URL (local):** `http://127.0.0.1:8000`

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Short JSON overview of the service |
| `GET` | `/health` | Liveness; shows configured model name |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/chat` | Main chat: body `{"message": "…", "session_id": "…"}` |
| `POST` | `/feedback` | Star rating: `{"session_id": "…", "rating": 1–5}` |
| `GET` | `/stats` | Aggregates from logged conversations |

CORS is configurable (default allows all origins for local development).

---

## Evaluation & quality checks

```bash
python evaluation/evaluate.py
```

- Prints routing pass/fail and writes **`evaluation/report.txt`**.
- Domain-specific scenarios live under **`evaluation/eval_data/`** (folders such as `SIM`, `BILL`, `NETWORK`, …), each with a `test_cases.json`.

To treat “RAG expected but Ollama offline” as non-fatal in automated runs:

```bash
BTL_SKIP_RAG_STRICT_EVAL=1 python evaluation/evaluate.py
```

---

## Limitations & responsible use

- Answers depend on **indexed content** and the **local model** — they may be incomplete or outdated; users should verify critical facts on **bt.bt** or via **1300**.
- This assistant is a **support aid**, not a binding policy or contract.
- Logged data may contain personal messages; protect **`database/conversations.db`** like any customer data.

---

## Repository layout

```
bt_chatbot/
├── config/                 # Central settings (env / .env)
├── data/raw/               # Source files for indexing
│   ├── json/  excel/  csv/  pdf/
├── database/               # chroma_db/ (index), conversations.db (logs)
├── pipeline/ingest.py      # Build / refresh the vector index
├── chatbot/                # RAG chain + orchestration
├── api/app.py              # FastAPI service
├── frontend/index.html     # Chat UI (open in browser)
├── evaluation/
│   ├── evaluate.py
│   ├── eval_data/          # Per-domain test cases
│   └── report.txt          # Generated report
├── requirements.txt
├── .env.example
└── README.md
```

---

## Document history

This README is intended for **stakeholders** (management, operations, and engineering). For the original technical specification, see the project **`prompt.md`** in the parent workspace where applicable.
