"""Layer 2: Chroma retrieval + Ollama with per-session memory (manual RAG for query control)."""

from __future__ import annotations

import re

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from config.settings import get_settings

_s = get_settings()
CHROMA_DIR = _s.chroma_persist_dir
COLLECTION_NAME = _s.chroma_collection_name
EMBED_MODEL = _s.embedding_model
OLLAMA_MODEL = _s.ollama_model

RAG_PROMPT = """You are the BT Virtual Assistant for Bhutan Telecom Limited (BTL).
Speak in a warm, natural, conversational tone—like a helpful human agent—but stay grounded in facts.

Use ONLY the information in the context below. The context may include website topics, structured
service notes, ticket summaries, or PDF/document excerpts.

Guidelines:
- Answer clearly and helpfully.
- When explaining procedures (recharge, USSD, registration, how-to questions), use numbered steps (1., 2., 3.)
  and put each step on its own line with a newline after each step. Add a blank line between sections if helpful.
  Do not cram several steps into one long sentence or a single line.
- For simple facts, one or two short paragraphs are fine.
- If the context supports it, mention practical next steps (e.g. dial a short code, visit bt.bt, call 1600/1300)
  only when those appear in the context.
- If the context names people in leadership or roles (CEO, Chief Executive Officer, directors, etc.), state
  those names and roles accurately from the context.
- If the context truly does not contain enough to answer, say honestly that you are not finding that detail
  in the materials you have, and suggest contacting BT (1300, 1600 where relevant, or bt.bt)—do not invent facts.
- Do not make up names, numbers, policies, or URLs that are not in the context.
- Never use bracket placeholders like [insert ...] or [phone number]—omit details you do not have.

Customer-facing answers (very important):
- Customers cannot see your internal PDFs or indexed documents. Never tell them to look at "page X",
  "Customer Charter page …", or any internal document page number—they do not have that file.
- For office locations, hours, or directions: give **town, area, address, phone, or opening hours** only
  when those details appear in the context. If the context only says a centre exists in a place without
  a street address, say what you can (e.g. town and hours) and suggest **calling 1600** or **bt.bt** for
  the exact location or directions.
- Answer the user’s actual question; do not tack on unrelated products or services from the same
  passage (e.g. premium numbers) unless they help answer that question.

Context:
{context}

Customer message: {question}

Your reply:
"""

# Leadership / governance questions: "CEO" often mismatches embeddings vs "Chief Executive Officer" in text.
_LEADERSHIP_Q = re.compile(
    r"\b(ceo|c\.e\.o\.|chief executive|chairman|management team|board of directors|"
    r"who runs btl|who is the (ceo|head)|leadership|managing director)\b",
    re.IGNORECASE,
)


def _expand_retrieval_query(user_message: str) -> str:
    q = user_message.strip()
    if _LEADERSHIP_Q.search(q):
        return (
            f"{q} Bhutan Telecom Limited BTL leadership Management page "
            "Chief Executive Officer directors board"
        )
    return q


def _dedupe_docs(docs: list, cap: int) -> list:
    seen: set[str] = set()
    out = []
    for d in docs:
        key = (d.page_content or "")[:160]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= cap:
            break
    return out


_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None
_llm: OllamaLLM | None = None

_session_memories: dict[str, ConversationBufferMemory] = {}


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if not CHROMA_DIR.is_dir():
            raise FileNotFoundError(
                f"Chroma DB not found at {CHROMA_DIR}. Run: python pipeline/ingest.py"
            )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=_get_embeddings(),
        )
    return _vectorstore


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=OLLAMA_MODEL, temperature=_s.ollama_temperature)
    return _llm


def _memory_for_session(session_id: str) -> ConversationBufferMemory:
    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
        )
    return _session_memories[session_id]


def _trim_memory(memory: ConversationBufferMemory, max_exchanges: int = 5) -> None:
    cm = getattr(memory, "chat_memory", None)
    msgs = getattr(cm, "messages", None) if cm is not None else None
    if msgs is not None and len(msgs) > max_exchanges * 2:
        cm.messages = msgs[-(max_exchanges * 2) :]


def _confidence_from_sources(n: int) -> float:
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.7
    if n == 1:
        return 0.4
    return 0.0


def get_rag_response(user_message: str, session_id: str) -> dict:
    """
    Retrieve with an expanded query (fixes CEO vs 'Chief Executive Officer' mismatch), then answer with Ollama.
    Chat history is passed only to the LLM, not to the retriever, so retrieval stays on-topic.
    """
    memory = _memory_for_session(session_id)
    hist = memory.load_memory_variables({}).get("chat_history") or ""

    vs = _get_vectorstore()
    k = _s.rag_retriever_k
    fk = max(k, _s.rag_mmr_fetch_k)
    rq = _expand_retrieval_query(user_message)

    try:
        if _s.rag_use_mmr:
            docs = vs.max_marginal_relevance_search(rq, k=k, fetch_k=fk)
        else:
            docs = vs.similarity_search(rq, k=k)

        if _LEADERSHIP_Q.search(user_message):
            extra = vs.similarity_search(
                "Bhutan Telecom Chief Executive Officer Management Jamyang leadership BTL CEO director",
                k=6,
            )
            docs = _dedupe_docs(list(docs) + list(extra), cap=max(k + 4, 10))

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        q_for_llm = user_message.strip()
        if hist:
            q_for_llm = (
                f"Prior conversation (for context only):\n{hist}\n\n"
                f"Current customer message: {user_message}"
            )

        prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["context", "question"])
        answer = (_get_llm().invoke(prompt.format(context=context, question=q_for_llm)) or "").strip()
    except Exception as e:
        return {
            "response": "",
            "source_services": [],
            "confidence": 0.0,
            "error": str(e),
        }

    services: list[str] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        svc = meta.get("service")
        if svc:
            services.append(str(svc))
    conf = _confidence_from_sources(len(docs))

    memory.save_context({"input": user_message}, {"output": answer})
    _trim_memory(memory, max_exchanges=5)

    return {
        "response": answer,
        "source_services": services,
        "confidence": conf,
    }
