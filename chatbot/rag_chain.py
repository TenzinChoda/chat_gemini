"""Layer 2: LangChain RetrievalQA over Chroma + Ollama with per-session memory."""

from __future__ import annotations

# LangChain 1.x: chains live in langchain-classic (not top-level langchain)
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
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

RAG_PROMPT = """You are a helpful customer support assistant for Bhutan Telecom Limited (BTL).
Use the following similar past customer support cases to help answer the current question.
Be concise, friendly, and specific to Bhutan Telecom services.
If you cannot find a relevant answer from the context, say you will escalate the issue.

Similar past cases:
{context}

Customer question: {question}

Helpful answer:
"""

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None
_llm: OllamaLLM | None = None
_qa_chain: RetrievalQA | None = None

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


def _get_qa_chain() -> RetrievalQA:
    global _qa_chain
    if _qa_chain is None:
        prompt = PromptTemplate(
            template=RAG_PROMPT,
            input_variables=["context", "question"],
        )
        retriever = _get_vectorstore().as_retriever(
            search_kwargs={"k": _s.rag_retriever_k}
        )
        _qa_chain = RetrievalQA.from_chain_type(
            llm=_get_llm(),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
    return _qa_chain


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
    Run RetrievalQA with optional chat-history prefix. Returns response, source service names, confidence.
    """
    memory = _memory_for_session(session_id)
    hist = memory.load_memory_variables({}).get("chat_history") or ""
    if hist:
        augmented = (
            f"Prior conversation (for context only):\n{hist}\n\n"
            f"Current customer question: {user_message}"
        )
    else:
        augmented = user_message

    qa = _get_qa_chain()
    try:
        out = qa.invoke({"query": augmented})
    except Exception as e:
        return {
            "response": "",
            "source_services": [],
            "confidence": 0.0,
            "error": str(e),
        }

    answer = (out.get("result") or "").strip()
    sources = out.get("source_documents") or []
    services: list[str] = []
    for doc in sources:
        meta = getattr(doc, "metadata", {}) or {}
        svc = meta.get("service")
        if svc:
            services.append(str(svc))
    conf = _confidence_from_sources(len(sources))

    memory.save_context({"input": user_message}, {"output": answer})
    _trim_memory(memory, max_exchanges=5)

    return {
        "response": answer,
        "source_services": services,
        "confidence": conf,
    }
