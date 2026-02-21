# streamlit_app.py
import os
import html
import uuid
import streamlit as st

from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------------------------------------------------------
# Embedding model path: read from environment variables instead of hardcoding
# platform-specific paths. Set EMBED_MODEL_PATH in your .env or shell to point
# to a local model directory; if unset the app falls back to HuggingFace Hub
# download behaviour.
# ---------------------------------------------------------------------------
_LOCAL_EMBED_PATH = os.getenv("EMBED_MODEL_PATH", "").strip()
if _LOCAL_EMBED_PATH and os.path.isdir(_LOCAL_EMBED_PATH):
    os.environ.setdefault("EMBED_MODEL", _LOCAL_EMBED_PATH)

os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))

_default_hf_home = os.getenv("HF_HOME") or os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface"
)
os.environ.setdefault("HF_HOME", _default_hf_home)

from rag_pipeline import answer_question
from runtime_settings import settings
from rag_engine import get_global_manager

from conversation_memory import hard_reset_memory, clear_qa_cache
from cache_manager import clear_cache as clear_rag_cache


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _sanitize_for_display(value: object) -> str:
    return html.escape(str(value or ""))


st.set_page_config(page_title="Syracuse Research Assistant", layout="wide")
st.title("Syracuse Research Assistant")

if "user_key" not in st.session_state:
    st.session_state["user_key"] = str(uuid.uuid4())
USER_KEY = st.session_state["user_key"]

if "engine_manager" not in st.session_state:
    st.session_state["engine_manager"] = get_global_manager()
ENGINE_MANAGER = st.session_state["engine_manager"]


def _session_metrics(user_key: str):
    store = ENGINE_MANAGER.store
    state = store.load(user_key)
    turns = state.get("turns", []) or []
    summary = state.get("rolling_summary", "") or ""
    return {
        "session_id": user_key,
        "turn_count": len(turns),
        "summary_len": len(summary),
        "retrieval_confidence": str(state.get("retrieval_confidence", "") or ""),
    }

with st.sidebar:
    st.subheader("Settings")

    _dbm = ENGINE_MANAGER.dbm
    _available_modes = _dbm.list_configs()
    if not _available_modes:
        st.error("No retrieval modes are configured.")
        st.stop()
    _requested_mode = str(getattr(settings, "active_mode", "") or "")
    _default_mode = _dbm.resolve_mode(_requested_mode)
    _default_index = _available_modes.index(_default_mode) if _default_mode in _available_modes else 0
    settings.active_mode = st.selectbox(
        "Chroma Database",
        _available_modes,
        index=_default_index,
    )

    selected_answer_model = st.selectbox(
        "Answer Model",
        ["llama-3.2-1b", "llama-3.2-3b"],
        index=1 if str(getattr(settings, "answer_model_key", "llama-3.2-3b")).strip().lower() == "llama-3.2-3b" else 0,
    )
    settings.answer_model_key = selected_answer_model
    settings.llm_model = selected_answer_model

    utility_path = str(getattr(settings, "llama_1b_path", "") or "").strip() or "(unset)"
    st.caption(f"Utility model path (LLAMA_1B): {utility_path}")

    settings.use_graph = st.checkbox("Enable Graph Retrieval", value=False)
    stateless_turn = st.checkbox("Stateless Turn", value=False)
    settings.debug_rag = st.checkbox("Debug RAG in terminal", value=False)

    st.subheader("Memory and Cache")

    if st.button("Clear Cache"):
        _safe_call(clear_qa_cache, USER_KEY)
        _safe_call(clear_rag_cache)
        st.success("Cache cleared for this session.")

    if st.button("Reset Memory"):
        _safe_call(hard_reset_memory, USER_KEY)
        st.success("Persistent memory cleared for this session.")

    if st.button("Restart Conversation"):
        st.session_state["messages"] = []
        _safe_call(hard_reset_memory, USER_KEY)
        st.success("Conversation restarted.")

    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_graph_query = st.checkbox("Show graph data", value=False)
    show_session_diag = st.checkbox("Show session diagnostics", value=True)

# Auto clear QA cache when settings change
sig = f"{settings.active_mode}|{settings.llm_model}"
if "settings_sig" not in st.session_state:
    st.session_state["settings_sig"] = sig
else:
    if st.session_state["settings_sig"] != sig:
        _safe_call(clear_qa_cache, USER_KEY)
        st.session_state["settings_sig"] = sig

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(_sanitize_for_display(msg["content"]))

prompt = st.chat_input("Ask about Syracuse research...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(_sanitize_for_display(prompt))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            out = answer_question(
                prompt,
                user_key=USER_KEY,
                use_graph=settings.use_graph,
                stateless=stateless_turn,
            )

            answer_text = (out.get("answer") or "").strip()
            if answer_text:
                st.write(_sanitize_for_display(answer_text))
            else:
                st.write("I could not generate an answer, but the retrieved sources are shown below.")

            llm_calls = out.get("llm_calls", {}) if isinstance(out.get("llm_calls"), dict) else {}
            answer_calls = int(llm_calls.get("answer_llm_calls", 0) or 0)
            utility_calls = int(llm_calls.get("utility_llm_calls", 0) or 0)
            st.caption(f"LLM calls this turn: answer={answer_calls} | utility={utility_calls}")

            if show_sources and out.get("sources"):
                with st.expander("Retrieved Sources", expanded=False):
                    for s in out["sources"]:
                        st.write(_sanitize_for_display(s))

            if settings.use_graph:
                if out.get("graph_error"):
                    st.warning(out["graph_error"])

                g = out.get("graph_graph", {}) or {}
                nodes_raw = g.get("nodes", []) or []
                edges_raw = g.get("edges", []) or []

                if nodes_raw and edges_raw:
                    nodes = []
                    edges = []

                    for n in nodes_raw:
                        nid = str(n.get("id", "")).strip()
                        label = str(n.get("label", "")).strip() or nid
                        ntype = str(n.get("type", "")).strip() or "node"

                        size = 18
                        if ntype == "paper":
                            size = 22
                        elif ntype == "researcher":
                            size = 26
                        elif ntype == "topic":
                            size = 20
                        elif ntype == "author":
                            size = 16

                        nodes.append(Node(id=nid, label=label, size=size))

                    for e in edges_raw:
                        src = str(e.get("source", "")).strip()
                        tgt = str(e.get("target", "")).strip()
                        etype = str(e.get("type", "")).strip() or ""
                        if src and tgt:
                            edges.append(Edge(source=src, target=tgt, label=etype))

                    cfg = Config(
                        width="100%",
                        height=int(g.get("height", 650)),
                        directed=True,
                        physics=True,
                        hierarchical=False,
                    )

                    with st.expander("Graph", expanded=True):
                        agraph(nodes=nodes, edges=edges, config=cfg)

                    if show_graph_query:
                        with st.expander("Graph Data", expanded=False):
                            st.json(g)
                else:
                    st.caption("No graph data available for visualization.")

            if show_session_diag:
                with st.expander("Session Diagnostics", expanded=False):
                    metrics = _session_metrics(USER_KEY)
                    timing = out.get("timing_ms", {}) if isinstance(out.get("timing_ms"), dict) else {}
                    st.json(
                        {
                            "session_id": metrics.get("session_id", ""),
                            "turn_count": metrics.get("turn_count", 0),
                            "summary_len": metrics.get("summary_len", 0),
                            "retrieval_confidence": metrics.get("retrieval_confidence", ""),
                            "answer_llm_calls": answer_calls,
                            "utility_llm_calls": utility_calls,
                            "timing_ms": timing,
                        }
                    )

    st.session_state["messages"].append({"role": "assistant", "content": answer_text})