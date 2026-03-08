# streamlit_app.py
import os
import html
import uuid
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

_LOCAL_EMBED_PATH = r"C:\codes\models\all-MiniLM-L6-v2"
if os.path.isdir(_LOCAL_EMBED_PATH):
    os.environ.setdefault("EMBED_MODEL", _LOCAL_EMBED_PATH)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", r"C:\codes\hf_cache")

# Force embeddings to CPU so they don't contend with the LLM for VRAM
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")

# Prevent tokenizer parallelism deadlock on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Raise LLM timeout to handle complex/long queries
os.environ.setdefault("RAG_LLM_TIMEOUT_S", "300")

from rag_pipeline import answer_question, sanitize_answer_for_display
from runtime_settings import settings
from rag_engine import get_global_manager
from conversation_memory import hard_reset_memory, clear_qa_cache
from cache_manager import clear_cache as clear_rag_cache

import psutil
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _esc(value: object) -> str:
    return html.escape(str(value or ""))


def _esc_answer(value: object) -> str:
    """Sanitise an LLM answer for display.

    We no longer html.escape() here because Streamlit's st.markdown with
    unsafe_allow_html=False already prevents XSS.  HTML-escaping was causing
    literal '&amp;' artefacts and preventing markdown rendering.
    """
    return sanitize_answer_for_display(str(value or ""))


def _ram_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "used_gb": vm.used / (1024 ** 3),
        "total_gb": vm.total / (1024 ** 3),
        "pct": vm.percent,
    }


def _vram_stats() -> dict | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        used_b = total_b - free_b
        return {
            "used_gb": used_b / (1024 ** 3),
            "total_gb": total_b / (1024 ** 3),
            "pct": (used_b / total_b * 100) if total_b else 0.0,
        }
    except Exception:
        return None


def _session_metrics(user_key: str) -> dict:
    state = ENGINE_MANAGER.store.load(user_key)
    return {
        "session_id": user_key,
        "turn_count": len(state.get("turns", []) or []),
        "summary_len": len(state.get("rolling_summary", "") or ""),
        "retrieval_confidence": str(state.get("retrieval_confidence", "") or ""),
    }


def _render_graph(g: dict, graph_key: str) -> None:
    nodes_raw = g.get("nodes", []) or []
    edges_raw = g.get("edges", []) or []
    if not nodes_raw or not edges_raw:
        return

    # ── Toggle state ──────────────────────────────────────────────────────────
    toggle_key = f"graph_open_{graph_key}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False  # hidden by default

    if st.button(
        "▼ Hide Graph" if st.session_state[toggle_key] else "▶ Show Graph",
        key=f"graph_btn_{graph_key}",
    ):
        st.session_state[toggle_key] = not st.session_state[toggle_key]

    if not st.session_state[toggle_key]:
        return

    # ── Node styling by type ──────────────────────────────────────────────────
    TYPE_CONFIG = {
        "paper":      {"size": 18, "color": "#4A90D9"},
        "researcher": {"size": 28, "color": "#E87B3A"},
        "topic":      {"size": 20, "color": "#5BAD72"},
        "author":     {"size": 14, "color": "#9B6BBE"},
    }
    DEFAULT_CFG = {"size": 16, "color": "#888888"}

    def _truncate(label: str, max_len: int = 48) -> str:
        return label if len(label) <= max_len else label[:max_len - 1].rstrip() + "…"

    nodes, edges = [], []

    for n in nodes_raw:
        nid   = str(n.get("id", "")).strip()
        label = _truncate(str(n.get("label", "")).strip() or nid)
        ntype = str(n.get("type", "")).strip().lower()
        cfg   = TYPE_CONFIG.get(ntype, DEFAULT_CFG)
        nodes.append(Node(
            id=nid,
            label=label,
            size=cfg["size"],
            color=cfg["color"],
            font={
                "size": 11,
                "color": "#1a1a1a",
                "background": "rgba(255,255,255,0.85)",
                "strokeWidth": 2,
                "strokeColor": "#ffffff",
            },
            borderWidth=2,
            borderWidthSelected=3,
        ))

    for e in edges_raw:
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if src and tgt:
            edges.append(Edge(
                source=src,
                target=tgt,
                label=str(e.get("type", "")),
                color={"color": "#aaaaaa", "opacity": 0.7},
                width=1.5,
                font={"size": 9, "color": "#555555",
                      "background": "rgba(255,255,255,0.75)"},
                smooth={"type": "curvedCW", "roundness": 0.1},
            ))

    config = Config(
        width="100%",
        height=520,
        directed=True,
        physics=True,
        hierarchical=False,
        fit=True,
        stabilization={
            "enabled": True,
            "iterations": 300,
            "updateInterval": 25,
            "fit": True,
        },
        solver="forceAtlas2Based",
        forceAtlas2Based={
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 120,
            "springConstant": 0.08,
            "damping": 0.9,
            "avoidOverlap": 0.8,
        },
        zoomView=True,
        dragNodes=True,
        dragView=True,
        navigationButtons=True,
        keyboard=True,
    )

    # Render directly in page — NOT inside st.expander (causes zero-height mount)
    agraph(nodes=nodes, edges=edges, config=config)


# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Syracuse Research Assistant", layout="wide")
st.title("Syracuse Research Assistant")

if "user_key" not in st.session_state:
    st.session_state["user_key"] = str(uuid.uuid4())
USER_KEY = st.session_state["user_key"]

if "engine_manager" not in st.session_state:
    st.session_state["engine_manager"] = get_global_manager()
ENGINE_MANAGER = st.session_state["engine_manager"]

# Always-on settings
settings.debug_rag = True
settings.use_graph = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("System Memory")
    ram_caption_placeholder  = st.empty()
    ram_bar_placeholder      = st.empty()
    vram_caption_placeholder = st.empty()
    vram_bar_placeholder     = st.empty()

    st.divider()
    st.subheader("Session")

    if st.button("Clear Cache"):
        _safe_call(clear_qa_cache, USER_KEY)
        _safe_call(clear_rag_cache)
        st.success("Cache cleared.")

    if st.button("Reset Memory"):
        _safe_call(hard_reset_memory, USER_KEY)
        st.success("Memory cleared.")

    if st.button("Restart Conversation"):
        st.session_state["messages"] = []
        _safe_call(hard_reset_memory, USER_KEY)
        st.success("Conversation restarted.")

    st.divider()
    st.subheader("Session Diagnostics")
    diagnostics_placeholder = st.empty()


def _refresh_sidebar(user_key: str) -> None:
    """Write current stats into sidebar placeholders — no rerun needed."""
    ram = _ram_stats()
    ram_caption_placeholder.caption(
        f"**RAM** {ram['used_gb']:.1f} / {ram['total_gb']:.1f} GB  ({ram['pct']:.0f}%)"
    )
    ram_bar_placeholder.progress(int(ram["pct"]))

    vram = _vram_stats()
    if vram:
        vram_caption_placeholder.caption(
            f"**VRAM** {vram['used_gb']:.1f} / {vram['total_gb']:.1f} GB  ({vram['pct']:.0f}%)"
        )
        vram_bar_placeholder.progress(int(vram["pct"]))
    else:
        vram_caption_placeholder.caption("**VRAM** — GPU not available")
        vram_bar_placeholder.empty()

    metrics = _session_metrics(user_key)
    diagnostics_placeholder.json({
        "session_id":           metrics["session_id"],
        "turn_count":           metrics["turn_count"],
        "summary_len":          metrics["summary_len"],
        "retrieval_confidence": metrics["retrieval_confidence"],
    })


# Initial population on every page load
_refresh_sidebar(USER_KEY)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for idx, msg in enumerate(st.session_state["messages"]):
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant":
            content = msg.get("content", "")
            st.markdown(_esc_answer(content), unsafe_allow_html=False)

            timing_caption = msg.get("timing_caption", "")
            if timing_caption:
                st.caption(timing_caption)

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"**[{i}]** {s}", unsafe_allow_html=False)

            graph_error = msg.get("graph_error", "")
            if graph_error:
                st.warning(graph_error)
            _render_graph(msg.get("graph", {}) or {}, graph_key=f"history_{idx}")
        else:
            st.write(_esc(msg.get("content", "")))

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask about Syracuse research...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(_esc(prompt))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            out = answer_question(
                prompt,
                user_key=USER_KEY,
                use_graph=True,
                stateless=False,
            )

        answer_text = (out.get("answer") or "").strip()
        if answer_text:
            st.markdown(_esc_answer(answer_text), unsafe_allow_html=False)
        else:
            answer_text = "I could not generate an answer, but the retrieved sources are shown below."
            st.markdown(answer_text, unsafe_allow_html=False)

        llm_calls = out.get("llm_calls", {}) if isinstance(out.get("llm_calls"), dict) else {}
        timing    = out.get("timing_ms", {})  if isinstance(out.get("timing_ms"),  dict) else {}
        timing_caption = (
            f"LLM calls — answer: {llm_calls.get('answer_llm_calls', 0)} | "
            f"utility: {llm_calls.get('utility_llm_calls', 0)} | "
            f"total: {timing.get('total_ms', 0):.0f} ms"
        )
        st.caption(timing_caption)

        sources = list(out.get("sources", []) or [])
        if sources:
            with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**[{i}]** {s}", unsafe_allow_html=False)

        graph_error = str(out.get("graph_error", "") or "")
        if graph_error:
            st.warning(graph_error)
        g = dict(out.get("graph_graph", {}) or {})
        new_msg_idx = len(st.session_state["messages"])
        _render_graph(g, graph_key=f"history_{new_msg_idx}")

    # Persist turn
    st.session_state["messages"].append({
        "role":           "assistant",
        "content":        answer_text,
        "sources":        sources,
        "graph":          g,
        "graph_error":    graph_error,
        "timing_caption": timing_caption,
    })

    # Refresh sidebar in-place — no rerun, graph stays mounted
    _refresh_sidebar(USER_KEY)