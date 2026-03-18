# streamlit_app.py
import os
import html
import uuid
import gc
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

_LOCAL_EMBED_PATH = os.environ.get("LOCAL_EMBED_PATH", "")
if _LOCAL_EMBED_PATH and os.path.isdir(_LOCAL_EMBED_PATH):
    os.environ.setdefault("EMBED_MODEL", _LOCAL_EMBED_PATH)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_HF_HOME = os.environ.get("HF_HOME_OVERRIDE", "")
if _HF_HOME and os.path.isdir(_HF_HOME):
    os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAG_LLM_TIMEOUT_S", "300")

from rag_pipeline import answer_question, sanitize_answer_for_display
from runtime_settings import settings
from rag_engine import get_global_manager
from conversation_memory import hard_reset_memory, clear_qa_cache
from cache_manager import clear_cache as clear_rag_cache

import psutil
import torch


def _safe_call(fn, *args, **kwargs):
    try: return fn(*args, **kwargs)
    except Exception: return None

def _esc(value) -> str:
    return html.escape(str(value or ""))

def _esc_answer(value) -> str:
    return sanitize_answer_for_display(str(value or ""))


def _render_graph(g: dict, graph_key: str) -> None:
    nodes_raw = g.get("nodes", []) or []
    edges_raw = g.get("edges", []) or []
    if not nodes_raw or not edges_raw:
        return

    toggle_key = f"graph_open_{graph_key}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False

    show = st.toggle("Show Graph", value=st.session_state[toggle_key],
                      key=f"graph_toggle_{graph_key}")
    st.session_state[toggle_key] = show
    if not show:
        return

    TYPE_CONFIG = {
        "paper": (18, "#4A90D9"), "researcher": (28, "#E87B3A"),
        "topic": (20, "#5BAD72"), "author": (14, "#9B6BBE"),
    }
    FONT = {"size": 11, "color": "#1a1a1a", "background": "rgba(255,255,255,0.85)",
            "strokeWidth": 2, "strokeColor": "#ffffff"}

    nodes = []
    for n in nodes_raw:
        nid = str(n.get("id", "")).strip()
        label = str(n.get("label", "")).strip() or nid
        if len(label) > 48: label = label[:47].rstrip() + "…"
        sz, color = TYPE_CONFIG.get(str(n.get("type", "")).strip().lower(), (16, "#888888"))
        nodes.append(Node(id=nid, label=label, size=sz, color=color, font=FONT,
                          borderWidth=2, borderWidthSelected=3))

    edges = [Edge(source=str(e.get("source", "")).strip(),
                  target=str(e.get("target", "")).strip(),
                  label=str(e.get("type", "")),
                  color={"color": "#aaaaaa", "opacity": 0.7}, width=1.5,
                  font={"size": 9, "color": "#555555", "background": "rgba(255,255,255,0.75)"},
                  smooth={"type": "curvedCW", "roundness": 0.1})
             for e in edges_raw if str(e.get("source", "")).strip() and str(e.get("target", "")).strip()]

    config = Config(
        width="100%", height=520, directed=True, physics=True, hierarchical=False, fit=True,
        stabilization={"enabled": True, "iterations": 300, "updateInterval": 25, "fit": True},
        solver="forceAtlas2Based",
        forceAtlas2Based={"gravitationalConstant": -50, "centralGravity": 0.01,
                          "springLength": 120, "springConstant": 0.08,
                          "damping": 0.9, "avoidOverlap": 0.8},
        zoomView=True, dragNodes=True, dragView=True, navigationButtons=True, keyboard=True,
    )
    with st.container():
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

settings.debug_rag = True
settings.use_graph = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("System Memory")
    ram_ph = st.empty()
    ram_bar_ph = st.empty()
    vram_ph = st.empty()
    vram_bar_ph = st.empty()

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
        _safe_call(clear_qa_cache, USER_KEY)
        _safe_call(clear_rag_cache)
        st.session_state["user_key"] = str(uuid.uuid4())
        try:
            mgr = ENGINE_MANAGER
            for rt_attr in ("answer_runtime", "utility_runtime"):
                rt = getattr(mgr, rt_attr, None)
                if rt is not None:
                    rt.close()
                    setattr(mgr, rt_attr, None)
                    setattr(mgr, f"active_{rt_attr.replace('_runtime', '')}_model_key", "")
            if mgr.utility_worker is not None:
                mgr.utility_worker.stop()
                mgr.utility_worker = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        st.success("Conversation restarted — models will reload on next question.")
        st.rerun()

    st.divider()
    st.subheader("Session Diagnostics")
    diag_ph = st.empty()


def _refresh_sidebar(user_key: str) -> None:
    vm = psutil.virtual_memory()
    ram_ph.caption(f"**RAM** {vm.used / 1e9:.1f} / {vm.total / 1e9:.1f} GB  ({vm.percent:.0f}%)")
    ram_bar_ph.progress(int(vm.percent))

    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            used = total_b - free_b
            pct = used / total_b * 100 if total_b else 0
            vram_ph.caption(f"**VRAM** {used / 1e9:.1f} / {total_b / 1e9:.1f} GB  ({pct:.0f}%)")
            vram_bar_ph.progress(int(pct))
        except Exception:
            vram_ph.caption("**VRAM** — error reading GPU info")
            vram_bar_ph.empty()
    else:
        vram_ph.caption("**VRAM** — GPU not available")
        vram_bar_ph.empty()

    state = ENGINE_MANAGER.store.load(user_key)
    diag_ph.json({
        "session_id": user_key,
        "turn_count": len(state.get("turns", []) or []),
        "summary_len": len(state.get("rolling_summary", "") or ""),
        "retrieval_confidence": str(state.get("retrieval_confidence", "") or ""),
    })


_refresh_sidebar(USER_KEY)

# ---------------------------------------------------------------------------
# Chat history & input
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(_esc_answer(msg.get("content", "")), unsafe_allow_html=False)
            if msg.get("timing_caption"):
                st.caption(msg["timing_caption"])
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"[{i}] {s}", unsafe_allow_html=False)
            if msg.get("graph_error"):
                st.warning(msg["graph_error"])
            _render_graph(msg.get("graph", {}) or {}, graph_key=f"history_{idx}")
        else:
            st.write(_esc(msg.get("content", "")))

prompt = st.chat_input("Ask about Syracuse research...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(_esc(prompt))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            out = answer_question(prompt, user_key=USER_KEY, use_graph=True, stateless=False)

        answer_text = (out.get("answer") or "").strip()
        if not answer_text:
            answer_text = "I could not generate an answer, but the retrieved sources are shown below."
        st.markdown(_esc_answer(answer_text), unsafe_allow_html=False)

        llm_calls = out.get("llm_calls", {}) if isinstance(out.get("llm_calls"), dict) else {}
        timing = out.get("timing_ms", {}) if isinstance(out.get("timing_ms"), dict) else {}
        timing_caption = (f"LLM calls — answer: {llm_calls.get('answer_llm_calls', 0)} | "
                          f"utility: {llm_calls.get('utility_llm_calls', 0)} | "
                          f"total: {timing.get('total_ms', 0):.0f} ms")
        st.caption(timing_caption)

        sources = list(out.get("sources", []) or [])
        if sources:
            with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"[{i}] {s}", unsafe_allow_html=False)

        graph_error = str(out.get("graph_error", "") or "")
        if graph_error:
            st.warning(graph_error)
        g = dict(out.get("graph_graph", {}) or {})
        _render_graph(g, graph_key=f"history_{len(st.session_state['messages'])}")

    st.session_state["messages"].append({
        "role": "assistant", "content": answer_text, "sources": sources,
        "graph": g, "graph_error": graph_error, "timing_caption": timing_caption,
    })
    _refresh_sidebar(USER_KEY)