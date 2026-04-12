import warnings
import logging
warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import os
import html
import uuid
import gc
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

_LOCAL_EMBED_PATH = os.environ.get("LOCAL_EMBED_PATH", "")
if _LOCAL_EMBED_PATH and os.path.isdir(_LOCAL_EMBED_PATH):
    os.environ.setdefault("EMBED_MODEL", _LOCAL_EMBED_PATH)
_HF_HOME = os.environ.get("HF_HOME_OVERRIDE", "")
if _HF_HOME and os.path.isdir(_HF_HOME):
    os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault("RAG_LLM_TIMEOUT_S", "300")

from rag_pipeline import answer_question, sanitize_answer_for_display
from runtime_settings import settings
from rag_engine import get_global_manager
from conversation_memory import hard_reset_memory

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

with st.sidebar:
    st.subheader("System Memory")
    ram_ph = st.empty()
    ram_bar_ph = st.empty()
    vram_ph = st.empty()
    vram_bar_ph = st.empty()

    st.divider()
    st.subheader("Session")

    if st.button("Reset Memory"):
        _safe_call(hard_reset_memory, USER_KEY)
        st.success("Memory cleared.")

    if st.button("Restart Conversation"):
        st.session_state["messages"] = []
        _safe_call(hard_reset_memory, USER_KEY)
        st.session_state["user_key"] = str(uuid.uuid4())
        st.success("Conversation restarted.")
        st.rerun()

    st.divider()
    st.subheader("Dataset")

    _db_labels = ENGINE_MANAGER.dbm.display_labels()
    _label_to_key = {v: k for k, v in _db_labels.items()}
    _options = list(_db_labels.values())

    if "active_dataset" not in st.session_state:
        st.session_state["active_dataset"] = _db_labels.get(ENGINE_MANAGER.active_mode, _options[0])

    selected_label = st.selectbox(
        "Choose dataset",
        options=_options,
        index=_options.index(st.session_state["active_dataset"]) if st.session_state["active_dataset"] in _options else 0,
        key="dataset_selector",
    )

    if selected_label != st.session_state["active_dataset"]:
        new_mode = _label_to_key[selected_label]
        ENGINE_MANAGER.switch_mode(new_mode)
        settings.active_mode = new_mode
        ENGINE_MANAGER.papers_vs_cache.pop(new_mode, None)
        st.session_state["active_dataset"] = selected_label
        st.success(f"Switched to **{selected_label}**")
        st.rerun()
    else:
        current_key = _label_to_key.get(selected_label, "full")
        if ENGINE_MANAGER.active_mode != current_key:
            ENGINE_MANAGER.switch_mode(current_key)
            settings.active_mode = current_key

    st.divider()
    st.subheader("Model")

    _MODEL_OPTIONS = {
        "LLaMA 3.2 3B": "llama-3.2-3b",
        "LLaMA 3.1 8B (8-bit)": "llama-3.1-8b",
        "Gemma 3 12B (8-bit)": "gemma-3-12b",
        "Qwen 2.5 14B (8-bit)": "qwen-2.5-14b",
        "GPT-OSS 20B (8-bit)": "gpt-oss-20b",
    }
    _model_labels = list(_MODEL_OPTIONS.keys())
    _model_keys = list(_MODEL_OPTIONS.values())

    if "active_model" not in st.session_state:
        current_key = getattr(settings, "answer_model_key", "llama-3.2-3b")
        st.session_state["active_model"] = next(
            (lbl for lbl, k in _MODEL_OPTIONS.items() if k == current_key),
            _model_labels[0],
        )

    selected_model_label = st.selectbox(
        "Answer model",
        options=_model_labels,
        index=_model_labels.index(st.session_state["active_model"]) if st.session_state["active_model"] in _model_labels else 0,
        key="model_selector",
    )

    if selected_model_label != st.session_state["active_model"]:
        new_model_key = _MODEL_OPTIONS[selected_model_label]
        settings.answer_model_key = new_model_key
        settings.llm_model = new_model_key
        st.session_state["active_model"] = selected_model_label

        with st.spinner(f"Loading **{selected_model_label}**..."):
            try:
                ENGINE_MANAGER.switch_answer_model(new_model_key)
                st.success(f"**{selected_model_label}** loaded and ready.")
            except Exception as e:
                st.error(f"Failed to load {selected_model_label}: {e}")
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
    active_cfg = ENGINE_MANAGER.dbm.get_active_config()
    _attn_info = ""
    _rt = getattr(ENGINE_MANAGER, "answer_runtime", None)
    if _rt and hasattr(_rt, "model"):
        _attn_info = str(getattr(getattr(_rt.model, "config", None), "_attn_implementation", "unknown"))
    _gpu_name = ""
    if torch.cuda.is_available():
        try: _gpu_name = torch.cuda.get_device_name(0)
        except Exception: pass
    diag_ph.json({
        "session_id": user_key,
        "active_dataset": ENGINE_MANAGER.active_mode,
        "chroma_collection": active_cfg.collection if active_cfg else "",
        "neo4j_db": ENGINE_MANAGER.dbm.get_active_neo4j_db(),
        "answer_model": getattr(settings, "answer_model_key", ""),
        "model_loaded": ENGINE_MANAGER.active_answer_model_key or "(not loaded)",
        "attention": _attn_info or "(unknown)",
        "gpu": _gpu_name or "(none)",
        "turn_count": len(state.get("turns", []) or []),
        "summary_len": len(state.get("rolling_summary", "") or ""),
        "retrieval_confidence": str(state.get("retrieval_confidence", "") or ""),
    })

_refresh_sidebar(USER_KEY)

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