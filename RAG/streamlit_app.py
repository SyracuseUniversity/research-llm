# streamlit_app.py
import uuid
import streamlit as st

from rag_pipeline import answer_question
from runtime_settings import settings
from graph_visualizer import render_graph_from_hits

from conversation_memory import hard_reset_memory, clear_qa_cache
from cache_manager import clear_cache as clear_rag_cache


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn()
        except TypeError:
            return None


st.set_page_config(page_title="Syracuse Research Assistant (Hybrid RAG)", layout="wide")
st.title("Syracuse Research Assistant (Hybrid RAG)")

if "user_key" not in st.session_state:
    st.session_state["user_key"] = str(uuid.uuid4())
USER_KEY = st.session_state["user_key"]

with st.sidebar:
    st.subheader("Settings")

    settings.active_mode = st.radio(
        "Chroma Database",
        ["full", "abstracts"],
        index=0,
    )

    settings.llm_model = st.selectbox(
        "LLM Model",
        ["llama-3.2-1b", "llama-3.2-3b"],
        index=1,
    )

    settings.use_graph = st.checkbox("Enable Graph Retrieval", value=False)

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
    show_graph_query = st.checkbox("Show graph query used", value=False)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask about Syracuse research…")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            out = answer_question(
                prompt,
                user_key=USER_KEY,
                use_graph=settings.use_graph,
            )

            if out.get("graph_error"):
                st.warning(out["graph_error"])

            answer_text = out.get("answer", "")
            st.markdown(answer_text, unsafe_allow_html=True)

            if show_sources and out.get("sources"):
                with st.expander("Retrieved Sources"):
                    for s in out["sources"]:
                        try:
                            st.markdown(s)
                        except Exception:
                            st.write(s)

            if settings.use_graph:
                hits = out.get("graph_hits", []) or []
                if hits:
                    graph_output, cypher_query, params = render_graph_from_hits(hits, height=650)
                    st.write(graph_output)
                    if show_graph_query:
                        with st.expander("Graph Query Used"):
                            if out.get("graph_cypher"):
                                st.code(out["graph_cypher"], language="cypher")
                            if out.get("graph_params"):
                                st.json(out["graph_params"])
                            if cypher_query:
                                st.code(cypher_query, language="cypher")
                            if params:
                                st.json(params)
                else:
                    st.caption("No graph data available for visualization.")

    st.session_state["messages"].append({"role": "assistant", "content": answer_text})
