#streamlit_app.py
import streamlit as st
from rag_pipeline import answer_question
from runtime_settings import settings
from cache_manager import clear_cache
from conversation_memory import memory_summary, recent_turns
from graph_visualizer import render_graph_from_hits

st.set_page_config(page_title="Syracuse Hybrid RAG", layout="wide")
st.title("Syracuse Research Assistant — Hybrid RAG")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Model & Database")

    settings.active_mode = st.radio("Chroma Database:", ["full", "abstracts"], index=0)
    settings.llm_model = st.selectbox("LLM Model:", ["llama-3.2-1b", "llama-3.2-3b"], index=1)
    settings.use_graph = st.checkbox("Enable Graph Expansion (Neo4j)", value=False)

    st.markdown("---")

    if st.button("🧹 Clear Cached Answers"):
        clear_cache()
        st.success("✅ Answer cache cleared.")

    if st.button("♻️ Reset Long-Term Memory Summary"):
        memory_summary = ""
        recent_turns.clear()
        st.success("✅ Conversation memory cleared.")

    if st.button("🔄 Reset Conversation (Chat Log)"):
        st.session_state.messages = []
        st.success("✅ Chat history cleared.")


# ---------------- Chat State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Show Previous Messages ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- Input ----------------
prompt = st.chat_input("Ask about Syracuse research…")


# ---------------- On New Message ----------------
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            out = answer_question(prompt)
            st.markdown(out["answer"])

            # ✅ Sources restored
            if out.get("sources"):
                with st.expander("📚 Sources (Newest → Oldest)"):
                    for s in out["sources"]:
                        s = s.strip()
                        if "doi.org" in s.lower():
                            st.markdown(f"- [{s}]({s})")
                        else:
                            st.markdown(f"- {s}")

            # ✅ Optional graph restored
            if settings.use_graph and out.get("graph_hits"):
                st.markdown("### 🌐 Graph — Researcher ↔ Papers ↔ Authors")
                graph_output, cypher_query, params = render_graph_from_hits(out["graph_hits"], height=600)
                st.write(graph_output)

                with st.expander("Graph Query Used"):
                    st.code(cypher_query.strip(), language="cypher")
                    st.json(params)

    # Store assistant reply
    st.session_state.messages.append({"role": "assistant", "content": out["answer"]})
