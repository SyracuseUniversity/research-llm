# # # # # #streamlit_app.py 
# # # # # import streamlit as st
# # # # # from rag_pipeline import answer_question

# # # # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # # # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # # # # with st.sidebar:
# # # # #     st.subheader("⚙️ Settings")
# # # # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # # # #     show_sources = st.checkbox("Show sources", True)

# # # # # if "messages" not in st.session_state:
# # # # #     st.session_state["messages"] = []

# # # # # for msg in st.session_state["messages"]:
# # # # #     with st.chat_message(msg["role"]):
# # # # #         st.markdown(msg["content"])

# # # # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # # # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # # # #     with st.chat_message("user"):
# # # # #         st.markdown(prompt)

# # # # #     with st.chat_message("assistant"):
# # # # #         with st.spinner("Thinking…"):
# # # # #             out = answer_question(prompt, n_ctx=n_ctx)
# # # # #             st.markdown(out["answer"])
# # # # #             if show_sources and "fused_text_blocks" in out:
# # # # #                 with st.expander("📚 Sources"):
# # # # #                     st.write("\n\n".join(out["fused_text_blocks"][:10]))

# # # # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # # # # streamlit_app.py
# # # # import streamlit as st
# # # # from rag_pipeline import answer_question

# # # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # # # with st.sidebar:
# # # #     st.subheader("⚙️ Settings")
# # # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # # #     show_sources = st.checkbox("Show sources", True)

# # # # if "messages" not in st.session_state:
# # # #     st.session_state["messages"] = []

# # # # for msg in st.session_state["messages"]:
# # # #     with st.chat_message(msg["role"]):
# # # #         st.markdown(msg["content"])

# # # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # # #     with st.chat_message("user"):
# # # #         st.markdown(prompt)

# # # #     with st.chat_message("assistant"):
# # # #         with st.spinner("Thinking…"):
# # # #             out = answer_question(prompt, n_ctx=n_ctx)
# # # #             st.markdown(out["answer"])
# # # #             if show_sources:
# # # #                 with st.expander("📚 Sources"):
# # # #                     for s in out.get("fused_text_blocks", [])[:12]:
# # # #                         st.markdown(s)

# # # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})


# # # # streamlit_app.py
# # # import streamlit as st
# # # from rag_pipeline import answer_question
# # # from graph_visualizer import render_graph

# # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # # with st.sidebar:
# # #     st.subheader("⚙️ Settings")
# # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # #     show_sources = st.checkbox("Show retrieved sources", True)
# # #     show_graph = st.checkbox("Show Graph View", False)

# # # if "messages" not in st.session_state:
# # #     st.session_state["messages"] = []

# # # for msg in st.session_state["messages"]:
# # #     with st.chat_message(msg["role"]):
# # #         st.markdown(msg["content"])

# # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # #     with st.chat_message("user"):
# # #         st.markdown(prompt)

# # #     with st.chat_message("assistant"):
# # #         with st.spinner("Retrieving and reasoning…"):
# # #             out = answer_question(prompt, n_ctx=n_ctx)

# # #             # --- Answer ---
# # #             st.markdown(out["answer"])

# # #             # --- Sources ---
# # #             if show_sources and out.get("sources"):
# # #                 with st.expander("📚 Retrieved Sources"):
# # #                     for s in out["sources"][:10]:
# # #                         st.markdown(s)

# # #             # --- Graph View ---
# # #             if show_graph and out.get("graph_hits"):
# # #                 st.markdown("### 📊 Graph View — Researcher ↔ Papers ↔ Relations")
# # #                 render_graph(out["graph_hits"])

# # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # # streamlit_app.py
# # import streamlit as st
# # from rag_pipeline import answer_question
# # from graph_visualizer import render_graph

# # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # st.title("Syracuse Research Assistant (Hybrid RAG)")

# # with st.sidebar:
# #     st.subheader("Settings")
# #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# #     show_sources = st.checkbox("Show retrieved sources", True)
# #     show_graph = st.checkbox("Show Graph Visualization", True)

# # if "messages" not in st.session_state:
# #     st.session_state["messages"] = []

# # for msg in st.session_state["messages"]:
# #     with st.chat_message(msg["role"]):
# #         st.markdown(msg["content"])

# # if prompt := st.chat_input("Ask about Syracuse research…"):
# #     st.session_state["messages"].append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)

# #     with st.chat_message("assistant"):
# #         with st.spinner("Retrieving context and graph…"):
# #             out = answer_question(prompt, n_ctx=n_ctx)
# #             st.markdown(out["answer"])

# #             if show_sources:
# #                 with st.expander("Retrieved Sources"):
# #                     for s in out.get("sources", []):
# #                         st.markdown(f"- {s}")

# #             if show_graph:
# #                 st.markdown("### Graph View — Researcher ↔ Papers ↔ Authors")

# #                 # Determine Cypher query dynamically
# #                 researcher_name = ""
# #                 if out.get("graph_hits"):
# #                     researcher_name = out["graph_hits"][0].get("researcher", "")
# #                 else:
# #                     for word in prompt.split():
# #                         if word[0].isupper() and len(word) > 3:
# #                             researcher_name = word
# #                             break

# #                 if researcher_name:
# #                     cypher_query = (
# #                         f"MATCH p=(r:Researcher {{name:'{researcher_name}'}})"
# #                         f"-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
# #                         f"RETURN p LIMIT 25;"
# #                     )
# #                 else:
# #                     cypher_query = (
# #                         "MATCH p=(r:Researcher)-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
# #                         "RETURN p LIMIT 25;"
# #                     )

# #                 graph_output = render_graph(cypher_query, height=650)
# #                 st.write(graph_output)

# #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # streamlit_app.py
# import streamlit as st
# from rag_pipeline import answer_question
# from graph_visualizer import render_graph

# st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# st.title("Syracuse Research Assistant (Hybrid RAG)")

# with st.sidebar:
#     st.subheader("Settings")
#     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
#     show_sources = st.checkbox("Show retrieved sources", True)
#     show_graph = st.checkbox("Show Graph Visualization", True)

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if prompt := st.chat_input("Ask about Syracuse research…"):
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Retrieving context and graph…"):
#             out = answer_question(prompt, n_ctx=n_ctx)
#             st.markdown(out["answer"])

#             if show_sources:
#                 with st.expander("Retrieved Sources"):
#                     for s in out.get("sources", []):
#                         st.markdown(f"- {s}")

#             if show_graph:
#                 st.markdown("### Graph View — Researcher ↔ Papers ↔ Authors")

#                 graph_hits = out.get("graph_hits", []) or []
#                 pids = [h.get("paper_id") for h in graph_hits if h.get("paper_id")]
#                 titles = [h.get("title") for h in graph_hits if h.get("title")]

#                 if pids:
#                     # Drive visualization by paper_id exactly as retrieved
#                     cypher_query = """
#                     UNWIND $pids AS pid
#                     MATCH p0=(pa:Paper {paper_id: pid})
#                     OPTIONAL MATCH p1=(r:Researcher)-[:WROTE]->(pa)
#                     OPTIONAL MATCH p2=(pa)-[:HAS_RESEARCHER]->(r)
#                     OPTIONAL MATCH p3=(pa)-[:HAS_AUTHOR]->(a:Author)
#                     RETURN collect(p0)+collect(p1)+collect(p2)+collect(p3) AS paths
#                     """
#                     graph_output = render_graph(cypher_query, params={"pids": pids}, height=650)

#                 elif titles:
#                     # Fallback: match by exact title (case-insensitive)
#                     cypher_query = """
#                     UNWIND $titles AS t
#                     MATCH p0=(pa:Paper)
#                     WHERE toLower(pa.title) = toLower(t)
#                     OPTIONAL MATCH p1=(r:Researcher)-[:WROTE]->(pa)
#                     OPTIONAL MATCH p2=(pa)-[:HAS_RESEARCHER]->(r)
#                     OPTIONAL MATCH p3=(pa)-[:HAS_AUTHOR]->(a:Author)
#                     RETURN collect(p0)+collect(p1)+collect(p2)+collect(p3) AS paths
#                     """
#                     graph_output = render_graph(cypher_query, params={"titles": titles}, height=650)

#                 else:
#                     # Last resort: small generic view
#                     researcher_name = ""
#                     if graph_hits:
#                         researcher_name = graph_hits[0].get("researcher", "")
#                     if not researcher_name:
#                         for word in prompt.split():
#                             if word[0].isupper() and len(word) > 3:
#                                 researcher_name = word
#                                 break

#                     if researcher_name:
#                         cypher_query = (
#                             "MATCH p=(r:Researcher {name:$name})-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
#                             "OPTIONAL MATCH pauth=(pa)-[:HAS_AUTHOR]->(a:Author) "
#                             "RETURN p, collect(pauth) AS author_paths "
#                             "LIMIT 60"
#                         )
#                         graph_output = render_graph(cypher_query, params={"name": researcher_name}, height=650)
#                     else:
#                         cypher_query = (
#                             "MATCH p=(r:Researcher)-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
#                             "RETURN p LIMIT 25"
#                         )
#                         graph_output = render_graph(cypher_query, height=650)

#                 st.write(graph_output)

#     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})


# streamlit_app.py
import streamlit as st
from rag_pipeline import answer_question
from graph_visualizer import render_graph_from_hits  # updated import

st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
st.title("Syracuse Research Assistant (Hybrid RAG)")

with st.sidebar:
    st.subheader("Settings")
    n_ctx = st.slider("Results per subsystem", 3, 12, 6)
    show_sources = st.checkbox("Show retrieved sources", True)
    show_graph = st.checkbox("Show Graph Visualization", True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Syracuse research…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and graph…"):
            out = answer_question(prompt, n_ctx=n_ctx)
            st.markdown(out["answer"])

            # ---- Show retrieved sources ----
            if show_sources:
                with st.expander("Retrieved Sources"):
                    for s in out.get("sources", []):
                        st.markdown(f"- {s}")

            # ---- Show graph from actual retrieval context ----
            if show_graph:
                st.markdown("### Graph View — Researcher ↔ Papers ↔ Authors")

                graph_hits = out.get("graph_hits", []) or []
                if not graph_hits:
                    st.info("No graph context retrieved from Neo4j for this query.")
                else:
                    graph_output, cypher_query, params = render_graph_from_hits(graph_hits, height=650)
                    st.write(graph_output)

                    with st.expander("Graph Query Used"):
                        st.code(cypher_query.strip(), language="cypher")
                        st.json(params)

    st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})
