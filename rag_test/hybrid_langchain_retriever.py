# # # #hybrid_langchain retriever.py
# # # import hashlib
# # # import chromadb
# # # from langchain_chroma import Chroma
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # import torch
# # # import config_full as config
# # # from graph_retriever import get_papers_by_researcher, get_paper_neighbors

# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # embed = HuggingFaceEmbeddings(
# # #     model_name="intfloat/e5-base-v2",
# # #     model_kwargs={"device": device}
# # # )

# # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # chroma = Chroma(
# # #     client=client,
# # #     collection_name="papers_all",
# # #     embedding_function=embed
# # # )

# # # def _dedupe(lst):
# # #     seen, out = set(), []
# # #     for x in lst:
# # #         h = hashlib.sha1(x.encode()).hexdigest()
# # #         if h not in seen:
# # #             seen.add(h)
# # #             out.append(x)
# # #     return out

# # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # #     # Step 1: Chroma
# # #     chroma_docs = chroma.similarity_search(query, k=k_chroma)

# # #     # Step 2: Extract candidates
# # #     titles = {d.metadata.get("title") for d in chroma_docs if d.metadata.get("title")}
# # #     researchers = {d.metadata.get("researcher") for d in chroma_docs if d.metadata.get("researcher")}

# # #     # Step 3: Query Neo4j for enrichment
# # #     graph_hits = []
# # #     for r in researchers:
# # #         graph_hits.extend(get_papers_by_researcher(r, k=k_graph))
# # #     for t in titles:
# # #         graph_hits.extend(get_paper_neighbors(t, k_authors=5))

# # #     # Step 4: Fuse
# # #     fused = []
# # #     for d in chroma_docs:
# # #         fused.append(f"[Chroma] {d.metadata.get('title','Untitled')} ({d.metadata.get('publication_date','?')})\n{d.page_content[:300]}")
# # #     for g in graph_hits:
# # #         fused.append(f"[Graph] {g.get('title','Untitled')} ({g.get('year','?')}) — {', '.join(g.get('authors', []))}")

# # #     return {
# # #         "graph_hits": graph_hits,
# # #         "chroma_ctx": [d.page_content for d in chroma_docs],
# # #         "fused_text_blocks": _dedupe(fused),
# # #     }


# # # hybrid_langchain_retriever.py
# # import hashlib
# # import torch
# # import chromadb
# # from langchain_chroma import Chroma
# # from langchain_huggingface import HuggingFaceEmbeddings
# # import config_full as config
# # from graph_retriever import search_graph

# # # ------------------ device + embedder ------------------
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # embed = HuggingFaceEmbeddings(
# #     model_name="intfloat/e5-base-v2",
# #     model_kwargs={"device": device}
# # )

# # # ------------------ chroma setup ------------------
# # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # chroma = Chroma(
# #     client=client,
# #     collection_name="papers_all",
# #     embedding_function=embed
# # )

# # # ------------------ helpers ------------------
# # def _dedupe(lst):
# #     """Deduplicate list preserving order."""
# #     seen, out = set(), []
# #     for x in lst:
# #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# #         if h not in seen:
# #             seen.add(h)
# #             out.append(x)
# #     return out

# # def _safe_str(val):
# #     if val is None:
# #         return ""
# #     return str(val).strip()

# # # ------------------ hybrid search ------------------
# # def hybrid_search(query: str, k_chroma: int = 10, k_graph: int = 8):
# #     """
# #     1. Retrieve top-k docs from Chroma (semantic similarity)
# #     2. For each retrieved doc, extract researcher/title
# #     3. Query Neo4j for related works using top keywords
# #     4. Merge, clean, and dedupe the results
# #     """

# #     # ---- Step 1: Chroma retrieval ----
# #     chroma_docs = chroma.similarity_search(query, k=k_chroma)
# #     if not chroma_docs:
# #         return {
# #             "graph_hits": [],
# #             "chroma_ctx": [],
# #             "fused_text_blocks": ["No Chroma context found."]
# #         }

# #     chroma_texts = [d.page_content for d in chroma_docs if d.page_content]
# #     chroma_metas = [d.metadata for d in chroma_docs]

# #     # ---- Step 2: derive search hints from Chroma metadata ----
# #     hint_titles = [_safe_str(m.get("title", "")) for m in chroma_metas if m.get("title")]
# #     hint_researchers = [_safe_str(m.get("researcher", "")) for m in chroma_metas if m.get("researcher")]
# #     expanded_queries = _dedupe([query] + hint_titles + hint_researchers)

# #     # ---- Step 3: Query Neo4j with top-hints ----
# #     graph_hits = []
# #     for subq in expanded_queries[:3]:  # top 3 to keep speed manageable
# #         res = search_graph(subq, k=k_graph)
# #         graph_hits.extend(res)
# #     # flatten + unique
# #     unique_titles = set()
# #     final_graph_hits = []
# #     for g in graph_hits:
# #         title = _safe_str(g.get("title"))
# #         if title and title not in unique_titles:
# #             unique_titles.add(title)
# #             final_graph_hits.append(g)

# #     # ---- Step 4: fuse Chroma + Neo4j ----
# #     fused = []
# #     for g in final_graph_hits:
# #         fused.append(f"[Graph] {g.get('title', 'Untitled')} ({g.get('year', 'N/A')}) "
# #                      f"— Researcher: {g.get('researcher', 'Unknown')}, "
# #                      f"Authors: {', '.join(g.get('authors', []) or [])}")

# #     for d in chroma_docs:
# #         meta = d.metadata or {}
# #         fused.append(f"[Chroma] Title: {meta.get('title', 'Untitled')} | "
# #                      f"Researcher: {meta.get('researcher', 'Unknown')} | "
# #                      f"Excerpt: {d.page_content[:200]}")

# #     fused_clean = [x for x in _dedupe(fused) if "N/A" not in x and len(x.strip()) > 20]

# #     return {
# #         "graph_hits": final_graph_hits,
# #         "chroma_ctx": chroma_texts,
# #         "fused_text_blocks": fused_clean,
# #     }

# # # ------------------ standalone test ------------------
# # if __name__ == "__main__":
# #     q = "papers by Gillian Youngs"
# #     res = hybrid_search(q)
# #     print(f"\n🔍 Query: {q}")
# #     print(f"Chroma docs: {len(res['chroma_ctx'])}, Graph hits: {len(res['graph_hits'])}")
# #     print("\nSample fused context:\n")
# #     for s in res["fused_text_blocks"][:10]:
# #         print("-", s)

# # hybrid_langchain_retriever.py
# import hashlib
# import torch
# import chromadb
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import config_full as config
# from graph_retriever import search_graph

# device = "cuda" if torch.cuda.is_available() else "cpu"

# embed = HuggingFaceEmbeddings(
#     model_name="intfloat/e5-base-v2",
#     model_kwargs={"device": device}
# )

# client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# chroma = Chroma(
#     client=client,
#     collection_name="papers_all",
#     embedding_function=embed
# )

# # ------------------ helpers ------------------
# def _dedupe(lst):
#     seen, out = set(), []
#     for x in lst:
#         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
#         if h not in seen:
#             seen.add(h)
#             out.append(x)
#     return out

# def _safe_str(v):
#     return str(v).strip() if v else ""

# # ------------------ hybrid retrieval ------------------
# def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 5):
#     """Chroma → Graph → merge."""
#     chroma_docs = chroma.similarity_search(query, k=k_chroma)
#     if not chroma_docs:
#         return {"graph_hits": [], "chroma_ctx": [], "fused_text_blocks": ["No Chroma matches found."]}

#     chroma_texts = [d.page_content for d in chroma_docs if d.page_content]
#     chroma_metas = [d.metadata for d in chroma_docs]

#     # graph enrichment based on retrieved chroma docs
#     graph_hits = []
#     for meta in chroma_metas:
#         title = _safe_str(meta.get("title"))
#         researcher = _safe_str(meta.get("researcher"))
#         doi = _safe_str(meta.get("doi"))
#         q = title or researcher or doi
#         if q:
#             graph_hits.extend(search_graph(q, k=k_graph))

#     # ------------------ fuse ------------------
#     fused = []
#     for d in chroma_docs:
#         m = d.metadata or {}
#         fused.append(
#             f"[Chroma] Title: {m.get('title','Untitled')} | "
#             f"Researcher: {m.get('researcher','Unknown')} | "
#             f"Excerpt: {d.page_content[:250]}"
#         )

#     for g in graph_hits:
#         fused.append(
#             f"[Graph] {g.get('title','Untitled')} ({g.get('year','N/A')}) — "
#             f"Researcher: {_safe_str(g.get('researcher'))}, "
#             f"Authors: {', '.join(g.get('authors', []) or [])}"
#         )

#     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
#     return {"graph_hits": graph_hits, "chroma_ctx": chroma_texts, "fused_text_blocks": fused_clean}

# if __name__ == "__main__":
#     res = hybrid_search("jeff")
#     print(f"Chroma={len(res['chroma_ctx'])}, Graph={len(res['graph_hits'])}")
#     for s in res["fused_text_blocks"][:8]:
#         print("-", s)

# hybrid_langchain_retriever.py
import hashlib, torch, chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import config_full as config
from graph_retriever import search_graph_from_chroma_meta

device = "cuda" if torch.cuda.is_available() else "cpu"
embed = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2", model_kwargs={"device": device})
client = chromadb.PersistentClient(path=config.CHROMA_DIR)
chroma = Chroma(client=client, collection_name="papers_all", embedding_function=embed)

def _dedupe(lst):
    seen, out = set(), []
    for x in lst:
        h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
        if h not in seen:
            seen.add(h); out.append(x)
    return out

def _safe_str(v): return str(v).strip() if v else ""

def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
    chroma_docs = chroma.similarity_search(query, k=k_chroma)
    if not chroma_docs:
        return {"graph_hits": [], "chroma_ctx": [], "fused_text_blocks": ["No Chroma matches found."]}

    chroma_texts = [d.page_content for d in chroma_docs if d.page_content]
    chroma_metas = [d.metadata for d in chroma_docs]

    # Graph expansion *only for those Chroma-identified papers/researchers*
    graph_hits = search_graph_from_chroma_meta(query, chroma_metas, k=k_graph)

    fused = []
    for d in chroma_docs:
        m = d.metadata or {}
        fused.append(
            f"[Chroma] Title: {m.get('title','Untitled')} | Researcher: {m.get('researcher','Unknown')} | "
            f"Excerpt: {d.page_content[:250]}"
        )

    for g in graph_hits:
        fused.append(
            f"[Graph] {g.get('title','Untitled')} ({g.get('year','N/A')}) — "
            f"Researcher: {_safe_str(g.get('researcher'))}, "
            f"Authors: {', '.join(g.get('authors',[]) or [])} | "
            f"Related: {', '.join(g.get('related',[])[:3])} | "
            f"Score: {g['score']}"
        )

    fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
    return {"graph_hits": graph_hits, "chroma_ctx": chroma_texts, "fused_text_blocks": fused_clean}
