# # # #rag_pipeleine.py
# # # import re, torch
# # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # import config_full as config
# # # from hybrid_langchain_retriever import hybrid_search

# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # tok, model = None, None

# # # def _load_llm():
# # #     global tok, model
# # #     if tok: return tok, model
# # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # #     model = AutoModelForCausalLM.from_pretrained(
# # #         config.LLAMA_MODEL_PATH,
# # #         torch_dtype=dtype,
# # #         device_map="auto" if device == "cuda" else None,
# # #         low_cpu_mem_usage=True,
# # #     )
# # #     if tok.pad_token_id is None and tok.eos_token_id:
# # #         tok.pad_token_id = tok.eos_token_id
# # #     return tok, model

# # # def sanitize(blocks, max_chars=6000):
# # #     out, total = [], 0
# # #     for b in blocks:
# # #         b = re.sub(r"<[^>]+>", " ", b).strip()
# # #         if not b or b in ["N/A", "Unknown"]:
# # #             continue
# # #         if total + len(b) > max_chars:
# # #             break
# # #         out.append(b)
# # #         total += len(b)
# # #     return out

# # # def build_prompt(question, context):
# # #     return (
# # #         f"<|system|>Use only the provided context to answer.<|/system|>\n"
# # #         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
# # #         "\n</|user|>\n<|assistant|>"
# # #     )

# # # def answer_question(question: str, n_ctx: int = 6):
# # #     retr = hybrid_search(question, k_graph=n_ctx, k_chroma=n_ctx)
# # #     ctx = sanitize(retr["fused_text_blocks"])
# # #     if not ctx:
# # #         return {"answer": "Not found in retrieved materials."}
# # #     prompt = build_prompt(question, ctx)

# # #     tok, model = _load_llm()
# # #     inputs = tok(prompt, return_tensors="pt").to(model.device)
# # #     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
# # #     out = tok.decode(gen[0], skip_special_tokens=True)
# # #     ans = out.split("<|assistant|>")[-1].strip()
# # #     return {"answer": ans, **retr}

# # # rag_pipeline.py
# # import re, torch
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # import config_full as config
# # from hybrid_langchain_retriever import hybrid_search

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # dtype = torch.float16 if device == "cuda" else torch.float32
# # tok, model = None, None

# # def _load_llm():
# #     global tok, model
# #     if tok: return tok, model
# #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# #     model = AutoModelForCausalLM.from_pretrained(
# #         config.LLAMA_MODEL_PATH,
# #         torch_dtype=dtype,
# #         device_map="auto" if device == "cuda" else None,
# #         low_cpu_mem_usage=True,
# #     )
# #     if tok.pad_token_id is None and tok.eos_token_id:
# #         tok.pad_token_id = tok.eos_token_id
# #     return tok, model

# # def sanitize(blocks, max_chars=6000):
# #     out, total = [], 0
# #     for b in blocks:
# #         b = re.sub(r"<[^>]+>", " ", b)
# #         if 20 < len(b) < 500:
# #             if total + len(b) > max_chars:
# #                 break
# #             out.append(b)
# #             total += len(b)
# #     return out

# # def build_prompt(question, context):
# #     return (
# #         f"<|system|>Answer using the context only.<|/system|>\n"
# #         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
# #         "\n</|user|>\n<|assistant|>"
# #     )

# # def answer_question(question: str, n_ctx: int = 6):
# #     retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
# #     ctx = sanitize(retr["fused_text_blocks"])
# #     if not ctx:
# #         return {"answer": "No relevant information found."}
# #     prompt = build_prompt(question, ctx)

# #     tok, model = _load_llm()
# #     inputs = tok(prompt, return_tensors="pt").to(model.device)
# #     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
# #     out = tok.decode(gen[0], skip_special_tokens=True)
# #     ans = out.split("<|assistant|>")[-1].strip()
# #     return {"answer": ans, **retr}


# # rag_pipeline.py
# import re, torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import config_full as config
# from hybrid_langchain_retriever import hybrid_search

# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float16 if device == "cuda" else torch.float32

# tok, model = None, None

# def _load_llm():
#     """Load local LLaMA model once."""
#     global tok, model
#     if tok and model:
#         return tok, model
#     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
#     model = AutoModelForCausalLM.from_pretrained(
#         config.LLAMA_MODEL_PATH,
#         torch_dtype=dtype,
#         device_map="auto" if device == "cuda" else None,
#         low_cpu_mem_usage=True,
#     )
#     if tok.pad_token_id is None and tok.eos_token_id:
#         tok.pad_token_id = tok.eos_token_id
#     return tok, model


# def sanitize(blocks, max_chars=6000):
#     """Remove tags and truncate overly long chunks."""
#     out, total = [], 0
#     for b in blocks:
#         b = re.sub(r"<[^>]+>", " ", b)
#         if 20 < len(b) < 500:
#             if total + len(b) > max_chars:
#                 break
#             out.append(b)
#             total += len(b)
#     return out


# def build_prompt(question, context):
#     """Format context for instruction-style LLaMA prompt."""
#     return (
#         f"<|system|>Use only the provided Syracuse University research context.<|/system|>\n"
#         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
#         "\n</|user|>\n<|assistant|>"
#     )


# def answer_question(question: str, n_ctx: int = 6):
#     """Main hybrid retrieval + generation pipeline."""
#     retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
#     ctx = sanitize(retr.get("fused_text_blocks", []))

#     if not ctx:
#         return {
#             "answer": "No relevant information found in Chroma or Graph retrievals.",
#             "sources": [],
#             "graph_hits": []
#         }

#     prompt = build_prompt(question, ctx)
#     tok, model = _load_llm()
#     inputs = tok(prompt, return_tensors="pt").to(model.device)
#     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
#     out = tok.decode(gen[0], skip_special_tokens=True)
#     ans = out.split("<|assistant|>")[-1].strip()

#     return {
#         "answer": ans,
#         "sources": ctx,
#         "graph_hits": retr.get("graph_hits", [])
#     }

import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config_full as config
from hybrid_langchain_retriever import hybrid_search

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tok, model = None, None

def _load_llm():
    global tok, model
    if tok and model:
        return tok, model
    tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLAMA_MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if tok.pad_token_id is None and tok.eos_token_id:
        tok.pad_token_id = tok.eos_token_id
    return tok, model


def sanitize(blocks, max_chars=6000):
    out, total = [], 0
    for b in blocks:
        b = re.sub(r"<[^>]+>", " ", b)
        if 20 < len(b) < 500:
            if total + len(b) > max_chars:
                break
            out.append(b)
            total += len(b)
    return out


def build_prompt(question, context):
    return (
        f"<|system|>Answer using only the provided Syracuse University research context.<|/system|>\n"
        f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
        "\n</|user|>\n<|assistant|>"
    )


def answer_question(question: str, n_ctx: int = 6):
    retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
    ctx = sanitize(retr.get("fused_text_blocks", []))

    if not ctx:
        return {
            "answer": "No relevant information found in Chroma or Graph retrievals.",
            "sources": [],
            "graph_hits": []
        }

    prompt = build_prompt(question, ctx)
    tok, model = _load_llm()
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
    out = tok.decode(gen[0], skip_special_tokens=True)
    ans = out.split("<|assistant|>")[-1].strip()

    return {
        "answer": ans,
        "sources": ctx,
        "graph_hits": retr.get("graph_hits", [])
    }
