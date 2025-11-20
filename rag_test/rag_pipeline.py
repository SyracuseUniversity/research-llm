# #rag_pipeline.py
# import torch, re
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import config_full as config
# from runtime_settings import settings
# from hybrid_langchain_retriever import hybrid_search
# from cache_manager import fetch, store
# from conversation_memory import add_turn, should_compress, compress_memory, get_memory_context

# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float16 if device=="cuda" else torch.float32
# _tok = _model = None

# def _load_llm():
#     global _tok,_model
#     if _tok and _model: return _tok,_model
#     path = config.LLAMA_3B if settings.llm_model=="llama-3.2-3b" else config.LLAMA_1B
#     _tok = AutoTokenizer.from_pretrained(path)
#     _model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
#     if _tok.pad_token_id is None: _tok.pad_token_id = _tok.eos_token_id
#     return _tok,_model

# def _free_vram_k():
#     if not torch.cuda.is_available(): return 10
#     free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
#     return max(min(int((free/(1024**3))*8),60),8)

# def answer_question(q):
#     cached = fetch(q)
#     if cached: 
#         ans,sources = cached
#         return {"answer":ans,"sources":sources}

#     k = _free_vram_k()
#     result = hybrid_search(q, k=k)
#     ctx = [re.sub(r"<[^>]+>"," ",x).strip() for x in result["chroma_ctx"]]

#     if not ctx:
#         return {"answer": "No relevant research found.", "sources":[]}

#     memory = get_memory_context()
#     prompt = f"""<|system|>You are a Syracuse Research RAG assistant.<|/system|>
# <|user|>
# {memory}
# Question: {q}

# Context:
# {chr(10).join(ctx)}
# </|user|>
# <|assistant|>
# """

#     tok,model = _load_llm()
#     inp = tok(prompt, return_tensors="pt").to(model.device)
#     out = model.generate(**inp, max_new_tokens=350, temperature=0.25, top_p=0.9)
#     ans = tok.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

#     store(q,ans,ctx)
#     add_turn(q,ans)
#     if should_compress(): compress_memory()

#     return {"answer": ans, "sources": ctx}


# rag_pipeline.py
# -----------------------------------------------------------
# Strict extractive RAG for Syracuse research
# -----------------------------------------------------------

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from runtime_settings import settings
from hybrid_langchain_retriever import hybrid_search
from conversation_memory import (
    add_turn,
    get_memory_context,
    should_compress,
    compress_memory,
    update_paper_memory,
)
from cache_manager import fetch, store
import config_full as config


# -------- LLaMA LOADER --------

_llm_cache = {}


def load_llm(model_name):
    if model_name in _llm_cache:
        return _llm_cache[model_name]

    path = config.LLAMA_1B if model_name == "llama-3.2-1b" else config.LLAMA_3B
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    _llm_cache[model_name] = (tok, model)
    return tok, model


# -------- STRICT PROMPT --------

def build_prompt(q, ctx, mem):
    return f"""
You are the Syracuse University Research Assistant.

Rules:
ONLY use the information from CONTEXT.
ALL retrieved papers are Syracuse affiliated by default.
DO NOT question affiliation.
DO NOT add anything not present in context.
Provide a concise summary directly answering the question.

---------------- CONTEXT ----------------
{ctx}
-----------------------------------------

---------------- MEMORY -----------------
{mem}
-----------------------------------------

QUESTION:
{q}

ANSWER:
""".strip()


# -------- GENERATION --------

def generate(tok, model, prompt):
    device = model.device
    enc = tok(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=350,
            temperature=0.2,
            do_sample=False,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    if "ANSWER:" in text:
        text = text.split("ANSWER:")[-1].strip()
    return text.strip()


# -------- NAME FILTER --------

def extract_name(question):
    m = re.findall(r"[A-Z][a-z]+\s+[A-Z][a-z]+", question)
    return m[0] if m else None


def name_matches(name, hit):
    if not name:
        return True
    block = (
        hit.get("title", "") + " "
        + hit.get("authors", "") + " "
        + hit.get("researcher", "") + " "
        + hit.get("snippet", "")
    ).lower()
    return name.lower() in block


# -------- MAIN --------

def answer_question(question):
    q = question.strip()

    # Cache check
    cached = fetch(q)
    if cached:
        ans, srcs = cached
        add_turn(q, ans)
        update_paper_memory(srcs)
        if should_compress():
            compress_memory()
        return {"answer": ans, "sources_detailed": srcs}

    # Retrieve with hybrid search
    hits = hybrid_search(q, top_k=20)

    if not hits:
        ans = "No relevant information found in the retrieved Syracuse research context."
        add_turn(q, ans)
        if should_compress():
            compress_memory()
        store(q, ans, [])
        return {"answer": ans, "sources_detailed": []}

    # Optional researcher name filtering
    name = extract_name(q)
    filtered = [h for h in hits if name_matches(name, h)]
    used = filtered if filtered else hits

    # Build context string from used hits
    ctx = "\n\n".join(
        f"Title: {h.get('title', 'Unknown')}\nSnippet: {h.get('snippet', '')}"
        for h in used
    )

    mem = get_memory_context()
    tok, model = load_llm(settings.llm_model)
    prompt = build_prompt(q, ctx, mem)
    ans = generate(tok, model, prompt)

    # Memory
    add_turn(q, ans)
    update_paper_memory(used)
    if should_compress():
        compress_memory()

    # Cache
    store(q, ans, used)

    return {"answer": ans, "sources_detailed": used}


