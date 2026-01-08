#rag_pipeline.py
import torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
import config_full as config
from runtime_settings import settings
from hybrid_langchain_retriever import hybrid_search
from cache_manager import fetch, store
from conversation_memory import add_turn, should_compress, compress_memory, get_memory_context

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device=="cuda" else torch.float32
_tok = _model = None

def _load_llm():
    global _tok,_model
    if _tok and _model: return _tok,_model
    path = config.LLAMA_3B if settings.llm_model=="llama-3.2-3b" else config.LLAMA_1B
    _tok = AutoTokenizer.from_pretrained(path)
    _model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
    if _tok.pad_token_id is None: _tok.pad_token_id = _tok.eos_token_id
    return _tok,_model

def _free_vram_k():
    if not torch.cuda.is_available(): return 10
    free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    return max(min(int((free/(1024**3))*8),60),8)

def answer_question(q):
    cached = fetch(q)
    if cached: 
        ans,sources = cached
        return {"answer":ans,"sources":sources}

    k = _free_vram_k()
    result = hybrid_search(q, k=k)
    ctx = [re.sub(r"<[^>]+>"," ",x).strip() for x in result["chroma_ctx"]]

    if not ctx:
        return {"answer": "No relevant research found.", "sources":[]}

    memory = get_memory_context()
    prompt = f"""<|system|>You are a Syracuse Research RAG assistant.<|/system|>
<|user|>
{memory}
Question: {q}

Context:
{chr(10).join(ctx)}
</|user|>
<|assistant|>
"""

    tok,model = _load_llm()
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=350, temperature=0.25, top_p=0.9)
    ans = tok.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

    store(q,ans,ctx)
    add_turn(q,ans)
    if should_compress(): compress_memory()

    return {"answer": ans, "sources": ctx}
