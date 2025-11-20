#coneversation_memory.py
import torch, gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

memory_summary = ""
recent_turns = []
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

def add_turn(user_msg, assistant_msg):
    recent_turns.append((user_msg, assistant_msg))

def _vr_free():
    if not torch.cuda.is_available(): return 4
    free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    return free / (1024**3)

def should_compress():
    return len(recent_turns) > max(int(_vr_free()*3), 4)

def compress_memory():
    global memory_summary, recent_turns
    if not recent_turns: return memory_summary

    tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL).to(device)

    text = "\n".join([f"User: {u}\nAssistant: {a}" for u,a in recent_turns])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    out = model.generate(**enc, max_new_tokens=180)
    summary = tok.decode(out[0], skip_special_tokens=True)

    memory_summary = (memory_summary + "\n" + summary).strip()
    recent_turns = []

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return memory_summary

def get_memory_context():
    return f"Previous conversation summary:\n{memory_summary}\n\n" if memory_summary else ""
