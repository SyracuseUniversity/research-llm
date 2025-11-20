# #coneversation_memory.py
# import torch, gc
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# device = "cuda" if torch.cuda.is_available() else "cpu"

# memory_summary = ""
# recent_turns = []
# SUMMARIZER_MODEL = "facebook/bart-large-cnn"

# def add_turn(user_msg, assistant_msg):
#     recent_turns.append((user_msg, assistant_msg))

# def _vr_free():
#     if not torch.cuda.is_available(): return 4
#     free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
#     return free / (1024**3)

# def should_compress():
#     return len(recent_turns) > max(int(_vr_free()*3), 4)

# def compress_memory():
#     global memory_summary, recent_turns
#     if not recent_turns: return memory_summary

#     tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
#     model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL).to(device)

#     text = "\n".join([f"User: {u}\nAssistant: {a}" for u,a in recent_turns])
#     enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
#     out = model.generate(**enc, max_new_tokens=180)
#     summary = tok.decode(out[0], skip_special_tokens=True)

#     memory_summary = (memory_summary + "\n" + summary).strip()
#     recent_turns = []

#     gc.collect()
#     if torch.cuda.is_available(): torch.cuda.empty_cache()
#     return memory_summary

# def get_memory_context():
#     return f"Previous conversation summary:\n{memory_summary}\n\n" if memory_summary else ""


# conversation_memory.py
import os
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MEM_DIR = os.path.expanduser("~/.rag_memory")
os.makedirs(MEM_DIR, exist_ok=True)

MEM_FILE = os.path.join(MEM_DIR, "session_memory.json")

memory_summary = ""
recent_turns = []

paper_summary = ""
paper_ids = []

SUMMARIZER = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"


def _save():
    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": memory_summary,
                "recent_turns": recent_turns,
                "paper_summary": paper_summary,
                "paper_ids": paper_ids,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _load():
    global memory_summary, recent_turns, paper_summary, paper_ids
    if os.path.exists(MEM_FILE):
        try:
            with open(MEM_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                memory_summary = d.get("summary", "")
                recent_turns = d.get("recent_turns", [])
                paper_summary = d.get("paper_summary", "")
                paper_ids = d.get("paper_ids", [])
        except Exception:
            memory_summary = ""
            recent_turns = []
            paper_summary = ""
            paper_ids = []


_load()


def reset_memory():
    global memory_summary, recent_turns, paper_summary, paper_ids
    memory_summary = ""
    recent_turns = []
    paper_summary = ""
    paper_ids = []
    _save()


def add_turn(q, a):
    recent_turns.append((q.strip(), a.strip()))
    _save()


def should_compress():
    return len(recent_turns) >= 5


def _run_summarizer(text, max_new_tokens=200):
    tok = AutoTokenizer.from_pretrained(SUMMARIZER)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER).to(device)

    enc = tok(text, return_tensors="pt", truncation=True).to(device)

    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens)

    summary = tok.decode(out[0], skip_special_tokens=True).strip()

    del model, tok, enc, out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def compress_memory():
    global memory_summary, recent_turns

    if not recent_turns:
        return

    text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in recent_turns])

    new_summary = _run_summarizer(text, max_new_tokens=200)

    if memory_summary:
        merged = f"Existing summary:\n{memory_summary}\n\nNew interactions:\n{new_summary}"
        memory_summary = _run_summarizer(merged, max_new_tokens=200)
    else:
        memory_summary = new_summary

    recent_turns = []
    _save()


def update_paper_memory(sources):
    global paper_summary, paper_ids

    if not sources:
        return

    new_entries = []
    new_ids = []

    for s in sources:
        pid = s.get("id") or s.get("title") or s.get("snippet", "")[:50]
        if not pid:
            continue
        if pid in paper_ids:
            continue

        new_ids.append(pid)
        title = s.get("title", "Unknown")
        snippet = s.get("snippet", "")
        new_entries.append(f"Title: {title}\nSnippet: {snippet}")

    if not new_entries:
        return

    paper_ids.extend(new_ids)

    raw = "\n\n".join(new_entries)
    new_summary = _run_summarizer(raw, max_new_tokens=256)

    if paper_summary:
        merged = f"Existing papers:\n{paper_summary}\n\nNew papers:\n{new_summary}"
        paper_summary = _run_summarizer(merged, max_new_tokens=256)
    else:
        paper_summary = new_summary

    _save()


def get_memory_context():
    _load()

    parts = []
    if memory_summary:
        parts.append("Conversation summary:\n" + memory_summary)
    if recent_turns:
        tail = recent_turns[-5:]
        parts.append(
            "Recent turns:\n"
            + "\n".join([f"User: {u}\nAssistant: {a}" for u, a in tail])
        )
    if paper_summary:
        parts.append("Relevant papers summary:\n" + paper_summary)

    return "\n\n".join(parts)
