# rag_engine.py
import os
import uuid
import json
import gc
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

import config_full as config
from session_store import SessionStore
from database_manager import DatabaseManager


MEMORY_DIR = "chroma_memory"
CACHE_DIR = "cache"
STATE_DB = "chat_state.sqlite"

CTX = 8192

BUDGET_SUMMARY = 700
BUDGET_MEMORY = 700
BUDGET_RECENT = 1200
BUDGET_PAPERS = 3200
TRIGGER = 7200


@dataclass
class Turn:
    role: str
    text: str


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def available_ram_mb() -> int:
    return int(psutil.virtual_memory().available / (1024 * 1024))


def available_vram_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        return int(free_b / (1024 * 1024))
    except Exception:
        return 0


def dynamic_budgets() -> Dict[str, int]:
    ram = available_ram_mb()
    vram = available_vram_mb()

    pressure = 0
    if ram < 2000:
        pressure += 2
    elif ram < 4000:
        pressure += 1

    if torch.cuda.is_available():
        if vram < 1500:
            pressure += 2
        elif vram < 3000:
            pressure += 1

    if pressure >= 3:
        return {"BUDGET_MEMORY": 450, "BUDGET_RECENT": 800, "BUDGET_PAPERS": 2200, "TRIGGER": 6400}
    if pressure == 2:
        return {"BUDGET_MEMORY": 550, "BUDGET_RECENT": 950, "BUDGET_PAPERS": 2700, "TRIGGER": 6800}
    if pressure == 1:
        return {"BUDGET_MEMORY": 650, "BUDGET_RECENT": 1100, "BUDGET_PAPERS": 3000, "TRIGGER": 7000}
    return {"BUDGET_MEMORY": BUDGET_MEMORY, "BUDGET_RECENT": BUDGET_RECENT, "BUDGET_PAPERS": BUDGET_PAPERS, "TRIGGER": TRIGGER}


class ModelRuntime:
    def __init__(self, model_id_or_path: str):
        self.model_id_or_path = model_id_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )

        gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
        self.llm = HuggingFacePipeline(pipeline=gen_pipe)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def pack_docs(docs: List[Document], budget: int, count_tokens_fn) -> List[Document]:
    out: List[Document] = []
    total = 0
    for d in docs:
        t = count_tokens_fn(d.page_content)
        if total + t > budget:
            break
        out.append(d)
        total += t
    return out


def format_docs(docs: List[Document], prefix: str) -> str:
    blocks: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        cid = meta.get("chunk_id") or meta.get("id") or meta.get("chunk") or "chunk"
        pid = meta.get("paper_id", "")
        title = meta.get("title", "")
        blocks.append(f"[{prefix} {cid}] paper_id={pid} title={title}\n{d.page_content}")
    return "\n\n".join(blocks)


def pack_recent_turns(turns: List[Turn], budget: int, count_tokens_fn) -> str:
    total = 0
    acc = ""
    for t in reversed(turns):
        block = f"{t.role.upper()}: {t.text}\n"
        bt = count_tokens_fn(block)
        if total + bt > budget:
            break
        acc = block + acc
        total += bt
    return acc.strip()


MEMORY_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract persistent memory items. Return STRICT JSON with keys: facts, decisions, preferences, tasks. "
            "Each value is a list of objects: {\"text\": str, \"salience\": 1-5}. "
            "Each text under 25 words. No extra keys.",
        ),
        ("human", "USER:\n{user}\n\nASSISTANT:\n{assistant}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Update a rolling summary. Output plain text with sections:\n"
            "Goal\nConstraints\nDecisions\nKey entities\nOpen questions\n"
            "Keep it compact.",
        ),
        ("human", "OLD SUMMARY:\n{old_summary}\n\nNEW TURNS:\n{new_turns}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a RAG assistant. Use PAPER CONTEXT as primary evidence. "
            "Use MEMORY only for continuity. If papers do not support a claim, say so.",
        ),
        (
            "human",
            "ROLLING SUMMARY:\n{rolling_summary}\n\n"
            "MEMORY SNIPPETS:\n{memory_snips}\n\n"
            "RECENT TURNS:\n{recent}\n\n"
            "PAPER CONTEXT:\n{papers}\n\n"
            "USER QUESTION:\n{question}",
        ),
    ]
)


class Engine:
    def __init__(
        self,
        runtime: ModelRuntime,
        papers_vs: Chroma,
        memory_vs: Chroma,
        store: SessionStore,
        session_id: str,
    ):
        self.runtime = runtime
        self.papers_vs = papers_vs
        self.memory_vs = memory_vs
        self.store = store
        self.session_id = session_id

        state = self.store.load(session_id)
        self.rolling_summary: str = state.get("rolling_summary", "") or ""
        turns_raw = state.get("turns", []) or []
        self.turns: List[Turn] = []
        for obj in turns_raw:
            role = (obj.get("role") or "").strip()
            text = (obj.get("text") or "").strip()
            if role and text:
                self.turns.append(Turn(role=role, text=text))

    def _persist_state(self) -> None:
        turns_json = [{"role": t.role, "text": t.text} for t in self.turns]
        self.store.save(self.session_id, self.rolling_summary, turns_json)

    def retrieve_papers(self, query: str, budget_papers: int) -> List[Document]:
        retriever = self.papers_vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 22, "fetch_k": 100, "lambda_mult": 0.6},
        )
        docs = retriever.invoke(query)
        return pack_docs(docs, budget_papers, self.runtime.count_tokens)

    def retrieve_memory(self, query: str, budget_memory: int) -> List[Document]:
        retriever = self.memory_vs.as_retriever(
            search_kwargs={
                "k": 12,
                "filter": {"session_id": self.session_id},
            }
        )
        docs = retriever.invoke(query)
        return pack_docs(docs, budget_memory, self.runtime.count_tokens)

    def update_summary(self, new_turns_text: str) -> None:
        msgs = SUMMARY_PROMPT.format_messages(
            old_summary=self.rolling_summary,
            new_turns=new_turns_text,
        )
        out = self.runtime.llm.invoke(msgs[0].content + "\n" + msgs[1].content)
        self.rolling_summary = (out or "").strip()

    def extract_memory(self, user: str, assistant: str) -> None:
        msgs = MEMORY_EXTRACT_PROMPT.format_messages(user=user, assistant=assistant)
        raw = self.runtime.llm.invoke(msgs[0].content + "\n" + msgs[1].content)

        try:
            data = json.loads(raw)
        except Exception:
            return

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        def add(key: str) -> None:
            items = data.get(key, [])
            if not isinstance(items, list):
                return
            for obj in items[:20]:
                if not isinstance(obj, dict):
                    continue
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                sal = int(obj.get("salience") or 3)
                sal = max(1, min(5, sal))
                texts.append(text)
                metas.append({"type": key, "salience": sal, "session_id": self.session_id})

        add("facts")
        add("decisions")
        add("preferences")
        add("tasks")

        if texts:
            ids = [str(uuid.uuid4()) for _ in texts]
            self.memory_vs.add_texts(texts=texts, metadatas=metas, ids=ids)
            try:
                self.memory_vs.persist()
            except Exception:
                pass

    def ask(self, question: str) -> Tuple[str, List[Document], List[Document]]:
        budgets = dynamic_budgets()
        budget_memory = budgets["BUDGET_MEMORY"]
        budget_recent = budgets["BUDGET_RECENT"]
        budget_papers = budgets["BUDGET_PAPERS"]
        trigger = budgets["TRIGGER"]

        query = (self.rolling_summary + "\n" + question).strip()

        mem_docs = self.retrieve_memory(query, budget_memory)
        paper_docs = self.retrieve_papers(query, budget_papers)

        memory_snips = format_docs(mem_docs, "MEM")
        papers_ctx = format_docs(paper_docs, "PAPER")
        recent = pack_recent_turns(self.turns, budget_recent, self.runtime.count_tokens)

        msgs = ANSWER_PROMPT.format_messages(
            rolling_summary=self.rolling_summary,
            memory_snips=memory_snips,
            recent=recent,
            papers=papers_ctx,
            question=question,
        )

        full_prompt = "\n\n".join(m.content for m in msgs)
        if self.runtime.count_tokens(full_prompt) > trigger:
            self.update_summary(recent)
            self.turns = []
            recent = ""
            msgs = ANSWER_PROMPT.format_messages(
                rolling_summary=self.rolling_summary,
                memory_snips=memory_snips,
                recent=recent,
                papers=papers_ctx,
                question=question,
            )

        answer = (self.runtime.llm.invoke("\n\n".join(m.content for m in msgs)) or "").strip()

        self.turns.append(Turn("user", question))
        self.turns.append(Turn("assistant", answer))

        self.extract_memory(question, answer)
        self._persist_state()
        return answer, paper_docs, mem_docs

    def reset_chat_state(self) -> None:
        self.rolling_summary = ""
        self.turns = []
        self.store.reset(self.session_id)

    def reset_memory_vectors(self) -> None:
        try:
            self.memory_vs.delete(where={"session_id": self.session_id})
            self.memory_vs.persist()
        except Exception:
            pass

    def reset_all_for_session(self) -> None:
        self.reset_memory_vectors()
        self.reset_chat_state()


def build_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
    )


def clear_runtime_cache() -> None:
    _ensure_dir(CACHE_DIR)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_llm_path(llm_model_key: str) -> str:
    key = (llm_model_key or "").strip().lower()
    if key in {"llama-3.2-1b", "llama_1b", "1b"}:
        return config.LLAMA_1B
    if key in {"llama-3.2-3b", "llama_3b", "3b"}:
        return config.LLAMA_3B
    return config.LLAMA_3B


class EngineManager:
    def __init__(self) -> None:
        _ensure_dir(CACHE_DIR)
        _ensure_dir(MEMORY_DIR)

        self.dbm = DatabaseManager()
        self.dbm.ensure_dirs_exist()

        self.store = SessionStore(STATE_DB)
        self.embeddings = build_embeddings()

        self.active_mode = "full"
        self.papers_vs_cache: Dict[str, Chroma] = {}

        self.memory_vs = Chroma(
            collection_name="memory",
            persist_directory=MEMORY_DIR,
            embedding_function=self.embeddings,
        )

        self.runtime: Optional[ModelRuntime] = None
        self.active_model_key: str = ""

    def get_papers_vs(self, mode: str) -> Chroma:
        m = (mode or "full").strip().lower()
        if m not in self.papers_vs_cache:
            cfg = self.dbm.get_config(m) or self.dbm.get_config("full")
            if cfg is None:
                raise RuntimeError("No database config available")
            _ensure_dir(cfg.chroma_dir)
            self.papers_vs_cache[m] = Chroma(
                collection_name=cfg.collection,
                persist_directory=cfg.chroma_dir,
                embedding_function=self.embeddings,
            )
        return self.papers_vs_cache[m]

    def switch_mode(self, mode: str) -> None:
        m = (mode or "full").strip().lower()
        if m not in {"full", "abstracts"}:
            m = "full"
        self.active_mode = m
        self.dbm.switch_config(m)

    def switch_model(self, llm_model_key: str) -> None:
        key = (llm_model_key or "").strip().lower()
        if key == self.active_model_key and self.runtime is not None:
            return
        clear_runtime_cache()
        model_path = _resolve_llm_path(key)
        self.runtime = ModelRuntime(model_path)
        self.active_model_key = key

    def get_engine(self, session_id: str, mode: str) -> Engine:
        if self.runtime is None:
            self.switch_model("llama-3.2-3b")
        papers_vs = self.get_papers_vs(mode)
        return Engine(
            runtime=self.runtime,
            papers_vs=papers_vs,
            memory_vs=self.memory_vs,
            store=self.store,
            session_id=session_id,
        )


_GLOBAL_MANAGER: Optional[EngineManager] = None

def get_global_manager() -> EngineManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = EngineManager()
    return _GLOBAL_MANAGER
