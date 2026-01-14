# runtime_settings.py
from dataclasses import dataclass

@dataclass
class RuntimeSettings:
    active_mode: str = "full"
    llm_model: str = "llama-3.2-3b"
    use_graph: bool = False

settings = RuntimeSettings()
