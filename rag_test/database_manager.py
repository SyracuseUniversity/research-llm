#database_manager.py
import os
from dataclasses import dataclass
from enum import Enum
from typing import List

class DatabaseMode(Enum):
    FULL = "full"
    ABSTRACTS = "abstracts"

@dataclass
class DatabaseConfig:
    mode: str
    chroma_dir: str
    collection: str
    description: str

class DatabaseManager:
    def __init__(self):
        self.configs = {}
        self.active_config_name = None
        self._load_defaults()

    def _load_defaults(self):
        self.register_config("full", DatabaseConfig(
            mode="full",
            chroma_dir=r"C:\codes\t5-db\chroma_store_full",
            collection="papers_all",
            description="Full-text papers with metadata"
        ))
        self.register_config("abstracts", DatabaseConfig(
            mode="abstracts",
            chroma_dir=r"C:\codes\t5-db\chroma_store_abstracts",
            collection="abstracts_all",
            description="Abstracts only — faster"
        ))
        self.active_config_name = "full"

    def register_config(self, name, config): self.configs[name] = config
    def switch_config(self, name):
        if name in self.configs:
            self.active_config_name = name; return True
        return False
    def get_active_config(self): return self.configs.get(self.active_config_name)
    def list_configs(self) -> List[str]: return list(self.configs.keys())
