import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import config_full as config


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
        self.register_config(
            "full",
            DatabaseConfig(
                mode="full",
                chroma_dir=config.CHROMA_DIR_FULL,
                collection="papers_all",
                description="Full-text papers with metadata",
            ),
        )
        self.register_config(
            "abstracts",
            DatabaseConfig(
                mode="abstracts",
                chroma_dir=config.CHROMA_DIR_ABSTRACTS,
                collection="abstracts_all",
                description="Abstracts only",
            ),
        )
        self.active_config_name = "full"

    def register_config(self, name: str, config_obj: DatabaseConfig):
        self.configs[name] = config_obj

    def switch_config(self, name: str) -> bool:
        if name in self.configs:
            self.active_config_name = name
            return True
        return False

    def get_active_config(self) -> Optional[DatabaseConfig]:
        return self.configs.get(self.active_config_name)

    def list_configs(self) -> List[str]:
        return list(self.configs.keys())

    def ensure_dirs_exist(self) -> None:
        for cfg in self.configs.values():
            if cfg and cfg.chroma_dir:
                os.makedirs(cfg.chroma_dir, exist_ok=True)
