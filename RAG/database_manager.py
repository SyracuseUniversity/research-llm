# database_manager.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import config_full as config


@dataclass
class DatabaseConfig:
    mode: str
    chroma_dir: str
    collection: str
    description: str


class DatabaseManager:
    def __init__(self) -> None:
        self.configs: Dict[str, DatabaseConfig] = {}
        self.active_config_name: str = "full"
        self._load_defaults()

    def _load_defaults(self) -> None:
        self.register_config(
            "full",
            DatabaseConfig(
                mode="full",
                chroma_dir=config.CHROMA_DIR_FULL,
                collection=getattr(config, "CHROMA_COLLECTION_FULL", "papers_all"),
                description="Full text papers with metadata",
            ),
        )
        self.register_config(
            "abstracts",
            DatabaseConfig(
                mode="abstracts",
                chroma_dir=getattr(config, "CHROMA_DIR_ABSTRACTS", config.CHROMA_DIR_FULL),
                collection=getattr(config, "CHROMA_COLLECTION_ABSTRACTS", "abstracts_all"),
                description="Abstracts only",
            ),
        )
        self.active_config_name = "full"

    def register_config(self, name: str, cfg: DatabaseConfig) -> None:
        self.configs[name] = cfg

    def switch_config(self, name: str) -> bool:
        if name in self.configs:
            self.active_config_name = name
            return True
        return False

    def get_active_config(self) -> Optional[DatabaseConfig]:
        return self.configs.get(self.active_config_name)

    def get_config(self, name: str) -> Optional[DatabaseConfig]:
        return self.configs.get(name)

    def list_configs(self) -> List[str]:
        return list(self.configs.keys())

    def ensure_dirs_exist(self) -> None:
        for cfg in self.configs.values():
            if cfg and cfg.chroma_dir:
                os.makedirs(cfg.chroma_dir, exist_ok=True)
