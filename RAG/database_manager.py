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
                collection=getattr(config, "CHROMA_COLLECTION_FULL", getattr(config, "CHROMA_COLLECTION", "papers_all")),
                description="Full text papers with metadata",
            ),
        )
        self.active_config_name = "full"

    def register_config(self, name: str, cfg: DatabaseConfig) -> None:
        self.configs[name] = cfg

    def resolve_mode(self, requested_mode: str) -> str:
        available_modes = self.list_configs()
        if not available_modes:
            return ""
        req = (requested_mode or "").strip()
        if req in self.configs:
            return req
        req_l = req.lower()
        by_lower = {k.lower(): k for k in self.configs.keys()}
        if req_l in by_lower:
            return by_lower[req_l]
        return available_modes[0]

    def switch_config(self, name: str) -> bool:
        target = self.resolve_mode(name)
        if target:
            self.active_config_name = target
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
