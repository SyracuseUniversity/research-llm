# cache_manager.py
import os
import gc
import shutil
import torch

CACHE_DIR = "cache"

def clear_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except Exception:
            pass
