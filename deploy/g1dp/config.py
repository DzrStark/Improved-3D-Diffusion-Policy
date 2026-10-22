import json
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    base = (path.parent / cfg.pop("path_base", ".")).resolve()
    for key in ("training_root", "checkpoint", "zarr", "raw_root", "urdf"):
        if cfg.get(key):
            cfg[key] = str((base / cfg[key]).resolve())
    return cfg


def seed_once(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
