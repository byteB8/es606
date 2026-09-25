"""Filesystem layout.

The same NAS is mounted at different paths on the two servers. The data root is EG606_DATA if
set, otherwise the first candidate that exists on this machine.
"""
import os
from pathlib import Path

_CANDIDATES = (
    "/mnt/nas/balbir/egdta",            # bhaskar
    "/mnt/nas_ramanujan/balbir/egdta",  # ramanujan
    os.path.expanduser("~/egdta"),      # Singularity HPC (no NAS mount; set EG606_DATA to override)
)


def _data_root() -> Path:
    if "EG606_DATA" in os.environ:
        return Path(os.environ["EG606_DATA"])
    for c in _CANDIDATES:
        if Path(c).is_dir():
            return Path(c)
    return Path(_CANDIDATES[0])


DATA_ROOT = _data_root()
RAW = DATA_ROOT / "raw"          # untouched downloads, one folder per dataset
DERIVED = DATA_ROOT / "derived"  # preprocessed caches, features, splits


def raw_dir(dataset: str) -> Path:
    return RAW / dataset


def derived_dir(dataset: str) -> Path:
    return DERIVED / dataset
