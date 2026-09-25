"""Network smoke test: fetch the smallest file of each dataset into a temp dir and check its size.

    PYTHONPATH=src python3 tests/smoke_download.py
"""
import logging
import tempfile
from importlib import import_module
from pathlib import Path

from eg606.data.download.__main__ import DATASETS
from eg606.data.download.common import fetch

logging.basicConfig(level=logging.INFO)
with tempfile.TemporaryDirectory() as tmp:
    for name in DATASETS:
        mod = import_module(f"eg606.data.download.{name}")
        items = [i for i in mod.remotes(mod.DEFAULT) if i.size]
        smallest = min(items, key=lambda i: i.size)
        path = fetch(smallest, Path(tmp) / name)
        ok = path.stat().st_size == smallest.size
        print(f"{name:12s} {'OK ' if ok else 'BAD'} {smallest.size:>10d} B  {smallest.rel_path}")
