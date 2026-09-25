#!/usr/bin/env python
"""Generic entry point: run a job described by a JSON config.

    python run.py --config configs/j01.json

The config names a module and its arguments, e.g.
    {"module": "eg606.eval.protocol", "args": ["--variant", "none", "--feature", "onset"]}
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
opts = ap.parse_args()

cfg = json.loads(Path(opts.config).read_text())
sys.argv = [cfg["module"], *cfg.get("args", [])]
sys.path.insert(0, str(Path(__file__).parent / "src"))
runpy.run_module(cfg["module"], run_name="__main__")
