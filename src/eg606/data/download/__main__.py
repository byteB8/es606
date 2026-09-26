"""Download datasets into the NAS raw folder.

    python -m eg606.data.download <dataset|all> [--subset NAME] [--list] [--status] [--workers N]

Each dataset module exposes SUBSETS, DEFAULT and remotes(subset) -> list[Remote].
"""
from __future__ import annotations

import argparse
import logging
import sys
from importlib import import_module

from eg606.paths import raw_dir

from .common import fetch_all

DATASETS = ("sparrkulee", "musin_g", "nmed_h", "nmed_t", "openmiir")


def _status(name: str, items) -> tuple[int, int, int]:
    """Return (expected bytes, bytes on disk, files complete) for one dataset."""
    root = raw_dir(name)
    expected = got = complete = 0
    for item in items:
        expected += item.size or 0
        path = root / item.rel_path
        part = path.with_name(path.name + ".part")
        if path.exists():
            size = path.stat().st_size
            got += size
            if item.size is None or size == item.size:
                complete += 1
        elif part.exists():
            got += part.stat().st_size
    return expected, got, complete


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.data.download")
    ap.add_argument("dataset", choices=(*DATASETS, "all"))
    ap.add_argument("--subset", help="dataset-specific subset (default: the module's DEFAULT)")
    ap.add_argument("--list", action="store_true", help="print files and sizes, download nothing")
    ap.add_argument("--status", action="store_true", help="report downloaded vs remaining, download nothing")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    failures = 0
    totals = [0, 0]
    if args.status:
        print(f"{'dataset':<12}{'expected':>10}{'on disk':>10}{'left':>10}   files")
    for name in names:
        mod = import_module(f"{__package__}.{name}")
        subset = args.subset if args.subset and args.dataset != "all" else mod.DEFAULT
        if subset not in mod.SUBSETS:
            ap.error(f"{name}: unknown subset {subset!r}; choose from {sorted(mod.SUBSETS)}")
        items = mod.remotes(subset)

        if args.status:
            expected, got, complete = _status(name, items)
            totals[0] += expected
            totals[1] += got
            print(f"{name:<12}{expected/1e9:9.1f}G{got/1e9:9.1f}G{(expected-got)/1e9:9.1f}G"
                  f"   {complete}/{len(items)}")
            continue

        logging.info("%s[%s]: %d files, %.2f GB", name, subset, len(items),
                     sum(i.size or 0 for i in items) / 1e9)
        if args.list:
            for i in items:
                print(f"{(i.size or 0) / 1e6:10.1f} MB  {i.rel_path}")
            continue
        failed = fetch_all(items, raw_dir(name), args.workers)
        for f in failed:
            logging.error("FAILED %s: %s", name, f.rel_path)
        failures += len(failed)

    if args.status:
        print(f"{'TOTAL':<12}{totals[0]/1e9:9.1f}G{totals[1]/1e9:9.1f}G"
              f"{(totals[0]-totals[1])/1e9:9.1f}G")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
