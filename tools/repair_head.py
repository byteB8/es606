"""Repair downloads whose first chunk is an HTML error page.

The resumable downloader appends from the current file size, so a server error page written as
chunk 0 leaves a file of exactly the right length whose head is HTML and whose tail is good data.
Find where the HTML ends and re-fetch just those bytes.

    python tools/repair_head.py nmed_t            # report
    python tools/repair_head.py nmed_t --fix
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from eg606.paths import raw_dir

MAGIC = (b"MATLAB", b"\x89HDF\r\n\x1a\n", b"PK\x03\x04", b"%PDF", b"\x1f\x8b")
PROBE = 1 << 20


def html_len(path: Path) -> int | None:
    """Length of a leading HTML error page, or None if the head looks like real data."""
    with open(path, "rb") as fh:
        head = fh.read(PROBE)
    if any(head.startswith(m) for m in MAGIC):
        return None
    low = head.lower()
    if not low.lstrip().startswith(b"<!doctype") and b"<html" not in low[:512]:
        return None
    end = low.rfind(b"</html>")
    if end < 0:
        raise SystemExit(f"{path.name}: HTML head with no </html> within {PROBE} bytes")
    end += len(b"</html>")
    while end < len(head) and head[end] in b"\r\n":
        end += 1
    return end


def refetch(url: str, path: Path, n: int) -> None:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise SystemExit(f"{path.name}: server ignored the range request (status {r.status})")
        good = r.read(n)
    if len(good) != n:
        raise SystemExit(f"{path.name}: got {len(good)} bytes, wanted {n}")
    with open(path, "r+b") as fh:
        fh.write(good)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    root = raw_dir(args.dataset)
    manifest = {m["rel_path"]: m for m in json.loads((root / "_manifest.json").read_text())}
    bad = 0
    for rel, m in manifest.items():
        path = root / rel
        if not path.exists():
            continue
        if path.stat().st_size != m["size"]:
            print(f"  {rel}: size {path.stat().st_size} != {m['size']} (incomplete, re-download)")
            bad += 1
            continue
        n = html_len(path)
        if n is None:
            continue
        bad += 1
        print(f"  {rel}: {n} bytes of HTML at the head", end="", flush=True)
        if args.fix:
            refetch(m["url"], path, n)
            print(" -> repaired" if html_len(path) is None else " -> STILL BAD")
        else:
            print()
    print(f"{args.dataset}: {bad} damaged file(s)" + ("" if args.fix else "; rerun with --fix"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
