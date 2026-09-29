"""Progress of an aria2 BitTorrent download, read from its .aria2 control file.

aria2 preallocates every file at full size, so the files on disk all look finished from the moment
the download starts. The control file carries a bitfield of completed pieces, which is the only
honest progress signal available from outside aria2 itself.

    python tools/aria2_progress.py /path/to/Something.aria2
"""
import struct
import sys
from pathlib import Path


def progress(path: Path) -> dict:
    b = path.read_bytes()
    o = 0

    def u(fmt, size):
        nonlocal o
        v = struct.unpack_from(fmt, b, o)[0]
        o += size
        return v

    version = u(">H", 2)
    u(">I", 4)                       # extension field
    ih_len = u(">I", 4)
    o += ih_len                      # info hash
    piece_length = u(">I", 4)
    total = u(">Q", 8)
    uploaded = u(">Q", 8)
    bf_len = u(">I", 4)
    bitfield = b[o:o + bf_len]
    done_pieces = sum(bin(x).count("1") for x in bitfield)
    n_pieces = (total + piece_length - 1) // piece_length
    return {"version": version, "piece_length": piece_length, "total": total,
            "uploaded": uploaded, "pieces": n_pieces, "done": done_pieces,
            "bytes_done": min(total, done_pieces * piece_length)}


for arg in sys.argv[1:] or ["."]:
    p = Path(arg)
    files = [p] if p.is_file() else sorted(p.rglob("*.aria2"))
    if not files:
        print(f"no .aria2 control file under {p} — the download has finished or has not started")
    for f in files:
        try:
            r = progress(f)
        except Exception as e:
            print(f"{f.name}: cannot read ({e})")
            continue
        pct = 100 * r["bytes_done"] / r["total"] if r["total"] else 0
        print(f"{f.stem}: {r['bytes_done'] / 1e9:5.2f} / {r['total'] / 1e9:5.2f} GB "
              f"({pct:5.1f}%)  {r['done']}/{r['pieces']} pieces  "
              f"uploaded {r['uploaded'] / 1e6:.0f} MB")
