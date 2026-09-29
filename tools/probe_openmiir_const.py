"""OpenMIIR condition codes and onset semantics, from deepthought's own source."""
import urllib.request

UA = {"User-Agent": "eg606-probe"}
BASE = "https://raw.githubusercontent.com/sstober/deepthought/master/deepthought/datasets/openmiir/"


def get(rel, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(BASE + rel, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


for rel in ("constants.py", "events.py"):
    print("=" * 72)
    print(rel)
    print("=" * 72)
    print(get(rel)[:3000])
