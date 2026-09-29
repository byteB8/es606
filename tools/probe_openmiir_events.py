"""OpenMIIR trial coding, taken from the authors' own preprocessing notebook.

Guessing this would be a mistake: the prep has to cut perception trials out of a continuous
recording, so the event id -> (stimulus, condition) mapping must come from the source.
"""
import json
import re
import urllib.parse
import urllib.request

BASE = "https://raw.githubusercontent.com/sstober/openmiir/master/"
NB = "eeg/preprocessing/notebooks/Subject P01.ipynb"


def get(rel, timeout=180):
    req = urllib.request.Request(BASE + urllib.parse.quote(rel),
                                 headers={"User-Agent": "eg606-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


data = json.loads(get(NB))
print("cells:", len(data.get("cells", [])))
for i, cell in enumerate(data.get("cells", [])):
    src = "".join(cell.get("source", []))
    hit = re.search(r"event|trigger|stimulus_id|condition|trial|label", src, re.I)
    outs = []
    for out in cell.get("outputs", []):
        t = "".join(out.get("text", []))
        if re.search(r"event|trial|condition|stimulus", t, re.I):
            outs.append(t[:600])
    if hit or outs:
        print("=" * 70)
        print(f"[cell {i}]")
        print(src[:800])
        for t in outs:
            print("   OUT>", t)
