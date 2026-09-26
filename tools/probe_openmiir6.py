"""OpenMIIR event coding, from the authors' own preprocessing notebook."""
import json
import re
import socket
import urllib.request

socket.setdefaulttimeout(60)
TREE = "https://api.github.com/repos/sstober/openmiir/git/trees/master?recursive=1"
RAW = "https://raw.githubusercontent.com/sstober/openmiir/master/"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "eg606-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


tree = json.loads(get(TREE))["tree"]
nb = [x["path"] for x in tree if x["path"].endswith(".ipynb")]
print("notebooks:", nb[:6])
if nb:
    data = json.loads(get(RAW + urllib.request.quote(nb[0])))
    print("=" * 60, "\n", nb[0])
    for cell in data.get("cells", [])[:40]:
        src = "".join(cell.get("source", []))
        if re.search(r"event|trigger|stimulus_id|condition|trial", src, re.I):
            print("-" * 50)
            print(src[:700])
        for out in cell.get("outputs", []):
            t = "".join(out.get("text", []))
            if re.search(r"event|trigger|condition", t, re.I):
                print("   OUT:", t[:500])
