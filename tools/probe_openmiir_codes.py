"""What do OpenMIIR's special event codes (1000, 1111, 2001) mean? Find the authors' definition."""
import json
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "eg606-probe"}


def get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


for repo in ("sstober/openmiir", "sstober/deepthought"):
    try:
        tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"))
    except Exception as e:
        print(repo, "tree failed:", e)
        continue
    py = [x["path"] for x in tree["tree"]
          if x["path"].endswith(".py") and re.search(r"preproc|event|openmiir|pipeline|trial", x["path"], re.I)]
    print("=" * 70)
    print(repo, "->", len(py), "candidate files")
    for p in py[:12]:
        print("  ", p)
        try:
            src = get(f"https://raw.githubusercontent.com/{repo}/master/{urllib.parse.quote(p)}").decode("utf-8", "replace")
        except Exception as e:
            print("      fetch failed:", e)
            continue
        for m in re.finditer(r"^.*(2001|1111|1000).*$", src, re.M):
            line = m.group(0).strip()
            if len(line) < 160 and not line.startswith("#"):
                print("       >", line)
