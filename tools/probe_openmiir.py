"""What does the OpenMIIR release contain, and where are its files?"""
import json
import urllib.request

FIGSHARE = "https://api.figshare.com/v2/articles/1541151"
GH_TREE = "https://api.github.com/repos/sstober/openmiir/git/trees/master?recursive=1"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "eg606-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


d = get(FIGSHARE)
print("TITLE:", d.get("title"))
print("DESC:", (d.get("description") or "")[:1200].replace("<br>", "\n"))
print("\nFIGSHARE FILES:")
for f in d.get("files", []):
    print("  {:<46}{:10.1f} MB".format(f["name"], f["size"] / 1e6))
    print("      ", f.get("download_url"))

print("\nGITHUB TREE (selected):")
try:
    t = get(GH_TREE)
except Exception as e:
    print("  tree fetch failed:", e)
else:
    paths = [x["path"] for x in t.get("tree", [])]
    for p in paths:
        low = p.lower()
        if any(k in low for k in (".wav", ".mp3", ".mid", "stimul", "meta", ".csv", ".json", "readme")):
            print("  ", p)
    print("  ... total entries:", len(paths))
