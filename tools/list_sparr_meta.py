"""List small metadata files in the SparrKULee RDR listing (to find the channel order)."""
import json, urllib.request, collections

API = "https://rdr.kuleuven.be/api"
DOI = "doi:10.48804/K3VSND"
UA = {"User-Agent": "eg606"}
files = json.load(urllib.request.urlopen(
    urllib.request.Request(f"{API}/datasets/:persistentId/versions/:latest/files?persistentId={DOI}",
                           headers=UA), timeout=120))["data"]
ext = collections.Counter()
small = []
for f in files:
    df = f["dataFile"]
    name = df.get("originalFileName", df["filename"])
    ext[name.rsplit(".", 1)[-1]] += 1
    if name.endswith((".tsv", ".tab", ".json", ".txt", ".md")) and not f.get("restricted"):
        small.append((f.get("directoryLabel", ""), name, df["id"], df.get("filesize", 0)))
print("extensions:", ext.most_common(12))
print(f"\nsmall metadata files: {len(small)}")
for d, n, i, s in small[:25]:
    print(f"  {s:>9}  {d}/{n}  id={i}")
