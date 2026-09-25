"""Fetch one SparrKULee channels.tsv from RDR to learn the channel order of the released arrays."""
import json
import urllib.request

API = "https://rdr.kuleuven.be/api"
DOI = "doi:10.48804/K3VSND"
UA = {"User-Agent": "eg606"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)


files = json.load(get(f"{API}/datasets/:persistentId/versions/:latest/files?persistentId={DOI}"))["data"]
cands = [f for f in files if "channels" in f["dataFile"]["filename"] and not f.get("restricted")]
print("channels.tsv files available:", len(cands))
if not cands:
    raise SystemExit("none found")

f = cands[0]
df = f["dataFile"]
q = "?format=original" if df.get("originalFileFormat") else ""
print("fetching:", f.get("directoryLabel"), df.get("originalFileName", df["filename"]))
text = get(f"{API}/access/datafile/{df['id']}{q}").read().decode(errors="replace")

lines = [l for l in text.splitlines() if l.strip()]
print("header:", lines[0])
names = [l.split("\t")[0] for l in lines[1:]]
print("n channels:", len(names))
print("order:", " ".join(names))
