"""Read the channel labels from a SparrKULee raw BDF without downloading the whole file.

BDF (like EDF) stores a fixed header: 256 bytes, then 16 bytes per channel of labels. The files are
gzipped, so we stream the first megabyte and decompress incrementally.
"""
import json, urllib.request, zlib

API = "https://rdr.kuleuven.be/api"
DOI = "doi:10.48804/K3VSND"
UA = {"User-Agent": "eg606"}

files = json.load(urllib.request.urlopen(
    urllib.request.Request(f"{API}/datasets/:persistentId/versions/:latest/files?persistentId={DOI}",
                           headers=UA), timeout=120))["data"]
bdf = [f for f in files
       if f["dataFile"].get("originalFileName", f["dataFile"]["filename"]).endswith(".bdf.gz")
       and not f.get("restricted")]
print("public raw BDFs:", len(bdf))
f = bdf[0]
df = f["dataFile"]
name = df.get("originalFileName", df["filename"])
q = "?format=original" if df.get("originalFileFormat") else ""
url = f"{API}/access/datafile/{df['id']}{q}"
print("reading header of:", f.get("directoryLabel"), name)

req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-1048575"})
resp = urllib.request.urlopen(req, timeout=180)
print("http status:", resp.status, "(206 = range honoured)")
head = zlib.decompressobj(31).decompress(resp.read(), 400_000)
print("decompressed bytes:", len(head))

ns = int(head[252:256].decode("ascii", "replace").strip())
print("channels in file:", ns)
labels = [head[256 + 16 * i: 256 + 16 * (i + 1)].decode("ascii", "replace").strip() for i in range(ns)]
print("order:", " ".join(labels))
