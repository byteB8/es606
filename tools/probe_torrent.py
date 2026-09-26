"""Does the OpenMIIR torrent carry HTTP web seeds? If so no BitTorrent client is needed."""
import socket
import urllib.request

socket.setdefaulttimeout(60)
URL = "https://academictorrents.com/download/c18c04a9f18ff7d133421012978c4a92f57f6b9c.torrent"


def bdecode(data, i=0):
    c = data[i:i + 1]
    if c == b"i":
        j = data.index(b"e", i)
        return int(data[i + 1:j]), j + 1
    if c == b"l":
        out, i = [], i + 1
        while data[i:i + 1] != b"e":
            v, i = bdecode(data, i)
            out.append(v)
        return out, i + 1
    if c == b"d":
        out, i = {}, i + 1
        while data[i:i + 1] != b"e":
            k, i = bdecode(data, i)
            v, i = bdecode(data, i)
            out[k] = v
        return out, i + 1
    j = data.index(b":", i)
    n = int(data[i:j])
    return data[j + 1:j + 1 + n], j + 1 + n


req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 eg606"})
with urllib.request.urlopen(req, timeout=60) as r:
    raw = r.read()
print("torrent:", len(raw), "bytes")
meta, _ = bdecode(raw)
for k in meta:
    if k != b"info":
        v = meta[k]
        print(" ", k.decode(), "=", str(v)[:300])
info = meta[b"info"]
print("  name:", info.get(b"name"))
print("  piece length:", info.get(b"piece length"))
total = 0
for f in info.get(b"files", []):
    path = b"/".join(f[b"path"]).decode()
    total += f[b"length"]
    print(f"    {path:<24}{f[b'length'] / 1e6:9.1f} MB")
print(f"  total {total / 1e9:.2f} GB")
seeds = meta.get(b"url-list") or meta.get(b"httpseeds")
print("\nWEB SEEDS:", seeds if seeds else "NONE — a BitTorrent client is required")
