"""What does the Academic Torrents entry offer, and can we fetch it over HTTP?"""
import re
import socket
import urllib.request

socket.setdefaulttimeout(40)
HASH = "c18c04a9f18ff7d133421012978c4a92f57f6b9c"
page = "https://academictorrents.com/details/" + HASH


def get(url, n=200000):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 eg606"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.status, r.read(n)


status, raw = get(page)
body = raw.decode("utf-8", "replace")
print("details page:", status, len(body), "bytes")
hrefs = sorted(set(re.findall(r'href="([^"]+)"', body)))
for h in hrefs:
    if any(k in h.lower() for k in ("download", "collection", ".fif", "browse", "mirror", "magnet")):
        print("  link:", h[:160])
# file table
for m in re.findall(r"(P\d\d-raw\.fif)", body):
    print("  file:", m)
print("\ntrying the HTTP file-download endpoints:")
for u in (f"https://academictorrents.com/download/{HASH}.torrent",
          f"https://academictorrents.com/collection/{HASH}"):
    try:
        s, b = get(u, 4000)
        print(f"  {u} -> {s}, {len(b)} bytes, head={b[:60]!r}")
    except Exception as e:
        print(f"  {u} -> FAIL {e}")
