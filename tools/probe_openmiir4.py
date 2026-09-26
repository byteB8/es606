"""Can the OpenMIIR raw EEG still be downloaded anywhere?"""
import json
import socket
import urllib.request

socket.setdefaulttimeout(30)
CANDIDATES = [
    "http://www.ling.uni-potsdam.de/mlcog/",
    "http://www.ling.uni-potsdam.de/mlcog/OpenMIIR-RawEEG_v1",
    "https://www.ling.uni-potsdam.de/mlcog/OpenMIIR-RawEEG_v1/",
    "http://bmi.ssc.uwo.ca/",
    "https://academictorrents.com/details/c18c04a9f18ff7d133421012978c4a92f57f6b9c",
    "https://academictorrents.com/apiv2/torrents/details?info_hash=c18c04a9f18ff7d133421012978c4a92f57f6b9c",
    "https://openneuro.org/crn/datasets?q=openmiir",
]
for url in CANDIDATES:
    print("-" * 70)
    print(url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 eg606"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read(4000).decode("utf-8", "replace")
        print(f"  status {r.status}, {len(body)} bytes of head")
        low = body.lower()
        for key in ("openmiir", ".fif", "raweeg", "download"):
            if key in low:
                i = low.index(key)
                print(f"   ...{body[max(0, i - 90):i + 130]!r}")
    except Exception as e:
        print("  FAIL:", type(e).__name__, e)
