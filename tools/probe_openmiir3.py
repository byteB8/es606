"""Are the OpenMIIR mirrors alive, and how are trials coded?"""
import re
import urllib.request

SITES = ["http://www.ling.uni-potsdam.de/mlcog/OpenMIIR-RawEEG_v1/",
         "http://bmi.ssc.uwo.ca/OpenMIIR-RawEEG_v1/"]


def fetch(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "eg606-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


for s in SITES:
    print("=" * 70)
    print(s)
    try:
        status, body = fetch(s)
        links = re.findall(r'href="([^"]+)"', body)
        fifs = [l for l in links if l.lower().endswith(".fif")]
        print(f"  status {status}, {len(links)} links, {len(fifs)} .fif")
        for f in fifs[:14]:
            print("   ", f)
    except Exception as e:
        print("  unreachable:", e)

print("=" * 70)
print("trigger coding, from the presentation script")
try:
    _, m = fetch("https://raw.githubusercontent.com/sstober/openmiir/master/"
                 "scripts/presentation/OpenMIIR_StimulusPresentation.m")
    keep = [l for l in m.splitlines()
            if re.search(r"trigger|condition|stimulus_id|port|marker|code", l, re.I)]
    print("\n".join(keep[:60]))
except Exception as e:
    print("  failed:", e)
