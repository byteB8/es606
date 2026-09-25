"""Read SparrKULee's preprocessing metadata to learn the channel selection and order."""
import json, urllib.request

API = "https://rdr.kuleuven.be/api"
UA = {"User-Agent": "eg606"}

raw = urllib.request.urlopen(
    urllib.request.Request(f"{API}/access/datafile/136121", headers=UA), timeout=180).read().decode(errors="replace")
print("bytes:", len(raw))
try:
    meta = json.loads(raw)
except json.JSONDecodeError:
    print(raw[:1500]); raise SystemExit

def walk(o, depth=0, path=""):
    if depth > 3:
        return
    if isinstance(o, dict):
        for k, v in list(o.items())[:12]:
            here = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                print("  " * depth + f"{k}: {type(v).__name__}({len(v)})")
                walk(v, depth + 1, here)
            else:
                s = str(v)
                print("  " * depth + f"{k}: {s[:110]}")
    elif isinstance(o, list) and o:
        print("  " * depth + f"[0] of {len(o)}:")
        walk(o[0], depth + 1, path)

walk(meta)
s = json.dumps(meta)
for kw in ("channel", "biosemi", "montage", "Fp1", "A1", "drop"):
    i = s.lower().find(kw.lower())
    if i != -1:
        print(f"\n--- context for '{kw}':\n{s[max(0,i-200):i+400]}")
