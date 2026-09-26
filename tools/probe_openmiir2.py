"""Where does the OpenMIIR EEG itself live, and how are trials coded?"""
import urllib.request

BASE = "https://raw.githubusercontent.com/sstober/openmiir/master/"
for rel in ("README.md", "eeg/README.md", "eeg/mne/README.md", "eeg/preprocessing/README.md",
            "audio/README.md", "meta/README.md"):
    print("=" * 70)
    print(rel)
    print("=" * 70)
    try:
        req = urllib.request.Request(BASE + rel, headers={"User-Agent": "eg606-probe"})
        with urllib.request.urlopen(req, timeout=45) as r:
            print(r.read().decode("utf-8", "replace")[:3000])
    except Exception as e:
        print("  (not available:", e, ")")
