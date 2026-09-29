"""Where inside the full stimulus does the music actually start?

Two candidates: the length of the cue wav, or the authors' CUE_LENGTH table (which is shorter).
If the full wav is literally the cue file with music appended, its first samples equal the cue
file's and the boundary is unambiguous. Otherwise the metadata table is the only authority.
"""
import numpy as np

from eg606.audio.features import read_wav
from eg606.paths import raw_dir

root = raw_dir("openmiir") / "audio"
meta = raw_dir("openmiir") / "meta"

for version in ("v1", "v2"):
    cues = sorted((root / f"cues.{version}").glob("*.wav"))
    if not cues:
        continue
    print(f"=== {version}")
    for p in cues[:6]:
        tag = p.name.split("_")[0]
        f = next((root / f"full.{version}").glob(f"{tag}_*.wav"), None)
        if f is None:
            continue
        c, csr = read_wav(p)
        x, fsr = read_wav(f)
        n = min(len(c), len(x))
        head = x[:n]
        same = np.allclose(head, c[:n], atol=1e-4)
        r = float(np.corrcoef(head, c[:n])[0, 1]) if n > 10 else float("nan")
        # where does the cue's energy stop inside the full wav?
        win = int(0.01 * fsr)
        m = len(x) // win * win
        rms = np.sqrt((x[:m].reshape(-1, win) ** 2).mean(1))
        quiet = rms < 0.02 * rms.max()
        # last quiet stretch before the music: search after the cue clicks
        print(f"  {tag}: sr={fsr} cue_n={len(c)} full_n={len(x)}  head==cue? {same}  r={r:.4f}")

# the authors say the authoritative cue length lives in the metadata workbook
for v in ("v1", "v2"):
    x = meta / f"Stimuli_Meta.{v}.xlsx"
    print(f"--- {x.name}: exists={x.exists()} size={x.stat().st_size if x.exists() else 0}")
    if x.exists():
        try:
            import openpyxl
            wb = openpyxl.load_workbook(x, data_only=True)
            for ws in wb.worksheets:
                print("   sheet:", ws.title, ws.dimensions)
                for row in ws.iter_rows(values_only=True):
                    if any(c is not None for c in row):
                        print("    ", row)
        except ImportError:
            print("   openpyxl not installed")
