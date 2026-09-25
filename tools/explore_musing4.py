"""Can we epoch from sourcedata using stm+/fxnd, and identify songs by duration?"""
import glob, os, wave, contextlib
import mne
from eg606.paths import raw_dir

root = raw_dir("musin_g")
wav = {}
for i in range(1, 13):
    with contextlib.closing(wave.open(str(root / "Code" / "ESongs" / f"{i}.esh.wav"))) as w:
        wav[i] = w.getnframes() / w.getframerate()

subs = sorted(os.path.basename(p) for p in glob.glob(str(root / "sourcedata" / "sub-*")))
print(f"{len(subs)} subjects in sourcedata\n")
print(f"{'sub':8}{'sfreq':>7}{'dur_min':>9}{'trials':>8}  song order (by duration match)   max_err_s")
allgood = True
for s in subs:
    evf = glob.glob(str(root / "sourcedata" / s / "eeg" / "*events.tsv"))
    setf = glob.glob(str(root / "sourcedata" / s / "eeg" / "*_eeg.set"))
    if not evf or not setf:
        print(f"{s:8} MISSING files"); allgood = False; continue
    raw = mne.io.read_raw_eeglab(setf[0], preload=False, verbose="ERROR")
    sfreq, dur = raw.info["sfreq"], raw.n_times / raw.info["sfreq"] / 60

    rows = [l.split("\t") for l in open(evf[0]).read().splitlines()[1:]]
    onsets = [(float(r[0]), r[-1]) for r in rows]
    starts = [o for o, v in onsets if v == "stm+"]
    ends = [o for o, v in onsets if v == "fxnd"]
    n = min(len(starts), len(ends))
    order, errs = [], []
    for st, en in zip(starts[:n], ends[:n]):
        d = en - st
        best = min(wav, key=lambda k: abs(d - wav[k]))
        order.append(best); errs.append(abs(d - wav[best]))
    ok = len(set(order)) == 12 and n == 12
    allgood &= ok
    print(f"{s:8}{sfreq:7.0f}{dur:9.1f}{n:8d}  {order}  {max(errs) if errs else -1:.2f} {'' if ok else '<-- CHECK'}")
print("\nall subjects give 12 unique songs:", allgood)
