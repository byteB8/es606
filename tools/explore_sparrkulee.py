"""What do SparrKULee's preprocessed EEG and stimulus files look like?"""
import glob, gzip, os
import numpy as np
from eg606.paths import raw_dir

root = raw_dir("sparrkulee")
print("top:", sorted(os.listdir(root))[:8])

eeg = sorted(glob.glob(str(root / "derivatives" / "preprocessed_eeg" / "sub-001" / "*" / "*.npy")))
print(f"\nsub-001 runs: {len(eeg)}")
for f in eeg[:4]:
    print("   ", os.path.basename(f))
x = np.load(eeg[0])
print("first array:", x.shape, x.dtype, "| std", float(np.nanstd(x)))

subs = sorted({os.path.basename(p) for p in glob.glob(str(root / "derivatives" / "preprocessed_eeg" / "sub-*"))})
print(f"\nsubjects with preprocessed EEG: {len(subs)}")
allf = glob.glob(str(root / "derivatives" / "preprocessed_eeg" / "sub-*" / "*" / "*.npy"))
print("total runs:", len(allf))

stim = sorted(glob.glob(str(root / "stimuli" / "eeg" / "*.npz.gz")))
print(f"\nstimuli: {len(stim)}; e.g. {[os.path.basename(s) for s in stim[:4]]}")
with gzip.open(stim[0]) as fh:
    z = np.load(fh, allow_pickle=True)
    print("keys:", list(z.keys()))
    for k in list(z.keys())[:4]:
        v = z[k]
        print(f"   {k}: shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)} "
              f"{v if getattr(v, 'size', 9) < 4 else ''}")

env = sorted(glob.glob(str(root / "derivatives" / "preprocessed_stimuli" / "*_envelope.npy")))
print(f"\ntheir envelopes: {len(env)}")
if env:
    e = np.load(env[0])
    print("  ", os.path.basename(env[0]), e.shape, e.dtype)
