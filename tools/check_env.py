"""Report which packages an environment has, for the runner setup."""
import importlib
for m in ["numpy", "scipy", "mne", "torch", "sklearn", "pandas", "h5py", "onnxruntime", "mne_icalabel"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14} {getattr(mod, '__version__', '?')}")
    except Exception:
        print(f"  {m:14} MISSING")
import torch
print("torch cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
