"""Walk the nested MATLAB cell structure of ImageryData.mat for one participant."""
import numpy as np, scipy.io as sio
from eg606.paths import raw_dir

m = sio.loadmat(str(raw_dir("bach_silence") / "ImageryData.mat"), squeeze_me=False)
x = m["eeg"]
path = "eeg"
for depth in range(6):
    print(f"{path}: type={type(x).__name__} dtype={getattr(x, 'dtype', None)} shape={getattr(x, 'shape', None)}")
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = x.flat[0]
        path += "[0]"
    else:
        break
s = m["stim"]
while isinstance(s, np.ndarray) and s.dtype == object:
    s = s.flat[0]
print("stim leaf:", getattr(s, "shape", None), "nonzero:", int(np.count_nonzero(s)))
