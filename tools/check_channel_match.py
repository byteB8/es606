"""Do SparrKULee's array channels and our interpolated music channels use the same order?"""
import numpy as np, mne
from eg606.paths import derived_dir

bdf = ("Fp1 AF7 AF3 F1 F3 F5 F7 FT7 FC5 FC3 FC1 C1 C3 C5 T7 TP7 CP5 CP3 CP1 P1 P3 P5 P7 P9 PO7 "
       "PO3 O1 Iz Oz POz Pz CPz Fpz Fp2 AF8 AF4 AFz Fz F2 F4 F6 F8 FT8 FC6 FC4 FC2 FCz Cz C2 C4 "
       "C6 T8 TP8 CP6 CP4 CP2 P2 P4 P6 P8 P10 PO8 PO4 O2").split()
mne_order = mne.channels.make_standard_montage("biosemi64").ch_names
music = [str(c) for c in np.load(derived_dir("musin_g") / "sub-001_bs64.npz", allow_pickle=True)["ch_names"]]

print("BDF EEG channels:", len(bdf))
print("MNE biosemi64   :", len(mne_order))
print("music bs64 file :", len(music))
print("\nBDF == MNE biosemi64 order:", bdf == mne_order)
print("music == MNE biosemi64 order:", music == mne_order)
if bdf != mne_order:
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(bdf, mne_order)) if a != b]
    print("first mismatches:", diff[:8])
print("\n=> speech-trained weights can be applied to music:", bdf == mne_order == music)
