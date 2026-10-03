"""Can match-mismatch be won from the audio alone?

The derangement control showed that some decoders score above chance with mismatched EEG. If the
true excerpt and its imposter differ systematically in their audio, a scorer can prefer one of them
without any EEG at all. This replays the exact window and imposter rule used for evaluation and
scores each pair by audio statistics only, so a result above chance here is a property of the task.

    python tools/audio_only_baseline.py
"""
import numpy as np

from eg606.train.contrastive import FS, GUARD, load_bach, load_music


def pairs(n, window, guard, rng=None):
    """(true_start, imposter_start) exactly as match_mismatch_accuracy builds them."""
    for s in range(0, n - window + 1, window):
        later = True if rng is None else bool(rng.integers(2))
        imp = s + window + guard if later else s - window - guard
        if imp < 0 or imp + window > n:
            imp = s - window - guard if later else s + window + guard
            if imp < 0 or imp + window > n:
                continue
        yield s, imp


def score(recs, window_s, symmetric=False):
    win, rng = int(window_s * FS), (np.random.default_rng(7) if symmetric else None)
    rules = {"more onset energy": [], "higher variance": [], "earlier in trial": [],
             "first window of trial": []}
    per_trial = []
    for r in recs:
        y = r["y"][0]
        k = 0
        for s, imp in pairs(len(y), win, GUARD, rng):
            t, i = y[s:s + win], y[imp:imp + win]
            rules["more onset energy"].append(t.mean() > i.mean())
            rules["higher variance"].append(t.std() > i.std())
            rules["earlier in trial"].append(s < imp)
            rules["first window of trial"].append(s == 0)
            k += 1
        per_trial.append(k)
    out = {k: float(np.mean(v)) for k, v in rules.items()}
    out["windows per trial"] = float(np.mean(per_trial))
    return out


def load_openmiir():
    from eg606.eval.transfer import openmiir_trials
    from eg606.paths import derived_dir
    root = derived_dir("openmiir")
    feats = {v: dict(np.load(root / f"audio_features_{v}.npz")) for v in ("v1", "v2")}
    return [{"y": y[None]} for p in sorted(root.glob("sub-*.npz"))
            for _, _, y in openmiir_trials(p, feats, "onset")]


if __name__ == "__main__":
    for name, recs, wins in (("MUSIN-G", load_music("onset", shift=4), (5, 10, 30)),
                             ("Bach", load_bach("onset"), (5, 10)),
                             ("OpenMIIR", load_openmiir(), (2, 3))):
        for w in wins:
            for sym in (False, True):
                r = score(recs, w, sym)
                tag = "either side" if sym else "after"
                print(f"{name:8s} {w:>2d}s imposter {tag:11s} "
                      + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))
