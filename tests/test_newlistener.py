"""Synthetic check of eval/newlistener: a planted, time-locked response must be found, and both
controls must fall to chance.

    PYTHONPATH=src python -m pytest tests/test_newlistener.py -q
"""
import numpy as np
from scipy.signal import lfilter

from eg606.eval.linear import zscore
from eg606.eval.newlistener import identify, run, window_scores

FS = 16.0
N_SUB, N_SONG, N_CH, SECONDS, DELAY = 6, 4, 8, 60, 2


def smooth_noise(rng, n, bands=1):
    return lfilter([1.0], [1.0, -0.8], rng.normal(size=(bands, n)), axis=1)


def synthetic(snr=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SECONDS * FS)
    targets = {s: {"one": smooth_noise(rng, n), "two": smooth_noise(rng, n, 2)}
               for s in range(1, N_SONG + 1)}
    cols = {"one": slice(0, 1), "two": slice(1, 3)}
    trials = []
    shared = rng.normal(size=(N_CH, 3))                # a common scalp projection ...
    for i in range(N_SUB):
        mix = shared + 0.3 * rng.normal(size=(N_CH, 3))  # ... that varies a little per listener
        for s in range(1, N_SONG + 1):
            src = np.concatenate([targets[s]["one"], targets[s]["two"]])        # (3, n)
            src = np.roll(src, DELAY, axis=1)          # the brain responds DELAY samples later
            eeg = snr * mix @ src + rng.normal(size=(N_CH, n))
            Y = np.concatenate([zscore(targets[s]["one"]), zscore(targets[s]["two"])]).T
            trials.append({"sub": f"sub-{i + 1:03d}", "song": s,
                           "eeg": zscore(eeg).astype(np.float32), "Y": Y})
    return trials, targets, cols


def test_window_scores_prefers_the_matching_candidate():
    rng = np.random.default_rng(1)
    C = rng.normal(size=(5, 200, 2))
    p = C[3] + 0.5 * rng.normal(size=(200, 2))
    assert int(np.argmax(window_scores(p, C))) == 3
    hits, total, guesses = identify(np.vstack([p, p]), np.concatenate([C, C], 1),
                                    [1, 2, 3, 4, 5], 4, 200, 400)
    assert (hits, total, guesses) == (2, 2, [4, 4])


def test_planted_response_is_found_and_the_control_is_at_chance():
    trials, targets, cols = synthetic()
    res = run(trials, targets, cols, FS, ["loso", "loso_song"], ["5", "all"], 5, 0)
    for split in ("loso", "loso_song"):
        for kind in cols:
            s = res["splits"][split]["summary"][kind]
            assert s["all"]["real"]["mean"] > 0.9, (split, kind, s["all"])
            assert s["5"]["real"]["mean"] > 0.6, (split, kind, s["5"])
            assert s["all"]["shift"]["mean"] < 0.45, (split, kind, s["all"])   # chance 0.25
    conf = np.array(res["splits"]["loso"]["confusion_all"]["one"])
    assert conf.sum() == N_SUB * N_SONG and np.trace(conf) > 0.9 * conf.sum()


def test_no_response_gives_chance():
    trials, targets, cols = synthetic(snr=0.0, seed=2)
    res = run(trials, targets, cols, FS, ["loso"], ["5"], 3, 0)
    acc = res["splits"]["loso"]["summary"]["one"]["5"]["real"]["mean"]
    assert abs(acc - 1 / N_SONG) < 0.12, acc
