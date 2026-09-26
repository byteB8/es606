"""Does the published within-recording protocol inflate song identification on other datasets too?

eval/songid.py established the effect on MUSIN-G: a 12-way classifier reaches 55% within a
recording, still reaches 20% on the *silence before each song*, and collapses to chance across
listeners. One dataset is an anecdote, so the same audit -- same features, same classifier -- runs
here on three:

  MUSIN-G   12 songs, 20 listeners   silence control: 10 s before every song
  NMED-T    10 songs, 20 listeners   silence control recovered from the raw sessions
  NMED-H     4 stimuli per listener, each heard TWICE in separate blocks

The two controls attack the same question from opposite sides. Silence holds the *recording* and
removes the music: anything decodable there is block identity. NMED-H's repetition holds the
*music* and changes the block: anything lost there was block identity. A protocol that measures a
response to music should survive the second and fail the first.

Chance differs per dataset, and on NMED-H per listener (each heard their own four stimuli), so
every number is also reported as a multiple of its own chance level.

    python -m eg606.eval.leakage --dataset nmed_t
    python -m eg606.eval.leakage --dataset all --out results/e2/leakage.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eg606.eval.songid import TRAIN_FRACTION, band_features, clf
from eg606.paths import derived_dir

log = logging.getLogger(__name__)


@dataclass
class Windows:
    """Feature windows with the labels every split needs."""
    subject: np.ndarray
    label: np.ndarray        # what the classifier predicts: song, or stimulus on NMED-H
    song: np.ndarray         # song identity, ignoring the stimulus version
    rep: np.ndarray          # which hearing: "a" or "b"
    X: np.ndarray

    def chance(self, subject=None) -> float:
        lab = self.label if subject is None else self.label[self.subject == subject]
        return 1 / max(1, len(set(lab)))


def _stack(rows: list) -> Windows:
    cols = [np.array([r[i] for r in rows]) for i in range(4)]
    return Windows(*cols, np.stack([r[4] for r in rows]))


def load_musin_g(variant: str = "basic"):
    """MUSIN-G keeps song{NN} / pre{NN} keys; the pre-song silence is the control."""
    root = derived_dir("musin_g")
    suffix = "" if variant == "none" else f"_{variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    music, silence = [], []
    for p in sorted(x for x in root.glob("sub-*.npz") if pattern.match(x.stem)):
        d = np.load(p, allow_pickle=True)
        sub = p.stem.split("_")[0]
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            for f in band_features(d[f"song{song:02d}"].astype(np.float64)):
                music.append((sub, str(song), song, "a", f))
            for f in band_features(d[f"pre{song:02d}"].astype(np.float64)):
                silence.append((sub, str(song), song, "a", f))
    return _stack(music), (_stack(silence) if silence else None)


def load_nmed(name: str):
    """NMED-T/-H derived files: one meta entry per trial; NMED-H tags version and repetition."""
    root = derived_dir(name)
    music, silence = [], []
    for p in sorted(root.glob("sub-*.npz")):
        is_silence = p.stem.endswith("_silence")
        sub = p.stem.replace("_silence", "")
        d = np.load(p, allow_pickle=True)
        for m in json.loads(str(d["meta"])):
            label = m.get("stimulus") or str(m["song"])
            rep = m.get("rep") or "a"
            for f in band_features(d[m["key"]].astype(np.float64)):
                (silence if is_silence else music).append((sub, label, m["song"], rep, f))
    return _stack(music), (_stack(silence) if silence else None)


def fit_score(w: Windows, tr, te, labels=None, rng=None, shuffle=False) -> float | None:
    y = w.label if labels is None else labels
    ytr = rng.permutation(y[tr]) if shuffle else y[tr]
    if len(set(ytr)) < 2 or not len(te):
        return None
    return float((clf().fit(w.X[tr], ytr).predict(w.X[te]) == y[te]).mean())


def within_naive(w: Windows, rng, shuffle=False) -> dict:
    """The published protocol: random windows drawn from one listener's own recording."""
    out = {}
    for s in sorted(set(w.subject)):
        idx = np.where(w.subject == s)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * TRAIN_FRACTION)
        a = fit_score(w, idx[:cut], idx[cut:], rng=rng, shuffle=shuffle)
        if a is not None:
            out[s] = a
    return out


def pooled_naive(w: Windows, rng) -> dict:
    idx = rng.permutation(len(w.label))
    cut = int(len(idx) * TRAIN_FRACTION)
    model = clf().fit(w.X[idx[:cut]], w.label[idx[:cut]])
    te = idx[cut:]
    pred = model.predict(w.X[te])
    return {s: float((pred[w.subject[te] == s] == w.label[te][w.subject[te] == s]).mean())
            for s in sorted(set(w.subject[te]))}


def loso(w: Windows, labels=None) -> dict:
    out = {}
    for s in sorted(set(w.subject)):
        a = fit_score(w, w.subject != s, np.where(w.subject == s)[0], labels)
        if a is not None:
            out[s] = a
    return out


def cross_repetition(w: Windows) -> dict:
    """Same listener, same stimuli, different block: train on hearing a, test on hearing b."""
    out = {}
    for s in sorted(set(w.subject)):
        mine = w.subject == s
        tr = np.where(mine & (w.rep == "a"))[0]
        te = np.where(mine & (w.rep == "b"))[0]
        if len(tr) and len(te):
            a = fit_score(w, tr, te)
            if a is not None:
                out[s] = a
    return out


def conditions(music: Windows, silence: Windows | None, rng) -> dict:
    """Every split the dataset can support. A listener with one stimulus cannot be classified."""
    shared_labels = all(set(music.label[music.subject == s]) == set(music.label)
                        for s in set(music.subject))
    conds = {
        "within_naive": lambda: within_naive(music, rng),
        "within_naive_shuffled": lambda: within_naive(music, rng, shuffle=True),
    }
    if shared_labels:
        conds["pooled_naive"] = lambda: pooled_naive(music, rng)
        conds["loso"] = lambda: loso(music)
    else:
        # listeners hold different stimulus sets, so only song identity is comparable across them
        conds["loso_song"] = lambda: loso(music, music.song.astype(str))
    if len(set(music.rep)) > 1:
        conds["cross_repetition"] = lambda: cross_repetition(music)
    if silence is not None:
        conds["silence_naive"] = lambda: within_naive(silence, rng)
        conds["silence_loso"] = lambda: loso(silence)

    out = {}
    for name, fn in conds.items():
        r = fn()
        if r:
            out[name] = r
            log.info("  %-22s mean %.3f over %d listeners", name, np.mean(list(r.values())), len(r))
    return out


def summarise(music: Windows, conds: dict) -> dict:
    """Per-listener chance, since on NMED-H each listener has their own stimulus set."""
    per_listener = {s: music.chance(s) for s in sorted(set(music.subject))}
    return {"chance_pooled": music.chance(), "chance_per_listener": per_listener,
            "n_listeners": len(per_listener), "n_labels": len(set(music.label)),
            "conditions": conds}


LOADERS = {"musin_g": load_musin_g, "nmed_t": lambda: load_nmed("nmed_t"),
           "nmed_h": lambda: load_nmed("nmed_h")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.leakage")
    ap.add_argument("--dataset", default="all", help="musin_g, nmed_t, nmed_h, or all")
    ap.add_argument("--variant", default="basic", help="MUSIN-G cleaning variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    names = list(LOADERS) if args.dataset == "all" else args.dataset.split(",")
    results = {}
    for name in names:
        rng = np.random.default_rng(args.seed)
        log.info("%s ...", name)
        music, silence = (load_musin_g(args.variant) if name == "musin_g" else LOADERS[name]())
        log.info("  %d music windows, %d silence windows, %d labels, %d listeners",
                 len(music.label), 0 if silence is None else len(silence.label),
                 len(set(music.label)), len(set(music.subject)))
        results[name] = summarise(music, conditions(music, silence, rng))

    print("\n=== song identification under each protocol, three datasets")
    for name, r in results.items():
        ch = r["chance_per_listener"]
        print(f"\n{name}: {r['n_labels']} labels, {r['n_listeners']} listeners, "
              f"chance {np.mean(list(ch.values())):.3f}")
        print(f"  {'condition':<24}{'mean':>7}{'sd':>7}{'x chance':>10}  above chance")
        for cond, accs in r["conditions"].items():
            a = np.array(list(accs.values()))
            rel = np.mean([accs[s] / ch[s] for s in accs])
            print(f"  {cond:<24}{a.mean():7.3f}{a.std():7.3f}{rel:9.1f}x"
                  f"  {sum(accs[s] > ch[s] for s in accs)}/{len(a)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
