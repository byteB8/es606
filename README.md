# eg606 — decoding naturalistic music from EEG, across listeners

Can we tell *which music someone is hearing*, and *when*, from their EEG — for a listener the model
has never seen? This repository contains the full pipeline: dataset acquisition, preprocessing,
linear baselines, a contrastive EEG↔audio model, and the evaluation protocols.

Course project for ES 606 (Computational Neuroscience), IIT Gandhinagar.

## Why the task is posed this way

Published EEG music-identification results are usually obtained by splitting *one continuous
recording* into training and test windows. Reproducing that protocol here shows what it measures:

| Protocol (same data, same features, same classifier) | 12-way song ID (chance 8.3%) |
|---|---|
| within-listener, random windows — the usual protocol | **55.1%** |
| the same, on the **10 s of silence before each song** | **20.0%** |
| leave-one-listener-out | **8.3%** (chance) |

A classifier "identifies" songs from silence, where no music is playing: the protocol reads block
identity, not the response to music. So this project uses a **time-locked match–mismatch** task —
given a window of EEG and two candidate audio excerpts (the true one and an imposter from the same
song one second later), decide which produced it. Both candidates share the recording, so slow drift
cannot separate them. Chance is 50%.

Under that formulation, with all training listeners held out:

| | 5 s | 10 s | 30 s |
|---|---|---|---|
| linear decoder (mTRF-style), new listener | 0.576 | 0.608 | 0.700 |
| linear decoder, **new song** | 0.575 | 0.600 | 0.697 |
| contrastive model, new listener (5-fold CV) | 0.584 | 0.599 | 0.688 |

The signal lives in 4–8 Hz (theta), note onsets beat the loudness envelope, and light cleaning
(bad-channel interpolation) helps while ICA-based artifact removal hurts.

## Cross-dataset timing matters

A decoder trained on **speech** listening transfers to music — but only when the two datasets agree
about when the sound began. One dataset's event markers lag its audio by ~62 ms; uncorrected, a
speech-trained decoder scores *below* chance on it (0.414), because in theta a 62 ms offset is about
a third of a cycle. Correcting it recovers 0.664, and the correction helps only models carrying
knowledge from the other dataset (+3.3 points, p=0.0007) and not models trained on that data alone
(no effect) — the signature of a timing offset rather than a modelling artefact.

## Layout

```
src/eg606/
  data/download/   dataset downloaders (stdlib only), resumable and size-verified
  data/import_local.py  install datasets that block scripted download (Dryad)
  prep/            preprocessing per dataset -> 64 Hz epochs aligned to audio features
  audio/           envelope and onset (spectral flux) features, one definition for all datasets
  eval/            protocol comparison, match-mismatch, transfer, latency analysis, song ID
  models/          contrastive EEG-audio model; VLAAI zero-shot wrapper
  train/           training loop and cross-validation (by listener, by song, or both)
scripts/           server sync, Slurm templates, publishing
tools/             one-off inspection scripts, kept because they document how facts were established
configs/           job configurations
```

## Getting the data

```bash
python -m eg606.data.download all --status     # what is present and what is missing
python -m eg606.data.download musin_g          # resumable; rerun to continue
python -m eg606.data.import_local --list       # datasets needing a browser download
```

| Dataset | Content | Audio |
|---|---|---|
| [MUSIN-G](https://openneuro.org/datasets/ds003774) | 20 listeners, 12 songs, 128-ch EGI | included |
| [SparrKULee](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND) | 85 listeners, 168 h speech, 64-ch BioSemi | included |
| [Bach: music of silence](https://datadryad.org/dataset/doi:10.5061/dryad.dbrv15f0j) | 21 musicians, 4 melodies × 11 repeats | included |
| [NMED-T](https://purl.stanford.edu/jn859kj8079) / [NMED-H](https://purl.stanford.edu/sd922db3535) | naturalistic music EEG | metadata only |

Set `EG606_DATA` to choose where data lives (default: the NAS path in `eg606/paths.py`).

## Running the pipeline

```bash
python -m eg606.prep.musin_g --subjects all --clean basic          # preprocess
python -m eg606.prep.musin_g --subjects all --to-montage biosemi64 # shared montage for transfer
python -m eg606.prep.sparrkulee --subjects all
python -m eg606.prep.bach

python -m eg606.eval.protocol --variant basic --feature onset --splits naive,loso,song
python -m eg606.eval.songid --variant basic                        # the leakage figure
python -m eg606.eval.latency --band 4,8                            # response-latency analysis
python -m eg606.train.cv --dataset music --model v2 --cv 5 --cv-by song
```

`scripts/sync.sh` mirrors the code to compute servers and submits jobs (`push`, `sbatch`, `status`);
copy `scripts/servers.env.example` to `scripts/servers.env` with your own machines first.

## Requirements

Python 3.10+, NumPy/SciPy, MNE-Python, scikit-learn, PyTorch (CUDA optional), ONNX Runtime for the
VLAAI baseline. `pip install -e ".[prep,audio,train]"`.

## Credits

Datasets are the work of their authors (cited above); please follow their licences. VLAAI weights
come from [exporl/vlaai](https://github.com/exporl/vlaai). The match–mismatch formulation follows the
auditory-EEG literature (Accou et al.; the ICASSP Auditory EEG Challenges).
