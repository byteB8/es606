"""OpenMIIR (Stober et al. 2015) — 10 listeners, 12 music fragments, BioSemi 64.

The release is split in two, and only one half can be fetched over HTTP:

  * **stimuli and metadata** live in the GitHub repository (~70 MB) and are fetched here. There are
    two stimulus versions, v1 for listeners P01-P08 and v2 for P09-P14, and the `full` variants
    include the cue clicks that precede each fragment.
  * **the raw EEG** (10 x ~700 MB FIF files, 7.0 GB) is *not* available over HTTP. Both mirrors
    named in the repository are dead (Potsdam returns 404, Western Ontario times out) and the
    Academic Torrents copy declares an empty web-seed list, so BitTorrent is the only route left.
    `--list` prints what is missing and where it would come from; `eeg_torrent_url()` returns the
    torrent so a client can be pointed at it.

    python -m eg606.data.download openmiir            # stimuli and metadata
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .common import USER_AGENT, Remote

REPO = "sstober/openmiir"
BRANCH = "master"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"

TORRENT = ("https://academictorrents.com/download/"
           "c18c04a9f18ff7d133421012978c4a92f57f6b9c.torrent")
EEG_SUBJECTS = ("P01", "P04", "P05", "P06", "P07", "P09", "P11", "P12", "P13", "P14")

KEEP_PREFIXES = ("audio/", "meta/")
KEEP_SUFFIXES = (".wav", ".txt", ".xlsx", ".md")

SUBSETS = {
    "stimuli": lambda p: p.startswith(KEEP_PREFIXES) and p.endswith(KEEP_SUFFIXES),
    "all": lambda p: p.startswith(KEEP_PREFIXES) and p.endswith(KEEP_SUFFIXES),
}
DEFAULT = "stimuli"


def eeg_torrent_url() -> str:
    """Where the raw EEG has to come from; see the module docstring for why."""
    return TORRENT


def _tree() -> list[tuple[str, int]]:
    req = urllib.request.Request(TREE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return [(x["path"], int(x.get("size") or 0)) for x in data["tree"] if x["type"] == "blob"]


def remotes(subset: str = DEFAULT) -> list[Remote]:
    keep = SUBSETS[subset]
    out = []
    for path, size in _tree():
        if not keep(path):
            continue
        # a GitHub path may contain spaces; quote it for the URL but keep it verbatim on disk
        out.append(Remote(url=RAW + urllib.parse.quote(path), rel_path=path, size=size))
    return out
