"""NMED-T — Stanford purl jn859kj8079 (CC-BY). ~39.7 GB; stimulus audio is not distributed."""
from __future__ import annotations

from .common import Remote
from .stanford import list_purl

DRUID = "jn859kj8079"

_META = ("behavioralRatings.mat", "participantInfo.mat", "Code.zip", "TapIt.zip", "NMED-T_README.pdf")
SUBSETS = {
    "all": lambda n: True,
    "clean": lambda n: n.startswith("song") or n in _META,   # per-song cleaned EEG, 125 Hz, ~6.9 GB
    "raw": lambda n: n.endswith("_raw.mat") or n in _META,   # per-participant raw, 1 kHz, ~32.8 GB
}
DEFAULT = "all"


def remotes(subset: str = DEFAULT) -> list[Remote]:
    keep = SUBSETS[subset]
    return [r for r in list_purl(DRUID) if keep(r.rel_path)]
