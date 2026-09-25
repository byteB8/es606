"""NMED-H — Stanford purl sd922db3535 (CC-BY). ~21 GB without the single-file examples."""
from __future__ import annotations

from .common import Remote
from .stanford import list_purl

DRUID = "sd922db3535"
# Standalone copies of files that are also inside the zip archives.
_EXAMPLES = {"S1_1_raw.mat", "song21_a_Imputed.mat", "rcaOut_orig.mat"}

SUBSETS = {
    "all": lambda n: n not in _EXAMPLES,
    "clean": lambda n: n.startswith("CleanEEG_") or n.endswith((".pdf",)) or n.startswith(("behaveStruct", "stimAssignment")),
    "raw": lambda n: n.startswith("RawEEG_") or n.endswith((".pdf",)) or n.startswith(("behaveStruct", "stimAssignment")),
}
DEFAULT = "all"


def remotes(subset: str = DEFAULT) -> list[Remote]:
    keep = SUBSETS[subset]
    return [r for r in list_purl(DRUID) if keep(r.rel_path)]
