"""SparrKULee — KU Leuven RDR, doi:10.48804/K3VSND (CC-BY-NC-4.0).

Public files only. The 196 restricted raw files (5 subjects) need an access request and are
skipped; the preprocessed EEG of all 85 subjects is public.
"""
from __future__ import annotations

import json
import urllib.request

from .common import USER_AGENT, Remote

API = "https://rdr.kuleuven.be/api"
DOI = "doi:10.48804/K3VSND"

SUBSETS = {
    # 64 Hz preprocessed EEG + stimulus audio + reference envelopes + top-level metadata, ~30 GB
    "minimal": lambda p: (p.startswith(("derivatives/preprocessed_eeg/", "stimuli/"))
                          or p.endswith("_envelope.npy") or "/" not in p),
    "derivatives": lambda p: p.startswith("derivatives/") or "/" not in p,
    "raw": lambda p: p.startswith("sub-") or "/" not in p,
    "all": lambda p: True,
}
DEFAULT = "minimal"


def _list_files() -> list[dict]:
    url = f"{API}/datasets/:persistentId/versions/:latest/files?persistentId={DOI}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["data"]


def remotes(subset: str = DEFAULT) -> list[Remote]:
    keep = SUBSETS[subset]
    out = []
    for f in _list_files():
        if f.get("restricted"):
            continue
        df = f["dataFile"]
        name, size, query = df["filename"], df.get("filesize"), ""
        if df.get("originalFileFormat"):  # Dataverse-ingested table: fetch the original bytes
            name = df.get("originalFileName", name)
            size = df.get("originalFileSize", size)
            query = "?format=original"
        rel = "/".join(p for p in (f.get("directoryLabel") or "", name) if p)
        if keep(rel):
            out.append(Remote(f"{API}/access/datafile/{df['id']}{query}", rel, size))
    return out
