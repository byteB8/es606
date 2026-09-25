"""MUSIN-G — OpenNeuro ds003774, listed and fetched from the public OpenNeuro S3 bucket."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

from .common import USER_AGENT, Remote

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds003774"

SUBSETS = {
    "all": lambda p: True,                                           # ~24.6 GB
    "preproc": lambda p: not p.startswith("sourcedata/"),            # per-song .set + wavs, ~10.9 GB
    "raw": lambda p: p.startswith("sourcedata/") or "/" not in p,    # continuous raw .set, ~13.7 GB
}
DEFAULT = "all"

_ENTRY = re.compile(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", re.S)


def _list_keys() -> list[tuple[str, int]]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": f"{DATASET}/", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        req = urllib.request.Request(f"{BUCKET}?{urllib.parse.urlencode(q)}",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            xml = r.read().decode()
        keys += [(k, int(s)) for k, s in _ENTRY.findall(xml)]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def remotes(subset: str = DEFAULT) -> list[Remote]:
    keep = SUBSETS[subset]
    out = []
    for key, size in _list_keys():
        rel = key[len(DATASET) + 1:]
        if not rel or rel.startswith(".datalad/") or not keep(rel):
            continue
        out.append(Remote(f"{BUCKET}/{urllib.parse.quote(key)}", rel, size))
    return out
