"""Stanford Digital Repository helper: list a PURL's published files from its public XML."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

from .common import USER_AGENT, Remote

_FILE = re.compile(r"<file\b([^>]*)>")


def list_purl(druid: str) -> list[Remote]:
    req = urllib.request.Request(f"https://purl.stanford.edu/{druid}.xml",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        xml = r.read().decode()
    out = []
    for attrs in _FILE.findall(xml):
        name = re.search(r'\bid="([^"]+)"', attrs)
        size = re.search(r'\bsize="(\d+)"', attrs)
        published = re.search(r'\bpublish="(\w+)"', attrs)
        if not name or (published and published.group(1) != "yes"):
            continue
        fname = name.group(1)
        url = f"https://stacks.stanford.edu/file/druid:{druid}/{urllib.parse.quote(fname)}"
        out.append(Remote(url, fname, int(size.group(1)) if size else None))
    return out
