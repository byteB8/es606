"""Resumable, size-verified HTTP downloads (stdlib only)."""
from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "eg606-downloader/0.1 (academic research, ES606 course project)"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Remote:
    url: str
    rel_path: str            # destination, relative to the dataset's raw folder
    size: int | None = None  # expected bytes; enables skip-if-complete and verification


def _open(url: str, start: int = 0, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if start:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=timeout)


def fetch(item: Remote, root: Path, retries: int = 6) -> Path:
    """Download one file, resuming a partial `.part` file when the server supports ranges."""
    dest = root / item.rel_path
    if dest.exists() and (item.size is None or dest.stat().st_size == item.size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        if item.size is not None and start > item.size:
            part.unlink()
            start = 0
        try:
            with _open(item.url, start) as resp:
                if start and resp.status != 206:  # server ignored the range: restart
                    start = 0
                with open(part, "ab" if start else "wb") as fh:
                    shutil.copyfileobj(resp, fh, CHUNK)
        except urllib.error.HTTPError as e:
            if e.code == 416 and start:  # partial already holds the whole file
                pass
            else:
                log.warning("%s: attempt %d/%d failed: %s", item.rel_path, attempt, retries, e)
                time.sleep(min(120, 2 ** attempt))
                continue
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log.warning("%s: attempt %d/%d failed: %s", item.rel_path, attempt, retries, e)
            time.sleep(min(120, 2 ** attempt))
            continue

        got = part.stat().st_size
        if item.size is not None and got != item.size:
            log.warning("%s: have %d of %d bytes, retrying", item.rel_path, got, item.size)
            continue
        part.rename(dest)
        return dest
    raise RuntimeError(f"giving up on {item.url}")


def fetch_all(items: list[Remote], root: Path, workers: int = 4) -> list[Remote]:
    """Download many files in parallel; write a manifest; return the items that failed."""
    root.mkdir(parents=True, exist_ok=True)
    total = sum(i.size or 0 for i in items)
    log.info("%d files, %.2f GB -> %s", len(items), total / 1e9, root)
    failed: list[Remote] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, i, root): i for i in items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                fut.result()
                done += 1
                if done % 25 == 0 or done == len(items):
                    log.info("%d/%d done", done, len(items))
            except Exception as e:  # keep going; report at the end
                log.error("%s: %s", item.rel_path, e)
                failed.append(item)
    manifest = root / "_manifest.json"
    manifest.write_text(json.dumps([asdict(i) for i in items], indent=1))
    return failed
