"""Install datasets that cannot be fetched by script into the NAS raw folder.

Dryad blocks scripted downloads (API 401, file stream 403), so those datasets are downloaded
in a browser and installed here, with a size check against the published file list:

    python -m eg606.data.import_local bach_silence ~/Downloads
    python -m eg606.data.import_local --list
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from eg606.paths import raw_dir

# dataset -> (source page, {filename: expected size in bytes or None}, files to unzip after copying)
MANUAL = {
    "bach_silence": (
        "https://datadryad.org/dataset/doi:10.5061/dryad.dbrv15f0j",
        {"ImageryData.mat": 1503601999, "original_stim.zip": 14043961, "README.txt": 1053},
        ["original_stim.zip"],
    ),
    "bach_expect": (
        "https://datadryad.org/dataset/doi:10.5061/dryad.g1jwstqmh",
        {"diliBach_4dryad_CND.zip": 5984510089, "diliBach_midi_4dryad.zip": 11281,
         "readme_DiliBach_EEG.txt": 5199},
        ["diliBach_midi_4dryad.zip"],
    ),
    "broderick": (
        "https://datadryad.org/dataset/doi:10.5061/dryad.070jc",
        {"Natural Speech.zip": None},  # optional second speech set
        [],
    ),
}


def _unpack_bundle(src: Path, expected: dict) -> Path:
    """If src is (or holds) a whole-dataset zip, extract it and return the folder with the files."""
    if src.is_dir() and all((src / n).exists() for n in expected):
        return src
    bundles = [src] if src.suffix == ".zip" else sorted(src.glob("*.zip"))
    for b in bundles:
        try:
            with zipfile.ZipFile(b) as z:
                names = {Path(n).name for n in z.namelist()}
                if not set(expected) <= names:
                    continue
                out = b.parent / (b.stem + "_unpacked")
                out.mkdir(exist_ok=True)
                for member in z.namelist():
                    name = Path(member).name
                    if name in expected and not (out / name).exists():
                        print(f"extracting {name} from {b.name}")
                        with z.open(member) as fh, open(out / name, "wb") as dst:
                            shutil.copyfileobj(fh, dst, 1 << 20)
                return out
        except zipfile.BadZipFile:
            continue
    return src


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.data.import_local")
    ap.add_argument("dataset", nargs="?", choices=sorted(MANUAL))
    ap.add_argument("source", nargs="?", help="folder holding the browser-downloaded files")
    ap.add_argument("--list", action="store_true", help="show what to download and from where")
    ap.add_argument("--unzip", action="store_true", help="also extract the small archives")
    args = ap.parse_args(argv)

    if args.list or not args.dataset:
        for name, (url, files, _) in MANUAL.items():
            total = sum(s or 0 for s in files.values())
            print(f"\n{name}  ({total/1e9:.2f} GB)\n  {url}")
            for f, s in files.items():
                print(f"    {f}" + (f"  ({s/1e6:.1f} MB)" if s else ""))
        return 0
    if not args.source:
        ap.error("give the folder holding the downloaded files, or use --list")

    url, expected, unzip = MANUAL[args.dataset]
    src, dest = Path(args.source).expanduser(), raw_dir(args.dataset)
    dest.mkdir(parents=True, exist_ok=True)

    # Dryad's "Download dataset" button gives one zip holding every file: unpack it first.
    src = _unpack_bundle(src, expected)
    missing, bad = [], []
    for name, size in expected.items():
        f = src / name
        if not f.exists():
            missing.append(name)
        elif size is not None and f.stat().st_size != size:
            bad.append(f"{name}: {f.stat().st_size} bytes, expected {size}")

    if missing or bad:
        print(f"Download these from {url} first:")
        for m in missing:
            print(f"  missing: {m}")
        for b in bad:
            print(f"  wrong size (incomplete download?): {b}")
        return 1

    for name in expected:
        target = dest / name
        if not target.exists() or target.stat().st_size != (src / name).stat().st_size:
            print(f"copying {name} -> {target}")
            shutil.copy2(src / name, target)
    if args.unzip:
        for name in unzip:
            print(f"extracting {name}")
            with zipfile.ZipFile(dest / name) as z:
                z.extractall(dest / Path(name).stem)
    print(f"{args.dataset}: ready in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
