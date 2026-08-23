"""
Fetch the Stage-1 training/eval sets into data/raw/ (gitignored — this script
IS the reproducibility artifact;).

Sets (configs/probes.yaml `training_data`):
  azaria_mitchell   The "true-false dataset" (Azaria & Mitchell 2023): six CSVs
                    of short factual statements labeled true/false. This is the
                    set SAPLMA was introduced on — the faithful headline condition.
  truthfulqa        TruthfulQA v1 (Lin et al. 2022), generation CSV.

Every download is SHA256-recorded in data/raw/MANIFEST.json so a later run can
detect upstream drift. Re-running is idempotent unless --force.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent
MANIFEST = RAW / "MANIFEST.json"

_LEVINSTEIN = "https://raw.githubusercontent.com/balevinstein/Probes/main/datasets"
_AM_FILES = [
    # Azaria & Mitchell (2023) six topic sets, as mirrored by Levinstein &
    # Herrmann's Probes repo (canonical host azariaa.com is blocked by some
    # network policies; the mirror is the artifact actually used — record it).
    "cities_true_false.csv", "companies_true_false.csv", "animals_true_false.csv",
    "elements_true_false.csv", "inventions_true_false.csv", "facts_true_false.csv",
    "generated_true_false.csv", "capitals_true_false.csv",
    # Levinstein & Herrmann negation sets — the known generalization-failure
    # axis; useful as a sanity companion to the MIRAGE typicality axis.
    "neg_companies_true_false.csv", "neg_facts_true_false.csv",
]

SOURCES = {
    **{
        f"azaria_mitchell/{f}": {
            "url": f"{_LEVINSTEIN}/{f}",
            "kind": "file",
            "dest": f"azaria_mitchell/{f}",
            "citation": ("Azaria & Mitchell (2023); negation sets from Levinstein & "
                         "Herrmann (2023). Mirror: github.com/balevinstein/Probes"),
        }
        for f in _AM_FILES
    },
    "truthfulqa": {
        "url": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        "kind": "file",
        "dest": "truthfulqa/TruthfulQA.csv",
        "citation": "Lin et al. (2022), TruthfulQA",
    },
}


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(name: str, spec: dict, force: bool = False) -> dict:
    dest = RAW / spec["dest"]
    if dest.exists() and not force:
        return {"name": name, "status": "already-present", "path": str(dest)}

    r = requests.get(spec["url"], timeout=120)
    r.raise_for_status()
    entry = {"name": name, "url": spec["url"], "sha256": _sha256(r.content),
             "bytes": len(r.content), "citation": spec["citation"],
             "fetched_at": datetime.now(timezone.utc).isoformat()}

    if spec["kind"] == "zip":
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(dest)
        entry["files"] = sorted(p.name for p in dest.rglob("*") if p.is_file())
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        entry["files"] = [dest.name]
    entry["status"] = "fetched"
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--only", help="fetch a single set (prefix match)")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name, spec in SOURCES.items():
        if args.only and not name.startswith(args.only):
            continue
        print(f"── {name}: {spec['url']}")
        entry = fetch(name, spec, force=args.force)
        print(f"   {entry['status']}" + (f" sha256={entry['sha256'][:16]}…" if "sha256" in entry else ""))
        if entry["status"] == "fetched":
            manifest[name] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
