"""
Artifact selection for tables and figures.

Result files are named `<stage>_<model>[_<domain>]_<date>.json`, so a re-run on a
different date leaves TWO files describing the same run key. Globbing them all in
double-counts every model. That bit us twice: once rendering a six-panel figure
where three panels were blank, once double-counting the diagnosis table.

`select` groups by the run's identity read from the JSON itself (detector, model,
domain, corpus) rather than from the filename, keeps only the newest per key, and
reports what it dropped so a stale file is visible instead of silent.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

_DATE = re.compile(r"_(\d{8})\.json$")


def _run_date(path: Path) -> str:
    m = _DATE.search(path.name)
    return m.group(1) if m else f"mtime:{path.stat().st_mtime:.0f}"


def run_key(art: dict) -> tuple:
    """Identity of a run, independent of when it was written."""
    return (art.get("detector", ""), art.get("model", ""),
            art.get("domain", "all"), art.get("corpus", ""))


def select(pattern: str | Path, results_dir: str | Path,
           domain: str | None = None, verbose: bool = True) -> list[dict]:
    """Newest artifact per run key matching `pattern` under `results_dir`.

    domain: keep only runs with this domain tag ("cities"), or None for no filter.
            Passing None does NOT merge domains — differing domains are distinct
            keys and all are returned, which is what you want for an audit and not
            what you want for a headline table. Be explicit.
    """
    best: dict[tuple, tuple[str, Path, dict]] = {}
    dropped: list[tuple[str, str]] = []
    for f in sorted(glob.glob(str(Path(results_dir) / pattern))):
        p = Path(f)
        try:
            art = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            dropped.append((p.name, "unparseable"))
            continue
        if domain is not None and art.get("domain", "all") != domain:
            continue
        k, d = run_key(art), _run_date(p)
        if k not in best or d > best[k][0]:
            if k in best:
                dropped.append((best[k][1].name, f"superseded by {p.name}"))
            best[k] = (d, p, art)
        else:
            dropped.append((p.name, f"superseded by {best[k][1].name}"))
    if verbose and dropped:
        print(f"  [artifacts] ignoring {len(dropped)} stale file(s):")
        for name, why in dropped:
            print(f"    - {name} ({why})")
    return [art for _, _, art in sorted(best.values(), key=lambda t: t[2].get("model", ""))]


def manifest(arts: list[dict]) -> list[dict]:
    """Provenance row per artifact, so a table can state what it was built from."""
    return [{"model": a.get("model"), "detector": a.get("detector"),
             "domain": a.get("domain", "all"), "corpus": a.get("corpus"),
             "seed": a.get("seed"), "provenance": a.get("provenance")} for a in arts]
