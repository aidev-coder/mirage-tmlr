"""
Build the entity-frequency cache for the D-007 typicality axis.

Scores every entity used in the structured-topic corpus against infini-gram
(Dolma v1.7 counts) ONCE and commits the result, so Stage-2 cell assignment is
reproducible and needs no network on the Modal box. Frequency is the D-007
primary typicality axis: entity commonness is separable from truth (unlike
perplexity, which encodes truth — see project notebook 2026-07-14).

Output: data/corpus/entity_freq.json  {entity: {count, log10}}
Run (needs network): python data/raw/fetch_entity_freq.py
Verify an existing cache:  python data/raw/fetch_entity_freq.py --verify

CORRECTNESS NOTE (2026-08-06). The first version wrote `count = 0` whenever
`status_code != 200` or the payload lacked a "count" key. The API rate-limits and
times out under load, so 2627 of 2884 entities (91%) were cached as count=0 —
including Amman (true count 1,183,125), Anaheim (2,972,569), Andorra (766,246).
The typicality axis was therefore not entity frequency at all but "did the API
call succeed", and every cell assignment built on it was invalid. A failed
request must NEVER be indistinguishable from a real zero: failures are recorded
as None and retried, and --verify re-queries a random sample so a silently
corrupted cache cannot ship again.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src import corpus_gen as cg  # noqa: E402

API = "https://api.infini-gram.io/"
INDEX = "v4_dolma-v1_7_llama"
OUT = _ROOT / "data" / "corpus" / "entity_freq.json"


def all_entities() -> list[str]:
    ents = set()
    for t in cg.TEMPLATES:
        for s, o in cg._parse_true_pairs(t):
            ents.add(s)
            ents.add(o)
        # objects that appear only in false rows also get swapped in — include them
        for st in cg._unedited_false(t):
            m = cg.TEMPLATES[t][0].match(st)
            if m:
                ents.add(m.group(1).strip())
                ents.add(m.group(2).strip())
    return sorted(ents)


def query_count(entity: str, session: requests.Session, tries: int = 5,
                pause: float = 0.3) -> int | None:
    """Real count, or None if it could not be established. Never 0-on-failure."""
    for attempt in range(tries):
        try:
            r = session.post(API, json={"index": INDEX, "query_type": "count",
                                        "query": entity}, timeout=30)
            if r.status_code == 200:
                payload = r.json()
                if "count" in payload:            # the only trustworthy outcome
                    return int(payload["count"])
                # HTTP 200 carrying an error body — the original silent-zero path
        except Exception:
            pass
        time.sleep(pause * (2 ** attempt) + random.random() * 0.2)
    return None


def summarize(cache: dict) -> dict:
    vals = [v.get("count") for v in cache.values()]
    counts = np.array([c for c in vals if isinstance(c, int) and c >= 0])
    return {
        "n": len(cache),
        "unresolved": sum(1 for c in vals if not isinstance(c, int) or c < 0),
        "zero": int((counts == 0).sum()) if len(counts) else 0,
        "nonzero": int((counts > 0).sum()) if len(counts) else 0,
        "median_nonzero": float(np.median(counts[counts > 0])) if (counts > 0).any() else None,
    }


def verify(cache: dict, n: int = 25, seed: int = 20260806) -> dict:
    """Re-query a random sample and compare against the cache. This is the check
    whose absence let a 91%-zero cache ship for three weeks."""
    rng = random.Random(seed)
    sample = rng.sample(sorted(cache), min(n, len(cache)))
    session = requests.Session()
    mismatches, unchecked = [], 0
    for e in sample:
        live = query_count(e, session)
        if live is None:
            unchecked += 1
            continue
        if cache[e].get("count") != live:
            mismatches.append({"entity": e, "cached": cache[e].get("count"), "live": live})
    return {"checked": len(sample) - unchecked, "unchecked": unchecked,
            "mismatches": mismatches, "ok": not mismatches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="sample-check an existing cache")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args()

    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    if args.verify:
        rep = verify(cache)
        print(json.dumps({"summary": summarize(cache), "verify": rep}, indent=2))
        raise SystemExit(0 if rep["ok"] else 1)

    ents = all_entities()
    todo = [e for e in ents
            if not isinstance(cache.get(e, {}).get("count"), int)
            or cache[e]["count"] < 0]
    print(f"{len(ents)} entities, {len(todo)} to (re)fetch", flush=True)

    session = requests.Session()
    failed = []
    for i, e in enumerate(todo, 1):
        c = query_count(e, session, pause=args.pause)
        if c is None:
            failed.append(e)
            cache[e] = {"count": None, "log10": None}
        else:
            cache[e] = {"count": c, "log10": float(np.log10(1 + c))}
        if i % 100 == 0 or i == len(todo):
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(cache, indent=0), encoding="utf-8")
            print(f"  {i}/{len(todo)} (last: {e}={c}) unresolved so far: {len(failed)}", flush=True)
        time.sleep(args.pause)

    OUT.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    print(json.dumps(summarize(cache), indent=2))
    if failed:
        print(f"UNRESOLVED ({len(failed)}): {failed[:20]}{' ...' if len(failed) > 20 else ''}")
        print("Written as None. Re-run to retry; do NOT treat them as zero.")
    else:
        print(f"all resolved -> {OUT}")


if __name__ == "__main__":
    main()
