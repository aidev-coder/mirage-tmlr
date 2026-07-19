"""
Build the entity-frequency cache for the D-007 typicality axis.

Scores every entity used in the structured-topic corpus against infini-gram
(Dolma v1.7 counts) ONCE and commits the result, so Stage-2 cell assignment is
reproducible and needs no network on the Modal box. Frequency is the D-007
primary typicality axis: entity commonness is separable from truth (unlike
perplexity, which encodes truth — see project notebook 2026-07-14).

Output: data/corpus/entity_freq.json  {entity: {count, log10}}
Run (needs network): python data/raw/fetch_entity_freq.py
"""
from __future__ import annotations

import json
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


def main() -> None:
    ents = all_entities()
    cache = {}
    if OUT.exists():
        cache = json.loads(OUT.read_text(encoding="utf-8"))
    todo = [e for e in ents if e not in cache]
    print(f"{len(ents)} entities, {len(todo)} to fetch")
    for i, e in enumerate(todo):
        try:
            r = requests.post(API, json={"index": INDEX, "query_type": "count",
                                         "query": e}, timeout=20)
            c = int(r.json().get("count", 0)) if r.status_code == 200 else 0
        except Exception:
            c = -1  # network failure marker (reported, not silently 0)
        cache[e] = {"count": c, "log10": (float(np.log10(1 + c)) if c >= 0 else None)}
        if i % 100 == 0:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  {i}/{len(todo)}  (last: {e}={c})", flush=True)
        time.sleep(0.05)
    OUT.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    miss = sum(1 for v in cache.values() if v["count"] < 0)
    print(f"done: {len(cache)} cached, {miss} network-miss -> {OUT}")


if __name__ == "__main__":
    main()
