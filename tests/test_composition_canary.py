"""
Regression test for the composition canary — the check whose absence cost MIRAGE
its headline (2026-08-03).

The canary must:
  1. FAIL the released pooled corpus on `domain`. That corpus put 32% non-cities in
     diagonal-TRUE and 0% in diagonal-FALSE while the off-diagonal reversed it, so a
     topic-reading probe scored BELOW chance off-diagonal (AUROC 0.46) and we read
     three weeks of that as a typicality confound.
  2. PASS the cities-only subset, where the collapse does not occur (off-diagonal
     AUROC 0.97). A check that fails everything proves nothing.
  3. NOT gate on metadata the probe cannot see. `provenance.strategy` predicts truth
     perfectly by construction but never enters the model's input; gating on it would
     fail every corpus of this design and train users to ignore the check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.corpus_build import composition_canary  # noqa: E402


def main() -> None:
    cands = sorted((_ROOT / "data" / "corpus").glob("mirage_2x2_v*.jsonl"))
    if not cands:
        raise SystemExit("no corpus in data/corpus/")
    items = [json.loads(l) for l in cands[-1].read_text(encoding="utf-8").splitlines() if l.strip()]
    cities = [it for it in items if it.get("domain") == "cities"]

    pooled = composition_canary(items)
    pure = composition_canary(cities)
    failures = []

    if pooled["pass"]:
        failures.append("pooled corpus PASSED — the domain shortcut is not being caught")
    if "domain" not in pooled["failed_fields"]:
        failures.append(f"'domain' not flagged; flagged={pooled['failed_fields']}")
    dom = pooled["fields"].get("domain", {})
    if dom.get("auroc") is not None and dom["auroc"] > 0.4:
        failures.append(f"domain AUROC {dom['auroc']} — expected well below chance (inverting shortcut)")
    if not dom.get("text_visible"):
        failures.append("domain should be recoverable from the statement text")

    if not pure["pass"]:
        failures.append(f"cities-only FAILED — false positive; flagged={pure['failed_fields']}")
    if "provenance.strategy" in pooled["failed_fields"] + pure["failed_fields"]:
        failures.append("gated on provenance.strategy, which the probe never sees")

    print(f"pooled      : pass={pooled['pass']} failed={pooled['failed_fields']} "
          f"(domain auroc={dom.get('auroc')}, text_visible={dom.get('text_visible')})")
    print(f"cities-only : pass={pure['pass']} failed={pure['failed_fields']}")
    print(f"reported-not-gated: {pooled['metadata_only_imbalances']}")
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("\nPASS — canary catches the domain shortcut, clears the domain-pure corpus, "
          "and ignores probe-invisible metadata.")


if __name__ == "__main__":
    main()
