"""
Self-test for the released probe auditor (mirage_hardness/audit_probe.py).

Two by-construction probes are pushed through the auditor on the real corpus with
synthetic activations:
  confounded  activations carry a strong typicality dimension + a weak truth one
              -> must be flagged CONFOUNDED (gap CI excludes zero), and must rate
                 the fluent-false (FT) cell TRUE.
  honest      activations carry truth only
              -> must NOT be flagged (gap CI includes zero).

The auditor is the paper's constructive artifact; if it cannot separate these two
it is not fit to audit anyone else's probe. Symmetric by design: a tool that only
ever says CONFOUNDED proves nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "mirage_hardness"))

from mirage_hardness.audit_probe import audit, resolve_probe, verdict  # noqa: E402

SEED = 20260802


def _corpus() -> list[dict]:
    cands = sorted((_ROOT / "data" / "corpus").glob("mirage_2x2_v*.jsonl"))
    if not cands:
        raise SystemExit("no corpus in data/corpus/")
    return [json.loads(l) for l in cands[-1].read_text(encoding="utf-8").splitlines() if l.strip()]


def _acts(items, confounded: bool, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = np.array([bool(it["truth"]) for it in items])
    typical = np.isin(np.array([it["cell"] for it in items]), ("TT", "FT"))
    X = rng.standard_normal((len(items), 16)).astype(np.float32)
    if confounded:
        X[:, 0] += 3.0 * typical      # the shortcut
        X[:, 1] += 1.5 * truth        # the real signal, weaker
    else:
        X[:, 1] += 3.0 * truth        # truth only
    return X


def main() -> None:
    items = _corpus()
    factory, _ = resolve_probe(None)
    failures = []

    conf = audit(_acts(items, True), items, factory, seed=SEED, n_boot=500)
    conf["verdict"] = verdict(conf)
    if not conf["verdict"].startswith("CONFOUNDED"):
        failures.append(f"confounded probe not flagged: {conf['verdict']}")
    if conf["gap"]["ci"][0] <= 0:
        failures.append(f"confounded gap CI includes zero: {conf['gap']['ci']}")
    ft = conf["per_cell_mean_p_true"].get("FT", 0.0)
    if ft <= 0.5:
        failures.append(f"confounded probe should rate FT true, got {ft}")

    hon = audit(_acts(items, False), items, factory, seed=SEED, n_boot=500)
    hon["verdict"] = verdict(hon)
    if hon["verdict"].startswith("CONFOUNDED"):
        failures.append(f"honest probe falsely flagged: gap {hon['gap']['ci']}")

    print(f"confounded: headline {conf['headline_auroc']['auroc']} "
          f"controlled {conf['controlled_auroc']['auroc']} "
          f"gap {conf['gap'].get('gap')} FT {ft}")
    print(f"honest    : headline {hon['headline_auroc']['auroc']} "
          f"controlled {hon['controlled_auroc']['auroc']} "
          f"gap {hon['gap'].get('gap')}")
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("\nPASS — auditor separates confounded from honest probes.")


if __name__ == "__main__":
    main()
