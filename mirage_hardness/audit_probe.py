"""
MIRAGE probe auditor — the second half of the released artifact (paper §5/§6).

`run_check.py` audits a CORPUS. This audits a DETECTOR: point it at any
internal-state hallucination probe and it returns the triple the paper argues
every such probe should publish:

    headline    AUROC on a held-out DIAGONAL split (truth and typicality aligned)
                — the number the field reports.
    controlled  AUROC on the OFF-DIAGONAL only (rare-true + fluent-false), where
                truth and typicality disagree — the honest number.
    gap         headline - controlled, with a bootstrap CI. The gap is the
                inflation attributable to the typicality confound.

Also reports the per-cell mean P(true). A confounded probe rates the fluent-false
(FT) cell TRUE; that cell is the mechanism, not a rounding error.

Your probe supplies a factory with the standard two-method contract:

    class MyProbe:
        def fit(self, X, y): ...      # X: [n, d] activations, y: bool truth
        def score(self, X): ...       # -> [n] P(true)

Usage
-----
Precomputed activations (no GPU needed):

    python mirage_hardness/audit_probe.py \
        --corpus data/corpus/mirage_2x2_v<hash>.jsonl \
        --activations acts.npy \
        --probe mypkg.myprobe:MyProbe

  acts.npy is [n, d] (one row per corpus item, same order) or [n, n_layers, d]
  with --layer L.

Extract activations here instead (needs GPU + transformers):

    python mirage_hardness/audit_probe.py --corpus <corpus> \
        --model meta-llama/Llama-3.1-8B --probe mypkg.myprobe:MyProbe

Omit --probe to audit the built-in SAPLMA reference implementation.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.eval.adversarial_split import adversarial_split  # noqa: E402
from src.eval.mediation import mediation  # noqa: E402
from src.eval.stratified_auroc import stratified_auroc  # noqa: E402

CELLS = ("TT", "TA", "FT", "FA")


def load_corpus(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_probe(spec: str | None):
    """'package.module:Attr' -> callable factory. None -> built-in SAPLMA."""
    if not spec:
        from src.probes.saplma import SaplmaProbe
        return lambda: SaplmaProbe(seed=0), "builtin:SaplmaProbe"
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--probe must be 'module:Attr' (a class or a zero-arg factory)")
    obj = getattr(importlib.import_module(mod_name), attr)
    for m in ("fit", "score"):
        target = obj if not isinstance(obj, type) else obj
        if not hasattr(target, m) and not callable(obj):
            raise SystemExit(f"probe {spec} must expose .{m}()")
    return (obj if callable(obj) and not isinstance(obj, type) else obj), spec


def _cell_means(scores: np.ndarray, cells: np.ndarray) -> dict:
    return {c: round(float(scores[cells == c].mean()), 4)
            for c in CELLS if (cells == c).any()}


def audit(X: np.ndarray, items: list[dict], probe_factory, seed: int = 0,
          n_boot: int = 1000) -> dict:
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    typicality = np.array([it["typicality"]["entity_freq_log10"] for it in items],
                          dtype=float)

    adv = adversarial_split(X, truth, cells, probe_factory, n_boot=n_boot, seed=seed)

    # the fielded instrument, scored on every item: train on the diagonal only
    # (the field's recipe), then read out on the whole corpus.
    from src.eval.adversarial_split import DIAGONAL
    diag = np.flatnonzero(np.isin(cells, DIAGONAL))
    probe = probe_factory()
    probe.fit(X[diag], truth[diag])
    fielded = np.asarray(probe.score(X))

    headline = adv["headline_heldout_diagonal"]
    controlled = adv["off_diagonal"]
    return {
        "headline_auroc": headline,
        "controlled_auroc": controlled,
        "gap": adv["gap"],
        "per_cell_mean_p_true": _cell_means(fielded, cells),
        "stratified_gap": stratified_auroc(fielded, truth, typicality, seed=seed).get("gap"),
        "mediation": mediation(fielded, truth, typicality),
        "n": int(len(items)),
        "cell_counts": {c: int((cells == c).sum()) for c in CELLS},
    }


def verdict(rep: dict) -> str:
    g = rep["gap"]
    lo = g["ci"][0] if g.get("ci") else None
    if lo is not None and lo > 0:
        return ("CONFOUNDED — headline AUROC is inflated relative to honest "
                "off-diagonal detection (gap CI excludes zero)")
    return ("NOT DISTINGUISHABLE FROM UNCONFOUNDED on this corpus "
            "(gap CI includes zero)")


def main() -> None:
    ap = argparse.ArgumentParser(description="MIRAGE probe auditor")
    ap.add_argument("--corpus", required=True, help="corpus .jsonl")
    ap.add_argument("--probe", default=None, help="module:Attr (default: built-in SAPLMA)")
    ap.add_argument("--activations", default=None, help="[n,d] or [n,L,d] .npy")
    ap.add_argument("--model", default=None, help="HF id; extract activations here (GPU)")
    ap.add_argument("--layer", type=int, default=None, help="layer index for [n,L,d] / model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()

    items = load_corpus(args.corpus)
    if args.activations:
        X = np.load(args.activations)
    elif args.model:
        from src.substrate import Substrate
        sub = Substrate(args.model)
        X = sub.hidden_states_matrix([it["text"] for it in items], batch_size=32)
    else:
        raise SystemExit("supply --activations or --model")

    if X.ndim == 3:
        L = args.layer if args.layer is not None else X.shape[1] // 2
        X = X[:, L, :]
    if len(X) != len(items):
        raise SystemExit(f"activations ({len(X)}) and corpus ({len(items)}) length mismatch")

    factory, name = resolve_probe(args.probe)
    rep = audit(X.astype(np.float32), items, factory, seed=args.seed)
    rep["probe"] = name
    rep["corpus"] = Path(args.corpus).name
    rep["verdict"] = verdict(rep)

    print(json.dumps(rep, indent=2))
    h, c, g = rep["headline_auroc"], rep["controlled_auroc"], rep["gap"]
    print(f"\nprobe      : {name}")
    print(f"headline   : {h['auroc']} {h['ci']}   (held-out diagonal)")
    print(f"controlled : {c['auroc']} {c['ci']}   (off-diagonal, honest)")
    print(f"gap        : {g.get('gap', g.get('point'))} {g['ci']}")
    print(f"per cell   : {rep['per_cell_mean_p_true']}")
    print(f"\nVERDICT: {rep['verdict']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
