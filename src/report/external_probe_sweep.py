"""
Run an external probe across every model and test whether the recoverability
mechanism holds for an architecture that is not ours.

The paper's mechanism claim is that a probe falls back on typicality exactly to the
extent that truth is not linearly available in the representation it reads. Measured
on our own SAPLMA probe that gives r = 0.964, but recoverability and the honest score
both came from the same apparatus, so the correlation could in principle be a
property of our probe rather than of the representation.

This script breaks that circularity two ways:

  1. It correlates an EXTERNAL probe's honest off-diagonal AUROC against the
     recoverability measured by OUR all-cell probe. Recoverability is a property of
     the representation; if the mechanism is real it should predict the confounding
     of a probe that had no part in measuring it.
  2. It also computes the external probe's OWN recoverability — the same probe fit
     on all four cells instead of the diagonal — so the relation can be stated
     entirely within the external architecture.

--layers reads the Modal `extlayers` artifacts instead and reports what the
benchmark can say about LAYER choice. The headline quantity there is the tie
spread: among layers the benchmark scores identically, how far apart is honest
detection? That needs no selection rule, so it is immune to argmax noise — which
matters because in-distribution saturates at 1.000 on many layers at once, and an
argmax over ties would be this script's tie-breaking rather than a finding.

Usage:
    python -m src.report.external_probe_sweep \
        --probe mirage_hardness.probes_external.geometry_of_truth_mmprobe:GeometryOfTruthMMProbe \
        --corpus data/corpus/mirage_2x2_v6206fe484650.jsonl \
        --acts-dir data/activations/local --tag mmprobe

    python -m src.report.external_probe_sweep --layers --corpus-hash 6206fe484650
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.eval.adversarial_split import DIAGONAL, OFF_DIAGONAL, adversarial_split  # noqa: E402
from src.stats import auroc_with_ci  # noqa: E402


def _load_probe(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _corpus(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _our_recoverability(model: str, corpus_name: str) -> float | None:
    """Recoverability as measured by OUR all-cell probe — a property of the
    representation, computed without the external probe's involvement."""
    for f in (_ROOT / "results").glob("stage3_saplma_*.json"):
        a = json.loads(f.read_text(encoding="utf-8"))
        if a.get("corpus") == corpus_name and a["model"].split("/")[-1].lower() == model.lower():
            e = a["per_layer"][a["headline_layer"]]
            return e["mediation_allcell"].get("truth_beta_partialled")
    return None


def run(probe_spec: str, corpus_path: Path, acts_dir: Path, seed: int = 0,
        n_folds: int = 5) -> list[dict]:
    factory = _load_probe(probe_spec)
    items = _corpus(corpus_path)
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    chash = corpus_path.stem.split("_v")[-1]
    off_idx = np.flatnonzero(np.isin(cells, OFF_DIAGONAL))
    diag_idx = np.flatnonzero(np.isin(cells, DIAGONAL))

    rows = []
    for npy in sorted(acts_dir.glob(f"*_{chash}_L*.npy")):
        m = re.match(rf"^(.+)_{chash}_L(\d+)\.npy$", npy.name)
        if not m:
            continue
        model, layer = m.group(1), int(m.group(2))
        X = np.load(npy).astype(np.float64)
        if len(X) != len(items):
            print(f"  skip {npy.name}: {len(X)} rows vs corpus {len(items)}")
            continue

        # the field's recipe: train on the diagonal, evaluate honestly off-diagonal
        adv = adversarial_split(X, truth, cells, lambda: factory(seed=seed), seed=seed)

        # The same external probe trained fairly on all four cells — its OWN
        # recoverability, stated entirely within this architecture.
        # CROSS-FITTED (2026-08-07 fix): fitting and scoring the same rows is
        # train-set-optimistic, and with d >> n a gradient-trained linear probe
        # memorizes outright — LRProbe returned exactly 1.000 on all seven models,
        # zero variance, which made the correlation undefined and would have been
        # reported as perfect recoverability. K-fold, matching stage3._oof_scores.
        fair_scores = np.zeros(len(truth), dtype=np.float64)
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(len(truth)), n_folds)
        for k in range(n_folds):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            fitted = factory(seed=seed).fit(X[tr], truth[tr])
            fair_scores[te] = np.asarray(fitted.score(X[te]))
        fair_off = auroc_with_ci(truth[off_idx], fair_scores[off_idx], seed=seed)

        # per-cell readout of the fielded probe
        fielded = factory(seed=seed).fit(X[diag_idx], truth[diag_idx])
        s = np.asarray(fielded.score(X))
        per_cell = {c: round(float((s[cells == c] > 0.5).mean()), 4)
                    for c in ("TT", "TA", "FT", "FA")}

        stab = fit_stability(X, truth, cells, factory, seed=seed)
        rows.append({
            "model": model, "layer": layer,
            "fit_stability": stab,
            "in_dist": adv["headline_heldout_diagonal"]["auroc"],
            "off": adv["off_diagonal"]["auroc"], "off_ci": adv["off_diagonal"]["ci"],
            "gap": adv["gap"]["gap"], "gap_ci": adv["gap"]["ci"],
            "gap_excludes_zero": adv["gap"].get("excludes_zero"),
            "external_own_recoverability_off_auroc": fair_off["auroc"],
            "our_recoverability_truth_beta": _our_recoverability(model, corpus_path.name),
            "per_cell_frac_called_true": per_cell,
        })
        print(f"  {model:24s} L{layer:<3d} in-dist {adv['headline_heldout_diagonal']['auroc']:.3f} "
              f"off {adv['off_diagonal']['auroc']:.3f} gap {adv['gap']['gap']:+.3f} "
              f"eval-CI [{adv['gap']['ci'][0]:+.3f}, {adv['gap']['ci'][1]:+.3f}] "
              f"refit-range [{stab['gap_range'][0]:+.3f}, {stab['gap_range'][1]:+.3f}] "
              f"| fair-trained off {fair_off['auroc']:.3f}", flush=True)
    return rows


IN_DIST_FLOOR = 0.70
TIE_TOLERANCES = (0.0, 0.01, 0.02)


def _probe_short(spec: str) -> str:
    return (spec.split(":")[-1].replace("GeometryOfTruth", "")
            .replace("ProbeAdapter", "").replace("Probe", "").lower() or spec)


def collect_layer_curves(chash: str) -> dict[str, dict[str, list[dict]]]:
    newest: dict[str, Path] = {}
    for p in sorted((_ROOT / "results").glob(f"extlayers_*_{chash}_*.json")):
        newest[json.loads(p.read_text(encoding="utf-8"))["model"]] = p
    out: dict[str, dict[str, list[dict]]] = {}
    for model, p in newest.items():
        by_probe: dict[str, list[dict]] = {}
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            if "error" not in r:
                by_probe.setdefault(_probe_short(r["probe"]), []).append(r)
        out[model] = {k: sorted(v, key=lambda r: r["layer"]) for k, v in by_probe.items()}
    return out


def _selection_rules(curve: list[dict], n_layers: int) -> dict:
    """What the field's ACTUAL layer-selection rules pick, and what they cost.

    An argmax is a strawman nobody uses. These are published practice:
      earliest_informative  earliest layer within 95% of the max in-distribution
                            AUROC (a stated criterion in the probing literature)
      fixed_depth_*         fixed fractional depth; PARALLAX (2026) taps
                            0.60-0.85*L citing Marks & Tegmark that truth lives in
                            the upper third
      our_mid_depth         n_layers // 2, what this project used
    All select on in-distribution only, never on the off-diagonal being reported.
    """
    top = max(r["in_dist"] for r in curve)
    by_layer = {r["layer"]: r for r in curve}
    picks = {}

    inf = [r for r in curve if r["in_dist"] >= 0.95 * top]
    if inf:
        picks["earliest_informative_95pct"] = min(inf, key=lambda r: r["layer"])
    for frac in (0.60, 0.70, 0.80, 0.85):
        L = min(by_layer, key=lambda z: abs(z - frac * (n_layers - 1)))
        picks[f"fixed_depth_{frac:.2f}"] = by_layer[L]
    mid = min(by_layer, key=lambda z: abs(z - (n_layers - 1) // 2))
    picks["our_mid_depth"] = by_layer[mid]
    best = max(curve, key=lambda r: r["off"])

    return {k: {"layer": v["layer"], "in_dist": v["in_dist"], "off": v["off"],
                "gap": v["gap"],
                "off_below_best": round(best["off"] - v["off"], 4)}
            for k, v in picks.items()}


def _ties(curve: list[dict], tol: float) -> dict:
    top = max(r["in_dist"] for r in curve)
    tied = [r for r in curve if r["in_dist"] >= top - tol]
    offs = [r["off"] for r in tied]
    return {"tolerance": tol, "top_in_dist": round(top, 4), "n_tied_layers": len(tied),
            "layers": [r["layer"] for r in tied],
            "honest_off_min": round(min(offs), 4), "honest_off_max": round(max(offs), 4),
            "honest_off_spread": round(max(offs) - min(offs), 4)}


def summarize_layers(curves: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for model, probes in sorted(curves.items()):
        for probe, curve in sorted(probes.items()):
            sel = max(curve, key=lambda r: r["in_dist"])
            best = max(curve, key=lambda r: r["off"])
            worst = min(curve, key=lambda r: r["off"])
            max_in = max(r["in_dist"] for r in curve)
            rows.append({
                "model": model.split("/")[-1], "probe": probe, "n_layers": len(curve),
                "selection_rules": _selection_rules(curve, len(curve)),
                "benchmark_ties": [_ties(curve, t) for t in TIE_TOLERANCES],
                "argmax_in_dist_layer": sel["layer"],
                "at_argmax": {"in_dist": sel["in_dist"], "off": sel["off"],
                              "gap": sel["gap"], "gap_ci": sel["gap_ci"]},
                "honest_best_layer": best["layer"],
                "at_honest_best": {"in_dist": best["in_dist"], "off": best["off"],
                                   "gap": best["gap"]},
                # meaningless unless a better-than-chance layer actually existed
                "off_forgone_by_argmax": (round(best["off"] - sel["off"], 4)
                                          if best["off"] > 0.5 else None),
                "no_layer_beats_chance": bool(best["off"] <= 0.5),
                "off_range_across_layers": [round(worst["off"], 4), round(best["off"], 4)],
                "max_in_dist_any_layer": round(max_in, 4),
                "clears_in_dist_floor_somewhere": bool(max_in >= IN_DIST_FLOOR),
            })
    return rows


def _headline_layers(corpus_name: str) -> dict[str, int]:
    out = {}
    for p in (_ROOT / "results").glob("stage3_saplma_*.json"):
        a = json.loads(p.read_text(encoding="utf-8"))
        if a.get("corpus") == corpus_name:
            out[a["model"]] = a["headline_layer"]
    return out


def at_two_layers(curves: dict[str, dict[str, list[dict]]], corpus_name: str) -> list[dict]:
    """Each probe read at OUR SAPLMA headline layer and at the layer the BENCHMARK
    selects for it. Selection uses in-distribution only, never the off-diagonal
    score being reported.

    Where several layers tie at the top in-distribution score the argmax is a
    tie-break, not a selection, and taking the first one systematically picks the
    lowest layer index — which for LRProbe means early layers whose honest score is
    far worse, manufacturing a gap the benchmark never chose. So the reported value
    is the MEDIAN across tied layers, which is what a practitioner picking any of
    the equally-best layers gets in expectation, plus the spread.
    """
    heads = _headline_layers(corpus_name)
    rows = []
    for model, probes in sorted(curves.items()):
        hl = heads.get(model)
        for probe, curve in sorted(probes.items()):
            top = max(r["in_dist"] for r in curve)
            tied = [r for r in curve if r["in_dist"] >= top]
            offs = sorted(r["off"] for r in tied)
            gaps = sorted(r["gap"] for r in tied)
            mid = tied[min(range(len(tied)),
                           key=lambda i: abs(tied[i]["off"] - float(np.median(offs))))]
            ours = next((r for r in curve if r["layer"] == hl), None)
            rows.append({
                "model": model.split("/")[-1], "probe": probe,
                "our_headline_layer": hl,
                "at_our_layer": None if ours is None else {
                    "layer": ours["layer"], "in_dist": ours["in_dist"],
                    "off": ours["off"], "gap": ours["gap"], "gap_ci": ours["gap_ci"],
                    "gap_excludes_zero": ours["gap_excludes_zero"]},
                "benchmark_selected": {
                    "top_in_dist": round(top, 4), "n_tied_layers": len(tied),
                    "layers": [r["layer"] for r in tied],
                    "median_off": round(float(np.median(offs)), 4),
                    "median_gap": round(float(np.median(gaps)), 4),
                    "off_spread": round(offs[-1] - offs[0], 4),
                    "representative_layer": mid["layer"],
                    "gap_ci_at_representative": mid["gap_ci"],
                    "layer_is_determined": len(tied) == 1},
                "own_layer_clears_floor": bool(top >= IN_DIST_FLOOR),
            })
    return rows


def _our_recoverability_map(corpus_name: str) -> dict[str, float]:
    out = {}
    for p in (_ROOT / "results").glob("stage3_saplma_*.json"):
        a = json.loads(p.read_text(encoding="utf-8"))
        if a.get("corpus") == corpus_name:
            e = a["per_layer"][a["headline_layer"]]
            b = e["mediation_allcell"].get("truth_beta_partialled")
            if b is not None:
                out[a["model"].split("/")[-1]] = b
    return out


def mechanism_by_probe(two: list[dict], corpus_name: str) -> dict:
    """Does recoverability still predict honest performance when each probe is read
    at the layer the benchmark selects for it rather than at ours? Reported over all
    models and over adequately powered ones only, both directions shown."""
    rec = _our_recoverability_map(corpus_name)
    out: dict = {}
    for probe in sorted({r["probe"] for r in two}):
        for label, get_off in (
                ("our_headline_layer",
                 lambda r: r["at_our_layer"]["off"] if r["at_our_layer"] else None),
                ("benchmark_selected_layer",
                 lambda r: r["benchmark_selected"]["median_off"])):
            for powered_only in (False, True):
                pairs = [(rec[r["model"]], get_off(r)) for r in two
                         if r["probe"] == probe and r["model"] in rec
                         and get_off(r) is not None
                         and (r["own_layer_clears_floor"] or not powered_only)]
                if len(pairs) > 2:
                    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
                    key = label + ("_powered_only" if powered_only else "")
                    out.setdefault(probe, {})[key] = _r_with_ci(x, y)
    return out


def fit_stability(X: np.ndarray, truth: np.ndarray, cells: np.ndarray, factory,
                  n_resample: int = 60, drop_frac: float = 0.10, seed: int = 0) -> dict:
    """How much does the gap move when the probe is refitted on a different sample?

    The bootstrap CI this project reports resamples the EVAL scores with the fitted
    probe held fixed, so it cannot see instability in the fit itself. MMProbe's gap
    on pythia-1.4b travels across [+0.045, +0.586] under random removal of 10% of
    rows — an interval its reported [0.373, 0.593] gives no hint of. Unregularised
    mean-difference directions are the worst case, since a handful of points move
    the class means outright.
    """
    rng = np.random.default_rng(seed)
    n = len(truth)
    keep_n = int(round(n * (1 - drop_frac)))
    gaps, offs = [], []
    for _ in range(n_resample):
        idx = np.sort(rng.choice(n, keep_n, replace=False))
        a = adversarial_split(X[idx], truth[idx], cells[idx],
                              lambda: factory(seed=seed), n_boot=1, seed=seed)
        gaps.append(a["gap"]["gap"])
        offs.append(a["off_diagonal"]["auroc"])
    gaps, offs = np.array(gaps), np.array(offs)
    return {"n_resample": n_resample, "drop_frac": drop_frac,
            "gap_median": round(float(np.median(gaps)), 4),
            "gap_range": [round(float(np.percentile(gaps, 2.5)), 4),
                          round(float(np.percentile(gaps, 97.5)), 4)],
            "gap_sd": round(float(gaps.std()), 4),
            "off_range": [round(float(np.percentile(offs, 2.5)), 4),
                          round(float(np.percentile(offs, 97.5)), 4)]}


def _r_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """At n=7 a correlation of 0.7 is not distinguishable from zero (§1.5)."""
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) > 1 and len(set(y[idx])) > 1:
            draws.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"r": round(r, 4), "n": int(len(x)),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run_layer_report(chash: str) -> Path:
    curves = collect_layer_curves(chash)
    if not curves:
        raise SystemExit(f"no extlayers_*_{chash}_*.json in results/")
    rows = summarize_layers(curves)
    corpus_name = f"mirage_2x2_v{chash}.jsonl"
    two = at_two_layers(curves, corpus_name)
    mech = mechanism_by_probe(two, corpus_name)

    print("LAYERS THE BENCHMARK SCORES EQUALLY, AND HOW FAR APART THEY HONESTLY ARE")
    print(f"{'model':22s} {'probe':6s} {'topInD':>7s} {'#tied':>6s} "
          f"{'honest off range':>19s} {'spread':>8s}")
    for r in rows:
        t = r["benchmark_ties"][0]
        print(f"{r['model']:22s} {r['probe']:6s} {t['top_in_dist']:7.4f} "
              f"{t['n_tied_layers']:6d} {t['honest_off_min']:9.3f} -"
              f"{t['honest_off_max']:8.3f} {t['honest_off_spread']:8.3f}")

    print("\nallowing layers within 0.01 in-distribution of the top:")
    for r in sorted(rows, key=lambda z: -z["benchmark_ties"][1]["honest_off_spread"])[:6]:
        t = r["benchmark_ties"][1]
        print(f"  {r['model']:22s} {r['probe']:6s} {t['n_tied_layers']:3d} layers "
              f"L{min(t['layers'])}-{max(t['layers'])}  honest off "
              f"{t['honest_off_min']:.3f}-{t['honest_off_max']:.3f}  "
              f"spread {t['honest_off_spread']:.3f}")

    under = [r for r in rows if not r["clears_in_dist_floor_somewhere"]]
    print(f"\nbelow the {IN_DIST_FLOOR} in-distribution floor at EVERY layer "
          f"(genuinely underpowered, not a layer-choice artifact): {len(under)}")
    for r in under:
        print(f"  {r['model']} / {r['probe']}: best in-dist anywhere "
              f"{r['max_in_dist_any_layer']:.3f}")

    print("\nWHAT THE FIELD'S PUBLISHED LAYER-SELECTION RULES ACTUALLY PICK "
          "(mean honest off-diagonal, and mean shortfall vs the best layer):")
    rules = sorted({k for r in rows for k in r["selection_rules"]})
    for rule in rules:
        offs = [r["selection_rules"][rule]["off"] for r in rows if rule in r["selection_rules"]]
        short = [r["selection_rules"][rule]["off_below_best"] for r in rows
                 if rule in r["selection_rules"]]
        print(f"  {rule:26s} honest {np.mean(offs):.3f}   leaves "
              f"{np.mean(short):.3f} on the table   (n={len(offs)})")

    dead = [r for r in rows if r["no_layer_beats_chance"]]
    print(f"\nno layer beats chance off-diagonal: {len(dead)}")
    for r in dead:
        print(f"  {r['model']:22s} {r['probe']:6s} best off "
              f"{r['at_honest_best']['off']:.3f} at L{r['honest_best_layer']}")

    print("\nEACH PROBE AT OUR HEADLINE LAYER vs WHERE THE BENCHMARK SELECTS")
    print(f"{'model':22s} {'probe':6s} {'ourL':>5s} {'inD':>6s} {'off':>6s} {'gap':>7s}"
          f"   {'topInD':>7s} {'#tied':>5s} {'medOff':>7s} {'medGap':>7s} {'spread':>7s}")
    for r in two:
        a, b = r["at_our_layer"], r["benchmark_selected"]
        flag = "" if r["own_layer_clears_floor"] else "  *underpowered"
        left = (f"{a['layer']:5d} {a['in_dist']:6.3f} {a['off']:6.3f} {a['gap']:+7.3f}"
                if a else f"{'-':>5s} {'-':>6s} {'-':>6s} {'-':>7s}")
        print(f"{r['model']:22s} {r['probe']:6s} {left}   {b['top_in_dist']:7.4f} "
              f"{b['n_tied_layers']:5d} {b['median_off']:7.3f} {b['median_gap']:+7.3f} "
              f"{b['off_spread']:7.3f}{flag}")

    print("\ncorrelation(our recoverability, each probe's honest off-diagonal):")
    for probe, d in sorted(mech.items()):
        for k, v in sorted(d.items()):
            print(f"  {probe:6s} {k:38s} r={v['r']:+.3f} "
                  f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] n={v['n']}"
                  f"{'' if v['excludes_zero'] else '   NOT distinguishable from 0'}")

    dest = _ROOT / "results" / f"extlayers_summary_{chash}.json"
    dest.write_text(json.dumps(
        {"corpus_hash": chash, "in_dist_floor": IN_DIST_FLOOR,
         "provenance": "measured", "summary": rows,
         "at_two_layers": two, "mechanism_by_probe": mech, "curves": curves},
        indent=2), encoding="utf-8")
    print(f"\n-> {dest}")
    return dest


def _hinge(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    best = None
    for c in np.arange(0.05, 0.95, 0.005):
        h = np.maximum(0.0, x - c)
        if h.std() < 1e-9:
            continue
        A = np.column_stack([np.ones_like(x), h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = float(((y - A @ coef) ** 2).sum())
        if best is None or ss < best[0]:
            best = (ss, float(c), coef)
    return best


def _fit_shape(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """A correlation assumes a line. It is not one: below a recoverability knee the
    probe is a pure frequency readout and additional recoverability buys nothing,
    so a hinge y = floor + slope*max(0, x - knee) is the right shape. Reported with
    AIC against the line and a step, and with the knee bootstrapped."""
    n = len(x)
    lin = np.polyfit(x, y, 1)
    ss_lin = float(((y - np.polyval(lin, x)) ** 2).sum())
    ss_step = None
    for c in np.unique(x):
        lo, hi = y[x < c], y[x >= c]
        if len(lo) < 2 or len(hi) < 2:
            continue
        s = float(((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum())
        ss_step = s if ss_step is None or s < ss_step else ss_step
    ss_hin, knee, coef = _hinge(x, y)

    def aic(ss, k):
        return round(float(n * np.log(ss / n) + 2 * k), 2)

    rng = np.random.default_rng(seed)
    knees, floors, slopes = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(x[idx].tolist())) < 4:
            continue
        b = _hinge(x[idx], y[idx])
        if b:
            knees.append(b[1]); floors.append(b[2][0]); slopes.append(b[2][1])

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return {"n": n,
            "linear": {"ss": round(ss_lin, 4), "k": 2, "aic": aic(ss_lin, 2)},
            "step": {"ss": round(ss_step, 4), "k": 3, "aic": aic(ss_step, 3)},
            "hinge": {"ss": round(ss_hin, 4), "k": 3, "aic": aic(ss_hin, 3),
                      "knee": round(knee, 4), "floor": round(float(coef[0]), 4),
                      "slope": round(float(coef[1]), 4),
                      "knee_ci": ci(knees), "floor_ci": ci(floors), "slope_ci": ci(slopes)},
            "preferred": "hinge" if aic(ss_hin, 3) < min(aic(ss_lin, 2), aic(ss_step, 3)) else "linear"}


def run_mechanism_report(corpus_name: str) -> Path:
    """The recoverability -> honest-performance relation for OUR probe, over every
    model with a Stage 3 artifact on this corpus. The draft quotes r = 0.964 at
    n = 7 with no interval; at that n almost nothing is distinguishable from zero,
    so the Pythia ladder is here to make the estimate mean something."""
    rows = []
    seen: dict[str, Path] = {}
    for p in sorted((_ROOT / "results").glob("stage3_saplma_*.json")):
        a = json.loads(p.read_text(encoding="utf-8"))
        if a.get("corpus") != corpus_name or "_cities_" in p.name:
            continue
        seen[a["model"]] = p
    for model, p in seen.items():
        a = json.loads(p.read_text(encoding="utf-8"))
        e = a["per_layer"][a["headline_layer"]]
        rec = e.get("mediation_allcell", {}).get("truth_beta_partialled")
        adv = e["adversarial"]
        if rec is None:
            continue
        rows.append({"model": model.split("/")[-1], "layer": a["headline_layer"],
                     "in_dist": adv["headline_heldout_diagonal"]["auroc"],
                     "off": adv["off_diagonal"]["auroc"],
                     "off_ci": adv["off_diagonal"]["ci"],
                     "gap": adv["gap"]["gap"], "recoverability": rec})
    rows.sort(key=lambda r: r["recoverability"])
    x = np.array([r["recoverability"] for r in rows])
    y = np.array([r["off"] for r in rows])
    stat = _r_with_ci(x, y)
    shape = _fit_shape(x, y)

    print(f"{'model':24s} {'L':>4s} {'in-dist':>8s} {'honest off':>11s} {'gap':>8s} "
          f"{'recoverability':>15s}")
    for r in rows:
        print(f"{r['model']:24s} {r['layer']:4d} {r['in_dist']:8.3f} {r['off']:11.3f} "
              f"{r['gap']:+8.3f} {r['recoverability']:15.3f}")
    print(f"\ncorrelation(recoverability, honest off-diagonal) = {stat['r']:+.4f} "
          f"[{stat['ci'][0]:+.3f}, {stat['ci'][1]:+.3f}]  n={stat['n']}"
          f"{'' if stat['excludes_zero'] else '   NOT distinguishable from 0'}")
    print(f"in-distribution range {min(r['in_dist'] for r in rows):.3f}-"
          f"{max(r['in_dist'] for r in rows):.3f}; honest range "
          f"{min(r['off'] for r in rows):.3f}-{max(r['off'] for r in rows):.3f}")

    h = shape["hinge"]
    print(f"\nshape: AIC linear {shape['linear']['aic']}, step {shape['step']['aic']}, "
          f"hinge {h['aic']}  -> {shape['preferred']}")
    print(f"  honest_off = {h['floor']:.3f} + {h['slope']:.2f} * max(0, recoverability "
          f"- {h['knee']:.3f})")
    print(f"  knee {h['knee']:.3f} {h['knee_ci']}   floor {h['floor']:.3f} {h['floor_ci']}"
          f"   slope {h['slope']:.2f} {h['slope_ci']}")
    below = [r for r in rows if r["recoverability"] < h["knee"]]
    print(f"  below the knee ({len(below)} models): honest "
          f"{min(r['off'] for r in below):.3f}-{max(r['off'] for r in below):.3f} across "
          f"recoverability {min(r['recoverability'] for r in below):.3f}-"
          f"{max(r['recoverability'] for r in below):.3f} — more recoverability buys nothing")

    dest = _ROOT / "results" / f"mechanism_saplma_{corpus_name.split('_v')[-1].split('.')[0]}.json"
    dest.write_text(json.dumps({"corpus": corpus_name, "provenance": "measured",
                                "n_models": len(rows), "correlation": stat,
                                "shape": shape, "rows": rows}, indent=2), encoding="utf-8")
    print(f"-> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", action="store_true")
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--corpus-hash", default="")
    ap.add_argument("--probe")
    ap.add_argument("--corpus")
    ap.add_argument("--acts-dir", default="data/activations/local")
    ap.add_argument("--tag", default="external")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    if args.mechanism:
        if not args.corpus_hash:
            raise SystemExit("--mechanism needs --corpus-hash")
        run_mechanism_report(f"mirage_2x2_v{args.corpus_hash}.jsonl")
        return
    if args.layers:
        if not args.corpus_hash:
            raise SystemExit("--layers needs --corpus-hash")
        run_layer_report(args.corpus_hash)
        return
    if not (args.probe and args.corpus):
        raise SystemExit("--probe and --corpus are required without --layers")

    rows = run(args.probe, Path(args.corpus), Path(args.acts_dir), args.seed, args.n_folds)
    if not rows:
        raise SystemExit("no activation files matched the corpus")

    out = {"probe": args.probe, "corpus": Path(args.corpus).name,
           "n_models": len(rows), "rows": rows, "provenance": "measured",
           "seed": args.seed, "n_folds": args.n_folds,
           "recoverability_cross_fitted": True}

    # A gap is only interpretable if the probe has in-distribution signal to lose.
    # This is the D-013 lesson: a detector that cannot read truth on the diagonal
    # either cannot exhibit a confound, and its gap of ~0 means "nothing to lose",
    # not "unconfounded". Report the full set as the headline and the adequately
    # powered subset as a sensitivity check — both, whichever direction they move.
    IN_DIST_FLOOR = 0.70
    strong = [r for r in rows if r["in_dist"] >= IN_DIST_FLOOR]
    out["in_dist_floor"] = IN_DIST_FLOOR
    out["underpowered_models"] = [
        {"model": r["model"], "in_dist": r["in_dist"], "gap": r["gap"]}
        for r in rows if r["in_dist"] < IN_DIST_FLOOR]

    for key, label in (("our_recoverability_truth_beta", "our all-cell probe (representation property)"),
                       ("external_own_recoverability_off_auroc", "the external probe's own fair training")):
        pairs = [(r[key], r["off"]) for r in rows if r.get(key) is not None]
        if len(pairs) > 2:
            x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
            r_all = float(np.corrcoef(x, y)[0, 1])
            entry = {"r": round(r_all, 4), "n": len(pairs), "measured_by": label}
            sp = [(z[key], z["off"]) for z in strong if z.get(key) is not None]
            if len(sp) > 2:
                xs = np.array([p[0] for p in sp]); ys = np.array([p[1] for p in sp])
                entry["r_adequately_powered_only"] = round(float(np.corrcoef(xs, ys)[0, 1]), 4)
                entry["n_adequately_powered"] = len(sp)
            out[f"correlation_{key}"] = entry
            extra = (f"; excluding {len(pairs) - len(sp)} model(s) with in-dist < {IN_DIST_FLOOR}: "
                     f"{entry['r_adequately_powered_only']:.3f} (n={len(sp)})") if len(sp) > 2 else ""
            print(f"\ncorrelation(recoverability [{label}], honest off-diagonal) "
                  f"= {r_all:.3f}  (n={len(pairs)} models){extra}")

    dest = _ROOT / "results" / f"external_sweep_{args.tag}_{Path(args.corpus).stem.split('_v')[-1]}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
