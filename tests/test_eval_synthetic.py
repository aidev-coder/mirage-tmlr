"""
Synthetic self-test of the Stage-3 eval stack (CPU, no models, ~1 min).

Per the project's standing directive §1.4 the instrument may be measuring itself — so before any real
probe is audited, the audit tools themselves are validated on a world where the
ground truth is known BY CONSTRUCTION:

  world      truth y ∈ {0,1}; typicality t ~ N(±1, 1) correlated with y on the
             benchmark-style corpus (true items typical, like TruthfulQA-style
             sets), decorrelated on the crossed corpus.
  confounded probe   score = f(typicality) + noise. Reads NO truth at all.
  honest probe       score = f(truth) + noise.

Required outcomes (asserted):
  A. Confounded probe posts a HIGH headline AUROC on the benchmark-style corpus
     (this is the illusion MIRAGE hunts).
  B. Stage 3a: its within-band AUROC collapses toward 0.5; the honest probe's
     survives.
  C. Stage 3b: its partialled truth-beta ≈ 0; the honest probe's is stable.
  D. Stage 3c: trained on diagonal cells it collapses on the off-diagonal
     (gap CI excludes 0); the honest probe's gap CI does not exclude 0.

If any assertion fails, the eval stack cannot be trusted on real probes.
Writes results/selftest_synthetic_<date>.json. Provenance: measured (synthetic).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.adversarial_split import adversarial_split
from src.eval.mediation import mediation
from src.eval.stratified_auroc import stratified_auroc
from src.stats import auroc_with_ci

SEED = 20260709
N = 2000
NOISE = 0.6


def make_world(rng, crossed: bool):
    """truth, typicality, cells. Benchmark-style: corr(truth, typicality) high.
    Crossed: independent by construction (the MIRAGE 2x2)."""
    y = rng.integers(0, 2, N)
    if crossed:
        typical = rng.integers(0, 2, N).astype(bool)
    else:
        # benchmark-style: true items are mostly typical, false mostly atypical
        typical = rng.random(N) < np.where(y == 1, 0.85, 0.15)
    t = rng.normal(np.where(typical, -1.0, 1.0), 1.0)  # high t = atypical (high ppl)
    cells = np.where(y & typical, "TT",
             np.where(y & ~typical, "TA",
             np.where(~y & typical, "FT", "FA")))
    return y, t, cells


def confounded_score(t, rng):
    """Reads typicality only: low perplexity -> 'true'. Never sees truth."""
    return -t + rng.normal(0, NOISE, len(t))


def honest_score(y, rng):
    return y + rng.normal(0, NOISE, len(y))


class LinearProbe:
    """Minimal fit/score probe over feature matrix X for the 3c harness."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


def main() -> dict:
    rng = np.random.default_rng(SEED)
    report: dict = {"seed": SEED, "n": N, "provenance": "measured (synthetic)"}
    failures = []

    def check(name, cond, detail):
        report[name] = {"pass": bool(cond), **detail}
        print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
        if not cond:
            failures.append(name)

    # ── A. the illusion: confounded probe looks great on a benchmark-style corpus
    y_b, t_b, _ = make_world(rng, crossed=False)
    s_conf_b = confounded_score(t_b, rng)
    headline = auroc_with_ci(y_b, s_conf_b, seed=SEED)
    print("A. headline on benchmark-style corpus (confounded probe)")
    check("A_headline_illusion", headline["auroc"] >= 0.75, {"headline": headline})

    # ── crossed corpus for the three tests
    y, t, cells = make_world(rng, crossed=True)
    s_conf = confounded_score(t, rng)
    s_hon = honest_score(y, rng)

    # On the crossed corpus the confounded probe should already be ~chance
    # pooled — but 3a must ALSO kill it on the benchmark-style corpus, which is
    # the realistic use: stratification recovers the truth signal (or absence)
    # without needing the crossed corpus at all.
    print("B. stage 3a — stratified AUROC (benchmark-style corpus)")
    strat_conf = stratified_auroc(s_conf_b, y_b, t_b, seed=SEED)
    y_hb, t_hb, _ = make_world(rng, crossed=False)
    strat_hon = stratified_auroc(honest_score(y_hb, rng), y_hb, t_hb, seed=SEED)
    check("B_confounded_collapses",
          strat_conf["within_band_auroc_weighted"] <= 0.60
          and strat_conf["gap"]["excludes_zero"],
          {"within_band": strat_conf["within_band_auroc_weighted"],
           "headline": strat_conf["headline_pooled"]["auroc"],
           "gap": strat_conf["gap"]})
    check("B_honest_survives",
          strat_hon["within_band_auroc_weighted"] >= 0.75,
          {"within_band": strat_hon["within_band_auroc_weighted"],
           "headline": strat_hon["headline_pooled"]["auroc"]})
    report["B_stratified_confounded"] = strat_conf
    report["B_stratified_honest"] = strat_hon

    # ── C. stage 3b — mediation (benchmark-style corpus, same realism argument)
    print("C. stage 3b — mediation")
    med_conf = mediation(s_conf_b, y_b, t_b)
    med_hon = mediation(honest_score(y_hb, rng), y_hb, t_hb)
    check("C_confounded_beta_dies",
          abs(med_conf["truth_beta_partialled"]) <= 0.10
          and med_conf["truth_beta_shrinkage"] >= 0.7,
          {"beta_marginal": med_conf["truth_beta_marginal"],
           "beta_partialled": med_conf["truth_beta_partialled"],
           "shrinkage": med_conf["truth_beta_shrinkage"]})
    check("C_honest_beta_stable",
          med_hon["truth_beta_partialled"] >= 0.5
          and (med_hon["truth_beta_shrinkage"] or 0) <= 0.3,
          {"beta_marginal": med_hon["truth_beta_marginal"],
           "beta_partialled": med_hon["truth_beta_partialled"],
           "shrinkage": med_hon["truth_beta_shrinkage"]})
    report["C_mediation_confounded"] = med_conf
    report["C_mediation_honest"] = med_hon

    # ── D. stage 3c — adversarial split on the crossed corpus.
    # Features encode what each probe "reads": the confounded probe's feature is
    # typicality; the honest probe's feature is a noisy truth signal.
    print("D. stage 3c — adversarial split")
    X_conf = np.column_stack([t + rng.normal(0, NOISE, N)])
    X_hon = np.column_stack([y + rng.normal(0, NOISE, N)])
    adv_conf = adversarial_split(X_conf, y, cells, LinearProbe, seed=SEED)
    adv_hon = adversarial_split(X_hon, y, cells, LinearProbe, seed=SEED)
    check("D_confounded_gap",
          adv_conf["gap"]["excludes_zero"] and adv_conf["gap"]["gap"] >= 0.15,
          {"headline": adv_conf["headline_heldout_diagonal"]["auroc"],
           "off_diagonal": adv_conf["off_diagonal"]["auroc"],
           "gap": adv_conf["gap"]})
    check("D_honest_no_gap",
          not adv_hon["gap"]["excludes_zero"],
          {"headline": adv_hon["headline_heldout_diagonal"]["auroc"],
           "off_diagonal": adv_hon["off_diagonal"]["auroc"],
           "gap": adv_hon["gap"]})
    report["D_adversarial_confounded"] = adv_conf
    report["D_adversarial_honest"] = adv_hon

    report["all_pass"] = not failures
    out = (Path(__file__).resolve().parent.parent / "results"
           / f"selftest_synthetic_{date.today().strftime('%Y%m%d')}.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    print(f"artifact: {out}")
    if failures:
        raise SystemExit(1)
    return report


# pytest entry point
def test_eval_stack_synthetic():
    assert main()["all_pass"]


if __name__ == "__main__":
    main()


def _ccs_world(n=400, d=64, seed=0, rogue=False, negation_offset=0.0):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(bool)
    direction = np.zeros(d)
    direction[0] = 1.0
    shared = rng.normal(size=(n, d))
    signal = np.where(y, 1.0, -1.0)[:, None] * direction * 3.0
    pos, neg = shared + signal, shared - signal
    if negation_offset:
        # the constant "contains not" direction; per-side scaling must remove it
        bump = np.zeros(d)
        bump[3] = negation_offset
        neg = neg + bump
    if rogue:
        pos[:, 7] += 300.0
        neg[:, 7] += 300.0
    return pos, neg, y


def test_ccs_adapter_is_invariant_to_a_constant_negation_offset():
    """Burns et al. z-score x+ and x- separately so the constant "contains not"
    direction cannot be read. Under per-side scaling a constant added to the
    negation side cancels exactly; under shared scaling it perturbs the positive
    side's normalization and the scores move. This pins the normalization itself,
    not a downstream AUROC that stays high either way."""
    import numpy as np
    import pytest
    pytest.importorskip("torch")
    from mirage_hardness.probes_external.ccs import CCSProbeAdapter, stack

    pos, neg, y = _ccs_world()
    plain = CCSProbeAdapter(seed=0, epochs=200).fit(stack(pos, neg), y).score(stack(pos, neg))
    bumped = np.zeros(pos.shape[1])
    bumped[3] = 50.0
    X2 = stack(pos, neg + bumped)
    shifted = CCSProbeAdapter(seed=0, epochs=200).fit(X2, y).score(X2)
    np.testing.assert_allclose(plain, shifted, rtol=1e-5, atol=1e-6)


def test_ccs_adapter_reads_truth_without_labels():
    import numpy as np
    import pytest
    pytest.importorskip("torch")
    from mirage_hardness.probes_external.ccs import CCSProbeAdapter, stack
    from src.stats import auroc_with_ci

    for rogue in (False, True):
        pos, neg, y = _ccs_world(rogue=rogue)
        X = stack(pos, neg)
        s = CCSProbeAdapter(seed=0, epochs=400).fit(X, y).score(X)
        assert auroc_with_ci(y, s, n_boot=200)["auroc"] > 0.9, rogue
        assert 0.0 < float(s.min()) and float(s.max()) < 1.0, "scores saturated"

    a = CCSProbeAdapter(seed=0, epochs=200).fit(X, y).score(X)
    b = CCSProbeAdapter(seed=0, epochs=200).fit(X, y).score(X)
    np.testing.assert_allclose(a, b)


def test_ccs_adapter_rejects_a_bare_matrix():
    import pytest
    pytest.importorskip("torch")
    from mirage_hardness.probes_external.ccs import CCSProbeAdapter, stack

    pos, neg, y = _ccs_world(d=63)
    with pytest.raises(ValueError):
        CCSProbeAdapter(seed=0, epochs=10).fit(pos, y)
    with pytest.raises(ValueError):
        stack(pos, neg[:5])
