"""
Stage 4 — the diagnosis table, regenerated from results/stage3_*.json ONLY.

Tables are generated, never authored (). Any number in the writeup
that cannot be traced to a results artifact via this script does not exist.

The spine of the paper is the consistency of the gap across models. Per model,
at its own headline (mid-depth) layer, the fielded instrument (the field's
diagonal-trained recipe) is audited by all three tests; the all-cell probe gives
the recoverability control.

Columns per (detector, model):
  in-dist AUROC (3c heldout diagonal) | off-diagonal AUROC (3c) | gap [CI]
  | FT-cell mean P(true) | 3a fielded gap [CI] | recoverability (all-cell truth β)
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _ROOT / "results"


def _auroc(entry: dict | None) -> str:
    if not entry or entry.get("auroc") is None:
        return "—"
    lo, hi = entry["ci"]
    return f"{entry['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"


def _gap(gap: dict | None) -> str:
    if not gap:
        return "—"
    lo, hi = gap["ci"]
    star = "*" if gap.get("excludes_zero") else ""
    val = gap.get("gap", gap.get("point"))
    return f"{val:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def _row(art: dict) -> dict:
    e = art["per_layer"][art["headline_layer"]]
    adv = e["adversarial"]
    return {
        "detector": art["detector"],
        "model": art["model"].split("/")[-1],
        "layer": art["headline_layer"],
        "in_dist": _auroc(adv["headline_heldout_diagonal"]),
        "off_diag": _auroc(adv["off_diagonal"]),
        "gap": _gap(adv["gap"]),
        "ft_mean": f"{e['fielded_cell_scores'].get('FT', {}).get('mean', float('nan')):.3f}",
        "strat_gap": _gap(e["stratified_fielded"].get("gap")),
        "recover": ("n/a" if e["mediation_allcell"].get("truth_beta_partialled") is None
                    else f"{e['mediation_allcell']['truth_beta_partialled']:.3f}"),
        "recover_auroc": _auroc(e.get("allcell_off_diagonal")),
        "artifact": None,
    }


def diagnosis_table(domain: str | None = None) -> str:
    """domain: restrict to runs with this domain tag. The pooled corpus confounds
    domain with truth across cells (2026-08-03), so a pooled table is not a valid
    headline — pass domain="cities" for the domain-pure result."""
    from .artifacts import select
    rows = []
    for art in select("stage3_*.json", RESULTS, domain=domain):
        r = _row(art)
        r["artifact"] = f"{art.get('model','?')} [{art.get('domain','all')}]"
        rows.append(r)
    if not rows:
        return "(no stage3 artifacts in results/ yet)"

    hdr = ("| Detector | Model | L | In-dist AUROC (3c) | Off-diagonal AUROC (3c) "
           "| Gap [95% CI] | FT mean P(true) | 3a fielded gap [CI] | Recoverable off-diag AUROC "
           "| Recoverability (all-cell β) |")
    sep = "|" + "---|" * 10
    lines = [hdr, sep] + [
        f"| {r['detector']} | {r['model']} | {r['layer']} | {r['in_dist']} | {r['off_diag']} "
        f"| {r['gap']} | {r['ft_mean']} | {r['strat_gap']} | {r['recover_auroc']} | {r['recover']} |"
        for r in rows]
    lines.append("")
    lines.append("`*` = CI excludes zero. Overlapping CIs are not a difference (.5).")
    lines.append("In-dist = held-out diagonal (the field's reported number). Off-diagonal = "
                 "truth detection on TA+FT. FT mean P(true) is the fluent-lie cell: BELOW 0.5 means "
                 "the probe correctly rejects fluent falsehood. Recoverability = truth β under an "
                 "all-cell (fairly trained) probe, typicality+fragmentation partialled out.")
    doms = sorted({r["artifact"].split("[")[-1].rstrip("]") for r in rows})
    lines.append(f"Domain scope: {', '.join(doms)}. A POOLED (all-domain) row is NOT a valid "
                 "headline for this corpus: domain is confounded with truth across the "
                 "diagonal/off-diagonal split, which by itself drives the off-diagonal below "
                 "chance (see notes/weakness_audit.md A1). Use domain=\"cities\".")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    dom = sys.argv[1] if len(sys.argv) > 1 else None
    table = diagnosis_table(domain=dom)
    print(table)
    out = RESULTS / "stage4_diagnosis_table.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\n-> {out}")
