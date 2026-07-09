"""
Stage 4 — the diagnosis table, regenerated from results/*.json ONLY.

Tables are generated, never authored (the project's standing directive §2). Any number in the writeup
that cannot be traced to a results artifact via this script does not exist.

Table columns per (detector, model):
  headline AUROC | typicality-controlled AUROC (3a) | off-diagonal AUROC (3c) | gap [CI]
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _ROOT / "results"


def _fmt(entry: dict | None) -> str:
    if not entry or entry.get("auroc") is None:
        return "—"
    lo, hi = entry["ci"]
    return f"{entry['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_gap(gap: dict | None) -> str:
    if not gap:
        return "—"
    lo, hi = gap["ci"]
    star = "*" if gap.get("excludes_zero") else ""
    return f"{gap.get('gap', gap.get('point')):+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def diagnosis_table() -> str:
    """Assemble the main table from stage1_* (headline) and stage3_* artifacts."""
    rows = []
    for f in sorted(RESULTS.glob("stage3_*.json")):
        art = json.loads(f.read_text())
        rows.append({
            "detector": art["detector"], "model": art["model"],
            "headline": _fmt(art.get("headline")),
            "stratified": _fmt({"auroc": art.get("stratified", {}).get("within_band_auroc_weighted"),
                                "ci": art.get("stratified", {}).get("gap", {}).get("ci", [0, 0])})
            if art.get("stratified", {}).get("within_band_auroc_weighted") is not None else "—",
            "off_diagonal": _fmt(art.get("adversarial", {}).get("off_diagonal")),
            "gap": _fmt_gap(art.get("adversarial", {}).get("gap")),
            "artifact": f.name,
        })
    if not rows:
        return "(no stage3 artifacts in results/ yet)"

    hdr = ("| Detector | Model | Headline AUROC | Typicality-controlled (3a) "
           "| Off-diagonal (3c) | Gap [95% CI] | Artifact |")
    sep = "|" + "---|" * 7
    lines = [hdr, sep] + [
        f"| {r['detector']} | {r['model']} | {r['headline']} | {r['stratified']} "
        f"| {r['off_diagonal']} | {r['gap']} | `{r['artifact']}` |" for r in rows]
    lines.append("\n`*` gap CI excludes zero. Overlapping CIs are not a difference.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(diagnosis_table())
