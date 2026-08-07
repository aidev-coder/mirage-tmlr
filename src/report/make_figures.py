"""
Figures, regenerated from results/stage3_saplma_*.json only (never hand drawn).

All figures take `domain=`. Pass domain="cities": the pooled corpus confounds
domain with truth across the diagonal/off-diagonal split, so pooled panels show an
artifact, not a confound (notes/weakness_audit.md A1).

fig_layer_sweep: off diagonal AUROC vs layer, with the in-distribution curve.
fig_cell_scores: fielded P(true) per cell at the headline layer.
fig_causal_manifold: mediation of the fluent-lie error by the frequency manifold.
fig_dissociation: probe readout vs the model's own judgment, per cell.
fig_generation_dissociation: the surviving result — probe vs model on statements
the model generated and then disowned.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _ROOT / "results"
CELL_ORDER = ["TT", "TA", "FT", "FA"]
CELL_LABEL = {"TT": "true common", "TA": "true rare",
              "FT": "false fluent", "FA": "false odd"}


def _saplma_artifacts(domain: str | None = None) -> list[dict]:
    from .artifacts import select
    return select("stage3_saplma_*.json", RESULTS, domain=domain)


def fig_layer_sweep(out: str | Path | None = None, domain: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arts = _saplma_artifacts(domain)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for art in arts:
        L = [e["layer"] for e in art["per_layer"]]
        off = [e["adversarial"]["off_diagonal"]["auroc"] for e in art["per_layer"]]
        ind = [e["adversarial"]["headline_heldout_diagonal"]["auroc"] for e in art["per_layer"]]
        depth = [x / (len(L) - 1) for x in L]
        name = art["model"].split("/")[-1]
        line, = ax.plot(depth, off, marker="o", markersize=3, label=name)
        ax.plot(depth, ind, lw=0.8, ls=":", color=line.get_color(), alpha=0.7)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("relative depth")
    ax.set_ylabel("AUROC")
    ax.set_title("off diagonal (solid) vs in distribution (dotted)\n"
                 "domain held fixed: no collapse", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_layer_sweep.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_cell_scores(out: str | Path | None = None, domain: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arts = _saplma_artifacts(domain)
    fig, axes = plt.subplots(1, len(arts), figsize=(4.2 * len(arts), 3.4), sharey=True)
    if len(arts) == 1:
        axes = [axes]
    for ax, art in zip(axes, arts):
        fh = art.get("fielded_scores_headline")
        if not fh:
            ax.set_title(art["model"].split("/")[-1] + " (no per item scores)")
            continue
        score = np.array(fh["score"])
        cell = np.array(fh["cell"])
        data = [score[cell == c] for c in CELL_ORDER]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(CELL_ORDER) + 1))
        ax.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], rotation=20, fontsize=8)
        ax.axhline(0.5, ls="--", lw=0.8, color="gray")
        ax.set_title(art["model"].split("/")[-1] + f" (L{fh['layer']})", fontsize=9)
    axes[0].set_ylabel("fielded probe P(true)")
    fig.suptitle("domain-pure: the probe rejects fluent falsehood correctly", fontsize=10)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_cell_scores.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_causal_manifold(out: str | Path | None = None, domain: str | None = None):
    """Fraction of the probe's hallucination (FT) error causally mediated by the
    frequency manifold vs a random subspace of equal dimension, swept over k, at
    each model's headline layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .artifacts import select
    arts = select("causal_*.json", RESULTS, domain=domain)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for art in arts:
        e = art["per_layer"][art["headline_layer"]]
        name = art["model"].split("/")[-1]
        man = e["ft_error_frac_mediated_by_k"]
        rnd = e["ft_error_random_subspace_by_k"]
        ks = sorted(int(k) for k in man)
        line, = ax.plot(ks, [man[str(k)] if str(k) in man else man[k] for k in ks],
                        marker="o", markersize=4, label=f"{name} (frequency manifold)")
        ax.plot(ks, [rnd.get(str(k), rnd.get(k)) for k in ks], ls=":", lw=1,
                color=line.get_color(), alpha=0.6)
    ax.axhline(0.0, ls="--", lw=0.8, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("manifold dimension k")
    ax.set_ylabel("fraction of FT error causally mediated")
    ax.set_title("frequency-manifold mediation of the fluent-lie error\n"
                 "(dotted = random subspace of equal size)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_causal_manifold.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_dissociation(out: str | Path | None = None, domain: str | None = None):
    """Per cell, the fielded probe's readout vs the model's own stated P(true), on
    the SAME judgment-prompt activations. The FT gap (probe says true, model says
    false) is the dissociation. Only models that actually perform the judgment task
    are shown (FA behavioral P(true) < 0.15 excludes base models)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .artifacts import select
    arts = [a for a in select("intervene_*.json", RESULTS, domain=domain)
            if "behavioral_baseline" in a["p_true"]["FA"]
            and a["p_true"]["FA"]["behavioral_baseline"] < 0.15]
    cells = ["TT", "TA", "FT", "FA"]
    labels = ["true\ncommon", "true\nrare", "false\nfluent", "false\nodd"]
    fig, axes = plt.subplots(1, len(arts), figsize=(3.6 * len(arts), 3.6), sharey=True)
    if len(arts) == 1:
        axes = [axes]
    x = np.arange(len(cells)); w = 0.38
    for ax, art in zip(axes, arts):
        probe = [art["p_true"][c]["probe_readout"] for c in cells]
        beh = [art["p_true"][c]["behavioral_baseline"] for c in cells]
        ax.bar(x - w / 2, probe, w, label="probe readout", color="#c44")
        ax.bar(x + w / 2, beh, w, label="model's answer", color="#48a")
        ax.axhline(0.5, ls="--", lw=0.8, color="gray")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(art["model"].split("/")[-1], fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("P(true)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("domain-pure: probe and model agree on authored fluent lies", fontsize=11)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_dissociation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _v3_rows(corpus: str | None = None):
    from .artifacts import select
    arts = select("stage3_saplma_*.json", RESULTS, verbose=False)
    if corpus:
        arts = [a for a in arts if a.get("corpus") == corpus]
    rows = []
    for a in arts:
        e = a["per_layer"][a["headline_layer"]]
        adv = e["adversarial"]
        rows.append({
            "model": a["model"].split("/")[-1],
            "in_dist": adv["headline_heldout_diagonal"]["auroc"],
            "off": adv["off_diagonal"]["auroc"],
            "off_ci": adv["off_diagonal"]["ci"],
            "recov": e["mediation_allcell"].get("truth_beta_partialled"),
        })
    return sorted(rows, key=lambda r: r["off"])


def fig_benchmark_blindness(out: str | Path | None = None, corpus: str | None = None):
    """The instrument-validity result. Every probe scores ~1.0 by the measure the
    field reports (held-out diagonal). Their honest off-diagonal scores span 0.11
    to 0.97. The benchmark cannot tell a frequency readout from a truth probe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _v3_rows(corpus)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, r in enumerate(rows):
        lo, hi = r["off_ci"]
        ax.plot([lo, hi], [i, i], color="#c44", lw=2, alpha=.5, solid_capstyle="round")
    ax.scatter([r["in_dist"] for r in rows], y, s=54, marker="D", color="#48a",
               zorder=3, label="in-distribution (what the field reports)")
    ax.scatter([r["off"] for r in rows], y, s=54, color="#c44", zorder=3,
               label="off-diagonal (honest)")
    ax.axvline(0.5, ls="--", lw=.8, color="gray")
    ax.set_yticks(y); ax.set_yticklabels([r["model"] for r in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("AUROC"); ax.set_xlim(0, 1.05)
    ax.text(0.512, -0.6, "chance", fontsize=8, color="gray")
    ax.set_title("every probe looks near-perfect on the benchmark;\n"
                 "honest truth detection ranges from 0.11 to 0.97", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=.92)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_benchmark_blindness.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def fig_recoverability_mechanism(out: str | Path | None = None, corpus: str | None = None):
    """The mechanism: a probe falls back on typicality exactly to the extent that
    truth is not linearly available in the representation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in _v3_rows(corpus) if r["recov"] is not None]
    x = np.array([r["recov"] for r in rows]); y = np.array([r["off"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(x, y, s=60, color="#c44", zorder=3)
    # alternate label placement: several models sit almost on top of each other
    # at the high-recoverability end, so a fixed offset collides.
    for i, r in enumerate(sorted(rows, key=lambda z: (z["recov"], z["off"]))):
        dx, dy = (6, -2) if i % 2 == 0 else (6, 7)
        ha = "left"
        if r["recov"] > 0.92:            # keep the rightmost labels inside the axes
            dx, ha = -6, "right"
        ax.annotate(r["model"], (r["recov"], r["off"]), fontsize=7,
                    xytext=(dx, dy), textcoords="offset points", ha=ha)
    if len(x) > 2:
        b, a0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a0 + b * xs, lw=1, ls="--", color="gray")
        ax.text(.03, .93, f"r = {np.corrcoef(x, y)[0, 1]:.3f}   (n={len(x)} models)",
                transform=ax.transAxes, fontsize=9)
    ax.axhline(0.5, ls=":", lw=.8, color="gray")
    ax.set_xlabel("truth recoverability  (β with fair training)")
    ax.set_ylabel("honest off-diagonal AUROC")
    ax.set_title("the probe substitutes typicality when truth is not available",
                 fontsize=10)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_recoverability_mechanism.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def fig_generation_dissociation(out: str | Path | None = None, domain: str | None = None):
    """Generation time, gold free. On statements the model produced and then judged
    false itself, what does a probe on that same model's hidden states say?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .artifacts import select
    arts = [a for a in select("gendis_*.json", RESULTS, domain=domain)
            if a.get("gold_free_dissociation")]
    names = [a["model"].split("/")[-1] for a in arts]
    probe = [a["gold_free_dissociation"]["probe_mean_p_true_on_disowned"] for a in arts]
    beh = [a["gold_free_dissociation"]["behavior_mean_p_true_on_disowned"] for a in arts]
    ns = [a["gold_free_dissociation"]["n_model_disowns_own_output"] for a in arts]

    fig, ax = plt.subplots(figsize=(6.8, 4))
    x = np.arange(len(arts)); w = 0.36
    ax.bar(x - w / 2, probe, w, label="probe on hidden states", color="#c44")
    ax.bar(x + w / 2, beh, w, label="the model itself", color="#48a")
    for i, (p, n) in enumerate(zip(probe, ns)):
        ax.text(i - w / 2, p - 0.06, f"n={n}", ha="center", fontsize=8, color="white")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=8)
    ax.set_ylabel("P(true)")
    ax.set_ylim(0, 1.18)
    ax.set_title("statements the model wrote, then called false itself\n"
                 "(no ground truth used)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_generation_dissociation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig_layer_sweep())
    print(fig_cell_scores())
    print(fig_causal_manifold())
    print(fig_dissociation())
    print(fig_generation_dissociation())
