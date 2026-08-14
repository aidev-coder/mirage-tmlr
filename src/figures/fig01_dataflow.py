"""Figure 1 — the study dataflow.

Structure comes from paper/schematics/dataflow.json; the three counts in the node
subtitles come from artifacts. Nothing numeric is typed into either file.

The diagram encodes three claims an earlier draft got wrong, so they are checked before
anything is drawn and the script raises rather than rendering a diagram that asserts them:

  * the released dataset does not feed corpus construction as training data — its role is
    the transfer path straight into probe training;
  * the training-free baselines are not derived from a corpus, so they take no incoming
    corpus edge;
  * the checker is an audit layer over every stage, not a downstream sink, so it receives
    no solid arrow.

    python -m src.figures.fig01_dataflow
"""

from __future__ import annotations

import csv
import json

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from ._common import CORPUS, DOUBLE, HARVEST, OI, RESULTS, ROOT, Sources, plt, save, style

SPEC = ROOT / "paper" / "schematics" / "dataflow.json"
INK = "#16181D"
MUTED = "#5A6068"
EDGE = "#8A9099"
FILL = "#F4F6F9"
LAYER = "#EAF1F8"


def counts(src: Sources) -> dict:
    """The only numbers in this figure, each read rather than typed."""
    man = src.load(ROOT / "data" / "raw" / "azaria_mitchell_official" / "MANIFEST.json")
    am_rows = man["total_rows"]                       # MANIFEST.json total_rows
    def lines(p):
        src.files[p.name] = "corpus"                  # recorded in the provenance stamp
        return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    return {
        "am_rows": f"{am_rows:,}",
        "authored_n": lines(RESULTS.parent / "data" / "corpus" / f"mirage_2x2_v{CORPUS}.jsonl"),
        "harvested_n": lines(RESULTS.parent / "data" / "corpus" / f"mirage_2x2_v{HARVEST}.jsonl"),
    }


def validate(spec: dict) -> None:
    nodes = {n["id"] for n in spec["nodes"]}
    edges = spec["edges"]
    by_id = {e["id"]: e for e in edges}

    e8 = by_id.get("e8")
    if e8 is None or e8["style"] != "dashed" or {e8["from"], e8["to"]} != {"n1", "n7"}:
        raise ValueError("e8 must exist, be dashed, and touch exactly n1 and n7")

    if any(e["to"] == "n9" for e in edges):
        raise ValueError("n9 (training-free baselines) must have no incoming edge")

    if any(e["to"] == "n10" or e["from"] == "n10" for e in edges):
        raise ValueError("n10 (checker) must receive no data arrow; it is an audit layer")

    if any(e["from"] == "n1" and e["to"] == "n4" and e["style"] != "solid" for e in edges):
        raise ValueError("e1 must be solid")
    if any(e["from"] == "n1" and e["to"] == "n5" for e in edges):
        raise ValueError("the released dataset must not flow into the corpus as training data")

    banned = ("honest", "retracted", "withdrawn", "robust", "comprehensive", "novel", "crucial")
    blob = " ".join([n["title"] for n in spec["nodes"]]
                    + [n.get("sub", "") for n in spec["nodes"]]
                    + [n.get("annotation", "") for n in spec["nodes"]]
                    + [e["label"] for e in edges]
                    + [l["text"] for l in spec["legend"]]).lower()
    hit = [w for w in banned if w in blob]
    if hit:
        raise ValueError(f"banned term in a label: {hit}")

    missing = [e["id"] for e in edges if e["from"] not in nodes or e["to"] not in nodes]
    if missing:
        raise ValueError(f"edge endpoints not in the node set: {missing}")
    print(f"  validated: {len(spec['nodes'])} nodes, {len(edges)} edges, "
          "e8 dashed n1->n7, n9 has no source, n10 takes no data arrow")


def box(ax, n, subs):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    face = LAYER if n.get("layer") else FILL
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=0.9, edgecolor="#B9C0C9", facecolor=face, zorder=3))
    ax.text(x, y + 1.5, n["title"], ha="center", va="center", fontsize=7.9,
            fontweight="bold", color=INK, zorder=4)
    ax.text(x, y - 2.2, n["sub"].format(**subs), ha="center", va="center",
            fontsize=6.8, color=MUTED, zorder=4)
    if n.get("annotation"):
        ax.text(x, y - h / 2 - 2.6, n["annotation"], ha="center", va="top",
                fontsize=6.2, color=MUTED, style="italic", zorder=4)


def anchor(n, side):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    return {"top": (x, y + h / 2), "bottom": (x, y - h / 2),
            "left": (x - w / 2, y), "right": (x + w / 2, y)}[side]


def arrow(ax, pts, style, label=None, lpos=0.5, lofs=(0, 1.6), ha="center"):
    dash = (0, (4, 2.6)) if style == "dashed" else "-"
    for i in range(len(pts) - 1):
        last = (i == len(pts) - 2)
        ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="-|>" if last else "-", color=EDGE,
                                    linewidth=0.95, linestyle=dash,
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=9), zorder=2)
    if label:
        i = max(0, int(len(pts) * lpos) - 1)
        mx = (pts[i][0] + pts[i + 1][0]) / 2 + lofs[0]
        my = (pts[i][1] + pts[i + 1][1]) / 2 + lofs[1]
        ax.text(mx, my, label, fontsize=6.3, color=MUTED, ha=ha, va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))


def main() -> int:
    style()
    src = Sources()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    validate(spec)
    subs = counts(src)
    N = {n["id"]: n for n in spec["nodes"]}

    fig, ax = plt.subplots(figsize=(DOUBLE, 4.6), layout="constrained")
    ax.set_xlim(0, 172)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- edges first so boxes sit on top -------------------------------------
    arrow(ax, [anchor(N["n1"], "bottom"), (26, 77), (N["n4"]["x"] - 13, 77),
               (N["n4"]["x"] - 13, N["n4"]["y"] + 5.5)], "solid",
          "source statements", lpos=0.34, lofs=(-1.5, 1.0), ha="right")
    arrow(ax, [anchor(N["n2"], "bottom"), (78, 77), (N["n4"]["x"] + 13, 77),
               (N["n4"]["x"] + 13, N["n4"]["y"] + 5.5)], "solid",
          "frequency axis", lpos=0.34, lofs=(1.5, 1.0), ha="left")
    arrow(ax, [anchor(N["n3"], "bottom"), anchor(N["n6"], "top")], "solid",
          "wrong answers only", lofs=(1.8, 0.4), ha="left")

    # e4 carries the gates: a filter glyph on the edge itself, not a box in the chain
    a, b = anchor(N["n4"], "bottom"), anchor(N["n5"], "top")
    arrow(ax, [a, b], "solid")
    gy = 57.0
    for dy, half in ((1.4, 3.4), (0.0, 2.2), (-1.4, 1.0)):
        ax.plot([a[0] - half, a[0] + half], [gy + dy, gy + dy], color=EDGE, lw=1.1, zorder=4)
    ax.text(a[0] - 5.0, gy, "through five gates", fontsize=6.3, color=MUTED,
            ha="right", va="center", zorder=5)

    arrow(ax, [anchor(N["n5"], "right"), anchor(N["n7"], "left")], "solid",
          "TT+FA diagonal,\ntrained on", lofs=(0, 4.6))
    arrow(ax, [anchor(N["n5"], "bottom"), (N["n5"]["x"], 24), (anchor(N["n8"], "left")[0], 24)],
          "solid", "TA+FT, evaluated only", lpos=0.34, lofs=(-2.0, -3.6), ha="left")
    arrow(ax, [anchor(N["n6"], "bottom"), (N["n6"]["x"], 34), (N["n8"]["x"] + 9, 34),
               (N["n8"]["x"] + 9, anchor(N["n8"], "top")[1])], "solid",
          "evaluated only,\nnever trained on", lpos=0.34, lofs=(2.0, 0.4), ha="left")

    # e8: the transfer experiment, down the left margin, bypassing n4 and n5 entirely
    arrow(ax, [anchor(N["n1"], "left"), (6, N["n1"]["y"]), (6, 51),
               (N["n7"]["x"] - 10, 51), (N["n7"]["x"] - 10, anchor(N["n7"], "top")[1])],
          "dashed", "transfer: trained on their data,\nevaluated on ours",
          lpos=0.62, lofs=(-16, 1.6))

    arrow(ax, [anchor(N["n7"], "bottom"), anchor(N["n8"], "top")], "solid",
          "probe scores", lofs=(1.4, 0.2), ha="left")
    arrow(ax, [anchor(N["n9"], "left"), anchor(N["n8"], "right")], "solid",
          "scored on\nthe same items", lofs=(0, 6.0))

    # ---- audit layer: dashed ticks upward, no data arrows ---------------------
    top_of_layer = N["n10"]["y"] + N["n10"]["h"] / 2
    for tid in spec["audit"]["touches"]:
        t = N[tid]
        ax.plot([t["x"] - 15, t["x"] - 15], [top_of_layer, t["y"] - t["h"] / 2],
                color=EDGE, lw=0.85, ls=(0, (2.5, 2.5)), zorder=1)

    for n in spec["nodes"]:
        box(ax, n, subs)

    ax.legend(handles=[
        Line2D([], [], color=EDGE, lw=1.0, label="data flow"),
        Line2D([], [], color=EDGE, lw=1.0, ls=(0, (4, 2.6)), label="contrast under audit"),
        Line2D([], [], color=EDGE, lw=0.85, ls=(0, (2.5, 2.5)), label="audited"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=7)

    save(fig, "fig01_dataflow", src, "Study dataflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
