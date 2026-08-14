"""Shared loading, style and provenance for the data figures.

Schematics (the 2x2, the apparatus, the architecture) are designed in Figma and live in
results/ as exported images. This package covers only figures that plot measurements, and
holds them to one contract:

  * every number is read from a committed artifact under results/; a missing key raises,
    naming the key, rather than falling back to a default;
  * source filenames and their sha256 prefixes go into the PDF metadata and stdout;
  * vector PDF plus 300 dpi PNG, identical across runs (fixed seed for any jitter).

Headless: matplotlib is set to Agg before pyplot is imported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
CORPUS = "44b4126cba1c"
HARVEST = "d279c2cae5f4"

# Okabe-Ito, colourblind-safe.
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F",
}
SINGLE, DOUBLE = 5.5, 6.9


def style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


class Sources:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def load(self, path: Path) -> dict:
        raw = Path(path).read_bytes()
        self.files[Path(path).name] = hashlib.sha256(raw).hexdigest()[:12]
        return json.loads(raw.decode("utf-8"))

    def load_glob(self, pattern: str) -> list[dict]:
        hits = sorted(RESULTS.glob(pattern))
        if not hits:
            raise FileNotFoundError(f"no artifact matches results/{pattern}")
        return [self.load(h) for h in hits]

    def stamp(self) -> str:
        return "; ".join(f"{k}@{v}" for k, v in sorted(self.files.items()))


def need(d, *path, where: str = ""):
    """Fetch a nested key or raise naming it. Never substitutes."""
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
            cur = cur[k]
        else:
            trail = ".".join(str(p) for p in path)
            raise KeyError(f"missing artifact key {trail!r}" + (f" in {where}" if where else ""))
    return cur


def chance(ax, axis: str = "x", label: bool = True) -> None:
    kw = dict(ls=(0, (4, 3)), lw=0.8, color=OI["grey"], zorder=0)
    if label:
        kw["label"] = "chance"
    (ax.axvline if axis == "x" else ax.axhline)(0.5, **kw)


def save(fig, name: str, src: Sources, title: str):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Title": title, "Subject": "sources: " + src.stamp(),
            "Creator": f"mirage/src/figures/{name}", "Keywords": src.stamp()}
    fig.savefig(OUT / f"{name}.pdf", metadata=meta)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"{name}: pdf+png -> results/figures/")
    print(f"  sources: {src.stamp()}")


def short(mid: str) -> str:
    return mid.split("/")[-1]
