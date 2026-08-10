"""
Beam runner for the external-probe layer sweep.

Modal remains the primary backend (`modal_app.py`); this exists because the Modal
credits ran out mid-campaign and the one outstanding job is the four-probe layer
sweep on the released corpus. Only that stage is ported. Nothing here is imported
by the Modal path, so a failure on this backend cannot affect it.

Two differences from Modal that shape the code. Beam's free tier allows five
concurrent GPU containers, so eighteen models arrive in four waves rather than at
once. And `.map()` takes a single positional argument per call, so the corpus and
probe list are module constants instead of mapped keyword arguments.

    cd mirage && python beam_app.py --models all
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from beam import Image, Volume, function

_HERE = Path(__file__).resolve().parent
CORPUS = "mirage_2x2_v44b4126cba1c.jsonl"
PROBES = [
    "mirage_hardness.probes_external.geometry_of_truth_mmprobe:GeometryOfTruthMMProbe",
    "mirage_hardness.probes_external.geometry_of_truth_lrprobe:GeometryOfTruthLRProbe",
    "mirage_hardness.probes_external.ccs:CCSProbeAdapter",
    "mirage_hardness.probes_external.ttpd:TTPDProbe",
]
MODELS = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-2-9b-it",
]

image = Image(python_version="python3.11", python_packages=[
    "torch>=2.1", "transformers>=4.44,<5", "accelerate>=0.30",
    "sentencepiece>=0.1.99", "numpy>=1.26,<3", "scipy>=1.11",
    "scikit-learn>=1.3", "statsmodels>=0.14", "pyyaml>=6.0", "requests>=2.28",
])

hf_cache = Volume(name="mirage-hf-cache", mount_path="./hf-cache")


# Only A10G, RTX4090 (24GB) and RTX5090 (32GB) are serverless on the free tier;
# H100 and A100 exist but are on-demand only and need a reserved machine, so
# `gpu="H100"` is never schedulable here. A list lets the scheduler take whatever
# is free. 32GB is the ceiling, which covers every model in bf16 except
# pythia-12b, whose weights alone are about 24GB.
SERVERLESS_GPUS = ["RTX5090", "A10G", "RTX4090"]


@function(image=image, gpu=SERVERLESS_GPUS, cpu=4, memory=32768, timeout=4 * 3600,
          volumes=[hf_cache], secrets=["HF_TOKEN"], retries=0)
def layer_sweep(model_id: str) -> dict:
    """Audit every external probe at every layer for one model."""
    import os
    import sys

    os.environ["HF_HOME"] = os.path.abspath("./hf-cache")
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", ""))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import importlib

    import numpy as np

    from src.corpus_gen import negate_all
    from src.eval.adversarial_split import adversarial_split
    from src.stage3 import load_corpus
    from src.substrate import Substrate

    def _load(spec: str):
        mod, _, attr = spec.partition(":")
        return getattr(importlib.import_module(mod), attr)

    factories = {spec: _load(spec) for spec in PROBES}
    items = load_corpus(str(Path(__file__).resolve().parent / "data" / "corpus" / CORPUS))
    texts = [it["text"] for it in items]
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    domains = np.array([it.get("domain", "") for it in items])
    n_topics = len(set(domains.tolist()))

    sub = Substrate(model_id, cache_dir="./hf-cache")
    H = sub.hidden_states_matrix(texts, batch_size=32)
    H_neg = sub.hidden_states_matrix(negate_all(texts), batch_size=32)
    n_layers = H.shape[1]

    rows = []
    for L in range(n_layers):
        X = H[:, L, :].astype(np.float64)
        Xn = H_neg[:, L, :].astype(np.float64)
        for spec, factory in factories.items():
            make = (lambda f=factory: f(seed=0))
            if getattr(factory, "needs_topics", False):
                from mirage_hardness.probes_external.ttpd import stack as ttpd_stack
                Xin = ttpd_stack(X, Xn, domains)
                make = (lambda f=factory: f(seed=0, n_topics=n_topics))
            elif getattr(factory, "paired", False):
                from mirage_hardness.probes_external.ccs import stack
                Xin = stack(X, Xn)
            else:
                Xin = X
            try:
                adv = adversarial_split(Xin, truth, cells, make, seed=0)
            except Exception as exc:
                rows.append({"probe": spec, "layer": L, "error": repr(exc)})
                continue
            rows.append({
                "probe": spec, "layer": L,
                "in_dist": adv["headline_heldout_diagonal"]["auroc"],
                "off": adv["off_diagonal"]["auroc"],
                "off_ci": adv["off_diagonal"]["ci"],
                "gap": adv["gap"]["gap"], "gap_ci": adv["gap"]["ci"],
                "gap_excludes_zero": adv["gap"].get("excludes_zero"),
            })
        print(f"  L{L}/{n_layers - 1} done", flush=True)

    return {"model": model_id, "corpus": CORPUS, "n_layers": n_layers, "seed": 0,
            "negated_pass": True, "rows": rows, "provenance": "measured",
            "backend": "beam"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="'all' or a comma-separated list of model ids")
    args = ap.parse_args()
    ids = MODELS if args.models == "all" else [
        m.strip() for m in args.models.split(",") if m.strip()]

    chash = CORPUS.split("_v")[-1].split(".")[0]
    out_dir = _HERE / "results"
    print(f"beam layer sweep: {len(ids)} models x {len(PROBES)} probes on {CORPUS}")
    done = 0
    for res in layer_sweep.map(ids):
        if not isinstance(res, dict) or "model" not in res:
            print(f"SKIPPED (failed): {str(res)[:200]}")
            continue
        short = res["model"].split("/")[-1].lower()
        out = out_dir / f"extlayers_{short}_{chash}_{date.today():%Y%m%d}.json"
        # merging, like the Modal path: re-running one probe set must not drop the rest
        if out.exists():
            prior = json.loads(out.read_text(encoding="utf-8"))
            fresh = {r["probe"] for r in res["rows"]}
            res["rows"] = [r for r in prior.get("rows", [])
                           if r["probe"] not in fresh] + res["rows"]
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        done += 1
        print(f"[{done}/{len(ids)}] {res['model']}: {res['n_layers']} layers -> {out}")


if __name__ == "__main__":
    main()
