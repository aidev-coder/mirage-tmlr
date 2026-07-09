"""
Modal runner for MIRAGE GPU stages (Stage 0 gate, Stage 1 activations).

Run from the repo root ON A MACHINE WITH MODAL CREDENTIALS (owner's local
machine — `modal token` already configured there):

    pip install modal
    modal secret create huggingface HF_TOKEN=hf_...     # once; gated Llama/Gemma need it
    modal run mirage/modal_app.py --stage sanity        # Stage 0 gate, all configured models
    modal run mirage/modal_app.py --stage sanity --model EleutherAI/pythia-1.4b

Results are written back to mirage/results/ locally by the entrypoint, so the
artifact lands in the repo the same as a local run. Activations persist in a
Modal Volume between runs (recomputing hidden states is the main cost, §5).

Hardware target per the project's standing directive §5: single A10G (24 GB) — 8B models in bf16 fit
(~16 GB weights + activations headroom). If a model OOMs, that is a flag for
the owner, not a reason to silently shard.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "mirage"
GPU = "A10G"
_HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44",
        "accelerate>=0.30",
        "sentencepiece>=0.1.99",
        "numpy>=1.26,<3",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "pyyaml>=6.0",
        "requests>=2.28",
    )
    # ship the mirage source tree into the container
    .add_local_dir(str(_HERE), remote_path="/root/mirage")
)

app = modal.App(APP_NAME)

# Persistent caches: HF weights + extracted activations survive across runs.
hf_cache = modal.Volume.from_name("mirage-hf-cache", create_if_missing=True)
activations = modal.Volume.from_name("mirage-activations", create_if_missing=True)

VOLUMES = {"/root/hf-cache": hf_cache, "/root/activations": activations}
SECRETS = [modal.Secret.from_name("huggingface")]


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=3600)
def stage0_sanity(model_id: str) -> dict:
    """Stage 0 gate for one model (the project's standing directive §3 Stage 0)."""
    import os
    import sys

    os.environ["HF_HOME"] = "/root/hf-cache"
    sys.path.insert(0, "/root/mirage")
    from src.substrate import Substrate

    sub = Substrate(model_id, cache_dir="/root/activations")
    report = sub.sanity_check()
    activations.commit()
    hf_cache.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "sanity", model: str = ""):
    import yaml

    if stage != "sanity":
        raise SystemExit(f"unknown/not-yet-implemented stage: {stage}")

    cfg = yaml.safe_load((_HERE / "configs" / "models.yaml").read_text())
    ids = [model] if model else [m["id"] for m in cfg["substrates"]]

    # run models in parallel on separate containers
    results = list(stage0_sanity.map(ids))
    for rep in results:
        print(f"{'PASS' if rep.get('pass') else 'FAIL'}  {rep['model']}  "
              f"vram={rep.get('vram_gb', '?')}GB")

    out = _HERE / "results" / "stage0_sanity.json"
    out.write_text(json.dumps(results, indent=2))
    ok = all(r.get("pass") for r in results)
    print(f"stage 0 gate: {'PASS' if ok else 'FAIL'} -> {out}")
    print("reminder: commit the artifact + a the project notebook entry (the project's standing directive §3/§7)")
