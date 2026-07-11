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


def _json_default(o):
    """Coerce numpy scalars/arrays returned across the Modal boundary so an
    artifact write can never fail on a stray numpy type (MIRAGE §1.2).
    A committed artifact is a hard gate precondition; serialization must not
    be the thing that loses it."""
    if hasattr(o, "tolist"):        # numpy arrays AND numpy scalars (np.bool_, np.float64, ...)
        return o.tolist()
    if hasattr(o, "item"):          # defensive fallback for 0-d scalar-likes
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44,<5",
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


def _setup():
    import os
    import sys

    os.environ["HF_HOME"] = "/root/hf-cache"
    sys.path.insert(0, "/root/mirage")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=3600)
def stage0_sanity(model_id: str) -> dict:
    """Stage 0 gate for one model (the project's standing directive §3 Stage 0)."""
    _setup()
    from src.substrate import Substrate

    sub = Substrate(model_id, cache_dir="/root/activations")
    report = sub.sanity_check()
    activations.commit()
    hf_cache.commit()
    return report


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=6 * 3600)
def stage1_saplma(model_id: str, max_per_topic: int = 0) -> dict:
    """Stage 1 SAPLMA headline reproduction for one model (the project's standing directive §3 Stage 1).

    Requires data/raw/ to be populated locally BEFORE `modal run` (the image
    ships the local mirage/ tree, so run `python mirage/data/raw/fetch.py` first).
    """
    _setup()
    from src.stage1 import run
    from src.substrate import Substrate

    sub = Substrate(model_id, cache_dir="/root/activations")
    result = run(sub, max_statements_per_topic=max_per_topic or None)
    activations.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "sanity", model: str = "", max_per_topic: int = 0):
    """
    modal run mirage/modal_app.py --stage sanity                     # Stage 0, all models
    modal run mirage/modal_app.py --stage stage1 --model <id>        # Stage 1, one model
    modal run mirage/modal_app.py --stage stage1                     # Stage 1, all models
    modal run mirage/modal_app.py --stage stage1 --max-per-topic 50  # smoke run
    """
    from datetime import date

    import yaml

    cfg = yaml.safe_load((_HERE / "configs" / "models.yaml").read_text())
    ids = [model] if model else [m["id"] for m in cfg["substrates"]]

    if stage == "sanity":
        results = list(stage0_sanity.map(ids))  # parallel containers
        for rep in results:
            print(f"{'PASS' if rep.get('pass') else 'FAIL'}  {rep['model']}  "
                  f"vram={rep.get('vram_gb', '?')}GB")
        out = _HERE / "results" / "stage0_sanity.json"
        out.write_text(json.dumps(results, indent=2, default=_json_default))
        ok = all(r.get("pass") for r in results)
        print(f"stage 0 gate: {'PASS' if ok else 'FAIL'} -> {out}")

    elif stage == "stage1":
        for result in stage1_saplma.map(ids, kwargs={"max_per_topic": max_per_topic}):
            short = result["model"].split("/")[-1].lower()
            out = _HERE / "results" / f"stage1_saplma_{short}_{date.today():%Y%m%d}.json"
            out.write_text(json.dumps(result, indent=2, default=_json_default))
            print(f"{result['model']}: {result['gate_note']}\n  -> {out}")

    else:
        raise SystemExit(f"unknown stage: {stage}")

    print("reminder: commit the artifact + a the project notebook entry (the project's standing directive §3/§7)")
