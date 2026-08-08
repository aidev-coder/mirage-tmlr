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
# Owner authorized a larger GPU for speed (overrides the §5 A10/L4 default).
# A100-40GB: fits 9B bf16 + batched activations with headroom, cost-sane for a
# $30 budget. Bump to "H100" for ~2x faster extraction if the budget allows.
GPU = "H100"
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
def stage1_saplma(model_id: str, max_per_topic: int = 0,
                  batch_size: int = 32, fast: bool = True) -> dict:
    """Stage 1 SAPLMA headline reproduction for one model (the project's standing directive §3 Stage 1).

    Requires data/raw/ to be populated locally BEFORE `modal run` (the image
    ships the local mirage/ tree, so run `python mirage/data/raw/fetch.py` first).

    Resumable: batched extraction is sharded per topic and the probe sweep is
    checkpointed per layer, both committed to the activations volume as they go —
    a container cut-off resumes from the last shard/layer instead of restarting.
    """
    _setup()
    from src.stage1 import run
    from src.substrate import Substrate

    sub = Substrate(model_id, cache_dir="/root/activations")
    result = run(
        sub, max_statements_per_topic=max_per_topic or None,
        fast=fast, device="cuda", batch_size=batch_size,
        commit_fn=activations.commit,  # durable checkpoints (cut-off safety)
    )
    activations.commit()
    hf_cache.commit()
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=3 * 3600)
def stage2_build(reference_model: str, canary_model: str,
                 edit_rate: float = 0.5, seed: int = 20260712,
                 version: int = 2) -> dict:
    """Stage 2 GPU stage: score typicality + run the 3 gates; return scored items
    + reports. finalize() is run locally by the entrypoint so the corpus lands in
    the repo (the project's standing directive §3 Stage 2; probes stay HELD per D-006)."""
    _setup()
    from src.build_corpus import score_and_gate
    from src.substrate import Substrate

    ref = Substrate(reference_model, cache_dir="/root/activations")
    can = Substrate(canary_model, cache_dir="/root/activations")
    if version >= 2:
        from src.build_corpus import score_and_gate_v2
        out = score_and_gate_v2(ref, can, edit_rate=edit_rate, seed=seed or 20260806)
    else:
        out = score_and_gate(ref, can, edit_rate=edit_rate, seed=seed)
    activations.commit()
    hf_cache.commit()
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=4 * 3600)
def stage3_run(model_id: str, corpus_name: str, detector: str = "saplma",
               domain: str = "") -> dict:
    """Stage 3: the three decorrelation tests on the finalized corpus (§3 Stage 3).
    Hidden states cached to the activations volume; full per-layer curves."""
    _setup()
    from src.stage3 import run
    from src.substrate import Substrate

    sub = Substrate(model_id, cache_dir="/root/activations")
    corpus_path = f"/root/mirage/data/corpus/{corpus_name}"
    if detector == "eigenscore":
        from src.stage3_eigen import run as run_eigen
        out = run_eigen(sub, corpus_path, commit_fn=activations.commit)
    else:
        out = run(sub, corpus_path, detector=detector, device="cuda",
                  commit_fn=activations.commit, domain=domain or None)
    activations.commit()
    hf_cache.commit()
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=2 * 3600)
def dump_layer_activations(model_id: str, corpus_name: str, layer: int | None = None) -> dict:
    """Extract ONE layer's hidden states for the full corpus and return them as
    npy bytes, for auditing EXTERNAL probes locally with mirage_hardness/
    audit_probe.py (which needs a local .npy, not the Modal activations volume).
    Returns only the headline layer (default: mid-depth), a [n, d] float32 array —
    small enough to transfer directly, unlike the full [n, n_layers, d] stack."""
    _setup()
    import io

    import numpy as np

    from src.stage3 import load_corpus
    from src.substrate import Substrate

    sub = Substrate(model_id, cache_dir="/root/activations")
    items = load_corpus(f"/root/mirage/data/corpus/{corpus_name}")
    texts = [it["text"] for it in items]
    H = sub.hidden_states_matrix(texts, batch_size=32)  # [n, n_layers+1, d]
    n_layers = H.shape[1]
    L = layer if layer is not None else n_layers // 2
    Xl = H[:, L, :].astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, Xl)
    activations.commit()
    hf_cache.commit()
    return {"layer": L, "n_layers": n_layers, "shape": list(Xl.shape),
            "npy_bytes": buf.getvalue()}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=4 * 3600)
def causal_run(model_id: str, corpus_name: str, domain: str = "") -> dict:
    """Causal mediation: fraction of the truth-probe response routed through the
    typicality direction, on matched twin pairs. Reuses cached activations."""
    _setup()
    from src.causal import run as run_causal
    from src.substrate import Substrate
    sub = Substrate(model_id, cache_dir="/root/activations")
    out = run_causal(sub, f"/root/mirage/data/corpus/{corpus_name}",
                     commit_fn=activations.commit, domain=domain or None)
    activations.commit()
    hf_cache.commit()
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=4 * 3600)
def intervene_run(model_id: str, corpus_name: str, k: int = 8,
                  domain: str = "") -> dict:
    """Model-level: ablate the frequency manifold from the residual and re-read the
    model's own stated P(true) per cell. Manifold vs random-subspace null."""
    _setup()
    from src.intervene import run as run_intervene
    from src.substrate import Substrate
    sub = Substrate(model_id, cache_dir="/root/activations")
    out = run_intervene(sub, f"/root/mirage/data/corpus/{corpus_name}", k=k,
                        commit_fn=activations.commit, domain=domain or None)
    activations.commit()
    hf_cache.commit()
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=4 * 3600)
def gendis_run(model_id: str, corpus_name: str, max_subjects: int = 400,
               domain: str = "") -> dict:
    """Generation-time dissociation: the model writes the statements, then we read
    the fielded probe and the model's own judgment on its natural hallucinations."""
    _setup()
    from src.gen_dissociation import run as run_gd
    from src.substrate import Substrate
    sub = Substrate(model_id, cache_dir="/root/activations")
    out = run_gd(sub, f"/root/mirage/data/corpus/{corpus_name}",
                 max_subjects=max_subjects, commit_fn=activations.commit,
                 domain=domain or None)
    activations.commit()
    hf_cache.commit()
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS,
              timeout=2 * 3600)
def harvest_fn(model_id: str, topic: str, max_subjects: int = 400) -> dict:
    """O-2/D-008 natural-error harvest from a disjoint model (Mistral)."""
    _setup()
    from src.harvest import harvest_topic
    from src.substrate import Substrate
    sub = Substrate(model_id, cache_dir="/root/activations")
    out = harvest_topic(sub, topic, max_subjects=max_subjects)
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(stage: str = "sanity", model: str = "", max_per_topic: int = 0,
         batch_size: int = 32, fast: bool = True, corpus: str = "", seed: int = 0,
         detector: str = "saplma", domain: str = ""):
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
        kw = {"max_per_topic": max_per_topic, "batch_size": batch_size, "fast": fast}
        for result in stage1_saplma.map(ids, kwargs=kw):
            short = result["model"].split("/")[-1].lower()
            tag = f"_k{max_per_topic}" if max_per_topic else ""
            impl = "" if result.get("probe_impl") == "torch_gpu" else "_sklearn"
            out = _HERE / "results" / f"stage1_saplma_{short}{tag}{impl}_{date.today():%Y%m%d}.json"
            out.write_text(json.dumps(result, indent=2, default=_json_default))
            print(f"{result['model']}: {result['gate_note']}\n  -> {out}")

    elif stage == "stage2":
        import sys as _sys
        _sys.path.insert(0, str(_HERE))
        from src import corpus_build

        ref_model = model or "Qwen/Qwen2.5-7B-Instruct"   # cross-family reference (D-002)
        canary_model = "meta-llama/Llama-3.1-8B"           # representative probed substrate
        kw2 = {"seed": seed} if seed else {}
        res = stage2_build.remote(reference_model=ref_model, canary_model=canary_model,
                                  version=2, **kw2)
        m = res["meta"]
        print(f"stage2: released n={m['n_released']} (full {m['n_full']}) | "
              f"ref={m['reference_model']} canary={m['canary_model']} @L{m['canary_layer']} | "
              f"cells raw={m['raw_counts']} released={m['released_counts']}")
        for g in ("crossing", "edit_canary", "fragmentation_canary", "composition_canary"):
            if g not in res:
                continue
            r = res[g]
            ci = f" ci={r.get('ci')}" if r.get("ci") else ""
            print(f"  {g}: pass={r.get('pass')} "
                  f"{('auroc=' + str(r.get('auroc'))) if 'auroc' in r else ''}{ci}")
        em = res.get("evidence_matched", {})
        print(f"  [xcheck, truth-matched subset n={em.get('n')}] "
              f"edit={em.get('edit_canary', {}).get('auroc')} "
              f"frag={em.get('fragmentation_canary', {}).get('auroc')} "
              f"{em.get('fragmentation_canary', {}).get('ci')}")
        try:
            path = corpus_build.finalize(
                res["items"], res["crossing"], res["edit_canary"],
                res["fragmentation_canary"], owner_signoff_decision_id="D-015",
                composition_report=res.get("composition_canary"))
            (_HERE / "results" / "stage2_corpus_report.json").write_text(
                json.dumps({"meta": m, "gates": {k: res[k] for k in
                            ("crossing", "edit_canary", "fragmentation_canary")},
                            "evidence_matched": em}, indent=2, default=_json_default))
            print(f"  CORPUS FINALIZED -> {path}  (D-011: full edit-clean corpus; Stage 3 next)")
        except RuntimeError as e:
            print(f"  NOT finalized: {e}")

    elif stage == "stage3":
        from datetime import date

        canary_model = model or "meta-llama/Llama-3.1-8B"
        corpus_name = corpus
        if not corpus_name:
            cdir = _HERE / "data" / "corpus"
            cands = sorted(cdir.glob("mirage_2x2_v*.jsonl"),
                           key=lambda p: p.stat().st_mtime)
            if not cands:
                raise SystemExit("no finalized corpus in data/corpus/ — run --stage stage2 first")
            corpus_name = cands[-1].name
        res = stage3_run.remote(model_id=canary_model, corpus_name=corpus_name,
                                detector=detector, domain=domain)
        hl = res["headline_layer"]
        e = res["per_layer"][hl]
        print(f"stage3[{res['detector']}] {res['model']} on {corpus_name} n={res['n']} "
              f"{res['cell_counts']}")
        print(f"  headline layer L{hl}:")
        print(f"    [fielded instrument = the field's recipe, audited by all 3 tests]")
        print(f"    3a stratified gap = {e['stratified_fielded'].get('gap')}")
        print(f"    3b truth_beta {e['mediation_fielded']['truth_beta_marginal']} -> "
              f"{e['mediation_fielded']['truth_beta_partialled']} "
              f"(shrink {e['mediation_fielded']['truth_beta_shrinkage']}); "
              f"frag_beta {e['mediation_fielded'].get('fragmentation_beta')}")
        print(f"    3c off-diagonal AUROC = {e['adversarial']['off_diagonal'].get('auroc')} "
              f"{e['adversarial']['off_diagonal'].get('ci')}; gap {e['adversarial']['gap']}")
        print(f"    per-cell fielded score: {e['fielded_cell_scores']}")
        rec = e.get("mediation_allcell", {}).get("truth_beta_partialled")
        if rec is not None:
            print(f"    [recoverability: all-cell probe] truth_beta_partialled = {rec}, "
                  f"strat_gap = {e.get('stratified_allcell', {}).get('gap', {}).get('point')}")
        else:
            print(f"    [unsupervised detector: no all-cell recoverability]")
        short = res["model"].split("/")[-1].lower()
        dtag = f"_{domain}" if domain else ""
        out = _HERE / "results" / f"stage3_{res['detector']}_{short}{dtag}_{date.today():%Y%m%d}.json"
        out.write_text(json.dumps(res, indent=2, default=_json_default))
        print(f"  -> {out}")

    elif stage == "dumpacts":
        model_id = model or "meta-llama/Llama-3.1-8B"
        corpus_name = corpus
        if not corpus_name:
            cdir = _HERE / "data" / "corpus"
            corpus_name = sorted(cdir.glob("mirage_2x2_v*.jsonl"),
                                 key=lambda p: p.stat().st_mtime)[-1].name
        res = dump_layer_activations.remote(model_id=model_id, corpus_name=corpus_name,
                                            layer=seed or None)
        out_dir = _HERE / "data" / "activations" / "local"
        out_dir.mkdir(parents=True, exist_ok=True)
        short = model_id.split("/")[-1].lower()
        chash = Path(corpus_name).stem.split("_v")[-1]
        out = out_dir / f"{short}_{chash}_L{res['layer']}.npy"
        out.write_bytes(res["npy_bytes"])
        print(f"dumpacts {model_id} corpus={corpus_name} layer={res['layer']}/{res['n_layers']} "
              f"shape={res['shape']} -> {out}")

    elif stage == "causal":
        from datetime import date
        model_id = model or "meta-llama/Llama-3.1-8B"
        corpus_name = corpus
        if not corpus_name:
            cdir = _HERE / "data" / "corpus"
            cands = sorted(cdir.glob("mirage_2x2_v*.jsonl"), key=lambda p: p.stat().st_mtime)
            if not cands:
                raise SystemExit("no finalized corpus — run --stage stage2 first")
            corpus_name = cands[-1].name
        res = causal_run.remote(model_id=model_id, corpus_name=corpus_name, domain=domain)
        hl = res["headline_layer"]; e = res["per_layer"][hl]
        print(f"causal {res['model']} L{hl} :: {res['contrast']}")
        print(f"  FT-error frac_mediated by k: {e['ft_error_frac_mediated_by_k']}")
        print(f"  random-subspace null   by k: {e['ft_error_random_subspace_by_k']}")
        print(f"  TT-ctrl  frac_mediated by k: {e['tt_control_frac_mediated_by_k']}")
        short = res["model"].split("/")[-1].lower()
        dtag = f"_{domain}" if domain else ""
        out = _HERE / "results" / f"causal_{short}{dtag}_{date.today():%Y%m%d}.json"
        out.write_text(json.dumps(res, indent=2, default=_json_default))
        print(f"  -> {out}")

    elif stage == "intervene":
        from datetime import date
        model_id = model or "meta-llama/Llama-3.1-8B"
        corpus_name = corpus
        if not corpus_name:
            cdir = _HERE / "data" / "corpus"
            cands = sorted(cdir.glob("mirage_2x2_v*.jsonl"), key=lambda p: p.stat().st_mtime)
            corpus_name = cands[-1].name
        res = intervene_run.remote(model_id=model_id, corpus_name=corpus_name,
                                   k=seed or 8, domain=domain)
        print(f"intervene {res['model']} L{res['layer']} k{res['k']}")
        for c, v in res["p_true"].items():
            print(f"  {c}: probe {v['probe_readout']} vs behavior {v['behavioral_baseline']} "
                  f"(manifold-ablated {v['manifold_ablated']})")
        short = res["model"].split("/")[-1].lower()
        dtag = f"_{domain}" if domain else ""
        out = _HERE / "results" / f"intervene_{short}{dtag}_{date.today():%Y%m%d}.json"
        out.write_text(json.dumps(res, indent=2, default=_json_default))
        print(f"  -> {out}")

    elif stage == "gendis":
        from datetime import date
        model_id = model or "Qwen/Qwen2.5-7B-Instruct"
        corpus_name = corpus
        if not corpus_name:
            cdir = _HERE / "data" / "corpus"
            corpus_name = sorted(cdir.glob("mirage_2x2_v*.jsonl"),
                                 key=lambda p: p.stat().st_mtime)[-1].name
        res = gendis_run.remote(model_id=model_id, corpus_name=corpus_name,
                                max_subjects=max_per_topic or 400, domain=domain)
        print(f"gendis {res['model']} L{res['layer']} n={res['n_generated']} "
              f"natural error rate {res['natural_error_rate']}")
        h, c = res["hallucinations"], res["correct_generations"]
        print(f"  hallucinated (n={h['n']}): probe {h['probe_mean_p_true']} | "
              f"model {h['behavior_mean_p_true']}")
        print(f"    high-freq subset (n={h['n_high_freq']}): probe "
              f"{h['probe_mean_p_true_high_freq']} | model {h['behavior_mean_p_true_high_freq']}")
        print(f"  correct      (n={c['n']}): probe {c['probe_mean_p_true']} | "
              f"model {c['behavior_mean_p_true']}")
        g = res.get("gold_free_dissociation", {})
        if g:
            print(f"  [gold-free] model disowns {g['n_model_disowns_own_output']} own outputs "
                  f"(model {g['behavior_mean_p_true_on_disowned']}) -> probe reads them "
                  f"{g['probe_mean_p_true_on_disowned']}, calls "
                  f"{g['probe_calls_disowned_true_frac']} true")
        if "probe_auroc_on_own_generations" in res:
            print(f"  AUROC on own generations: probe "
                  f"{res['probe_auroc_on_own_generations']['auroc']} "
                  f"{res['probe_auroc_on_own_generations']['ci']} | model "
                  f"{res['behavior_auroc_on_own_generations']['auroc']} "
                  f"{res['behavior_auroc_on_own_generations']['ci']}")
        short = res["model"].split("/")[-1].lower()
        dtag = f"_{domain}" if domain else ""
        out = _HERE / "results" / f"gendis_{short}{dtag}_{date.today():%Y%m%d}.json"
        out.write_text(json.dumps(res, indent=2, default=_json_default))
        print(f"  -> {out}")

    elif stage == "harvest":
        import json as _j
        hm = "mistralai/Mistral-7B-Instruct-v0.2"   # disjoint from all probed substrates
        topic = model or "cities"
        res = harvest_fn.remote(model_id=hm, topic=topic,
                                max_subjects=max_per_topic or 400)
        print(f"harvest[{topic}] stats: {res['stats']}")
        for it in res["items"][:8]:
            print("  FALSE:", it["text"])
        out = _HERE / "data" / "corpus" / f"harvest_{topic}.json"
        out.write_text(_j.dumps(res["items"], indent=1))
        print(f"saved {len(res['items'])} natural-error items -> {out}")

    else:
        raise SystemExit(f"unknown stage: {stage}")

    print("reminder: commit the artifact + a the project notebook entry (the project's standing directive §3/§7)")
