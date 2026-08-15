# MIRAGE

**A validity audit of white-box (internal-state) LLM hallucination detectors.**

When a probe on an LLM's hidden states reports high hallucination-detection AUROC, is it
reading **falsehood** — or **atypicality** (perplexity / distribution shift / low frequency)
that merely *correlates* with falsehood in the benchmarks everyone trains and tests on?

MIRAGE answers that with a decorrelation corpus (a crossed truth × typicality 2×2), three
orthogonal statistical tests, and a released validity suite any future probe must pass.
See `the project's standing directive` for the standing directive, guardrails, and stage gates.

## Layout

```
configs/        substrate models, probe hyperparams, corpus cell definitions
src/
  substrate.py            per-layer hidden-state extraction (uniform interface)
  typicality.py           the contested axis: cross-family ppl + frequency + self-ppl
  corpus_build.py         2x2 construction + crossing/canary gates (Stage 2)
  stats.py                bootstrap AUROC CIs (every AUROC ships with one)
  probes/                 saplma | eigenscore | semantic_entropy
  eval/                   stratified_auroc | mediation | adversarial_split (Stage 3)
  report/                 tables/figures regenerated from results/*.json only
data/           raw (gitignored, fetch script) | corpus (versioned) | activations (gitignored)
results/        canonical result artifacts — the only source of truth for the writeup
notes/          decisions.md (append-only) | the project notebook (append-only)
tests/          synthetic self-test of the eval stack (runs on CPU, no models)
```

## Status

| Stage | State |
|---|---|
| 0 Substrate | code ready; gate (extraction verified on GPU box) **pending** |
| 1 Probe bank | code ready; headline reproduction **pending** |
| 2 Decorrelation corpus | gates implemented; generation **blocked on owner-signed design** (§4.2) |
| 3 Three tests | implemented; validated on synthetic data (`tests/`) |
| 4 Diagnosis table | generator ready |
| 5 Validity suite | scaffold only |

## Quickstart

```bash
pip install -r env/requirements.txt        # torch/transformers only needed for Stages 0-2
python tests/test_eval_synthetic.py        # verify the Stage-3 eval stack end-to-end (CPU)
python -m src.substrate --sanity           # Stage 0 gate (GPU box)
```

## Provenance

MIRAGE is the successor to an earlier, unpublished project by the same authors. That project's
certification audit found that its own published 14.3% natural-violation rate was entirely a
measurement artifact of the extraction stage rather than a property of the data. MIRAGE relocates
the same instrument-validity question to internal-state hallucination detectors.

The standing directive, guardrails and stage gates are in `the project's standing directive`. `HASH_MAP.md` maps the
registration commits cited in the paper to their equivalents here, since this repository's history
was rewritten for anonymisation and every hash changed.
