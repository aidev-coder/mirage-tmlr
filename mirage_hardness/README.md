# MIRAGE hardness probe

The released constructive artifact of the MIRAGE paper (§5/§6). Two tools:

| Tool | Audits | Question it answers |
|---|---|---|
| `run_check.py` | a **corpus** | can this truth × typicality corpus support an un-confounded truth claim? |
| `audit_probe.py` | a **detector** | how much of this probe's reported AUROC is the typicality confound? |

Internal-state hallucination probes (SAPLMA, EigenScore, …) report high AUROC for
true-vs-false. MIRAGE finds that number is inflated by *typicality*: statements that
look ordinary are usually true in every benchmark these probes train on, so a probe
that only learned "ordinary means true" scores just as well. The cell that exposes
it is **confident, fluent, in-distribution falsehood (FT)** — a fluent lie.

## 1. Corpus check — `run_check.py`

| # | Check | Needs GPU? | Fails when |
|---|---|---|---|
| 1 | crossing | no | typical/atypical not separated on the assignment axis |
| 2 | FT yield | no | the false-typical cell is empty/near-empty (< 15%) |
| 3 | edit canary | yes (substrate) | a "was-edited" probe reads the generation signature |
| 4 | fragmentation canary | yes | truth is predictable from tokenization alone |

```bash
python mirage_hardness/run_check.py --corpus your_corpus.jsonl   # data-only checks
# full check incl. hidden-state canaries needs a GPU substrate:
# mirage_hardness.run_check.run(items, substrate=<Substrate>, layer=L)
```

Verdict: `USABLE` / `USABLE (data-only checks)` / `CONFOUNDED / UN-BUILDABLE`, plus
the failed checks.

**What it took us to pass.** Early attempts failed: assigning cells by *perplexity*
starved FT (perplexity encodes truth, so fluent-false barely exists under it), and
entity-swap generation tripped the edit canary at small n. Switching the assignment
axis to entity frequency (D-007) made FT constructable, and a larger-n canary showed
the swap signature was faint and seed-variable rather than disqualifying (D-010/D-012).
The released corpus passes crossing + the edit canary; fragmentation is handled as an
analysis covariate rather than by subsampling (D-011), because matching to clean it
introduced an artifact of its own. Run this before trusting your own corpus — the
failure modes above are easy to hit and invisible without the checks.

## 2. Probe audit — `audit_probe.py`

Returns the triple every internal-state probe paper should publish:

- **headline** — AUROC on a held-out **diagonal** split (truth and typicality aligned).
  The number the field reports.
- **controlled** — AUROC on the **off-diagonal** only (rare-true + fluent-false),
  where truth and typicality disagree. The honest number.
- **gap** — headline − controlled, with a bootstrap CI. The inflation attributable
  to the confound.

Plus per-cell mean P(true). A confounded probe rates the **FT** cell *true*; that
cell is the mechanism, not a rounding error.

Your probe supplies the standard two-method contract:

```python
class MyProbe:
    def fit(self, X, y): ...    # X: [n, d] activations, y: bool truth
    def score(self, X): ...     # -> [n] P(true)
```

```bash
# precomputed activations, no GPU:
python mirage_hardness/audit_probe.py \
    --corpus data/corpus/mirage_2x2_v<hash>.jsonl \
    --activations acts.npy \
    --probe mypkg.myprobe:MyProbe

# or extract here (GPU):
python mirage_hardness/audit_probe.py --corpus <corpus> \
    --model meta-llama/Llama-3.1-8B --probe mypkg.myprobe:MyProbe
```

`acts.npy` is `[n, d]` in corpus order, or `[n, n_layers, d]` with `--layer L`.
Omit `--probe` to audit the built-in SAPLMA reference.

Verdict: `CONFOUNDED` when the gap CI excludes zero, else `NOT DISTINGUISHABLE FROM
UNCONFOUNDED on this corpus`. The tool is symmetric on purpose — `tests/test_audit_probe.py`
checks it flags a by-construction confounded probe *and* clears an honest one. A
check that only ever says CONFOUNDED proves nothing.

## Corpus format

One JSON object per line, at least:
`{truth, cell, edited, entity, text, typicality:{entity_freq_log10, reference_ppl}}`
where `cell` ∈ {TT, TA, FT, FA} (True/False × Typical/Atypical).
