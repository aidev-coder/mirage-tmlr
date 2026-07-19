# MIRAGE hardness probe

A standing check for **truth × typicality decorrelation corpora** — the released
constructive artifact of the MIRAGE paper (§5/§6).

Internal-state hallucination probes (SAPLMA, EigenScore, …) report high AUROC for
true-vs-false. MIRAGE argues that number is confounded with *typicality*, and that
the one cell that would expose it — **confident, fluent, in-distribution falsehood
(FT)** — is intrinsically hard to construct. This tool makes that testable: point
it at any candidate corpus and it reports whether the corpus can support an
un-confounded truth claim.

## Checks

| # | Check | Needs GPU? | Fails when |
|---|---|---|---|
| 1 | crossing | no | typical/atypical not separated on the assignment axis |
| 2 | FT yield | no | the false-typical cell is empty/near-empty (< 15%) |
| 3 | edit canary | yes (substrate) | a "was-edited" probe reads the generation signature |
| 4 | fragmentation canary | yes | truth is predictable from tokenization alone |

A corpus is **USABLE** only if all four pass. In this project's own attempts it
never did: FT was starved (checks 1–2 under a perplexity axis) or the canaries
fired (checks 3–4 under swaps). See `../results/stage2_construction_evidence.json`.

## Use

```bash
# data-only checks (crossing + FT yield):
python mirage_hardness/run_check.py --corpus your_corpus.jsonl

# full check incl. hidden-state canaries: run on a GPU via Modal
# (mirage_hardness.run_check.run(items, substrate=<Substrate>, layer=L))
```

Corpus format: one JSON object per line with at least
`{truth, cell, edited, entity, text, typicality:{entity_freq_log10, reference_ppl}}`.

## Verdict

`USABLE` / `USABLE (data-only checks)` / `CONFOUNDED / UN-BUILDABLE`, plus the list
of failed checks. Publish it next to any internal-state-probe AUROC.
