# MIRAGE decisions log (append-only)

Every resolved §4 open decision gets an entry: date, decision, reasoning, who decided.

---

## D-001 — Repo location (provisional)

- **Date:** 2026-07-09
- **Decision:** Scaffold lives at `mirage/` inside the parent project repository, on the session's
  designated branch. Extraction to a standalone `mirage` repo is planned once one exists;
  the directory is self-contained to make that a pure copy.
- **Reasoning:** Current agent session is scoped to `the project repository` only; creating and
  pushing a new repository is outside its authorization. Self-containment preserved (own
  README, .gitignore, requirements, configs).
- **Decided by:** Owner approved agent recommendation in session (2026-07-09). **Provisional** —
  revisit when a standalone repo is created.

## D-002 — Typicality operationalization (§4.1)

- **Date:** 2026-07-09
- **Decision:** Triangulate three axes. **Primary** (used for cell assignment):
  perplexity under a *different-family* reference LM (llama items scored under qwen and
  vice versa; mapping in `configs/models.yaml`). **Secondary** (cross-check): entity corpus
  frequency via infini-gram counts. **Tertiary** (reported only, flagged circular, never
  used for assignment): the substrate model's own perplexity.
- **Reasoning:** If typicality is measured under the substrate being probed, probe score and
  typicality are two functions of the same forward pass and the Stage-3b mediation becomes
  circular. Cross-family perplexity breaks that loop; frequency grounds it in pretraining
  data reality. Disagreement among the three axes is itself reportable (§1.6).
- **Decided by:** Owner approved agent recommendation in session (2026-07-09).

## D-003 — False+typical generation strategy (§4.2, the recursive trap)

- **Date:** 2026-07-09
- **Decision:** **Symmetric-edit design plus a second natural source.**
  1. False+typical items are minimal entity swaps to *frequency-matched, same-type*
     entities (Paris→Madrid, not Paris→Ouagadougou), keeping the false cell typical.
  2. **Every cell — including true cells — passes a matched fraction of items through the
     same edit pipeline** (truth-preserving swaps/paraphrases), so edit provenance is
     orthogonal to truth *by construction*, not just by canary.
  3. Second, independent false+typical source: the substrate model's own confident natural
     errors, filtered to low reference-perplexity. Two generation processes agreeing is the
     answer to "your corpus is the artifact."
  4. The "was-this-edited" canary probe is built and run anyway (gate: AUROC ≤ 0.55,
     `configs/corpus.yaml`) before any Stage-3 number is trusted.
- **Reasoning:** Canary alone detects contamination; the symmetric design *prevents* it.
  Natural errors alone resist typicality control; synthetic edits alone risk the
  edit-signature shortcut. The combination covers both failure modes.
- **Decided by:** Owner approved agent recommendation in session (2026-07-09).
  **Note:** the concrete source datasets and the edit pipeline's implementation still require
  owner review at the Stage 2 gate, including the external-reviewer step (the project's standing directive §7).

## D-004 — Layer choice (§4.3)

- **Date:** 2026-07-09
- **Decision:** Sweep every layer; report the full curve. Additionally overlay the
  layer-wise AUROC of a *typicality* probe on the same axis — co-peaking layers are
  evidence for the confound and belong in Figure 1.
- **Reasoning:** Per the project's standing directive §4.3, cherry-picking the best layer is itself the confound
  being hunted. The overlay turns the sweep from a robustness check into evidence.
- **Decided by:** Owner approved agent recommendation in session (2026-07-09).
