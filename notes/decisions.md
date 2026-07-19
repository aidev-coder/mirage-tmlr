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

## D-005 — Stage-2 infra defaults adopted; C3 confound added; gate held (§4/§7)

- **Date:** 2026-07-12
- **Decision (agent-adopted defaults, PROVISIONAL — owner may veto):**
  - **O-1 Wikidata:** frozen dated SPARQL snapshot committed by hash (the doc's
    recommendation; matches data/raw/MANIFEST pattern). ADOPTED.
  - **O-3 frequency:** infini-gram API primary with a local Dolma n-gram count
    fallback; a network check on the Modal box decides at build time. ADOPTED.
  - **O-2 natural-error source:** RECOMMEND harvesting from a DISJOINT model (not
    a probed substrate) and reporting swap-generated vs harvested FT separately so
    neither source is load-bearing alone. NOT adopted — touches the FT cell; owner
    call. (No GPT-4o; a Modal-hosted disjoint model or Groq is the practical source.)
  - **C3 (new, from stage2_self_review.md):** add `entity_subword_count` to the
    cell match keys and a second tokenization/fragmentation canary. Folds into
    the D-003 control set. Owner signoff item.
- **Held:** generation functions remain stubbed. Per §3/§7 the corpus is NOT built
  or probed until (i) owner signoff here, and (ii) a HUMAN interpretability reviewer
  sees the design + a 50-item sample (O-4). The self-review reduces but does not
  discharge (ii).
- **Escalation to owner (the one integrity decision, not execution):** proceed to
  BUILD on owner signoff with the automated crossing + edit-canary + fragmentation
  gates as backstops and an external review invited async-but-non-blocking; OR hold
  the build until a named human reviewer (O-4) responds. Agent recommendation:
  the former, because the design is strong, the automated gates are hard, and an
  indefinite hold on an unnamed reviewer stalls the project — but this is the
  owner's scientific-integrity call by §4/§7, so it is surfaced, not resolved.
- **Decided by:** agent recommendation; awaiting owner veto/confirm.

## D-006 — Owner build-signoff for Stage-2 corpus (§3 iii); §7 gap accepted

- **Date:** 2026-07-12
- **Decision:** Owner ("build" / "Continue" in session, after the conflict was named
  explicitly) signs off building the Stage-2 corpus on the approved design
  (D-002/D-003) + adopted defaults (D-005). Generation stubs may be implemented.
- **§7 external-reviewer gap — ACCEPTED AS RISK by owner:** no human interpretability
  reviewer is named (O-4). The agent's adversarial self-review (stage2_self_review.md,
  found confound C3) is an explicit SUBSTITUTE, not a discharge. Mitigation:
  (a) the corpus is built + gated but marked PROVISIONAL; (b) **Stage-3 probe runs
  are HELD** until either a human reviewer signs off or the owner explicitly waives
  it a second time — building is reversible, trusting Stage-3 numbers is not;
  (c) hard automated gates (crossing, edit-canary, fragmentation-canary C3) must
  pass or finalize() refuses to write the corpus.
- **Reasoning:** owner overrode §7 ordering with eyes open after loud escalation
  (§Role permits in-chat override once the conflict is named). Deepest intent of
  §7 — don't trust confounded numbers — preserved by holding Stage 3.
- **Decided by:** owner (session), agent recommended.

## D-007 — Typicality assignment axis: perplexity -> entity frequency (§4.1 revisit)

- **Date:** 2026-07-14
- **Status:** PROVISIONAL (agent recommendation; owner veto/confirm pending).
- **Trigger (measured):** Stage-2 v1+v2 builds show perplexity encodes truth (LMs
  score false statements high-ppl), so the FT (false+typical) cell is intrinsically
  starved when cells are assigned by ppl tercile (FT=72-78 vs FA~960).
- **Recommendation:** promote the D-002 SECONDARY axis (entity corpus frequency,
  infini-gram Dolma index) to PRIMARY for 2x2 cell assignment. Perplexity retained
  as the reported cross-check; the ppl-vs-frequency divergence becomes a finding.
  Rationale: entity frequency is separable from truth; ppl is not. This makes FT
  constructable (common-entity falsehoods) and makes the confound a genuine
  alternative explanation rather than a near-tautology (ppl≈truth).
- **Supersedes:** D-002's choice of ppl-primary FOR ASSIGNMENT only (triangulation
  + cross-family reasoning of D-002 otherwise stands).
- **Decided by:** agent recommendation; awaiting owner nod (surfaced per §4.1/§7).

## D-008 / D-009 — Harvest insufficient; REFRAME around construction-impossibility

- **Date:** 2026-07-19
- **D-008 (harvest) outcome:** natural-error harvest from disjoint Mistral yields
  2.5% (5/200 cities), obscure-skewed -> cannot populate FT. Combined with the
  swap edit-signature (v1-v3, ~0.67), both construction paths are exhausted.
- **D-009 DECISION (owner: "continue", 2026-07-19):** reframe the paper. The
  finding is that the FT cell (confident fluent in-distribution falsehood) is
  intrinsically hard to construct cleanly, so internal-state probes are validated
  only on the diagonal — blind to the confound cell. Retire the Stage-3 AUROC-gap
  plan (no clean corpus exists to run it on); the three builds + harvest yield ARE
  the evidence. Constructive half: release the build-attempt harness as a
  "hardness probe" any future corpus/probe must pass.
- **Reasoning:** the project's standing directive §0 pre-authorizes "corpus can't be built cleanly" as an
  equally-publishable outcome; §6 says lead with the audit. The obstacle is more
  novel than the original gap-measurement plan.
- **Decided by:** owner ("continue"); agent recommended after exhaustive attempts.
