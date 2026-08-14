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

## D-010 — CI-based canary gate (owner ratified)
- Date: 2026-07-19. Owner: "yes, use the CI-gate".
- Decision: canary gates (edit, fragmentation) pass iff the AUROC bootstrap CI includes 0.50 (not distinguishable from chance), replacing the arbitrary point<=threshold rule. Aligns with 1.5. Symmetric: still fails the uncontrolled fragmentation (CI [0.567,0.681] excludes 0.50).
- Released corpus = truth-subword-matched (clean on both canaries by CI); the full corpus's uncontrolled fragmentation kept as evidence the control is necessary.
- Reasoning: point-0.55 tripped on 0.5501 seed-noise though the CI spanned chance; CI-inclusion-of-0.5 is the correct "at chance" criterion. Not result-tuning (1.1) — surfaced and ratified, applied symmetrically.

## D-011 — fragmentation is a Stage-3 covariate, not a corpus gate (owner ratified)
- Date: 2026-07-19. Status: RATIFIED, owner: "yes to D-011".
- Trigger: truth-subword matching cleans fragmentation (0.55) but induces a spurious edit-canary signal (0.646); the full corpus is edit-clean (0.55) with a real fragmentation confound (0.625).
- Proposal: finalize the FULL corpus using the edit-canary (generation artifact, 4.2) + crossing as the hard gates; treat fragmentation (a tokenization confound) as a covariate partialled out in the Stage-3b mediation, exactly as typicality is. Rationale: confounds are controlled analytically, not by distorting the corpus; matching introduced its own artifact.
- Decided by: owner ("yes to D-011", 2026-07-19); agent recommended.
- Consequence: score_and_gate returns the full corpus as items (n=2164); finalize gates on edit-canary (CI includes 0.5) + crossing; fragmentation_report kept in the corpus report and partialled out in Stage-3b mediation. Truth-matched subset retained as evidence_matched cross-check.

## D-012 — edit signature as a Stage-3 covariate (owner deferred to agent)
- Date: 2026-07-20. Owner: "as you think appropriate". Agent chose option A.
- Trigger: second corpus seed (20260801) fails the edit-canary (0.593 CI excludes 0.5) where seed 20260712 passes (0.550). The symmetric-swap edit signature is faint (~0.55-0.59 at n=2164) and seed-variable around the CI-includes-0.5 gate (§4.2).
- Decision: extend D-011 to the edit axis. Partial out an out-of-fold edit-probe score (P(edited) from hidden states at the mid layer) as a Stage-3 mediation covariate alongside typicality + fragmentation. The off-diagonal collapse is trustworthy only if it survives removing the edit-predictable component. Report the multi-seed canary spread honestly (§1.6).
- Not chosen (kept open): regenerating a signature-free corpus (B) or larger-n resolution (C) — A subsumes the immediate need without regeneration and directly tests the confound; revisit B if the covariate materially moves truth_beta.
- Applies to: SAPLMA Stage-3 (hidden-state probe). EigenScore keeps typicality+fragmentation (its own signal is hidden-state-derived; edit-covariate extension deferred unless needed).

## D-013 — scope the strong claim to trained probes (owner ratified)
- Date: 2026-07-21. Owner: "Scope to trained probes".
- Trigger: EigenScore self-variant is a weak truth detector on this corpus (diagonal AUROC 0.54-0.62, flat per-cell), so it cannot test the confound and the SAPLMA collapse does not replicate on it (measured, 2026-07-21).
- Decision: the paper's strong claim is scoped to SAPLMA-style trained linear/MLP internal-state probes (the dominant white-box family). The EigenScore self-variant is reported honestly as an inconclusive secondary (too weak to adjudicate), with the §6 caveat that it is not the strongest INSIDE operationalization. Faithful sampling-based EigenScore and semantic-entropy are left as future work, not blockers.
- Reasoning: §6 (audit the strongest real version) is satisfied for trained probes and honestly flagged as unmet for eigenscore; overclaiming generalization from a weak variant would repeat the the prior project overreach. The trained-probe result is clean, cross-model consistent, and robust to typicality+fragmentation+edit controls.
- Decided by: owner ("Scope to trained probes"); agent recommended A.

## D-014 — surface the typicality triangulation to answer the perplexity-confound objection (owner ratified)
- Date: 2026-07-23. Owner: "Surface triangulation".
- Trigger: rebuttal audit R2 — a reviewer will say entity frequency is not typicality and the probe really reads perplexity, which the mediation never partials out. The mechanism claim ("reads typicality") rested on one axis in the draft.
- Decision: surface the D-002 triangulation into the draft's corpus section, framed honestly as a divergence not an agreement. Frequency separates the cells (Cliff's 0.77/0.57, p 3.6e-217/2.9e-29); reference-LM perplexity is flat across the frequency axis within a truth class (TT/TA 14.9/14.6, FT/FA 57.4/58.0) because it tracks truth, not frequency (the D-007 basis). The mechanistic teeth: the two false cells are perplexity-matched (57 vs 58) yet the probe splits them (FT P(true) 0.56, FA 0.05), so perplexity is held equal across the probe's error and cannot be the read. Substrate claim-probability stays excluded as circular (D-002 tertiary).
- Reasoning: this closes R2 with measured data already in results/stage2_corpus_report.json rather than softening the claim; it is a mechanistic rebuttal (perplexity equalized across the error) not just a construction defense. No new run.
- Decided by: owner ("Surface triangulation"); agent recommended route (b) over softening.

## D-015 — corpus v2 rebuilt on verified frequencies with composition balanced by construction
- Date: 2026-08-06. Owner: "get the best of the best data we can have"; agent executed.
- Trigger: two fatal v1 defects. (a) 2627/2884 entity frequencies (91%) were rate-limit failures written as count=0, so the typicality axis was "did the API call succeed" — Amman cached 0, true count 1,183,125. (b) cell assignment ranked frequency globally across pooled domains, so `domain` predicted truth off-diagonal at AUROC 0.24 and manufactured the reported "collapse".
- Decision: rebuild. Frequency cache refetched with retries, failures recorded as None and never zero, verified by live re-query (25 sampled, 0 mismatches, 2884/2884 resolved). Cell assignment now ranks terciles WITHIN each domain; unresolved frequencies drop the item rather than zeroing it; a domain must fill all four cells with >= 20 items or it is excluded whole; items balanced per (domain, cell). Composition canary promoted to a hard finalize() gate.
- Result: `data/corpus/mirage_2x2_vdc291b459fc1.jsonl`, n=568, TT/TA/FT/FA = 142 each, two domains (cities, inventions) at 71 per (domain,cell). Excluded honestly: `elements` (13-17 per cell, under the floor), `generated` (too few measured frequencies). Gates: crossing PASS, edit-canary 0.4645 CI[0.361,0.577] PASS, fragmentation 0.4915 PASS, composition PASS (no text-visible field predicts truth off-diagonal).
- Axes verified crossed within domain: cities TT/FT median log10 freq 6.73/6.74 (typical) vs TA/FA 5.28/5.30 (atypical); inventions 4.40/4.38 vs 2.32/2.15. Perplexity tracks truth (TT 13.7, TA 21.0, FT 53.0, FA 82.4), which is why it is the cross-check and not the axis (D-007).
- Cost accepted: n falls 2164 -> 568. Balanced composition on measured frequencies is worth the sample-size loss; the v1 n was inflated by items placed on a fabricated axis.
- Supersedes: the v1 corpus (mirage_2x2_v165941295e9a) and every Stage 3+ result computed on it.

## D-016 — corpus v3 textual duplicates: disclose or rebuild (OWNER DECISION, OPEN)
- Date: 2026-08-08. Status: OPEN, surfaced to owner, agent recommends (a).
- Trigger: 70 of 696 rows in `mirage_2x2_v6206fe484650.jsonl` are exact textual duplicates, produced by `truth_preserving_swap` not deduplicating against the existing item pool. They are asymmetric by truth: TT +33, TA +35, FA +2, FT 0. Unique statements per cell are 141/139/174/172, so the finalize() balance gate passed on ROW counts while the true cells are ~20% repeats.
- Measured impact: no duplicate spans two cells, carries conflicting truth labels, or crosses the diagonal/off-diagonal boundary, so there is no train-on-test leakage. Edit canary recomputed on unique texts only: 0.5312 [0.432, 0.622] as shipped -> 0.5165 [0.421, 0.621] deduplicated, PASS either way, so §4.2 holds despite 14.5% of canary rows being label-ambiguous (identical string appearing once edited and once not). SAPLMA and LRProbe gaps move <= 0.07 under deduplication.
- Option (a) DISCLOSE: report n=696 rows / 626 unique statements, add a duplicate check to the finalize() gates so it cannot recur, keep every existing number. Cheapest, no number changes, and the defect is real but demonstrably not load-bearing.
- Option (b) REBUILD as v4 with deduplication in the generator: cleaner corpus, but a fourth rebuild, every Stage 3+ number recomputed, and n falls below 626 after rebalancing per (domain, cell).
- Agent recommendation: (a). The defect does not touch any headline claim on measurement, and §1.1 counsels against a rebuild whose main effect would be cosmetic. Escalated rather than resolved because corpus composition is a §4-class decision.

## D-016 — RESOLVED: rebuild as v4, deduplicated, with within-domain balancing
- Date: 2026-08-09. Owner: "rebuild and dups as well", scope "dedupe only, minimal v4"; balancing rule left to agent ("your call, go for the best").
- Decision: rebuild rather than disclose. v4 = the same 2x2, deduplicated, no new domains and no multi-clause items (those become auxiliary sets outside the 2x2 so the headline stays comparable).
- Fix at the source: `corpus_gen.generate_topic` now refuses a statement already emitted for that topic. First occurrence wins, deterministic by seed. Candidate pool 3814 -> 3200 rows, rows == uniques in every domain.
- Two new hard `finalize()` gates: (i) no duplicate statements; (ii) no domain may exceed 50% of items. Both are the class of check whose absence produced v1 (domain composition) and D-016 (duplicates).
- Balancing changed from GLOBAL equality per (domain, cell) to WITHIN-domain equality. v3's rule let the smallest domain cap every other one (companies has 133 unique true statements, so every domain got 58). Orthogonality only requires P(true|domain) = P(typical|domain) = 0.5, which within-domain balance already gives; domains need not match each other.
- Why not uncapped: uncapped puts cities at ~76% of items and adds NOTHING to companies/inventions, which are what actually bound the generality claim. A project whose headline was destroyed once by domain composition should not ship a pooled number that is mostly one topic. Hence the majority rule.
- Stated plainly and not spun: this does NOT close the small-n limitation. n is bounded by companies and inventions capacity; only new source data moves it.
- Consequence: every artifact keyed to the v3 hash (18 stage3 runs, the 7-model layer sweep, causal, gendis, intervene, tables, figures) must be regenerated on v4 before any of it is quoted again.

## D-017 — v5: `edited` balanced 50/50 inside every cell (owner ratified)
- Date: 2026-08-09. Owner chose rebuild ("a") over shipping v4 with a disclosed bound.
- Trigger: the §4.2 edited-truth independence invariant FAILS on both released corpora. Per-cell P(edited): v3 TT 0.500 / TA 0.489 / FT 0.471 / **FA 0.661** (spread 0.190); v4 TT 0.422 / TA 0.417 / FT 0.422 / **FA 0.578** (spread 0.161). Pre-existing, not introduced by the D-016 deduplication, though dedup worsened the candidate-pool gap 0.0718 -> 0.0806 by dropping items that were all edited+true.
- Why it matters: the diagonal is TT+FA, so "edited" predicts FALSE in training and predicts nothing off-diagonal (TA 0.417 vs FT 0.422). That is the same signature as the confound under study — a shortcut that works in-distribution and vanishes off it. Measured upper bound, using the `edited` flag ITSELF as the predictor: diagonal AUROC 0.578, off-diagonal 0.503, **gap +0.075 on v4** (+0.089 on v3) before any probe reads a hidden state.
- Scope of the damage: negligible for the large gaps (0.80-0.92 on Pythia, so at most ~8% attributable), but it EXCEEDS the small gaps at the strong-model end. gemma-2-9b-it's +0.032 [0.012, 0.052] sits inside the bound, so the 2026-08-09 claim that "the clean negative is retired, every model shows a significant positive gap" was NOT supported and is withdrawn pending v5.
- Decision: fix by construction, not by regression. `_balance_by_domain` now draws `edited` 50/50 within EVERY cell, so edit provenance is orthogonal to truth and to typicality by design. New hard `finalize()` gate `edit_balance_check` (max per-cell spread 0.02) plus the same check in the released `run_check.py`. Both retracted corpora fail it.
- Note D-012 already partialled an edit covariate out of the MEDIATION arm; the ADVERSARIAL arm that produces the headline gap never had that control. This closes it at the source instead.
- Cost: n falls again (balancing on the smaller of edited/unedited within each cell). Accepted.

## D-023 - Pre-registered criterion for the recoverability-vs-n curve (2026-08-14)

Registered BEFORE the curve is computed. Neither the owner-side reviewer nor I knows the
answer at the time of this commit. Recorded so the outcome meets a decision rule rather
than a story about one.

Context. B6 showed pythia-6.9b, a below-knee model, reaching 0.882 off-diagonal when its
probe is trained on 9,270 derivative rows and 0.697 on 4,991 official rows, against 0.122
from our 304-item diagonal. Section 3.2 currently calls the hinge "a property of the
representation, not of the probe". If the knee moves with training size, that sentence is
false and the section describes a data-budget regime instead.

Deviation from the proposed wording, stated. The proposal conditions on "two or more
below-knee models", but the hinge is fitted ACROSS models, not per model - there is one
knee per fit. So migration is defined on the fitted knee itself.

PROCEDURE. Refit recoverability (fair all-cell training) and the hinge at training sizes
n in {100, 200, 300, 450, 608} drawn from our crossed corpus, 5 subsamples per rung, all
18 models at each rung. The knee k(n) is refitted at each rung with its own bootstrap
interval.

CRITERION. Let W be the width of the bootstrap interval on the knee at the reference fit
(currently k = 0.765, interval [0.57, 0.80], so W = 0.23). The knee has MIGRATED if
|k(608) - k(100)| > W, AND the shift is monotone in n across at least three of the five
rungs, so a single noisy rung cannot trigger it.

CONSEQUENCE, fixed in advance.
  - Migration -> section 3.2 is REWRITTEN, not rescoped. The hinge becomes a claim about
    the standard protocol's probe regime at realistic training sizes, and the phrase
    "property of the representation" is withdrawn.
  - No migration -> the representational reading survives and the section states the
    scope explicitly: the knee is stable across a 6x range of training size.
  - Shift between 0.5W and W, or non-monotone -> reported as indeterminate at this corpus
    size, with no claim either way, and the curve is published as measured.

Known limitation of this design, stated now. Our corpus caps at 608, so the ladder cannot
reach the 4,991 and 9,270 sizes that produced the B6 anomaly. A null result therefore
bounds migration only within 100-608 and does not rule out a knee that moves at the sizes
the field actually trains on. That limitation goes in the section whichever way it falls.
