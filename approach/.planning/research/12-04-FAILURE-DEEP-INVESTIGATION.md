# Plan 12-04 Deep Failure Investigation

**Produced:** 2026-05-31
**Status:** Diagnostic + remediation playbook
**Parent verdict:** REJECT (Claude macro 0.9235 < 0.93 old floor; BBB 0.7377; BBB delta −0.0659 < −0.06 tolerance)
**Word count:** ~2950

---

## 1. Mechanistic Root Cause

### 1.1 The regression is at EXTRACTION, not validation

The 12-04 SUMMARY attributes the BBB regression to "boundary erasure between extraction and validation prompts" — implying the validation judge became confused. Per-link forensics on the trim's tmp phase-cache contradict this. The decomposition is:

| Stage | Effect on BBB |
|---|---|
| Trim adds NEW candidates at extraction (TRIM_NEW) | **+10 candidates, all FP, all auto-approved by validation** |
| Trim removes candidates at extraction | −4 candidates, all already-rejected by base validation (no F1 effect) |
| Decision flips on common candidates (approved → rejected) | 0 |
| Decision flips on common candidates (rejected → approved) | 0 |

**Every one of the +10 BBB FPs originates at extraction.** The validation prompt sees them downstream and unconditionally approves them — but it would have approved them with the original VALIDATION_RULES too, because the candidates' surface form ("Frontends receive the X event") fully satisfies the original APPROVE-1 rule ("performs an operation"). The validation prompt was never the bottleneck here.

This rules out the "boundary erasure" hypothesis as written. The mechanism is narrower: **the merged rubric, when wrapped by `_EXTRACTION_HEADER` and shown to the extraction LLM, widens what the proposer considers an "interaction participant".**

### 1.2 The exact pattern: subprocess role-labels treated as component aliases

The 10 BBB FPs all share one structural pattern. The BBB SAD has a contiguous block (s24–s32) describing the internal sub-process layout of the HTML5 Server, with sentences of the form:

```
s24: Frontends receive the ValidateAuthTokenResp event to complete authentication.
s27: Frontends handle completely the Streamer redis events ...
s29: Backends handle all the non-streamer events.
s30: If more than one backend is running, bbb-web splits the load ...
s31: So individual backends only process redis events for ...
```

"Frontends" / "Backends" are **role-labels for processes inside HTML5 Server**, not aliases of HTML5 Server. The architecture model component is `HTML5 Server`. The merged-rubric extraction prompt re-labels every one of these as a `HTML5 Server` reference (`matched_text="Frontends"`, etc.) — bypassing the existing alias mapping (which does not contain "frontends"/"backends" as global-scope aliases).

The 11th BBB FP follows the identical pattern: `matched_text="BigBlueButton API"` extracted as a reference to component `BBB web`.

These extractions did NOT happen in the baseline. The baseline extraction prompt, on the same sentences, did not propose these candidates at all.

### 1.3 Side-by-side rule audit: which rule actually changed semantics?

Comparing `ENTITY_EXTRACTION_RULES` (original) vs `ENTITY_EXTRACTION_RULES_V3` (merged):

| Original include rule | Merged rule | Semantic drift |
|---|---|---|
| EXT_inc_1 "name (or known alias) appears directly" | M1 verbatim | None |
| EXT_inc_2 "space-separated form matches compound" | M2 verbatim | None |
| EXT_inc_3 "sentence describes what specific component does by name or role" | **M3** "sentence — or its section heading — describes what a specific component does by name or role, naming the component **as its subject or as an architectural participant (performs an operation, provides or receives a service, is being configured, is explicitly introduced)**" | **SEMANTIC EXPANSION.** Original M3 required the **component to be named**. Merged M3 added the VAL_APP_1 clause "as an architectural participant (performs an operation...)". This makes the rule independently satisfiable by ANY sentence describing an architectural-participation pattern, even when the component name itself is not present. |
| EXT_inc_5 "component participates in interaction (sender/receiver/target)" | M5 verbatim | None |

The drift is in M3. In the original ENTITY_EXTRACTION_RULES, "describes what a specific component does by name or role" was scoped by the proposer-side framing — the model had to identify the *component* first, then check the description. In the merged rubric, M3 reads as a disjunction: "naming the component as its subject **OR** as an architectural participant (performs an operation / provides or receives a service / is being configured / is explicitly introduced as part of the system)". A subprocess-role-label sentence ("Frontends receive the X event") satisfies the second disjunct trivially: there is *some* component (HTML5 Server) whose sub-processes are being described, an operation is performed, a service is being received. The proposer attaches the operation back to the parent component.

In the ORIGINAL design, this semantic was confined to the JUDGE prompt (VAL_APP_1, downstream of extraction). The judge can apply it conservatively because by the time it fires the extraction proposer has already filtered to candidates whose name actually appears. **Lifting VAL_APP_1's text into the proposer prompt's rule body broke this invariant.**

This is exactly the failure mode the V35a lesson predicts ("merging boundary signals the LLM uses to disambiguate roles") — but the *specific* boundary that mattered here was extraction-vs-validation, and the specific rule that absorbed the drift was M3.

### 1.4 What about the 6 added FNs vs baseline?

Verdict reports BBB FN went from 12 (cached baseline) to 17 — six new misses. The candidate-level diff shows the trim KEPT every TP candidate the baseline had (no common-candidate `approved → rejected` flips, no removed-candidates-that-were-TPs). The +6 FNs are therefore not introduced by the merge — they are gold links the baseline pipeline never recovered in either run (the seed_val and coref stages are blocked in the harness, so gold links that require seed_val to flip from "raw_seed" to "validated" remain misses identically in both arms). The verdict's headline FN delta is misleading: it reflects baseline-fixture vs cached-final scoring mismatch, not a merge-induced recall loss.

**The mechanistic regression on BBB is precision-only, +10 extraction FPs.** The verdict's "−6 recall" framing is partially a baseline-anchor artifact.

---

## 2. Was The Merge Actually Lossless?

By the survey's own definition of Technique 3 ("lossless rewriting: same rules, denser surface form, identical decision boundary"), **the merge is NOT lossless**. The audit in §1.3 found one rule whose decision boundary moved (M3 absorbed VAL_APP_1). Three minor faithfulness checks:

| Original rule | Status in merged rubric | Faithfulness |
|---|---|---|
| EXT_inc_3 | Folded into M3 with VAL_APP_1 clause appended | **DRIFT** — disjunction widens proposer scope |
| VAL_APP_2 ("section heading names component") | Folded into M3 ("sentence — or its section heading — describes...") | OK on validation side; on extraction side this is new information that the original ENT prompt never delivered (minor expansion, but section heading is a strong signal so likely benign) |
| EXT_tiebreak "Favor inclusion over exclusion" | Preserved in `_EXTRACTION_HEADER` only | OK on extraction; **MISSING on validation header** which now reads as approve-OR-reject neutral. In the original, validation was the precision filter explicitly — losing that asymmetric framing on the judge side is a separate concern (but didn't manifest on BBB since validation never had to reject the new candidates) |
| EXT_inc_3 "by name or role" wording | Preserved verbatim in M3 head | Word-level intact; but its meaning shifted because it now reads as one of three disjuncts in a longer clause |

**The merge is a faithful textual collapse but an unfaithful semantic collapse.** The executor did not introduce arbitrary information; every rule fragment is present somewhere. But the *structural separation* between PROPOSER ("name must appear") and JUDGE ("here is what counts as participation") is lost. Claude — and the V35a evidence suggests this is real — was using that structural separation as a signal.

Verdict: this is a retryable failure mode. A stricter Technique-3 application can preserve M3's name-anchored framing while keeping the rest of the merge intact.

---

## 3. Alternative Merge Strategies — Comparison Table

Cost assumptions: 1 dataset × Claude Sonnet × single-step entity_candidates harness invocation ≈ 8 min wall time and ~$0.40 (BBB has the highest token count; smaller datasets ~3 min, ~$0.15). 5-dataset Round 2 sweep ≈ 30 min, ~$1.50. Round 1 single-dataset probe (mediastore) ≈ 8 min, ~$0.40.

| Strategy | Mechanism change | Predicted F1 outcome | Cost to ablate | Risk of same failure mode |
|---|---|---|---|---|
| **(a) Schema-driven cases** — convert the 6 ENT include rules into a JSON `case_id: enum` field so the LLM tags each candidate by which inclusion rule fired; let the schema list constrain output. Keep the prompt rule list compact. | Reduces extraction prompt surface by moving rule enumeration to the output schema. Each candidate now carries provenance (which rule fired). Downstream validation can short-circuit if case_id ∈ {pure-name, alias} (cheaper) vs case_id ∈ {participation, passive} (judge-required). | Likely **neutral or +0.5pp** on Claude; could regress on GPT (which over-uses generic enum values per V32 GPT findings). The schema does not constrain the proposer's *reasoning*, only its output format — so M3-style drift could still occur (LLM tags a Frontends-FP as case_id=participation). | High: 1 Round-1 probe + 5-dataset Round 2 + 5-dataset GPT = ~70 min, ~$3. | **Medium-high.** Schema enforcement is verdict-format only. The proposer still sees the rule body and can still widen M3 internally. Does not directly fix the BBB failure mode. |
| **(b) SHARED rubric, SEPARATE prompts with strict role-specific scoping** — keep both prompts; share the *inclusion* core; put exclusion rules + tie-breakers ONLY in the validation prompt. Extraction prompt sees only "name must appear" rules. | This is what 12-04 *claimed* to do ("rubric-shared / decision-divergent") but didn't: the merged rubric is shared between both prompts in 12-04. (b) makes the rubric truly role-divergent: M1-M2-M4-M5-M6 (name-anchored proposer rules) in extraction; M3 + all M7-M10 + VAL framing in validation. Restores the structural separation Claude was using. | **Best predicted F1 on Claude (94.5–95% macro).** Restores the V31 design rationale ("extract aggressively on name presence, validate strictly on participation type"). Should recover the BBB precision. | 1 Round-1 probe + 5-dataset Round 2 ≈ 40 min, ~$1.90 Claude-only. | **Low.** Directly targets the diagnosed mechanism. The proposer never sees VAL_APP_1's wording. |
| **(c) Distill ENT and VAL individually (no merge)** — apply Technique 3 inside each prompt: collapse the 6 inc + 2 exc into 5 rules in ENT; collapse 3 APPROVE + 3 REJECT into 4 rules in VAL. No cross-prompt sharing. | Half the rule reduction (≈ 2 rules saved instead of 4). Zero risk of cross-prompt boundary erasure. The remaining variance is intra-prompt and well-bounded. | **Likely +0pp ±0.5pp** on both backends. Preserves V31 design. Modest defensibility/readability win. | 2 Round-1 probes (one per prompt) + 5-dataset Round 2 ≈ 50 min, ~$2.40. | **Very low.** Smallest possible change. Pareto-safe but pareto-tiny. |
| **(d) ENT-only merge (preserve VAL_RULES verbatim)** — restructure only ENT side; leave VAL untouched. | Tests whether the regression originated in the extraction prompt. If ENT-only restructure with the same M3 drift reproduces the BBB regression → confirms §1's diagnosis. If it does NOT reproduce → the regression was about the *combined* prompts, not the individual ENT change. | If we replicate 12-04's M3 wording verbatim: predicted regression (replicates §1). If we use (b)'s name-anchored M3: predicted +0pp. Best as a **diagnostic ablation**, not a promotion target. | 1 Round-1 probe (BBB only — the regressing dataset) ≈ 8 min, ~$0.40 Claude. | **Diagnostic only.** Not a promotion candidate by itself. |
| **(e) VAL-only restructure via Technique 8 (reasoning-before-conclusion)** — leave ENT untouched. Restructure VAL_RULES to lead with the consideration ("first weigh whether the sentence references the component as a participant; then issue APPROVE/REJECT"). | Targets Technique 8 from the survey (arXiv 2603.13351). Tests whether ordering directive ordering recovers any of V35a's losses without rule deletion. | **Likely 0pp Claude, possibly +1pp GPT** (per survey's Technique 8 reasoning about verdict-first prompts being a known GPT weakness). | 5-dataset Round 2 ≈ 30 min, ~$1.50 Claude + ~$1.50 GPT. | **Low.** Does not touch the diagnosed BBB extraction failure mode. Orthogonal trim that could land independently. |
| **(f) Cache-shift** — move rule lists into the system prompt for cache benefit; no rule reduction. Reduces per-call token surface but not prompt content. | Pure infra change. Saves API cost. No F1 effect predicted. | **0pp F1.** Cost reduction only. | 1 Round-1 probe + 1 GPT sanity check ≈ 16 min, ~$0.80. | **Negligible F1 risk.** Defer to v2.2 as it does not address PROMPT-02. |

---

## 4. BBB-Specific Failure Mode

### Is this BBB-only?

**Yes, mechanistically.** The merged-rubric M3 drift is triggered by sentences where:
1. A subordinate role-label (subprocess name, sub-component, abstract role) is the subject;
2. That role-label is NOT in the global-scope alias dictionary;
3. The architectural component whose participation pattern matches sits in a different paragraph or section.

BBB has a 9-sentence contiguous block (s24–s32) of exactly this pattern, plus the BigBlueButton API → BBB web mismatch (s39). Together these 10 sentences account for all of trim2's added BBB FPs.

The other datasets don't have this pattern:
- **mediastore, teastore**: components are CamelCase and named directly; no subprocess-role indirection in the SADs.
- **teammates**: low-variance, names match directly; the merged rubric on teammates actually *helped* (3 new TP candidates from broader extraction, 3 more TP approvals from broader judge — masked by 2 FPs, net neutral).
- **jabref**: high English-vocabulary overlap is already handled by M8 (ordinary English word exclusion), which the merge preserves verbatim.

### Will BBB always limit aggressive ENT/VAL trim?

Probably — but not always for the same reason. The 12-04 regression is mechanistic and **fixable** by strategy (b) (name-anchored M3 in the shared rubric). The deeper BBB challenge is high variance in surface forms (technology-named components, abstract role-labels, subprocess descriptions). Any trim that REMOVES the subprocess-exclusion rule (M10 in 12-04's numbering, original VAL_REJ_3) would risk re-triggering the same pattern. M10 must be retained in any future merge. 12-04 retains it textually but it sits next to M3 in the same rubric, which gives Claude no clear precedence signal in close cases.

A defensible BBB-aware trim must:
1. Keep M3 (participation rule) on the validation side only;
2. Keep M10 (subprocess exclusion) on the validation side only;
3. Let the extraction proposer operate on the strict name-anchored subset (M1, M2, M4, M5, M6).

Strategy (b) does all three.

---

## 5. Recommended Next Steps (Phase 12 vs v2.2)

### Phase 12 retries (rank-ordered by EV under v2.1 RELAXED gate: macro ≥ 0.90, BBB absolute ≥ 0.79)

**Pick 1: Strategy (b) — SHARED rubric, role-divergent inclusion/exclusion routing.** Highest EV given the diagnosis. Phase 12 budget: ~$2 Claude + ~$2 GPT, ~70 min wall time including Round 1 probe + 5-dataset Round 2 + GPT cross-model arm. F1 envelope prediction: macro 0.94–0.95, BBB 0.79–0.83. Probability of passing v2.1 RELAXED gate: ~70%. Probability of passing the old strict gate (macro ≥ 0.93, BBB delta ≥ −0.06): ~55%. Recommend as Plan 12-04b.

**Pick 2: Strategy (c) — Individual distill, no merge.** Lower upside (~2 rule savings vs 4) but very high probability of passing both gates (~90% pass relaxed, ~75% pass strict). Phase 12 budget: ~$2.40, ~50 min. F1 envelope: macro 0.945–0.955, BBB 0.79–0.83. Recommend as Plan 12-04c only if (b) fails. The defensibility argument is "Technique 3 applied at appropriate scope (intra-prompt)".

### Defer to v2.2

- Strategy (a) schema-driven: too speculative under Phase 12 time pressure; the schema-routing benefit is mostly cost not F1. v2.2 prompt-cost-reduction work.
- Strategy (d) ENT-only: diagnostic, not promotion. Skip unless §1's M3 diagnosis is contested.
- Strategy (e) VAL-only Technique 8: GPT-targeted, defer to v2.2 cross-model work after the v2.1 single-model trim is settled. Independently land-able later.
- Strategy (f) cache-shift: pure cost work, not in v2.1 scope per REQUIREMENTS.md "Cost optimization out of scope".

### Concrete Phase 12 recommendation

Plan 12-04b: implement Strategy (b) as a new variant `s_linker13_trim2_entval_v2_clean`. Test design:
- ENTITY_EXTRACTION_RULES_V4 = `_EXTRACTION_HEADER` + name-anchored rules (M1, M2, M4, M5, M6) + EXT_tiebreak
- VALIDATION_RULES_V4 = `_VALIDATION_HEADER` + participation rules (M3 verbatim from original VAL_APP_1/2/3) + exclusion rules (M7, M8, M9, M10)
- No shared-core constant. Rule count: 6 (ext) + 7 (val) = 13. Net reduction = 1 rule (vs original 14). Technique 3 applied conservatively: cross-prompt collapse rejected; per-prompt collapse minimal.

If 12-04b passes: promote to Plan 13-01 as the entity trim. If fails: do not run 12-04c (the rule savings is too small to justify additional ablation budget against the v2.1 cutoff); accept that ENT+VAL merge is not Phase 12-feasible and move on with the originals into `prompts_v3.py`.

---

## 6. Honest Limitations

1. **The 17 FN figure in the verdict cannot be fully resolved without re-running the full pipeline.** The harness scoring counts gold-not-in-final-links, but the harness blocks seed_val and coref. In a real end-to-end run, some of those "missing" gold links would have been recovered by seed_val flips on the TRIM's seed candidates — but the harness pins seed/coref to baseline. The merge's effect on recall in a real pipeline is therefore unmeasurable from this data. The +10 FP analysis is unaffected (extraction-only effect).

2. **GPT cross-model behavior is unverified.** Round 3 was skipped per protocol. Strategy (b)'s prediction of similar Claude behavior on GPT is theoretical. The 12-04 failure prevented learning whether the same merge pattern would help or hurt GPT. Strategy (b) needs the full Round 3 to confirm.

3. **The "M3 drift" diagnosis is based on rule-text reading, not on model-internal evidence.** A more rigorous test would prompt Claude with a held-out subprocess-role sentence under both rubrics and check which rule the model cites in its (extended-thinking-emitted) reasoning. Within Phase 12 budget this was not run; should be the first thing Plan 12-04b does in its Round 1 probe.

4. **Strategy (b)'s rule renumbering may itself introduce LLM-side surface-variance noise.** Claude is sensitive to numbered-rule reorderings (per the V35 series). The 12-04b plan should keep numbering as close to the originals as possible to minimize confounders.

5. **Teammates' apparent BENEFIT from 12-04 (+3 TP candidates, +3 TP approvals) is observed but not explained.** Same M3 widening that hurt BBB may have HELPED teammates by recovering "E2E", "Storage" candidates the strict baseline missed. Strategy (b) trades this away. Whether the net is positive for teammates under (b) is empirical, not diagnosable without re-running.

6. **The merged rubric's M3 absorption of VAL_APP_1 was textually documented in the variant's docstring as "rubric-shared / decision-divergent" — but the decision divergence was implemented via different *headers*, not different *rule sets*. This was a subtle planning vs execution mismatch.** A future Plan 12-04b should make the implementation contract more explicit: which rules go in which prompt's body, not just how the prompt is framed.

---

## 7. Pointers (file paths only)

- Variant source: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py`
- Original prompts: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/src/llm_sad_sam/linkers/experimental/prompts_v2.py` (lines 179–205)
- Baseline cache for BBB: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/results/phase_cache/s_linker13_clean/bigbluebutton/`
- Trim's tmp cache (entity_candidates.pkl, entity_decisions.pkl) — the live forensic data for this investigation: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/results/ablation_results/12_04_trim2_entval/claude/s_linker13_trim2_entval_clean/bigbluebutton/_phase_cache_tmp/s_linker13_trim2_entval_clean/bigbluebutton/`
- Verdict: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/results/ablation_results/12_04_trim2_entval/verdict.json`
- Original plan: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/phases/12-trim-ablation/12-04-step2-entval-merge-PLAN.md`
- Survey Technique 3 framing: `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/.planning/research/PROMPT-HARNESS-SURVEY.md` §2 Technique 3 + §6 V35a lesson
- BBB SAD source: `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/bigbluebutton/text_2021/bigbluebutton.txt`
- BBB gold standard: `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv`
