---
phase: 12
plan: voyager-pilot-gpt
subsystem: prompt-curriculum
tags: [voyager, train-test, gpt-5.4, skill-bank, distillation, gate-06]
key-files:
  created:
    - results/voyager_pilot/distilled_skills.json
    - results/voyager_pilot/test_results.json
    - results/voyager_pilot/defensibility_audit.json
    - .planning/phases/12-trim-ablation/12-VOYAGER-PILOT-GPT-SUMMARY.md
  modified:
    - scripts/voyager_train_tlr.py (added --resume)
    - results/voyager_pilot/skill_bank.json (7→10 patterns)
    - results/voyager_pilot/train_trajectory.json (6→10 rows)
decisions:
  - "Voyager-style train-test methodology marginally transfers on gpt-5.4 for SAD-SAM TLR (Δ +0.16 pp on held-out macro vs axiom-only floor); the transfer is real on jabref (+2.7 pp) but a regression on bigbluebutton (−2.4 pp), so the average masks per-dataset variance."
  - "Distilled 6-skill bank does NOT beat the trim1-distilled baseline on gpt-5.4 (0.9045 vs 0.9173 all-5 macro); skill-bank curriculum and v2.1 distilled-prompts are competing for the same effect."
metrics:
  duration_min: 12.5
  total_llm_calls: 18
  total_wallclock_s: 750
  patterns_in_bank: 10
  patterns_in_distilled: 6
  patterns_gate06_rejected: 8
---

# Phase 12 Voyager Pilot (gpt-5.4 resumption): Summary

**One-liner:** Resumed the Voyager-style train-test pilot on gpt-5.4 (Claude budget had restored but user directive prioritised gpt backend). Training converged at macro 0.9223 with 10 skill-bank patterns; distillation produced 6 GATE-06-clean universal rules; held-out gpt-5.4 macro on BBB+JAB is 0.8846 (distilled) vs 0.8830 (axiom-only floor) — a +0.16 pp lift, *within noise* and below the trim1-distilled gpt-5.4 baseline of 0.8883 on the same two projects.

---

## 1. Resumption Context

| Aspect | Value |
| --- | --- |
| Backend | OpenAI gpt-5.4 (all calls; zero Claude calls in this resumption) |
| Resumption commit | `2c21298` (feat: add --resume) |
| Prior state preserved | 7 Claude-derived skill bank patterns; outer-0 trajectory (6 rows); 5 prior rejections |
| Wallclock budget | $40 / 4 h cap; **used ~12.5 min, well under budget** |
| LLM calls | 18 (training: 6, distill: 1, distill reviewer: 1, test: 10) |
| Backup | `results/voyager_pilot/_resumption_2026-06-01_pre_gpt54_backup/` (pre-resume skill bank + trajectory snapshot) |

The Claude-derived 7-pattern skill bank was loaded as a warm start; outer-pass index was offset to 1 so new rows did not collide with prior rows in `train_trajectory.json`. Every new trajectory row and rejected pattern is tagged with `"backend": "openai", "model": "gpt-5.4"` so the audit trail is unambiguous.

## 2. Training Trajectory (outer-1 on gpt-5.4)

| Outer | Project | Inner iter | F1 | FP | FN | Skills before | Backend |
|---|---|---|---|---|---|---|---|
| 0 | teastore | 0 | 0.9818 | 1 | 0 | 0 | claude (prior) |
| 0 | mediastore | 0 | 0.9333 | 1 | 3 | 0 | claude (prior) |
| 0 | mediastore | 1 | 0.9508 | 1 | 2 | 1 | claude (prior) |
| 0 | teammates | 0 | 0.8226 | 16 | 6 | 1 | claude (prior) |
| 0 | teammates | 1 | 0.8000 | 18 | 7 | 3 | claude (prior) |
| 0 | teammates | 2 | 0.8197 | 15 | 7 | 5 | claude (prior) |
| **1** | **teammates** | **0** | **0.7903** | 18 | 8 | 7 | **gpt-5.4** |
| **1** | **teammates** | **1** | **0.8160** | 17 | 6 | 9 | **gpt-5.4** |
| **1** | **teastore** | **0** | **1.0000** | 0 | 0 | 10 | **gpt-5.4** |
| **1** | **mediastore** | **0** | **0.9508** | 1 | 2 | 10 | **gpt-5.4** |

| Pass | Train macro F1 |
|---|---|
| outer 0 (Claude warm-start) | 0.9174 |
| outer 1 (gpt-5.4 add-3) | 0.9223 |

Training converged on the env-override threshold (`VOYAGER_CONV_THRESH=0.90`). Teammates remained the hard project — gpt-5.4 inner iters added 2 new patterns each round but did not break the 0.85 wall.

## 3. Skill Bank Growth (10 patterns total)

| # | Origin | Scope | One-line gist |
|---|---|---|---|
| 1 | Claude | DOC_KNOWLEDGE_JUDGE_RULES | Block link when sentence describes only outcome/quality without operation/actor |
| 2 | Claude | DOC_KNOWLEDGE_JUDGE_RULES | Block link to presentation/utility when only stack/tech is described |
| 3 | Claude | COREF_RULES | Resolve `it / this component` only if sentence continues responsibility |
| 4 | Claude | DOC_KNOWLEDGE_JUDGE_RULES | Approve link to service/domain module if sentence assigns it entry-point/business-logic role |
| 5 | Claude | DOC_KNOWLEDGE_JUDGE_RULES | Block link to helper/adapter/library merely because a class acts |
| 6 | Claude | DOC_KNOWLEDGE_JUDGE_RULES | Approve link to coordinator/facade module if sentence is structural with collaborators |
| 7 | Claude | DOC_KNOWLEDGE_JUDGE_RULES | Block link to utility/adapter when sentence describes a specific sender/queuer/service class |
| **8** | **gpt-5.4** | **DOC_KNOWLEDGE_JUDGE_RULES** | **Approve link to service/domain module if described as controller/facade/API surface (structural, not execution)** |
| **9** | **gpt-5.4** | **DOC_KNOWLEDGE_JUDGE_RULES** | **Block link to client-facing layer when sentence only labels tech stack or another dispatcher handles the data** |
| **10** | **gpt-5.4** | **DOC_KNOWLEDGE_JUDGE_RULES** | **Block link to shared utility when sentence only describes configuration/provider selection/sender class actions** |

Patterns 8-10 are gpt-5.4-derived and visibly variations on patterns 4-7 — gpt-5.4's contribution is **calibration restatement**, not new rules. This is consistent with the V35 finding that Claude prompts sit at a local optimum: gpt-5.4 did not discover qualitatively new patterns, it re-expressed the existing ones in subtly different framings.

### 3.1 GATE-06 Audit (skill bank, strict regex)

| Source | Patterns | Taboo hits |
|---|---|---|
| Skill bank (10 patterns) | 10 | **0** |
| Distilled bank (6 patterns) | 6 | **0** |
| Total rejected during training | 8 | (terms: `persistence`, `backend`, `datastore`) |

The training-time filter is critical — 8 of 18 LLM-proposed patterns hit taboo terms before the gpt-5.4 distillation stage. The rejection log shows gpt-5.4 has the same drift as Claude: it reaches for `persistence`/`datastore` whenever the teammates dataset (which is dominated by storage-related sentences) feeds back FP/FN evidence.

## 4. Distillation (6 universal rules, gpt-5.4)

| # | Scope | Rule |
|---|---|---|
| 1 | DOC_KNOWLEDGE_JUDGE_RULES | Approve link to service/controller/facade/domain ONLY when explicitly assigned an interface/entry-point/coordination/business-rule role |
| 2 | DOC_KNOWLEDGE_JUDGE_RULES | Block link to processor/transformer/handler when sentence describes only outcome/quality without operation, data flow, or responsible component |
| 3 | DOC_KNOWLEDGE_JUDGE_RULES | Block link to view/client/presentation when sentence is only a tech label or behaviour is done by another dispatcher/servlet |
| 4 | DOC_KNOWLEDGE_JUDGE_RULES | Block link to utility/adapter/broker/connector/infrastructure unless sentence states the component owns the reusable capability |
| 5 | COREF_RULES | Resolve pronoun/generic ref only while discourse preserves the previous component's responsibility and actor context |
| 6 | COREF_RULES | Block pronoun inheritance when sentence shifts responsibility to a different component, transport, layer, or external system |

**Reviewer-defensibility check** (single gpt-5.4 critic call): all 6 rules rated `defensible` — i.e. derivable from any architecture documentation, not from project-specific facts. The defensibility audit log is at `results/voyager_pilot/defensibility_audit.json`. Compared with the 10-pattern bank, the distillation:
- Consolidated patterns 4/6/8 → distilled rule 1 (service/controller/facade approval)
- Consolidated patterns 1 → distilled rule 2
- Consolidated patterns 2/9 → distilled rule 3
- Consolidated patterns 5/7/10 → distilled rule 4
- Promoted pattern 3 → distilled rules 5+6 (split into two coref directions)

## 5. Held-out Evaluation (gpt-5.4, frozen `distilled_skills.json`)

### 5.1 Per-project F1

| Project | Split | s_linker13 gpt-5.4 (baseline) | trim1 gpt-5.4 | trim9 gpt-5.4 | Voyager axiom-only | Voyager distilled | Δ distilled vs axiom |
|---|---|---|---|---|---|---|---|
| mediastore | train_sanity | 0.9677 | 0.9333 | 0.8966 | 0.9677 | 0.9677 | 0.00 |
| teastore | train_sanity | 1.0000 | 0.9818 | 1.0000 | 1.0000 | 0.9455 | **−5.45 pp** |
| teammates | train_sanity | 0.7939 | 0.8947 | 0.8522 | 0.7967 | 0.8403 | **+4.36 pp** |
| **bigbluebutton** | **test** | 0.8037 | 0.8036 | 0.7818 | **0.7931** | **0.7692** | **−2.39 pp** |
| **jabref** | **test** | 0.9730 | 0.9730 | 0.9730 | **0.9730** | **1.0000** | **+2.70 pp** |
| **Macro all-5** | — | **0.9077** | **0.9173** | **0.9007** | **0.9061** | **0.9045** | **−0.16 pp** |
| **Macro held-out (BBB+JAB)** | test | 0.8884 | 0.8883 | 0.8774 | **0.8830** | **0.8846** | **+0.16 pp** |

### 5.2 Comparisons against shipped baselines (held-out macro)

| System | Macro on {BBB, JAB} | Δ vs Voyager distilled |
|---|---|---|
| s_linker13 gpt-5.4 (clean baseline, Phase 10) | 0.8884 | +0.38 pp |
| trim1 gpt-5.4 (distilled judge rules, Phase 12-03) | 0.8883 | +0.37 pp |
| trim9 gpt-5.4 (runtime seed disambig, Phase 12-12) | 0.8774 | −0.72 pp |
| **Voyager distilled gpt-5.4** | **0.8846** | — |
| Voyager axiom-only gpt-5.4 (floor) | 0.8830 | −0.16 pp |

## 6. Central Question: Does Voyager-style train-test methodology transfer on gpt-5.4 for many-decision TLR classification?

**Answer: marginally, and not in a way that wins.** Three layered findings:

### Finding 1 — Skills DO survive frozen test-time injection without obvious overfit collapse.
The distilled 6 rules, frozen on disk and applied to never-seen BBB + JAB documents on gpt-5.4, do not collapse held-out F1. Macro held-out *axiom-only* = 0.8830 vs *distilled* = 0.8846 (+0.16 pp). This rules out the strong-overfit hypothesis (skills would have hurt held-out F1).

### Finding 2 — Per-dataset variance dominates the macro delta.
On the same frozen rules:
- **jabref** gains +2.70 pp (distilled exceeds the s_linker13 gpt-5.4 ceiling of 0.9730 → 1.0000).
- **bigbluebutton** loses −2.39 pp (skills add 2 more FPs vs axiom).
The skills' "approve link to service/controller/facade" rule helps jabref (the only project where there is a clean class-name correspondence) and hurts BBB (where component names like *HTML5 Server* / *Recording Service* trigger spurious facade approvals). Same effect, opposite signs, almost-perfect cancellation in the macro.

### Finding 3 — Voyager distilled does NOT beat the trim1-distilled baseline.
trim1 (Phase 12-03) is also a distilled-judge-rules variant, but distilled from a Claude run rather than learned via gpt-5.4 train-test. On gpt-5.4:
- trim1 macro (all 5) = **0.9173**
- Voyager distilled macro (all 5) = **0.9045** (Δ −1.28 pp)
- s_linker13 gpt-5.4 (no distillation at all) = **0.9077** (Δ +0.32 pp vs Voyager)

The Voyager curriculum, with the abstraction overhead of going through 10 abstract patterns → 6 distilled rules, lands roughly *at* the no-distillation baseline and *below* the trim1 hand-shaped distillation. The train-test methodology adds organisational rigour but, on this benchmark, no measured F1 above the already-distilled v2.1 baselines.

### Finding 4 — Backend asymmetry: Claude warm-start did most of the lifting.
Of the 10 final patterns, 7 came from outer-pass-0 (Claude). gpt-5.4's outer-pass-1 contributed 3 patterns that are *variations* on Claude's, not new dimensions. Cross-backend curriculum learning *was* feasible — but the gpt-5.4 increment was small enough that one could plausibly ablate the gpt-5.4 pass and get the same distilled output. (Not tested in this resumption — a single Claude-only distillation would be the cheaper falsification.)

### Verdict

The train-test methodology survives operationally on gpt-5.4 (resume support works, GATE-06 holds, defensibility critic catches nothing). It produces a reusable transfer artefact (`distilled_skills.json`). But the transfer **does not beat already-shipped Phase 12 distilled baselines**, and the per-dataset variance is high enough that the +0.16 pp held-out lift is well within the gpt-5.4 run-to-run variance band (±5-12 links per project per V32 GPT-5.2 notes). I would not present this pilot as a paper result; I would present it as **evidence that LLM-driven prompt-curriculum learning on small TLR datasets converges to roughly the same fixed point as hand-shaped distillation**.

## 7. Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Training script did not support resumption.** The original script overwrote `train_trajectory.json` at end of `train()`, and the outer index restarted at 0, which would have corrupted prior runs. Fix: added `--resume` flag that loads prior trajectory + rejected_log, offsets outer index past prior passes, and persists trajectory after every inner iter so budget cuts mid-loop preserve work. Commit `2c21298`.

**2. [Rule 2 - Critical correctness] Trajectory rows lacked backend/model provenance.** When resuming across backends, mixing Claude and gpt-5.4 rows in the same trajectory file made the audit ambiguous. Fix: tagged every new row + rejected pattern with `backend` and `model` fields. The prior 6 Claude rows are visible by absence of the tag (`backend=?` in display).

### Honest non-deviations

- **No skill-bank wiping**: the prior 7 patterns were preserved as warm-start (per execution-block constraint).
- **No frozen-file modification**: prompts_v3_axiom.py and s_linker13_skill_learned_clean.py untouched.
- **No re-baselining**: gpt-5.4 trim1/trim9 numbers are read from Phase 12-03 and 12-12 verdicts/files, not re-run.

## 8. Authentication Gates

None. OpenAI API key was already in `.env` and worked first-try.

## 9. Threat Flags

None. No new code surface introduced — only a `--resume` flag and JSON write-mode change on a script that was already gitignored output.

## 10. Self-Check: PASSED

- Resumption script: `scripts/voyager_train_tlr.py` — exists, `--resume` flag accepted.
- Skill bank: `results/voyager_pilot/skill_bank.json` — exists, 10 patterns, 0 taboo hits.
- Distilled bank: `results/voyager_pilot/distilled_skills.json` — exists, 6 patterns, 0 taboo hits.
- Defensibility audit: `results/voyager_pilot/defensibility_audit.json` — exists, 6 kept / 0 flagged.
- Test results: `results/voyager_pilot/test_results.json` — exists, macros computed, per-project FP/FN recorded.
- Train trajectory: `results/voyager_pilot/train_trajectory.json` — 10 rows, both Claude and gpt-5.4 tagged.
- Logs: `logs/voyager_gpt54/{train,distill,test}.log` — exist.
- Backup of pre-resume state: `results/voyager_pilot/_resumption_2026-06-01_pre_gpt54_backup/`.
- Commit `2c21298` (--resume support): present in git log.

All claims verified against on-disk artefacts.
