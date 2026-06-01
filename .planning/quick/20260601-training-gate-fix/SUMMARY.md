---
status: complete
quick_id: 20260601-training-gate-fix
date: 2026-06-01
---

# Quick Task: Training Gate + Feedback Scope Fixes

## Changes

### 1. Convergence gate: F1 → D-proposes-zero (commit 238821f)
`voyager_train_tlr_v4_beta.py` line 881:

Before: `"converged": committed_macro >= CONVERGENCE_THRESHOLD`  
After: `"converged": len(accepted) == 0 and len(removals) == 0`

F1 gate said "good enough to stop" but a project at F1=0.97 with 1-2 clean FPs still has generalizable signal. D-proposes-zero means "nothing left to learn" — the true convergence condition.

### 2. FP sentence context added to Oracle prompt (commit 6cd9f9f)
Before: Oracle received only `fn_context_sample` (sentence text around FNs).  
After: Also receives `fp_context_sample` (±1 sentence window around FPs).

Without FP text, Oracle couldn't identify "effect-only sentence" or "algorithm description" patterns — exactly the clean generalizable errors from high-F1 projects.

### 3. Per-project D calls replace combined call (commit 215e70f)

**Ablation result** using probe pass-1 oracle JSONs:

| Condition | Patterns | Unique slots |
|-----------|----------|-------------|
| Combined D (1 call) | 3 | 3: EXTRACTION, JUDGE, VALIDATION |
| Per-project D (3 calls) | 5 unique | 5: + SEED_DISAMBIGUATION, COREF |

TM's `evidence_count=14` FM drowned out MS/TS distinct signals in combined mode. Per-project D gives every project equal representation before dedup + GATE-06.

## Ablation artifacts
- `results/voyager_v4_beta/mainline/ablation_d_combined.json` — combined D output (iter91)
- `results/voyager_v4_beta/mainline/ablation_d_per_{ms,ts,tm}.json` — per-project D outputs (iter92-94)
- `scripts/ablation_d_scope.py` — reusable ablation script

## Verification
`python scripts/voyager_train_tlr_v4_beta.py probe --projects mediastore --dry-run`  
→ `converged=False (accepted=1, removals=0)` — gate driven by D proposals, not F1. ✓
