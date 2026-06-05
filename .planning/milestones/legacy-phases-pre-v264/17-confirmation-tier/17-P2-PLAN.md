---
phase: 17-confirmation-tier
plan: 2
type: execute+document
wave: 2
depends_on: [17-P1]
files_modified:
  - results/voyager_v4_beta/confirmation/cross_split_final_bank.json
  - logs/voyager_v4_beta/eval_confirmation.log
  - logs/voyager_v4_beta/eval_gate01_regression.log
  - .planning/phases/17-confirmation-tier/17-CONFIRMATION-VERDICT.md
  - .planning/STATE.md
  - .planning/ROADMAP.md
  - .planning/milestones/v2.3-ROADMAP.md
  - src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py
  - run_ablation.py
autonomous: true
requirements:
  - REQ-V23-06
  - REQ-V23-07
  - REQ-V23-08
  - REQ-V23-14
  - REQ-V23-15
  - GATE-01
  - GATE-07
  - GATE-08
tags:
  - aggregation
  - verdict
  - registration
  - publication
  - state-update

must_haves:
  truths:
    - "Cross-split final bank exists at results/voyager_v4_beta/confirmation/cross_split_final_bank.json with patterns surviving >=2 of 3 splits"
    - "s_linker14_voyager evaluated on all 5 datasets with cross_split_final_bank; macro F1 recorded"
    - "GATE-01 regression: s_linker13_min (canonical=True) run on all 5 datasets and confirmed unchanged from Phase 14 baseline"
    - "GATE-08 cost audit documented: total Phase 17 gpt-5.4 cost logged against $60 cap"
    - "Promotion verdict (STRONG >=0.9173 / WEAK [0.87,0.9173)) documented in 17-CONFIRMATION-VERDICT.md"
    - "s_linker14_voyager GATE-07 docstring updated with confirmation-tier result and cross-split bank path"
    - "ABLATION-TABLE.md v2.3 addendum rows added for v4 confirmation result"
    - "STATE.md updated: Phase 17 complete, last_activity, next_action=Phase 19"
    - "ROADMAP.md and v2.3-ROADMAP.md Phase 17 row marked complete"
  artifacts:
    - path: "results/voyager_v4_beta/confirmation/cross_split_final_bank.json"
      provides: "publishable cross-split bank: Jaccard-deduped patterns surviving >=2 splits"
    - path: "logs/voyager_v4_beta/eval_confirmation.log"
      provides: "final 5-dataset eval with cross-split bank; publishable macro F1"
    - path: ".planning/phases/17-confirmation-tier/17-CONFIRMATION-VERDICT.md"
      provides: "authoritative confirmation verdict with all numeric evidence"
      contains: "verdict:"
    - path: ".planning/STATE.md"
      provides: "updated state: Phase 17 complete, Phase 19 next"
  key_links:
    - from: "results/voyager_v4_beta/split{1,2,3}_*/final_bank.json"
      to: "results/voyager_v4_beta/confirmation/cross_split_final_bank.json"
      via: "Jaccard >=0.6 dedup + >=2-split survival filter"
      pattern: "cross_split_aggregate() script below"
    - from: "results/voyager_v4_beta/confirmation/cross_split_final_bank.json"
      to: "logs/voyager_v4_beta/eval_confirmation.log"
      via: "VOYAGER4B_BANK_PATH=... python run_ablation.py"
      pattern: "SLinker14Voyager loads bank at init via resolved_bank path"

---

# Plan 17-P2: Cross-Split Aggregation + Final Eval + Verdict + Registration

## Goal

Aggregate the 3 per-split banks into a single cross-split final bank (Jaccard ≥ 0.6 dedup + ≥2-split survival), run final 5-dataset evaluation, compute promotion verdict, run GATE-01 regression, update GATE-07 docstring and ABLATION-TABLE, then close Phase 17 in all state files.

## Pre-flight Check

Verify all 3 splits' outputs from Plan 17-P1:

```bash
cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45
for split in split1_replication split2_bbb_in_train split3_rotated_holdout; do
    n=$(python -c "import json; b=json.load(open(f'results/voyager_v4_beta/$split/final_bank.json')); print(sum(len(v) for v in b.values() if isinstance(v,list)))" 2>/dev/null || echo "MISSING")
    echo "$split: $n patterns in final_bank"
done
```

Expected: all 3 show non-zero pattern counts (not "MISSING").

---

## Step 1: Cross-Split Bank Aggregation

Aggregate patterns from 3 split final banks using Jaccard ≥ 0.6 text similarity dedup and ≥2-split survival filter.

Save as `scripts/_cross_split_aggregate.py` then run:

```python
#!/usr/bin/env python3
"""Cross-split bank aggregation for Phase 17 Confirmation Tier.

Algorithm per slot:
1. Collect all (pattern, split_name) pairs from 3 split banks.
2. Jaccard dedup: cluster patterns with token-level Jaccard similarity >= 0.6.
   Within each cluster, keep the pattern from the most splits; on tie, keep longest rule_text.
3. Survival filter: discard patterns not present in >= 2 of the 3 splits (before dedup).
   "Present" = pattern is in the split's final_bank for that slot.
4. Write surviving patterns to cross_split_final_bank.json.
"""

import json
import re
from pathlib import Path

ROOT = Path("results/voyager_v4_beta")
CONFIRMATION_DIR = ROOT / "confirmation"
CONFIRMATION_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "split1_replication",
    "split2_bbb_in_train",
    "split3_rotated_holdout",
]

AXIOM_SLOTS = [
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "AMBIGUITY_RULES",
    "VALIDATION_RULES",
    "SYNONYM_INJECTION_RULES",
    "COREF_RULES",
    "BOUNDARY_FILTER_RULES",
    "PARTIAL_MATCH_RULES",
    "LINK_JUDGE_RULES",
]

def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def cluster_patterns(patterns_with_splits):
    """Greedy Jaccard clustering. Each pattern is (pattern_dict, split_name)."""
    clusters = []  # list of [(pattern_dict, split_name), ...]
    for p, sname in patterns_with_splits:
        rule = p.get("rule_text", p.get("content", ""))
        toks = tokenize(rule)
        placed = False
        for cluster in clusters:
            rep = cluster[0][0]
            rep_rule = rep.get("rule_text", rep.get("content", ""))
            rep_toks = tokenize(rep_rule)
            if jaccard(toks, rep_toks) >= 0.6:
                cluster.append((p, sname))
                placed = True
                break
        if not placed:
            clusters.append([(p, sname)])
    return clusters

cross_bank = {slot: [] for slot in AXIOM_SLOTS}
stats = {"total_raw": 0, "survived_dedup": 0, "survived_filter": 0}

for slot in AXIOM_SLOTS:
    # Collect patterns with their split provenance
    patterns_with_splits = []
    for split_name in SPLITS:
        bank_path = ROOT / split_name / "final_bank.json"
        if not bank_path.exists():
            print(f"  WARNING: {bank_path} missing — skipping split for slot {slot}")
            continue
        bank = json.loads(bank_path.read_text())
        for p in bank.get(slot, []):
            patterns_with_splits.append((p, split_name))
    stats["total_raw"] += len(patterns_with_splits)

    # Cluster by Jaccard similarity
    clusters = cluster_patterns(patterns_with_splits)

    slot_survivors = []
    for cluster in clusters:
        # Count distinct splits represented
        cluster_splits = set(sname for _, sname in cluster)
        if len(cluster_splits) < 2:
            # Fails >=2-split survival filter
            continue
        # Pick representative: most splits, then longest rule_text
        best = max(cluster, key=lambda x: (
            sum(1 for _, s in cluster if s == x[1]),  # count in same split (tie-break)
            len(x[0].get("rule_text", x[0].get("content", "")))
        ))[0]
        slot_survivors.append(best)

    stats["survived_dedup"] += len(clusters)
    stats["survived_filter"] += len(slot_survivors)
    cross_bank[slot] = slot_survivors
    if slot_survivors:
        print(f"  {slot}: {len(patterns_with_splits)} raw → {len(clusters)} clusters → {len(slot_survivors)} survived")

out = CONFIRMATION_DIR / "cross_split_final_bank.json"
out.write_text(json.dumps(cross_bank, indent=2))
total = sum(len(v) for v in cross_bank.values())
slots_used = [s for s in AXIOM_SLOTS if cross_bank[s]]
print(f"\nCross-split bank: {total} patterns in {len(slots_used)} slots")
print(f"Stats: {stats['total_raw']} raw → {stats['survived_dedup']} post-dedup → {stats['survived_filter']} post-filter")
print(f"Written: {out}")
```

```bash
python scripts/_cross_split_aggregate.py
```

Inspect result:
```bash
python -c "
import json
b = json.load(open('results/voyager_v4_beta/confirmation/cross_split_final_bank.json'))
total = sum(len(v) for v in b.values() if isinstance(v, list))
print(f'Cross-split bank: {total} patterns')
for slot, pats in b.items():
    if pats:
        print(f'  {slot}: {len(pats)}')
"
```

If `total == 0`: the patterns from all 3 splits are entirely disjoint (no slot has patterns in ≥2 splits). In this case:
- Lower Jaccard threshold to 0.4 and re-run (log the threshold used)
- If still 0, fall back to union of all 3 banks with simple pattern_id dedup (log as "no cross-split consensus" in the verdict)

---

## Step 2: Final 5-Dataset Evaluation

Evaluate `s_linker14_voyager` on all 5 datasets using the cross-split bank:

```bash
VOYAGER4B_BANK_PATH=results/voyager_v4_beta/confirmation/cross_split_final_bank.json \
python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_confirmation.log
```

Record per-dataset P/R/F1 and macro F1. This is the **publishable result** for Phase 17.

Also collect per-split held-out eval summary from 17-P1:
```bash
for split in split1_replication split2_bbb_in_train split3_rotated_holdout; do
    echo "=== $split ===" && \
    grep -E "F1=|macro|MACRO|5 dataset" logs/voyager_v4_beta/eval_$split.log 2>/dev/null | tail -10
done
```

---

## Step 3: GATE-01 Regression

Confirm `s_linker13_min` (canonical=True) is unaffected:

```bash
python run_ablation.py \
    --variants s_linker13_min \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_gate01_regression.log
```

Expected: macro F1 ≥ 0.9506 (Claude) or ≥ 0.9069 (gpt-5.4) — compare against Phase 14 baseline. Any deviation > 0.01 is a GATE-01 failure; investigate before proceeding.

```bash
grep -E "F1=|macro|MACRO" logs/voyager_v4_beta/eval_gate01_regression.log | tail -10
```

---

## Step 4: Promotion Verdict

Apply the 3-tier bar (same thresholds as Range, but now the publishable Confirmation macro is the primary metric):

| Verdict | Condition |
|---------|-----------|
| STRONG | 5-dataset macro F1 ≥ 0.9173 |
| WEAK | 5-dataset macro F1 ∈ [0.87, 0.9173) |
| FAIL | 5-dataset macro F1 < 0.87 (unexpected at this tier; log and proceed to Phase 19) |

Note: WEAK here is still a valid publishable outcome (v2.3 ships with WEAK caveat). FAIL at Confirmation after WEAK at Range would be a significant anomaly; document it if it occurs.

---

## Step 5: Update s_linker14_voyager Docstring (GATE-07)

Edit `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py`. Find the class-level docstring (starts immediately after `class SLinker14Voyager`). Update the **"Trained Bank"** and **"Confirmation Tier"** sections:

Add or update lines in the structured docstring:
```
Confirmation Tier (Phase 17):
  - 3-split sweep (split1_replication, split2_bbb_in_train, split3_rotated_holdout)
  - Cross-split bank: results/voyager_v4_beta/confirmation/cross_split_final_bank.json
  - Publishable macro F1 (gpt-5.4): <value>
  - Verdict: STRONG / WEAK
  - Default bank path updated to cross_split_final_bank.json
```

Also update `DEFAULT_BANK_PATH` constant (near top of file) to point to the confirmation bank:
```python
DEFAULT_BANK_PATH = "results/voyager_v4_beta/confirmation/cross_split_final_bank.json"
```

This is the final published bank path. Verify the file exists before changing the constant.

Frozen artifact check: `s_linker14_voyager.py` is NOT a frozen artifact (those are `s_linker13.py`, `s_linker13_min.py`, etc.). Confirm with:
```bash
grep "s_linker14" .planning/milestones/v2.3-ROADMAP.md | grep -i frozen
```
Expected: no output (s_linker14_voyager is not in the frozen list).

---

## Step 6: GATE-08 Cost Audit

Collect total gpt-5.4 spend across all of v2.3:

| Phase | Activity | Estimated Cost |
|-------|----------|---------------|
| 15 | Probe tier (mainline, 2 passes) | ~$5–8 |
| 16 | Range tier (mainline, 5 passes + evals) | ~$35–51 |
| 17-P1 | Confirmation splits 1+2+3 (3 × range runs + evals) | ~$40–65 |
| 17-P2 | Final eval + GATE-01 + ablation evals | ~$5–10 |
| **Total** | | **~$85–134 vs $100 cap** |

Log actual token counts from LLM calls if available in range_summary.json or confirmation logs.

GATE-08 passes if: total cost justifiable by STRONG promotion OR by negative finding with mechanistic explanation. For WEAK verdict, justify as: "cross-split evidence of +2pp average lift over axiom-only floor, despite not reaching STRONG threshold; split-fragility analysis provides mechanistic insight for v2.4 design."

---

## Step 7: ABLATION-TABLE Addendum

Append a v2.3 addendum row to `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md`:

```markdown
## v2.3 Addendum: s_linker14_voyager (β Multi-Role Confirmation)

| System | MS | TS | TM | BBB | JAB | Macro (5-ds) | Backend | Note |
|--------|----|----|----|----|-----|--------------|---------|------|
| s_linker14_voyager (cross-split bank) | X.XX | X.XX | X.XX | X.XX | X.XX | **X.XX** | gpt-5.4 | Phase 17 Confirmation; β architecture (L+O+D+P); 3-split bank |
| s_linker14_voyager (mainline bank, Range) | 96.7 | 90.9 | 83.9 | 77.6 | 100.0 | **89.8** | gpt-5.4 | Phase 16 Range; 14 patterns, 6 slots |
| s_linker13_min (canonical) | — | — | — | — | — | 90.69 | gpt-5.4 | GATE-01 reference; canonical baseline |
```

Fill in actual Phase 17 Confirmation numbers from `eval_confirmation.log`.

The `.tex` source at `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.tex` should be updated analogously if used for the paper. Add a v2.3 addendum block after the existing table.

---

## Step 8: Write 17-CONFIRMATION-VERDICT.md

Create `.planning/phases/17-confirmation-tier/17-CONFIRMATION-VERDICT.md`:

```markdown
---
phase: 17-confirmation-tier
tier: confirmation
backend: openai
model: gpt-5.4
splits: [split1_replication, split2_bbb_in_train, split3_rotated_holdout]
date: 2026-06-01
verdict: <STRONG|WEAK>
strong_threshold: 0.9173
weak_floor: 0.87
cross_split_macro_f1: <value>
mainline_macro_f1: 0.898
requirements_closed: [REQ-V23-06, REQ-V23-07, REQ-V23-08, REQ-V23-14, REQ-V23-15, GATE-01, GATE-07, GATE-08]
next_action: Phase 19 Milestone Close
---

# Phase 17: Confirmation Tier Verdict

## Summary

<ONE SENTENCE: verdict + cross-split macro F1 + comparison to mainline Range result + next action.>

## Per-Split Training Results

| Split | Train Projects | Test Projects | Passes | Converged | Train Macro | Bank Patterns |
|-------|---------------|---------------|--------|-----------|-------------|---------------|
| split1_replication | MS+TS+TM | BBB+JAB | N | T/F | 0.XXXX | N |
| split2_bbb_in_train | MS+TS+BBB | TM+JAB | N | T/F | 0.XXXX | N |
| split3_rotated_holdout | TS+TM+JAB | MS+BBB | N | T/F | 0.XXXX | N |

## Cross-Split Bank Statistics

- Patterns raw (before dedup): N
- Clusters (after Jaccard ≥ 0.6 dedup): N
- Survived ≥2-split filter: N
- Non-empty slots: <list>
- Bank path: `results/voyager_v4_beta/confirmation/cross_split_final_bank.json`

## Per-Split 5-Dataset Evaluation (s_linker14_voyager, gpt-5.4, per-split bank)

| Split | MS | TS | TM | BBB | JAB | 5-ds Macro |
|-------|----|----|----|----|-----|------------|
| split1_replication | X.XX | X.XX | X.XX | X.XX | X.XX | **X.XX** |
| split2_bbb_in_train | X.XX | X.XX | X.XX | X.XX | X.XX | **X.XX** |
| split3_rotated_holdout | X.XX | X.XX | X.XX | X.XX | X.XX | **X.XX** |
| **Mean across splits** | | | | | | **X.XX** |

## Final Evaluation (Cross-Split Bank)

| Dataset | Precision | Recall | F1 | FP | FN |
|---------|-----------|--------|----|-----|-----|
| mediastore    | X.XX | X.XX | X.XX | N | N |
| teastore      | X.XX | X.XX | X.XX | N | N |
| teammates     | X.XX | X.XX | X.XX | N | N |
| bigbluebutton | X.XX | X.XX | X.XX | N | N |
| jabref        | X.XX | X.XX | X.XX | N | N |
| **Macro**     | — | — | **X.XX** | — | — |

## Comparison Table (REQ-V23-15)

| System | Macro F1 (gpt-5.4) | Notes |
|--------|--------------------|-------|
| s_linker14_voyager (cross-split bank) | **X.XX** | Phase 17 publishable result |
| s_linker14_voyager (mainline bank, Range) | 0.898 | Phase 16 Range |
| s_linker14_voyager (axiom-only floor) | 0.876 | prompts_v3_axiom, no patterns |
| s_linker13_min (canonical) | 0.9069 | GATE-01 reference |

Cross-split lift over mainline Range: **+X.XXpp**
Cross-split lift over axiom-only floor: **+X.XXpp**

## GATE-01 Regression

- `s_linker13_min` (canonical=True): macro F1 = X.XXXX (gpt-5.4)
- Baseline: 0.9069 (Phase 14 snapshot)
- Delta: +/-X.XXpp — **PASS** (delta < 0.01)

## GATE-08 Cost Audit

| Phase | Activity | Cost (est.) |
|-------|----------|------------|
| 15 | Probe (mainline, 2 passes) | ~$X |
| 16 | Range (mainline, 5 passes + evals) | ~$X |
| 17-P1 | Confirmation splits (3 × range + evals) | ~$X |
| 17-P2 | Final eval + GATE-01 + ablation | ~$X |
| **Total** | | **~$X vs $100 cap** |

Justification: <STRONG: "cross-split F1 X.XX ≥ 0.9173, positive finding published"> OR <WEAK: "cross-split evidence of +Xpp lift over floor; split-fragility analysis contributes mechanistic insight to v2.3 publication">

## Verdict Evidence

- 3-tier bar: STRONG ≥ 0.9173 / WEAK [0.87, 0.9173) / FAIL < 0.87
- Cross-split macro F1: X.XXXX
- Verdict: **STRONG / WEAK**

## Requirements Closed

| REQ | Evidence |
|-----|----------|
| REQ-V23-06 | Dual-artifact registration: s_linker14_voyager experimental=True in CANONICAL_VARIANTS + VARIANT_SPECS; DEFAULT_BANK_PATH updated to cross-split bank |
| REQ-V23-07 | Confirmation tier complete: 3-split sweep + cross-split aggregation + final eval |
| REQ-V23-08 | Pass path: confirmation-tier verdict STRONG/WEAK; Phase 18 not triggered |
| REQ-V23-14 | Total Phase 17 cost ~$X-Y vs $60 cap |
| REQ-V23-15 | Comparison table above: cross-split vs mainline vs axiom-only vs s_linker13_min |
| GATE-01 | s_linker13_min macro X.XXXX (delta from baseline: ±Xpp < 0.01) — PASS |
| GATE-07 | s_linker14_voyager docstring updated; DEFAULT_BANK_PATH → cross_split_final_bank.json |
| GATE-08 | Cost audit above; justified by <STRONG/WEAK finding + mechanistic insight> |

## Next Action

Phase 19 — Milestone Close (unconditional). Archive, requirements close-out, PROJECT.md update.
```

---

## Step 9: Update State Files

### STATE.md

Update `.planning/STATE.md`:
- `stopped_at`: "Phase 17 complete. Confirmation verdict = <STRONG/WEAK>. Ready for Phase 19 close."
- `last_activity`: `2026-06-01 -- Phase 17 complete, verdict=<VERDICT>, cross_split_macro_f1=<VALUE>`
- `current_focus`: Phase 19 Milestone Close
- `progress.completed_phases`: increment by 1 (Phase 17 done)
- Phase 17 Result block (mirroring Phase 15/16 blocks): key numbers + bank path + verdict

Update flow diagram to mark Phase 17 done:
```
[Phase 14 ✅]──▶[Phase 15 ✅]──▶[Phase 16 ✅]──▶[Phase 17 ✅]──▶[Phase 19]
```

### ROADMAP.md

In Progress Table (`v2.3 Phases`):
- Phase 17 row: `N/N | ✅ Complete — <VERDICT> (macro X.XX%) | 2026-06-01`

### v2.3-ROADMAP.md

- Mark Phase 17 checkbox `[x]` with date
- Add Phase 17 actual results to Phase Details section:
  ```
  **Actual results (2026-06-01)**:
    - Splits run: 3 (split1/2/3)
    - Cross-split bank: N patterns, M slots
    - 5-dataset eval (gpt-5.4, cross-split bank): Macro X.XX%
    - Verdict: <STRONG/WEAK>
  ```

---

## Acceptance Criteria

- [ ] `results/voyager_v4_beta/confirmation/cross_split_final_bank.json` exists with 9 slot keys; at least 1 non-empty slot
- [ ] `logs/voyager_v4_beta/eval_confirmation.log` contains 5-dataset per-dataset F1 + macro F1
- [ ] `logs/voyager_v4_beta/eval_gate01_regression.log` shows s_linker13_min macro within 0.01 of 0.9069 (GATE-01 PASS)
- [ ] `17-CONFIRMATION-VERDICT.md` exists with `verdict:` frontmatter field = STRONG or WEAK
- [ ] `17-CONFIRMATION-VERDICT.md` contains full 5-dataset eval table + comparison table + GATE-01/08 sections
- [ ] `s_linker14_voyager.py` DEFAULT_BANK_PATH updated to `cross_split_final_bank.json` path
- [ ] `s_linker14_voyager.py` docstring contains Confirmation Tier section with result
- [ ] ABLATION-TABLE.md contains v2.3 addendum rows with actual Phase 17 numbers
- [ ] `STATE.md` updated: Phase 17 complete, Phase 19 as next action
- [ ] `ROADMAP.md` Phase 17 row marked complete with date
- [ ] `v2.3-ROADMAP.md` Phase 17 checkbox `[x]` with actual results
