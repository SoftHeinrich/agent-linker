---
phase: 17-confirmation-tier
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - logs/voyager_v4_beta/confirmation_split1.log
  - logs/voyager_v4_beta/confirmation_split2.log
  - logs/voyager_v4_beta/confirmation_split3.log
  - logs/voyager_v4_beta/eval_split1.log
  - logs/voyager_v4_beta/eval_split2.log
  - logs/voyager_v4_beta/eval_split3.log
  - results/voyager_v4_beta/split1_replication/range_summary.json
  - results/voyager_v4_beta/split1_replication/final_bank.json
  - results/voyager_v4_beta/split2_bbb_in_train/range_summary.json
  - results/voyager_v4_beta/split2_bbb_in_train/final_bank.json
  - results/voyager_v4_beta/split3_rotated_holdout/range_summary.json
  - results/voyager_v4_beta/split3_rotated_holdout/final_bank.json
autonomous: true
requirements:
  - REQ-V23-07
  - REQ-V23-13
  - REQ-V23-14
  - GATE-06
tags:
  - voyager
  - training-run
  - gpt-5.4
  - confirmation-tier
  - 3-split
user_setup:
  - service: openai
    why: "gpt-5.4 LLM calls for L/O/D roles during β training (3 splits × max 5 passes) + per-split 5-dataset evaluation"
    env_vars:
      - name: OPENAI_API_KEY
        source: ".env file at repo root (already present from Phase 16)"

must_haves:
  truths:
    - "Split 1 range run completes (MS+TS+TM train); range_summary.json and final_bank.json exist at results/voyager_v4_beta/split1_replication/"
    - "Split 2 range run completes (MS+TS+BBB train); range_summary.json and final_bank.json exist at results/voyager_v4_beta/split2_bbb_in_train/"
    - "Split 3 range run completes (TS+TM+JAB train); range_summary.json and final_bank.json exist at results/voyager_v4_beta/split3_rotated_holdout/"
    - "Each per-split final_bank.json aggregates per-project banks for that split (union, dry-run placeholder removed)"
    - "s_linker14_voyager evaluated on all 5 datasets per split (3 evals total); per-dataset F1 logged"
    - "All 3 splits complete without crash; pass summaries and train macros recorded"
  artifacts:
    - path: "results/voyager_v4_beta/split1_replication/range_summary.json"
      provides: "split 1 range result: passes_run, final_train_macro_f1, converged"
      contains: '"split": "split1_replication"'
    - path: "results/voyager_v4_beta/split1_replication/final_bank.json"
      provides: "split 1 aggregated per-project bank (9 slot keys)"
    - path: "results/voyager_v4_beta/split2_bbb_in_train/range_summary.json"
      provides: "split 2 range result"
      contains: '"split": "split2_bbb_in_train"'
    - path: "results/voyager_v4_beta/split2_bbb_in_train/final_bank.json"
      provides: "split 2 aggregated per-project bank"
    - path: "results/voyager_v4_beta/split3_rotated_holdout/range_summary.json"
      provides: "split 3 range result"
      contains: '"split": "split3_rotated_holdout"'
    - path: "results/voyager_v4_beta/split3_rotated_holdout/final_bank.json"
      provides: "split 3 aggregated per-project bank"
    - path: "logs/voyager_v4_beta/eval_split1.log"
      provides: "per-dataset F1 for all 5 datasets using split 1 bank"
    - path: "logs/voyager_v4_beta/eval_split2.log"
      provides: "per-dataset F1 for all 5 datasets using split 2 bank"
    - path: "logs/voyager_v4_beta/eval_split3.log"
      provides: "per-dataset F1 for all 5 datasets using split 3 bank"
  key_links:
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_range"
      to: "results/voyager_v4_beta/split1_replication/range_summary.json"
      via: "--split split1_replication --projects mediastore,teastore,teammates"
      pattern: "fresh start (no existing banks in split1 dir) → range to convergence"
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_range"
      to: "results/voyager_v4_beta/split2_bbb_in_train/range_summary.json"
      via: "--split split2_bbb_in_train --projects mediastore,teastore,bigbluebutton"
      pattern: "fresh start → range to convergence"
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_range"
      to: "results/voyager_v4_beta/split3_rotated_holdout/range_summary.json"
      via: "--split split3_rotated_holdout --projects teastore,teammates,jabref"
      pattern: "fresh start → range to convergence"

---

# Plan 17-P1: 3-Split Confirmation Runs + Per-Split Banks + Per-Split Eval

## Goal

Run β training Range tier on all 3 Voyager v2 splits (fresh start each), aggregate per-project banks into per-split `final_bank.json` files, and evaluate `s_linker14_voyager` on all 5 datasets per split. Produces 3 range summaries + 3 final banks + 3 eval logs consumed by Plan 17-P2 for cross-split aggregation and final verdict.

## Split Assignments (Voyager v2 convention)

| Split | Name | Train Projects | Test Projects |
|-------|------|---------------|---------------|
| 1 | split1_replication | mediastore, teastore, teammates | bigbluebutton, jabref |
| 2 | split2_bbb_in_train | mediastore, teastore, bigbluebutton | teammates, jabref |
| 3 | split3_rotated_holdout | teastore, teammates, jabref | mediastore, bigbluebutton |

Split 1 = same train set as mainline (Phase 15/16). Starts fresh (no warm-start from mainline banks — clean science).

## Pre-flight Check

```bash
cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45
# Confirm .env and API key present
grep OPENAI_API_KEY .env | wc -c

# Confirm mainline results not accidentally overwritten
ls results/voyager_v4_beta/mainline/final_bank.json

# Confirm split dirs don't exist yet (fresh start guaranteed)
ls results/voyager_v4_beta/ 2>/dev/null
```

Expected: `mainline/` directory present, no `split1_replication/` etc yet.

---

## Split 1: MS + TS + TM (train) | BBB + JAB (test)

### Step 1.1: Run Range — Split 1

```bash
mkdir -p logs/voyager_v4_beta
python scripts/voyager_train_tlr_v4_beta.py range \
    --projects mediastore,teastore,teammates \
    --backend openai \
    --model gpt-5.4 \
    --split split1_replication \
    2>&1 | tee logs/voyager_v4_beta/confirmation_split1.log
```

Expected: fresh-start banks (no prior split1 banks). Runs max 5 passes; convergence at macro ≥ 0.90 or pass 5 cap. GATE-06 advisory warnings for project names are non-blocking.

Verify:
```bash
python -c "
import json
s = json.load(open('results/voyager_v4_beta/split1_replication/range_summary.json'))
print(f'Split 1: passes_run={s[\"passes_run\"]} converged={s[\"converged\"]} final_macro={s[\"final_train_macro_f1\"]:.4f}')
"
```

### Step 1.2: Aggregate Per-Project Banks → split1 final_bank.json

```python
import json
from pathlib import Path

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

split_dir = Path("results/voyager_v4_beta/split1_replication")
projects = ["mediastore", "teastore", "teammates"]

final_bank = {slot: [] for slot in AXIOM_SLOTS}
seen_ids = set()

for project in projects:
    bank = json.loads((split_dir / f"{project}_bank.json").read_text())
    for slot in AXIOM_SLOTS:
        for pattern in bank.get(slot, []):
            pid = pattern.get("pattern_id", "")
            rule = pattern.get("rule_text", pattern.get("content", ""))
            if pid in seen_ids:
                continue
            if "DRY_RUN" in rule.upper() or "PLACEHOLDER" in rule.upper() or "MOCK" in rule.upper():
                print(f"  Skip dry-run placeholder: {pid}")
                continue
            seen_ids.add(pid)
            final_bank[slot].append(pattern)

total = sum(len(v) for v in final_bank.values())
slots_used = [s for s in AXIOM_SLOTS if final_bank[s]]
print(f"Split 1 final_bank: {total} patterns in {len(slots_used)} slots")
(split_dir / "final_bank.json").write_text(json.dumps(final_bank, indent=2))
print("Written: results/voyager_v4_beta/split1_replication/final_bank.json")
```

Run as: `python -c "$(cat <<'PYEOF' ... PYEOF)"` or save to `/tmp/agg_split1.py` and run.

### Step 1.3: Evaluate Split 1 Bank on All 5 Datasets

```bash
VOYAGER4B_BANK_PATH=results/voyager_v4_beta/split1_replication/final_bank.json \
python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_split1.log
```

Record macro F1 from stdout.

---

## Split 2: MS + TS + BBB (train) | TM + JAB (test)

### Step 2.1: Run Range — Split 2

```bash
python scripts/voyager_train_tlr_v4_beta.py range \
    --projects mediastore,teastore,bigbluebutton \
    --backend openai \
    --model gpt-5.4 \
    --split split2_bbb_in_train \
    2>&1 | tee logs/voyager_v4_beta/confirmation_split2.log
```

Verify:
```bash
python -c "
import json
s = json.load(open('results/voyager_v4_beta/split2_bbb_in_train/range_summary.json'))
print(f'Split 2: passes_run={s[\"passes_run\"]} converged={s[\"converged\"]} final_macro={s[\"final_train_macro_f1\"]:.4f}')
"
```

### Step 2.2: Aggregate Per-Project Banks → split2 final_bank.json

Same aggregation script as Step 1.2, substituting:
- `split_dir = Path("results/voyager_v4_beta/split2_bbb_in_train")`
- `projects = ["mediastore", "teastore", "bigbluebutton"]`
- Output: `results/voyager_v4_beta/split2_bbb_in_train/final_bank.json`

### Step 2.3: Evaluate Split 2 Bank on All 5 Datasets

```bash
VOYAGER4B_BANK_PATH=results/voyager_v4_beta/split2_bbb_in_train/final_bank.json \
python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_split2.log
```

---

## Split 3: TS + TM + JAB (train) | MS + BBB (test)

### Step 3.1: Run Range — Split 3

```bash
python scripts/voyager_train_tlr_v4_beta.py range \
    --projects teastore,teammates,jabref \
    --backend openai \
    --model gpt-5.4 \
    --split split3_rotated_holdout \
    2>&1 | tee logs/voyager_v4_beta/confirmation_split3.log
```

Verify:
```bash
python -c "
import json
s = json.load(open('results/voyager_v4_beta/split3_rotated_holdout/range_summary.json'))
print(f'Split 3: passes_run={s[\"passes_run\"]} converged={s[\"converged\"]} final_macro={s[\"final_train_macro_f1\"]:.4f}')
"
```

### Step 3.2: Aggregate Per-Project Banks → split3 final_bank.json

Same aggregation script, substituting:
- `split_dir = Path("results/voyager_v4_beta/split3_rotated_holdout")`
- `projects = ["teastore", "teammates", "jabref"]`
- Output: `results/voyager_v4_beta/split3_rotated_holdout/final_bank.json`

### Step 3.3: Evaluate Split 3 Bank on All 5 Datasets

```bash
VOYAGER4B_BANK_PATH=results/voyager_v4_beta/split3_rotated_holdout/final_bank.json \
python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_split3.log
```

---

## Step 4: Summarize Per-Split Results

After all 3 splits complete, collect per-split summary:

```bash
python -c "
import json

splits = [
    ('split1_replication', ['mediastore','teastore','teammates'], ['bigbluebutton','jabref']),
    ('split2_bbb_in_train', ['mediastore','teastore','bigbluebutton'], ['teammates','jabref']),
    ('split3_rotated_holdout', ['teastore','teammates','jabref'], ['mediastore','bigbluebutton']),
]

print('Split | Train | Test | Passes | Converged | Train Macro | Bank Patterns')
print('------|-------|------|--------|-----------|-------------|---------------')
for name, train, test in splits:
    s = json.load(open(f'results/voyager_v4_beta/{name}/range_summary.json'))
    b = json.load(open(f'results/voyager_v4_beta/{name}/final_bank.json'))
    n_patterns = sum(len(v) for v in b.values() if isinstance(v, list))
    print(f'{name} | {train} | {test} | {s[\"passes_run\"]} | {s[\"converged\"]} | {s[\"final_train_macro_f1\"]:.4f} | {n_patterns}')
"
```

Record this table — consumed by Plan 17-P2.

---

## Cost Note

Phase 16 (mainline, 5 passes) cost ~$35–51. Phase 17 runs 3 splits × up to 5 passes. Estimate:
- Each split converges in 1–3 passes (pattern bank may saturate faster with accumulated context): ~$10–17/split
- 3 evals × 5 datasets each: ~$10–15 total
- **Estimated total Phase 17-P1**: ~$40–65 (within $40–60 target; monitor closely)

If any single split hits $20 before convergence, log the anomaly and continue (budget cap is for full phase, not per-split).

---

## Acceptance Criteria

- [ ] `results/voyager_v4_beta/split1_replication/range_summary.json` exists with `"split": "split1_replication"` and numeric `final_train_macro_f1`
- [ ] `results/voyager_v4_beta/split1_replication/final_bank.json` exists with 9 slot keys
- [ ] `results/voyager_v4_beta/split2_bbb_in_train/range_summary.json` exists with `"split": "split2_bbb_in_train"`
- [ ] `results/voyager_v4_beta/split2_bbb_in_train/final_bank.json` exists with 9 slot keys
- [ ] `results/voyager_v4_beta/split3_rotated_holdout/range_summary.json` exists with `"split": "split3_rotated_holdout"`
- [ ] `results/voyager_v4_beta/split3_rotated_holdout/final_bank.json` exists with 9 slot keys
- [ ] `logs/voyager_v4_beta/eval_split1.log`, `eval_split2.log`, `eval_split3.log` each contain per-dataset F1 for all 5 datasets
- [ ] No split crashes mid-run; all 3 range_summary.json files have valid `passes_run` and `final_train_macro_f1`
