---
phase: 16-range-tier
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - logs/voyager_v4_beta/range.log
  - results/voyager_v4_beta/mainline/range_summary.json
  - results/voyager_v4_beta/mainline/pass1_summary.json
  - results/voyager_v4_beta/mainline/pass2_summary.json
  - results/voyager_v4_beta/mainline/final_bank.json
  - results/voyager_v4_beta/mainline/mediastore_bank.json
  - results/voyager_v4_beta/mainline/teastore_bank.json
  - results/voyager_v4_beta/mainline/teammates_bank.json
autonomous: true
requirements:
  - REQ-V23-07
  - REQ-V23-13
  - REQ-V23-14
  - REQ-V23-15
  - GATE-06
tags:
  - voyager
  - training-run
  - gpt-5.4
  - range-tier
  - evaluation
user_setup:
  - service: openai
    why: "gpt-5.4 LLM calls for L/O/D roles during β training range + 5-dataset evaluation"
    env_vars:
      - name: OPENAI_API_KEY
        source: ".env file at repo root (already present per Phase 15 execution)"

must_haves:
  truths:
    - "Range harness runs end-to-end on mediastore, teastore, teammates to convergence or pass 5 cap without crash"
    - "range_summary.json exists at results/voyager_v4_beta/mainline/ with converged and final_train_macro_f1 fields"
    - "Per-project bank files updated: mediastore_bank.json, teastore_bank.json, teammates_bank.json"
    - "final_bank.json created at results/voyager_v4_beta/mainline/final_bank.json with union of all per-project patterns"
    - "s_linker14_voyager evaluated on all 5 datasets (mediastore, teastore, teammates, bigbluebutton, jabref) via run_ablation.py"
    - "Per-dataset F1 and 5-dataset macro F1 recorded in range.log or a separate eval output file"
    - "Axiom-only comparison F1 recorded (empty-bank run or reference to Phase 14 dry-run results)"
    - "Token usage estimate logged to range.log"
  artifacts:
    - path: "logs/voyager_v4_beta/range.log"
      provides: "stdout capture of full range run: per-pass F1, convergence, GATE-06, cost estimate"
    - path: "results/voyager_v4_beta/mainline/range_summary.json"
      provides: "range tier result: passes_run, final_train_macro_f1, converged, pass_summaries"
      contains: '"tier": "range"'
    - path: "results/voyager_v4_beta/mainline/final_bank.json"
      provides: "aggregated slot-uniform bank for s_linker14_voyager evaluation"
    - path: "results/voyager_v4_beta/mainline/mediastore_bank.json"
      provides: "MS per-project trained bank after range convergence"
    - path: "results/voyager_v4_beta/mainline/teastore_bank.json"
      provides: "TS per-project trained bank after range convergence"
    - path: "results/voyager_v4_beta/mainline/teammates_bank.json"
      provides: "TM per-project trained bank after range convergence"
  key_links:
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_range"
      to: "results/voyager_v4_beta/mainline/range_summary.json"
      via: "up to MAX_OUTER_PASSES passes, warm-start from existing probe banks"
      pattern: "_load_bank → run_outer_pass → _save_bank"
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_range"
      to: "results/voyager_v4_beta/mainline/{project}_bank.json"
      via: "per-project bank updated on each committed pass"
      pattern: "_save_bank"
    - from: "run_ablation.py::run_variant"
      to: "src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py::SLinker14Voyager"
      via: "s_linker14_voyager variant with final_bank.json as default bank_path"
      pattern: "SLinker14Voyager(backend=backend, model=model)"

---

# Plan 16-P1: Range Tier Run + final_bank Aggregation + 5-Dataset Evaluation

## Goal

Execute the β training Range tier on the mainline split (MS+TS+TM) to convergence, aggregate per-project banks into `final_bank.json`, then evaluate `s_linker14_voyager` on all 5 datasets (gpt-5.4) to produce the 5-dataset macro F1 needed for the 3-tier verdict in Plan 16-P2.

## Pre-flight Check

Before running, verify:
- `.env` file at repo root contains `OPENAI_API_KEY` (used by Phase 15 — should still be valid)
- Existing probe banks exist: `results/voyager_v4_beta/mainline/{mediastore,teastore,teammates}_bank.json`
- No stale lock files or partially-written JSONs from Phase 15 (check `range_summary.json` doesn't already exist)

```bash
cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45
ls results/voyager_v4_beta/mainline/
python -c "import json; b=json.load(open('results/voyager_v4_beta/mainline/mediastore_bank.json')); print('MS bank patterns:', sum(len(v) for v in b.values() if isinstance(v, list)))"
```

## Step 1: Run Range Tier

```bash
mkdir -p logs/voyager_v4_beta
python scripts/voyager_train_tlr_v4_beta.py range \
    --projects mediastore,teastore,teammates \
    --backend openai \
    --model gpt-5.4 \
    2>&1 | tee logs/voyager_v4_beta/range.log
```

Expected behavior:
- Pass 1: loads probe banks → runs L (expect high F1 ~0.91+) → O (fresh failure modes) → D (may propose new patterns or remove dry-run placeholders) → GATE-06 → P
- Convergence: when D proposes 0 accepted patterns AND 0 removals
- If probe D had nothing left after pass 1, range also converges at pass 1 (fast path)
- GATE-06 advisory taboo warnings for project names expected (non-blocking per Phase 15)

Verify success:
```bash
python -c "
import json
s = json.load(open('results/voyager_v4_beta/mainline/range_summary.json'))
print(f'passes_run={s[\"passes_run\"]} converged={s[\"converged\"]} final_macro={s[\"final_train_macro_f1\"]:.4f}')
"
```

## Step 2: Aggregate final_bank.json

Merge all per-project banks into a single `final_bank.json`. Union by pattern_id (keep first occurrence if duplicate IDs).

```python
import json
from pathlib import Path

split_dir = Path("results/voyager_v4_beta/mainline")
projects = ["mediastore", "teastore", "teammates"]

# 9 axiom slot keys (must match s_linker14_voyager.py AXIOM_SLOTS)
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

final_bank = {slot: [] for slot in AXIOM_SLOTS}
seen_ids = set()

for project in projects:
    bank_path = split_dir / f"{project}_bank.json"
    if not bank_path.exists():
        print(f"WARNING: {bank_path} missing")
        continue
    bank = json.loads(bank_path.read_text())
    for slot in AXIOM_SLOTS:
        for pattern in bank.get(slot, []):
            pid = pattern.get("pattern_id", str(pattern))
            if pid not in seen_ids:
                seen_ids.add(pid)
                # Skip dry-run placeholders (content == "DRY_RUN_PLACEHOLDER" or similar)
                content = pattern.get("content", "")
                if "DRY_RUN" in content.upper() or "PLACEHOLDER" in content.upper():
                    print(f"  Skipping dry-run placeholder: {pid}")
                    continue
                final_bank[slot].append(pattern)

total = sum(len(v) for v in final_bank.values())
print(f"final_bank.json: {total} patterns across {len([s for s in AXIOM_SLOTS if final_bank[s]])} non-empty slots")
for slot in AXIOM_SLOTS:
    if final_bank[slot]:
        print(f"  {slot}: {len(final_bank[slot])} pattern(s)")

(split_dir / "final_bank.json").write_text(json.dumps(final_bank, indent=2))
print("Written: results/voyager_v4_beta/mainline/final_bank.json")
```

Run as:
```bash
python -c "$(cat <<'PYEOF'
<paste inline python above>
PYEOF
)"
```
Or save as a temporary script and run it. Verify final_bank.json is valid JSON with the 9 slot keys.

## Step 3: Evaluate s_linker14_voyager on All 5 Datasets

Run evaluation using the aggregated bank:

```bash
python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_range.log
```

This uses `final_bank.json` at the default path (`results/voyager_v4_beta/mainline/final_bank.json`).

Record per-dataset F1 and macro F1 from stdout. Format expected:
```
mediastore        F1=X.XXXX  P=X.XXXX  R=X.XXXX
teastore          F1=X.XXXX  ...
teammates         F1=X.XXXX  ...
bigbluebutton     F1=X.XXXX  ...
jabref            F1=X.XXXX  ...
MACRO             F1=X.XXXX
```

## Step 4: Axiom-Only Comparison (REQ-V23-15)

Run `s_linker14_voyager` in empty-bank mode to get the axiom-only floor:

```bash
VOYAGER4B_BANK_PATH=/dev/null python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_axiom_only.log
```

Record axiom-only per-dataset F1 and macro. This is the "prompts_v3_axiom floor" for REQ-V23-15.

Note: If `/dev/null` as bank path causes a parse error, create an empty bank JSON:
```bash
python -c "
import json
SLOTS = ['DOC_KNOWLEDGE_EXTRACTION_RULES','DOC_KNOWLEDGE_JUDGE_RULES','AMBIGUITY_RULES',
         'VALIDATION_RULES','SYNONYM_INJECTION_RULES','COREF_RULES',
         'BOUNDARY_FILTER_RULES','PARTIAL_MATCH_RULES','LINK_JUDGE_RULES']
json.dump({s:[] for s in SLOTS}, open('/tmp/empty_bank.json','w'), indent=2)
print('empty bank written')
"
VOYAGER4B_BANK_PATH=/tmp/empty_bank.json python run_ablation.py \
    --variants s_linker14_voyager \
    --datasets mediastore teastore teammates bigbluebutton jabref \
    2>&1 | tee logs/voyager_v4_beta/eval_axiom_only.log
```

## Step 5: Record Summary in Log

Append a cost summary to `range.log`:

```bash
echo "
=== RANGE TIER COST ESTIMATE ===" >> logs/voyager_v4_beta/range.log
python -c "
import json
s = json.load(open('results/voyager_v4_beta/mainline/range_summary.json'))
passes = s['passes_run']
projects = len(s['projects'])
# Estimate: per pass = 3L + 3O + 3D + 1D_merge + 3P = ~13 calls @ ~\$0.50-0.70/call
calls_per_pass = projects * 4  # L+O+D+P per project, D shared = approx
total_calls = calls_per_pass * passes
print(f'Passes run: {passes}')
print(f'Estimated LLM calls: ~{total_calls} (training)')
print(f'Estimated cost (training): ~\${total_calls * 0.60:.0f}-{total_calls * 0.80:.0f}')
print(f'Estimated evaluation cost: ~\$5-10 (5 datasets x s_linker14_voyager)')
" >> logs/voyager_v4_beta/range.log
```

## Acceptance Criteria

- [ ] `results/voyager_v4_beta/mainline/range_summary.json` exists with `"tier": "range"` and numeric `final_train_macro_f1`
- [ ] `results/voyager_v4_beta/mainline/final_bank.json` exists with 9 slot keys (non-empty for at least 3 slots)
- [ ] `logs/voyager_v4_beta/eval_range.log` contains per-dataset F1 for all 5 datasets
- [ ] `logs/voyager_v4_beta/eval_axiom_only.log` contains axiom-only per-dataset F1 for all 5 datasets
- [ ] 5-dataset macro F1 recorded (primary metric for Phase 16-P2 verdict)
- [ ] No crash mid-run; range converged or hit pass 5 cap
