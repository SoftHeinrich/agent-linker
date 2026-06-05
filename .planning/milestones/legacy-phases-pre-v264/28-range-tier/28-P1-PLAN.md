---
phase: 28-range-tier
plan: 1
type: execute
wave: 1
depends_on: [27-probe-tier]
files_modified:
  - logs/voyager_v4_beta/range_p28.log
  - logs/voyager_v4_beta/eval_range_p28.log
  - results/voyager_v4b_v25/mainline/range_summary.json
  - results/voyager_v4b_v25/mainline/final_bank.json
  - results/voyager_v4b_v25/mainline/mediastore_bank.json
  - results/voyager_v4b_v25/mainline/teastore_bank.json
  - results/voyager_v4b_v25/mainline/teammates_bank.json
  - .planning/phases/28-range-tier/28-RANGE-VERDICT.md
autonomous: true
requirements:
  - REQ-V25-10
  - GATE-06
  - GATE-08
tags:
  - voyager
  - training-run
  - gpt-5.4
  - range-tier
  - v2.5
---

<objective>
Run the β Range tier with clean v2.5 infrastructure (continuing from probe bank with 12 patterns across 8 slots) on the mainline training split, up to 5 passes or convergence. Then aggregate final_bank.json and evaluate on all 5 datasets.

Outputs:
- logs/voyager_v4_beta/range_p28.log — training stdout
- logs/voyager_v4_beta/eval_range_p28.log — 5-dataset eval stdout
- results/voyager_v4b_v25/mainline/range_summary.json
- results/voyager_v4b_v25/mainline/final_bank.json
- .planning/phases/28-range-tier/28-RANGE-VERDICT.md
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Range training run</name>
  <action>
    VOYAGER4B_OUT_ROOT=results/voyager_v4b_v25 python scripts/voyager_train_tlr_v4_beta.py range \
        --projects mediastore,teastore,teammates \
        --backend openai --model gpt-5.4 \
        2>&1 | tee logs/voyager_v4_beta/range_p28.log
  </action>
</task>

<task type="auto">
  <name>Task 2: Aggregate final_bank.json (15-slot union)</name>
  <action>
    Python inline: union all 3 per-project banks by pattern_id, using all 15 SLOT_NAMES.
    Write to results/voyager_v4b_v25/mainline/final_bank.json.
  </action>
</task>

<task type="auto">
  <name>Task 3: 5-dataset evaluation</name>
  <action>
    VOYAGER4B_BANK_PATH=results/voyager_v4b_v25/mainline/final_bank.json \
    python run_ablation.py --variants s_linker14_voyager \
        --datasets mediastore teastore teammates bigbluebutton jabref \
        --backend openai --model gpt-5.4 \
        2>&1 | tee logs/voyager_v4_beta/eval_range_p28.log
  </action>
</task>

<task type="auto">
  <name>Task 4: Write 28-RANGE-VERDICT.md</name>
  <action>
    Parse range_summary.json, eval output. Write verdict with per-pass training table,
    5-dataset F1 table, axiom-only comparison, lift vs v2.4 baseline (87.6%),
    and 3-tier verdict (STRONG ≥0.9173 / WEAK [0.87,0.9173) / FAIL <0.87).
  </action>
</task>

</tasks>
