---
phase: 27-probe-tier
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - logs/voyager_v4_beta/probe_p27.log
  - results/voyager_v4b_v25/mainline/probe_summary.json
  - results/voyager_v4b_v25/mainline/pass1_summary.json
  - results/voyager_v4b_v25/mainline/pass2_summary.json
  - results/voyager_v4b_v25/mainline/mediastore_bank.json
  - results/voyager_v4b_v25/mainline/teastore_bank.json
  - results/voyager_v4b_v25/mainline/teammates_bank.json
  - .planning/phases/27-probe-tier/27-PROBE-VERDICT.md
autonomous: true
requirements:
  - REQ-V25-09
  - GATE-06
  - GATE-08
tags:
  - voyager
  - training-run
  - gpt-5.4
  - probe-tier
  - v2.5
---

<objective>
Run the β probe tier with clean v2.5 infrastructure (oracle cache fix + 15-slot bank) on the mainline training split (mediastore, teastore, teammates) using gpt-5.4 via OpenAI backend, for up to 2 outer passes, producing the per-project trained banks and the `probe_summary.json` verdict artifact required by REQ-V25-09.

Output:
- logs/voyager_v4_beta/probe_p27.log — full stdout capture
- results/voyager_v4b_v25/mainline/probe_summary.json — verdict JSON
- results/voyager_v4b_v25/mainline/pass{1,2}_summary.json — per-pass evidence
- results/voyager_v4b_v25/mainline/{mediastore,teastore,teammates}_bank.json — trained banks
- .planning/phases/27-probe-tier/27-PROBE-VERDICT.md — human-readable verdict
</objective>

<context>
v2.5 changes active in `voyager_train_tlr_v4_beta.py`:
- REQ-V25-01: Oracle cache key includes `bank_content_hash` (line 461–463)
- REQ-V25-02: `MIN_COMMIT_DELTA = 0.005` — skip O+D if delta < 0.005
- REQ-V25-03: D prompt includes `{underfilled_slots}` steering toward zero-pattern slots

v2.5 changes active in `s_linker14_voyager.py`:
- REQ-V25-04: 15 slot constants (SLOT_NAMES has 15 entries)
- REQ-V25-05: ILinker3Injected subclass wires SEED_EXTRACTION_RULES + SEED_ACTOR_RULES
- REQ-V25-06: 4 inline prompts replaced by bank-slot injection

Output dir: results/voyager_v4b_v25 (VOYAGER4B_OUT_ROOT env var) — avoids overwriting v2.4 results.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Pre-flight and run probe</name>
  <action>
    Command:
    VOYAGER4B_OUT_ROOT=results/voyager_v4b_v25 python scripts/voyager_train_tlr_v4_beta.py probe \
        --projects mediastore,teastore,teammates \
        --backend openai \
        --model gpt-5.4\
        2>&1 | tee logs/voyager_v4_beta/probe_p27.log
  </action>
  <acceptance_criteria>
    - probe_summary.json exists with "verdict" field
    - pass1_summary.json and pass2_summary.json exist
    - 3 bank JSON files exist
  </acceptance_criteria>
</task>

<task type="auto">
  <name>Task 2: Write PROBE-VERDICT.md</name>
  <action>
    Parse probe_summary.json and pass summaries. Write 27-PROBE-VERDICT.md with:
    - Pass 1 and Pass 2 per-project F1 + macro F1
    - Gate A / Gate B fired confirmation (evidence from log)
    - New slots with committed patterns (SEED_* / GENERIC_* slots)
    - Oracle cache key verification (bank hash in cache filenames)
    - Verdict: CONTINUE / KILL with numeric evidence
    - Cost estimate from token usage in log
  </action>
</task>

</tasks>
