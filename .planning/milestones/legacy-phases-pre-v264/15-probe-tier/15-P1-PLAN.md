---
phase: 15-probe-tier
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - logs/voyager_v4_beta/probe.log
  - results/voyager_v4_beta/mainline/probe_summary.json
  - results/voyager_v4_beta/mainline/pass1_summary.json
  - results/voyager_v4_beta/mainline/pass2_summary.json
  - results/voyager_v4_beta/mainline/mediastore_bank.json
  - results/voyager_v4_beta/mainline/teastore_bank.json
  - results/voyager_v4_beta/mainline/teammates_bank.json
autonomous: true
requirements:
  - REQ-V23-07
  - REQ-V23-13
  - REQ-V23-14
tags:
  - voyager
  - training-run
  - gpt-5.4
  - probe-tier
user_setup:
  - service: openai
    why: "gpt-5.4 LLM calls for L/O/D roles during β training probe"
    env_vars:
      - name: OPENAI_API_KEY
        source: ".env file at repo root (already present per RESEARCH.md)"

must_haves:
  truths:
    - "Probe harness runs end-to-end on mediastore, teastore, teammates without crash"
    - "Pass 1 macro F1 computed and persisted to pass1_summary.json"
    - "Pass 2 macro F1 computed (if pass-1 macro >= 0.80) and persisted to pass2_summary.json"
    - "probe_summary.json contains verdict field equal to 'CONTINUE' or 'KILL'"
    - "Per-project _bank.json file exists for all 3 train projects after final pass"
    - "Token usage from LLMClient is logged to probe.log"
  artifacts:
    - path: "logs/voyager_v4_beta/probe.log"
      provides: "stdout capture of full probe run incl. per-pass F1, verdict, token usage"
    - path: "results/voyager_v4_beta/mainline/probe_summary.json"
      provides: "top-level verdict file with final_train_macro_f1 and pass_summaries"
      contains: '"verdict"'
    - path: "results/voyager_v4_beta/mainline/pass1_summary.json"
      provides: "pass-1 per-project F1s + macro + committed flag"
    - path: "results/voyager_v4_beta/mainline/mediastore_bank.json"
      provides: "MS per-project trained bank (slot-uniform 9 slots)"
    - path: "results/voyager_v4_beta/mainline/teastore_bank.json"
      provides: "TS per-project trained bank"
    - path: "results/voyager_v4_beta/mainline/teammates_bank.json"
      provides: "TM per-project trained bank"
  key_links:
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_probe"
      to: "s_linker14_voyager.SLinker14Voyager"
      via: "L role per-project linker run with current bank state"
      pattern: "_run_linker_l"
    - from: "scripts/voyager_train_tlr_v4_beta.py::run_outer_pass"
      to: "results/voyager_v4_beta/mainline/{project}_bank.json"
      via: "_save_bank() after probation commit/rollback"
      pattern: "_save_bank"
    - from: "LLMClient.get_session_usage()"
      to: "logs/voyager_v4_beta/probe.log"
      via: "explicit print of token totals after run_probe completes"
      pattern: "get_session_usage"
---

<objective>
Run the β probe tier on the mainline training split (mediastore, teastore, teammates) using gpt-5.4 via the OpenAI backend, for up to 2 outer passes, producing the per-project trained banks and the `probe_summary.json` verdict artifact required by REQ-V23-07 / REQ-V23-13 / REQ-V23-14.

Purpose: Phase 15 is purely operational — the harness was shipped in Phase 14 and dry-run verified. This task invokes it against real LLM calls so Plan 2 can convert the verdict JSON into a human-readable verdict document and STATE.md update.

Output:
- logs/voyager_v4_beta/probe.log — full stdout capture incl. token usage
- results/voyager_v4_beta/mainline/probe_summary.json — verdict JSON
- results/voyager_v4_beta/mainline/pass{1,2}_summary.json — per-pass F1 evidence
- results/voyager_v4_beta/mainline/{mediastore,teastore,teammates}_bank.json — trained banks
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/15-probe-tier/15-CONTEXT.md
@.planning/phases/15-probe-tier/15-RESEARCH.md
@scripts/voyager_train_tlr_v4_beta.py

<interfaces>
<!-- Key entry points the executor needs from the harness. Already implemented in Phase 14. -->
<!-- Do NOT modify these files — only invoke via CLI. -->

CLI entry point (scripts/voyager_train_tlr_v4_beta.py):
```
python scripts/voyager_train_tlr_v4_beta.py probe \
    --projects mediastore,teastore,teammates \
    --backend openai \
    --model gpt-5.4
```

Subparser args (verified from main(), lines 1025-1040):
- --projects: comma-separated list, default = MAINLINE_TRAIN = ["mediastore","teastore","teammates"]
- --backend: choices ["openai","claude"], default "openai"
- --model: default "gpt-5.4"
- --dry-run: action="store_true" (do NOT pass — this is a real run)

Output locations (constants in harness):
- OUT_ROOT = Path(os.environ.get("VOYAGER4B_OUT_ROOT", "results/voyager_v4_beta"))
- split_dir = OUT_ROOT / "mainline"
- CHEAP_KILL_THRESHOLD = 0.87 (applied after pass 2 inside run_probe)
- MAINLINE_TRAIN = ["mediastore","teastore","teammates"]

Behaviour (verified from run_probe, lines 895-952):
- Iterates pass_num in range(1, 3) — always runs both passes UNLESS pass 2 cheap-kills
- After pass 2: if committed_macro_f1 < 0.87 prints CHEAP-KILL line and breaks
- Final verdict = "CONTINUE" if final_macro >= 0.87 else "KILL"
- probe_summary.json written deterministically at end
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Pre-flight checks and log directory creation</name>
  <files>logs/voyager_v4_beta/</files>
  <read_first>
    - .planning/phases/15-probe-tier/15-RESEARCH.md (Pitfall 1: missing log directory; Environment Availability table)
    - .planning/phases/15-probe-tier/15-CONTEXT.md (logged path locked at logs/voyager_v4_beta/probe.log)
  </read_first>
  <action>
    Perform pre-flight checks for the probe run (per RESEARCH.md Pitfall 1 — harness does NOT auto-create logs/ subdirs):

    1. Verify OPENAI_API_KEY is set in environment OR present in .env at repo root:
       `grep -q '^OPENAI_API_KEY=' .env && echo OK_ENVFILE || ([ -n "$OPENAI_API_KEY" ] && echo OK_ENV || echo MISSING)`
       If MISSING → STOP and surface error. Do not proceed.

    2. Confirm the β harness script exists and is executable:
       `test -f scripts/voyager_train_tlr_v4_beta.py && echo OK || echo MISSING`

    3. Confirm s_linker14_voyager is importable (Phase 14 deliverable, frozen):
       `python -c "from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager; print('OK')"`

    4. Create the log target directory (harness does NOT create it):
       `mkdir -p logs/voyager_v4_beta`

    5. Ensure the results split directory parent exists (harness creates split_dir but parent must exist):
       `mkdir -p results/voyager_v4_beta`

    6. Sanity check disk space (probe writes ~5-10MB of JSON; cache ~50MB):
       `df -h . | tail -1`

    Do NOT pre-create logs/voyager_v4_beta/probe.log — the shell redirect in Task 2 will create it.
    Do NOT modify any frozen artifact listed in `<frozen>` (s_linker13.py, prompts_v2.py, ilinker*.py,
    data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py, s_linker13_min.py, s_linker14_voyager.py).
  </action>
  <verify>
    <automated>test -d logs/voyager_v4_beta &amp;&amp; test -f scripts/voyager_train_tlr_v4_beta.py &amp;&amp; python -c "from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager"</automated>
  </verify>
  <acceptance_criteria>
    - `logs/voyager_v4_beta/` directory exists (verifiable via `test -d logs/voyager_v4_beta`)
    - `scripts/voyager_train_tlr_v4_beta.py` exists (verifiable via `test -f scripts/voyager_train_tlr_v4_beta.py`)
    - `from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager` succeeds (exit 0)
    - OPENAI_API_KEY is sourceable from `.env` or environment (no MISSING)
    - No files in `<frozen>` list were modified during preflight (verifiable via `git diff --name-only` showing no entries in that list)
  </acceptance_criteria>
  <done>
    Log dir exists, harness script present, linker importable, OpenAI key available.
    Ready to invoke the probe.
  </done>
</task>

<task type="auto">
  <name>Task 2: Run β probe harness on mainline train split (gpt-5.4)</name>
  <files>
    logs/voyager_v4_beta/probe.log,
    results/voyager_v4_beta/mainline/probe_summary.json,
    results/voyager_v4_beta/mainline/pass1_summary.json,
    results/voyager_v4_beta/mainline/pass2_summary.json,
    results/voyager_v4_beta/mainline/mediastore_bank.json,
    results/voyager_v4_beta/mainline/teastore_bank.json,
    results/voyager_v4_beta/mainline/teammates_bank.json
  </files>
  <read_first>
    - .planning/phases/15-probe-tier/15-RESEARCH.md (Architecture Patterns / Probe Tier Data Flow, Pitfalls 2-5)
    - scripts/voyager_train_tlr_v4_beta.py lines 895-952 (run_probe behaviour, cheap-kill at pass 2)
    - .planning/phases/15-probe-tier/15-CONTEXT.md (3 train projects locked; budget $5-10)
  </read_first>
  <action>
    Invoke the Phase 14 β harness in probe mode against the mainline train split. The harness drives the
    full L+O+D+P loop for up to 2 outer passes, applies GATE-06 taboo grep + advisory reviewer_critic_stub
    per RESEARCH.md (advisory mode kept — do NOT upgrade the stub), and writes verdict JSON deterministically.

    EXACT COMMAND (run from repo root /mnt/hostshare/ardoco-home/llm-sad-sam-v45):

    ```bash
    python scripts/voyager_train_tlr_v4_beta.py probe \
        --projects mediastore,teastore,teammates \
        --backend openai \
        --model gpt-5.4 \
        2>&1 | tee logs/voyager_v4_beta/probe.log
    ```

    Notes:
    - Use `tee` (not `>`) so stdout is visible in the agent terminal AND captured to probe.log.
    - `2>&1` folds stderr into stdout so any exception traceback also lands in probe.log.
    - DO NOT pass `--dry-run`. Per CONTEXT D-01 / D-02 this is the real gpt-5.4 run.
    - DO NOT split into two invocations (one per pass). The harness loops passes 1 and 2 internally
      and applies the cheap-kill threshold (0.87) after pass 2. Per RESEARCH.md Open Question 2 + Pitfall 2,
      the 0.80 pass-2 trigger from CONTEXT SC#3 is a human-decision sentinel — the harness always runs both
      passes; we observe the outcome rather than pre-empting.
    - DO NOT override VOYAGER4B_OUT_ROOT or VOYAGER4B_CACHE_ROOT. Defaults (results/voyager_v4_beta/{,cache/})
      are locked per REQ-V23-10.
    - DO NOT modify scripts/voyager_train_tlr_v4_beta.py or any frozen artifact during the run.
    - Expected runtime: per RESEARCH.md Pitfall 5, each pass = 3 L runs + 3 O runs + 1 D run + 3 probation L runs
      = ~10 LLM-heavy calls/pass. Two passes ≈ 20 calls. Wall clock typically 30-90 minutes total on gpt-5.4.
    - Run as a long-running command (timeout up to 600000 ms via Bash). If the LLM-call duration exceeds this,
      retry; the per-(text_stem, comp_hash, backend, model) cache at results/voyager_v4_beta/cache/ will skip
      completed O/D calls on rerun (REQ-V23-10).
    - If a single project crashes mid-run (per CONTEXT Claude's Discretion item 3): do NOT manually patch state.
      Capture the traceback in probe.log, then re-invoke the same command — the on-disk cache will replay
      completed work and the partial bank files will resume from the last committed pass.

    After the harness exits:
    1. Verify probe_summary.json exists and contains a `verdict` field:
       `python -c "import json; d=json.load(open('results/voyager_v4_beta/mainline/probe_summary.json')); print(d['verdict'], d['final_train_macro_f1'])"`
    2. Append the LLM token usage summary to the tail of probe.log (Pitfall 4 — harness logs tokens
       per-role via `llm.get_session_usage()` calls but no aggregate cost line). If the per-role lines
       are present in probe.log already, no action needed; otherwise add a final line of the form:
       `echo "[TOKENS] grep '\[TOKENS\]' logs/voyager_v4_beta/probe.log for per-role token totals" >> logs/voyager_v4_beta/probe.log`
       (this is a navigation hint for Plan 2; the harness already prints get_session_usage per role.)

    Do NOT compute the verdict document here. Do NOT update STATE.md here. Those belong to Plan 2 (Wave 2).
  </action>
  <verify>
    <automated>test -f results/voyager_v4_beta/mainline/probe_summary.json &amp;&amp; python -c "import json; d=json.load(open('results/voyager_v4_beta/mainline/probe_summary.json')); assert d['verdict'] in ('CONTINUE','KILL'), d; assert d['tier']=='probe'; assert d['split']=='mainline'; assert set(d['projects'])=={'mediastore','teastore','teammates'}; assert isinstance(d['final_train_macro_f1'], float); assert len(d['pass_summaries'])>=1; print('OK', d['verdict'], d['final_train_macro_f1'])"</automated>
  </verify>
  <acceptance_criteria>
    - File `logs/voyager_v4_beta/probe.log` exists and is non-empty (size > 0): `test -s logs/voyager_v4_beta/probe.log`
    - File `results/voyager_v4_beta/mainline/probe_summary.json` exists
    - probe_summary.json field `verdict` ∈ {"CONTINUE","KILL"}
    - probe_summary.json field `tier` == "probe"
    - probe_summary.json field `split` == "mainline"
    - probe_summary.json field `projects` == ["mediastore","teastore","teammates"] (any order)
    - probe_summary.json field `final_train_macro_f1` is a numeric value in [0.0, 1.0]
    - probe_summary.json field `pass_summaries` is a list of length 1 or 2
    - File `results/voyager_v4_beta/mainline/pass1_summary.json` exists with fields `train_f1s_after_l`, `committed_macro_f1`, `committed` (bool)
    - File `results/voyager_v4_beta/mainline/mediastore_bank.json` exists
    - File `results/voyager_v4_beta/mainline/teastore_bank.json` exists
    - File `results/voyager_v4_beta/mainline/teammates_bank.json` exists
    - probe.log contains the substring `[PROBE TIER]` (run_probe banner line)
    - probe.log contains the substring `verdict=` (final line printed by run_probe)
    - If passes_run == 2: file `results/voyager_v4_beta/mainline/pass2_summary.json` exists
    - No frozen artifact in `<frozen>` was modified during run (verifiable via `git diff --name-only` showing no entries from: s_linker13.py, prompts_v2.py, ilinker*.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py, s_linker13_min.py, s_linker14_voyager.py, voyager_train_tlr_v4_beta.py)
  </acceptance_criteria>
  <done>
    Probe harness has completed end-to-end. probe_summary.json contains a binary verdict (CONTINUE or KILL)
    backed by per-pass F1 evidence. Per-project _bank.json files exist for all 3 train projects.
    probe.log captures the full run incl. per-role token usage. No frozen artifact modified.
  </done>
</task>

</tasks>

<verification>
After both tasks complete, the following commands MUST all succeed:

```bash
# Pre-flight assets
test -d logs/voyager_v4_beta
test -f scripts/voyager_train_tlr_v4_beta.py

# Run output assets
test -s logs/voyager_v4_beta/probe.log
test -f results/voyager_v4_beta/mainline/probe_summary.json
test -f results/voyager_v4_beta/mainline/pass1_summary.json
test -f results/voyager_v4_beta/mainline/mediastore_bank.json
test -f results/voyager_v4_beta/mainline/teastore_bank.json
test -f results/voyager_v4_beta/mainline/teammates_bank.json

# Verdict shape check
python -c "
import json
d = json.load(open('results/voyager_v4_beta/mainline/probe_summary.json'))
assert d['verdict'] in ('CONTINUE','KILL')
assert d['tier'] == 'probe'
assert d['split'] == 'mainline'
assert set(d['projects']) == {'mediastore','teastore','teammates'}
assert isinstance(d['final_train_macro_f1'], float)
assert 0.0 <= d['final_train_macro_f1'] <= 1.0
assert 1 <= len(d['pass_summaries']) <= 2
print('Verdict:', d['verdict'], 'Macro F1:', d['final_train_macro_f1'])
"

# Frozen artifact check
git diff --name-only | grep -E '(s_linker13(\.py|_min\.py)|prompts_v2\.py|ilinker[0-9]*\.py|data_types_v2\.py|document_loader_v2\.py|pcm_parser_v2\.py|s_linker14_voyager\.py|voyager_train_tlr_v4_beta\.py)' && echo "FROZEN MODIFIED — FAIL" || echo "Frozen artifacts intact"
```
</verification>

<success_criteria>
- Probe tier harness completed on 3 train projects (REQ-V23-07)
- 1 or 2 passes executed (REQ-V23-13, probe cap = 2)
- Per-pass macro F1 logged to probe.log and persisted to pass{N}_summary.json
- probe_summary.json verdict ∈ {"CONTINUE","KILL"} backed by `final_train_macro_f1`
- Per-project _bank.json persisted for MS, TS, TM
- gpt-5.4 token usage visible in probe.log (REQ-V23-14 cost evidence; dollar conversion deferred to Plan 2)
- No frozen artifact modified
- No changes to STATE.md or 15-PROBE-VERDICT.md (those are Plan 2's deliverables)
</success_criteria>

<output>
After completion, create `.planning/phases/15-probe-tier/15-01-SUMMARY.md` capturing:
- Final command run
- Number of passes executed
- Numeric verdict (CONTINUE/KILL) and final_train_macro_f1
- Total token usage (prompt + completion) parsed from probe.log [TOKENS] lines
- Path list of every artifact created (probe.log, probe_summary.json, pass{1,2}_summary.json, {project}_bank.json)
- Any anomalies (project crashes, retries, cheap-kill triggered)
</output>
