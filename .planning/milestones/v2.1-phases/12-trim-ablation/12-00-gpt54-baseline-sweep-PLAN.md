---
phase: 12-trim-ablation
plan: 00
type: execute
wave: 1
depends_on: []
files_modified:
  - results/phase_cache_gpt54/s_linker13_clean/mediastore/
  - results/phase_cache_gpt54/s_linker13_clean/teastore/
  - results/phase_cache_gpt54/s_linker13_clean/teammates/
  - results/phase_cache_gpt54/s_linker13_clean/bigbluebutton/
  - results/phase_cache_gpt54/s_linker13_clean/jabref/
  - results/ablation_results/12_00_gpt54_baseline/
  - .planning/phases/12-trim-ablation/12-00-SUMMARY.md
autonomous: false
requirements: [PROMPT-02]
must_haves:
  truths:
    - "gpt-5.4 baseline checkpoints exist for s_linker13_clean on all 5 datasets"
    - "gpt-5.4 macro F1 anchor recorded for s_linker13_clean (used as the per-trim reference)"
    - "downstream trim plans (12-03/04/05) can resume gpt-5.4 single-step ablation from these checkpoints"
  artifacts:
    - path: "results/phase_cache_gpt54/s_linker13_clean/mediastore/final.pkl"
      provides: "gpt-5.4 final-phase checkpoint for mediastore"
    - path: "results/phase_cache_gpt54/s_linker13_clean/teastore/final.pkl"
      provides: "gpt-5.4 final-phase checkpoint for teastore"
    - path: "results/phase_cache_gpt54/s_linker13_clean/teammates/final.pkl"
      provides: "gpt-5.4 final-phase checkpoint for teammates"
    - path: "results/phase_cache_gpt54/s_linker13_clean/bigbluebutton/final.pkl"
      provides: "gpt-5.4 final-phase checkpoint for bigbluebutton"
    - path: "results/phase_cache_gpt54/s_linker13_clean/jabref/final.pkl"
      provides: "gpt-5.4 final-phase checkpoint for jabref"
    - path: "results/ablation_results/12_00_gpt54_baseline/baseline.json"
      provides: "per-dataset gpt-5.4 F1 anchor for s_linker13_clean"
  key_links:
    - from: "PHASE_CACHE_DIR env var"
      to: "results/phase_cache_gpt54"
      via: "subprocess env override in baseline-sweep command"
      pattern: "PHASE_CACHE_DIR=results/phase_cache_gpt54"
    - from: "LLM_BACKEND env var"
      to: "OPENAI backend with gpt-5.4 model"
      via: "env LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4"
      pattern: "LLM_BACKEND=openai.*OPENAI_MODEL_NAME=gpt-5\\.4"
---

<objective>
Run a one-time 5-dataset baseline sweep of `s_linker13_clean` on gpt-5.4 to populate `results/phase_cache_gpt54/s_linker13_clean/` with the layer1/layer2/entity_candidates/entity_decisions/final pickles required by every downstream single-step trim ablation. Without this anchor, plans 12-03/12-04/12-05 cannot satisfy the GATE-01 cross-model arm (PROMPT-02), because there is no gpt-5.4 baseline to compare per-trim deltas against and no upstream checkpoint to resume from.

Purpose: closes the gpt-5.4 cross-model precondition before any trim begins. The sweep is conditional — skip if the directory already exists with all five datasets populated.

Output: per-dataset gpt-5.4 phase checkpoints + a baseline.json recording macro F1, per-dataset F1, FP/FN counts on gpt-5.4 for the unmodified `s_linker13_clean`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/STATE.md
@.planning/phases/12-trim-ablation/12-CONTEXT.md

<interfaces>
<!-- Existing run_ablation.py CLI surface for env-driven sweeps -->

From run_ablation.py:
- DATASETS keys: "mediastore", "teastore", "teammates", "bigbluebutton", "jabref"
- get_backend() reads LLM_BACKEND env (claude | openai | checkpoint | codex)
- OPENAI_MODEL_NAME default "gpt-5.2" — must be overridden to "gpt-5.4"
- CLI: python run_ablation.py --variants s_linker13_clean --datasets <list>

From src/llm_sad_sam/linkers/experimental/s_linker13_clean.py:
- _checkpoint_dir() reads `PHASE_CACHE_DIR` env (default "./results/phase_cache")
- writes to `{PHASE_CACHE_DIR}/{_VARIANT_NAME}/{dataset_stem}/`
- _VARIANT_NAME == "s_linker13_clean"
- _save_phase emits 5 pickles per dataset: layer1, layer2, entity_candidates, entity_decisions, final
</interfaces>
</context>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <name>Task 0: Confirm gpt-5.4 sweep authorization and verify the conditional</name>
  <read_first>
    - .planning/phases/12-trim-ablation/12-CONTEXT.md (decisions section "gpt-5.4 baseline" line ~95)
    - .planning/REQUIREMENTS.md (GATE-01 cross-model tolerance T=1.0pp, absolute floor 0.8977)
    - .planning/STATE.md (Standing Gates section, GATE-01 cross-model)
    - ls results/phase_cache_gpt54/ 2>/dev/null
    - ls results/phase_cache_gpt54/s_linker13_clean/ 2>/dev/null (verify the gating condition)
  </read_first>
  <decision>Run the full 5-dataset gpt-5.4 baseline sweep on s_linker13_clean now?</decision>
  <context>
    The CONTEXT calls this sweep "explicitly user-authorized at the time it's needed, not assumed."
    Verification: `ls results/phase_cache_gpt54/s_linker13_clean/` MUST return either (a) all 5
    dataset subdirs each containing final.pkl — in which case option-b applies; or (b) the directory
    does not exist / is incomplete — in which case option-a is the path forward.
    Cost: 5 datasets × 5 phases × ~hundreds of LLM calls per dataset on gpt-5.4. User has confirmed
    "no LLM budget limit" (memory + CLAUDE.md). Time cost: tens of minutes to ~hours.
  </context>
  <options>
    <option id="option-a">
      <name>Run the full sweep (sweep dir absent or incomplete)</name>
      <pros>Unblocks plans 12-03/04/05; produces the gpt-5.4 anchor required by GATE-01 cross-model arm.</pros>
      <cons>Real LLM cost on gpt-5.4 (acceptable per policy).</cons>
    </option>
    <option id="option-b">
      <name>Skip — gpt-5.4 checkpoints already populated for all 5 datasets</name>
      <pros>Saves LLM spend; Plan 12-00 completes immediately.</pros>
      <cons>None if and only if all 5 final.pkl files exist AND ran with OPENAI_MODEL_NAME=gpt-5.4 (verify via the corresponding ablation_results JSON if present).</cons>
    </option>
  </options>
  <resume-signal>Select: option-a (run sweep) or option-b (skip; cite existence proof)</resume-signal>
  <acceptance_criteria>
    - If option-a: executor proceeds to Task 1.
    - If option-b: executor proceeds directly to Task 2 (validate-and-record) — Task 1 is skipped, but Task 2 still produces `results/ablation_results/12_00_gpt54_baseline/baseline.json` from the existing data.
  </acceptance_criteria>
  <done>User has selected option-a or option-b; the choice is recorded in 12-00-SUMMARY.md.</done>
</task>

<task type="auto">
  <name>Task 1: Run gpt-5.4 baseline sweep on s_linker13_clean (option-a path)</name>
  <files>results/phase_cache_gpt54/s_linker13_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/{layer1,layer2,entity_candidates,entity_decisions,final}.pkl</files>
  <read_first>
    - run_ablation.py lines 433-446 (get_backend, env defaults)
    - run_ablation.py lines 380-411 (DATASETS dict)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 1084-1102 (_checkpoint_dir, _save_phase)
  </read_first>
  <action>
    Invoke the 5-dataset sweep against the gpt-5.4 backend with checkpoints redirected to `results/phase_cache_gpt54`.

    Exact command (run from repo root):
    `LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 PHASE_CACHE_DIR=results/phase_cache_gpt54 python run_ablation.py --variants s_linker13_clean --datasets mediastore teastore teammates bigbluebutton jabref --results-dir results/ablation_results/12_00_gpt54_baseline`

    Run sequentially (one variant × 5 datasets). Do NOT parallelize datasets — the sweep writes to per-dataset subdirs, but the LLMClient state and rate-limit handling assumes serial execution.

    On a per-dataset transient failure (HTTP 5xx, rate-limit, JSON parse fail), retry the single dataset with the same command but `--datasets <single_dataset>`. Do not silently skip.

    Stream stdout to `results/ablation_results/12_00_gpt54_baseline/sweep.log` (`tee` or `>`). Capture both stdout and stderr.

    The sweep will produce, per dataset, all 5 phase pickles via `_save_phase` and one `s_linker13_clean_<dataset>_<timestamp>.csv` under the results dir. Do not delete any of these.
  </action>
  <verify>
    <automated>test -f results/phase_cache_gpt54/s_linker13_clean/mediastore/final.pkl &amp;&amp; test -f results/phase_cache_gpt54/s_linker13_clean/teastore/final.pkl &amp;&amp; test -f results/phase_cache_gpt54/s_linker13_clean/teammates/final.pkl &amp;&amp; test -f results/phase_cache_gpt54/s_linker13_clean/bigbluebutton/final.pkl &amp;&amp; test -f results/phase_cache_gpt54/s_linker13_clean/jabref/final.pkl</automated>
  </verify>
  <acceptance_criteria>
    - Each of the 5 dataset subdirs under `results/phase_cache_gpt54/s_linker13_clean/` contains all 5 pickles (`layer1.pkl`, `layer2.pkl`, `entity_candidates.pkl`, `entity_decisions.pkl`, `final.pkl`) — verified by `find results/phase_cache_gpt54/s_linker13_clean -name '*.pkl' | wc -l` returning 25.
    - The ablation JSON exists at `results/ablation_results/12_00_gpt54_baseline/ablation_<timestamp>.json` with per-dataset F1, P, R, fp, fn keys present.
    - Sweep log at `results/ablation_results/12_00_gpt54_baseline/sweep.log` is non-empty.
    - No edits to `src/llm_sad_sam/linkers/experimental/s_linker13_clean.py`, `prompts_v2.py`, or any v2.0 frozen file: `git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
  </acceptance_criteria>
  <done>5 datasets each have 5 pickles under the gpt-5.4 cache dir; ablation results JSON written.</done>
</task>

<task type="auto">
  <name>Task 2: Validate the anchor and write baseline.json + 12-00-SUMMARY.md</name>
  <files>
    - results/ablation_results/12_00_gpt54_baseline/baseline.json
    - .planning/phases/12-trim-ablation/12-00-SUMMARY.md
  </files>
  <read_first>
    - results/ablation_results/12_00_gpt54_baseline/ablation_*.json (read the most recent timestamp)
    - .planning/REQUIREMENTS.md GATE-01 cross-model row (T=1.0pp, floor 0.8977)
    - .planning/STATE.md "Standing Gates" section
  </read_first>
  <action>
    Aggregate the per-dataset gpt-5.4 results from `results/ablation_results/12_00_gpt54_baseline/ablation_<timestamp>.json` into a stable `baseline.json` with shape:
    `{ "variant": "s_linker13_clean", "backend": "openai", "model": "gpt-5.4", "per_dataset": {<dataset>: {"P": float, "R": float, "F1": float, "tp": int, "fp": int, "fn": int}}, "macro_F1": float, "absolute_floor": 0.8977, "tolerance_pp": 1.0, "baseline_target_from_v2_0": 0.9077, "captured_at": "<ISO timestamp>" }`.

    Validate: every dataset present (5 keys); each F1 in [0, 1]; macro_F1 = mean of per-dataset F1 to 4 decimal places.

    If macro_F1 < 0.8977 OR any dataset F1 = 0 (likely a sweep failure), do NOT mark the plan done — instead, write `12-00-SUMMARY.md` with a `## BLOCKER` section detailing which dataset(s) failed and surface the failure to the orchestrator. The downstream trim plans cannot proceed without a healthy anchor.

    If healthy: write `12-00-SUMMARY.md` recording the macro and per-dataset numbers as the gpt-5.4 reference. Reference D-CONTEXT decision "gpt-5.4 baseline" and PROMPT-02 requirement. List the 25 pickle paths produced and the ablation JSON path.
  </action>
  <verify>
    <automated>python -c "import json; d = json.load(open('results/ablation_results/12_00_gpt54_baseline/baseline.json')); assert d['variant']=='s_linker13_clean'; assert d['model']=='gpt-5.4'; assert len(d['per_dataset'])==5; assert all(0&lt;=v['F1']&lt;=1 for v in d['per_dataset'].values()); assert 0 &lt;= d['macro_F1'] &lt;= 1; print('OK macro_F1=', d['macro_F1'])"</automated>
  </verify>
  <acceptance_criteria>
    - `results/ablation_results/12_00_gpt54_baseline/baseline.json` exists; the verify command exits 0.
    - `baseline.json` records `model == "gpt-5.4"` and contains all 5 dataset keys: `mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`.
    - If `baseline.json.macro_F1 >= 0.8977`: SUMMARY.md states ANCHOR_HEALTHY; plan is unblocked for downstream trim plans.
    - If `baseline.json.macro_F1 < 0.8977`: SUMMARY.md states ANCHOR_BELOW_FLOOR and lists failing datasets; plan is NOT marked complete — orchestrator must decide whether the gpt-5.4 backend itself is below the v2.0-codified floor (a milestone-level concern, not a trim concern).
    - `12-00-SUMMARY.md` references decision "gpt-5.4 baseline" and requirement PROMPT-02.
  </acceptance_criteria>
  <done>baseline.json validated; SUMMARY records anchor state (healthy or blocked) with full per-dataset numbers.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Local Python → OpenAI API | gpt-5.4 calls cross a network boundary; API key in `.env` |
| Local Python → file system | Pickle writes under `results/phase_cache_gpt54/` |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-00-01 | Information disclosure | `.env` OPENAI_API_KEY | mitigate | `.env` is already gitignored; sweep does not log env vars; do not include OPENAI_API_KEY in any committed artifact or summary |
| T-12-00-02 | Tampering | pickle files under results/phase_cache_gpt54/ | accept | Local-only artifacts; pickle load is performed only by trusted in-repo code paths; not deployed externally |
| T-12-00-03 | Repudiation | which model/version produced each checkpoint | mitigate | baseline.json records model = "gpt-5.4" and ISO timestamp; sweep.log captures stdout of the entire run |
</threat_model>

<verification>
- All 5 gpt-5.4 dataset checkpoints exist (25 pickle files total).
- baseline.json validates: 5 datasets, macro_F1 within [0,1], model == "gpt-5.4".
- ANCHOR_HEALTHY iff macro_F1 ≥ 0.8977 (GATE-01 cross-model floor).
- No frozen-v2.0 file was modified.
</verification>

<success_criteria>
- gpt-5.4 baseline checkpoints in place for all 5 datasets.
- baseline.json records per-dataset and macro F1 numbers as the gpt-5.4 anchor.
- Downstream plans (12-03, 12-04, 12-05) can read these as their cross-model reference.
- If anchor is below 0.8977, the plan surfaces the blocker rather than silently letting downstream trim plans use a degraded reference.
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-00-SUMMARY.md` recording the anchor state, per-dataset F1, model, timestamp, and exact paths to the 25 pickle artifacts.
</output>
