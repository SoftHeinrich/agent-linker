---
phase: 12-trim-ablation
plan: 04
type: execute
wave: 2
depends_on: [00, 01, 02]
files_modified:
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py
  - run_ablation.py
  - tests/test_s_linker13_trim2_entval_registration.py
  - results/ablation_results/12_04_trim2_entval/
  - .planning/phases/12-trim-ablation/12-04-SUMMARY.md
autonomous: false
requirements: [PROMPT-01, PROMPT-02]
must_haves:
  truths:
    - "Variant s_linker13_trim2_entval_clean exists, importable, registered with canonical=False"
    - "The variant overrides BOTH ENTITY_EXTRACTION_RULES and VALIDATION_RULES with a merged Technique-3 (lossless rubric distillation) rubric block that collapses the overlapping architectural-participant criteria"
    - "All include/exclude semantic categories from the originals are present in the merged rubric (coverage preservation guard)"
    - "Single-step ablation invoked via the 12-02 harness re-running BOTH entity_candidates AND entity_decisions phases (per the 12-02 contract — Step 2 spans both sub-phases)"
    - "Claude Sonnet × 5 datasets + gpt-5.4 × 5 datasets ablation completed"
    - "Accept/reject verdict recorded against GATE-01 Claude + GATE-01 cross-model + GATE-06 spot probe"
    - "FP/FN delta table per dataset committed for both backends"
  artifacts:
    - path: "src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py"
      provides: "Variant overriding entity extraction + validation prompts; inherits everything else from s_linker13_clean"
      exports: ["SLinker13Trim2EntvalClean", "ENTVAL_MERGED_RUBRIC_V3", "ENTITY_EXTRACTION_RULES_V3", "VALIDATION_RULES_V3"]
    - path: "results/ablation_results/12_04_trim2_entval/claude/<dataset>/entity_decisions.json"
      provides: "Per-dataset Claude single-step ablation results"
    - path: "results/ablation_results/12_04_trim2_entval/gpt54/<dataset>/entity_decisions.json"
      provides: "Per-dataset gpt-5.4 single-step ablation results"
    - path: "results/ablation_results/12_04_trim2_entval/verdict.json"
      provides: "PASS/FAIL verdict against GATE-01 + GATE-06"
      contains: "claude_macro_F1, gpt54_macro_F1, claude_gate_pass, gpt54_gate_pass, overall_verdict"
  key_links:
    - from: "src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py"
      to: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      via: "subclass — overrides _extract_entities_enriched + _validate_with_evidence to rebind ENTITY_EXTRACTION_RULES + VALIDATION_RULES in parent module scope"
      pattern: "class SLinker13Trim2EntvalClean\\(SLinker13Clean"
    - from: "single_step harness, phase=entity_candidates"
      to: "results/phase_cache(_gpt54)?/s_linker13_clean/<dataset>/layer1.pkl"
      via: "upstream-checkpoint load — Step 2 reuses layer1, re-runs entity_candidates + entity_decisions + final"
      pattern: "PHASE_CACHE_DIR=results/phase_cache(_gpt54)?"
---

<objective>
Implement Step 2 from the Phase 11 survey §5 row 2: produce trim variant `s_linker13_trim2_entval_clean` that merges `ENTITY_EXTRACTION_RULES` (6 include + 2 exclude) and `VALIDATION_RULES` (3 APPROVE + 3 REJECT) into a single shared "architectural-participant" rubric block via Technique 3 (lossless rubric distillation). The two original prompts have structural overlap: rule 1 of EXTRACTION ("name appears directly") mirrors APPROVE-clause-1 of VALIDATION ("named as architectural participant"). The merged rubric collapses the duplicated boundary while preserving every coverage case.

Ablate via the 12-02 harness on Claude Sonnet × 5 datasets and gpt-5.4 × 5 datasets, re-running BOTH `entity_candidates` AND `entity_decisions` phases (per 12-02 contract row — Step 2 spans both because the merge spans both prompts that fire in different sub-phases). Accept iff both GATE-01 arms hold and GATE-06 probe is clean; reject and document otherwise.

Purpose: closes PROMPT-02 for the entity-pipeline prompt pair (#7 + #8 in the survey table). Tests structural-redundancy collapse without coverage loss.

Output: standalone variant file, registered; 10 ablation result JSONs (5 datasets × 2 backends, each at the entity_decisions phase); verdict.json; SUMMARY with FP/FN delta table.
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
@.planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md
@.planning/research/PROMPT-HARNESS-SURVEY.md
@BENCHMARK_TABOO.md
@src/llm_sad_sam/linkers/experimental/prompts_v2.py
@src/llm_sad_sam/linkers/experimental/s_linker13_clean.py

<interfaces>
<!-- The exact prompts being merged — extracted from prompts_v2.py + s_linker13_clean.py at planning time -->

From prompts_v2.py lines 179-191 — ENTITY_EXTRACTION_RULES (include 1-6 + exclude 1-2 + "Favor inclusion over exclusion — later verification will filter borderline cases.")
  - include 1: name (or known alias) appears directly in sentence
  - include 2: space-separated form matches compound name
  - include 3: sentence describes what specific component does by name or role
  - include 4: known synonym used
  - include 5: component participates in interaction (sender/receiver/target)
  - include 6: passive/prepositional phrase mentioning component
  - exclude 1: name only inside dotted path
  - exclude 2: name used as ordinary English word

From prompts_v2.py lines 194-205 — VALIDATION_RULES (APPROVE 1-3 + REJECT 1-3)
  - APPROVE 1: component named as architectural participant (performs operation / provides or receives service / configured / introduced)
  - APPROVE 2: section heading names component
  - APPROVE 3: sentence describes component's responsibilities/behavior/interactions
  - REJECT 1: name used as ordinary technical or English word (e.g., "proxy" in "proxy pattern")
  - REJECT 2: name modifies noun phrase without being standalone architectural ref (e.g., "observer pattern", "pipeline stage")
  - REJECT 3: sentence describes algorithm/subprocess/implementation technique sharing the name but not the component

From s_linker13_clean.py lines 47-52 — import block to mirror.
From s_linker13_clean.py lines 725-780 — `_run_single_extraction_pass` is the consumer of ENTITY_EXTRACTION_RULES (line 743).
From s_linker13_clean.py lines 818-1080 — `_validate_with_evidence` is the consumer of VALIDATION_RULES (line 984 from grep).

From .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md:
  - entity_candidates row: upstream = layer1; downstream = entity_decisions + final; fires ENTITY_EXTRACTION_RULES
  - entity_decisions row: upstream = layer1 + entity_candidates; downstream = final; fires VALIDATION_RULES
  - Step 2 modifies BOTH — harness invocation runs phase=entity_candidates first (which writes its own entity_candidates.pkl), then phase=entity_decisions reuses that, then propagates to final
  - Implementation: a single harness invocation with phase=entity_candidates will trigger the cascade through entity_decisions → final because DOWNSTREAM_DEPS["entity_candidates"] = ("entity_decisions", "final"); both modified prompts fire in the cascade because the variant overrides both at the class level

From .planning/research/PROMPT-HARNESS-SURVEY.md §5 row 2 — Technique 3 application.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Author the merged ENT+VAL rubric + create the variant + register it</name>
  <files>
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py
    - run_ablation.py
    - tests/test_s_linker13_trim2_entval_registration.py
  </files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/prompts_v2.py lines 179-205 (the two prompts being merged — extract verbatim)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 725-780 (_run_single_extraction_pass — uses ENTITY_EXTRACTION_RULES at line 743)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 818-1080 (_validate_with_evidence — uses VALIDATION_RULES at line 984)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py line 136 (_VARIANT_NAME — must override to "s_linker13_trim2_entval_clean")
    - .planning/research/PROMPT-HARNESS-SURVEY.md §5 row 2 (the prescription)
    - .planning/research/PROMPT-HARNESS-SURVEY.md §6 (V35a lesson: example-driven simplification loses coverage)
    - BENCHMARK_TABOO.md (lexical surface)
    - run_ablation.py lines 40-87 (CANONICAL_VARIANTS), 324-330 (the spec entry pattern)
  </read_first>
  <behavior>
    - Test: `from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import SLinker13Trim2EntvalClean, ENTVAL_MERGED_RUBRIC_V3, ENTITY_EXTRACTION_RULES_V3, VALIDATION_RULES_V3` succeeds.
    - Test: `SLinker13Trim2EntvalClean._VARIANT_NAME == "s_linker13_trim2_entval_clean"`.
    - Test: `SLinker13Trim2EntvalClean` is a subclass of `SLinker13Clean`.
    - Test: `ENTITY_EXTRACTION_RULES_V3` and `VALIDATION_RULES_V3` are BOTH derived from `ENTVAL_MERGED_RUBRIC_V3` plus prompt-specific framing (the merged rubric is the shared core; each constant adds a small role-specific header so the extraction prompt and the validation prompt each get a context-appropriate framing).
    - Test: `ENTITY_EXTRACTION_RULES_V3 != prompts_v2.ENTITY_EXTRACTION_RULES` (it IS modified).
    - Test: `VALIDATION_RULES_V3 != prompts_v2.VALIDATION_RULES` (it IS modified).
    - Test: rule-count delta — original total rules across both prompts = 6+2+3+3 = 14; merged rubric numbered rule count ≤ 10 (4-rule reduction per the survey's "estimated 4-rule reduction" estimate). Measure by counting `\n\d\. ` markers in `ENTVAL_MERGED_RUBRIC_V3`.
    - Test: coverage preservation — the merged rubric body contains keyword markers for every semantic category present in the originals: ["alias", "synonym", "compound", "interaction", "passive", "prepositional", "dotted", "heading", "ordinary"]. Each must be present at least once (case-insensitive substring match).
    - Test: the merged rubric body contains zero benchmark-component leakage probes (same 9-name regex from Plan 12-03).
    - Test: the merged rubric body contains "Favor inclusion" (the extraction-side tie-breaker MUST be preserved — V31 phase-contribution analysis showed it is load-bearing).
    - Test: the merged rubric body has both extraction-side ("include when") and validation-side ("approve when" / "reject when") framings detectable; or alternatively, the merged rubric is structurally pre-decision rules that BOTH consumers' prompts append the appropriate decision-format directive to. (Implementation choice — document in the variant docstring whether the merge is rubric-shared-decision-divergent or rubric-shared-decision-shared.)
    - Test: `s_linker13_trim2_entval_clean` registered in CANONICAL_VARIANTS and VARIANT_SPECS with `canonical=False`.
    - Test: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
  </behavior>
  <action>
    Author a shared core rubric `ENTVAL_MERGED_RUBRIC_V3` that collapses the duplicated architectural-participant boundary. Decision design:

    The two prompts serve different but overlapping purposes:
      - ENTITY_EXTRACTION_RULES is a PROPOSER prompt (broad recall, "favor inclusion").
      - VALIDATION_RULES is a JUDGE prompt (precision filter on the proposed set).
    Their overlap is in the architectural-participant criterion (extraction rules 1, 3, 5, 6 ↔ validation APPROVE 1, 3) and the ordinary-English exclusion (extraction exclude 2 ↔ validation REJECT 1, 2).

    Author `ENTVAL_MERGED_RUBRIC_V3` as the shared core covering:
      - Component-mention forms: direct name, known alias, known synonym, compound space-separated form, abbreviation
      - Architectural participation patterns: sender/receiver/target in described interaction; passive or prepositional phrase ("data is stored in X", "handled by X"); section heading naming the component; sentence describing the component's responsibilities/behavior
      - Non-component uses (always exclude): name inside dotted path; name used as ordinary English word; name modifying a noun phrase without standalone architectural reference (e.g., "observer pattern", "pipeline stage" — same surface terms as prompts_v2 line 203; NOT benchmark terms); name describing algorithm/subprocess/implementation technique sharing the component's name but not the component itself

    Aim for 9-10 numbered rules in the merged body (the survey estimated a 4-rule reduction from 14 → 10). The merged rubric is the SHARED CORE; it does NOT contain prompt-specific tie-breakers.

    Then author `ENTITY_EXTRACTION_RULES_V3` and `VALIDATION_RULES_V3` as small prompt-specific framings around the shared core:

      _EXTRACTION_HEADER = "RULES — include a reference when the criteria below indicate a component reference; favor inclusion over exclusion — later verification will filter borderline cases."
      _VALIDATION_HEADER = "DECISION RULES — for each candidate, APPROVE if any of the criteria below indicates this sentence references the component as an architectural participant; REJECT if the sentence uses the name in a non-component sense per the exclusion criteria."

      ENTITY_EXTRACTION_RULES_V3 = _EXTRACTION_HEADER + "\n\n" + ENTVAL_MERGED_RUBRIC_V3
      VALIDATION_RULES_V3 = _VALIDATION_HEADER + "\n\n" + ENTVAL_MERGED_RUBRIC_V3

    Then create the variant class via subclassing:

      class SLinker13Trim2EntvalClean(SLinker13Clean):
          """Step 2 trim variant: ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged via Technique 3.

          Override surface:
            - ENTITY_EXTRACTION_RULES → ENTITY_EXTRACTION_RULES_V3 (shared core + extraction-header)
            - VALIDATION_RULES → VALIDATION_RULES_V3 (shared core + validation-header)

          Rule count: <N> rules in shared core (down from 14 across the two originals).
          Coverage: every semantic category in the originals is represented in the merged rubric.

          All other prompts and pipeline phases inherit from SLinker13Clean unchanged.
          """

          _VARIANT_NAME = "s_linker13_trim2_entval_clean"

          def _run_single_extraction_pass(self, sentences, comp_names, mappings, name_to_id, sent_map, pass_label=""):
              import llm_sad_sam.linkers.experimental.s_linker13_clean as _parent_mod
              orig = _parent_mod.ENTITY_EXTRACTION_RULES
              try:
                  _parent_mod.ENTITY_EXTRACTION_RULES = ENTITY_EXTRACTION_RULES_V3
                  return super()._run_single_extraction_pass(sentences, comp_names, mappings, name_to_id, sent_map, pass_label)
              finally:
                  _parent_mod.ENTITY_EXTRACTION_RULES = orig

          def _validate_with_evidence(self, candidates, bundles, components, sent_map):
              import llm_sad_sam.linkers.experimental.s_linker13_clean as _parent_mod
              orig = _parent_mod.VALIDATION_RULES
              try:
                  _parent_mod.VALIDATION_RULES = VALIDATION_RULES_V3
                  return super()._validate_with_evidence(candidates, bundles, components, sent_map)
              finally:
                  _parent_mod.VALIDATION_RULES = orig

    Append to run_ablation.py CANONICAL_VARIANTS and VARIANT_SPECS as in Plan 12-03 (mirror the pattern; `canonical=False`; description starts with "S-Linker13 Trim2 — Phase 12 Step 2: ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged via Technique 3 (lossless rubric distillation)").

    Create `tests/test_s_linker13_trim2_entval_registration.py` testing all behaviors above.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; pytest tests/test_s_linker13_trim2_entval_registration.py -x -q &amp;&amp; python -c "from llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean import SLinker13Trim2EntvalClean, ENTVAL_MERGED_RUBRIC_V3, ENTITY_EXTRACTION_RULES_V3, VALIDATION_RULES_V3; from llm_sad_sam.linkers.experimental import prompts_v2; assert ENTITY_EXTRACTION_RULES_V3 != prompts_v2.ENTITY_EXTRACTION_RULES; assert VALIDATION_RULES_V3 != prompts_v2.VALIDATION_RULES; import re; assert not re.search(r'(?i)\\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\\b', ENTVAL_MERGED_RUBRIC_V3); assert 'Favor inclusion' in ENTITY_EXTRACTION_RULES_V3; import re as _re; n = len(_re.findall(r'\\n\\d\\.\\s', ENTVAL_MERGED_RUBRIC_V3)); assert n &lt;= 10, f'merged rubric has {n} rules, expected &lt;= 10'" &amp;&amp; python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS; assert 's_linker13_trim2_entval_clean' in CANONICAL_VARIANTS; assert VARIANT_SPECS['s_linker13_trim2_entval_clean']['canonical'] is False" &amp;&amp; git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py</automated>
  </verify>
  <acceptance_criteria>
    - All registration-test assertions pass.
    - Coverage preservation guard: all 9 semantic keyword markers present in the merged rubric.
    - Rule count: merged rubric ≤ 10 numbered rules (down from 14 across the two originals).
    - "Favor inclusion" preserved on the extraction side.
    - GATE-06 spot-probe: 9-name benchmark-component regex returns no match.
    - GATE-02 unaffected — variant registered under `missing` in baseline fixture; `pytest tests/test_v20_baseline_regression.py -q` exits 0.
    - Zero edits to v2.0 frozen files or to s_linker13_clean.py.
  </acceptance_criteria>
  <done>Variant + merged rubric authored, registered, coverage tests green.</done>
</task>

<task type="auto">
  <name>Task 2: Run Claude Sonnet single-step ablation × 5 datasets via 12-02 harness</name>
  <files>
    - results/ablation_results/12_04_trim2_entval/claude/{mediastore,teastore,teammates,bigbluebutton,jabref}/entity_decisions.json
    - results/phase_cache/s_linker13_trim2_entval_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/
  </files>
  <read_first>
    - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md (entity_candidates → cascade to entity_decisions + final)
    - results/phase_cache/s_linker13_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.pkl (upstream — must exist)
    - src/llm_sad_sam/ablation/single_step.py (engine entry)
    - .planning/REQUIREMENTS.md GATE-01 Claude arm
  </read_first>
  <action>
    For each dataset in {mediastore, teastore, teammates, bigbluebutton, jabref} run sequentially:
      `LLM_BACKEND=claude CLAUDE_MODEL=sonnet PHASE_CACHE_DIR=results/phase_cache python -m llm_sad_sam.ablation single_step --variant s_linker13_trim2_entval_clean --dataset <ds> --phase entity_candidates --results-dir results/ablation_results/12_04_trim2_entval/claude --backend claude`

    The harness loads `results/phase_cache/s_linker13_clean/<ds>/layer1.pkl` as upstream, runs entity_candidates with `ENTITY_EXTRACTION_RULES_V3` injected via the override, cascades into entity_decisions (which uses `VALIDATION_RULES_V3` via the same override), then final dedup. Per DOWNSTREAM_DEPS["entity_candidates"] = ("entity_decisions", "final"), the cascade is automatic.

    NOTE: the result JSON emitted by the harness has filename `<phase>.json` where phase is the LAST phase scored — in this case `entity_decisions.json` is the natural name because that is where the validated set lands; but the harness writes one JSON per top-level invocation phase. If the harness writes `entity_candidates.json` instead of `entity_decisions.json`, accept that filename and adjust the verify command accordingly. Document the actual filename in SUMMARY.

    Each invocation writes per-variant checkpoints under `results/phase_cache/s_linker13_trim2_entval_clean/<ds>/` (enforced by `_VARIANT_NAME` override). Stream stdout to `results/ablation_results/12_04_trim2_entval/claude/sweep.log`.

    Per-dataset transient failure: retry the single dataset. Do not skip.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; for ds in mediastore teastore teammates bigbluebutton jabref; do ls results/ablation_results/12_04_trim2_entval/claude/$ds/*.json 1&gt;/dev/null 2&gt;&amp;1 || { echo "MISSING $ds"; exit 1; }; done &amp;&amp; python -c "import json, glob; rows=[json.load(open(glob.glob(f'results/ablation_results/12_04_trim2_entval/claude/{ds}/*.json')[0])) for ds in ['mediastore','teastore','teammates','bigbluebutton','jabref']]; macro = sum(r['F1'] for r in rows)/5; print(f'Claude macro F1 = {macro:.4f}'); assert all(0 &lt;= r['F1'] &lt;= 1 for r in rows)"</automated>
  </verify>
  <acceptance_criteria>
    - All 5 Claude per-dataset result JSONs exist with valid F1.
    - Macro F1 printed.
    - Per-variant cache populated at `results/phase_cache/s_linker13_trim2_entval_clean/<ds>/`.
    - No edits to v2.0 frozen files or s_linker13_clean.py.
  </acceptance_criteria>
  <done>Claude × 5 datasets done; F1 captured.</done>
</task>

<task type="auto">
  <name>Task 3: Run gpt-5.4 single-step ablation × 5 datasets via 12-02 harness</name>
  <files>
    - results/ablation_results/12_04_trim2_entval/gpt54/{mediastore,teastore,teammates,bigbluebutton,jabref}/*.json
    - results/phase_cache_gpt54/s_linker13_trim2_entval_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/
  </files>
  <read_first>
    - results/phase_cache_gpt54/s_linker13_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.pkl (upstream — must exist; produced by Plan 12-00)
    - results/ablation_results/12_00_gpt54_baseline/baseline.json (anchor)
    - .planning/REQUIREMENTS.md GATE-01 cross-model arm
  </read_first>
  <action>
    For each dataset run sequentially:
      `LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 PHASE_CACHE_DIR=results/phase_cache_gpt54 python -m llm_sad_sam.ablation single_step --variant s_linker13_trim2_entval_clean --dataset <ds> --phase entity_candidates --results-dir results/ablation_results/12_04_trim2_entval/gpt54 --backend openai --model gpt-5.4`

    Stream stdout to `results/ablation_results/12_04_trim2_entval/gpt54/sweep.log`.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; for ds in mediastore teastore teammates bigbluebutton jabref; do ls results/ablation_results/12_04_trim2_entval/gpt54/$ds/*.json 1&gt;/dev/null 2&gt;&amp;1 || { echo "MISSING $ds"; exit 1; }; done &amp;&amp; python -c "import json, glob; rows=[json.load(open(glob.glob(f'results/ablation_results/12_04_trim2_entval/gpt54/{ds}/*.json')[0])) for ds in ['mediastore','teastore','teammates','bigbluebutton','jabref']]; macro = sum(r['F1'] for r in rows)/5; print(f'gpt-5.4 macro F1 = {macro:.4f}'); assert all(0 &lt;= r['F1'] &lt;= 1 for r in rows)"</automated>
  </verify>
  <acceptance_criteria>
    - All 5 gpt-5.4 per-dataset result JSONs exist.
    - Macro F1 printed.
    - No edits to v2.0 frozen files or s_linker13_clean.py.
  </acceptance_criteria>
  <done>gpt-5.4 × 5 datasets done.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Adjudicate verdict + write SUMMARY (PASS or REJECT)</name>
  <what-built>10 per-dataset ablation JSONs + verdict.json aggregating against GATE-01 Claude + GATE-01 cross-model + GATE-06 spot probe.</what-built>
  <read_first>
    - results/ablation_results/12_04_trim2_entval/claude/*/<phase>.json
    - results/ablation_results/12_04_trim2_entval/gpt54/*/<phase>.json
    - results/ablation_results/12_00_gpt54_baseline/baseline.json
    - tests/fixtures/v2_0_baseline.json
    - .planning/REQUIREMENTS.md GATE-01 rows
    - BENCHMARK_TABOO.md
  </read_first>
  <how-to-verify>
    The executor writes `results/ablation_results/12_04_trim2_entval/verdict.json` AND `12-04-SUMMARY.md`. The user (this checkpoint) confirms:

    1. **Verdict schema** (identical to Plan 12-03 schema, adapted for trim_id "trim2_entval"):
       `{ "trim_id": "trim2_entval", "claude": {...}, "gpt54": {...}, "gate06_probe": {...}, "overall_verdict": ... }`

    2. **GATE-01 Claude check**:
       - claude.macro_F1 ≥ 0.93
       - For each ds: delta_F1 ≥ -0.06 (BBB) / -0.02 (other) vs `tests/fixtures/v2_0_baseline.json`'s s_linker13_clean baseline (the parent's Phase 10 anchor)
       - claude.gate_pass = ALL hold

    3. **GATE-01 cross-model check**:
       - gpt54.macro_F1 ≥ 0.8977
       - All 5 datasets reported

    4. **GATE-06 probe**: clean (already enforced by Task 1's behavior test)

    5. **Overall verdict**: ACCEPT iff all three pass; REJECT with explicit gate_reason otherwise

    6. **SUMMARY** contains:
       - Verdict prominent at top
       - Per-dataset FP/FN delta table (Claude + gpt-5.4)
       - Rule-count reduction stat (e.g., "14 → 10 rules, 4-rule reduction per survey §5 row 2")
       - For ACCEPT: confirmation the variant carries to Plan 12-06 + Plan 13-01
       - For REJECT: failing arm + dataset(s) explicit; variant NOT carried
       - References PROMPT-01 + PROMPT-02
  </how-to-verify>
  <resume-signal>Type "approved" or describe corrections needed.</resume-signal>
  <acceptance_criteria>
    - verdict.json exists, validates against schema.
    - 12-04-SUMMARY.md exists; verdict explicit; FP/FN delta table present; references PROMPT-01 + PROMPT-02.
    - If REJECT: SUMMARY lists failing arm + datasets, NOT registered for Plan 13-01.
    - If ACCEPT: SUMMARY confirms carry to Plan 12-06 + (subject to that audit) Plan 13-01.
  </acceptance_criteria>
  <done>Verdict recorded; SUMMARY shipped; accept/reject signal canonical for Plan 12-06 + 13-01.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Two-method override on SLinker13Clean | Both `_run_single_extraction_pass` and `_validate_with_evidence` are overridden via parent-module monkey-patch with try/finally — same pattern as Plan 12-03, doubled |
| Variant checkpoint dir | Namespaced via `_VARIANT_NAME` (assert in `_checkpoint_dir`) — cannot overwrite baseline |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-04-01 | Tampering | accidental edits to prompts_v2.py while authoring merged rubric | mitigate | Task 1 verify: `git diff --quiet prompts_v2.py` exits 0; new constants live in the variant file |
| T-12-04-02 | Information disclosure | merged rubric leaks coverage of one prompt's edge case the other prompt never needed → recall regression on a dataset where that edge case mattered | mitigate | Task 1's coverage-preservation guard (9 keyword markers) + the per-dataset ablation surfaces any uncaught coverage loss as F1 delta in Task 4 verdict |
| T-12-04-03 | Tampering | monkey-patch in `_validate_with_evidence` collides with concurrent `_run_single_extraction_pass` from a different thread of `_run_parallel` | mitigate | Document non-thread-safety in docstring; harness runs single-variant, single-dataset sequentially per Task 2/3 invocations; the inner DAG parallelism in s_linker13_clean only parallelizes across distinct phase functions, not multiple instances of the same phase function |
| T-12-04-04 | Information disclosure | benchmark-component leakage in merged rubric body | mitigate | Task 1 behavior test runs the 9-name probe; Plan 12-06 runs the full TABOO sweep |
</threat_model>

<verification>
- Variant + merged rubric authored; coverage / rule-count / benchmark-component guards all pass.
- 5 Claude + 5 gpt-5.4 per-dataset result JSONs exist with valid F1.
- verdict.json validates against schema; overall_verdict explicit.
- SUMMARY shipped with FP/FN delta tables + rule-count reduction stat.
- Zero edits to v2.0 frozen files or to s_linker13_clean.py.
- GATE-02 unaffected.
</verification>

<success_criteria>
- PROMPT-02 progressed for Step 2: trim variant designed, ablated on both backends, verdict recorded.
- PROMPT-01 progressed: the two merged constants join the v2→v3 mapping table (under "renamed/merged" status).
- Plan 12-06 inherits this trim's merged rubric + variant file as a primary audit target.
- Plan 13-01 inherits the accept/reject signal.
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-04-SUMMARY.md`.
</output>
</content>
</invoke>