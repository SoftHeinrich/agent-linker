---
phase: 12-trim-ablation
plan: 03
type: execute
wave: 2
depends_on: [00, 01, 02]
files_modified:
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py
  - run_ablation.py
  - tests/test_s_linker13_trim1_judge_registration.py
  - results/ablation_results/12_03_trim1_judge/
  - .planning/phases/12-trim-ablation/12-03-SUMMARY.md
autonomous: false
requirements: [PROMPT-01, PROMPT-02]
must_haves:
  truths:
    - "Variant s_linker13_trim1_judge_clean exists, importable, registered with canonical=False"
    - "The variant overrides DOC_KNOWLEDGE_JUDGE_RULES + DOC_KNOWLEDGE_JUDGE_EXAMPLES with a trimmed rubric applying Technique 3 (lossless rubric distillation) + Technique 8 (reasoning-before-conclusion directive order)"
    - "All 7 worked examples from DOC_KNOWLEDGE_JUDGE_EXAMPLES are PRESERVED verbatim (V35a lesson: example-driven simplification regresses Claude)"
    - "The 3 numbered rules in DOC_KNOWLEDGE_JUDGE_RULES are collapsed into a single APPROVE-biased rubric block with verdict format AFTER reasoning (per arXiv 2603.13351)"
    - "Single-step ablation invoked via the 12-02 harness: Claude Sonnet × 5 datasets + gpt-5.4 × 5 datasets, re-running ONLY layer1 (the phase that fires the judge prompt)"
    - "Accept/reject verdict recorded: PASS iff Claude macro F1 ≥ 0.93 AND BBB drop ≤ 6pp AND other-dataset drop ≤ 2pp AND gpt-5.4 macro F1 ≥ 0.8977"
    - "FP/FN delta table per dataset committed for both backends"
    - "GATE-06 spot check on the new trimmed rubric body returns clean (benchmark-component probe = 0 hits)"
  artifacts:
    - path: "src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py"
      provides: "Variant overriding the alias-judge prompts; inherits everything else from s_linker13_clean"
      exports: ["SLinker13Trim1JudgeClean", "DOC_KNOWLEDGE_JUDGE_RUBRIC_V3", "DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3"]
    - path: "results/ablation_results/12_03_trim1_judge/claude/<dataset>/layer1.json"
      provides: "Per-dataset Claude Sonnet single-step ablation results for the trim"
    - path: "results/ablation_results/12_03_trim1_judge/gpt54/<dataset>/layer1.json"
      provides: "Per-dataset gpt-5.4 single-step ablation results for the trim"
    - path: "results/ablation_results/12_03_trim1_judge/verdict.json"
      provides: "PASS/FAIL verdict against GATE-01 Claude + GATE-01 cross-model"
      contains: "claude_macro_F1, gpt54_macro_F1, claude_gate_pass, gpt54_gate_pass, overall_verdict"
  key_links:
    - from: "src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py"
      to: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      via: "class SLinker13Trim1JudgeClean(SLinker13Clean) — overrides only the judge prompt-fragment constants"
      pattern: "class SLinker13Trim1JudgeClean\\(SLinker13Clean"
    - from: "src/llm_sad_sam/ablation single_step CLI"
      to: "results/phase_cache/s_linker13_clean/<dataset>/layer1.pkl  AND  results/phase_cache_gpt54/s_linker13_clean/<dataset>/layer1.pkl"
      via: "harness loads upstream-empty for layer1 (first phase) but the trim re-runs Tier 1 doc-knowledge enriched call which is INSIDE layer1; downstream replay uses cached prompts"
      pattern: "PHASE_CACHE_DIR=results/phase_cache(_gpt54)?"
---

<objective>
Implement Step 1 from the Phase 11 survey §5: produce a trim variant `s_linker13_trim1_judge_clean` that replaces `DOC_KNOWLEDGE_JUDGE_RULES` (3 numbered rules + IMPORTANT closer) with a single APPROVE-biased rubric, applying Technique 3 (lossless rubric distillation — same boundary, denser surface) + Technique 8 (reasoning-before-conclusion directive order — the "When in doubt, APPROVE" tie-breaker must precede the verdict format, per arXiv 2603.13351). The 7 worked examples in `DOC_KNOWLEDGE_JUDGE_EXAMPLES` are KEPT verbatim (V35a lesson: example removal regresses Claude). Ablate via the 12-02 harness on Claude Sonnet × 5 datasets and gpt-5.4 × 5 datasets, re-running only the phase where the judge prompt fires. Accept iff both GATE-01 arms hold and GATE-06 probe is clean; reject and document otherwise.

Purpose: closes PROMPT-02 for the highest-rule-mass prompt pair (#4 + #5 in the survey table). Tests the V35 escape hypothesis: lossless distillation + reasoning-order fix preserves information density Claude exploits.

Output: standalone variant file, registered in CANONICAL_VARIANTS + VARIANT_SPECS with canonical=False; 10 ablation result JSONs (5 datasets × 2 backends); verdict.json; SUMMARY with FP/FN delta table.
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
<!-- The exact prompts being trimmed — extracted from prompts_v2.py at planning time -->

From prompts_v2.py lines 87-121 — DOC_KNOWLEDGE_JUDGE_EXAMPLES (7 worked examples). KEEP VERBATIM in the trim.
From prompts_v2.py lines 124-139 — DOC_KNOWLEDGE_JUDGE_RULES:
  - Rule 1: AUTO-APPROVE list (4 sub-categories: abbreviations from initials/words; trailing-word; CamelCase; multi-word phrases)
  - Rule 2: APPROVE clause with generic-word exclusion (system/process/utility/component/module)
  - Rule 3: REJECT only if clearly generic OR refers to different component / whole system
  - IMPORTANT closer: "When in doubt, APPROVE. False approvals are filtered by later pipeline stages; false rejections cause permanent recall loss."

From s_linker13_clean.py lines 47-52 — the import block; the trim variant inherits this and overrides the JUDGE constants only.
From s_linker13_clean.py lines 428-451 — the judge prompt assembly; it embeds DOC_KNOWLEDGE_JUDGE_EXAMPLES + DOC_KNOWLEDGE_JUDGE_RULES via f-string. Subclass override pattern: rebind module-level names by writing them as class attributes on the subclass AND override the one method that references them at module scope (`_learn_document_knowledge_enriched` line 366) so the prompt assembly reads `self.DOC_KNOWLEDGE_JUDGE_RULES` instead of the module global.

From .planning/research/PROMPT-HARNESS-SURVEY.md §5 row 1 — Technique 3 + Technique 8 prescription for this prompt pair.
From .planning/research/PROMPT-HARNESS-SURVEY.md §6 — V35c failure mode: concrete output examples bias sentence-number distribution. Do NOT change `DOC_KNOWLEDGE_JUDGE_EXAMPLES`'s example structure.

From BENCHMARK_TABOO.md — the surface terms the new rubric MUST NOT include.

From .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md:
  - layer1 modifications: re-run layer1 → cascades to layer2, entity_candidates, entity_decisions, final
  - layer1 prompts include DOC_KNOWLEDGE_JUDGE_RULES + DOC_KNOWLEDGE_JUDGE_EXAMPLES (target of this trim)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Author the trimmed rubric + create the variant + register it</name>
  <files>
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py
    - run_ablation.py
    - tests/test_s_linker13_trim1_judge_registration.py
  </files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/prompts_v2.py lines 87-139 (the exact DOC_KNOWLEDGE_JUDGE_EXAMPLES + DOC_KNOWLEDGE_JUDGE_RULES source — extract verbatim into the trim variant)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 47-52 (import block to mirror)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 366-466 (_learn_document_knowledge_enriched — the consumer method to override)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py line 136 (_VARIANT_NAME — must override to "s_linker13_trim1_judge_clean")
    - .planning/research/PROMPT-HARNESS-SURVEY.md §5 row 1 (the prescription)
    - .planning/research/PROMPT-HARNESS-SURVEY.md §6 (V35c — what NOT to do)
    - .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md §4 (the inference-time rubric pattern — Step 3's territory; do NOT apply here, this is Step 1's static rubric only)
    - BENCHMARK_TABOO.md (the lexical surface)
    - run_ablation.py lines 40-87 (CANONICAL_VARIANTS), 324-330 (the existing s_linker13_clean entry shape to mirror)
  </read_first>
  <behavior>
    - Test: `from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import SLinker13Trim1JudgeClean, DOC_KNOWLEDGE_JUDGE_RUBRIC_V3, DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3` succeeds.
    - Test: `SLinker13Trim1JudgeClean._VARIANT_NAME == "s_linker13_trim1_judge_clean"` (distinct checkpoint namespace).
    - Test: `SLinker13Trim1JudgeClean` is a subclass of `SLinker13Clean`.
    - Test: `DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 == prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES` (all 7 worked examples kept verbatim — V35a guard).
    - Test: `DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 != prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES` (the rubric IS modified).
    - Test: the trimmed rubric body contains the substring "When in doubt" BEFORE the substring "Return" or "JSON" or "verdict" (reasoning-before-conclusion ordering check — the tie-breaker is emitted in the rubric body, not at the end).
    - Test: the trimmed rubric body contains exactly ONE block of decision criteria (not three numbered rules) — measured by counting `\n\d\. ` numbered-rule markers — count == 0 (rubric is prose, not enumerated rules; consistent with Technique 3 lossless distillation).
    - Test: the trimmed rubric body contains the 4 AUTO-APPROVE sub-categories present in the original Rule 1 (abbreviations, trailing-word, CamelCase, multi-word phrases) — verified by `all(kw in rubric_lower for kw in ["abbreviation", "trailing", "camelcase", "multi-word"])`. Coverage preservation guard.
    - Test: the trimmed rubric body contains zero benchmark-component leakage probes — `re.search(r'(?i)\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|AudioWatermarking|MediaManagement|WebUI|Recommender|Persistence|SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|bibentry)\b', rubric) is None`.
    - Test: `s_linker13_trim1_judge_clean` is registered in run_ablation.CANONICAL_VARIANTS and `VARIANT_SPECS["s_linker13_trim1_judge_clean"]["canonical"] is False`.
    - Test: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
  </behavior>
  <action>
    Author the trimmed rubric as a Python string constant `DOC_KNOWLEDGE_JUDGE_RUBRIC_V3` inside the new variant file. The rubric MUST:
      - Apply Technique 3 (lossless rubric distillation): merge the 3 numbered rules of the original `DOC_KNOWLEDGE_JUDGE_RULES` into a single APPROVE-biased rubric block. The merged rubric covers all 4 AUTO-APPROVE sub-categories (abbreviations, trailing-word, CamelCase, multi-word phrases) AND the APPROVE clause (plausibly refers to one component, not a generic word like "system/process/utility/component/module") AND the REJECT clause (clearly generic OR refers to different component / whole system).
      - Apply Technique 8 (reasoning-before-conclusion): place the "When in doubt, APPROVE — false approvals filtered downstream, false rejections lose recall permanently" tie-breaker BEFORE any verdict-format directive in the rubric body. The verdict-format directive itself remains in the consumer method's `Return JSON: {{"approved": [...]}}` line — the rubric body never contains JSON template strings (V35c guard).
      - Use prose-level distillation, NOT numbered rules. The decision criteria flow as continuous text rather than 3 bullet points (Technique 3 application). This is the structural change that distinguishes lossless distillation from V35a-style replacement.
      - NOT include any of the BENCHMARK_TABOO terms listed in the behavior test. All illustrative phrasing must use safe SE-textbook terms (parser, scheduler, dispatcher, broker, lexer, code generator, render engine — same surface AMBIGUITY_FEW_SHOT already uses).
      - Be approximately 80-130% of the byte-length of the original `DOC_KNOWLEDGE_JUDGE_RULES` (Technique 3 is lossless density compression, not aggressive deletion; significant size deltas indicate either V35-style coverage loss or unjustified expansion). Document the actual length delta in the docstring.

    Author `DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 = DOC_KNOWLEDGE_JUDGE_EXAMPLES` (byte-equal alias — examples are preserved verbatim per V35a lesson).

    Then create the variant class. Use SUBCLASSING (not file-copy — this trim is a surgical 1-prompt override, not a structural fork):

      from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean
      from llm_sad_sam.linkers.experimental.prompts_v2 import (
          DOC_KNOWLEDGE_JUDGE_EXAMPLES as _V2_JUDGE_EXAMPLES,
          # we deliberately do NOT import DOC_KNOWLEDGE_JUDGE_RULES — the trim overrides it
      )

      DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 = _V2_JUDGE_EXAMPLES  # byte-equal, V35a guard

      DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 = """<authored rubric body here>"""

      class SLinker13Trim1JudgeClean(SLinker13Clean):
          """Step 1 trim variant: alias-judge prompts trimmed via Technique 3 + Technique 8.

          Override surface:
            - DOC_KNOWLEDGE_JUDGE_RULES → DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 (lossless distillation, reasoning-before-conclusion order)
            - DOC_KNOWLEDGE_JUDGE_EXAMPLES → DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 (byte-equal, kept verbatim — V35a guard)

          All other prompts and pipeline phases inherit from SLinker13Clean unchanged.
          Length delta vs original DOC_KNOWLEDGE_JUDGE_RULES: <N bytes vs M bytes, X% delta>.
          """

          _VARIANT_NAME = "s_linker13_trim1_judge_clean"

          def _learn_document_knowledge_enriched(self, sentences, components):
              # Identical to parent, with the judge prompt constants rebound to V3.
              # Parent method assembles prompt2 via:
              #   {DOC_KNOWLEDGE_JUDGE_EXAMPLES}
              #   {DOC_KNOWLEDGE_JUDGE_RULES}
              # We need to override JUST that prompt assembly. The cleanest pattern:
              # monkey-patch the two names in the parent module's scope via a context
              # manager, then call super().
              import llm_sad_sam.linkers.experimental.s_linker13_clean as _parent_mod
              orig_rules = _parent_mod.DOC_KNOWLEDGE_JUDGE_RULES
              orig_examples = _parent_mod.DOC_KNOWLEDGE_JUDGE_EXAMPLES
              try:
                  _parent_mod.DOC_KNOWLEDGE_JUDGE_RULES = DOC_KNOWLEDGE_JUDGE_RUBRIC_V3
                  _parent_mod.DOC_KNOWLEDGE_JUDGE_EXAMPLES = DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3
                  return super()._learn_document_knowledge_enriched(sentences, components)
              finally:
                  _parent_mod.DOC_KNOWLEDGE_JUDGE_RULES = orig_rules
                  _parent_mod.DOC_KNOWLEDGE_JUDGE_EXAMPLES = orig_examples

    Implementation note: monkey-patching the parent module's name scope is the pragmatic minimal-invasive pattern here because the parent's `_learn_document_knowledge_enriched` references the names at module scope inside its f-string. This is reviewer-defensible because it is explicit (named overrides, scoped via try/finally) and confines the override to a single method. Alternative would be to fork the entire method body — that would duplicate 100 lines and is rejected as scope-bloat. Document the choice in the class docstring.

    Append to `run_ablation.py`:
      - `"s_linker13_trim1_judge_clean",` to CANONICAL_VARIANTS (after `"s_linker13_clean",`).
      - New VARIANT_SPECS entry:
          "s_linker13_trim1_judge_clean": dict(
              aliases=(),
              module="llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean",
              class_name="SLinker13Trim1JudgeClean",
              description="S-Linker13 Trim1 — Phase 12 Step 1: DOC_KNOWLEDGE_JUDGE_RULES distilled (Technique 3 + 8); 7 worked examples preserved verbatim",
              canonical=False,
          ),

    Create `tests/test_s_linker13_trim1_judge_registration.py` testing all behaviors above.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; pytest tests/test_s_linker13_trim1_judge_registration.py -x -q &amp;&amp; python -c "from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import SLinker13Trim1JudgeClean, DOC_KNOWLEDGE_JUDGE_RUBRIC_V3, DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3; from llm_sad_sam.linkers.experimental import prompts_v2; assert DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 == prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES; assert DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 != prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES; r = DOC_KNOWLEDGE_JUDGE_RUBRIC_V3; assert 'When in doubt' in r; import re; assert not re.search(r'(?i)\\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\\b', r)" &amp;&amp; python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS; assert 's_linker13_trim1_judge_clean' in CANONICAL_VARIANTS; assert VARIANT_SPECS['s_linker13_trim1_judge_clean']['canonical'] is False" &amp;&amp; git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py</automated>
  </verify>
  <acceptance_criteria>
    - All registration-test assertions pass.
    - The 4 AUTO-APPROVE sub-categories (abbreviations / trailing-word / CamelCase / multi-word phrases) are present in the rubric body (coverage preservation).
    - The rubric body is prose, not numbered rules (zero `\n\d\. ` markers).
    - The tie-breaker ("When in doubt") appears BEFORE any verdict-format directive in the rubric body.
    - All 7 worked examples in DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 are byte-equal to v2.
    - GATE-06 spot-probe is clean (9-name benchmark-component regex returns no match).
    - GATE-02 unaffected — register the new variant under `missing` in `tests/fixtures/v2_0_baseline.json` per the documented "snapshot it before promotion" pattern; `pytest tests/test_v20_baseline_regression.py -q` exits 0.
    - Zero edits to any v2.0 frozen file or to s_linker13_clean.py (verify command exits 0).
  </acceptance_criteria>
  <done>Variant + rubric authored, registered, isolated coverage tests green.</done>
</task>

<task type="auto">
  <name>Task 2: Run Claude Sonnet single-step ablation × 5 datasets via 12-02 harness</name>
  <files>
    - results/ablation_results/12_03_trim1_judge/claude/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/phase_cache/s_linker13_trim1_judge_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/
  </files>
  <read_first>
    - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md (layer1 → cascade to layer2/entity_candidates/entity_decisions/final)
    - results/phase_cache/s_linker13_clean/ (verify the 5 baseline dataset dirs exist before invoking)
    - src/llm_sad_sam/ablation/single_step.py (the engine — refresh how to invoke)
    - .planning/REQUIREMENTS.md GATE-01 Claude arm (macro F1 ≥ 0.93; BBB ≤ 6pp; other ≤ 2pp)
  </read_first>
  <action>
    Invoke the 12-02 harness on Claude Sonnet for each of the 5 datasets. The judge prompt fires inside `_learn_document_knowledge_enriched` which is part of layer1 — so the harness target is phase="layer1".

    For each dataset in {mediastore, teastore, teammates, bigbluebutton, jabref} run sequentially:
      `LLM_BACKEND=claude CLAUDE_MODEL=sonnet PHASE_CACHE_DIR=results/phase_cache python -m llm_sad_sam.ablation single_step --variant s_linker13_trim1_judge_clean --dataset <ds> --phase layer1 --results-dir results/ablation_results/12_03_trim1_judge/claude --backend claude`

    Notes for the executor:
      - PHASE_CACHE_DIR points at the BASELINE cache because the harness READS upstream-empty (layer1 is the first phase) but WRITES the modified variant's outputs into `${PHASE_CACHE_DIR}/s_linker13_trim1_judge_clean/<ds>/` per `_VARIANT_NAME` override. Variant isolation is enforced by the `_checkpoint_dir` assertion in s_linker13_clean.py:1090-1093.
      - Downstream phases (layer2, entity_candidates, entity_decisions, final) WILL re-run because layer1's doc_knowledge change cascades. This is the documented behavior per the 12-02 harness contract.
      - Each invocation writes one JSON per dataset under `results/ablation_results/12_03_trim1_judge/claude/<ds>/layer1.json` with F1/P/R/fp/fn/baseline_F1/delta_F1 keys.
      - Stream stdout to `results/ablation_results/12_03_trim1_judge/claude/sweep.log` (`tee`).
      - On per-dataset transient failure (HTTP, rate-limit, JSON parse), retry the SINGLE dataset with the same command. Do not silently skip.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; for ds in mediastore teastore teammates bigbluebutton jabref; do test -f "results/ablation_results/12_03_trim1_judge/claude/$ds/layer1.json" || { echo "MISSING $ds"; exit 1; }; done &amp;&amp; python -c "import json; rows=[json.load(open(f'results/ablation_results/12_03_trim1_judge/claude/{ds}/layer1.json')) for ds in ['mediastore','teastore','teammates','bigbluebutton','jabref']]; macro = sum(r['F1'] for r in rows)/5; print(f'Claude macro F1 = {macro:.4f}'); assert all(0 &lt;= r['F1'] &lt;= 1 for r in rows)"</automated>
  </verify>
  <acceptance_criteria>
    - All 5 Claude per-dataset result JSONs exist with valid F1 values.
    - Macro F1 = mean of per-dataset F1 is printed (will be assessed against GATE-01 in Task 4).
    - No edits to v2.0 frozen files: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
    - Per-variant checkpoint dir `results/phase_cache/s_linker13_trim1_judge_clean/<ds>/` is populated for downstream Task 3 (gpt-5.4 arm references the same variant; checkpoints are backend-tagged via different PHASE_CACHE_DIR roots).
  </acceptance_criteria>
  <done>Claude Sonnet × 5 datasets ablation done; per-dataset F1 captured.</done>
</task>

<task type="auto">
  <name>Task 3: Run gpt-5.4 single-step ablation × 5 datasets via 12-02 harness</name>
  <files>
    - results/ablation_results/12_03_trim1_judge/gpt54/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/phase_cache_gpt54/s_linker13_trim1_judge_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/
  </files>
  <read_first>
    - results/phase_cache_gpt54/s_linker13_clean/ (verify the 5 baseline dirs exist — Plan 12-00 acceptance)
    - results/ablation_results/12_00_gpt54_baseline/baseline.json (the gpt-5.4 anchor)
    - .planning/REQUIREMENTS.md GATE-01 cross-model arm (gpt-5.4 macro F1 ≥ 0.8977)
  </read_first>
  <action>
    For each dataset in {mediastore, teastore, teammates, bigbluebutton, jabref} run sequentially:
      `LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 PHASE_CACHE_DIR=results/phase_cache_gpt54 python -m llm_sad_sam.ablation single_step --variant s_linker13_trim1_judge_clean --dataset <ds> --phase layer1 --results-dir results/ablation_results/12_03_trim1_judge/gpt54 --backend openai --model gpt-5.4`

    Notes:
      - PHASE_CACHE_DIR pointed at the gpt-5.4 root (per 12-00 anchor).
      - The variant writes to `results/phase_cache_gpt54/s_linker13_trim1_judge_clean/<ds>/`.
      - Stream stdout to `results/ablation_results/12_03_trim1_judge/gpt54/sweep.log`.
      - Per-dataset failure: retry the single dataset. Do not skip.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; for ds in mediastore teastore teammates bigbluebutton jabref; do test -f "results/ablation_results/12_03_trim1_judge/gpt54/$ds/layer1.json" || { echo "MISSING $ds"; exit 1; }; done &amp;&amp; python -c "import json; rows=[json.load(open(f'results/ablation_results/12_03_trim1_judge/gpt54/{ds}/layer1.json')) for ds in ['mediastore','teastore','teammates','bigbluebutton','jabref']]; macro = sum(r['F1'] for r in rows)/5; print(f'gpt-5.4 macro F1 = {macro:.4f}'); assert all(0 &lt;= r['F1'] &lt;= 1 for r in rows)"</automated>
  </verify>
  <acceptance_criteria>
    - All 5 gpt-5.4 per-dataset result JSONs exist with valid F1.
    - Macro F1 printed.
    - No edits to v2.0 frozen files (git diff --quiet check).
  </acceptance_criteria>
  <done>gpt-5.4 × 5 datasets ablation done.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Adjudicate verdict + write SUMMARY (PASS or REJECT)</name>
  <what-built>10 per-dataset ablation JSONs (5 Claude + 5 gpt-5.4) plus a verdict.json aggregating against GATE-01 Claude + GATE-01 cross-model + GATE-06 spot probe.</what-built>
  <read_first>
    - results/ablation_results/12_03_trim1_judge/claude/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/ablation_results/12_03_trim1_judge/gpt54/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/ablation_results/12_00_gpt54_baseline/baseline.json (gpt-5.4 baseline anchor)
    - tests/fixtures/v2_0_baseline.json (Claude per-dataset baseline anchor — for the BBB ≤ 6pp / other ≤ 2pp deltas)
    - .planning/REQUIREMENTS.md GATE-01 rows (Claude + cross-model)
    - BENCHMARK_TABOO.md (Universal Taboo section)
  </read_first>
  <how-to-verify>
    The executor (Claude) writes `results/ablation_results/12_03_trim1_judge/verdict.json` AND `12-03-SUMMARY.md`. The user (this checkpoint) confirms:

    1. **Verdict aggregation** (Claude does the math, user reads it):
       Open `results/ablation_results/12_03_trim1_judge/verdict.json`. Verify schema:
         `{ "trim_id": "trim1_judge", "claude": {"per_dataset": {<ds>: {"F1": float, "baseline_F1": float, "delta_F1": float}}, "macro_F1": float, "gate_pass": bool, "gate_reason": str}, "gpt54": {<same shape>, "absolute_floor": 0.8977, "gate_pass": bool, "gate_reason": str}, "gate06_probe": {"taboo_hits": int, "pass": bool}, "overall_verdict": "ACCEPT" | "REJECT" }`

    2. **GATE-01 Claude check** (claude.gate_pass):
       - claude.macro_F1 ≥ 0.93 (absolute floor)
       - For each dataset:
         - if ds == "bigbluebutton": delta_F1 ≥ -0.06 (BBB tolerance 6pp)
         - else: delta_F1 ≥ -0.02 (other-dataset tolerance 2pp)
       - claude.gate_pass = True only if ALL of the above hold.

    3. **GATE-01 cross-model check** (gpt54.gate_pass):
       - gpt54.macro_F1 ≥ 0.8977 (absolute floor T=1.0pp from 0.9077 baseline)
       - All 5 datasets reported.

    4. **GATE-06 probe** (gate06_probe.pass):
       - The trimmed rubric source text passes the project-name probe (Task 1 already enforced this).
       - The probe runs again here as a defense-in-depth check.

    5. **Overall verdict**:
       - ACCEPT iff claude.gate_pass AND gpt54.gate_pass AND gate06_probe.pass
       - REJECT otherwise; gate_reason field must explain which dataset(s) failed which arm

    6. **SUMMARY.md**: open `.planning/phases/12-trim-ablation/12-03-SUMMARY.md`. Verify it contains:
       - The verdict (ACCEPT or REJECT) prominently at the top
       - The full per-dataset FP/FN delta table (Claude + gpt-5.4)
       - For REJECT: which dataset(s) failed which arm, and explicit listing in the "rejected trims" register (which Plan 12-06 will audit and the milestone summary will report)
       - For ACCEPT: confirmation that the variant should be carried into Plan 12-06's GATE-06 + reviewer-defensibility audit, and ultimately into Plan 13-01's `s_linker13_min` union
       - References PROMPT-02 and PROMPT-01 requirements
  </how-to-verify>
  <resume-signal>Type "approved" to confirm verdict adjudication is correct; otherwise describe corrections needed (e.g., "the gpt-5.4 baseline anchor changed, recompute deltas").</resume-signal>
  <acceptance_criteria>
    - `results/ablation_results/12_03_trim1_judge/verdict.json` exists and validates against the schema.
    - `12-03-SUMMARY.md` exists; verdict explicit; FP/FN delta table present; references PROMPT-01 + PROMPT-02.
    - If REJECT: SUMMARY lists the failing arm + datasets, and explicitly NOT registered for Plan 13-01 promotion.
    - If ACCEPT: SUMMARY confirms the variant carries to Plan 12-06 audit and (subject to that audit) to Plan 13-01.
    - Plan does NOT modify the variant or rubric in response to gate failure — that is Plan 12-06's responsibility (or a subsequent Phase 12 revision) per the lineage clarification "trims must be measured, not pre-rejected; failed trims are documented, not silently rescued".
  </acceptance_criteria>
  <done>Verdict recorded, SUMMARY shipped, accept/reject decision is the canonical signal for Plan 12-06 and Plan 13-01.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| New trim variant file → SLinker13Clean monkey-patched module scope | Override pattern is in-process and confined via try/finally; reviewer-defensible |
| Harness → external Claude + gpt-5.4 APIs | Network boundary; API keys in `.env` (gitignored) |
| Variant checkpoint dir → variant-namespaced subdirectory | `_VARIANT_NAME` assertion in `_checkpoint_dir` guarantees the new variant cannot overwrite the baseline cache |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-03-01 | Tampering | accidental edits to prompts_v2.py while authoring the trimmed rubric | mitigate | Task 1 verify: `git diff --quiet prompts_v2.py` exits 0; rubric is authored as a new constant in the variant file, not by editing v2 |
| T-12-03-02 | Information disclosure | benchmark-component leakage into the trimmed rubric body | mitigate | Task 1's behavior test runs the 9-name probe (Reencoding/FreeSWITCH/.../JabRef); Task 4's verdict re-runs it; Plan 12-06 runs the full TABOO sweep |
| T-12-03-03 | Tampering | monkey-patch in `_learn_document_knowledge_enriched` leaks past try/finally (concurrent use) | mitigate | Document in the class docstring that this variant is NOT thread-safe vs the parent SLinker13Clean module scope; the ablation harness runs variants sequentially per dataset so no contention |
| T-12-03-04 | Repudiation | verdict.json doesn't record which checkpoint + rubric version produced it | mitigate | verdict.json embeds variant name, ISO timestamp, anchor sources (claude baseline fixture path + gpt-5.4 baseline.json path) |
| T-12-03-05 | Denial of service | downstream Plan 13-01 promotes a rubric that secretly leaks benchmark terms | mitigate | Plan 12-06 is the gate; this plan emits accept verdict only as a precondition, never as final approval |
</threat_model>

<verification>
- Variant + trimmed rubric authored, all coverage / structure / benchmark-component guards in place.
- 5 Claude + 5 gpt-5.4 per-dataset result JSONs exist with valid F1.
- verdict.json validates against schema; overall_verdict is ACCEPT or REJECT with explicit reasoning.
- SUMMARY shipped with FP/FN delta tables.
- Zero edits to v2.0 frozen files or to s_linker13_clean.py.
- GATE-02 unaffected (variant snapshotted under `missing` in baseline fixture).
</verification>

<success_criteria>
- PROMPT-02 progressed for Step 1: trim variant designed, ablated on both backends, verdict recorded.
- PROMPT-01 progressed: the v3 rubric joins the v2→v3 mapping table maintained by Plan 12-01 + 12-06.
- Plan 12-06 inherits this trim's rubric + variant file as a primary audit target.
- Plan 13-01 inherits the accept/reject signal: if ACCEPT, carry into `s_linker13_min` union; if REJECT, document in milestone summary's rejected-trims register.
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-03-SUMMARY.md`.
</output>
</content>
</invoke>