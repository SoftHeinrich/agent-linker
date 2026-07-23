---
phase: 12-trim-ablation
plan: 05
type: execute
wave: 2
depends_on: [00, 01, 02]
files_modified:
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py
  - run_ablation.py
  - tests/test_s_linker13_trim3_runtime_rubric_registration.py
  - results/ablation_results/12_05_trim3_runtime_rubric/
  - .planning/phases/12-trim-ablation/12-05-SUMMARY.md
autonomous: false
requirements: [PROMPT-01, PROMPT-02]
must_haves:
  truths:
    - "Variant s_linker13_trim3_runtime_rubric_clean exists, importable, registered with canonical=False"
    - "The variant replaces the STATIC DOC_KNOWLEDGE_JUDGE_RULES with an INFERENCE-TIME RUBRIC BUILDER (per supplement Techniques 2 + 3 — AHE + Agentic Rubrics): a small LLM call generates a 4-6-item rubric from a generic SE-textbook seed example AND the architecture document, then the generated rubric flows into the actual judge call"
    - "The rubric builder seed example is from a safe SE-textbook domain (parser/scheduler/dispatcher/broker), NOT a benchmark project; GATE-06 probe is clean"
    - "The 7 worked examples in DOC_KNOWLEDGE_JUDGE_EXAMPLES are PRESERVED verbatim (Plan 12-03's V35a guard transfers here)"
    - "Single-step ablation invoked via the 12-02 harness re-running layer1 + cascading downstream"
    - "Claude Sonnet × 5 datasets + gpt-5.4 × 5 datasets ablation completed"
    - "Accept/reject verdict recorded against GATE-01 Claude + GATE-01 cross-model + GATE-06 spot probe"
    - "Risk acknowledged: this trim has the highest design risk in Phase 12 because the pattern is new to this codebase (inference-time rubric is not previously deployed in s_linker13)"
  artifacts:
    - path: "src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py"
      provides: "Variant overriding _learn_document_knowledge_enriched to insert a rubric-builder LLM call before the judge call"
      exports: ["SLinker13Trim3RuntimeRubricClean", "RUBRIC_BUILDER_PROMPT", "RUBRIC_BUILDER_SEED_EXAMPLE", "DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3"]
    - path: "results/ablation_results/12_05_trim3_runtime_rubric/claude/<dataset>/layer1.json"
    - path: "results/ablation_results/12_05_trim3_runtime_rubric/gpt54/<dataset>/layer1.json"
    - path: "results/ablation_results/12_05_trim3_runtime_rubric/verdict.json"
      provides: "PASS/FAIL verdict against GATE-01 + GATE-06"
      contains: "claude_macro_F1, gpt54_macro_F1, claude_gate_pass, gpt54_gate_pass, overall_verdict, risk_notes"
  key_links:
    - from: "src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py"
      to: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      via: "subclass — overrides _learn_document_knowledge_enriched to inject rubric-builder LLM call"
      pattern: "class SLinker13Trim3RuntimeRubricClean\\(SLinker13Clean"
    - from: "rubric builder call"
      to: "self.llm.query"
      via: "LLM call number +1 per dataset (vs baseline) — builds the per-document rubric"
      pattern: "self\\.llm\\.query"
---

<objective>
Implement Step 3 from the Phase 11 survey SUPPLEMENT §4 item 1: produce trim variant `s_linker13_trim3_runtime_rubric_clean` that replaces the STATIC `DOC_KNOWLEDGE_JUDGE_RULES` rubric with an INFERENCE-TIME rubric generator. Mechanism (per supplement Techniques 2 + 3, sourced from AHE arXiv 2604.25850 and Agentic Rubrics arXiv 2601.04171):

  1. Before the judge call, run a small "rubric builder" LLM call.
  2. The rubric builder consumes (a) a GENERIC SE-textbook seed example (e.g., a parser/scheduler system — never a benchmark project), (b) the architecture document, and (c) the list of candidate aliases to be judged.
  3. The rubric builder emits a 4-6-item project-grounded rubric tailored to the current document.
  4. The generated rubric is then injected into the judge prompt IN PLACE OF the static `DOC_KNOWLEDGE_JUDGE_RULES`.

This pattern is hypothesized to break the V35 ceiling (per supplement §3 cross-cutting theme 2: "generate the rubric, don't write it — information density is preserved via regeneration rather than retention"). The 7 worked examples in `DOC_KNOWLEDGE_JUDGE_EXAMPLES` are preserved verbatim (V35a guard transfers from Plan 12-03 — example removal regresses Claude).

Ablate via 12-02 harness on Claude Sonnet × 5 datasets and gpt-5.4 × 5 datasets, re-running phase=layer1. Accept iff both GATE-01 arms hold and GATE-06 probe is clean; reject and document otherwise.

Purpose: tests the strongest theoretical V35-escape mechanism identified in the survey supplement (Papers 4 + 5). This is the HIGHEST-RISK trim in Phase 12 because the pattern is new to this codebase (the rubric builder adds one LLM call to the layer1 budget) and depends on the rubric builder NOT introducing benchmark-derived phrasing through over-fitting to the document.

Output: standalone variant file, registered; 10 ablation result JSONs (5 datasets × 2 backends); verdict.json with risk_notes section; SUMMARY with FP/FN delta table + rubric-call cost analysis.
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
@.planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md
@BENCHMARK_TABOO.md
@src/llm_sad_sam/linkers/experimental/prompts_v2.py
@src/llm_sad_sam/linkers/experimental/s_linker13_clean.py

<interfaces>
<!-- The exact prompts and code path being modified — extracted from prompts_v2.py + s_linker13_clean.py at planning time -->

From prompts_v2.py lines 87-121 — DOC_KNOWLEDGE_JUDGE_EXAMPLES (7 worked examples). KEEP VERBATIM.
From prompts_v2.py lines 124-139 — DOC_KNOWLEDGE_JUDGE_RULES — this is what the runtime rubric REPLACES.

From s_linker13_clean.py lines 366-466 — _learn_document_knowledge_enriched is the override target. The current method:
  - Calls prompt1 (extraction) to get candidate aliases.
  - Calls prompt2 (judge) embedding DOC_KNOWLEDGE_JUDGE_EXAMPLES + DOC_KNOWLEDGE_JUDGE_RULES.

The new method inserts a rubric-builder call BETWEEN prompt1 and prompt2:
  - Step 1: prompt1 → all_mappings (same as before).
  - Step 2 (NEW): rubric_builder_prompt → generated_rubric (4-6 lines of bullet-style criteria).
  - Step 3: prompt2 modified — embeds DOC_KNOWLEDGE_JUDGE_EXAMPLES + generated_rubric (NOT the static DOC_KNOWLEDGE_JUDGE_RULES).

From .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md §3 cross-cutting theme 2 — "generate the rubric, don't write it".
From .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md §4 item 1 — concrete prescription targeting DOC_KNOWLEDGE_JUDGE_RULES.
From .planning/research/PROMPT-HARNESS-SURVEY.md §6 — V35c failure mode (concrete JSON output examples bias distribution). The runtime rubric is NOT an output example; it is a model-generated rubric used as model INPUT in the next call. The V35c risk pattern does NOT apply here.

From BENCHMARK_TABOO.md — both the seed example AND the rubric-builder prompt body MUST be project-agnostic; the document text passed in is project-specific but is INPUT, not part of the prompt template.

From .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md — layer1 modification cascades to layer2/entity_*/final.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Author the rubric builder + variant override + register it</name>
  <files>
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py
    - run_ablation.py
    - tests/test_s_linker13_trim3_runtime_rubric_registration.py
  </files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/prompts_v2.py lines 87-139 (DOC_KNOWLEDGE_JUDGE_EXAMPLES + DOC_KNOWLEDGE_JUDGE_RULES — the original surface)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 366-466 (_learn_document_knowledge_enriched — the method body to override)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py line 136 (_VARIANT_NAME)
    - .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md §3 + §4 (the mechanism and prescription)
    - .planning/research/PROMPT-HARNESS-SURVEY.md §6 (V35c failure mode — what the runtime rubric MUST NOT do)
    - BENCHMARK_TABOO.md (lexical surface)
    - run_ablation.py lines 40-87 (CANONICAL_VARIANTS), 324-330 (spec pattern)
  </read_first>
  <behavior>
    - Test: `from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import SLinker13Trim3RuntimeRubricClean, RUBRIC_BUILDER_PROMPT, RUBRIC_BUILDER_SEED_EXAMPLE, DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3` succeeds.
    - Test: `SLinker13Trim3RuntimeRubricClean._VARIANT_NAME == "s_linker13_trim3_runtime_rubric_clean"`.
    - Test: `SLinker13Trim3RuntimeRubricClean` is a subclass of `SLinker13Clean`.
    - Test: `DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 == prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES` (byte-equal, V35a guard).
    - Test: `RUBRIC_BUILDER_SEED_EXAMPLE` mentions only safe-SE terms — the regex check is `re.search(r'(?i)\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|AudioWatermarking|MediaManagement|WebUI|Recommender|Persistence|SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|bibentry)\b', RUBRIC_BUILDER_SEED_EXAMPLE)` returns None.
    - Test: `RUBRIC_BUILDER_PROMPT` contains the substring "4-6" (the target rubric size, per supplement §4 item 1) AND contains a placeholder for the document text (e.g., `{document_text}`) AND a placeholder for the candidate mappings (e.g., `{candidate_mappings}`) AND a placeholder for the seed example (e.g., `{seed_example}`).
    - Test: `RUBRIC_BUILDER_PROMPT` itself (the template) contains zero benchmark-component leakage probes (same 9-name regex).
    - Test: `RUBRIC_BUILDER_PROMPT` does NOT contain a JSON output template that pre-biases the rubric's content (e.g., no template line like `{"rules": ["rule1", "rule2"]}` with example content). It MAY define the output shape (e.g., a list of strings) but example content MUST be empty/abstract.
    - Test: `s_linker13_trim3_runtime_rubric_clean` registered in CANONICAL_VARIANTS and VARIANT_SPECS with `canonical=False`.
    - Test: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
  </behavior>
  <action>
    Author three module-level constants in `s_linker13_trim3_runtime_rubric_clean.py`:

      RUBRIC_BUILDER_SEED_EXAMPLE = (
          'EXAMPLE (a generic compiler-style system, for reference shape only — NOT '
          'the project you will analyze):\n'
          'Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n'
          'Candidate mappings to judge:\n'
          '  "AST" -> AbstractSyntaxTree (abbrev)\n'
          '  "Table" -> SymbolTable (synonym)\n'
          '  "the generator" -> CodeGenerator (synonym)\n'
          'A good 5-item rubric for this example would be:\n'
          '  - Approve abbreviations whose letters appear in the component name\n'
          '  - Approve trailing words of multi-word component names when unambiguous\n'
          '  - Approve descriptive phrases that consistently refer to one component\n'
          '  - Reject ordinary words used in their dictionary sense, not as a name\n'
          '  - When uncertain, prefer approve — downstream filters catch false approvals\n'
          'The rubric above is illustrative; build YOUR rubric from the project document below.'
      )

      RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for judging whether candidate alias mappings are valid for the architecture components in this specific document.

{seed_example}

PROJECT DOCUMENT:
{document_text}

CANDIDATE MAPPINGS (to be judged later, not now):
{candidate_mappings}

Produce a 4-6 item rubric grounded in patterns the document actually uses. Cover both approval criteria (when an alias clearly refers to a component) and rejection criteria (when a term is too generic). Do NOT pre-decide any mapping — the rubric is the decision criteria, not the decisions themselves.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""

      DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 = prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES  # byte-equal, V35a guard

    Implementation strategy: override `_learn_document_knowledge_enriched` end-to-end in the subclass. Copy the parent method body (lines 366-466) into the subclass, then insert the rubric-builder call between prompt1 and prompt2, and replace `{DOC_KNOWLEDGE_JUDGE_RULES}` in prompt2 with `{generated_rubric}`. The parent method is ~100 lines so copying is acceptable here (one-method fork rather than the parent-monkey-patch pattern used in Plans 12-03/04, because this trim INSERTS a new call rather than swapping a constant).

    Pseudocode of the new method body:

      def _learn_document_knowledge_enriched(self, sentences, components):
          comp_names = [c.name for c in components]
          doc_lines = [s.text for s in sentences]
          # ── Step 1: extraction (unchanged from parent) ──
          # ... copy prompt1 + data1 + all_mappings extraction logic verbatim ...

          if not all_mappings:
              return DocumentKnowledge()  # nothing to judge

          # ── Step 2: NEW — build per-document rubric ──
          mapping_list = [f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]]
          rubric_prompt = RUBRIC_BUILDER_PROMPT.format(
              seed_example=RUBRIC_BUILDER_SEED_EXAMPLE,
              document_text=chr(10).join(doc_lines),
              candidate_mappings=chr(10).join(mapping_list),
          )
          for attempt in range(2):
              rubric_data = self.llm.extract_json(self.llm.query(rubric_prompt, timeout=180))
              if rubric_data and rubric_data.get("rubric"):
                  break
              if attempt == 0:
                  print("    Rubric builder: empty response, retrying...")
          # Fallback if rubric builder fails: degrade to the static parent rubric so the trim never silently drops downstream
          if rubric_data and rubric_data.get("rubric"):
              generated_rubric_lines = rubric_data["rubric"]
              if not isinstance(generated_rubric_lines, list):
                  generated_rubric_lines = []
          else:
              from llm_sad_sam.linkers.experimental.prompts_v2 import DOC_KNOWLEDGE_JUDGE_RULES
              # Fallback path — log it for the SUMMARY's risk_notes section
              print("    Rubric builder fell back to static parent rubric")
              generated_rubric = DOC_KNOWLEDGE_JUDGE_RULES
              fallback_used = True
          else:
              generated_rubric = "DECISION RUBRIC (generated for this document):\n" + "\n".join(f"- {r}" for r in generated_rubric_lines)
              fallback_used = False

          # ── Step 3: judge — same structure as parent, but with generated rubric ──
          prompt2 = f\"\"\"JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3}

{generated_rubric}

Return JSON:
{{{{"approved": ["term1", "term2"]}}}}
JSON only:\"\"\"
          # ... rest of method body (judge call, approval filtering, alias entry creation) — unchanged from parent ...

    NOTE: the subclass `_learn_document_knowledge_enriched` records whether the fallback was used (e.g., a self._trim3_fallback_count counter incremented per fallback). This is surfaced in Task 4's verdict.json risk_notes.

    Append to run_ablation.py CANONICAL_VARIANTS and VARIANT_SPECS:
      - description: "S-Linker13 Trim3 — Phase 12 Step 3: DOC_KNOWLEDGE_JUDGE_RULES replaced by inference-time rubric builder (AHE + Agentic Rubrics mechanism — supplement Techniques 2+3); 7 worked examples preserved verbatim"

    Create `tests/test_s_linker13_trim3_runtime_rubric_registration.py` testing all behaviors above.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; pytest tests/test_s_linker13_trim3_runtime_rubric_registration.py -x -q &amp;&amp; python -c "from llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean import SLinker13Trim3RuntimeRubricClean, RUBRIC_BUILDER_PROMPT, RUBRIC_BUILDER_SEED_EXAMPLE, DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3; from llm_sad_sam.linkers.experimental import prompts_v2; assert DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 == prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES; import re; assert not re.search(r'(?i)\\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\\b', RUBRIC_BUILDER_SEED_EXAMPLE); assert not re.search(r'(?i)\\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\\b', RUBRIC_BUILDER_PROMPT); assert '4-6' in RUBRIC_BUILDER_PROMPT; assert '{document_text}' in RUBRIC_BUILDER_PROMPT; assert '{candidate_mappings}' in RUBRIC_BUILDER_PROMPT; assert '{seed_example}' in RUBRIC_BUILDER_PROMPT" &amp;&amp; python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS; assert 's_linker13_trim3_runtime_rubric_clean' in CANONICAL_VARIANTS; assert VARIANT_SPECS['s_linker13_trim3_runtime_rubric_clean']['canonical'] is False" &amp;&amp; git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py</automated>
  </verify>
  <acceptance_criteria>
    - All registration-test assertions pass.
    - RUBRIC_BUILDER_SEED_EXAMPLE uses only safe-SE terms (compiler-style example).
    - RUBRIC_BUILDER_PROMPT has the three required placeholders and the "4-6 item" target.
    - GATE-06 spot-probe clean on both the seed example and the prompt template.
    - 7 worked examples preserved byte-equal.
    - GATE-02 unaffected — variant registered under `missing` in baseline fixture; `pytest tests/test_v20_baseline_regression.py -q` exits 0.
    - Zero edits to v2.0 frozen files or to s_linker13_clean.py.
  </acceptance_criteria>
  <done>Variant + rubric builder authored, registered, coverage tests green.</done>
</task>

<task type="auto">
  <name>Task 2: Run Claude Sonnet single-step ablation × 5 datasets via 12-02 harness</name>
  <files>
    - results/ablation_results/12_05_trim3_runtime_rubric/claude/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/phase_cache/s_linker13_trim3_runtime_rubric_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/
  </files>
  <read_first>
    - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md (layer1 → cascade)
    - results/phase_cache/s_linker13_clean/ (no upstream needed — layer1 is first phase; harness validates this)
    - src/llm_sad_sam/ablation/single_step.py
    - .planning/REQUIREMENTS.md GATE-01 Claude arm
  </read_first>
  <action>
    For each dataset in {mediastore, teastore, teammates, bigbluebutton, jabref} run sequentially:
      `LLM_BACKEND=claude CLAUDE_MODEL=sonnet PHASE_CACHE_DIR=results/phase_cache python -m llm_sad_sam.ablation single_step --variant s_linker13_trim3_runtime_rubric_clean --dataset <ds> --phase layer1 --results-dir results/ablation_results/12_05_trim3_runtime_rubric/claude --backend claude`

    This invocation runs layer1 fresh (adds the rubric-builder LLM call to the budget — +1 call per dataset, accepted per CLAUDE.md "no LLM budget limit"), then cascades through layer2, entity_candidates, entity_decisions, final. Each result JSON should record whether the rubric-builder fallback was triggered (if the variant's instance carries a `_trim3_fallback_count` attribute, the harness can read it via a known accessor; alternatively, the variant logs to stdout and Task 4 parses the sweep.log).

    Stream stdout to `results/ablation_results/12_05_trim3_runtime_rubric/claude/sweep.log` so Task 4 can audit rubric-builder fallback rates and inspect generated rubric content for taboo terms.

    Per-dataset transient failure: retry the single dataset. Do not skip.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; for ds in mediastore teastore teammates bigbluebutton jabref; do test -f "results/ablation_results/12_05_trim3_runtime_rubric/claude/$ds/layer1.json" || { echo "MISSING $ds"; exit 1; }; done &amp;&amp; python -c "import json; rows=[json.load(open(f'results/ablation_results/12_05_trim3_runtime_rubric/claude/{ds}/layer1.json')) for ds in ['mediastore','teastore','teammates','bigbluebutton','jabref']]; macro = sum(r['F1'] for r in rows)/5; print(f'Claude macro F1 = {macro:.4f}'); assert all(0 &lt;= r['F1'] &lt;= 1 for r in rows)"</automated>
  </verify>
  <acceptance_criteria>
    - All 5 Claude per-dataset result JSONs exist with valid F1.
    - Macro F1 printed.
    - Per-variant cache populated at `results/phase_cache/s_linker13_trim3_runtime_rubric_clean/<ds>/`.
    - sweep.log captured for fallback-rate analysis in Task 4.
    - No edits to v2.0 frozen files or s_linker13_clean.py.
  </acceptance_criteria>
  <done>Claude × 5 datasets done.</done>
</task>

<task type="auto">
  <name>Task 3: Run gpt-5.4 single-step ablation × 5 datasets via 12-02 harness</name>
  <files>
    - results/ablation_results/12_05_trim3_runtime_rubric/gpt54/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/phase_cache_gpt54/s_linker13_trim3_runtime_rubric_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/
  </files>
  <read_first>
    - results/ablation_results/12_00_gpt54_baseline/baseline.json (anchor)
    - .planning/REQUIREMENTS.md GATE-01 cross-model arm
  </read_first>
  <action>
    For each dataset run sequentially:
      `LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 PHASE_CACHE_DIR=results/phase_cache_gpt54 python -m llm_sad_sam.ablation single_step --variant s_linker13_trim3_runtime_rubric_clean --dataset <ds> --phase layer1 --results-dir results/ablation_results/12_05_trim3_runtime_rubric/gpt54 --backend openai --model gpt-5.4`

    Stream stdout to `results/ablation_results/12_05_trim3_runtime_rubric/gpt54/sweep.log`.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; for ds in mediastore teastore teammates bigbluebutton jabref; do test -f "results/ablation_results/12_05_trim3_runtime_rubric/gpt54/$ds/layer1.json" || { echo "MISSING $ds"; exit 1; }; done &amp;&amp; python -c "import json; rows=[json.load(open(f'results/ablation_results/12_05_trim3_runtime_rubric/gpt54/{ds}/layer1.json')) for ds in ['mediastore','teastore','teammates','bigbluebutton','jabref']]; macro = sum(r['F1'] for r in rows)/5; print(f'gpt-5.4 macro F1 = {macro:.4f}'); assert all(0 &lt;= r['F1'] &lt;= 1 for r in rows)"</automated>
  </verify>
  <acceptance_criteria>
    - All 5 gpt-5.4 per-dataset result JSONs exist.
    - Macro F1 printed.
    - sweep.log captured.
    - No edits to v2.0 frozen files or s_linker13_clean.py.
  </acceptance_criteria>
  <done>gpt-5.4 × 5 datasets done.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Adjudicate verdict + audit generated rubrics for taboo + write SUMMARY</name>
  <what-built>10 per-dataset ablation JSONs + verdict.json aggregating against GATE-01 Claude + GATE-01 cross-model + GATE-06 spot probe + an EXTRA audit: every rubric the builder generated across all 10 runs is grep-checked for benchmark-component leakage (the runtime rubric is the unique risk surface in this trim — the builder COULD emit a taboo term derived from the input document).</what-built>
  <read_first>
    - results/ablation_results/12_05_trim3_runtime_rubric/claude/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/ablation_results/12_05_trim3_runtime_rubric/gpt54/{mediastore,teastore,teammates,bigbluebutton,jabref}/layer1.json
    - results/ablation_results/12_05_trim3_runtime_rubric/claude/sweep.log
    - results/ablation_results/12_05_trim3_runtime_rubric/gpt54/sweep.log
    - results/ablation_results/12_00_gpt54_baseline/baseline.json
    - tests/fixtures/v2_0_baseline.json
    - .planning/REQUIREMENTS.md GATE-01 rows
    - BENCHMARK_TABOO.md (full Universal Taboo section — the generated-rubric audit uses the same surface Plan 12-06 will use, not just the 9-name probe)
  </read_first>
  <how-to-verify>
    The executor writes `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` AND `12-05-SUMMARY.md`. The user confirms:

    1. **Generated-rubric audit (NEW for this trim)** — Plan 12-05 carries an extra risk: the rubric is generated AT RUNTIME from the project document, so the model COULD emit a taboo term. Audit protocol:
       - Parse each sweep.log (Claude + gpt-5.4) to extract every rubric body the builder emitted (the variant logs "DECISION RUBRIC (generated for this document):" followed by the rubric).
       - For each rubric body, run the full TABOO sweep: `grep -wEi "logic|UI|client|storage|common|model|database|DB|cache|registry|auth|server|persistence|facade|recording|cascade|conversion|validation|dedicated|preferences|config|internal|adapter|order|processor|event|socket|layer|Reencoding|FreeSWITCH|kurento|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|AudioWatermarking|MediaManagement|WebUI|Recommender|SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|bibentry"`.
       - Tally hits per (backend, dataset). Record in `verdict.json.generated_rubric_audit`.
       - If hits > 0 on ANY (backend, dataset), gate06_probe.pass = False — the rubric generator has leaked benchmark terms even though the static seed example + prompt were clean. This is a known risk of the inference-time rubric pattern and the trim is REJECTED with rationale "rubric builder leaked taboo terms in generated rubric — pattern not GATE-06-safe for this regime".

    2. **Verdict schema** (extended for Plan 12-05):
       `{ "trim_id": "trim3_runtime_rubric", "claude": {..., "fallback_count": int}, "gpt54": {..., "fallback_count": int}, "generated_rubric_audit": {"per_run": [{"backend": str, "dataset": str, "taboo_hits": int, "leaked_terms": [str]}], "total_hits": int}, "gate06_probe": {"taboo_hits": total_hits, "pass": bool}, "overall_verdict": ... , "risk_notes": str }`

    3. **GATE-01 Claude** (same as Plans 12-03/04):
       - claude.macro_F1 ≥ 0.93
       - For each ds: BBB delta ≥ -0.06, other ≥ -0.02
       - claude.gate_pass = ALL hold

    4. **GATE-01 cross-model** (same):
       - gpt54.macro_F1 ≥ 0.8977
       - All 5 datasets reported

    5. **Overall verdict**: ACCEPT iff all three (Claude gate + gpt-5.4 gate + GATE-06 generated-rubric audit) pass. REJECT otherwise.

    6. **risk_notes** section MUST capture:
       - Rubric-builder fallback rate per backend (how many runs degraded to the static parent rubric)
       - The +1 LLM call per dataset cost (10 extra calls total for the full ablation)
       - Whether the generated rubric varied significantly across datasets (sanity check — if all 5 datasets produced near-identical rubrics, the pattern is not actually adapting to the document and the trim's mechanism is suspect even if F1 passes)

    7. **SUMMARY** contains:
       - Verdict prominent at top
       - Per-dataset FP/FN delta table (Claude + gpt-5.4)
       - The generated-rubric audit summary
       - The risk_notes section
       - For ACCEPT: confirmation the variant carries to Plan 12-06 (extra-scrutiny audit required for this trim) + (subject to that audit) Plan 13-01
       - For REJECT: failing arm + datasets + which gate failed
       - References PROMPT-01 + PROMPT-02
  </how-to-verify>
  <resume-signal>Type "approved" or describe corrections needed.</resume-signal>
  <acceptance_criteria>
    - verdict.json exists, validates against schema (including the new generated_rubric_audit + risk_notes fields).
    - 12-05-SUMMARY.md exists; verdict explicit; FP/FN delta table present; generated-rubric audit summary present; risk_notes present; references PROMPT-01 + PROMPT-02.
    - If REJECT: failing reason explicit (GATE-01 Claude / GATE-01 cross-model / GATE-06 generated-rubric leakage), variant NOT carried to Plan 13-01.
    - If ACCEPT: SUMMARY notes the variant requires extra-scrutiny audit in Plan 12-06 (this is the highest-risk trim).
  </acceptance_criteria>
  <done>Verdict recorded; generated-rubric audit complete; SUMMARY shipped.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Rubric builder LLM call | NEW LLM call in layer1; receives document text as input and produces decision-criteria as output that flows back into the same LLM in the next call |
| Generated rubric → judge prompt | The rubric (model output) becomes part of the next prompt template — this is the unique GATE-06 risk of this trim |
| Variant checkpoint dir | Namespaced via `_VARIANT_NAME` (assert in `_checkpoint_dir`) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-05-01 | Information disclosure | rubric builder emits benchmark-component names from the input document into the generated rubric | mitigate | Task 4's generated-rubric audit runs the full TABOO sweep against every rubric emitted across 10 runs; hits → REJECT verdict. The RUBRIC_BUILDER_PROMPT explicitly instructs "Do NOT pre-decide any mapping" but model compliance is empirical, not guaranteed. |
| T-12-05-02 | Tampering | accidental edits to prompts_v2.py or s_linker13_clean.py while authoring the new variant | mitigate | Task 1 verify: `git diff --quiet` on all v2.0 frozen files + s_linker13_clean.py exits 0 |
| T-12-05-03 | Denial of service | rubric builder consistently fails (empty response), variant silently degrades to static parent rubric and the trim is effectively a no-op | mitigate | Variant tracks fallback_count; Task 4 verdict.risk_notes reports fallback rate; if fallback rate > 20% on any (backend, dataset) the trim's mechanism is non-functional even if F1 happens to pass — note in risk_notes for Plan 13-01 to consider |
| T-12-05-04 | Repudiation | which rubric was used in which run not recorded | mitigate | sweep.log captures every emitted rubric body (verbose stdout from variant); verdict.generated_rubric_audit ties each rubric to (backend, dataset) |
| T-12-05-05 | Information disclosure | the +1 LLM call per dataset adds ~10 calls total for the full ablation; cost is acceptable per CLAUDE.md "no LLM budget limit" but should be recorded for transparency | accept | risk_notes records the call-count delta vs baseline |
</threat_model>

<verification>
- Variant + rubric builder authored; GATE-06 clean on the static prompt + seed.
- 5 Claude + 5 gpt-5.4 per-dataset result JSONs exist.
- verdict.json validates against extended schema including generated_rubric_audit + risk_notes.
- Generated-rubric audit completed; pass/fail recorded.
- SUMMARY shipped with FP/FN delta + audit summary + risk notes.
- Zero edits to v2.0 frozen files or s_linker13_clean.py.
- GATE-02 unaffected.
</verification>

<success_criteria>
- PROMPT-02 progressed for Step 3: highest-risk trim ablated on both backends, verdict recorded WITH explicit generated-rubric leakage audit.
- PROMPT-01 progressed: DOC_KNOWLEDGE_JUDGE_RULES status in the v2→v3 mapping table updates to "replaced by inference-time rubric builder" (under Plan 12-06's mapping-doc maintenance).
- Plan 12-06 inherits this trim as PRIMARY AUDIT TARGET — highest-risk trim, requires extra reviewer scrutiny on the inference-time rubric pattern.
- Plan 13-01 inherits the accept/reject signal with the fallback-rate caveat.
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-05-SUMMARY.md`.
</output>
