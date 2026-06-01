---
phase: 10-scaffolding
plan: 03
type: execute
wave: 2
depends_on:
  - 10-02
files_modified:
  - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py
  - run_ablation.py
autonomous: false
requirements:
  - CLEAN-01
  - CLEAN-02
user_setup: []

must_haves:
  truths:
    - "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py exists as a standalone file (no inheritance from SLinker13 — duplicate code is intentional per user preference)"
    - "s_linker13_clean imports helpers from helper_v3 instead of inlining them"
    - "s_linker13_clean imports prompts from prompts_v2 unchanged (prompts_v3 lands in Phase 12)"
    - "run_ablation.CANONICAL_VARIANTS contains the string \"s_linker13_clean\" and run_ablation.VARIANT_SPECS contains the matching dict entry"
    - "On Claude Sonnet, s_linker13_clean produces F1 identical to s_linker13 on all 5 datasets (mediastore, teastore, teammates, bigbluebutton, jabref) within abs diff < 1e-4"
    - "s_linker13.py remains byte-identical (frozen)"
    - "tests/test_v20_baseline_regression.py still exits 0 after registration"
  artifacts:
    - path: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      provides: "Standalone _clean variant: thin orchestration class importing helper_v3 helpers + prompts_v2 prompts"
      contains: "class SLinker13Clean"
      min_lines: 200
    - path: "run_ablation.py"
      provides: "Registration of s_linker13_clean in CANONICAL_VARIANTS list and VARIANT_SPECS dict"
      contains: "s_linker13_clean"
  key_links:
    - from: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      to: "src/llm_sad_sam/linkers/experimental/helper_v3.py"
      via: "module-level import of coerce_mention_type, format_mention_string, has_standalone_mention, build_component_profile, parse_snum, get_comp_names"
      pattern: "from llm_sad_sam\\.linkers\\.experimental\\.helper_v3 import"
    - from: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      to: "src/llm_sad_sam/linkers/experimental/prompts_v2.py"
      via: "module-level import of prompt constants"
      pattern: "from llm_sad_sam\\.linkers\\.experimental\\.prompts_v2 import"
    - from: "run_ablation.py CANONICAL_VARIANTS"
      to: "src/llm_sad_sam/linkers/experimental/s_linker13_clean.py"
      via: "VARIANT_SPECS[\"s_linker13_clean\"][\"module\"]"
      pattern: "llm_sad_sam\\.linkers\\.experimental\\.s_linker13_clean"
---

<objective>
Ship the standalone s_linker13_clean variant: a thin orchestration class that imports the
extracted helpers from helper_v3 (Plan 10-02) and prompt constants from the unchanged
prompts_v2, registered in CANONICAL_VARIANTS + VARIANT_SPECS, and verified to produce
identical F1 to s_linker13 on all 5 datasets via Claude Sonnet.

Purpose: CLEAN-01 + CLEAN-02 — this is the foundation variant the v2.1 trim / prompts_v3
chain (Phases 12-13) builds on. Nothing currently runnable breaks: s_linker13 stays frozen.

Output:
  1. src/llm_sad_sam/linkers/experimental/s_linker13_clean.py — class SLinker13Clean
  2. run_ablation.py — CANONICAL_VARIANTS list updated + VARIANT_SPECS dict entry added
  3. A parity sweep result demonstrating identical F1 to s_linker13 on all 5 datasets
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/10-scaffolding/10-CONTEXT.md
@src/llm_sad_sam/linkers/experimental/s_linker13.py
@src/llm_sad_sam/linkers/experimental/prompts_v2.py
@run_ablation.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create standalone s_linker13_clean.py</name>
  <files>src/llm_sad_sam/linkers/experimental/s_linker13_clean.py</files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/s_linker13.py (FULL file — this is the byte-for-byte source the _clean variant copies from, then refactors to import helpers from helper_v3 instead of inlining)
    - src/llm_sad_sam/linkers/experimental/helper_v3.py (just produced by Plan 10-02 — read it to know the exact import surface)
    - src/llm_sad_sam/linkers/experimental/prompts_v2.py (frozen — read the top of the file to confirm exported prompt constant names: AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES — these are imported by s_linker13.py at lines 55-60)
    - src/llm_sad_sam/core/data_types_v2.py (frozen — used for SadSamLink, CandidateLink, ModelKnowledge, DocumentKnowledge, EvidenceBundle)
    - src/llm_sad_sam/core/document_loader_v2.py (frozen — used for Sentence, load_sentences, build_sent_map)
    - src/llm_sad_sam/linkers/experimental/ilinker3.py (frozen — s_linker13 imports ILinker3 as the seed; _clean inherits this dependency unchanged)
    - .planning/phases/10-scaffolding/10-CONTEXT.md (Decisions block: "duplicate code is fine per user pref" — confirm standalone class, NOT subclass)
    - run_ablation.py lines 40-360 (registration patterns)
  </read_first>
  <behavior>
    - Behavior 1 (standalone file, no inheritance): class SLinker13Clean does NOT subclass SLinker13. The file is a self-contained copy of s_linker13.py with the helper bodies removed and replaced by helper_v3 imports. Duplicate code preserved at all other layers.
    - Behavior 2 (frozen imports): prompts_v2 imports unchanged from s_linker13 (same eight constants). data_types_v2 and document_loader_v2 imports unchanged. ILinker3 import unchanged.
    - Behavior 3 (helper imports): the six required helper names (coerce_mention_type, format_mention_string, has_standalone_mention, build_component_profile, parse_snum, get_comp_names) come from helper_v3 — NOT defined inside SLinker13Clean.
    - Behavior 4 (parity contract): for any (text_path, model_path) input, SLinker13Clean.link(...) returns a SadSamLink list semantically equivalent (same set of (sentence, component) pairs and confidences within float tolerance) to SLinker13.link(...) for the same input on Claude Sonnet.
    - Behavior 5 (no semantic changes): the _clean variant does NOT trim any rule, prompt, or heuristic. Trim work is Phase 12's job. _clean is a structural refactor only.
    - Behavior 6 (GATE-06 clean): no benchmark component names anywhere in the file (greppable forbidden tokens listed in acceptance).
    - Behavior 7 (GATE-07 registration discipline): structured docstring at top of file with REMOVED_FROM, RULES_REMOVED (empty for _clean — no rules removed yet), KEEP, and a "scaffolding sibling of s_linker13" provenance line.
  </behavior>
  <action>
    Create src/llm_sad_sam/linkers/experimental/s_linker13_clean.py as a standalone file derived from s_linker13.py.

    Recipe (do each step explicitly — do not summarize):

    1. Copy the full body of src/llm_sad_sam/linkers/experimental/s_linker13.py.

    2. Replace the module docstring with one matching this skeleton (fill in details):
       """S-Linker13 Clean: structural cleanup sibling of s_linker13 (Phase 10, v2.1).

       REMOVED_FROM: s_linker13 (structural refactor only — zero rules removed)
       RULES_REMOVED: []
       KEEP: ["_has_standalone_mention (Spike 002 RISKY; unchanged from s_linker13; EXT-01/EXT-02 still deferred)"]
       CLEAN: ["helpers extracted to helper_v3", "prompt imports unchanged from prompts_v2 (prompts_v3 lands in Phase 12)"]

       s_linker13_clean is the structural cleanup foundation for the v2.1 trim chain
       (Phase 12). On Claude Sonnet it produces F1 identical to s_linker13 on all 5
       datasets — this is the CLEAN-01 acceptance contract. Frozen siblings (s_linker13,
       prompts_v2, data_types_v2, document_loader_v2, pcm_parser_v2) are NOT touched.
       """

    3. Rename the class from SLinker13 to SLinker13Clean. Do NOT subclass — keep it standalone.

    4. Remove the inlined definitions of:
         _build_component_profile, _parse_snum, _has_standalone_mention, _get_comp_names,
         _coerce_mention_type (if present), _format_mention_string (if present)
       Replace every call site inside the class:
         self._build_component_profile(x)        ->  build_component_profile(x, self.components, self.knowledge)  # parameter list per the actual helper_v3 signature decided in Plan 10-02
         self._parse_snum(v)                     ->  parse_snum(v)
         self._has_standalone_mention(c, t)      ->  has_standalone_mention(c, t)
         self._get_comp_names(c)                 ->  get_comp_names(c)
         self._coerce_mention_type(v)            ->  coerce_mention_type(v)
         self._format_mention_string(mt, alias)  ->  format_mention_string(mt, alias)
       Add the matching import at the top of the file:
         from llm_sad_sam.linkers.experimental.helper_v3 import (
             coerce_mention_type,
             format_mention_string,
             has_standalone_mention,
             build_component_profile,
             parse_snum,
             get_comp_names,
         )

    5. Leave every other piece of code byte-identical to s_linker13.py: link() main loop, _run_seed, _run_seed_validation, _run_entity_pipeline, _run_coreference, _extract_entities_enriched, _validate_with_evidence, _coref_cases_in_context, checkpoint/log infra, dataclasses. Do NOT trim, simplify, or "improve" anything. That is Phase 12's job.

    6. Update the class's checkpoint_dir convention so its on-disk cache cannot collide with s_linker13's: if s_linker13 writes to results/phase_cache/<text_path_stem>/, s_linker13_clean writes to results/phase_cache/<text_path_stem>__clean/ (or whatever the existing variant convention is — match the established suffix pattern other variants use; verify by reading _checkpoint_dir in s_linker13.py).

    7. Do NOT add any new heuristic, prompt, or filter. Do NOT remove any heuristic, prompt, or filter. The _clean variant is a structural identity transform modulo the helper extraction.

    8. Confirm GATE-06: grep -E "Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper" src/llm_sad_sam/linkers/experimental/s_linker13_clean.py must return nothing. Note that Kafka Broker / Nginx Proxy / Zookeeper appear in prompts_v2 (which is FROZEN and imported, not duplicated here) — they must not be re-pasted into s_linker13_clean.py itself.

    9. Confirm frozen-file untouched: s_linker13.py, prompts_v2.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py must show 0 git-diff changes after this task.

    Do NOT:
      - Subclass SLinker13 (standalone-file rule)
      - Edit s_linker13.py or any frozen module
      - Add prompts_v3 imports (does not exist yet — lands in Phase 12)
      - Trim any rule (Phase 12 territory)
      - Change LLM backend defaults (Claude Sonnet remains the default — never opus)
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && python -c "from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean; import inspect, llm_sad_sam.linkers.experimental.s_linker13_clean as m; src=inspect.getsource(m); leaks=[w for w in ['Reencoding','FreeSWITCH','kurento','Recording Service','Redis PubSub','HTML5 Server','Nginx Proxy','Kafka Broker','Zookeeper'] if w in src]; assert not leaks, leaks; assert 'from llm_sad_sam.linkers.experimental.helper_v3 import' in src, 'must import from helper_v3'; assert 'from llm_sad_sam.linkers.experimental.prompts_v2 import' in src, 'must import from prompts_v2 (frozen)'; assert SLinker13Clean.__bases__ == (object,), f'must be standalone (no subclass), got bases {SLinker13Clean.__bases__}'; print('s_linker13_clean OK')" && git diff --quiet -- src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py && echo "frozen files untouched"</automated>
  </verify>
  <acceptance_criteria>
    - File src/llm_sad_sam/linkers/experimental/s_linker13_clean.py exists
    - `python -c "from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean"` exits 0
    - class SLinker13Clean has __bases__ == (object,) — standalone, not subclassed
    - File imports from llm_sad_sam.linkers.experimental.helper_v3 (the six helper names)
    - File imports from llm_sad_sam.linkers.experimental.prompts_v2 (same eight constants s_linker13 imports)
    - File does NOT redefine any of: _build_component_profile, _parse_snum, _has_standalone_mention, _get_comp_names, _coerce_mention_type, _format_mention_string (those now live in helper_v3)
    - Module docstring contains: "REMOVED_FROM: s_linker13", "RULES_REMOVED: []", "KEEP:", "CLEAN:"
    - `grep -E "Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper" src/llm_sad_sam/linkers/experimental/s_linker13_clean.py` returns nothing
    - `git diff --quiet -- src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py` exits 0
    - Checkpoint directory convention is distinct from s_linker13's (so caches don't collide)
  </acceptance_criteria>
  <done>
    Standalone SLinker13Clean class importable, helper imports wired to helper_v3, prompts_v2 imports unchanged, frozen files untouched, GATE-06 clean.
  </done>
</task>

<task type="auto">
  <name>Task 2: Register s_linker13_clean in run_ablation.py</name>
  <files>run_ablation.py</files>
  <read_first>
    - run_ablation.py lines 40-86 (CANONICAL_VARIANTS — locate the s_linker13 entry at line 79; insert s_linker13_clean directly after it so it groups with the canonical 13-family)
    - run_ablation.py lines 316-322 (existing s_linker13 VARIANT_SPECS entry — use it as the structural template; note canonical=True is set there and indicates a promoted variant)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py (just created — confirms module path and class name to register)
    - .planning/phases/10-scaffolding/10-CONTEXT.md (CLEAN-01 wording: "Standalone s_linker13_clean.py variant ships, importable via run_ablation.py and registered in CANONICAL_VARIANTS / VARIANT_SPECS")
    - .planning/STATE.md Standing Gates (GATE-07 registration discipline)
  </read_first>
  <action>
    Edit run_ablation.py to register the new variant.

    Step 1 — CANONICAL_VARIANTS list: insert "s_linker13_clean" on its own line immediately AFTER the existing "s_linker13" line (currently line 79). Use the same indentation and trailing-comma style as the surrounding entries. Add a short inline comment: `# v2.1 scaffolding sibling of s_linker13 (Phase 10, CLEAN-01)`.

    Step 2 — VARIANT_SPECS dict: insert a new entry immediately AFTER the existing "s_linker13" entry (currently ending near line 322). Exact entry:
      "s_linker13_clean": dict(
          aliases=(),
          module="llm_sad_sam.linkers.experimental.s_linker13_clean",
          class_name="SLinker13Clean",
          description="S-Linker13 Clean: v2.1 scaffolding sibling — helpers in helper_v3, prompts_v2 unchanged, zero rules removed (CLEAN-01).",
          canonical=False,
      ),
    canonical=False is correct at scaffolding time (no promotion yet). Phase 13 (PROMPT-03) may flip canonical on a downstream _min variant if both gates hold; that decision is deferred.

    Step 3 — verify the dict-to-list reconciliation block at the bottom of the module still works: the line `VARIANTS = {canonical: ... for canonical in CANONICAL_VARIANTS}` requires every CANONICAL_VARIANTS entry to be present as a key in VARIANT_SPECS. Adding s_linker13_clean to both satisfies this.

    Do NOT:
      - Remove or modify any existing CANONICAL_VARIANTS entry
      - Reorder existing entries
      - Touch any other section of run_ablation.py (argparse, sweep loop, output paths)
      - Flip canonical=True on the new entry — s_linker13 stays the only canonical=True 13-family entry until Phase 13 decides otherwise
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS, VARIANTS; assert 's_linker13_clean' in CANONICAL_VARIANTS, 'missing from CANONICAL_VARIANTS'; assert 's_linker13_clean' in VARIANT_SPECS, 'missing from VARIANT_SPECS'; spec=VARIANT_SPECS['s_linker13_clean']; assert spec['module']=='llm_sad_sam.linkers.experimental.s_linker13_clean', spec; assert spec['class_name']=='SLinker13Clean', spec; assert spec.get('canonical', False) is False, 'canonical must be False at scaffolding time'; assert 's_linker13_clean' in VARIANTS, 'VARIANTS reconciliation broke'; assert VARIANTS['s_linker13']['canonical']=='s_linker13', 's_linker13 registration must be untouched'; print('registration OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS"` exits 0
    - "s_linker13_clean" appears in CANONICAL_VARIANTS list
    - VARIANT_SPECS["s_linker13_clean"]["module"] == "llm_sad_sam.linkers.experimental.s_linker13_clean"
    - VARIANT_SPECS["s_linker13_clean"]["class_name"] == "SLinker13Clean"
    - VARIANT_SPECS["s_linker13_clean"]["canonical"] is False (or key absent — equivalent)
    - VARIANT_SPECS["s_linker13"] is byte-identical to its pre-edit state (canonical=True preserved)
    - The dict-to-list reconciliation block at the bottom of run_ablation.py still resolves successfully (no KeyError raised at import)
    - `git diff run_ablation.py` shows additions only — no deletions of existing CANONICAL_VARIANTS or VARIANT_SPECS entries
  </acceptance_criteria>
  <done>
    Registration committed, importability confirmed, s_linker13 registration untouched, dict-to-list reconciliation still passes.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Run 5-dataset parity sweep s_linker13_clean vs s_linker13</name>
  <what-built>
    s_linker13_clean.py (standalone variant) registered in CANONICAL_VARIANTS / VARIANT_SPECS,
    importing helpers from helper_v3 and prompts from prompts_v2. Structural refactor only —
    zero rules removed.
  </what-built>
  <how-to-verify>
    1. Confirm Claude Sonnet is the configured backend (NOT opus). Check .env: `grep -E "CLAUDE_MODEL|claude" .env`. Default per project memory: claude-sonnet.

    2. Run the parity sweep — exact command, single line:
       `python run_ablation.py --variant s_linker13_clean s_linker13 --datasets mediastore teastore teammates bigbluebutton jabref 2>&1 | tee results/10-03-parity-sweep.log`
       (Adjust the flag names to match the actual run_ablation.py argparse surface if they differ — read run_ablation.py argparse section first; the intent is: run BOTH variants on ALL 5 datasets in one sweep so the output JSON has both for direct diff.)

    3. Locate the produced ablation JSON under results/ablation_results/ablation_<timestamp>.json — it is the most recent file.

    4. Run the parity diff:
       `python -c "import json,glob,os; latest=max(glob.glob('results/ablation_results/ablation_*.json'), key=os.path.getmtime); d=json.load(open(latest)); ds_keys=['mediastore','teastore','teammates','bigbluebutton','jabref']; diffs=[]; [diffs.append((ds,'F1',d[ds]['s_linker13']['F1'],d[ds]['s_linker13_clean']['F1'])) for ds in ds_keys if abs(d[ds]['s_linker13']['F1']-d[ds]['s_linker13_clean']['F1'])>=1e-4]; print('FILE:',latest); print('DIFFS:',diffs); assert not diffs, f'PARITY FAILED: {diffs}'; macro_orig=sum(d[ds]['s_linker13']['F1'] for ds in ds_keys)/5; macro_clean=sum(d[ds]['s_linker13_clean']['F1'] for ds in ds_keys)/5; print(f'macro s_linker13={macro_orig:.4f} s_linker13_clean={macro_clean:.4f}'); assert abs(macro_orig-macro_clean)<1e-4"`

    5. Confirm the macro F1 anchor: s_linker13 macro must land at 0.9509 ± 5e-3 (LLM variance) and s_linker13_clean must match it within abs diff < 1e-4.

    6. Confirm tests/test_v20_baseline_regression.py still passes:
       `python -m pytest tests/test_v20_baseline_regression.py -v`

    Expected outcome (paste into resume signal):
      - Diff list empty (no dataset breaks abs-diff threshold 1e-4)
      - s_linker13 macro within 0.9504..0.9514
      - s_linker13_clean macro within 1e-4 of s_linker13 macro
      - Regression test exits 0

    If parity fails on any dataset, do NOT proceed. Report the dataset, the absolute F1 delta,
    the sources/fp_by_source/fn_details diff, and the most likely cause (helper extraction
    error, hidden self.* reference dropped, prompt import drift, etc.). The plan must be
    revised before s_linker13_clean ships.
  </how-to-verify>
  <resume-signal>
    Reply "parity-confirmed" with the produced log path + the 5 per-dataset F1 deltas + the
    macro F1 for both variants, or describe the failing dataset and observed delta.
  </resume-signal>
</task>

</tasks>

<verification>
1. `python -c "from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean; from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS; assert 's_linker13_clean' in CANONICAL_VARIANTS and 's_linker13_clean' in VARIANT_SPECS"` exits 0
2. `python -m pytest tests/test_v20_baseline_regression.py -v` exits 0
3. 5-dataset parity sweep log shows abs F1 diff < 1e-4 for s_linker13 vs s_linker13_clean on every dataset
4. `git diff --quiet -- src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py` exits 0
</verification>

<success_criteria>
- s_linker13_clean.py standalone file importable; class SLinker13Clean is not a subclass
- helper_v3 + prompts_v2 imports wired correctly
- CANONICAL_VARIANTS and VARIANT_SPECS register the variant with canonical=False
- 5-dataset Claude Sonnet parity sweep: abs F1 diff < 1e-4 on every dataset; macro F1 ≈ 0.9509
- Frozen files untouched (s_linker13.py, prompts_v2.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py)
- GATE-02 regression test still passes
- GATE-06 grep clean
</success_criteria>

<output>
After completion, create `.planning/phases/10-scaffolding/10-03-SUMMARY.md` recording:
- Path to the parity sweep log + ablation JSON
- Per-dataset F1 table (s_linker13 vs s_linker13_clean, abs diff)
- Macro F1 for both variants
- Final CANONICAL_VARIANTS length after registration
- Confirmation frozen files untouched
- Confirmation regression test still passes
</output>
