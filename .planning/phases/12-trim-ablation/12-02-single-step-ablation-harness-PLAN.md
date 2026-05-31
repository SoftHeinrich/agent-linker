---
phase: 12-trim-ablation
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - src/llm_sad_sam/ablation/__init__.py
  - src/llm_sad_sam/ablation/single_step.py
  - src/llm_sad_sam/ablation/__main__.py
  - tests/test_single_step_harness.py
  - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md
  - .planning/phases/12-trim-ablation/12-02-SUMMARY.md
autonomous: true
requirements: [PROMPT-02]
must_haves:
  truths:
    - "A documented single-step ablation entry point exists: python -m llm_sad_sam.ablation single_step --variant X --dataset D --phase P"
    - "Running the harness on s_linker13_clean for any phase reproduces final.pkl F1 within run-to-run variance vs the cached final.pkl (no semantic change is a no-op when re-running)"
    - "The harness writes a per-run results JSON: results/ablation_results/12_02_harness/<variant>/<dataset>/<phase>.json with F1, P, R, fp, fn, baseline_F1, delta_F1"
    - "The harness rejects unknown phase names with a non-zero exit and a clear error"
    - "The harness CANNOT run if the upstream checkpoint for the requested phase is missing — it surfaces the missing path, does not silently regenerate"
    - "Phase→upstream-checkpoint dependency table is committed: which downstream phases must re-run after each modified phase"
  artifacts:
    - path: "src/llm_sad_sam/ablation/single_step.py"
      provides: "Single-step ablation engine — loads upstream checkpoint, runs target phase on a given variant, propagates through downstream phases, scores against gold standard"
      exports: ["run_single_step", "PHASE_ORDER", "DOWNSTREAM_DEPS"]
    - path: "src/llm_sad_sam/ablation/__main__.py"
      provides: "CLI: python -m llm_sad_sam.ablation single_step ..."
      contains: "argparse, single_step subcommand"
    - path: "tests/test_single_step_harness.py"
      provides: "Smoke tests for the harness — exit codes, missing-checkpoint behavior, results-JSON shape, baseline-equivalence on s_linker13_clean"
    - path: ".planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md"
      provides: "Documented phase-to-upstream-checkpoint dependency table that trim plans 12-03/04/05 cite"
      contains: "PHASE_ORDER, DOWNSTREAM_DEPS"
  key_links:
    - from: "src/llm_sad_sam/ablation/single_step.py"
      to: "results/phase_cache/<variant>/<dataset>/{layer1,layer2,entity_candidates,entity_decisions,final}.pkl"
      via: "pickle.load on the upstream-checkpoint path resolved from variant+dataset+phase"
      pattern: "pickle.load(open(.*\\.pkl"
    - from: "src/llm_sad_sam/ablation/single_step.py"
      to: "VARIANT_SPECS in run_ablation.py"
      via: "import + instantiation of the requested variant class"
      pattern: "from run_ablation import VARIANT_SPECS"
    - from: "src/llm_sad_sam/ablation/single_step.py"
      to: "tests/fixtures/v2_0_baseline.json"
      via: "baseline F1 lookup for delta computation"
      pattern: "v2_0_baseline.json"
---

<objective>
Build (or extend) the single-step ablation harness so that plans 12-03 / 12-04 / 12-05 can each ablate ONE prompt by re-running ONLY the affected phase on top of the existing layer1/layer2/entity_candidates/entity_decisions checkpoints — instead of re-running the full 5-phase pipeline per trim. The harness must (a) load the correct upstream checkpoint for a given target phase, (b) run the target phase with the variant's modified prompts, (c) propagate output through the downstream phases that depend on the modified phase's output (per a committed dependency table), (d) score the resulting links against the gold standard, and (e) write a results JSON with F1/P/R/fp/fn plus delta vs baseline.

Purpose: satisfies the USER DIRECTIVE in 12-CONTEXT.md (checkpoint-loaded single-step ablation, not full-pipeline sweeps) and unblocks every per-trim plan in Wave 2. Without this harness, plans 12-03/04/05 would have to invent their own per-trim runner and would risk inconsistent measurement.

Output: a small Python package `llm_sad_sam.ablation` with `single_step` CLI, a documented phase-to-upstream-checkpoint contract, smoke tests, and a SUMMARY linking to the contract.
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
@tests/test_mention_type_ablation.py

<interfaces>
<!-- Checkpoint surface and variant instantiation contract — extracted from s_linker13_clean.py + run_ablation.py -->

From src/llm_sad_sam/linkers/experimental/s_linker13_clean.py:
- _save_phase emits 5 pickles per dataset under `${PHASE_CACHE_DIR:-./results/phase_cache}/${_VARIANT_NAME}/${dataset}/`
- Phase names (in pipeline order): "layer1", "layer2", "entity_candidates", "entity_decisions", "final"
- Layer1 state keys: "model_knowledge", "doc_knowledge", "raw_seed_links"
- Layer2 state keys: "seed_links", "validated", "coref_links"
- entity_candidates state keys: "entity_candidates", "bundles"
- entity_decisions state keys: "decisions"
- final state keys: "final"
- The class signature: SLinker13Clean(backend=None, model=None, checkpoint_fallback=None, checkpoint_fallback_model=None)
- `_VARIANT_NAME` is a class attribute used to namespace the checkpoint dir

From run_ablation.py:
- VARIANT_SPECS[name] = dict(aliases=, module=, class_name=, description=, canonical=)
- DATASETS dict maps dataset name → {text, model, gold_sam, transarc_sam} paths
- get_backend() reads LLM_BACKEND env (claude | openai | checkpoint)
- run_variant(...) is the existing full-pipeline runner — DO NOT call from the harness; the harness needs partial-pipeline replay

From tests/test_mention_type_ablation.py:
- Loads pickles via `pickle.load(open(os.path.join(CACHE, dataset, f"{phase}.pkl"), "rb"))`
- This is the offline-analysis pattern; the harness extends this to ALSO re-execute the phase with a different variant

From tests/fixtures/v2_0_baseline.json (referenced by GATE-02):
- Per-variant, per-dataset baseline F1 used as the comparison anchor
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Define the phase-to-upstream-checkpoint dependency contract</name>
  <files>
    - src/llm_sad_sam/ablation/__init__.py
    - src/llm_sad_sam/ablation/single_step.py
    - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md
    - tests/test_single_step_harness.py
  </files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 216-320 (the `link()` method — pipeline DAG)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 1080-1102 (_checkpoint_dir, _save_phase)
    - .planning/phases/12-trim-ablation/12-CONTEXT.md decisions section (the dependency-enumeration requirement: "the executor MUST either re-run downstream phases or use existing checkpoint")
    - .planning/research/PROMPT-HARNESS-SURVEY.md §0 (the phase-to-prompt map)
  </read_first>
  <behavior>
    - Test: `from llm_sad_sam.ablation.single_step import PHASE_ORDER` returns the tuple `("layer1", "layer2", "entity_candidates", "entity_decisions", "final")`.
    - Test: `from llm_sad_sam.ablation.single_step import DOWNSTREAM_DEPS` returns a dict mapping each phase to the tuple of phases that MUST be re-run when the keyed phase is modified.
    - Test: DOWNSTREAM_DEPS["layer1"] == ("layer2", "entity_candidates", "entity_decisions", "final") — every downstream phase consumes layer1.
    - Test: DOWNSTREAM_DEPS["entity_candidates"] == ("entity_decisions", "final") — entity validation reads bundles produced by entity_candidates.
    - Test: DOWNSTREAM_DEPS["entity_decisions"] == ("final",) — final dedup reads validated entity links.
    - Test: DOWNSTREAM_DEPS["layer2"] == ("final",) — final dedup only.
    - Test: DOWNSTREAM_DEPS["final"] == () — terminal phase, nothing downstream.
  </behavior>
  <action>
    Create the package skeleton. `src/llm_sad_sam/ablation/__init__.py` is empty.

    In `src/llm_sad_sam/ablation/single_step.py`, define two module-level constants:

      PHASE_ORDER = ("layer1", "layer2", "entity_candidates", "entity_decisions", "final")

      DOWNSTREAM_DEPS = {
          "layer1": ("layer2", "entity_candidates", "entity_decisions", "final"),
          "layer2": ("final",),
          "entity_candidates": ("entity_decisions", "final"),
          "entity_decisions": ("final",),
          "final": (),
      }

    Add a docstring at module top documenting the contract and citing 12-CONTEXT.md decisions section.

    Write `.planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md` containing a Markdown table with columns: `phase | upstream_checkpoint_required | downstream_phases_to_rerun | which_prompts_fire_in_this_phase | trim_plan_that_modifies_it`. Populate from the survey §0 phase-to-prompt map:

      | layer1 | (none — first phase) | layer2, entity_candidates, entity_decisions, final | AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES | 12-03 (Step 1 — judge), 12-05 (Step 3 — runtime rubric on judge) |
      | layer2 | layer1 | final | ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES, SEED_DISAMBIGUATION_RULES | (none — Step 2 modifies entity_candidates / entity_decisions phases inside the layer2 DAG; see row below) |
      | entity_candidates | layer1 | entity_decisions, final | ENTITY_EXTRACTION_RULES | 12-04 (Step 2 — ent+val merge) |
      | entity_decisions | layer1, entity_candidates | final | VALIDATION_RULES | 12-04 (Step 2 — ent+val merge) |
      | final | layer2, entity_candidates, entity_decisions | — | (no prompts — pure dedup) | (none) |

    NOTE: Step 2 (ent+val merge) re-runs BOTH entity_candidates AND entity_decisions because the merge collapses prompts across both sub-phases. Step 1 / Step 3 re-run layer1 only and let layer2/entity_*/final replay deterministically from the new layer1.

    **CRITICAL HARNESS CONTRACT (entity_candidates / entity_decisions reuse rule):** `layer2.pkl` is NOT a re-runnable phase — it is the synthesis pickle of the `_run_parallel({seed_val, coref, entity})` block inside `s_linker13_clean.link()`. The harness uses `layer2.pkl` purely as a CACHE READ for the seed_val and coref tracks when re-running entity_candidates or entity_decisions; the entity track is overridden surgically. This means Step 2 (12-04) re-runs entity_candidates AND entity_decisions but does NOT pay live LLM cost for seed_val or coref. The harness MUST assert zero live LLM calls on seed_val and coref tracks when phase ∈ {entity_candidates, entity_decisions}. See Task 2 step 8 for the implementation requirement.

    **Harness coupling note (technical debt to be tracked):** The harness reaches into `_run_entity_pipeline`, `_extract_entities_enriched`, `_validate_with_evidence`, `_run_seed_validation`, `_run_coreference` by method name. These are semi-private but stable in `s_linker13_clean`. Any future refactor of these methods will silently break the harness. Plan 12-02 SUMMARY must record this constraint; Phase 13 promotion (`s_linker13_min`) should preserve these method names or update the harness in lock-step.

    Create `tests/test_single_step_harness.py` with `test_phase_order_constant`, `test_downstream_deps_layer1`, `test_downstream_deps_layer2`, `test_downstream_deps_entity_candidates`, `test_downstream_deps_entity_decisions`, `test_downstream_deps_final` exactly matching the behaviors above.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; python -c "from llm_sad_sam.ablation.single_step import PHASE_ORDER, DOWNSTREAM_DEPS; assert PHASE_ORDER == ('layer1','layer2','entity_candidates','entity_decisions','final'); assert DOWNSTREAM_DEPS['final'] == (); assert DOWNSTREAM_DEPS['layer1'] == ('layer2','entity_candidates','entity_decisions','final')" &amp;&amp; pytest tests/test_single_step_harness.py::test_phase_order_constant tests/test_single_step_harness.py::test_downstream_deps_layer1 tests/test_single_step_harness.py::test_downstream_deps_layer2 tests/test_single_step_harness.py::test_downstream_deps_entity_candidates tests/test_single_step_harness.py::test_downstream_deps_entity_decisions tests/test_single_step_harness.py::test_downstream_deps_final -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `python -c "from llm_sad_sam.ablation.single_step import PHASE_ORDER, DOWNSTREAM_DEPS"` exits 0.
    - All 6 contract tests pass.
    - `.planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md` exists with the 5-row dependency table (the verify command implicitly checks this by importing the constants the doc describes; reviewer reads the doc).
    - No edits to v2.0 frozen files: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
  </acceptance_criteria>
  <done>Contract committed, downstream-deps table is the canonical reference plans 12-03/04/05 cite.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement run_single_step engine + CLI + smoke tests</name>
  <files>
    - src/llm_sad_sam/ablation/single_step.py
    - src/llm_sad_sam/ablation/__main__.py
    - tests/test_single_step_harness.py
  </files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 216-320 (the `link()` orchestrator — what phase emits what)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 366-466 (_learn_document_knowledge_enriched — Tier 1 judge call: target of Step 1 + Step 3 trims)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 688-714 (_run_entity_pipeline — target of Step 2 trims)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 818-end (_validate_with_evidence — target of Step 2 trims)
    - run_ablation.py lines 89-367 (VARIANT_SPECS shape)
    - run_ablation.py lines 433-560 (run_variant, get_backend, normalize_variants — DO NOT re-use directly; copy the pieces you need)
    - tests/fixtures/v2_0_baseline.json (the per-variant per-dataset F1 anchor — read structure)
    - tests/test_mention_type_ablation.py (the pickle-load + analysis pattern to match)
  </read_first>
  <behavior>
    - Test: `run_single_step(variant="s_linker13_clean", dataset="mediastore", phase="layer1", results_dir=tmp_path, backend="checkpoint")` runs end-to-end against the existing `results/phase_cache/s_linker13_clean/mediastore/` checkpoint, emits `<results_dir>/s_linker13_clean/mediastore/layer1.json` with keys `["variant", "dataset", "phase", "F1", "P", "R", "fp", "fn", "baseline_F1", "delta_F1"]`, and the F1 is within ±0.02 of the cached `final.pkl`-derived F1 because no semantic change happened (variant is unchanged baseline).
    - Test: `run_single_step(variant="s_linker13_clean", dataset="mediastore", phase="not_a_real_phase", ...)` raises ValueError mentioning the unknown phase.
    - Test: `run_single_step(variant="s_linker13_clean", dataset="mediastore", phase="entity_decisions", ...)` — requires both layer1.pkl AND entity_candidates.pkl to exist; if either is missing, raises FileNotFoundError naming the missing pickle path.
    - Test: CLI invocation `python -m llm_sad_sam.ablation single_step --variant s_linker13_clean --dataset mediastore --phase layer1 --results-dir <tmp> --backend checkpoint` exits 0 and produces the same JSON.
    - Test: CLI exits non-zero when `--variant` is not in VARIANT_SPECS.
    - Test: CLI exits non-zero when `--dataset` is not in DATASETS.
  </behavior>
  <action>
    Implement `run_single_step` in `src/llm_sad_sam/ablation/single_step.py`. Signature:

      def run_single_step(
          variant: str,
          dataset: str,
          phase: str,
          results_dir: str | Path,
          backend: str = "claude",
          model: str | None = None,
          phase_cache_dir: str | None = None,  # overrides PHASE_CACHE_DIR env
      ) -> dict

    Algorithm:
      1. Validate phase in PHASE_ORDER, raise ValueError if not.
      2. Resolve variant_spec from VARIANT_SPECS; raise KeyError with available variants if missing.
      3. Resolve dataset paths from DATASETS; raise KeyError if missing.
      4. Resolve upstream-checkpoint requirement: for phase X, the set of pickles {p1.pkl, ...} that must EXIST on disk under `${phase_cache_dir or PHASE_CACHE_DIR or './results/phase_cache'}/{variant}/{dataset}/`. The required upstream set is PHASE_ORDER[:idx_of(phase)] — every phase before X. If phase == "layer1" the upstream set is empty.
      5. For each required upstream pickle, assert os.path.exists; if missing, raise FileNotFoundError listing the missing paths (in order). Do NOT silently regenerate.
      6. Instantiate the variant class (mirroring run_variant's instantiation pattern in run_ablation.py — backend object built from `get_backend()` semantics; for "checkpoint" backend, use `LLMBackend.CHECKPOINT`).
      7. Load upstream checkpoints (e.g., for phase == "entity_decisions": layer1 + entity_candidates).
      8. Re-execute target phase only:
         - phase == "layer1": call variant.link() with PHASE_CACHE_DIR pointing at a per-run tmp subdir so we don't overwrite the baseline cache. Capture the final links after the full pipeline runs.
         - phase == "layer2": load layer1 state into the variant instance, then call the layer2-DAG runner (the `_run_parallel({...})` block inside link()); subsequent dedup uses the loaded raw_seed_links/coref/etc.
         - phase == "entity_candidates" / "entity_decisions":
           * **CRITICAL — DO NOT re-run seed_val OR coreference tracks live.** `_run_entity_pipeline` lives inside `_run_parallel({seed_val, coref, entity})` in `s_linker13_clean.link()`; `layer2.pkl` is the synthesis pickle that captures ALL three tracks. The harness must reuse the cached non-entity state.
           * Load layer1 state into the variant. ALSO load `layer2.pkl` into a `cached_layer2` dict for the variant+dataset; from it pluck `cached_seed_links` and `cached_coref_links` (the seed_val + coref output the baseline already paid for).
           * phase == "entity_candidates": invoke ONLY `_extract_entities_enriched(...)` to produce a fresh `entity_candidates` + `bundles`. Skip `_run_seed_validation` and `_run_coreference` entirely.
           * phase == "entity_decisions": additionally load `entity_candidates.pkl` from the baseline cache (do not re-run extraction) and invoke ONLY `_validate_with_evidence(...)` to produce fresh `validated_entity_links`.
           * After the surgical re-run, reconstruct a layer2-equivalent state by combining {seed_links: cached_seed_links, coref_links: cached_coref_links, validated_entity_links: <new from this phase>} and feed it into final dedup.
           * The acceptance test for `phase=entity_candidates` MUST count live LLM calls (count via `LLMClient`'s call counter or by inspecting the phase log) and assert that seed_val and coref tracks made ZERO live calls — proving the surgical override.
         - phase == "final": just re-run dedup — primarily a sanity path.
      9. Score final links against gold_sam pairs using the SAME scoring helper run_ablation.py uses (`eval_metrics` / `load_gold_sam` — import them).
      10. Read `tests/fixtures/v2_0_baseline.json` and look up `baseline_F1` for (variant, dataset); if not present, set baseline_F1 to null.
      11. Write `<results_dir>/<variant>/<dataset>/<phase>.json` with the result dict.
      12. Return the result dict.

    Implementation note: re-using the existing pipeline orchestration without forking the entire `link()` method requires either (a) a small refactor in s_linker13_clean (out of scope — frozen-modulo-Plan-12 constraint), or (b) selective method invocation from outside. Choose option (b): the harness instantiates the variant, sets `self._current_text_path = text_path` and `self.model_knowledge`/`self.doc_knowledge` from the loaded layer1 pickle, then calls the lower-level methods (`_run_seed_validation`, `_run_entity_pipeline`, `_run_coreference`, `_extract_entities_enriched`, `_validate_with_evidence`). This is permitted — these are public-ish method names already used by the link() orchestrator. No modification to s_linker13_clean.py is required; the harness just calls into it.

    Implement `src/llm_sad_sam/ablation/__main__.py` as a thin argparse wrapper:

      `python -m llm_sad_sam.ablation single_step --variant X --dataset D --phase P --results-dir R [--backend claude|openai|checkpoint] [--model M] [--phase-cache-dir DIR]`

    Use a subcommand structure so future commands (e.g., `multi_step`, `sweep`) can be added.

    Extend `tests/test_single_step_harness.py` with the 6 behaviors above. For the baseline-equivalence smoke test, use backend="checkpoint" so no live LLM is called — the checkpoint backend replays cached LLM responses.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; pytest tests/test_single_step_harness.py -x -q &amp;&amp; python -m llm_sad_sam.ablation single_step --variant s_linker13_clean --dataset mediastore --phase layer1 --results-dir /tmp/12_02_harness_smoke --backend checkpoint &amp;&amp; test -f /tmp/12_02_harness_smoke/s_linker13_clean/mediastore/layer1.json &amp;&amp; python -c "import json; d=json.load(open('/tmp/12_02_harness_smoke/s_linker13_clean/mediastore/layer1.json')); assert d['variant']=='s_linker13_clean'; assert d['phase']=='layer1'; assert 'F1' in d; assert 'delta_F1' in d"</automated>
  </verify>
  <acceptance_criteria>
    - All 6 tests in `tests/test_single_step_harness.py` pass.
    - CLI smoke run for (s_linker13_clean, mediastore, layer1, checkpoint backend) exits 0 and writes the expected JSON.
    - Unknown phase, variant, dataset raise with non-zero exit and a clear message (verified by 2 of the 6 tests).
    - The harness REQUIRES the upstream checkpoints to exist — does NOT silently regenerate them (verified by the FileNotFoundError test).
    - No edits to v2.0 frozen files: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
    - No edits to s_linker13_clean.py: `git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker13_clean.py` exits 0 (the harness CALLS INTO it, does not modify it).
  </acceptance_criteria>
  <done>Harness CLI runs; smoke test passes; trim plans can resume per-phase against either Claude or gpt-5.4 checkpoint trees.</done>
</task>

<task type="auto">
  <name>Task 3: Wire baseline-equivalence sanity check + write 12-02-SUMMARY</name>
  <files>
    - .planning/phases/12-trim-ablation/12-02-SUMMARY.md
    - results/ablation_results/12_02_harness/
  </files>
  <read_first>
    - tests/fixtures/v2_0_baseline.json (the per-variant per-dataset baseline F1)
    - results/phase_cache/s_linker13_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/final.pkl (verify each exists before invoking harness)
    - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md (the contract Task 1 committed)
  </read_first>
  <action>
    Run a no-LLM equivalence sweep using the harness against the existing `s_linker13_clean` baseline checkpoints, one per dataset, to prove the harness reproduces the baseline F1 within run-to-run variance when no semantic change is applied.

    For each dataset in {mediastore, teastore, teammates, bigbluebutton, jabref}:
      `python -m llm_sad_sam.ablation single_step --variant s_linker13_clean --dataset <ds> --phase final --results-dir results/ablation_results/12_02_harness --backend checkpoint`

    NOTE: phase=="final" re-runs only dedup (deterministic, no LLM); this is the cheapest verification that the harness load/score path is correct. If a CI-friendly stronger check is needed, also run phase=="entity_decisions" with backend=="checkpoint" on a single dataset (mediastore) to exercise the validation-replay path. Stay on backend=="checkpoint" for all of these — no live LLM cost.

    Aggregate results into `results/ablation_results/12_02_harness/equivalence_summary.json` with:
      `{ "variant": "s_linker13_clean", "phase": "final", "per_dataset": {<ds>: {"harness_F1": float, "baseline_F1": float, "delta_F1": float}}, "max_abs_delta": float, "within_variance": bool }`
    where `within_variance` is true iff `max_abs_delta <= 0.02`.

    Write `.planning/phases/12-trim-ablation/12-02-SUMMARY.md`:
      - References PROMPT-02 (the harness is the precondition for every trim ablation).
      - Lists the 6 artifact paths (package init, single_step.py, __main__.py, test file, contract doc, equivalence_summary.json).
      - States the equivalence-sweep verdict (PASS if `within_variance`, FAIL otherwise) with the per-dataset deltas.
      - Cites 12-CONTEXT.md decisions section ("checkpoint-loaded single-step ablation").
      - Explicitly notes that Step 2 (12-04) requires re-running TWO sub-phases (entity_candidates + entity_decisions) because the merge spans both — confirms the harness handles this case.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; python -c "import json; d=json.load(open('results/ablation_results/12_02_harness/equivalence_summary.json')); assert d['within_variance'] is True, d; assert len(d['per_dataset'])==5; print('OK max_abs_delta=', d['max_abs_delta'])"</automated>
  </verify>
  <acceptance_criteria>
    - `results/ablation_results/12_02_harness/equivalence_summary.json` exists and `within_variance` is True (max |delta_F1| ≤ 0.02 across all 5 datasets).
    - `12-02-SUMMARY.md` exists, references PROMPT-02 and 12-CONTEXT decisions, lists all 6 artifact paths.
    - No live-LLM calls were made (backend=="checkpoint" for every harness invocation).
    - GATE-02 unaffected (no edits to canonical-variant registry): `pytest tests/test_v20_baseline_regression.py -q` exits 0.
  </acceptance_criteria>
  <done>Harness equivalence verified across 5 datasets; SUMMARY ships; trim plans 12-03/04/05 can run.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Harness → pickle files | The harness loads pickle artifacts produced by trusted in-repo code; treats them as authoritative |
| Harness → variant class (instantiation + method invocation) | Calls semi-private methods (`_run_seed_validation`, `_run_entity_pipeline`, `_validate_with_evidence`); coupling is explicit and tested |
| Harness → results JSON on disk | Local-only artifact under `results/ablation_results/` |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-02-01 | Tampering | accidental harness edits to s_linker13_clean.py while wiring method calls | mitigate | Task 2 + Task 3 acceptance: `git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker13_clean.py` exits 0 |
| T-12-02-02 | Information disclosure | LLM-response pickles containing intermediate state could leak project-specific text into logs | accept | Results live under `results/` which is already gitignored from public artifacts; not a deployment concern |
| T-12-02-03 | Repudiation | results JSON doesn't record which checkpoint version was loaded | mitigate | Embed `phase_cache_dir`, `variant`, `dataset`, ISO timestamp in every results JSON for traceability |
| T-12-02-04 | Denial of service | downstream trim plans (12-03/04/05) build on harness — if harness regresses, all three break | mitigate | Task 3's equivalence sweep is the regression gate; `within_variance` flag must be True before Wave 2 starts |
</threat_model>

<verification>
- Package `llm_sad_sam.ablation` importable with `single_step` module.
- CLI `python -m llm_sad_sam.ablation single_step ...` runs end-to-end.
- Contract doc lists every phase's upstream checkpoints + downstream re-runs.
- Equivalence sweep on s_linker13_clean shows ≤ 0.02 |ΔF1| on all 5 datasets, all phases probed.
- No frozen file modified; no s_linker13_clean.py edit.
- GATE-02 still green.
</verification>

<success_criteria>
- PROMPT-02 precondition closed: a documented, tested single-step ablation entry point exists.
- Wave 2 trim plans (12-03, 12-04, 12-05) can invoke `python -m llm_sad_sam.ablation single_step ...` with consistent measurement semantics.
- Phase-to-upstream-checkpoint dependency contract is the canonical source; trim plans cite it instead of re-deriving.
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-02-SUMMARY.md`.
</output>
</content>
</invoke>