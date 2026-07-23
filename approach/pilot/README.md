# s21 Prompt-Integrated Router Pilot

This pilot tests whether the useful part of the router work can be folded into
s21's prompt structure without using the promoted agentic router.

The experiment keeps canonical `s_linker21.py` untouched. It uses frozen s21
outputs from `results/v2.6.6_extracts_s21/gpt` as the floor, runs live typed
batch extraction prompts over the benchmark corpus, validates new candidates
with s21-style gates, and reports macro P/R/F1.

## Where Things Are Implemented

- `pilot/s21_prompt_router_live.py` implements the live prompt-integration
  harness. It owns the typed extraction prompt variants, contrast validation,
  baseline/augmented scoring, F1/F2 reporting, and extraction-vs-validation
  diagnostics.
- `pilot/f2_validation_grid.py` implements the second-stage F2 optimization
  grid. It reuses the cached `typed_all_filter_named` extraction output, applies
  structural validation filters, optionally runs cached IMPLICIT/ANAPHORA
  context validators, and ranks policies by macro F2.
- `pilot/cache/` stores live LLM caches and JSON summaries. Re-running the same
  commands should reuse these caches instead of spending new calls, unless cache
  files are deleted.
- `pilot/RESULTS.md` records the workflow, score tables, and interpretation.

Canonical `s_linker21.py` is not modified by this pilot. The integrated follow-up
variant is implemented as `SLinker22` in
`src/llm_sad_sam/linkers/experimental/s_linker22.py` and registered as
`s_linker22` in `run_ablation.py`.

## Workflow

1. **Use frozen s21 as the floor.**
   The scripts load links from `results/v2.6.6_extracts_s21/gpt/run*/`.

2. **Run typed extraction prompts.**
   `s21_prompt_router_live.py` asks for component references with modes such as
   `AFFIRMATIVE`, `CONTRAST`, `IMPLICIT`, `ANAPHORA`, and `CODEPATH`.

3. **Validate only deployable model-doc candidates.**
   `AFFIRMATIVE` candidates use the unchanged s21 P1/P2 entity validator.
   `CONTRAST` candidates use a contrast-specific validator. `CODEPATH` is not
   accepted as a model-doc link.

4. **Score augmented links.**
   The augmented set is frozen s21 plus newly validated proposals. The scripts
   report macro P/R/F1/F2 and diagnostics showing how many base-missed gold links
   were surfaced by extraction and kept by validation.

5. **Optimize for F2 with structural filters.**
   `f2_validation_grid.py` tests policies over the cached extraction output. The
   best current policy is `exact_or_terminal_no_code`: reject code/test/path-like
   evidence, allow contrast via the contrast judge, and for `AFFIRMATIVE` require
   exact component evidence or terminal-word evidence such as `the client` for
   `HTML5 Client`.

## Run

```bash
python pilot/s21_prompt_router_live.py --variants typed_all_filter_named typed_named_only scratchpad_named
python pilot/f2_validation_grid.py
```

Outputs and caches are written under `pilot/cache/`.

## Integrated Runtime Variant

`s_linker22` carries the pilot's best policy into the normal s21-style workflow:

1. Phase 2 keeps live s21 Framing-C extraction as the floor and adds live typed
   extraction for candidates the floor missed.
2. Phase 4 sends floor candidates through the unchanged s21 P1/P2 validator.
3. Typed `IMPLICIT`, `ANAPHORA`, and `CODEPATH` candidates are rejected as
   model-doc links.
4. Typed `AFFIRMATIVE` candidates pass the exact/terminal/no-code evidence filter, then
   the unchanged s21 P1/P2 validator.
5. Typed `CONTRAST` candidates pass a contrast-specific claim-before-verdict
   validator.
6. Later s21 phases, including coreference and dedup merge, continue to run.

Run the integrated workflow with the default OpenAI `gpt-5.4` backend:

```bash
python run_ablation.py --variants s_linker22 --datasets mediastore teastore teammates bigbluebutton jabref
```
