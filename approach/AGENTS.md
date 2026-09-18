# AGENTS.md

This is the **s126-only consolidation**: `master` was pruned down to the paper
arm (`s_linker126`) and the modules its own validation suite depends on. The
full variant history (every other linker family: i1/i2, s_linker through
s_linker125, the router, the S23 verification family, s26-s125 exploration
rounds) is preserved on `origin/archive/master-pre-s126-consolidation` —
check that branch out if you need to see or resurrect a retired variant.

## Active Surface

- `run_ablation.py` — ablation runner; registry now holds only `s_linker126`
  (the paper arm) and `s_linker126_noknow` (RQ4 knowledge A/B, same module,
  `no_knowledge=True`). `python run_ablation.py --list-variants` prints both.
- `src/llm_sad_sam/linkers/experimental/s_linker126.py` — the paper arm
  (`class SLinker126`), STANDALONE: the `s122 -> s123 -> s125 -> s126`
  subclass chain is flattened into this one file, so it imports no other
  `s_linkerNNN` module at runtime. Only shared infra: `core/`,
  `pcm_parser{,_v2}.py`, `llm_client.py`, `helper_v3.py`, `linker_infra.py`.
- `src/llm_sad_sam/linkers/experimental/{s_linker122,s_linker123,s_linker125}.py`
  — kept only because s126's own validation suite compares against them:
  `pilot/test_s126.py` imports `SLinker125` as a baseline and
  `pilot/test_s126_standalone.py` imports `s_linker122` as the pre-flatten
  ancestor to check the flattening is behaviour-preserving.
  `pilot/s127_greedy_merge_audit.py` also imports `s_linker122`/`s_linker123`.
- `src/llm_sad_sam/linkers/experimental/s_linker25.py` — kept because
  `pilot/design_audit.py` imports it, and `design_audit.py` is a dependency
  of `pilot/score_runs.py`, which `pilot/coref_exact_pilots.py` imports,
  which `pilot/test_s126.py` imports.
- `pilot/{test_s126,test_s126_standalone,s127_greedy_merge_audit,
  coref_exact_pilots,reading_pilots,score_runs,ab_stats,design_audit}.py`
  and `pilot/{run_s126_e2e,run_s126_e2e_noknow}.sh` — the s126 validation
  chain. `reading_pilots.py` was trimmed to just the benchmark/gold-loading
  helpers this chain uses (its old multi-variant stage-pilot comparison was
  archived with the variants it compared).
- `src/llm_sad_sam/linkers/experimental/{helper_v3,linker_infra,__init__}.py`
- `src/llm_sad_sam/core/`, `src/llm_sad_sam/{llm_client,pcm_parser,pcm_parser_v2}.py`

`experimental/__init__.py` exports `SLinker25`, `SLinker122`, `SLinker123`,
`SLinker125`, `SLinker126`. `run_ablation.py` also imports by full module
path via `importlib`, so no other namespace-level re-export is required.

## Build & Run

```bash
pip install -e ".[openai]"
python run_ablation.py --list-variants
python run_ablation.py --variants s_linker126 --datasets mediastore
```

The host provides the OpenAI credential as `OAI_KEY`, not `OPENAI_API_KEY`.
For OpenAI-backed commands, map it only in the process environment:

```bash
OPENAI_API_KEY="$OAI_KEY" python run_ablation.py ...
```

Never write either credential value to `.env`, logs, results, or tracked
files.

## Standing Gates

- **GATE-01**: the reported arm stays byte-stable —
  `src/llm_sad_sam/linkers/experimental/s_linker126.py` above all. New work
  subclasses or forks it; edits to shared files (`__init__.py`,
  `run_ablation.py`) are purely additive.
- **GATE-06**: no benchmark-derived vocabulary in any new code — prompts and
  rubrics stay generic English; the runtime catalog (component names, code
  identifiers) is the only project-specific input.
- **GATE-07**: every prompt clause and every code gate must stand on a
  general rule, general software-engineering practice, or prior work
  measured on this branch or in the literature — never a surface form or
  syntax that merely happens to occur in the five benchmark documents.

Facts about a case stay in code; judgment or weighting about a case goes in
the prompt a judge reads (the design law `approach/CLAUDE.md` derives and
applies repeatedly). A proposed rewording of a load-bearing criterion needs a
fixed-input validation (level 1/2 of the measurement policy) before an
end-to-end run is paid for.

Paper-facing claims also follow the repository-wide restrictions in the root
`AGENTS.md`: report the measured scope, distinguish observation from
interpretation, avoid unsupported causal or superiority language.

## Notes

- Default benchmarking backend is set in `.env` (`LLM_BACKEND=openai`,
  `gpt-5.4`). `.env` is untracked.
- For the full round-by-round ledger behind why s126 looks the way it does
  (s26 through s125's design rounds, every refused arm, every measured
  gate), see `approach/CLAUDE.md` and, for the complete history including
  archived files, `git log` on
  `origin/archive/master-pre-s126-consolidation`.
