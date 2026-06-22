# CLAUDE.md

This is the **s20U branch**: the experimental linker repo trimmed down to only
the files needed to run the `s_linker20_union` ("s20U") SAD-SAM sweep. The full
history (all other linker families, planning docs, logs, results, archives,
tests) lives on `master`.

## Active Surface

Runtime files retained on this branch:

- `run_ablation.py` — lightweight ablation runner (only the s20U variants are
  exercised here; benchmark inputs are read from the sibling `../ardoco` repo).
- `src/llm_sad_sam/linkers/experimental/s_linker20_union.py` — the linker
  (`class SLinker20Union`). Standalone: no inheritance from other linkers; all
  constants inlined. Used by both `s_linker20_union` and
  `s_linker20_union_noknow` (the latter passes `no_knowledge=True`).
- `src/llm_sad_sam/linkers/experimental/{helper_v3,ilinker3,__init__}.py`
- `src/llm_sad_sam/core/` — `data_types`, `data_types_v2`, `document_loader`,
  `document_loader_v2`, `model_analyzer`
- `src/llm_sad_sam/{llm_client,pcm_parser,pcm_parser_v2}.py`
- `run_s20union_*.sh` — N=3 sweep runners (gpt / sonnet / re_medium / noknow)

`experimental/__init__.py` was reduced to export only `ILinker3` and
`SLinker20Union`; the run path imports submodules by full path via `importlib`,
so the historical eager imports of the whole linker family are unnecessary.

## Build & Run

```bash
pip install -e ".[openai]"
python run_ablation.py --variants s_linker20_union --datasets mediastore
python run_ablation.py --list-variants
```

## Notes

- The variant registry in `run_ablation.py` still lists many non-retained
  variants (their modules were removed on this branch). Only the
  `s_linker20_union*` entries resolve to a present module here.
- Default benchmarking backend is set in `.env` (`LLM_BACKEND=openai`,
  `gpt-5.4`). `.env` is untracked.
