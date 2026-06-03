# CLAUDE.md

This repository keeps a curated subset of the experimental linker history.

## Active Surface

Retained runtime files:

- [src/llm_sad_sam/linkers/experimental/ilinker1.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker1.py)
- [src/llm_sad_sam/linkers/experimental/ilinker2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker2.py)
- [src/llm_sad_sam/linkers/experimental/ilinker3.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker3.py)
- [src/llm_sad_sam/linkers/experimental/s_linker.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker.py) through [src/llm_sad_sam/linkers/experimental/s_linker11a.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker11a.py)
- [src/llm_sad_sam/linkers/experimental/s_linker15.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker15.py) — v2.6.1 production (axiom-only, no training)
- [src/llm_sad_sam/linkers/experimental/s_linker15b.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker15b.py) — v2.6.2 negative result (alias-recovery, −2.5pp)
- [src/llm_sad_sam/linkers/experimental/s_linker15c.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker15c.py) — v2.6.2 negative result (ILinker4+entity hybrid, −0.6pp)
- [src/llm_sad_sam/linkers/experimental/s_linker17a.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker17a.py) — v2.6.2 renaming-only (ICSE Framing A/B/C names)
- [src/llm_sad_sam/linkers/experimental/s_linker17b.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker17b.py) — v2.6.2 unified k=2 voting architecture
- [src/llm_sad_sam/linkers/experimental/prompts.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/prompts.py)
- [src/llm_sad_sam/linkers/experimental/prompts_v2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/prompts_v2.py)
- [src/llm_sad_sam/core/data_types_v2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/core/data_types_v2.py)
- [src/llm_sad_sam/core/document_loader_v2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/core/document_loader_v2.py)
- [src/llm_sad_sam/pcm_parser_v2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/pcm_parser_v2.py)

Archived families such as `agent_linker*`, `cnr*`, `alinker`, and `ilinker2_v*` are kept under [archive/README.md](/home/yu/project/adc/agent-linker/archive/README.md).

## Build & Run

```bash
pip install -e ".[dev,openai]"
python run_ablation.py
python run_ablation.py --list-variants
pytest
```

[run_ablation.py](/home/yu/project/adc/agent-linker/run_ablation.py) is lightweight and only supports the retained ILinker/S-Linker set.

## Current Milestone (v2.6.2) — Multi-Framing Extraction Design

v2.6.2 explores unified multi-framing architecture for the ICSE paper. Two new variants:

- **`s_linker17a`** — rename-only variant of `s_linker15` using ICSE-friendly names (Framing A/B/C
  instead of Tier 1/2/3). Zero logic change. Validates that renaming is safe (expected F1 ≈ s15).
- **`s_linker17b`** — unified k=2 voting architecture. Alias discovery runs first (sequential),
  then all three Framings (A/B/C) extracted in parallel (all alias-aware), k=2 voting merge, unified
  evidence-bundle validation.

v2.7 (BBB Recall Closure) is frozen behind v2.6.2.

Run: `python run_ablation.py --variants s_linker17a s_linker17b`

## Production Linker (v2.6.1) — `s_linker15`

`src/llm_sad_sam/linkers/experimental/s_linker15.py` is the v2.6.1 experimental
linker: a **no-training, axiom-only** SAD-SAM linker (`experimental=True`,
`canonical=False`). It is `s_linker14_voyager` with ALL Voyager trained-bank
machinery removed — no bank loading, no `_wrap` injection, no `reload_bank`, no
training coupling. The axiom prompts are **inlined** in the file (copied from
`prompts_v4_axiom` B-variant + three v2.6.1 FP root-cause fixes: tier/platform
alias, code-path prefix leakage, functional-alias-as-workflow). Seed extractor is
`ILinker4` with empty seed rules (pure axiom seed).

- `s_linker14_voyager` (trained-bank consumer) is **retained alongside** s_linker15.
- Canonical `s_linker13_min` is unchanged.
- Run it: `python run_ablation.py --variants s_linker15`.
- Rationale (drop training): across v2.3–v2.6 the trained bank gave only ~+1.6pp
  over the axiom-only floor at high cost and WEAK verdicts, with BBB split-fragility.
  v2.6.1 commits to the axiom-only floor. See `.planning/milestones/v2.6.1-ROADMAP.md`.

## Notes

- `ilinker3` is not a standalone historical pipeline; the runner wraps it with a small adapter so it can be benchmarked like the other retained variants.
- `s_linker1` is represented by [src/llm_sad_sam/linkers/experimental/s_linker.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker.py).
- Default model policy is Claude Sonnet for the canonical line, but `.env` sets the
  active benchmarking backend to `gpt-5.4` (`LLM_BACKEND=openai`).
