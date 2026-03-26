# CLAUDE.md

This repository keeps a curated subset of the experimental linker history.

## Active Surface

Retained runtime files:

- [src/llm_sad_sam/linkers/experimental/ilinker1.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker1.py)
- [src/llm_sad_sam/linkers/experimental/ilinker2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker2.py)
- [src/llm_sad_sam/linkers/experimental/ilinker3.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker3.py)
- [src/llm_sad_sam/linkers/experimental/s_linker.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker.py) through [src/llm_sad_sam/linkers/experimental/s_linker11a.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker11a.py)
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

## Notes

- `ilinker3` is not a standalone historical pipeline; the runner wraps it with a small adapter so it can be benchmarked like the other retained variants.
- `s_linker1` is represented by [src/llm_sad_sam/linkers/experimental/s_linker.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker.py).
- Default model policy remains Claude Sonnet unless there is an explicit reason to change it.
