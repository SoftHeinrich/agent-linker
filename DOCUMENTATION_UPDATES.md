# Documentation Updates: Linker Retention Correction (2026-03-26)

## Summary

The active repository was corrected to keep:

- `ILinker1`
- `ILinker2`
- `ILinker3`
- `SLinker` through `SLinker11a`
- their prompt files and v2-stack dependencies

Everything else remains archived.

## Active Runtime Paths

- [src/llm_sad_sam/linkers/experimental/ilinker1.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker1.py)
- [src/llm_sad_sam/linkers/experimental/ilinker2.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker2.py)
- [src/llm_sad_sam/linkers/experimental/ilinker3.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/ilinker3.py)
- [src/llm_sad_sam/linkers/experimental/s_linker.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker.py) through [src/llm_sad_sam/linkers/experimental/s_linker11a.py](/home/yu/project/adc/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker11a.py)

## Archived Paths

- [archive/linkers/experimental](/home/yu/project/adc/agent-linker/archive/linkers/experimental): non-retained linker families and `ilinker2_v*`
- [archive/test_scripts/2026-03-26-v40c-retention](/home/yu/project/adc/agent-linker/archive/test_scripts/2026-03-26-v40c-retention): historical root-level tests
- [archive/old_scripts/2026-03-26-v40c-retention](/home/yu/project/adc/agent-linker/archive/old_scripts/2026-03-26-v40c-retention): historical helper scripts

## Updated Files

- [run_ablation.py](/home/yu/project/adc/agent-linker/run_ablation.py): supports only the retained variants
- [README.md](/home/yu/project/adc/agent-linker/README.md): updated usage and retention scope
- [AGENTS.md](/home/yu/project/adc/agent-linker/AGENTS.md): updated repository instructions
- [CLAUDE.md](/home/yu/project/adc/agent-linker/CLAUDE.md): updated repository instructions
