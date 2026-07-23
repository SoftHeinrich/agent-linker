# Spike 004 harness — validator-layer ablation

Replays ONLY the entity + coref validation gates on the frozen `s_linker20_union`
nothink phase_caches (upstream candidates loaded from cache, never re-extracted), so
different validators can be A/B'd at effort 0 without rerunning the whole pipeline.
All runs use the Claude Sonnet CLI backend; effort 0 = `CLAUDE_DISABLE_THINKING=1`.

## Files

| file | role |
|------|------|
| `cache_io.py` | load layer1/3/4/final pickles for a cell + benchmark sent_map/components/gold; build per-link contexts. Reuses `run_ablation` loaders so scoring is identical. |
| `traps.py` | Mode 2 structural trap predicates (taboo-safe). |
| `stage0_reproduce.py` | reproduce note baselines (92.8 / 89.7) from cached ablation JSONs. |
| `stage0b_trap.py` | apply rule traps to cached links ($0); per-trap precision-recovery vs TP-removal. |
| `layered_validator.py` | `LayeredValidator(SLinker20Union)` — Mode 5 justification scaffold + Mode 1/2 rubric (v1–v4, env `SPIKE_RUBRIC`) + Mode 4 coref skeptic (env `SPIKE_CORE_SKEPTIC`). Validator-layer override only. |
| `replay.py` | drive the gates on cached candidates; write per-cell result JSON. |
| `summarize.py` | aggregate a label vs cached nothink/thinking-on (macro-F1, FP-by-source, implicit-FN guardrail, latency). |

## Reproduce

```bash
cd <agent-linker repo root>
python .planning/spikes/004-nogap-validator-ab/harness/stage0_reproduce.py
python .planning/spikes/004-nogap-validator-ab/harness/stage0b_trap.py
# winning config (entity-lenient / coref-strict), effort 0, one run × all datasets:
python .planning/spikes/004-nogap-validator-ab/harness/replay.py \
    --run run1 --datasets "mediastore teastore teammates bigbluebutton jabref" \
    --validator layered --rubric v4 --thinking off
python .planning/spikes/004-nogap-validator-ab/harness/summarize.py --labels layered_v4_offthink
```

Per-cell results land in `../results/<label>/<run>/<dataset>.json`. `<label>` encodes
validator+rubric+skeptic+thinking (e.g. `layered_v4_offthink`).

## Notes

- The Claude CLI backend does not surface token usage, so cost is reported as latency +
  call count, not output tokens. effort-0 ≈ 14–16 s/call (zero thinking tokens).
- Baselines are NOT re-run: `results/v2.6.5_s20union_sonnet{,_nothink_20260627}` on disk.
- TABOO: all rubric/trap text is generic English structure — no benchmark vocabulary.
