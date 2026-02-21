# Memory

## User Preferences
- **Always use Claude Sonnet** as the LLM backend (never opus). Set in `run_ablation.py` and linker constructors.
- **No dataset-specific examples in prompts** — data leakage. Use safe SE textbook domains (compiler, OS, e-commerce). See `BENCHMARK_TABOO.md`.
- User prefers standalone linker files (duplicate code intentionally, not inheritance chains)

## Current Best: V26a = 95.4% macro F1 (Feb 21)
- MS 98.4%, TS 100%, TM 94.2%, BBB 89.6%, JAB 94.7%
- 22 FP total (mostly TransArc baseline), 2 FN
- Key: synonym-safe judge bypass + ambiguous-only deliberation (advocate-prosecutor-jury)
- 13 stubborn TransArc FPs survive: TM(5), BBB(5), JAB(2), MS(1)

## V29 Experiment Series (Feb 21) — TransArc Judge Redesign
See [v29_experiments.md](v29_experiments.md) for full details.

| Version | Strategy | Macro F1 | Outcome |
|---------|----------|----------|---------|
| V29-intersect | ALL TransArc → decomposed 3-question judge, intersection voting | 77.7% | Catastrophic. Killed TPs everywhere. Q2 "not technology" wrong for this domain. |
| V29-union | Same prompt, union voting instead | 74.1% | Even worse. Prompt itself is the problem, not voting. |
| V29-gate | Widened ambiguous gate + softened 2-question prompt + intersection + safe examples | **94.2%** | Close to V26a. JAB 100%, TM 90.4%. |

## Key Architectural Lessons
- **Never judge ALL TransArc links** — non-ambiguous names (FreeSWITCH, kurento, WebRTC-SFU) get killed
- **Ambiguous gate is essential** but V26a's was too narrow (only single-word, non-CamelCase, non-uppercase)
- **V29's widened gate** catches ALL Phase 1 ambiguous + names with generic word parts
- **Decomposed sub-questions work** when scoped to ambiguous names only
- **Intersection voting safe** when only ambiguous names are judged (small batches)
- **Technology/tool rejection (P2) is wrong** for architecture docs — components ARE often named after technologies

## ILinkers (Feb 21)
- All CLEAN, no data leakage (audited). Safe SE textbook prompt examples.
- **I2**: 86.5-87.9% macro F1. High precision (~95%) but lower recall (~76%), especially BBB.
- **V26a+I2**: **92.7-95.7% macro F1**. V26a pipeline with I2 replacing TransArc as seed. Best pure-LLM approach — competitive with TransArc-based V26a (95.4%). V26a's coref/entity/validation phases recover I2's recall gap.

## LLM Variance (Critical Finding)
- Same model gives DIFFERENT behavior across days (Phase 1 ambiguity, Phase 3 synonyms)
- This is NOT code change — affects entire phases, not individual links
- V29 results vary by ~2-3pp across runs due to this

## Version History
See [linker_history.md](linker_history.md) for full table of all linker versions and results.

## Key Files
- `agent_linker_v26a.py` — Current best (95.4% macro)
- `agent_linker_v29.py` — V29 experiment (94.2% macro, widened gate variant)
- `BENCHMARK_TABOO.md` — Safe/taboo terms for prompt examples
- `run_ablation.py` — Ablation runner
- `model_analyzer.py` — Phase 1 component classification
- `llm_client.py` — LLM abstraction (claude/openai/codex)
