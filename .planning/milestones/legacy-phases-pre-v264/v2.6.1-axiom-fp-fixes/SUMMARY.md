---
milestone: v2.6.1
phase: v2.6.1-01
title: Axiom FP Root-Cause Fixes
status: complete
---

# Phase v2.6.1-01 SUMMARY

**One-liner:** Three axiom FP root-cause fixes (tier/platform alias, code-path prefix,
functional-alias-as-workflow) baked into s_linker15; validated combined + attributed against gold —
fixes fire on Claude (TM FP 17→6), inert on GPT-5.4.

FP attribution (cached links vs gold, zero cost): Claude removed all 6 Cause-A targets (UI×4 +
GAE×2), cleared Cause-C cluster (5+→1), caught 2/3 Cause-B (Storage@125 residual). GPT kept every
target (reads nuanced rules literally). Isolated per-fix ablation not run (combined + attribution
sufficient). Commit: 8b5601c.
