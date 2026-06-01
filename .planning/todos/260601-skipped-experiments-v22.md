---
created: 2026-06-01
status: parked
context: v2.2 milestone close — set of queued experiments skipped per user directive
backend_policy: prefer gpt-5.4; Claude only if super necessary
related:
  - .planning/quick/260601-bfe-ship-v2-2-as-s-linker13-min-opt-in-probe/260601-bfe-PLAN.md
  - .planning/v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md
  - .planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md
  - .planning/v2.2-prep/voyager-v2-rollup.md
---

# Skipped Experiments — v2.2 Close Wave

User directive 2026-06-01: skip remaining queued experiments. Items captured here for traceability so v2.3 (or later) can revisit deliberately.

## 1. Probe D Claude re-test with cache-key fix — SKIPPED (Claude backend)

- **Context:** Per-backend cache-key fix landed in `s_linker14_probe_d_upstream_clean.py` and was sanity-verified on gpt-5.4 BBB (SANITY_PASS, +1.12pp vs anchor). Original Range D Claude run reused gpt-5.4-authored cached rubric (cache-key bug). Methodologically ready to re-test on Claude with fresh per-backend cache.
- **Why skipped:** Claude backend run — new feedback rule prefers gpt-5.4. Cross-model cache-fix evidence not blocking v2.2 ship (s_linker13_min unchanged carries v2.1 Claude numbers; Probe D ships as opt-in gpt-5.4-only).
- **When to revisit:** If a v2.3+ canonical wants Probe D promoted to cross-model, this run becomes required (GATE-01 floor proof). Until then, gpt-5.4-only carve-out is sufficient.
- **Cost estimate:** ~$0.5-1 Claude (BBB Range only).
- **Reference:** `.planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md` (cache-fix code change + gpt-5.4 sanity verified).

## 2. Voyager v3 Claude — splits 2 & 3 — SKIPPED (Claude backend)

- **Context:** Voyager v3 = Claude-Sonnet sibling of v2 (gpt-5.4). Split 1 train converged, test ran (see `logs/voyager_v3_split1*.log`); splits 2 and 3 pending.
- **Why skipped:** Claude backend runs — new feedback rule prefers gpt-5.4. v2 (gpt-5.4) already declined for v2.2 (split-fragile, mean −0.05pp); waiting for splits 2+3 of v3 to potentially surface a Claude-vs-gpt-5.4 abstraction-tendency contrast is not required to start v2.3 design.
- **When to revisit:** If v2.3 Voyager v4 design needs cross-model evidence on the multi-role architecture, finish v3 first. Else permanently park.
- **Cost estimate:** ~$30-60 Claude (2 splits, ~$15-30 each).
- **Reference:** `scripts/voyager_train_tlr_v3.py`, `logs/voyager_v3_split1.log`, `.planning/v2.2-prep/voyager-v2-rollup.md` (gpt-5.4 sibling already declined).

## 3. trim1 baseline cache fill for Voyager rollup — DEFERRED (low priority)

- **Context:** Voyager v2 test_results.json records `s_linker13_trim1_F1_cached: null` for all 5 projects. Comparison vs trim1 in Voyager rollups falls back to the Phase 12 pilot number (0.9173). A re-fill would compare distilled-bank vs trim1 on the SAME run.
- **Why deferred:** Low signal-to-cost. Rollup already concludes Voyager underperforms trim1 by ~3pp; per-project trim1 cache only sharpens the comparison.
- **When to revisit:** Only if v2.3 wants a clean apples-to-apples per-project comparison in the publication artifact.
- **Cost estimate:** ~$1-2 gpt-5.4 (5 datasets × 1 short run with phase_cache hit) — acceptable under new rule.
- **Backend:** gpt-5.4 (does NOT violate new rule, but still parked as low priority).

## 4. Runtime-rubric pattern in other tiers (alias / extraction) — v2.3 SPIKE candidate

- **Context:** Probe D STRONG_PASS proves runtime LLM-built rubric → injected into a static-prompt slot. Same mechanism class is applicable to other slots: `ENTITY_EXTRACTION_RULES`, `DOC_KNOWLEDGE_JUDGE_RULES` (alias), `VALIDATION_RULES`.
- **Why not now:** v2.2 ship is closing scope-trimmed; spike work belongs in v2.3 `/gsd:explore` → `/gsd:spike` phase. Listed here so it isn't lost.
- **When to revisit:** v2.3 ideation (next step after this milestone closes).
- **Backend:** gpt-5.4.
- **Reference:** `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py` (template).

## 5. Voyager v4 multi-role architecture (deferred to v2.3, not skipped)

Not in this list because v2.3 anchor is exactly this. See `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md` (created by quick-ship Task 3).
