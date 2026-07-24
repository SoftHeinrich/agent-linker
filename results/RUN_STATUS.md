# Live-run status

This directory preserves the text evidence from experimental runs attempted
after the package was assembled. It is intentionally separate from the released
link artifacts in `sota-links/`; completed runs below are experimental evidence
and are not automatically part of the paper tables.

| Run | Variant / backend / dataset | Status | Preserved evidence |
| --- | --- | --- | --- |
| 2026-07-23 10:38 UTC | `s_linker21` / Claude / mediastore | Interrupted; runner wrote an empty CSV. | `approach/results/replication-one-run/` and `llm_logs/*103826*` |
| 2026-07-23 10:40 UTC | `s_linker21` / OpenAI / mediastore | Superseded before completion when the router-specific variant was selected. | `llm_logs/*104025*` |
| 2026-07-23 10:41 UTC | `s_linker21_agentrouter` / OpenAI GPT-5.4 / mediastore | Stopped after 14 successful calls when one coreference request remained unresponsive for more than 11 minutes; no final link CSV was produced. | `llm_logs/*104111*` and `s_linker21_agentrouter_openai_mediastore_20260723_104624*` |
| 2026-07-23 11:53 UTC | `s_linker23_verify1p_all` / GPT-5.6-terra / Flex / all five datasets | Invalid configuration attempt: API rejected legacy `temperature=0.1`; output is all-zero and is not a score. | `llm_logs/*115337*` and `s_linker23_verify1p_all_openai_mediastore_20260723_115350*` |
| 2026-07-23 12:03–12:08 UTC | `s_linker23_verify1p_all` / GPT-5.6-terra / Flex / explicit `reasoning_effort=none` / all five datasets | Completed N=1. `mini-src` macro P 82.31%, R 97.07%, F1 88.88%, F2 93.56%; 186 TP, 52 FP, 9 FN. | `s23_verify1p_all_gpt56terra_flex_noreasoning_central_20260723/` and matching `llm_logs/*1203*`–`*1208*` files |
| 2026-07-23 15:30–15:35 UTC | `s_linker23_verify` / GPT-5.6-terra / Flex / explicit `reasoning_effort=none` / all five datasets | Completed changed-phase checkpoint replay. `mini-src` macro P 81.58%, R 94.81%, F1 87.11%, F2 91.37%; 179 TP, 51 FP, 16 FN. Unchanged S21 prompts were cache hits; router/P2 prompts were fresh. | `s23_verify_gpt56terra_flex_noreasoning_changed_phase_replay_20260723/` and matching `s_linker23_verify_checkpoint_*` traces |
| 2026-07-23 15:40–15:45 UTC | `s_linker21` / OpenAI GPT-5.4 / Flex / explicit `reasoning_effort=none` / all five datasets | Completed fresh N=1 control. `mini-src` macro P 98.85%, R 90.74%, F1 94.42%, F2 92.14%; 170 TP, 3 FP, 25 FN. | `s21_gpt54_openai_flex_noreasoning_20260723/` and matching `s_linker21_openai_*` traces |
| 2026-07-23 15:59–16:10 UTC | `s_linker24` / OpenAI GPT-5.4 / enforced Flex / explicit `reasoning_effort=none` / all five datasets | Completed fresh N=1. `mini-src` macro P 97.31%, R 90.55%, F1 93.73%; pooled F2 88.82% (170 TP, 7 FP, 25 FN). Negative marginal result: resolver-approved anchored cases were all rejected by the inherited coref gate, so S24 made 0 additions. | `s24_gpt54_openai_flex_noreasoning_20260723/` |
| 2026-07-23 16:19–16:37 UTC | `s_linker24` dedicated anchored validator / OpenAI GPT-5.4 / enforced Flex / explicit `reasoning_effort=none` / all five datasets | **Invalid for full-score comparison.** BBB had 11 exhausted Flex 429s silently converted into empty phase outputs, collapsing its floor to 27 links. Marginal S24 evidence remains: 3 gold additions, 0 marginal FP from 40 eligible / 6 resolver-approved cases. | `s24_anchored_validator_gpt54_openai_flex_noreasoning_20260723/` |
| 2026-07-23 16:36–16:40 UTC | `s_linker21` / OpenAI GPT-5.6-terra / enforced Flex / explicit `reasoning_effort=none` / all five datasets | Completed fresh N=1 control. `mini-src` macro P 94.29%, R 91.70%, F1 92.77%; pooled F2 88.75% (172 TP, 17 FP, 23 FN). | `s21_gpt56terra_openai_flex_noreasoning_20260723/` |
| 2026-07-24 00:29–00:34 UTC | `s_linker24` dedicated anchored validator / OpenAI GPT-5.4 / enforced Flex / explicit `reasoning_effort=none` / teammates + BBB recovery | Completed fail-closed with zero failed calls. Combined with valid original mediastore, teastore, and jabref outputs: macro P 97.71%, R 91.96%, F1 94.67%; pooled F2 90.20% (173 TP, 6 FP, 22 FN). | `s24_anchored_validator_gpt54_openai_flex_noreasoning_tm_bbb_recovery_20260724/RESULTS.md` |
| 2026-07-24 00:59–01:02 UTC | `s_linker24_agentic` pilot / Codex controller+tools / saved S21 floors / all five datasets | Iteration 1 failed marginal precision: 8 TP, 4 FP, 66.67%; macro F1 still rose 93.34%→94.75%. Failure evidence retained rather than promoted. | `s24_agentic_tools_pilot_20260724/RESULTS.md` |
| 2026-07-24 01:03–01:05 UTC | `s_linker24_agentic` corrected pilot / Codex controller+tools / saved S21 floors / all five datasets | Passed: 6 TP, 0 FP, 100% marginal precision, four distinct plans; macro F1 93.34%→94.88% (+1.54pp). | `s24_agentic_tools_pilot_iter2_20260724/RESULTS.md` |
| 2026-07-24 01:06–01:08 UTC | promoted `s_linker24_agentic` / Codex controller+tools / saved S21 floors / all five datasets | Production class replay passed: 5 TP, 0 FP, 100% marginal precision, four distinct plans; macro F1 93.34%→94.68% (+1.34pp). | `s24_agentic_promoted_fixed_floor_20260724/RESULTS.md` |
| 2026-07-24 01:04–01:10 UTC | promoted `s_linker24_agentic` / Codex / mediastore | Normal end-to-end runner smoke passed. Same-run internal S21 floor F1 96.77%; alias tool added 1 TP / 0 FP; final F1 98.41% (+1.64pp). | `s24_agentic_codex_e2e_mediastore_20260724/RESULTS.md` |
| 2026-07-24 09:55–10:05 UTC | semantic appeal pilot / Codex / saved S21 rejected candidates / all five datasets | **Invalidated.** Iteration 1: 7 TP / 12 FP and macro F1 below S21. Identity/ownership iteration: 5 TP / 6 FP, above S21 but below dynamic S24. User then rejected the refine-not-replace architecture. | `s24_semantic_appeal_pilot_iter1_20260724/` and `s24_semantic_appeal_pilot_iter2_identity_20260724/` |
| 2026-07-24 10:10–10:22 UTC | replacement orchestrator pilot / Codex controller+audit / recorded phase tools / all five datasets | **Passed F2-first objective.** 174 TP / 9 FP / 21 FN; macro F1 94.01%, macro F2 92.96%, pooled F1 92.06%, pooled F2 90.34%. S21 comparison: macro F2 90.83%, pooled F2 88.14%. | `s24_replacement_orchestrator_pilot_all_iter1_20260724/` |
| 2026-07-24 10:40–10:45 UTC | `s_linker24_orchestrator` / Codex / mediastore | Fresh replacement smoke completed from raw inputs: 31 TP / 1 FP / 0 FN, F1 98.41%, F2 99.36%. Audit safely accepted no additions after fresh entity coverage. | `s24_orchestrator_codex_e2e_mediastore_20260724/` |
| 2026-07-24 11:05–11:08 UTC | replacement orchestrator feedback-parity replay / Codex / recorded phase tools / all five datasets | **Failed before scoring.** Full raw phase feedback recursively enlarged controller history until the Codex subprocess failed with `Argument list too long`. | `s24_replacement_orchestrator_pilot_all_iter3_feedback_parity_20260724/FAILURE.md` |
| 2026-07-24 11:10–11:16 UTC | replacement orchestrator compact-feedback replay / Codex / recorded phase tools / all five datasets | **Passed F2 objective, but route hypothesis failed.** 175 TP / 12 FP / 20 FN; macro F2 93.23%, pooled F2 90.49%; same route on all projects. Full outputs retained, controller state normalized. | `s24_replacement_orchestrator_pilot_all_iter4_compact_feedback_20260724/` |
| 2026-07-24 11:20–11:28 UTC | replacement orchestrator participation-audit replay / Codex / recorded phase tools / all five datasets | **Passed causal F2 hypothesis.** 181 TP / 12 FP / 14 FN; macro F1 94.50%, macro F2 94.75%, pooled F1 93.30%, pooled F2 93.01%. Recovered predicted relational, negated, multi-target, and structural-discourse cases. Same route; route diversity remains unvalidated. | `s24_replacement_orchestrator_pilot_all_iter5_participation_20260724/` |

The `.jsonl`, `.log`, and `*_calls.json` files are text provenance: they record
the request sequence, model, response status, and phase-level trace. The local
phase caches are deliberately excluded because they are binary, transient cache
files rather than reportable results.

To obtain a score from `mini-src`, a completed run must first produce a
`<variant>_<project>_links.csv` file. The command is:

```bash
TRANSARC_BENCHMARK="$PWD/benchmark" \
python3 evaluation/mini-src/metrics.py --task sad-sam --project mediastore \
  --results-dir approach/results/<run-dir> \
  --result-pattern '<variant>_{project}_links.csv'
```
