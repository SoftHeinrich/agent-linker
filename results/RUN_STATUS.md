# Live-run status

This directory preserves the text evidence from live runs attempted after the
package was assembled. It is intentionally separate from the released link
artifacts in `sota-links/`: none of the runs below is a completed result that
should be used in the paper tables.

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
