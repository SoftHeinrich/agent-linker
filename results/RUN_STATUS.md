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
