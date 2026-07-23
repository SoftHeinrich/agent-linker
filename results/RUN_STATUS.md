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
