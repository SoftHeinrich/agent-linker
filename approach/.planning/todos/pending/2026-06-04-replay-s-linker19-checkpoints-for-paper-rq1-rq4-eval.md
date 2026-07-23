---
created: 2026-06-04T13:58:59.191Z
title: Replay s_linker19 checkpoints for paper RQ1-RQ4 eval
area: eval
files:
  - src/llm_sad_sam/linkers/experimental/s_linker19.py
  - results/phase_cache/s_linker19/{claude,openai}/<project>/{layer1,layer2,layer3,layer4,final}.pkl
  - writing/working/sections/eval.tex
  - writing/working/sections/results.tex
  - writing/working/sections/approach.tex
  - ../transarc-emp/src/lib/metrics_api.py
  - ../transarc-emp/src/lib/transarc_error_analysis.py
---

## Problem

The paper's `writing/working/sections/eval.tex` defines RQ1–RQ4 over link-level P/R/F1, per-component F1, sentence coverage, noise rate, validator contribution, and per-agent overlap. Current s_linker19 artefacts are only summary JSONs (`results/s19_*.json`) — the paper's `\todo{}` cells cannot be filled from those. The pipeline does, however, write full per-phase pickles to `results/phase_cache/s_linker19/{claude,openai}/<project>/{layer1,layer2,layer3,layer4,final}.pkl` for all 5 benchmark projects (mediastore, teastore, teammates, bigbluebutton, jabref) on both Claude Sonnet and gpt-5.4. The fields recorded — `framing_c_pass1`/`pass2`, per-key entity `decisions` with `p1`/`p2`/`approved`, `coref_raw`/`coref_validated`/`coref_decisions`, and `final_provenance` per link — are exactly what RQ3 and RQ4 measure. Results.tex itself already commits to offline replay framing for RQ3 ("logged decisions rather than re-running") and RQ4 ("set overlap rather than leave-one-out"). So the full RQ1–RQ4 eval is achievable with zero LLM calls.

Paper-side mismatches surfaced during the analysis (per [[feedback-code-is-canonical]], code is canonical):

1. **RQ4 linker count** — `approach.tex` describes 2 linkers (\linkerB named + \linkerC coref) and s_linker19 implements that shape, but `eval.tex` §exp:rq4 says 3 agents (Explicit/Contextual/Anaphoric) and `results.tex` §results:rq4 prose says 4 (canonical/alias/pronoun/partial). Eval+results are outdated and must be rewritten to the 2-linker shape.
2. **RQ3 NoConsensus wording** — `eval.tex` says "each extraction agent runs a single LLM pass instead of two voting passes; the vote is replaced by simply keeping every proposed link." The code still runs both passes; consensus removal = union of pass1, pass2 (intersection swapped for union). The "single LLM pass" phrasing is wrong; rewrite to match what the code actually does.
3. **Doc-to-code for RQ1** — s_linker19 emits doc-to-model only. Compose with the SAM→code mapping via transarc-emp infrastructure (the same path TransArc uses) to derive doc-to-code numbers.

## Solution

Promote to a phase (likely under `agent-linker/.planning/` since paper sections and s19 live here; the eval pipeline runs against `../transarc-emp/src/lib/metrics_api.py`).

Tasks (all stdlib + pickle reads, no LLM):

- **T5 — RQ1 doc-to-model + RQ2.** Adapter from `final.pkl` (final + final_provenance) → TransArc sad-sam result CSV format expected by `transarc-emp/src/lib/transarc_error_analysis.py::load_result_sad_sam_standalone`. Run `python3 ../transarc-emp/src/lib/metrics_api.py --task sad-sam` → produces `reports/metrics_sad-sam.csv` + `writing/tables/metrics_sad-sam.tex` for both Claude and gpt-5.4.
- **T9 — RQ1 doc-to-code.** Compose s19 (sentence → component_id) links with transarc-emp's SAM→code mapping (`load_code_model_files` + `load_gs_sam_code_maps`), emit per-project sad-code CSVs, run `metrics_api.py --task sad-code`. Same composition path TransArc uses → apples-to-apples comparison.
- **T6 — RQ3 validator counterfactuals.** Offline from `layer{2,3,4}.pkl`:
  - `NoConsensus`: `framing_c_pass1 ∪ framing_c_pass2` (was ∩). Pass through layer3+layer4 gates as logged (accept the lossy interaction with downstream validators — see decision below).
  - `NoEntityValid`: layer3 `candidates` (skip p1∧p2 gate).
  - `NoCitation`: layer4 `coref_raw` (skip coref validator).
  - `NoValidator`: composition of all three.
  - Per-validator gold-vs-spurious / killed-vs-kept counts feed `\autoref{fig:rq3-validator}`. Derived from `decisions` dicts on each candidate.
- **T7 — RQ4 set overlap on the 2-linker shape.** From layer3 (entity-validated) + layer4 (coref-validated) + gold:
  - `|entity ∩ gold|`, `|coref ∩ gold|`, unique-TP per linker, overlap-TP (both linkers found), `|entity ∪ coref ∩ gold|`.
  - Compact UpSet-style summary table.
- **T10 — Paper updates.** Rewrite `eval.tex` §exp:rq3 (NoConsensus wording → union) and §exp:rq4 (3 agents → 2 linkers); rewrite `results.tex` §results:rq4 prose (4 agents → 2 linkers); reconcile §results:rq3's "~2× LLM calls" claim with what's actually doubled (entity validator p1∧p2, not the two extraction passes).
- **T8 — Final report.** Both backends side-by-side, populate the `\todo{}` cells in results.tex. Single MD report under the phase dir.

**Decision points before execution:**
1. RQ3 NoConsensus replay strategy: (a) accept the union directly as the NoConsensus link set, or (b) re-run only Phase 4 on the union with `LLM_BACKEND=checkpoint` (replays cached calls; new calls would fail). Default to (a) for simplicity; document the assumption in T10.
2. Phase home: `agent-linker/.planning/` (paper-side close) vs. `transarc-emp/.planning/` (eval-side phase). Probably agent-linker since the paper sections live in `writing/working/` which is symlinked from agent-linker.

**Phase entry:** `/gsd-plan-phase` with this todo as the scope sketch.
