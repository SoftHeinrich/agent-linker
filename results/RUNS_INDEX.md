# Run results index

Catalog of `agent-linker/results/` (decluttered layout). Generated, non-destructive. `results/` is gitignored.

Layout: canonical + cache + `manual/` at top level (stable paths scripts depend on); superseded runs under `experimental/`; loose probe files under `scratch/`.

**19 run dirs**, 28 scratch files. CANONICAL=paper-feeding · EXPERIMENTAL=superseded · CACHE=regenerable · MANUAL=ad-hoc.

## CANONICAL

| location | version | variant | backend | knowledge | files | size | consumer | note |
|---|---|---|---|---|---|---|---|---|
| `v2.6.5_s20union` | v2.6.5 | s20union | gpt | full | 181 | 4.9M | none (noenroll.py retired 2026-08-26) | raw N=3 doc-model CSV runs |
| `v2.6.5_s20union_sonnet` | v2.6.5 | s20union | sonnet | full | 180 | 4.9M | none (noenroll.py retired 2026-08-26) | raw N=3 doc-model CSV runs |
| `v2.6.6_extracts` | v2.6.6 | s20union | gpt+sonnet | full | 30 | 3.1M | build_unified.py (recovered-links) | clean doc-model link set (final.links) |
| `v2.6.6_extracts_noknow` | v2.6.6 | s20union | gpt | noknow | 15 | 1.5M | - | clean noknow extracts (gpt only; sonnet pending) |
| `v2.6.6_s20union_noknow` | v2.6.6 | s20union | gpt | noknow | 108 | 1.1M | - | raw noknow sweep (gpt only) |

## MANUAL

| location | version | variant | backend | knowledge | files | size | consumer | note |
|---|---|---|---|---|---|---|---|---|
| `manual` | - | s20union | gpt+sonnet | full+noknow | 519 | 15M | - | ad-hoc hand-runs incl. early noknow |

## EXPERIMENTAL

| location | version | variant | backend | knowledge | files | size | consumer | note |
|---|---|---|---|---|---|---|---|---|
| `experimental/s19_proxy_ablation` | v2.6.5 | s19_proxy | - | full | 3 | 16K | - | proxy ablation probe |
| `experimental/v2.6.5_s19U` | v2.6.5 | s19U | gpt | full | 180 | 4.9M | - | un-minimized parent; s19U-vs-s20union head-to-head |
| `experimental/v2.6.5_s19U_sonnet` | v2.6.5 | s19U | sonnet | full | 180 | 4.8M | - | un-minimized parent (sonnet) |
| `experimental/v2.6.5_s20union_gpt_re_medium` | v2.6.5 | s20union | gpt | full | 180 | 5.4M | - | reasoning-effort=medium re-run probe |
| `experimental/v2.6.5_union_aliasb` | v2.6.5 | s20union_aliasb | gpt+sonnet | full | 27 | 184K | - | aliasb prompt-swap variant |
| `experimental/voyager_v4_beta` | v2.6.x | voyager | - | - | 8 | 40K | - | voyager skill-bank training probe |
| `v2.6.5` | v2.6.5 | mixed | mixed | full | 162 | 1.1M | - | earlier per-project ablations - IN PLACE (host-owned uid1000, unmovable) |

## CACHE

| location | version | variant | backend | knowledge | files | size | consumer | note |
|---|---|---|---|---|---|---|---|---|
| `ablation_results` | - | - | - | - | 21 | 112K | run_ablation.py | per-run ablation metric JSON |
| `llm_checkpoint` | - | - | - | - | 36 | 288K | LLM_BACKEND=checkpoint | replay cache |
| `llm_checkpoint_dotted_sonnet` | - | - | - | - | 1 | 48K | - | replay cache (probe) |
| `llm_logs` | - | - | - | - | 1017 | 54M | run_ablation.py (writes) | raw LLM call logs (regenerable) |
| `llm_sessions` | - | - | - | - | 0 | 28K | - | session traces |
| `phase_cache` | - | - | - | - | 341 | 4.2M | run_ablation.py | pickled layer caches (regenerable) |

## scratch/ (28 files)

Ad-hoc probe outputs; no fixed consumer.

- `scratch/4b_prompt_absorption_20260604_063756.json`
- `scratch/ablation_evjudge_20260605_125900.json`
- `scratch/ablation_evjudge_20260605_131354.json`
- `scratch/ablation_evjudge_20260605_131543.json`
- `scratch/ablation_evjudge_min_20260605_140801.json`
- `scratch/ablation_evjudge_rest_20260608_151656.json`
- `scratch/ablation_evjudge_rest_20260608_154944.json`
- `scratch/analyze_framings_openai_gpt54_20260604_031126.json`
- `scratch/analyze_framings_openai_gpt54_20260604_031228.json`
- `scratch/analyze_framings_openai_gpt54_20260604_031247.json`
- `scratch/analyze_framings_openai_gpt54_20260604_031401.json`
- `scratch/c5pass_coref_gpt54_20260604_050600.json`
- `scratch/c5pass_gpt54_20260604_044641.json`
- `scratch/compute_chain_20260604_061334.json`
- `scratch/coref_c2_eval.csv`
- `scratch/coref_through_twopass.csv`
- `scratch/dotted_path_rename_20260604_072803.json`
- `scratch/dotted_path_rename_20260604_072850.json`
- `scratch/dotted_path_rename_20260604_073336.json`
- `scratch/feasibility_study_20260604_055810.json`
- `scratch/rc1_alias_promotion_test.csv`
- `scratch/rc1_embed_cache.json`
- `scratch/rc1_embedding_test.csv`
- `scratch/rc4_union_test.csv`
- `scratch/s19_clean_20260604_065728.json`
- `scratch/s19_sonnet_post_rename_20260604_075606.json`
- `scratch/smoketest_mediastore_1780542216.json`
- `scratch/verify_steps_20260604_063028.json`
