# Semantic gold for the rustc core chapters (`semgold/`)

Replaces the hyperlink/verbatim-name gold (`../build_dataset.py`, now one *vote* among
several) with a label model over semantic sources. Rationale and numbers:
`../../README.md` §8; literature behind the design: `../../LITERATURE.md` (synthesis
designs 1–3).

Pipeline (all paths relative to this directory; `PY=../../../../.venv/bin/python3`):

| step | script | input → output | LLM |
|---|---|---|---|
| 0 | `profiles.py` | `compiler/<crate>/src` → `out/profiles.json` (crate `//!` doc, README, module docs, modules, public items, files) | no |
| 0 | `symbols.py` | public items in `compiler/` → `out/symbols.json` (identifier → defining crates) | no |
| 0 | `cochange.py` | rust monorepo history of each chapter → `out/cochange_pairs.csv` | no |
| 1 | `evidence.py` | sentences + profiles + symbols → `out/evidence.json` (identifiers resolved, BM25 top-8 candidates) | no |
| 2 | `annotate.py --backend openai --model gpt-5.6-terra [--salt _r2]` | grounded sentence-view labels ABOUT/REFERS → `out/annotations_<backend>_<model>[salt].json` | yes |
| 2 | `annotate.py --backend claude_cli --model sonnet` | same, other model family | yes |
| 2' | `annotate_crateview.py --backend openai --model gpt-5.6-terra` | crate-view labels (component first, whole chapter) → `out/annotations_crateview_*.json` | yes |
| 3 | `label_model.py` | all votes → `out/semantic_labels.csv` (tiers), `out/gold_semantic*.csv`, `out/label_model_report.json` | no |
| 4 | `validate.py` | tiers vs anchors / co-change; `out/human_check_sheet.csv` (255 stratified pairs, blank verdict column) | no |
| 5 | `rescore.py <links.csv> --gold gold_plus|gold [--show-fp N --show-implicit N]` | linker output vs semantic gold | no |

Environment: `RUST_TREE` = `git archive` of `compiler/` at rust `a69a6326` (default
`/tmp/oss-case/rustc/tree`), `RUST_REPO` = full rust clone (co-change only). OpenAI calls
follow the paper's convention (terra, flex, `OPENAI_REASONING_EFFORT=none`, key passed as
`OPENAI_API_KEY="$OAI_KEY"`); the Claude family goes through the local `claude -p` CLI.
Every LLM response is cached under `cache/<backend>/<sha256(model,salt,prompt)>.json`, so
re-running a step is free and the dataset is reproducible from the cache without keys.

Tiers in `semantic_labels.csv`: `gold` = ABOUT by both families; `gold_plus_only` = ABOUT by
one family and supported by a deterministic vote (symbol index, anchor, co-change);
`silver` = ABOUT by one family, unsupported; `refers` = REFERS by either and not ABOUT.
`consistency` = share of three terra runs that reproduced the pair; `crateview` = the
crate-view annotator also said ABOUT. `gold_semantic.csv` (= gold ∪ gold_plus_only) is the
runner-format gold; `gold_semantic_strict.csv` (gold), `gold_semantic_3way.csv` (gold ∩
crate view), `gold_semantic_a2only.csv` (Claude family alone; robustness against the
linker's own family) are the alternatives.
