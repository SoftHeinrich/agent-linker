# Phase 43: Replay s_linker19 checkpoints for paper RQ1–RQ4 eval - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Populate every `\todo{}` cell in `writing/working/sections/{eval,results}.tex` for RQ1 (doc-to-model + doc-to-code), RQ3 (LLM-call validator counterfactuals), and RQ4 (2-linker overlap), and reconcile paper text with `s_linker19.py` where the prose disagrees with the code (per [[code-is-canonical]]). All numbers derive from existing `results/phase_cache/s_linker19/{claude,openai}/<project>/{layer1..4,final}.pkl` artefacts. **Zero new LLM calls.** GATE-01 invariants (`s_linker13_min` + `s_linker19` byte-equal at phase close) must hold.

**Out of phase 43:**
- RQ2 (metrics validation / bias pillar): owned by `evaluation/` (transarc-emp). Out of scope.
- Any change to `s_linker19.py` or `s_linker13_min.py` logic.
- Any LLM call (no `claude` CLI subprocess, no `openai` API key). `LLM_BACKEND` is not set.

</domain>

<decisions>
## Implementation Decisions

### Pipeline Layering (Area 1)

- **D-01:** **2-stage pipeline split across the two repos.**
  - **Replay stage (approach/):** `approach/scripts/v2.6.3/replay_s19_to_csv.py`, `replay_s19_rq3.py`, `replay_s19_rq4.py` unpickle `results/phase_cache/s_linker19/{claude,openai}/<project>/{layer1..4,final}.pkl` and emit CSVs to `approach/results/v2.6.3/{backend}/{project}/{sad-sam,sad-code,rq3,rq4}.csv`. These scripts live in `agent-linker` because pickle deserialization requires the `Link` dataclass from `src/llm_sad_sam/`.
  - **Format stage (evaluation/):** Existing `evaluation/src/lib/metrics_api.py --task sad-{sam,code}` reads the CSVs unchanged. New `evaluation/src/paper/rq3_table.py` and `evaluation/src/paper/rq4_table.py` render TeX from the RQ3/RQ4 CSVs to `writing/working/table/` and `writing/working/figures/`.
- **D-02:** **Stdlib-only constraint in `evaluation/` preserved.** Pickle is stdlib; the format-stage scripts do not import from `agent-linker`. No `pip install` introduced in `evaluation/`.

### Table & Figure Layout (Area 2 + Area 4)

- **D-03:** **RQ1 tables: wide, Claude-first.** 5 project rows (MediaStore, TeaStore, TeaMmates, BigBlueButton, JabRef) + Macro row. Two column groups: `Claude | GPT-5.4`. Backend labels: `Claude` and `GPT-5.4`. Applies to both `rq1` sad-sam and sad-code tables.
- **D-04:** **RQ3 & RQ4 main body = single backend, the other backend mirrors in appendix.** Main body backend = **Claude** (matches CLAUDE.md default-model policy and the Claude-first table convention from D-03; stronger numbers for s_linker19). Appendix carries the identical RQ3/RQ4 figures + tables for **GPT-5.4** as the cross-backend robustness check.
- **D-05:** **RQ4 per-linker table (2 rows: Entity, Coref).** Columns per linker: `TPs caught`, `Unique TPs`, `FPs`, `ΔF1-if-removed`. Footer row: `|Entity ∩ Coref ∩ gold|` = overlap-TP. Existing `writing/working/table/rq4-agents.tex` retrofitted from 4 rows (Canonical/Alias/Pronoun/Partial-name) to 2 rows (Entity/Coref).
- **D-06:** **RQ4 figure (UpSet) redrawn for 2 sets.** `writing/working/figures/rq4-upset.tex` collapses from 4-agent / 7-intersection UpSet to 2-linker / 3-cell UpSet: `only_E`, `both`, `only_C`. One subfigure per backend if main-body and appendix share a label; otherwise two separate figures.

### RQ3 Redesign (Area 3)

- **D-07:** **NoConsensus dropped. RQ3 covers only LLM-call validators.** Consensus voting (`pass1 ∩ pass2` at `s_linker19.py:637`) is an extractor design choice, not a validator — it sits inside `_run_framing_c` upstream of layer3. Including it as a separate variant would require reconstructing a counterfactual for the `pass1 △ pass2` symmetric-difference set (no cached layer3/layer4 decisions exist for those candidates), which conflicts with the "zero new LLM calls" constraint.
- **D-08:** **RQ3 variants = {Full, NoEntityValid, NoCitation, NoValidator}.** All four derive deterministically from pickle fields — no LLM calls, no counterfactual reconstruction:

  | Variant         | Definition                                              | Derivation                                                |
  | --------------- | ------------------------------------------------------- | --------------------------------------------------------- |
  | `Full`          | Entity-validator ON + Coref/Citation-validator ON       | `layer3.validated ∪ layer4.coref_validated` (= `final.pkl`) |
  | `NoEntityValid` | Skip layer3 entity validator                            | `layer3.candidates ∪ layer4.coref_validated`              |
  | `NoCitation`    | Skip layer4 coref/citation validator                    | `layer3.validated ∪ layer4.coref_raw`                     |
  | `NoValidator`   | Skip both LLM-call validators                           | `layer3.candidates ∪ layer4.coref_raw`                    |

- **D-09:** **RQ3 figure & table reduced to 2 validator rows.** `writing/working/figures/rq3-validator.tex` and `writing/working/table/rq3-validators.tex` retrofitted from 3 rows (Consensus voter / Entity valid. / Citation valid.) to 2 rows (`\entValidator{}` / `\corefValidator{}`), plus a combined-footer row.

### TeX Macros for Renameable Names

- **D-10:** **Add validator + variant macros to `writing/working/abbrev.tex`** so any rename is a single-file edit. Suggested:
  ```latex
  \newcommand{\entValidator}{entity validator\xspace}
  \newcommand{\corefValidator}{coref validator\xspace}
  \newcommand{\fullVariant}{Full\xspace}
  \newcommand{\noEntityValid}{NoEntity\xspace}
  \newcommand{\noCitation}{NoCitation\xspace}
  \newcommand{\noValidator}{NoValidator\xspace}
  ```
  All RQ3 prose, table headers, figure labels, and `\autoref` targets in eval.tex / results.tex use these macros. Existing `\linkerB` (Entity) and `\linkerC` (Coreference) are reused for RQ4.

### Paper Text Reconciliation (revised by D-07)

- **D-11:** **Four paper rewrites finalised.**
  1. **`eval.tex` §exp:rq3** — drop the `NoConsensus` bullet entirely (do **not** "rewrite to consensus = union of pass1 and pass2"); the variant is no longer in RQ3. Add a one-line note that consensus voting is kept inside `\fullVariant{}` as part of the extractor.
  2. **`eval.tex` §exp:rq4** — agent count `3 (Explicit/Contextual/Anaphoric) → 2 linkers (\linkerB{} entity + \linkerC{} coreference)`.
  3. **`results.tex` §results:rq4** — narrative `4 agents (canonical/alias/pronoun/partial) → 2 linkers (\linkerB{} + \linkerC{})`. UpSet decomposition reframed as `|only_E|`, `|both|`, `|only_C|`.
  4. **`results.tex` §results:rq3** — `"~2× LLM calls"` claim reconciled with what is actually doubled: the **entity validator's p1∧p2 evidence pattern** doubles entity-validation calls, not the framing-c extraction (which is also 2-pass but that is part of the extractor and was already in baseline cost). Consensus voting's 2-pass cost is in `\fullVariant{}`, not attributable to a validator.

### ROADMAP Success-Criteria Updates Needed at Plan Time

- **D-12:** Phase 43's ROADMAP entry (`.planning/ROADMAP.md` lines 143–) lists 8 success criteria authored before the RQ3 redesign. The following need updates during `/gsd-plan-phase 43`:
  - **Criterion #3 (RQ3 variants):** Replace 4-variant list (with NoConsensus) by 3 ablations + Full per D-08. Drop the explicit `framing_c_pass1 ∪ framing_c_pass2` derivation.
  - **Criterion #5 ("2× LLM calls"):** Reconcile per D-11 item 4 — the doubling is in the entity validator, not the framing-c extractor.
  - **Criterion #8 (NoConsensus replay docs):** **OBSOLETE** — no NoConsensus variant exists; criterion can be removed.

### Out of Scope (explicitly confirmed)

- **D-13:** **RQ2** is out of phase 43. The metrics-validation pillar lives in `evaluation/` (transarc-emp `src/bias/`) and produces its tables independently via `evaluation/src/paper/generate_tables.py`.
- **D-14:** **No algorithm changes** to `s_linker19.py` or `s_linker13_min.py` during this phase. GATE-01 byte-equality verified at phase close.

### Claude's Discretion

- Precise filenames inside `approach/scripts/v2.6.3/` are suggestions — planner can split differently as long as the 2-stage boundary (D-01) and stdlib-only constraint (D-02) hold.
- Exact macro names in `abbrev.tex` (D-10) are starter shapes; planner may align them to surrounding `\linkerA`/`\linkerB`/`\linkerC` conventions.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & gates
- `.planning/ROADMAP.md` §"Phase 43: Replay s_linker19 checkpoints for paper RQ1–RQ4 eval" (lines 143–168) — phase goal and 8 success criteria (criteria #3, #5, #8 to be revised per D-12 during planning).
- `.planning/STATE.md` — current milestone is v2.6.3 (paper-eval). Active phase 43 hint at line 169.
- `.planning/PROJECT.md` §"Core Value" — defines GATE-01 floors that must hold byte-equal across this phase.
- `.planning/REQUIREMENTS.md` — currently lacks v2.6.3-specific REQs; `/gsd-plan-phase 43` must add `REQ-V263-XX` entries reflecting D-01..D-14.

### Canonical artefact (frozen during this phase)
- `src/llm_sad_sam/linkers/experimental/s_linker19.py` — paper variant. Key landmarks:
  - `_run_framing_c` (line 621) — consensus voting at line 637; sets `_framing_c_pass1` / `_framing_c_pass2` on `self` for pickle export.
  - `_validate_with_evidence` (line ~690) — entity validator (layer3); returns `validated, entity_decisions`.
  - `_validate_coref_links` (line ~860) — coref/citation validator (layer4); returns `coref_validated, coref_decisions`.
  - Output composition (line 387): `all_links = entity_links + coref_validated`.
- `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` — frozen byte-equal during this phase (GATE-01).

### Phase-cache inputs (read-only)
- `results/phase_cache/s_linker19/{claude,openai}/{mediastore,teastore,teammates,bigbluebutton,jabref}/{layer1.pkl,layer2.pkl,layer3.pkl,layer4.pkl,final.pkl}` — 5 pickles × 5 projects × 2 backends = 50 pickle files. Source of every number in this phase.
- `layer2.pkl` fields: `framing_c`, `framing_c_pass1`, `framing_c_pass2`.
- `layer3.pkl` fields: `candidates`, `validated`, `decisions` (entity_decisions).
- `layer4.pkl` fields: `coref_raw`, `coref_validated`, `coref_decisions`.

### Evaluation toolkit (`evaluation/`, stdlib-only)
- `evaluation/src/lib/metrics_api.py` — entry point: `python3 src/lib/metrics_api.py --task sad-{sam,code}`. Consumes TransArc-format CSV; produces `reports/metrics_*.csv` + `writing/tables/metrics_*.tex`. **Used unchanged** for RQ1 doc-to-model + doc-to-code.
- `evaluation/src/lib/` — `load_code_model_files`, `load_gs_sam_code_maps` and other shared loaders for the doc-to-code composition.
- `evaluation/src/paper/generate_tables.py` — pattern reference for the new `rq3_table.py` / `rq4_table.py` to follow.
- `evaluation/README.md` — per-pillar script → report table.
- `evaluation/CLAUDE.md` — stdlib-only constraint, sys.path import pattern.

### Paper artefacts (writing/working — active dir per [[project_writing_dirs]])
- `writing/working/sections/eval.tex` — §exp:rq3 (lines 112–139) and §exp:rq4 (lines 140–158) to be rewritten per D-11.
- `writing/working/sections/results.tex` — §results:rq3 (lines 61–79) and §results:rq4 (lines 81–101) to be rewritten per D-11.
- `writing/working/abbrev.tex` — TeX macro file; add D-10 macros here.
- `writing/working/figures/rq3-validator.tex` — TikZ stacked-bar template; reduce to 2 rows per D-09.
- `writing/working/figures/rq4-upset.tex` — TikZ UpSet template; reduce to 2 sets per D-06.
- `writing/working/table/rq3-validators.tex` — table template; reduce to 2 rows per D-09.
- `writing/working/table/rq4-agents.tex` — table template; reduce to 2 rows per D-05.
- `writing/working/main.tex` — appendix structure (for D-04 GPT-5.4 mirror tables).

### Conventions & guardrails (`mono/`, both repos)
- `mono/CLAUDE.md` — default-model policy (Claude); benchmark-taboo; writing-dir conventions.
- `approach/BENCHMARK_TABOO.md` — applies to any new prompt code (not expected here, but if RQ3/RQ4 scripts surface example sentences in error reports, they must respect the taboo).
- `approach/CLAUDE.md` — approach-side repo rules.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`s_linker19.py` `_validate_with_evidence` / `_validate_coref_links` decision dicts** — every validator decision is keyed by `(sentence_number, component_id)` and stored in `layer3.pkl.decisions` / `layer4.pkl.coref_decisions`. The RQ3 replay reads `approved` and `path` fields directly; no recomputation needed.
- **`evaluation/src/lib/metrics_api.py`** — already produces `reports/metrics_sad-sam.csv`, `reports/metrics_sad-code.csv`, and the matching TeX tables. RQ1 plugs in via CSV input; no code change in evaluation/.
- **`evaluation/src/lib/load_code_model_files` + `load_gs_sam_code_maps`** — provide the SAM→code expansion needed for the doc-to-code composition (criterion #2).
- **`writing/working/abbrev.tex`** — existing macro conventions (`\approach`, `\linkerA`, `\linkerB`, `\linkerC`) extend cleanly to the validator + variant additions in D-10.
- **`writing/working/table/rq*.tex` + `writing/working/figures/rq*-*.tex`** — template skeletons already exist with `% Cells reported as 0 are not yet obtained` placeholders; phase 43 fills the cells and updates the row/column counts.

### Established Patterns
- **CSV → TeX two-step** — `evaluation/src/paper/generate_tables.py` reads CSVs from `reports/` and writes TeX into `writing/tables/`. New RQ3/RQ4 formatters follow the same pattern.
- **Pickle dataclass coupling** — pickle deserialization requires the originating module on `sys.path`. The replay scripts therefore live in `approach/`, not `evaluation/`.
- **Pickle file naming** — `{layer1,layer2,layer3,layer4,final}.pkl` per `{backend}/{project}`. Stable across all 5 projects × 2 backends.

### Integration Points
- **CSV handoff** — `approach/results/v2.6.3/{backend}/{project}/{sad-sam,sad-code,rq3,rq4}.csv`. This is the contract between replay-stage and format-stage; the planner should fix this schema before either side starts.
- **TeX writes** — RQ1 writes to `writing/working/tables/metrics_*.tex` (via `metrics_api.py`); RQ3/RQ4 writes to `writing/working/table/rq3-validators.tex` + `rq4-agents.tex` and `writing/working/figures/rq3-validator.tex` + `rq4-upset.tex` (in place, overwriting the placeholder zeros). Appendix tables/figures land under `writing/working/appendix/` (planner to confirm exact filename).
- **No `LLM_BACKEND` env var** — every replay script must hard-fail if any code path that could trigger an LLM call is reached. Defensive `assert "OPENAI_API_KEY" not in os.environ` and similar guards are reasonable.

</code_context>

<specifics>
## Specific Ideas

- **TeX macros for renameable names (D-10).** User explicitly asked for macros so validator/variant names can be renamed in one place. The macro file is `writing/working/abbrev.tex`.
- **Single-backend main body + appendix mirror (D-04).** User's explicit instruction: "for rq3,4, only use result of 1 backend and put the other one at appendix". RQ1 stays both-backends per D-03.
- **RQ3 redesign — drop NoConsensus (D-07/D-08).** User's explicit instruction: "remove the nonConsensus validator in that rq, only compare valadators that run LLM calls. lets resign the rq". This is a scope reshape, not creep: it narrows RQ3 to the LLM-call validator question and avoids reconstructing a counterfactual for which the pickles have no cached layer3/layer4 decisions.

</specifics>

<deferred>
## Deferred Ideas

- **NoConsensus as a standalone validator-design study.** If a future paper wants to ablate consensus voting, it should re-run `_run_framing_c` with `intersected = pass1 | pass2` (line 637) and re-fit layer3+layer4 against the union distribution. That requires new LLM calls and re-validation — explicitly out of phase 43 per D-07. Belongs in a future milestone, not v2.6.3.
- **Per-project breakdown for RQ3/RQ4 in main body.** Current RQ3/RQ4 tables aggregate across projects (`rq3-validators.tex` has one row per validator; `rq4-agents.tex` one row per linker). Per-project granularity is in the appendix mirror tables (GPT-5.4 backend) and could be added for Claude too if a reviewer asks. Out of scope unless review feedback demands it.
- **Consensus-voting stability metric.** `|pass1 ∩ pass2| / |pass1 ∪ pass2|` per project is a useful diagnostic (extractor stability), but is **not** an RQ3 ablation. If wanted, add as a side note in §approach.tex or appendix — out of phase 43.
- **Phase 37 / v2.6 close.** Per STATE.md, GATE-06 'Persistence' taboo fix and v2.6 audit are deferred. Unaffected by phase 43 but worth noting.

</deferred>

---

*Phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval*
*Context gathered: 2026-06-04*
