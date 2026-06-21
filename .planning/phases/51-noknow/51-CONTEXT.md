# Phase 51: NOKNOW - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the No-Knowledge ablation **real**: add a knowledge-disable path to
`s_linker20_union` (skips the alias table + ambiguity map), run it on
5 projects × {gpt, sonnet} × N=3, and extract the results into the **same**
neutral JSON schema Phase 50 produced — so the scorer treats Full and
No-Knowledge cells uniformly.

This is the **only LLM-bound, live-call** phase in milestone v2.6.6. All other
phases replay frozen caches. Satisfies NOKNOW-01 (disable path + GATE-01) and
NOKNOW-02 (live runs + neutral extracts).

**In scope:** the `no_knowledge` flag on the variant; its `run_ablation.py`
registration; the live N=3 sweep on both backends; extraction of those runs
into the Phase-50 JSON shape; GATE-01 proof that flag-off behavior is unchanged.

**Out of scope:** the metric core / scorer (Phase 52), RQ3/RQ4 computation
(Phases 53/54), `../working/` bundle (Phase 55), and re-running/re-tuning the
frozen Full runs.

</domain>

<decisions>
## Implementation Decisions

### Disable mechanism
- **D-01:** Knowledge-disable is a **constructor flag** on the existing class —
  `SLinker20Union(no_knowledge=True)`. The default-off path must be
  snapshot-identical to today's behavior. (Chosen over a standalone
  `s_linker20_noknow.py` copy and over a thin subclass: duplicating the
  ~1086-line frozen union for a two-field toggle invites drift, and a subclass
  is an inheritance chain the project disfavors. NOKNOW-01 literally says
  "behind a flag/variant"; GATE-01 constrains full-knowledge *behavior*, not
  source bytes.)
- **D-02:** Surface the No-Knowledge config as a **registered sibling variant**
  `s_linker20_union_noknow` in `run_ablation.py` — a registration entry that
  constructs `SLinker20Union(no_knowledge=True)`, **no linker logic duplicated**.
  This yields a distinct `--variants` name and a distinct `_links.csv` prefix,
  so the results tree and the Phase-50 extractor separate Full vs No-Knowledge
  cells without any hidden out-of-band env toggle.

### Disable scope (what `no_knowledge=True` does)
- **D-03:** **Skip the 3 layer1 LLM call-sites entirely** — do not call the
  ambiguity prompt (`_learn_world_knowledge` / `_prompt_ambiguity`) nor the
  doc-knowledge extract+judge prompts (`_learn_document_knowledge` /
  `_prompt_doc_knowledge_extract` + `_prompt_doc_knowledge_judge`). Set
  `model_knowledge = ModelKnowledge()` (∅ `ambiguous_names`) and
  `doc_knowledge = DocumentKnowledge()` (`{}` `aliases`) directly. Still call
  `_save_phase(..., "layer1", {...})` with the empty knowledge so the phase_cache
  / extract **shape** is identical to Full.
- **D-04:** Downstream degrades **automatically** — 2-pass alias-injected entity
  extraction runs with empty mappings; the structural alias-aware antecedent
  gate matches canonical component names only (no aliases); coref has no known
  aliases. No downstream code change beyond passing empty knowledge. (Confirm
  during research that no *other* knowledge source leaks in — there should be
  none per GATE-06: zero benchmark-derived vocabulary anywhere.)

### Run depth & output annotation
- **D-05:** **N=3, symmetric with the Full extracts** — 30 live runs total
  (5 projects × 2 backends × 3 runs). This **supersedes the N≥1 floor** in
  NOKNOW-02 and intentionally pulls the deferred `NOKNOW-N` variance work into
  this milestone, giving variance bands on the knowledge axis for a clean
  Full(N=3) vs No-Knowledge(N=3) A/B in RQ4-02.
- **D-06:** **Output folders + extracts MUST be annotated as no-knowledge** and
  kept distinct from the Full tree (e.g. a `…_noknow` results root and a
  `noknow`-annotated extract tree parallel to `results/v2.6.6_extracts/`).
  Exact path naming is Claude's discretion as long as Full vs No-Knowledge is
  unambiguous on disk and to the extractor.

### Execution & cost
- **D-07:** **Phase 51 runs the sweep end-to-end** within the phase:
  build the flag → verify GATE-01 → launch the 30-run live sweep → extract into
  neutral JSON. (User chose this over the Phase-48 "ship scripts, user triggers"
  gating pattern — no separate go-ahead checkpoint.)
- **D-08:** **Soft cost cap ≈ $60** (in the v2.5 $80 / Phase-48 $20 range).
  Log cumulative live-call cost to the sweep PROGRESS log (success criterion 2
  requires cost logged); **do not hard-abort** — the sweep is unattended and
  resumable via per-(run,dataset) `.done` markers. User reviews the logged total
  after. No-Knowledge runs are cheaper per run than Full (3 fewer LLM call-sites).

### GATE-01 evidence
- **D-09:** Prove flag-off == today's `s_linker20_union` via a **zero-LLM
  snapshot replay against the 30 frozen phase_caches**: with the flag OFF,
  replay and assert the produced `layer1` (`ambiguous_names` + `aliases`) and
  final `_links.csv` are byte-identical to the frozen Full outputs — reusing
  Phase 50's faithfulness-oracle lineage (set-equality on `(sentence,
  component_id, source)`). Plus a **structural check** that the only source
  change is a strictly additive branch guarded by `if self.no_knowledge:`, so
  the flag-off path executes the identical pre-existing statements.

### Claude's Discretion
- Exact No-Knowledge results/extract directory names (D-06) — any scheme that is
  unambiguously annotated and machine-distinguishable from Full.
- Whether GATE-01 replay covers all 30 cells or a representative subset
  sufficient to prove byte-equality (the structural guard backstops it).
- Run-script ergonomics (cooldowns, retry-once, resume markers) — mirror the
  existing `run_s20union_{gpt,sonnet}_n3.sh` skeleton.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §"Phase 51 — NOKNOW" — goal + 3 success criteria
  (disable flag with GATE-01 snapshot check; 5×{gpt,sonnet}×N≥1 run with cost
  logged; extract into Phase-50-identical neutral JSON).
- `.planning/REQUIREMENTS.md` §"No-Knowledge Ablation (NOKNOW)" — NOKNOW-01,
  NOKNOW-02; plus §"Future Requirements" `NOKNOW-N` (N≥3 variance — now pulled
  forward by D-05) and the §"Out of Scope" rows (no s19 recompute, no Full
  re-run, no canonical-linker behavior change).
- `.planning/PROJECT.md` §"Current Milestone v2.6.6" + §"Standing gates" —
  GATE-01 (canonical/paper artifacts byte-/snapshot-stable, incl. full-knowledge
  `s_linker20_union`), GATE-06 (no benchmark-derived vocabulary), PARITY.

### Output contract (what the extract must match)
- `.planning/phases/50-extract/50-01-SUMMARY.md` — the neutral JSON schema +
  serialization decisions the No-Knowledge extracts MUST reproduce (coref as
  lists not dict-keyed maps; `aliases` as list-of-records; `raw_resolution`
  omitted; `final.links` authoritative from `final.pkl`).
- `.planning/phases/50-extract/50-RESEARCH.md` and `50-01-PLAN.md` — extractor
  design + open-question rationale (read if reusing/extending the extractor).
- `scripts/extract_s20union_caches.py` — the extractor to extend to the
  No-Knowledge results root (`load_cell`, `to_neutral`, `write_json`,
  `keyed_to_records`, faithfulness oracle, `rederive_final` guard). Note its
  cell-discovery roots are currently hardcoded to `results/v2.6.5_s20union*` —
  a No-Knowledge root must be added.

### Linker getting the flag
- `src/llm_sad_sam/linkers/experimental/s_linker20_union.py` — the variant.
  Knowledge layer: `_run_parallel({model, doc})` (~L502) → `_save_phase(...,
  "layer1", ...)` (~L511); ambiguity builder `_learn_world_knowledge` /
  `_prompt_ambiguity` (~L623); alias builder `_learn_document_knowledge`
  (~L632, extract ~L639 + judge ~L662); empty-knowledge types `ModelKnowledge`
  / `DocumentKnowledge`. Downstream consumers: 2-pass extraction (~L684),
  structural antecedent gate (~L749), coref (`COREF_RULES`).
- `run_ablation.py` — variant registry; add the `s_linker20_union_noknow` entry.

### Run-script templates
- `run_s20union_gpt_n3.sh`, `run_s20union_sonnet_n3.sh` (repo root, currently
  untracked) — the N=3 sweep skeleton to mirror: per-(run,dataset) `.done`
  resume markers; `PHASE_CACHE_DIR` / `LLM_LOG_DIR` / `CHECKPOINT_DIR` env
  routing per run; light→heavy dataset order; cooldowns + retry-once. Backends:
  gpt `LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4`; sonnet `LLM_BACKEND=claude
  CLAUDE_MODEL=sonnet`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase-50 extractor** (`scripts/extract_s20union_caches.py`): `to_neutral`,
  `write_json` (`json.dump(indent=2, sort_keys=True, allow_nan=False)`),
  `keyed_to_records`, the set-equality **faithfulness oracle**, and the
  `rederive_final` guard all transfer directly to No-Knowledge cells — extend
  the discovery roots rather than re-implement.
- **N=3 sweep scripts** (`run_s20union_{gpt,sonnet}_n3.sh`): copy as the
  No-Knowledge sweep skeleton; change variant → `s_linker20_union_noknow`,
  results root → annotated `…_noknow`, and add cumulative cost logging (D-08).

### Established Patterns
- **PHASE_CACHE_DIR per-run isolation** gives genuine N independence — each run
  writes its own `phase_cache`; reuse so No-Knowledge N=3 runs are independent.
- **Single registration import** registers pickled classes without LLM side
  effects (Phase-50 extractor pattern) — the No-Knowledge extract path needs the
  same `import …s_linker20_union` to resolve cache classes.
- **Strictly-additive guarded branch** is the GATE-01-safe way to add behavior
  to a frozen artifact (D-09): all new logic under `if self.no_knowledge:`.

### Integration Points
- `run_ablation.py` variant registry — new `s_linker20_union_noknow` entry
  constructing the flagged instance.
- `s_linker20_union.py` `_run_parallel` knowledge-learning site (~L502) — the
  single guard point where `no_knowledge` short-circuits to empty knowledge.
- Extractor cell-discovery roots — add the annotated No-Knowledge results tree
  so the 30 No-Knowledge cells extract into the Phase-50 JSON shape.

</code_context>

<specifics>
## Specific Ideas

- User explicitly: **"the output folder should annotate as no-knowledge"** —
  the No-Knowledge results + extract trees must be visibly/machine-distinctly
  labeled (D-06), never co-mingled with the Full extracts.
- User explicitly chose **N=3 symmetric** (D-05) and **phase runs the sweep
  end-to-end** (D-07) with a **soft cap, log-and-continue** cost posture (D-08).

</specifics>

<deferred>
## Deferred Ideas

- **Pending axiom/prompt-design todos** (6 in `.planning/todos/pending/`, e.g.
  "design better axioms for section-context", "refined v3-style axiom diffs",
  "prompts_v4 three root-cause FP fixes") — these are v2.6.1/v2.6.2 axiom-era
  work, **out of scope** for v2.6.6 eval infrastructure. Not folded.
- **NOKNOW-N as a *separate* requirement** is effectively absorbed by D-05
  (N=3 this milestone); REQUIREMENTS.md should be updated at phase close to
  reflect that the N≥3 variance work landed here rather than being deferred.

</deferred>

---

*Phase: 51-noknow*
*Context gathered: 2026-06-21*
