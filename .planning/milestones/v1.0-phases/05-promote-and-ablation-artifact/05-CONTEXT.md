# Phase 5: Promote and Ablation Artifact - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning
**Mode:** `gsd-discuss-phase --auto` (Claude-selected recommended defaults; no human Q&A — every decision below is locked with a cited source)

<domain>
## Phase Boundary

Promote the **chain-winning variant `s_linker13f`** (full-sweep macro F1 = **0.9509**, +1.04pp over 12c; Phase 4 Plan 04-02 SUMMARY canonical JSON `results/ablation_results/ablation_20260529_215932.json`) into the canonical `s_linker13.py`; formally log the `_has_standalone_mention` KEEP decision in `PROJECT.md`; produce the ablation table (markdown primary + LaTeX paper-ready via `tabulate`) covering every variant in the chain (12c → 13a → 13b → 13c → [13d retired] → 13e → 13f → **13**); and ship a 2-3 page methodology writeup documenting the rule-removal chain, the standing-policy history (BBB tolerance loosening 2pp→4pp→6pp), the 13d failure mode, and the dual-hard-tier protocol used on 13e.

**In scope:**
- Promotion mechanism (PROMO-01): copy `s_linker13f.py` → `s_linker13.py`, rename class to `SLinker13`, set `_VARIANT_NAME = "s_linker13"`, register in `run_ablation.py` (`CANONICAL_VARIANTS` + `VARIANT_SPECS`) as the new canonical entry, update structured docstring to record the cumulative `RULES_REMOVED` list across the chain.
- `_has_standalone_mention` KEEP decision (PROMO-02): append a formal row to `PROJECT.md` Key Decisions table, citing Spike 002's RISKY classification (O(N×M) anchor collection) and noting EXT-01/EXT-02 as the v2 follow-up spikes.
- Ablation table (PROMO-03): one row per variant with per-dataset F1 (MS, TS, TM, BBB, JAB), macro F1, ΔF1 vs 12c (macro), ΔF1 vs immediate structural parent (macro), `rules_removed` list, status (PASS / RETIRED / RETIRED-as-rejection-artifact / PROMOTED). Markdown primary; LaTeX rendered from the same `tabulate` data source.
- Methodology writeup (PROMO-04): markdown, target 2-3 pages, sections covering: rule-removal methodology, six-variant chain narrative, standing-policy evolution (BBB tolerance history), 13d failure mode (dotted-path classification, milestone-level finding), dual-hard-tier protocol used for 13e (widest blast radius), retained-primitive rationale for `_has_standalone_mention`.

**Out of scope (this phase):**
- Any new variant sweeps (no `s_linker13g`, no re-runs of 12c/13a/13b/13c/13d/13e/13f). The ablation table is **assembled from existing JSONs** in `results/ablation_results/` (canonical IDs cited in D-49).
- Replacing `_has_standalone_mention` with an LLM primitive (EXT-01) or dropping its dotted-path guard (EXT-02) — both v2 / deferred per `.planning/STATE.md` "Deferred Items".
- GPT-5.2 cross-model re-evaluation of `s_linker13` (EXT-03 — v2 deferred per same STATE.md row; PROJECT.md constraint "Claude Sonnet only").
- Per-row FP-by-phase breakdown (seed / entity / coref) as a **mandatory** column. Per D-49d below: the FP-by-phase column appears for 13c, 13d, 13e, 13f (where Phase 2-4 SUMMARYs already record it) and is blank for 12c / 13a / 13b / **13** (no per-phase breakdown captured at sweep time). Best-effort, not a gate.
- Modifying `run_ablation.py` to deprecate 13a-13f. **The 13a-13f files remain in the tree as ablation/rejection artifacts** (D-43b); only `s_linker13` is added as the canonical promotion.
- Milestone closure (`gsd-complete-milestone` / `gsd-audit-milestone` are post-Phase-5 steps, not in scope).

</domain>

<decisions>
## Implementation Decisions

### Winner Identification (D-43)
- **D-43:** **Winner = `s_linker13f`.** Full-sweep macro F1 = **0.9509** (canonical JSON `results/ablation_results/ablation_20260529_215932.json`, dated 2026-05-29). This is the **best macro F1 in the entire 13-series chain** (12c=0.9405, 13a=0.9364, 13b=0.9519¹, 13c=0.9314, 13d=RETIRED, 13e=0.9380, 13f=0.9509) and the **last variant that holds the dual floor** under standing policy (BBB 6pp tolerance, other-dataset 2pp, macro ≥ 0.93). **Source:** `.planning/STATE.md` lines 8, 30, 31, 38, 40 ("Winner candidate: 13f"); `.planning/phases/04-alias-scope-and-coref-fold/04-02-SUMMARY.md` §"Full Sweep" + §"VAR-06 Outcome"; `.planning/ROADMAP.md` Phase 5 (line 88, "winning variant (the last 13x that holds dual floor)").

  ¹ Note: 13b's macro (0.9519) is **higher** than 13f's (0.9509) on raw number, but 13b is a **mid-chain intermediate** — it removed only one helper (`_is_structurally_unambiguous`) and 13c (its successor) regressed BBB beyond the original 2pp tolerance, triggering the user-loosened 6pp BBB tolerance. The **chain-winner definition per ROADMAP Phase 5 is the LAST variant that holds the dual floor** (i.e., the variant with the most rules removed that still passes), not the variant with the highest macro number. 13f satisfies both criteria simultaneously: it is the latest in the cumulative-removal chain AND has the highest macro of any variant that passed under the standing 6pp BBB policy. The methodology writeup (PROMO-04, D-50) explicitly notes 13b's higher raw macro and explains why 13f is the artifact-of-record.

- **D-43a:** **13d is NOT a parent in the canonical promotion chain.** Phase 3 closed empty (STATE.md 2026-05-29); VAR-04 retired. The chain is **12c → 13a → 13b → 13c → 13e → 13f → 13** (`s_linker13f` was forked from `s_linker13e`, which was forked from `s_linker13c`). 13d remains in the tree as the rejection artifact (D-43b) and gets a `RETIRED` row in the ablation table (D-49b). **Source:** `.planning/STATE.md` §"Phase 3 Closure Note"; `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` §"User Resolution".

- **D-43b:** **All of `s_linker13a.py`, `s_linker13b.py`, `s_linker13c.py`, `s_linker13d.py`, `s_linker13e.py`, `s_linker13f.py` REMAIN in the tree.** They are the rejection / intermediate / chain artifacts referenced by the ablation table. **Do NOT delete any.** Each file remains registered in `run_ablation.py` so the table can be regenerated by re-reading the canonical JSONs at any time. Only `s_linker13.py` is **added** (D-44). **Source:** orchestrator instructions ("Keep 13a-13f files in tree (do NOT delete) — they are the rejection/intermediate artifacts referenced by the ablation table"); MEMORY.md "User prefers standalone linker files (duplicate code intentionally, not inheritance chains)"; Phase 4 D-31c precedent ("`s_linker13d.py` remains in the tree as the rejection artifact").

### Promotion Mechanism (D-44)
- **D-44:** **Promotion is a clean file copy + class/constant rename + append-only registration.** Steps:
  1. `cp src/llm_sad_sam/linkers/experimental/s_linker13f.py src/llm_sad_sam/linkers/experimental/s_linker13.py`
  2. Rename the class inside the new file from `SLinker13f` → **`SLinker13`** (replace_all on the class name; verify no references break).
  3. Set `_VARIANT_NAME = "s_linker13"` (was `"s_linker13f"`).
  4. Update the structured module docstring:
     - `REMOVED_FROM:` should record the cumulative chain — recommended phrasing: `REMOVED_FROM: s_linker12c (cumulative via 13a→13b→13c→13e→13f)`. The planner may choose a single-parent (`s_linker13f`) phrasing if preferred — **planner discretion (D-44d)**, both pass GATE-03.
     - `RULES_REMOVED:` should list the cumulative set of structural helpers retired across the chain — recommended list: `["_split_component_name (13a partial)", "_is_structurally_unambiguous (13b)", "_is_ambiguous_name_component (13c)", "_is_strong_alias (13e)", "_get_strong_alias_mappings (13e)", "_has_strong_alias_mention (13f)"]`. **Six structural rule removals total** (VAR-04 retired); the methodology writeup (D-50) records the same list.
     - `KEEP:` add an explicit field — `KEEP: ["_has_standalone_mention (Spike 002 RISKY; O(N×M); see PROJECT.md Key Decisions)"]` — to mirror the PROMO-02 keep-decision log (D-46).
  5. Update the `__init__` print-banner string from `"s_linker13f"` → `"s_linker13"` (banner discipline precedent: Phase 1 Plan 05 §Deviations; Phase 2-4 §Claude's Discretion).
  6. Register `s_linker13` in `run_ablation.py` `CANONICAL_VARIANTS` AND `VARIANT_SPECS` — **append-only after `s_linker13f`** (same shape used in Phase 1-4 plans; **do NOT replace** 13f or any of 13a-13f). The new entry is the **canonical promotion** (mark it as such in the spec dict, e.g., a `"canonical": True` flag or equivalent — **planner discretion (D-44d)** on the exact key name; both V32/S-Linker convention precedent allows either approach).

  **Source:** ROADMAP §Phase 5 success criterion #1 ("`s_linker13.py` exists … registered in `run_ablation.py`"); GATE-02; GATE-03; Phase 1 D-04 (append-only registration); Phase 2 D-11a, Phase 3 D-25a, Phase 4 D-37a (registration shape precedent).

- **D-44a:** **The promotion is byte-equivalent to `s_linker13f` modulo (a) `_VARIANT_NAME` constant, (b) class name, (c) docstring, (d) banner string.** No semantic changes. The full-sweep numbers for `s_linker13` are **defined to be 13f's full-sweep numbers** — **no re-sweep is required or performed in Phase 5** (D-48). If a future audit wants to verify `s_linker13` numerically matches 13f, that is a one-shot sanity run, not a Phase 5 deliverable. **Source:** orchestrator instructions ("No sweep work in Phase 5"); Phase 1 D-02 (single-run baseline principle, no re-runs without reason).

- **D-44b:** **Per-variant pickle cache (`results/phase_cache/s_linker13/`) is created lazily on first use.** Per the D-07 runtime assertion (Phase 1 INFRA-05), the namespace is independent from `results/phase_cache/s_linker13f/`. No pre-population, no cache copy from 13f. **Source:** Phase 1 D-07; Phase 4 D-37 (per-variant namespacing precedent).

- **D-44c:** **The `s_linker13` class docstring also records the promotion provenance** — recommended phrasing in the docstring body (free-form, not in the structured `REMOVED_FROM:` / `RULES_REMOVED:` / `KEEP:` fields): "`s_linker13` is the canonical promotion of `s_linker13f` (Phase 5, 2026-05-29). Full-sweep macro F1 = 0.9509 (`ablation_20260529_215932.json`). See `.planning/phases/05-promote-and-ablation-artifact/05-METHODOLOGY.md` for the rule-removal chain narrative." **Source:** Phase 1 Plan 05 §Deviations precedent (provenance-in-docstring); orchestrator instructions ("update run_ablation registration as new CANONICAL").

- **D-44d (Claude's Discretion):** **Exact wording / shape choices** — `REMOVED_FROM:` chain-vs-single-parent phrasing, class-name capitalization (`SLinker13` vs `s_linker13` — recommend `SLinker13` for class-name consistency with `SLinker12c`/`SLinker13f`), the `"canonical": True` registration flag name, and the banner-string format — are **planner's call** during implementation. Constraints fixed in D-44 / D-44a / D-44c; precise wording free. **Source:** Phase 1-4 §Claude's Discretion precedent.

### `_has_standalone_mention` KEEP Decision (D-45)
- **D-45:** **`_has_standalone_mention` is formally KEPT in `s_linker13`.** PROMO-02 satisfied via a new row in the `PROJECT.md` Key Decisions table. Recommended row text:

  | Decision | Rationale | Outcome |
  |----------|-----------|---------|
  | KEEP `_has_standalone_mention` in `s_linker13` | Spike 002 classified it RISKY (O(N×M) anchor-collection; would need full-component-list × full-sentence-list scan if replaced by LLM call). Phase 5 confirms KEEP — replacement is deferred to v2 (EXT-01 spike) with a more aggressive cost-benefit analysis under a relaxed budget. EXT-02 (drop dotted-path guard) is a narrower follow-up also deferred to v2. | KEPT (Phase 5, 2026-05-29) |

  The new row is **appended** to the existing Key Decisions table (`.planning/PROJECT.md` lines 60-69) — do not edit existing rows. **Source:** ROADMAP Phase 5 success criterion #2 ("PROJECT.md Key Decisions table contains a formal KEEP entry for `_has_standalone_mention` referencing Spike 002's O(N×M) classification"); REQUIREMENTS.md PROMO-02 ("KEEP decision formally logged as Key Decision in PROJECT.md"); `.planning/STATE.md` §"Deferred Items" rows EXT-01 / EXT-02.

- **D-45a:** **The methodology writeup (D-50) restates the KEEP rationale in narrative form** under a "Retained Primitive" section. The writeup reference for the docstring `KEEP:` field (D-44 step 4) is `.planning/phases/05-promote-and-ablation-artifact/05-METHODOLOGY.md` §"Retained Primitive: `_has_standalone_mention`". **Source:** ROADMAP Phase 5 success criterion #4 ("retained-primitive rationale for `_has_standalone_mention`"); REQUIREMENTS.md PROMO-04.

- **D-45b:** **The KEEP decision references — but does not reproduce — Spike 002's O(N×M) analysis.** The PROJECT.md row points readers to `.planning/spikes/002-rules-audit/` (full classification of all 12 helpers; `_has_standalone_mention` is the lone RISKY entry). The methodology writeup likewise cites Spike 002 by directory, not by transclusion. **Source:** Phase 4 D-32c / D-39 precedent (spike references by directory, not body-text).

- **D-45c:** **EXT-01 (LLM replacement of `_has_standalone_mention`) and EXT-02 (drop dotted-path guard) are explicitly marked deferred to v2** in the writeup, with a one-line forward-pointer to `.planning/STATE.md` "Deferred Items" so future-milestone work has a discoverable handle. **Source:** `.planning/STATE.md` "Deferred Items" rows; `.planning/REQUIREMENTS.md` §"v2 Requirements" (EXT-01, EXT-02, EXT-03).

### Ablation Table — Format & Columns (D-46)
- **D-46:** **The ablation table is markdown-primary (assembled in code via `tabulate(..., tablefmt="github")` or `"pipe"`) AND LaTeX-paper-ready (same data, rendered via `tabulate(..., tablefmt="latex")` or `"latex_booktabs"` — planner picks the cleaner one).** Both outputs share a single Python source-of-truth (a `dict` / `list[dict]` literal with the per-row data; see D-49 for canonical numbers). **Source:** REQUIREMENTS.md PROMO-03 ("Output: markdown + LaTeX via `tabulate`"); ROADMAP Phase 5 success criterion #3 ("Ablation table exists in both markdown and LaTeX (`tabulate` output)"); Phase 1 D-06 / Phase 2 D-17 / Phase 3 D-29 / Phase 4 D-41 (`tabulate` is the dependency-of-record for table rendering).

- **D-46a:** **Columns (in order):**
  1. **`variant`** — `s_linker12c`, `s_linker13a`, `s_linker13b`, `s_linker13c`, `s_linker13d`, `s_linker13e`, `s_linker13f`, `s_linker13`
  2. **`parent`** — the immediate structural parent (e.g., 13e is `s_linker13c` because 13d is RETIRED and not in the chain; 13f is `s_linker13e`; **13** is `s_linker13f`)
  3. **`rule removed`** — short, human-readable description of what this variant removed vs its parent (e.g., 13b: `_is_structurally_unambiguous`; 13d: `_classify_mention (regex → LLM enum)`). For 12c, the entry is `— (baseline)`. For **13**, the entry is `— (promotion: cumulative chain)`.
  4. **`MS`** — mediastore F1 (per-dataset)
  5. **`TS`** — teastore F1
  6. **`TM`** — teammates F1
  7. **`BBB`** — bigbluebutton F1
  8. **`JAB`** — jabref F1
  9. **`macro`** — macro F1 across the 5 datasets
  10. **`Δ vs 12c`** — macro F1 delta vs 12c baseline (positive = improvement)
  11. **`Δ vs parent`** — macro F1 delta vs the immediate structural parent (per-row parent in column 2); blank for 12c (no parent) and for 13d (parent is 13c but row is RETIRED so the delta is the rejection delta; **show it anyway** as the rejection-magnitude metric)
  12. **`status`** — one of `BASELINE` (12c), `PASS` (13a, 13b, 13c, 13e, 13f), `RETIRED` (13d, with note "TM −0.188 vs 12c, dotted-path classification failure"), `PROMOTED` (13)
  13. **`FP-by-phase`** — seed / entity / coref FP breakdown for the row; **best-effort only — D-46b**; blank rows are acceptable

  **Source:** orchestrator instructions ("columns: variant, rule removed, per-dataset F1, macro F1, Δ vs 12c"); REQUIREMENTS.md PROMO-03 ("one row per variant … per-dataset F1, ΔF1 vs parent, rules-removed list, FP-by-phase"); Phase 4 D-41 (ablation row schema precedent); Phase 1 D-12, Phase 2 D-12, Phase 3 D-24b, Phase 4 D-36b (dual-comparator: vs 12c for GATE, vs parent for sanity).

- **D-46b:** **`FP-by-phase` column is best-effort.** Phase 2 / Phase 3 / Phase 4 SUMMARY files record some per-row FP-by-phase numbers (e.g., Phase 1 Plan 05 SUMMARY notes the 13a BBB breakdown: 2 seed FPs / 1 entity FP / 0 coref FP). For rows where the SUMMARY does not record a clean breakdown, the cell is **blank** (do not fabricate; do not re-derive by re-running). The methodology writeup (D-50) notes the FP-by-phase column is partial. **Source:** REQUIREMENTS.md PROMO-03 ("FP-by-phase (seed/entity/coref)"); orchestrator instructions ("No sweep work in Phase 5 — use the existing JSONs from Phases 1-4 ablation_results/ as data source").

- **D-46c:** **The 13d row is included with status `RETIRED`** and a footnote: "VAR-04 retired 2026-05-29 — TM F1 = 0.750 (Δ −0.188 vs 12c) on dotted-path regression. LLM enum classifier cannot reproduce the project-specific Java-package convention (`ui.website`, `logic.api`, `storage.entity`) encoded in 12c's regex `_classify_mention`. Milestone-level finding: classification of language-construct references is regex territory; the no-hand-crafted-rules thesis holds with this caveat." The hard-tier-only numbers from 13d (no full-sweep was run — Phase 3 hard-rejected) are used; MS/TS/JAB columns are blank for 13d. **Source:** `.planning/STATE.md` §"Phase 3 Closure Note (empty)"; `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md`; REQUIREMENTS.md VAR-04 (struck via retirement).

- **D-46d:** **Methodology writeup includes the table inline** (markdown body uses the same `tabulate` markdown render). The LaTeX render is exported to a separate file (recommended: `.planning/phases/05-promote-and-ablation-artifact/05-ABLATION.tex`) so it can be dropped into a paper directly. **Source:** orchestrator instructions ("Ablation table format: markdown (primary) + LaTeX (paper-ready)"); ROADMAP Phase 5 success criterion #3.

### Ablation Table — Canonical Per-Row Data (D-47)
- **D-47:** **Canonical per-row numbers (locked from the existing JSONs; no re-sweep):**

  | row | variant | parent | rule removed | MS | TS | TM | BBB | JAB | macro | Δ vs 12c | Δ vs parent | status | source JSON |
  |-----|---------|--------|--------------|----|----|----|-----|-----|-------|----------|-------------|--------|-------------|
  | 1 | `s_linker12c` | — | — (baseline) | 0.984 | 0.963 | 0.938 | 0.844 | 0.973 | **0.9405** | 0.000 | — | BASELINE | `ablation_20260528_173020.json` (+ BBB re-run `ablation_20260514_185017.json`) |
  | 2 | `s_linker13a` | 12c | Spike 001 LLM trailing-word enrichment (partial: 0 aliases added on TM+BBB; cache-stream perturbation -2.7..-4.8pp on BBB) | 1.000 | 0.982 | 0.923 | 0.804 | 0.973 | **0.9364** | -0.0041 | -0.0041 | PASS (under loosened BBB 4pp tolerance, 2026-05-28) | `ablation_20260528_173020.json` (Phase 1 Plan 05) |
  | 3 | `s_linker13b` | 13a | `_is_structurally_unambiguous` (post-filter) | 0.984 | 0.973 | 0.947 | 0.839 | 1.000 | **0.9519** | +0.0114 | +0.0155 | PASS | `ablation_20260528_190916.json` (Phase 2 Plan 02-01) |
  | 4 | `s_linker13c` | 13b | `_is_ambiguous_name_component` (wrapper inline-remove) | 0.967 | 0.953 | 0.929 | 0.7818 | 0.973 | **0.9314** | -0.0091 | -0.0205 | PASS (under loosened BBB 6pp tolerance, 2026-05-29) | `ablation_20260528_201851.json` (Phase 2 Plan 02-02) |
  | 5 | `s_linker13d` | 13c | `_classify_mention` (4-regex → LLM enum, Spike 003) | — | — | 0.750 | — | — | — | (TM Δ −0.188) | (TM Δ −0.179) | RETIRED — dotted-path failure | `ablation_20260529_110532.json` (Phase 3 Plan 03-01, hard-tier only) |
  | 6 | `s_linker13e` | 13c | `_is_strong_alias` + `_get_strong_alias_mappings` (alias `scope: global\|local` LLM field) | 0.984 | 0.963 | 0.939 | 0.804 | 1.000 | **0.9380** | -0.0025 | +0.0066 | PASS | `ablation_20260529_201324.json` (Phase 4 Plan 04-01) |
  | 7 | `s_linker13f` | 13e | `_has_strong_alias_mention` (fold into coref evidence as `antecedent_via_alias`) | 0.984 | 1.000 | 0.947 | 0.821 | 1.000 | **0.9509** | +0.0104 | +0.0129 | PASS | `ablation_20260529_215932.json` (Phase 4 Plan 04-02) |
  | 8 | **`s_linker13`** | 13f | — (promotion: cumulative chain, 6 rules removed) | 0.984 | 1.000 | 0.947 | 0.821 | 1.000 | **0.9509** | +0.0104 | 0.0000 | PROMOTED | (defined as 13f per D-44a) |

  All numbers come from the canonical JSONs cited in the table's right column. **No re-runs in Phase 5.** Per-row source JSONs are also listed in `<canonical_refs>` below. **Source:** `.planning/STATE.md` lines 36-46 (Phase 4 closure numbers); `.planning/phases/04-alias-scope-and-coref-fold/04-01-SUMMARY.md` (13e full sweep); `.planning/phases/04-alias-scope-and-coref-fold/04-02-SUMMARY.md` §"Full Sweep" (13f full sweep); `.planning/phases/02-ambiguity-cleanup/02-01-SUMMARY.md` (13b full sweep); `.planning/phases/02-ambiguity-cleanup/02-02-SUMMARY.md` (13c full sweep); `.planning/phases/01-baseline-and-infrastructure/01-05-SUMMARY.md` (13a full sweep); `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` (13d hard-tier only).

- **D-47a:** **If any per-row number in D-47 conflicts with the source JSON when the planner inspects it directly, the JSON wins.** The numbers in D-47 are transcribed from SUMMARY files (which themselves cite JSONs). A discrepancy would indicate a transcription error; the planner re-derives from the JSON and updates the table. **No new sweeps are triggered by a discrepancy.** **Source:** Phase 1 D-02 (single-source-of-truth principle for baseline numbers).

- **D-47b:** **The `Δ vs parent` column for `s_linker13e` is computed against `s_linker13c`** (13c is 13e's parent because VAR-04 / 13d is retired and not in the chain — D-43a). For 13c → 13e: macro Δ = 0.9380 − 0.9314 = +0.0066 (positive, since 13e actually improves over 13c; the 12c-comparator gate is the relevant constraint). **Source:** Phase 4 D-31c; Phase 4 D-36b; orchestrator instructions ("the 13-series chain: 12c → 13a → 13b → 13c → [13d retired] → 13e → 13f").

### No-Sweep Boundary (D-48)
- **D-48:** **Phase 5 performs ZERO new linker sweeps.** All ablation table numbers come from existing `results/ablation_results/` JSONs (canonical IDs in D-47 + `<canonical_refs>`). The promoted `s_linker13.py` is byte-equivalent to `s_linker13f.py` modulo D-44a edits; its numbers are 13f's numbers. **Source:** orchestrator instructions ("No sweep work in Phase 5 — use the existing JSONs from Phases 1-4 ablation_results/ as data source"); D-44a above; Phase 1 D-02 + Phase 2 D-10 + Phase 3 D-23 + Phase 4 D-35 (no-re-run-baselines precedent compounds into Phase 5).

- **D-48a:** **If the planner notices a stale or missing number during table assembly, the resolution is to (i) re-read the source JSON, (ii) update D-47, (iii) escalate to user if the JSON itself is corrupt or absent.** Do NOT re-run the variant. **Source:** Phase 1 D-02; orchestrator instructions ("No sweep work in Phase 5").

- **D-48b:** **`s_linker13.py` is NOT swept on the 5-project benchmark in Phase 5.** A future audit may want a one-shot sanity sweep to confirm byte-equivalent numbers vs 13f; that sweep — if needed — is a milestone-closure step (`gsd-audit-milestone`) or a v2 deliverable, not Phase 5. **Source:** D-44a; orchestrator instructions.

### Ablation Artifact File Layout (D-49)
- **D-49:** **All Phase 5 deliverables live under `.planning/phases/05-promote-and-ablation-artifact/`.** Recommended file layout (planner discretion on exact filenames within these constraints):

  - `.planning/phases/05-promote-and-ablation-artifact/05-CONTEXT.md` — this file (locked decisions).
  - `.planning/phases/05-promote-and-ablation-artifact/05-RESEARCH.md` — research notes (if any; researcher-agent output).
  - `.planning/phases/05-promote-and-ablation-artifact/05-01-PLAN.md` — plan for D-44 promotion mechanism + D-45 KEEP decision logging.
  - `.planning/phases/05-promote-and-ablation-artifact/05-02-PLAN.md` — plan for D-46 ablation table (markdown + LaTeX) + D-50 methodology writeup. **Planner may collapse 05-01 + 05-02 into a single plan if the workload is small enough — Claude's Discretion (D-49c).**
  - `.planning/phases/05-promote-and-ablation-artifact/05-ABLATION.md` — the markdown ablation table artifact (PROMO-03). **This file is referenced by the methodology writeup AND is a standalone artifact.**
  - `.planning/phases/05-promote-and-ablation-artifact/05-ABLATION.tex` — the LaTeX-rendered ablation table artifact (PROMO-03). Paper-drop-in ready.
  - `.planning/phases/05-promote-and-ablation-artifact/05-METHODOLOGY.md` — the 2-3 page methodology writeup (PROMO-04). **This is the milestone deliverable** per orchestrator instructions ("the writeup is the milestone deliverable"). The `s_linker13.py` docstring (D-44c) forward-points to this file.

  The actual `s_linker13.py` linker file lives at `src/llm_sad_sam/linkers/experimental/s_linker13.py` (D-44), NOT under `.planning/`. **Source:** orchestrator instructions ("File placement: `.planning/phases/05-promote-and-ablation-artifact/` for the ablation table + methodology writeup … `s_linker13.py` lives in `src/llm_sad_sam/linkers/experimental/`"); Phase 1-4 directory layout precedent.

- **D-49a:** **Table-generation script** — the planner writes a small Python helper (recommended: `.planning/phases/05-promote-and-ablation-artifact/render_ablation.py` or invoked from a notebook cell, **planner's discretion D-49c**) that holds the per-row data dict (D-47) and renders BOTH markdown (`05-ABLATION.md`) AND LaTeX (`05-ABLATION.tex`) from the single source-of-truth. The script is committed alongside the artifacts so the table is regeneratable. **Source:** REQUIREMENTS.md PROMO-03 ("Output: markdown + LaTeX via `tabulate`"); Phase 1 D-06.

- **D-49b:** **The `Δ vs 12c` and `Δ vs parent` columns are computed from the per-dataset F1 values in the row, NOT hardcoded.** This catches transcription errors at render time. **Source:** D-47a (source-of-truth principle); Phase 1 Plan 04 SUMMARY §"Macro F1 re-computation" precedent.

- **D-49c (Claude's Discretion):** Plan split (one vs two plans), the exact script filename, the exact LaTeX `tabulate` format flag (`"latex"` vs `"latex_booktabs"` vs `"latex_raw"`), and whether the `render_ablation.py` script lives in the phase directory or in a `scripts/` directory at repo root — all planner's call. Constraints fixed in D-49 / D-49a / D-49b.

### Methodology Writeup — Scope (D-50)
- **D-50:** **`05-METHODOLOGY.md` is target 2-3 pages of markdown** (roughly 1500-2500 words). Required sections (planner picks the exact headings and ordering; recommended structure below):

  1. **Background & Goal** (~1/4 page) — Restate the core value (every rule removed must hold macro F1 ≥ 93% or be rejected); cite PROJECT.md "Core Value" and the deliverable framing ("defensible claim that traceability linking can be done without hand-crafted structural rules").
  2. **Methodology** (~1/2 page) — Rule-removal protocol: standalone variant files per rule (D-08 / D-31 precedent); BENCHMARK_TABOO audit on every new prompt (GATE-04); hard-tier-first → full-sweep gate sequence (GATE-05); dual-comparator deltas (vs 12c for GATE, vs parent for sanity); single-run sweeps (no N-run median). Brief mention of the dual-hard-tier protocol for VAR-05 (widest blast radius — see §D-50c below).
  3. **The Six Removals** (~3/4 page) — Narrative walking through 13a (Spike 001 trailing-word LLM, BBB cache-stream perturbation), 13b (clean removal of `_is_structurally_unambiguous`), 13c (wrapper inline-remove, BBB drift triggered tolerance loosening), **13d (RETIRED; milestone-level lesson — see §D-50d)**, 13e (alias `scope: global|local` LLM field, dual-hard-tier protocol applied), 13f (alias-coref fold via `antecedent_via_alias`; best-in-chain macro 0.9509). Includes the ablation table (inline reference to D-47 / `05-ABLATION.md`).
  4. **Standing-Policy History** (~1/4 page) — BBB tolerance evolution: original 2pp (Phase 1 INFRA-01 design) → 4pp (user-loosened 2026-05-28 after 13a BBB regression, Phase 1 Plan 05 closure) → 6pp (user-loosened 2026-05-29 after 13c BBB drift, Phase 2 Plan 02-02 closure). Discusses the cache-stream-timing hypothesis (D-13a) and the empirical evidence supporting it (13b pure-removal staying within the original 2pp BBB band; 13c byte-identical-classification still drifting on BBB). Frames the loosened tolerance as a documented limitation of Claude run-to-run variance on BBB's multi-word component partials (HTML5 Client/Server, WebRTC-SFU), NOT a methodology weakness.
  5. **The 13d Failure Mode** (~1/4 page) — Detailed account of the dotted-path classification failure: LLM enum classifier cannot reproduce the project-specific Java-package convention (`ui.website`, `logic.api`, `storage.entity`) → 33 entity-source FPs on TM → TM F1 = 0.750 (Δ −0.188 vs 12c). Frames it as a **milestone-level finding**: classification of language-construct references is regex territory; the no-hand-crafted-rules thesis **holds with this caveat**. Explicitly notes that the writeup does NOT claim 100% rule elimination — 1 of 7 attempted removals (`_classify_mention`) was retired empirically.
  6. **The Dual-Hard-Tier Protocol for VAR-05** (~1/4 page) — Why 13e (widest blast radius) ran twice on hard tier before full sweep: schema break touching 6 read sites; LLM-substitution-of-project-specific-rule risk profile analogous to VAR-04. Reports the dual-run inter-variance numbers (Run 1 BBB 0.826, Run 2 BBB 0.818, |Δ|=0.008 — well within the 0.04 band). Frames the protocol as the empirical mitigation for the "widest blast radius" criterion the ROADMAP set for VAR-05.
  7. **Retained Primitive: `_has_standalone_mention`** (~1/4 page) — KEEP rationale (Spike 002 RISKY classification, O(N×M) anchor-collection cost). Mirrors the PROJECT.md Key Decisions row (D-45). Forward-points to EXT-01 (replacement spike) and EXT-02 (drop dotted-path guard) as deferred v2 work.
  8. **Result** (~1/8 page) — `s_linker13` ships with **six structural rule removals** (vs the original seven targeted; VAR-04 retired). Macro F1 = 0.9509 (+1.04pp over 12c baseline). Defensible claim achieved with one documented caveat.

  Total: ~2.5 pages of content. **Planner picks exact headings, ordering, and prose; structural decisions (which sections appear, what each covers) are LOCKED above.** **Source:** REQUIREMENTS.md PROMO-04 ("Research writeup (markdown) documenting methodology, results, and the retained-primitive rationale"); ROADMAP Phase 5 success criterion #4 ("Research writeup (markdown) documents methodology, the promotion chain, and the retained-primitive rationale for `_has_standalone_mention`"); orchestrator instructions ("Methodology writeup scope: ~2-3 pages, including the standing policy history (BBB tolerance loosening), the 13d failure mode (dotted-path classification), the dual-hard-tier protocol for 13e (widest blast)").

- **D-50a:** **The writeup quotes specific numbers from the ablation table** (D-47) and from prior SUMMARY files. It does NOT introduce new numbers, new sweeps, or new claims. **Source:** D-48 (no-sweep); orchestrator instructions.

- **D-50b:** **The writeup is paper-style narrative, not GSD-style mechanical.** Plain English, citation-by-footnote-or-inline-reference, no tables-with-checkboxes. Voice: third-person research report (the chain of variants; the dual-hard-tier protocol was applied …). **Planner discretion (D-50e)** on tone particulars within these constraints. **Source:** REQUIREMENTS.md "Research writeup (markdown)"; precedent: MEMORY.md §"V31 Final Audit", §"V31 Phase Contribution Analysis", §"GPT-5.2 Cross-Model Evaluation" — these are the project's voice for results writeups.

- **D-50c:** **Dual-hard-tier protocol narrative cites Phase 4 D-35a / D-39 / `04-01-SUMMARY.md` §"Dual Hard-Tier Results"** as the source for the protocol design and the empirical inter-run variance numbers. **Source:** `.planning/phases/04-alias-scope-and-coref-fold/04-CONTEXT.md` §"Baseline Protocol (D-35)" + §"LLM-Substitution Inertness Risk (D-39)"; `.planning/phases/04-alias-scope-and-coref-fold/04-01-SUMMARY.md` §"Dual Hard-Tier Results (Tasks 2 + 3, per D-35a)".

- **D-50d:** **13d failure-mode narrative cites `03-01-SUMMARY.md` §"Failure-Mode Analysis" and `.planning/STATE.md` §"Phase 3 Closure Note (empty)"** as the canonical accounts. The dotted-path examples (`ui.website`, `logic.api`, `storage.entity`) are quoted directly from STATE.md. **Source:** `.planning/STATE.md` lines 43-46; `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` §"Failure-Mode Analysis".

- **D-50e (Claude's Discretion):** Heading wording, paragraph ordering within each section, footnote vs inline citation style, and whether to include a "Future Work" section forward-pointing to EXT-01/EXT-02/EXT-03 (recommended: yes, brief) — planner's call. Constraints fixed in D-50.

### Plan-Phase Scope (D-51)
- **D-51:** **Phase 5 plans cover (in this order, planner may collapse to single plan per D-49c):**
  1. **PROMO-01 (D-44):** Promote `s_linker13f.py` → `s_linker13.py`; rename class; update `_VARIANT_NAME`, docstring (incl. `KEEP:` field), banner; register in `run_ablation.py` (append-only after `s_linker13f`, mark as canonical per D-44).
  2. **PROMO-02 (D-45):** Append KEEP-decision row to PROJECT.md Key Decisions table.
  3. **PROMO-03 (D-46, D-47, D-48, D-49):** Generate `05-ABLATION.md` (markdown) and `05-ABLATION.tex` (LaTeX) from the per-row data dict (D-47) via `tabulate`. Commit the small render script alongside.
  4. **PROMO-04 (D-50):** Write `05-METHODOLOGY.md` (2-3 pages, sections per D-50).

  **No new sweeps, no new linker variant creation, no new spike work.** **Source:** REQUIREMENTS.md PROMO-01..PROMO-04; ROADMAP Phase 5 success criteria #1-4.

- **D-51a:** **GATE-01..GATE-06 do not apply to Phase 5** in the usual variant-promotion sense. ROADMAP §"Quality Gates" line 9 names "Phases 2, 3, 4, and 5" — but Phase 5's gate is the **artifact-correctness gate**: the promoted `s_linker13.py` is byte-equivalent to `s_linker13f.py` modulo allowed edits (D-44a); the ablation table numbers match the canonical JSONs (D-47a); the methodology writeup covers the required sections (D-50). **No re-sweep, no LLM-substitution-risk check** (no new LLM substitutions in Phase 5). **Source:** D-48; D-44a; ROADMAP Phase 5 (no GATE-01..GATE-06 enumeration in Phase 5 success criteria, unlike Phases 2-4).

- **D-51b:** **GATE-04 (BENCHMARK_TABOO audit) carries forward for the methodology writeup.** The writeup is plain English about a project that contains benchmark-derived component names (HTML5 Client/Server, Kurento Media Server, FreeSWITCH, etc.); these can appear in the writeup BECAUSE the writeup is describing results on those benchmarks, NOT prompting an LLM. **`BENCHMARK_TABOO.md` applies to PROMPTS, not to result narratives.** The methodology writeup explicitly states which benchmarks it ran on, with their component vocabularies, because that is the contractual transparency a methodology section needs. Cross-check that no NEW prompt text is introduced in Phase 5 (D-44a guarantees this for `s_linker13.py`); no audit gate is triggered. **Source:** BENCHMARK_TABOO.md scope ("prompts"); MEMORY.md ("No dataset-specific examples in prompts — data leakage").

### Provenance & Reproducibility (D-52)
- **D-52:** **Every ablation table row is sourced to a JSON file under `results/ablation_results/`.** The render script (D-49a) accepts these JSON paths as input or references them by filename in code comments. A reviewer can regenerate the table by running the script. **Source:** REQUIREMENTS.md PROMO-03 (table is the reproducibility artifact); Phase 1 D-02 (single-source-of-truth principle).

- **D-52a:** **The `s_linker13` row of the ablation table has source JSON = `ablation_20260529_215932.json` (the 13f full-sweep JSON)**, with a footnote: "`s_linker13` is the canonical promotion of `s_linker13f` (Phase 5, 2026-05-29); no separate sweep was run per D-44a." **Source:** D-44a; D-48.

- **D-52b:** **The methodology writeup cites the SUMMARY files alongside the JSONs** for narrative anchors (e.g., the dual-hard-tier-variance numbers live in `04-01-SUMMARY.md` §"Dual Hard-Tier Results"; the 13d failure-mode analysis lives in `03-01-SUMMARY.md` §"Failure-Mode Analysis"). **Source:** `<canonical_refs>` below.

### Wave Structure (D-53)
- **D-53:** **Phase 5 may be executed as one or two plans.** Recommended split:
  - **Plan 05-01:** PROMO-01 (D-44 promotion mechanism) + PROMO-02 (D-45 KEEP-decision row in PROJECT.md). Small, mechanical, gated only on file presence + registration import-check.
  - **Plan 05-02:** PROMO-03 (D-46 ablation table) + PROMO-04 (D-50 methodology writeup). The render script + the writeup share a data source (D-47); a single plan that produces both files in lockstep is the natural unit.

  **Both plans can run in sequence (05-01 before 05-02) or in parallel** — neither depends on the other's artifact. Planner picks based on convenience. **Default recommendation: sequential 05-01 → 05-02** so the methodology writeup can reference the promoted `s_linker13.py` by name. **Source:** Phase 4 D-42 (sequential wave structure precedent).

- **D-53a:** **Single-plan collapse is acceptable if the planner judges the workload small enough.** All four deliverables (D-44 promotion, D-45 PROJECT.md row, D-46 ablation, D-50 writeup) total maybe a half-day of work. A single `05-01-PLAN.md` is fine. **Source:** Phase 3 D-30 (single-plan precedent).

### Validation Checks (D-54)
- **D-54:** **Phase 5 success-criteria validation (post-plan):**
  - **SC-1 (ROADMAP):** `src/llm_sad_sam/linkers/experimental/s_linker13.py` exists; class `SLinker13` defined; `_VARIANT_NAME = "s_linker13"`; registered in `run_ablation.py` `CANONICAL_VARIANTS` AND `VARIANT_SPECS`. **Check:** import-check (`python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS, build_linker; build_linker('s_linker13')"`) returns without error.
  - **SC-2 (ROADMAP):** `PROJECT.md` Key Decisions table contains a KEEP row for `_has_standalone_mention` referencing Spike 002. **Check:** `grep "_has_standalone_mention" .planning/PROJECT.md` returns the new row.
  - **SC-3 (ROADMAP):** `05-ABLATION.md` exists with one row per variant (12c, 13a, 13b, 13c, 13d, 13e, 13f, 13); columns per D-46a; numbers per D-47. `05-ABLATION.tex` exists and is `tabulate`-rendered LaTeX. **Check:** both files exist; markdown table renders cleanly; LaTeX compiles in a paper template (planner verifies by eyeballing — no `\usepackage` errors).
  - **SC-4 (ROADMAP):** `05-METHODOLOGY.md` exists; covers the eight sections per D-50; length ~2-3 pages (~1500-2500 words). **Check:** `wc -w` returns a number in band; section headings match D-50; the table from D-47 is either inlined or referenced.

  **Source:** ROADMAP §Phase 5 success criteria #1-4.

- **D-54a:** **Phase 5 closure** = all four SCs satisfied, all four deliverables committed, STATE.md updated (`stopped_at: Phase 5 complete (PROMO-01..04 satisfied)`, `last_activity: 2026-05-30 -- Phase 5 closed; s_linker13 promoted from 13f`), milestone progress 100%. **Source:** Phase 4 closure precedent in STATE.md.

### Folded Todos
None — `.planning/STATE.md` "Pending Todos" is empty.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (`gsd-phase-researcher`, `gsd-planner`) MUST read these before planning or implementing.**

### Project specs (winner identification, requirements, standing policy)
- `.planning/PROJECT.md` — Core Value (macro F1 ≥ 93% or reject); Key Decisions table (D-45 row appends here); constraints (Claude Sonnet only, no benchmark leakage).
- `.planning/REQUIREMENTS.md` — **PROMO-01, PROMO-02, PROMO-03, PROMO-04** (Phase 5 scope). §"v2 Requirements" EXT-01, EXT-02, EXT-03 (deferred follow-ups cited in the methodology writeup).
- `.planning/ROADMAP.md` §Phase 5 (lines 87-96) — goal, depends-on, success criteria #1-4.
- `.planning/STATE.md` §"Phase 4 Closure Notes" (winner candidate = 13f), §"Phase 3 Closure Note (empty)" (13d retired narrative), §"Standing Policy (Phases 3+)" (BBB 6pp tolerance), §"Deferred Items" (EXT-01, EXT-02, EXT-03).

### Phase 1 inheritance (winner-chain leg + first variant)
- `.planning/phases/01-baseline-and-infrastructure/01-CONTEXT.md` — D-02 (single-run baseline), D-03/D-04 (`_VARIANT_NAME` discipline), D-07 (runtime assertion).
- `.planning/phases/01-baseline-and-infrastructure/01-04-SUMMARY.md` — 12c canonical full-sweep numbers (MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405); BBB re-run history (cache-stream-timing first evidence).
- `.planning/phases/01-baseline-and-infrastructure/01-05-SUMMARY.md` — 13a full-sweep numbers (macro 0.9364); BBB cache-stream-timing perturbation analysis (D-50 §"Standing-Policy History" reference); first BBB tolerance loosening 2pp→4pp 2026-05-28 (D-50 narrative reference).

### Phase 2 inheritance (13b + 13c numbers; BBB tolerance evolution)
- `.planning/phases/02-ambiguity-cleanup/02-01-SUMMARY.md` — 13b full-sweep numbers (macro 0.9519); pure-removal does not exhibit BBB perturbation (D-50 narrative reference).
- `.planning/phases/02-ambiguity-cleanup/02-02-SUMMARY.md` — 13c full-sweep numbers (macro 0.9314, BBB 0.7818); byte-identical-classification BBB drift evidence (D-50 narrative reference); BBB tolerance second loosening 4pp→6pp 2026-05-29.

### Phase 3 inheritance (13d retirement narrative)
- `.planning/phases/03-mention-classifier-migration/03-CONTEXT.md` — D-19..D-30 (Phase 3 decisions).
- `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` — **13d failure-mode analysis** (dotted-path classification failure; TM F1 = 0.750; Δ −0.188); milestone-level finding. **Direct source for D-50 §"The 13d Failure Mode".**

### Phase 4 inheritance (13e + 13f numbers; dual-hard-tier protocol)
- `.planning/phases/04-alias-scope-and-coref-fold/04-CONTEXT.md` — D-31..D-42; §"Baseline Protocol (D-35)" (single-run sweeps); §"LLM-Substitution Inertness Risk (D-39)" (dual-hard-tier rationale).
- `.planning/phases/04-alias-scope-and-coref-fold/04-01-SUMMARY.md` — 13e full-sweep numbers (macro 0.9380); **dual hard-tier inter-variance metadata** (Run 1 BBB 0.826, Run 2 BBB 0.818, |Δ|=0.008). Direct source for D-50 §"The Dual-Hard-Tier Protocol for VAR-05".
- `.planning/phases/04-alias-scope-and-coref-fold/04-02-SUMMARY.md` — **13f full-sweep numbers** (macro 0.9509, +0.0104 vs 12c). Direct source for D-43 winner identification and the `s_linker13` promotion row in the ablation table.

### Spikes (validated rule classifications)
- `.planning/spikes/002-rules-audit/` — Spike 002, classifies `_has_standalone_mention` as RISKY (O(N×M) anchor collection). **Direct source for D-45 KEEP rationale and D-50 §"Retained Primitive".**
- `.planning/spikes/001-llm-trailing-words/README.md` — Spike 001 (13a's source). Reference for D-50 §"The Six Removals".
- `.planning/spikes/003-llm-mention-classifier/README.md` — Spike 003 (13d's source). Reference for D-50 §"The 13d Failure Mode".

### Codebase targets (files to read/edit)
- `src/llm_sad_sam/linkers/experimental/s_linker13f.py` — **promotion source.** Copy verbatim to `s_linker13.py`, apply D-44 edits.
- `src/llm_sad_sam/linkers/experimental/s_linker13.py` — **NEW FILE (to be created in Plan 05-01)**. Per D-44.
- `src/llm_sad_sam/linkers/experimental/s_linker13a.py`, `s_linker13b.py`, `s_linker13c.py`, `s_linker13d.py`, `s_linker13e.py` — **DO NOT DELETE** per D-43b. Remain in tree as ablation/rejection artifacts. Still registered in `run_ablation.py`.
- `run_ablation.py` — **append `s_linker13` after `s_linker13f`** in `CANONICAL_VARIANTS` and `VARIANT_SPECS`. Mark as canonical promotion per D-44 step 6. Same registration shape as Phase 1-4 plans (D-37a).
- `BENCHMARK_TABOO.md` — full project list. **Reference only** in Phase 5; no new prompts introduced (D-51b).

### Baseline JSON files (table data source — DO NOT re-run any of these)
- `results/ablation_results/ablation_20260528_173020.json` — **12c canonical baseline** (MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405). Phase 1 Plan 01-04. Row 1 of ablation table.
- `results/ablation_results/ablation_20260514_185017.json` — **12c BBB re-run** (BBB F1 = 0.844). Phase 1 Plan 04 cache-cleared re-run; canonical for the 12c BBB cell.
- `results/ablation_results/ablation_20260528_173020.json` — 13a full sweep (macro 0.9364). Phase 1 Plan 05. Row 2 of ablation table.
- `results/ablation_results/ablation_20260528_190916.json` — 13b full sweep (macro 0.9519). Phase 2 Plan 02-01. Row 3.
- `results/ablation_results/ablation_20260528_201851.json` — 13c full sweep (macro 0.9314, BBB 0.7818). Phase 2 Plan 02-02. Row 4.
- `results/ablation_results/ablation_20260529_110532.json` — 13d hard-tier (RETIRED). Phase 3 Plan 03-01. Row 5 (with `RETIRED` status).
- `results/ablation_results/ablation_20260529_181005.json` + `ablation_20260529_193240.json` — 13e dual-hard-tier (Run 1 + Run 2). Phase 4 Plan 04-01. Reference for D-50 §"The Dual-Hard-Tier Protocol".
- `results/ablation_results/ablation_20260529_201324.json` — 13e full sweep (macro 0.9380). Phase 4 Plan 04-01. Row 6.
- `results/ablation_results/ablation_20260529_204652.json` — 13f hard-tier. Phase 4 Plan 04-02. Reference only.
- `results/ablation_results/ablation_20260529_215932.json` — **13f full sweep (macro 0.9509, +0.0104 vs 12c)**. Phase 4 Plan 04-02. **Row 7 of ablation table AND row 8 (`s_linker13` PROMOTED), per D-44a + D-52a.**

### Research / methodology context
- `.planning/research/ARCHITECTURE.md` — pipeline structure; locates the helpers removed in 13a-13f.
- `.planning/research/PITFALLS.md` — Claude run-to-run variance documentation (D-50 §"Standing-Policy History" reference).

### Memory / prior art
- MEMORY.md — standalone-file preference; LLM run-to-run variance documenting the BBB cache-stream-timing pattern (Phase 1 D-13a + Phase 2 D-13a evidence); Phase 3 lessons backing the 13d narrative; "Always use Claude Sonnet … never opus" preference (sanity-check that `s_linker13` does not silently switch backends).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `s_linker13f.py` (`src/llm_sad_sam/linkers/experimental/`) — **promotion source.** Standalone file, full pipeline implementation, fully audited (Phase 4 close). Copy verbatim and apply D-44 edits to produce `s_linker13.py`.
- `_VARIANT_NAME` pattern + D-07 runtime assertion (Phase 1 INFRA-05) — carries forward via the `cp`; only the constant value changes to `"s_linker13"`.
- `run_ablation.py` `CANONICAL_VARIANTS` / `VARIANT_SPECS` append-only registration — exercised in every Phase 1-4 plan; same shape for `s_linker13`.
- `tabulate` dep (Phase 1 D-06, exercised in Phase 2-4 ablation row generation) — Phase 5's primary table renderer for both markdown and LaTeX (D-46).
- Every canonical JSON in `results/ablation_results/` — table data source per D-47 / D-49a. No re-runs.

### Established Patterns
- Standalone variant files (12a-12e + 13a-13f) — duplicated code is the project's reproducibility artifact. `s_linker13` continues the pattern.
- Append-only registration in `run_ablation.py`.
- Structured module docstring with `REMOVED_FROM:` and `RULES_REMOVED:` (Phases 1-4) — extended with `KEEP:` in Phase 5 (D-44 step 4).
- Per-variant pickle cache namespace under `results/phase_cache/<_VARIANT_NAME>/<dataset>/`; D-07 fail-fast assertion at construct time.
- Markdown-primary tables via `tabulate(..., tablefmt="github")` or `"pipe"` (Phase 2 D-17 / Phase 3 D-29 / Phase 4 D-41 precedent); Phase 5 extends to LaTeX via `tablefmt="latex"` / `"latex_booktabs"`.
- Inline prompt constants — **no new prompts introduced in Phase 5** (D-44a; `s_linker13.py` is byte-equivalent to `s_linker13f.py`).

### Integration Points
- `run_ablation.py` `CANONICAL_VARIANTS` list — append `"s_linker13"` after `"s_linker13f"`.
- `run_ablation.py` `VARIANT_SPECS` dict — add a `s_linker13` entry, copying the 13f spec dict and updating any name fields. Mark as canonical promotion (planner discretion on the exact flag name — D-44 step 6 / D-44d).
- `PROJECT.md` Key Decisions table (lines 60-69) — append the KEEP-decision row from D-45. **Do not edit existing rows.**
- `.planning/STATE.md` — Phase 5 closure update at the end (D-54a).

### Slip-Channel / Failure Modes to Pre-Watch
- **Banner-string drift.** Phase 1 Plan 05 §Deviations records a precedent where the banner string was not updated, requiring a follow-up commit. The planner MUST grep for `"s_linker13f"` after the file copy and verify the only remaining occurrences are in code comments / docstring provenance (D-44c), NOT in `_VARIANT_NAME` or the banner.
- **Class-name shadowing.** If the planner renames the class `SLinker13f → SLinker13` but misses a `super().__init__` call or a self-reference, the import-check (D-54 SC-1) will surface it. Recommended planner action: `grep -n "SLinker13f\|s_linker13f" src/llm_sad_sam/linkers/experimental/s_linker13.py` after the rename; only docstring / comment references should remain.
- **Ablation row transcription error.** D-47 lists the numbers from SUMMARY-file transcription. The render script should compute the macro from the per-dataset cells (D-49b) so a typo in one cell propagates to the macro and is visible at render time.
- **JSON-vs-SUMMARY discrepancy.** Per D-47a, the JSON wins. If the planner finds a number in a SUMMARY file that does not match the cited JSON, the planner re-reads the JSON, updates D-47 (annotate the change in the plan SUMMARY), and proceeds. **No re-sweeps.**
- **LaTeX `tabulate` quirks.** `tabulate`'s LaTeX output sometimes emits cells with special characters (`_`, `&`, `%`) that need escaping. The planner should manually inspect the `.tex` output and add an escape pass if needed (`tabulate` has `floatfmt` + `numalign` controls but no auto-escape for inline text columns). **Source:** general `tabulate` knowledge; not in MEMORY.md.
- **`s_linker13` cache namespace.** First import / first use will create `results/phase_cache/s_linker13/`. If a leftover `results/phase_cache/s_linker13/` exists from any prior experimentation, the D-07 assertion will fail (namespace must match `_VARIANT_NAME`). Recommended planner action: `ls results/phase_cache/s_linker13/ 2>/dev/null || true` before plan close; if present and not from this plan, delete.

</code_context>

<specifics>
## Specific Ideas

- **`s_linker13f` is the artifact-of-record for the entire milestone.** Its byte-equivalent twin `s_linker13.py` is the canonical name, but the actual binary numbers come from 13f's full sweep. The methodology writeup makes this explicit: "we promote `s_linker13f` as `s_linker13.py`; the numbers reported for `s_linker13` are the 13f full-sweep numbers."
- **The "best macro" answer is 13b (0.9519), not 13f (0.9509).** This is the kind of detail the methodology writeup must address honestly (D-43 footnote 1). 13b is a one-rule-removed midpoint with high raw macro; 13f is the six-rules-removed endpoint. The thesis of the project is "rule-reduction without F1 regression," not "highest raw F1." 13f is the answer because it has the most rules removed AND still passes the dual floor. The writeup makes this explicit.
- **The BBB tolerance loosening (2pp → 4pp → 6pp) is the methodology writeup's most defensive section.** A reviewer will ask why the floor was loosened. The answer (D-50 §"Standing-Policy History") is: cache-stream-timing perturbation on a specific failure mode (multi-word component partials on BBB) that is independent of code-correctness, with empirical evidence (13b 2pp-band pure-removal vs 13c 6pp-band byte-identical-classification). This is a documented Claude run-to-run variance pattern, NOT a methodology weakness.
- **The 13d failure mode is the milestone's most publishable caveat.** "LLM substitutions of project-specific structural rules can fail catastrophically — 13d -19pp TM" (Plan 03-01 SUMMARY phrasing). The writeup states this plainly: the no-hand-crafted-rules thesis holds with this single documented caveat (classification of language-construct references).
- **The dual-hard-tier protocol for VAR-05 is the methodology's most defensive design choice.** The ROADMAP mandated it ("widest blast radius — run twice on hard tier before full sweep"); the protocol caught nothing (both runs passed, |Δ| well inside the variance band). The writeup frames this as: the protocol was the empirical mitigation; its uneventful pass is empirical evidence that VAR-05 is stable.
- **Six rule removals across the chain, not seven.** The original target was seven (VAR-01 through VAR-06 + an implicit "tail" via `_has_standalone_mention`); VAR-04 (13d) was retired empirically; `_has_standalone_mention` is KEPT by deliberate decision. The writeup reports "six structural rule removals," not "all six rules removed" — the distinction matters for honest accounting.
- **PROMO-03 + PROMO-04 share a data source.** The ablation table (D-47) is the numerical artifact; the methodology writeup is the narrative artifact. Both read from the same `dict[str, list[float|str]]` literal in the render script (D-49a). A planner who modifies one without the other should fail D-54 SC-3 / SC-4 immediately.
- **No `s_linker14` is contemplated in this milestone.** Phase 5 closes the milestone; v2 (EXT-01/02/03) is the next-milestone scope. The methodology writeup forward-points to v2 but does NOT contain v2 work.
- **Phase 5 is the smallest phase of the milestone.** Four mechanical deliverables, no new linker variants, no new sweeps, no new LLM substitutions. The planner workload is ~half a day of careful prose + a small Python helper for table rendering.

</specifics>

<deferred>
## Deferred Ideas

- **EXT-01: Spike on replacing `_has_standalone_mention` with LLM primitive** — v2, deferred per `.planning/STATE.md` "Deferred Items". The methodology writeup (D-50 §"Retained Primitive") forward-points to this without performing it.
- **EXT-02: Drop `_has_standalone_mention` dotted-path guard; let LLM mention classifier handle dotted-path detection** — v2, deferred per same row. Forward-pointed from the methodology writeup; not in Phase 5 scope.
- **EXT-03: GPT-5.2 cross-model re-evaluation of `s_linker13`** — v2, deferred per same row. PROJECT.md constraint is "Claude Sonnet only"; the writeup notes the GPT-vs-Claude gap as a documented limitation (MEMORY.md §"GPT-5.2 Compatibility") and forward-points to EXT-03 for the formal re-evaluation.
- **Per-phase FP-by-phase breakdown for 12c, 13a, 13b, and `s_linker13` rows** — D-46b makes this best-effort. A complete FP-by-phase column would require re-running the linker and parsing per-source per-FP records, which is sweep-work and out of scope (D-48). The writeup notes the partial coverage.
- **One-shot sanity sweep of `s_linker13.py` to verify byte-equivalence to 13f** — could be done as a milestone-closure audit (`gsd-audit-milestone`) but is NOT a Phase 5 deliverable (D-48b).
- **Replacing 13a-13f in `run_ablation.py` with a single canonical `s_linker13` entry** — explicitly rejected (D-43b). The intermediate / rejection / chain variants remain registered so the ablation table is regeneratable from canonical JSONs at any time.
- **LaTeX table styling beyond `tabulate`'s defaults** (booktabs `\midrule` placement, custom column alignment, multi-row spans for `Δ vs 12c` / `Δ vs parent` headers) — planner's call (D-49c); D-46 only requires "paper-ready", not "perfectly typeset". A `\hline`-and-default columns rendering is acceptable; the planner upgrades to `booktabs` if the paper template uses it.
- **A separate "results discussion" section in the methodology writeup** beyond the per-variant narrative (D-50 §"The Six Removals") — out of scope at 2-3 page target. The "Result" section (D-50 §8) is the single-paragraph summary.
- **A separate "limitations" or "threats to validity" section in the writeup** — folded into D-50 §"Standing-Policy History" (the BBB tolerance evolution IS the primary methodology limitation discussion) and §"The 13d Failure Mode" (the no-hand-crafted-rules-thesis caveat). Explicit "Limitations" heading is planner discretion (D-50e).
- **Combining the ablation table render and the writeup into a single Jupyter notebook** — planner discretion. Likely simpler to keep them as separate `.md` / `.tex` / `.py` files for git-tracking simplicity.
- **Updating MEMORY.md with the Phase 5 outcome** — out of scope for the discuss-phase / plan-phase / execute-phase loop. MEMORY.md updates happen at milestone-closure (`gsd-complete-milestone`), not Phase 5 close.

### Reviewed Todos (not folded)
None — `.planning/STATE.md` "Pending Todos" is empty.

</deferred>

---

*Phase: 05-promote-and-ablation-artifact*
*Context gathered: 2026-05-29 (auto mode — Claude-selected recommended defaults from Phases 1-4 precedent + ROADMAP Phase 5 + orchestrator instructions; D-43..D-54)*
