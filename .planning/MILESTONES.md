# Milestones — llm-sad-sam-v45

Historical record of shipped milestones. See `milestones/v[X.Y]-ROADMAP.md`, `milestones/v[X.Y]-REQUIREMENTS.md`, and `milestones/v[X.Y]-MILESTONE-AUDIT.md` for full per-milestone archives.

---

## v1.0 — Rule-to-LLM Ablation (`s_linker12c` → `s_linker13`)

**Shipped:** 2026-05-29 (re-audit `passed`: 2026-05-30)
**Audit verdict:** `passed` (upgraded from `tech_debt` on 2026-05-30 after BBB root-cause investigation)
**Final artifact:** `src/llm_sad_sam/linkers/experimental/s_linker13.py`

### Delivered

Defensible empirical claim that 6 of 7 targeted structural rules in `s_linker12c` can be replaced by LLM primitives without regressing macro F1 below 0.93. Final macro F1 = **0.9509** (+1.04 pp vs the 0.9405 `s_linker12c` baseline).

Per-dataset (13f sweep, used as `s_linker13` row): MediaStore 0.984 / TeaStore 1.000 / TeaMMates 0.947 / BigBlueButton 0.821 / JabRef 1.000.

### Stats

| Item | Count |
|------|-------|
| Phases | 5 (5 complete) |
| Plans | 13 (13 complete; Phase 3 closed empty with negative result) |
| Variant files | 7 (`s_linker13a`-`s_linker13f` + canonical `s_linker13`) |
| Helpers retired | 6 of 7 targeted (`_split_component_name`, `_is_structurally_unambiguous`, `_is_ambiguous_name_component`, `_is_strong_alias`, `_get_strong_alias_mappings`, `_has_strong_alias_mention`) |
| Helpers retired-as-rejection | 1 (`_classify_mention` — VAR-04 negative result) |
| Helpers KEPT on cost grounds | 1 (`_has_standalone_mention` — RISKY per Spike 002 O(N×M)) |
| Timeline | 2026-04-21 (project init) → 2026-05-29 (milestone close) → 2026-05-30 (re-audit `passed`) |

### Key Accomplishments

1. **Phase 1 (Baseline + Infrastructure):** 12c baseline JSON captured; per-variant `_checkpoint_dir` namespacing landed; `diskcache>=5.6.1` + `tabulate>=0.9.0` added; `s_linker13a` (Spike 001 LLM trailing-word) ships under user-loosened BBB 4pp tolerance — macro F1 0.9364.
2. **Phase 2 (Ambiguity Cleanup):** `s_linker13b` removes `_is_structurally_unambiguous` (+0.0114 macro vs 12c); `s_linker13c` inlines and removes `_is_ambiguous_name_component` (parity probe 5/5 byte-identical), macro 0.9314 under user-loosened BBB 6pp.
3. **Phase 3 (Mention Classifier Migration — closed empty):** `s_linker13d` collapses TeaMMates F1 from 0.938 to 0.750 on 33 entity-source FPs from dotted-path Java-package references; VAR-04 retired-as-rejection per user direction; `s_linker13d.py` left in tree as rejection artifact. **This is the milestone's primary publishable finding.**
4. **Phase 4 (Alias Scope + Coref Fold):** `s_linker13e` introduces `scope: global|local` LLM field, retires `_is_strong_alias` + `_get_strong_alias_mappings`; dual-hard-tier protocol clean (|Δ|=0.008). `s_linker13f` folds `_has_strong_alias_mention` into coref prompt — macro **0.9509, best in chain**.
5. **Phase 5 (Promote + Ablation Artifact):** `s_linker13.py` promoted (byte-equivalent to `s_linker13f.py` modulo class/banner per D-44a); `_has_standalone_mention` KEEP-decision logged in `PROJECT.md`; `ABLATION-TABLE.md` + `.tex` generated via `tabulate` (8 rows); `METHODOLOGY.md` shipped (7 sections covering thesis, chain, policy evolution, 13d negative result, dual-hard-tier protocol, deferred items).
6. **Post-milestone root-cause analysis (2026-05-30):** `BBB-ROOT-CAUSE.md` and `BBB-DEEP-SEMANTIC-ANALYSIS.md` produced; identified alias-count → recovery-handle-count correlation as mechanism for intermediate-variant BBB drift; deliverable confirmed not to consume tolerance (BBB band 0.821-0.842 overlaps 12c band 0.818-0.844); audit re-classified `tech_debt` → `passed`.

### Deferred to v2 (4 items)

- **EXT-01** — Spike on replacing `_has_standalone_mention` with LLM primitive (relaxed budget)
- **EXT-02** — Drop dotted-path guard in `_has_standalone_mention` (narrower follow-up to EXT-01)
- **EXT-03** — GPT-5.2 cross-model re-evaluation of `s_linker13`
- **EXT-04** — Emit-biased boundary prompting on alias-discovery to shrink BBB borderline-4 variance band from ~3pp to ~1pp (NEW; motivated by BBB-ROOT-CAUSE.md / BBB-DEEP-SEMANTIC-ANALYSIS.md)

### Standing-Policy Decisions

- BBB per-dataset tolerance loosened from 2 pp → 4 pp → 6 pp during the chain (used by intermediate variants 13a/13c/13e; NOT consumed by deliverable `s_linker13`)
- Macro F1 floor stayed at 0.93 throughout
- Other-dataset tolerance stayed at 2 pp throughout
- Dual-hard-tier protocol applied to widest-blast-radius variant (VAR-05 / `s_linker13e`)

### Known Limitations (Documented Empirical Findings)

- **VAR-04 retirement** — `_classify_mention` cannot be LLM-replaced for dotted-path Java-package conventions; documented as publishable negative result. The no-hand-crafted-rules thesis holds with this caveat: classification of project-specific language-construct references is regex territory.
- **15-sentence BBB structural dead zone** — HTML5 Client/Server and WebRTC-SFU partial mentions (S6, S9-13, S19, S39, S47, S65, S73). Identical across 12c and all variants — neither regex nor LLM globally aliases "the client"/"the server" (over-fire risk). Remediation requires per-sentence partial-injection (EXT-01).
- **BBB borderline-4 variance band** (~3 pp) — S38 BBB web, S73/S76/S79 HTML5 Client. Recovery correlates monotonically with alias-discovery emit count. EXT-04 addresses.

### Archive

- `milestones/v1.0-ROADMAP.md`
- `milestones/v1.0-REQUIREMENTS.md`
- `milestones/v1.0-MILESTONE-AUDIT.md`
- Phase directories retained under `.planning/phases/` (user explicitly requested no `gsd-cleanup` at this time)
