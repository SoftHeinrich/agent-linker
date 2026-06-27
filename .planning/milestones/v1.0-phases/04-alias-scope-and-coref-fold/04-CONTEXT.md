# Phase 4: Alias Scope and Coref Fold - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning
**Mode:** `gsd-discuss-phase --auto` (recommended defaults selected by Claude; no human Q&A — every decision below is locked with a cited source)

<domain>
## Phase Boundary

Retire the three structural alias helpers from `s_linker13c` by shipping two sequential standalone variants:

1. **VAR-05 / `s_linker13e.py`** — extend the alias-discovery prompt (`_learn_document_knowledge_enriched`, `s_linker13c.py:319`) so each emitted alias carries a `scope: "global" | "local"` field. Define an `AliasEntry` dataclass (or `TypedDict`) inline with `component: str`, `scope: str` (and the existing alias-term key). Replace the two consumer-side calls to `_is_strong_alias` / `_get_strong_alias_mappings` with reads off `scope == "global"`. Delete both methods. **Widest blast radius in the entire 13-series — VAR-05 is the only rule replacement where the LLM signal is *consumed everywhere downstream* (entity-extraction prompts L739-741, coref antecedent verification L1015, evidence-bundle alias-match formatting L569-573).**
2. **VAR-06 / `s_linker13f.py`** — fold the `_has_strong_alias_mention` antecedent-verification check (`s_linker13c.py:1068-1080`) into the coref prompt evidence schema. The coref LLM emits an `antecedent_via_alias: bool` (or `antecedent_alias_used: str | null`) field; the consumer trusts that field instead of re-running the regex against `ant_sent.text`. Delete `_has_strong_alias_mention`. **Smaller scope:** the check fires at exactly one site (`_coref_cases_in_context`, L1015), and the coref LLM call already has full sentence context — the fold is structurally clean.

Both variants must pass **GATE-01 dual floor** on the full 5-project sweep:
- **macro F1 ≥ 0.93**
- **no dataset more than 2pp below the 12c Plan 04 baseline (MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405) — except BBB, which gets a 6pp tolerance** carried over from the 2026-05-29 standing policy set during Phase 2 closure (BBB floor = 0.844 - 0.06 = **0.784**).

**Parent baselines for ΔF1 (D-12 / D-24b carry-over):**
- **13e parent = 13c** (Phase 2 final winner; macro 0.9314; canonical full-sweep JSON `results/ablation_results/ablation_20260528_201851.json`). 13d is **NOT** a parent — VAR-04 was retired empirically (Phase 3 closed empty per STATE.md 2026-05-29) and 13d does not enter the promotion chain.
- **13f parent = the 13e that ships** (if 13e passes GATE-01 and is admitted). **Fallback:** if 13e is rejected, 13f's parent falls back to 13c so the chain does not stall on VAR-06.

**In scope:** Variant files `s_linker13e.py` + `s_linker13f.py`, inline `AliasEntry` dataclass + prompt-schema extension on `_learn_document_knowledge_enriched` (VAR-05), coref prompt-schema extension on `_coref_cases_in_context` (VAR-06), registration in `run_ablation.py`, BENCHMARK_TABOO audits on the new `ALIAS_SCOPE_SCHEMA` / coref-evidence prompt-schema text, hard-tier (TM + BBB) gate then full 5-project sweep, ablation log rows for both variants, **dual-run hard-tier protocol for 13e (run twice, both passes auto-approve before full sweep)**.

**Out of scope (this phase):**
- `_has_standalone_mention` keep/replace decision (Phase 5 / EXT-01; `_has_standalone_mention` is still consumed at `s_linker13c.py:1014` and remains untouched in both 13e and 13f).
- `_classify_components` ambiguity-set changes (Phase 2 territory; 13e + 13f inherit the 13c removal and prompt unchanged).
- `_classify_mention` (VAR-04, RETIRED in Phase 3 — closed empty 2026-05-29; 13e/13f keep 13c's regex `_classify_mention` since 13c never removed it).
- Promotion artifact (PROMO-* → Phase 5).
- GPT-5.2 cross-model gating (EXT-03; Claude Sonnet only per PROJECT.md).
- Combining VAR-05 + VAR-06 into a single variant — explicitly rejected by the ROADMAP Phase 4 two-criterion structure and by the "one rule removal per variant" Key Decision (D-08).
- Changes to retained upstream components (`ilinker*`, `prompts_v2`, `data_types_v2`) unless required by a rule removal — neither VAR-05 nor VAR-06 requires it.

</domain>

<decisions>
## Implementation Decisions

### Variant File Layout (D-31)
- **D-31:** `s_linker13e.py` is a **standalone file**, full copy of `s_linker13c.py`, edited in place. No inheritance, no shared helpers module. `s_linker13f.py` is then a full copy of the 13e that ships (or of 13c if 13e is rejected per D-35a). **Source:** Phase 1 D-03/D-04; Phase 2 D-08; Phase 3 D-19; MEMORY.md ("User prefers standalone linker files (duplicate code intentionally, not inheritance chains)").
- **D-31a:** Each variant's module docstring includes `REMOVED_FROM: <parent>` and `RULES_REMOVED: [...]`. For 13e: `REMOVED_FROM: s_linker13c`, `RULES_REMOVED: ["_is_strong_alias", "_get_strong_alias_mappings"]`. For 13f: `REMOVED_FROM: s_linker13e` (or `s_linker13c` if 13e rejected), `RULES_REMOVED: ["_has_strong_alias_mention"]`. **Source:** GATE-03; Phase 2 D-08; Phase 3 D-19a.
- **D-31b:** The `__init__` print-banner string in each variant identifies the variant (Phase 1 Plan 05 deviation #3 lesson; Phase 2 D-08 §Claude's Discretion; Phase 3 D-19b carry-over). **Source:** Phase 1 Plan 05 SUMMARY §Deviations; Phase 2 §Claude's Discretion.
- **D-31c:** **`s_linker13d.py` is NOT in the inheritance chain.** Phase 3 closed empty (STATE.md 2026-05-29); VAR-04 retired. The structural cumulative-removal chain skips 13d entirely: 12c → 13a → 13b → 13c → **13e** → **13f**. `s_linker13d.py` remains in the tree as the rejection artifact (do not delete — per Plan 03-01 SUMMARY §"User Resolution"). **Source:** STATE.md §"Phase 3 Closure Note"; ROADMAP §Phase 3.

### Prompt-Schema Extension — VAR-05 (D-32)
- **D-32:** Extend the existing `_learn_document_knowledge_enriched` LLM prompt (`s_linker13c.py:319-355`, the first prompt that emits `{"abbreviations": {...}, "synonyms": {...}}`) so each value entry carries a `scope: "global" | "local"` field. **The expanded JSON shape is the author's choice during planning** but must remain back-compatible with the existing consumer loop at L348-355 (which currently iterates `abbreviations` + `synonyms` as flat `{short: full}` dicts). **Recommended planner shape** (one of two acceptable):
  - **Option A (record-per-alias, flat):** `{"abbreviations": [{"term": "X", "component": "Y", "scope": "global"}], "synonyms": [{"term": "X", "component": "Y", "scope": "global"}]}` — clean schema; requires consumer loop rewrite (one line each).
  - **Option B (paired dicts, parallel):** `{"abbreviations": {"X": "Y"}, "synonyms": {"X": "Y"}, "scopes": {"X": "global"}}` — minimal disruption to existing consumer; `scope` looked up at consumer site.
  - **Default recommendation:** Option A — closer to a record / dataclass; pairs naturally with the inline `AliasEntry` dataclass per D-32a.
- **D-32a:** Define `AliasEntry` inline in `s_linker13e.py` as a `@dataclass` (or `NamedTuple`) with fields `component: str`, `scope: str` (the alias-term itself is the dict key in `ModelKnowledge.aliases`). The current `doc_knowledge.aliases: dict[str, str]` (term → component name) migrates to `dict[str, AliasEntry]`. Touchpoints: `s_linker13c.py:389` (alias assignment), `s_linker13c.py:533-541` (alias listing in evidence prompts), `s_linker13c.py:569-573` (alias-match formatter in `_format_evidence` — must read `.component` off the entry), `s_linker13c.py:1077-1079` (`_has_strong_alias_mention` — being removed in 13f, but 13e still has it for the one-rule-per-variant rule; in 13e it reads `entry.scope == "global"` instead of calling `_is_strong_alias`). **Source:** ROADMAP Phase 4 success criterion #1 ("`AliasEntry` defined inline with `component` and `scope` fields").
- **D-32b:** Prompt-schema text stays **inline** in `s_linker13e.py` (no `prompts_v2.py` edits). **Source:** Phase 2 D-09; Phase 3 D-20b.
- **D-32c:** The five scope-classification rules (taxonomy of when the LLM should mark an alias `global` vs `local`) are stated **as positive natural-language rules in the prompt**, not as code rules. The taxonomy MUST be expressed in safe-SE-textbook placeholders only (D-37a); the planner authors the exact wording. The empirical taxonomy that `_is_strong_alias` encodes — multi-word → global, hyphenated → global, CamelCase → global, all-caps acronyms → global, lowercase single-word → **local** — is captured in the prompt as a reasoning guide, not a regex. **Source:** ROADMAP Phase 4 success criterion #1 ("`AliasEntry` defined inline with `component` and `scope` fields; no call to `_is_strong_alias` or `_get_strong_alias_mappings`"); MEMORY.md ("LLM substitutions of project-specific structural rules can fail catastrophically — 13d -19pp TM").
- **D-32d:** **`_get_strong_alias_mappings` is replaced by a list-comprehension at its single callsite** (`s_linker13c.py:741`): `mappings = [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items() if entry.scope == "global"]`. The method is then deleted from 13e. **Source:** ROADMAP Phase 4 success criterion #1 (no `_get_strong_alias_mappings` call); D-08 standalone-file principle (consumer-side substitution, no helper module).

### Prompt-Schema Extension — VAR-06 (D-33)
- **D-33:** Extend the existing `_coref_cases_in_context` coref prompt (`s_linker13c.py:974-989`, the LLM call that emits `{"resolutions": [{"case": N, "sentence": M, "pronoun": "it", "component": "X", "antecedent_sentence": K, "antecedent_text": "quote"}]}`) so each resolution entry carries a new boolean field: **`antecedent_via_alias: bool`** (the cleanest fold). Default-on for clarity at consumer: `True` iff the antecedent sentence references the component **via a known global alias** (not by the component's canonical name). **Source:** ROADMAP Phase 4 success criterion #3 ("coref prompt output schema includes `antecedent_via_alias` field"); MEMORY.md §"V39 Coref Unification & Judge Analysis" (Variant E cases-in-context already provides the antecedent quote, so the LLM already has sentence-text context to make the alias call).
- **D-33a:** **Consumer migration at `s_linker13c.py:1014-1016`:**
  ```python
  if not (self._has_standalone_mention(comp, ant_sent.text) or
          self._has_strong_alias_mention(comp, ant_sent.text)):
      continue
  ```
  becomes (in 13f):
  ```python
  if not (self._has_standalone_mention(comp, ant_sent.text) or
          res.get("antecedent_via_alias", False)):
      continue
  ```
  Then `_has_strong_alias_mention` (`s_linker13c.py:1068-1080`) is deleted. `_has_standalone_mention` is NOT touched (out of scope; deferred to EXT-01). **Source:** ROADMAP Phase 4 success criterion #3 ("`s_linker13f` contains no call to `_has_strong_alias_mention`").
- **D-33b:** The fold is **inert with respect to the rest of the coref logic.** The coref LLM call already has full sentence context (±5 bidirectional, per the cases-in-context prompt L967), so adding `antecedent_via_alias` to the output schema does not change the input the LLM sees — it only adds an output bit. Compare to VAR-04 / 13d, where the LLM had to learn a new classification taxonomy (dotted-path detection) from project-specific patterns and failed catastrophically (-19pp TM). VAR-06 asks the LLM to make a much simpler call: "is the antecedent quote a canonical-name reference or an alias reference?" — a question the LLM is already implicitly answering when it picks an antecedent sentence. **Source:** Plan 03-01 SUMMARY §"Failure-Mode Analysis" (dotted-path failure mode); MEMORY.md §"Key Architectural Lessons" ("LLM CAN replace P8c boundary filters ... Convention-aware filter (3-step reasoning guide) catches 11 FPs vs 5 regex, 0 TPs killed").
- **D-33c:** **`coref_alias_used: str | null` is an acceptable alternative output shape** (string is the alias text, null is "canonical name"; truthiness check at consumer is equivalent). **Default: boolean `antecedent_via_alias`** for minimal LLM output complexity. Planner may switch to the string form if a downstream FN-debugging pattern wants the alias text on hand. **Source:** ROADMAP Phase 4 success criterion #3 (schema field name `antecedent_via_alias` is canonical).

### Exact-String / Backward-Compat Contract (D-34)
- **D-34:** **`doc_knowledge.aliases` type signature change (VAR-05) is a structural break.** The migration from `dict[str, str]` to `dict[str, AliasEntry]` touches every read site. The 13e plan MUST identify each read site (planner does a `grep` for `doc_knowledge.aliases` and `\.aliases\.items\(\)` / `\.aliases\.get\(\)`) and migrate each one. Specifically (in 13c):
  - L174 / L179: `len(self.doc_knowledge.aliases)` — unchanged (still len of dict)
  - L389: `knowledge.aliases[term] = comp` → `knowledge.aliases[term] = AliasEntry(component=comp, scope=...)` (read scope off the LLM record)
  - L533-535: `aliases.items()` iteration in `_format_evidence` — read `target.component` and optionally `target.scope`
  - L569-571: `aliases.items()` in `_match_alias` text formatter — same
  - L741 (replaces `_get_strong_alias_mappings`): list-comp filter on `entry.scope == "global"`
  - L1077: `_has_strong_alias_mention` filter (deleted in 13f; in 13e it filters on `entry.scope == "global"`)
- **D-34a:** **Side-by-side LLM-vs-structural classification log is a Phase 4 acceptance artifact.** Per ROADMAP Phase 4 success criterion #1 ("side-by-side log of LLM scope vs structural classification on all aliases is recorded and shows no unexpected divergence"), the 13e plan MUST emit during the dual hard-tier run a comparison table: for every alias the LLM emits, log `(alias_term, llm_scope, structural_scope_via_old_is_strong_alias_method, match: bool)`. This is **diagnostic** — it does not gate GATE-01 — but a divergence rate above ~20% is a strong signal the LLM is mis-classifying and the planner should surface it. **Source:** ROADMAP Phase 4 success criterion #1; MEMORY.md §"V32 Final Audit" (taboo-leakage debug pattern: side-by-side comparison surfaced 13d's failure mode).
- **D-34b:** **`_is_strong_alias` is reused IN 13e ONLY for the diagnostic side-by-side log** (D-34a). It is deleted from 13e once the log is captured. Planner has two clean options: (i) keep `_is_strong_alias` in 13e as a `@deprecated` static helper used only by the diagnostic logger, deleted before the full sweep; (ii) write a one-line in-test reimplementation of the structural rule in the plan's parity test and delete `_is_strong_alias` immediately. **Default: option (ii)** — cleaner `rules_removed` claim. **Source:** GATE-03 (structured docstring); Phase 3 D-19c (deletion vs preservation analogue).

### Baseline Protocol (D-35)
- **D-35:** **Single run on full 5-project sweep for 13e (after the dual hard-tier passes — see D-35a) and single run for 13f.** No N-run median, no best-of-N. **Do not re-run 12c; do not re-run 13c.** Canonical 12c baseline JSON: `results/ablation_results/ablation_20260528_173020.json` (Plan 01-04). Canonical 13c baseline JSON: `results/ablation_results/ablation_20260528_201851.json` (Plan 02-02 canonical, MS 0.967 / TS 0.953 / TM 0.929 / BBB 0.7818 / JAB 0.973 / macro 0.9314). **Source:** Phase 1 D-02; Phase 2 D-10; Phase 3 D-23; STATE.md "D-02 single-run baseline applies".
- **D-35a:** **VAR-05 dual hard-tier protocol (ROADMAP-mandated).** Per ROADMAP Phase 4 success criterion #2 ("13e is run twice on teammates + BBB before full sweep; both runs agree within Claude's normal run-to-run variance"), the 13e plan runs the hard-tier sweep **twice** (cache cleared between runs to ensure independence — see D-37c), with the **standing GATE-05 auto-approve thresholds applied INDEPENDENTLY to each run**. Promotion to full sweep requires:
  - Run 1: TM delta ≥ -0.01 AND BBB delta ≥ -0.06 vs 12c (D-24a / D-36a thresholds), **AND**
  - Run 2: TM delta ≥ -0.01 AND BBB delta ≥ -0.06 vs 12c, **AND**
  - Run 1 ↔ Run 2 inter-run variance: |Δ| ≤ 0.02 on TM macro AND |Δ| ≤ 0.04 on BBB macro (Claude run-to-run variance band documented in MEMORY.md §"LLM Variance").
  Any of the three conditions failing → no full sweep, rework variant. **Source:** ROADMAP Phase 4 success criterion #2 ("widest blast radius — run twice on hard tier before full sweep"); MEMORY.md §"LLM Variance (Critical Finding)" ("Same model gives DIFFERENT behavior across days").
- **D-35b:** **VAR-06 single-hard-tier protocol.** 13f runs the standard single hard-tier pass (D-24a thresholds, same as 13b/13c/13d) → if auto-approve, run the single full sweep. The dual-run requirement is **specifically for VAR-05's widest-blast-radius schema change**; VAR-06's narrower-scope schema fold inherits the simpler protocol. **Source:** ROADMAP Phase 4 §Phase Details — VAR-06 ("smaller scope, single hard-tier is fine" — orchestrator instruction explicitly distinguishes); Phase 3 D-23 (precedent for single-variant hard-tier).

### Standing Policy Carry-Over (D-36)
- **D-36:** **GATE-01 BBB tolerance: 6pp. Other 4 datasets: 2pp. Macro floor: 0.93.** Standing policy set on 2026-05-29 in Phase 2 closure (Plan 02-02 SUMMARY §"User Resolution"); inherited by Phase 3+ without further direction; **inherited by Phase 4 under the same standing**. BBB floor = 0.844 - 0.06 = **0.784**. Other-dataset floors: MS 0.964 / TS 0.943 / TM 0.918 / JAB 0.953. Macro floor: 0.93. **Source:** STATE.md §"Standing Policy (Phases 3+)"; Phase 3 D-24.
- **D-36a:** **GATE-05 hard-tier auto-approve under the standing policy.** Per dataset: TM delta ≥ -0.01 vs 12c AND BBB delta ≥ -0.06 vs 12c → auto-approve to full sweep; marginal (-0.01 to -0.02 on TM, -0.01 to -0.06 on BBB) → halt and flag, surface checkpoint; hard reject (delta < -0.02 on TM OR < -0.06 on BBB) → no full sweep, rework. **Source:** Phase 3 D-24a; STATE.md "GATE-05 hard-tier auto-approve thresholds carry over (TM ≥ -0.01, BBB ≥ -0.06 with the wider tolerance)".
- **D-36b:** **Dual comparator (GATE vs sanity).** GATE-01 enforcement is **vs 12c** (per ROADMAP Phase 4 success criteria #1 and #3, which both reference "dual floor met"). The ablation-row "ΔF1 vs parent" column is **vs the immediate structural parent**: 13e ΔF1 vs 13c, 13f ΔF1 vs 13e (or vs 13c if 13e is rejected — D-31c chain). The 12c comparator is for GATE pass/fail; the parent comparator is for ablation-table sanity. **Source:** Phase 2 D-12; Phase 3 D-24b; orchestrator instructions ("dual-comparator deltas (vs 12c for GATE, vs parent for sanity)").

### Checkpoint Namespacing (D-37)
- **D-37:** Each variant declares its own `_VARIANT_NAME` class constant: `s_linker13e._VARIANT_NAME = "s_linker13e"`; `s_linker13f._VARIANT_NAME = "s_linker13f"`. The D-07 runtime assertion in `_checkpoint_dir` (Phase 1 INFRA-05) carries forward unchanged via the `cp`. **Source:** Phase 1 D-03 + D-07; Phase 2 D-11; Phase 3 D-25.
- **D-37a:** `run_ablation.py` registration is **append-only** — `s_linker13e` appended after `s_linker13d` (or after `s_linker13c` if the planner reorders to skip the rejected 13d slot); `s_linker13f` appended after `s_linker13e`. Same `CANONICAL_VARIANTS` / `VARIANT_SPECS` shape used in Phase 1 / Phase 2 / Phase 3. **Source:** Phase 1 D-04; Phase 2 D-11a; Phase 3 D-25a.
- **D-37b:** **Clear `results/phase_cache/s_linker13e/` and `results/phase_cache/s_linker13f/` before the first hard-tier run of each variant.** Per-variant pickle namespaces are independent (D-07 assertion), so the standard `cp` from parent does **NOT** carry cache leakage, but the planner MUST explicitly verify no stray cache dirs exist (Phase 1 Plan 05 §Issues "stray smoke-test pickle dir leaked into 13a cache" precedent). **Source:** Phase 1 Plan 05 SUMMARY §Issues; Phase 3 D-27a (cache hygiene carry-over).
- **D-37c:** **For VAR-05's dual hard-tier (D-35a): clear `results/phase_cache/s_linker13e/` between Run 1 and Run 2.** Inter-run independence is the whole point of the dual run — re-using Run 1's pickle cache in Run 2 would defeat the variance measurement. The clear step is part of the plan. **Source:** ROADMAP Phase 4 success criterion #2 ("both runs agree within Claude's normal run-to-run variance" — independence required); MEMORY.md §"LLM Variance" ("Not fixable by temperature/seed", run-to-run is the actual measurement).

### Variance Re-Run Trigger (D-38)
- **D-38:** **Within-variant variance re-run only on marginal flag.** For 13f (single-hard-tier, D-35b): same as Phases 2-3 — run a second hard-tier pass (cache cleared) only if the first hits the marginal band (D-36a). No variance re-run on auto-approve, no variance re-run on hard reject. For 13e (dual-hard-tier, D-35a): the dual run **IS** the variance measurement — no additional variance re-run unless **Run 3** would be needed because Run 1 and Run 2 disagreed (in which case the variant is already rejected per D-35a's third condition). **Source:** Phase 2 D-14; Phase 3 D-26 (carry-over with VAR-05 dual-run override).
- **D-38a:** **If 13e full-sweep BBB drifts into the [-0.04, -0.06] band: treat as D-13a cache-stream-timing artifact, not classification-coverage failure.** Same admit pattern as 13c (Plan 02-02 SUMMARY) and 13d (D-27b). The 6pp BBB carry-over admits this band. If BBB drops > 6pp, surface for user direction — do NOT auto-loosen the standing policy further. **Source:** Phase 2 D-13a, Plan 02-02 SUMMARY §"Evidence for D-13a"; Phase 3 D-27b.

### LLM-Substitution Inertness Risk (D-39)
- **D-39:** **VAR-05's risk is materially higher than VAR-04's (13d) was.** Per Plan 03-01 SUMMARY §"Failure-Mode Analysis", VAR-04 (`_classify_mention` regex → LLM enum) failed because the LLM could not learn the **project-specific** Java-package convention (`ui.website`, `logic.api` → `indirect`) without explicit per-pattern training data. **VAR-05 has analogous risk** — the LLM must classify alias *scope* (global vs local) based on a taxonomy that `_is_strong_alias` derives from **structural pattern matching** (multi-word, hyphenated, CamelCase, all-caps, capitalized-leading-letter). Some of these are universal naming conventions (CamelCase) but others encode project conventions (e.g., "single-word lowercase alias collides with English vocabulary" is an empirical observation from the BBB/TS benchmarks, not a universal rule). **Pre-mitigation in the prompt (D-32c) bakes the taxonomy in as positive reasoning rules, but the failure mode of "LLM emits `global` for everything" (defeating the gate) or "LLM emits `local` for true global aliases" (FN tail) is real.**
- **D-39a:** **The dual hard-tier protocol (D-35a) is the explicit mitigation.** Per ROADMAP Phase 4 success criterion #2 ("run twice on hard tier before full sweep — widest blast radius"), the dual run catches both (i) Claude run-to-run variance and (ii) a systematically wrong taxonomy (which would manifest as a stable bad-delta in BOTH runs, not just one). **If both runs fail GATE-05, the variant is rejected — no third run.** **Source:** ROADMAP Phase 4 §Phase Details; MEMORY.md §"Phase 3 lessons" (LLM substitutions of project-specific structural rules can fail catastrophically — 13d -19pp TM).
- **D-39b:** **VAR-06's risk is materially lower than VAR-05's.** The coref-prompt fold (D-33) does not add a new taxonomy — it asks the LLM "is the antecedent quote a canonical-name reference or an alias reference?", a question the LLM is already implicitly answering when picking the antecedent. **VAR-06 inherits the single-hard-tier protocol (D-35b).** **Source:** MEMORY.md §"V39 Coref Unification & Judge Analysis" (Variant E coref is the cross-model Pareto winner — 0 FP on both Claude and GPT-5.2, evidence the coref prompt is robust to schema extensions).
- **D-39c:** **Cache-stream timing perturbation (Spike-001 mechanism) applies to both variants.** Both 13e and 13f extend prompts on calls that already exist (D-32 modifies the `_learn_document_knowledge_enriched` call; D-33 modifies the `_coref_cases_in_context` call). No new LLM calls are added; only the response schema grows. **One-time cache miss on the modified prompts is expected**, but the same-call ordering is preserved. The 6pp BBB carry-over (D-36) admits the empirical [-0.04, -0.06] band of cache-stream perturbation. **Source:** Phase 2 D-15; Phase 3 D-27.

### Taboo Audit (D-40)
- **D-40:** **Real audit, not smoke test.** Both variants add new prompt-schema text (D-32 alias scope rules; D-33 coref antecedent-via-alias guidance). Every new prompt constant runs through the same substring-match BENCHMARK_TABOO audit Phases 1-3 used. **Source:** GATE-04; Phase 1 Plan 05 §Issues "gui in ambiguity"; Phase 3 D-28.
- **D-40a:** **Specific hazards for the scope-taxonomy prompt (VAR-05, D-32c):** The universal-taboo list includes `client`, `server`, `storage`, `common`, `logic`, `cache`, `auth`, `recording`, `persistence`, `facade`, `database/DB`, `registry`, `UI`, `model`, `preferences`, `conversion`, `validation`, `dedicated`, `cascade`. Any example or guidance phrase ("e.g. `auth` is a single-word lowercase alias", "e.g. `Cache` is a CamelCase global") must avoid these. Use safe SE-textbook placeholders: `TaskScheduler`, `Scheduler`, `Dispatcher`, `Broker`, `Parser`, `Lexer`. **Source:** BENCHMARK_TABOO.md; MEMORY.md ("No dataset-specific examples in prompts — data leakage. Use safe SE textbook domains").
- **D-40b:** **Specific hazards for the coref-fold prompt (VAR-06, D-33):** The coref prompt's existing wording (`s_linker13c.py:974-989`) is already audited. The new field guidance must avoid alias example terms that overlap with benchmark vocabulary. Reuse the same safe-SE placeholders as the upstream alias-scope prompt for consistency. **Source:** Phase 2 D-16 (smoke-audit pattern); Phase 3 D-28b (Spike 003 placeholder reuse).

### Ablation Log Rows (D-41)
- **D-41:** Two new rows in the ablation table (`PROMO-03`) for 13e and 13f. Each row records: per-dataset F1 (5 columns), macro F1, ΔF1 vs 12c (5 columns + macro), ΔF1 vs parent (1 column for macro per D-36b), `rules_removed` list, FP-by-phase breakdown (seed / entity / coref). Markdown via `tabulate` (LaTeX output deferred to Phase 5 PROMO-03). The 13c row from Phase 2 and the 13d "RETIRED" row from Phase 3 remain in the table. **Source:** Phase 2 D-17; Phase 3 D-29; REQUIREMENTS.md PROMO-03; ROADMAP Phase 4 success criterion #4.
- **D-41a:** **The 13e row also records inter-run variance metadata** (Run 1 F1, Run 2 F1, |Δ| on TM and BBB) for the methodology writeup (PROMO-04, Phase 5). The full sweep row is from a single run (D-35); the dual-run data lives in the SUMMARY, not in the canonical ablation table row. **Source:** ROADMAP Phase 4 success criterion #2; MEMORY.md §"V32 Final Audit" (variance metadata supports the writeup).

### Wave Structure (D-42)
- **D-42:** **Two sequential plans**, one per variant:
  - **Plan 04-01: VAR-05** — `s_linker13e.py` creation, prompt-schema extension, `AliasEntry` dataclass, consumer migration (all 6 read sites per D-34), registration, BENCHMARK_TABOO audit on the new alias-scope prompt text, dual hard-tier (TM + BBB) gate per D-35a, full 5-project sweep, ablation row + variance metadata.
  - **Plan 04-02: VAR-06** — `s_linker13f.py` creation (from 13e if 13e passed, from 13c if 13e was rejected — D-31c fallback), coref prompt-schema extension, consumer migration (single site at L1014-1016), registration, BENCHMARK_TABOO audit on the new coref-fold prompt text, single hard-tier (TM + BBB) gate per D-35b, full 5-project sweep, ablation row.
  - **04-02 MUST NOT start until 04-01 closes** — 13f is defined as 13e-minus-the-coref-alias-check; the 13e that ships is 13f's parent file. If 13e is rejected, the planner spins 13f from 13c (D-31c fallback) and the cumulative-removal claim narrows accordingly. **Source:** ROADMAP Phase 4 §Success Criteria (#1 + #2 lock 13e; #3 + #4 lock 13f as downstream); Phase 2 D-18 wave-structure precedent.

### Claude's Discretion
- 13e and 13f docstring exact wording (constraints fixed in D-31a; precise phrasing free).
- Exact JSON shape of the alias-scope prompt response — Option A (record-per-alias, flat) vs Option B (paired-dicts with `scopes`). Default Option A per D-32; planner may switch if a downstream prompt-consumer site reads more cleanly with Option B. Both are valid under the GATE-03 spec.
- Coref schema field exact name — `antecedent_via_alias: bool` (D-33 default) vs `coref_alias_used: str | null` (D-33c alternative). Planner picks during implementation.
- Whether `_is_strong_alias` is kept as a `@deprecated` diagnostic helper in 13e for the side-by-side log (D-34b option (i)) or rewritten as a single in-test line and deleted immediately (D-34b option (ii)). Default option (ii) for cleaner `rules_removed` claim; planner may pick (i) if the side-by-side log is wanted as a one-shot diagnostic before deletion.
- Whether to thread `AliasEntry` through the existing `_format_evidence` consumer (L533-541) as a single dataclass read or as separate `.component` / `.scope` accesses — both byte-identical at the consumer site.
- Whether the parity unit test (D-34a side-by-side log) lives in `tests/` (preferred, discoverable by `pytest`) or inline as a temporary one-shot script. Default: `tests/`; Spike pattern is acceptable.
- Print-banner string update in `__init__` per D-31b.
- Whether to clear `results/phase_cache/s_linker13e/` and `results/phase_cache/s_linker13f/` between the hard-tier and full sweep within a variant (within-variant, not between-runs — D-37c is the BETWEEN-runs requirement). Default: **no clearing** between hard-tier and full sweep within a variant — per-dataset checkpoints are independent and full sweep extends hard-tier (matches Phase 2 / Phase 3 §Claude's Discretion precedent).

### Folded Todos

None — STATE.md "Pending Todos" is empty.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (`gsd-phase-researcher`, `gsd-planner`) MUST read these before planning or implementing.**

### Project specs (gate definitions, requirement IDs, standing policy)
- `.planning/PROJECT.md` — Core Value (macro F1 ≥ 93% or reject), Key Decisions (base=12c, ablation unit = variant, standalone files), constraints (Claude Sonnet only, no benchmark leakage).
- `.planning/REQUIREMENTS.md` — **VAR-05, VAR-06** (Phase 4 scope: `s_linker13e.py` + `s_linker13f.py`); **GATE-01..GATE-06** (every variant must satisfy); Traceability rows for VAR-05 and VAR-06.
- `.planning/ROADMAP.md` §Phase 4 (lines 74-85) — goal, depends-on, success criteria #1-4 (including the "widest blast radius — run twice on hard tier" instruction for 13e and the "no call to `_has_strong_alias_mention`" + `antecedent_via_alias` field requirement for 13f).
- `.planning/STATE.md` §"Standing Policy (Phases 3+)" — 6pp BBB carry-over, 2pp others, 0.93 macro floor; D-02 single-run baseline; GATE-05 thresholds carry-over.

### Phase 1 inheritance (decisions that carry forward)
- `.planning/phases/01-baseline-and-infrastructure/01-CONTEXT.md` — D-02 (single-run baseline), D-03/D-04 (`_VARIANT_NAME` discipline), D-07 (runtime assertion).
- `.planning/phases/01-baseline-and-infrastructure/01-05-SUMMARY.md` — Spike 001 lessons (LLM-substitution inertness, prompt-cache-stream timing perturbation, BBB variance); §"Gate Resolution (2026-05-28)" + §"Variance Re-Run (2026-05-16)" precedent.

### Phase 2 inheritance (decisions that carry forward)
- `.planning/phases/02-ambiguity-cleanup/02-CONTEXT.md` — D-08 (standalone file pattern), D-09 (inline prompts), D-10 (single-run sweep), D-11 (`_VARIANT_NAME` per variant), D-12 (ΔF1 vs parent for ablation; vs 12c for gate), D-13 (BBB tolerance), D-13b (GATE-05 thresholds), D-14 (variance re-run trigger), D-15 (LLM-substitution inertness), D-17 (ablation row schema), D-18 (sequential plans).
- `.planning/phases/02-ambiguity-cleanup/02-01-SUMMARY.md` — 13b shipped clean; pure-removal does not exhibit BBB perturbation (D-13a); canary-probe pattern.
- `.planning/phases/02-ambiguity-cleanup/02-02-SUMMARY.md` — 13c canonical full-sweep JSON `results/ablation_results/ablation_20260528_201851.json`, macro 0.9314, BBB 0.7818 (0.0022 above 6pp floor); D-13a RECONFIRMED via 5/5 parity probe; user 6pp BBB resolution on 2026-05-29.

### Phase 3 inheritance (decisions and rejection artifact)
- `.planning/phases/03-mention-classifier-migration/03-CONTEXT.md` — D-19..D-30 (Phase 3 decisions); D-24 standing policy carry-over confirmation; D-27 LLM-substitution inertness with prompt-content vs call-count framing.
- `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` — **VAR-04 RETIRED** (hard reject -19pp TM on dotted-path FPs); milestone-level lesson that LLM substitutions of project-specific structural rules can fail catastrophically; user resolution 2026-05-29 (path 3: drop VAR-04, close Phase 3 empty); `s_linker13d.py` left in tree as rejection artifact. **Direct precedent for VAR-05's elevated-risk framing (D-39).**

### Spikes (validated rule classifications)
- `.planning/spikes/002-rules-audit/` — confirms `_is_strong_alias`, `_get_strong_alias_mappings`, and `_has_strong_alias_mention` are REPLACEABLE (not RISKY, not ESSENTIAL). Background: `_has_standalone_mention` is the lone RISKY classification, deferred to Phase 5 / EXT-01.
- `.planning/spikes/001-llm-trailing-words/README.md` — variant-shipping protocol template; cite-evidence pattern reference.
- `.planning/spikes/003-llm-mention-classifier/README.md` — piggyback-on-existing-LLM-call pattern reference (relevant template for D-32 schema extension on existing call).

### Codebase targets (lines to read/edit)
- `src/llm_sad_sam/linkers/experimental/s_linker13c.py` — **copy source for 13e** (1132 lines).
  - L46: `EvidenceBundle.mention_type` comment (string contract documented).
  - L174, L179: `self.doc_knowledge.aliases` length reads — unchanged after `AliasEntry` migration (dict-length still works).
  - L254-275: `_is_strong_alias` static method (to delete in 13e; may be kept as a `@deprecated` diagnostic helper per D-34b option (i)).
  - L277-282: `_get_strong_alias_mappings` method (to delete in 13e; replaced by inline list-comp at L741 per D-32d).
  - L319-355: `_learn_document_knowledge_enriched` — primary VAR-05 prompt-schema extension site (D-32).
  - L389: `knowledge.aliases[term] = comp` — alias assignment, change to `AliasEntry(...)` (D-34).
  - L533-541: `aliases.items()` iteration in `_format_evidence` — read `.component` (D-34).
  - L569-573: `aliases.items()` in alias-match text formatter — read `.component` (D-34).
  - L665-672: `_run_coreference` entry point — uses `_coref_cases_in_context` (Variant E).
  - L739-741: `_get_strong_alias_mappings()` callsite — replaced by inline list-comp on `entry.scope == "global"` (D-32d).
  - L952-1020: `_coref_cases_in_context` — primary VAR-06 prompt-schema extension site (D-33) and consumer migration site (L1014-1016, D-33a).
  - L1014-1016: `_has_strong_alias_mention` callsite in coref antecedent verification — migration to `res.get("antecedent_via_alias", False)` (D-33a).
  - L1068-1080: `_has_strong_alias_mention` method definition (to delete in 13f).
  - `_VARIANT_NAME` constant + D-07 assertion (in `_checkpoint_dir`) — preserve as-is via the `cp`; only `_VARIANT_NAME` value changes (D-37).
- `run_ablation.py` — append `s_linker13e` after `s_linker13d` in `CANONICAL_VARIANTS` and `VARIANT_SPECS`; append `s_linker13f` after `s_linker13e`. Same registration shape as Phase 1-3 plans (D-37a).
- `BENCHMARK_TABOO.md` — full project list + universal-taboo list; D-40a hazard list for alias-scope prompt; D-40b hazard list for coref-fold prompt.

### Baseline JSON files (reuse, do NOT re-run)
- `results/ablation_results/ablation_20260528_173020.json` — **12c canonical baseline** (MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405). GATE-01 comparator for both variants (D-36b).
- `results/ablation_results/ablation_20260528_201851.json` — **13c canonical full-sweep** (Plan 02-02; macro 0.9314, BBB 0.7818). Parent comparator for 13e (D-35).
- `results/ablation_results/ablation_20260528_190916.json` — 13b full-sweep (Plan 02-01; macro 0.9519). Reference only — NOT in the 13e→13f parent chain (D-31c).
- `results/ablation_results/ablation_20260529_110532.json` — 13d hard-tier (Plan 03-01; RETIRED). Reference only — NOT in the parent chain (D-31c).

### Research context (background, not action)
- `.planning/research/ARCHITECTURE.md` — pipeline structure (Tier 1 / Tier 2 / coref / boundary-filter wave); locates `_learn_document_knowledge_enriched` and `_coref_cases_in_context` in the pipeline.
- `.planning/research/PITFALLS.md` — Claude run-to-run variance documentation backing D-35a / D-39.

### Memory / prior art
- MEMORY.md — standalone-file preference (D-31); LLM run-to-run variance backing dual hard-tier protocol (D-35a, D-39); Spike-001-style LLM-substitution inertness (D-39, D-39c); §"V39 Coref Unification & Judge Analysis" Variant E cross-model Pareto framing (D-33b backing low VAR-06 risk); §"Phase 3 lessons" backing elevated VAR-05 risk (D-39); "No dataset-specific examples in prompts — data leakage" (D-40a, D-40b); GPT compatibility is a side concern, not a gate (out of scope per PROJECT.md constraint).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_VARIANT_NAME` pattern + D-07 runtime assertion (Phase 1 INFRA-05) — carries forward unchanged via the `cp` from 13c. 13e sets `_VARIANT_NAME = "s_linker13e"`; 13f sets `"s_linker13f"`.
- `run_ablation.py` `CANONICAL_VARIANTS` / `VARIANT_SPECS` append-only registration — exercised in Phase 1 Plan 05 and Phase 2 / Phase 3 plans; same shape for 13e and 13f.
- 12c full-sweep baseline JSON (Phase 1 Plan 04, `results/ablation_results/ablation_20260528_173020.json`) — reuse for ΔF1 vs 12c per D-35 + D-36b. **Do not re-run 12c.**
- 13c canonical full-sweep baseline JSON (Phase 2 Plan 02-02, `results/ablation_results/ablation_20260528_201851.json`) — reuse for ΔF1 vs 13c per D-35 + D-36b. **Do not re-run 13c.**
- `tabulate` dep (Phase 1 D-06, exercised in Phase 2 D-17 and Phase 3 D-29) — reused for the 13e and 13f ablation rows.
- Existing `_coref_cases_in_context` Variant E prompt (s_linker13c.py:974-989) — drop-in extension point for VAR-06's `antecedent_via_alias` field; the LLM already has ±5 bidirectional sentence context, so the new field requires only an output-schema bullet, not new input context.
- Existing `_learn_document_knowledge_enriched` prompt (s_linker13c.py:319-355) — drop-in extension point for VAR-05's `scope` field; the LLM already sees the full document and the component list, so the new field requires only an output-schema bullet + a scope-reasoning guide.

### Established Patterns
- Standalone variant files (12a/b/c/d/e + 13a/b/c + 13d-as-rejection-artifact) — duplicated code is the project's reproducibility artifact.
- Append-only registration in `run_ablation.py`.
- Per-variant pickle cache namespacing under `results/phase_cache/<_VARIANT_NAME>/<dataset>/`; D-07 fail-fast assertion catches namespace bugs at construct time.
- Hard-tier-first (TM + BBB) → full 5-project sweep gate sequence (with the VAR-05-specific dual-run override per D-35a).
- Inline prompt constants; BENCHMARK_TABOO substring audit on every new prompt text.
- Single-run sweeps (no N-run median, no best-of-N); variance re-run only on D-38 marginal-band trigger (or D-35a explicit dual-run for VAR-05).
- Schema extension on an existing LLM call (no new LLM call added) — the Spike-003 piggyback pattern, validated in Phase 3 13d (despite VAR-04's retirement, the *schema-extension* pattern itself is sound; what failed was the taxonomy the LLM was asked to learn).

### Integration Points
- `_learn_document_knowledge_enriched` (s_linker13c.py:319) is the LLM call whose output schema extends for VAR-05. The existing consumer at L347-355 iterates `data1.get("abbreviations", {})` and `data1.get("synonyms", {})`; the new code reads each record's `scope` field and constructs `AliasEntry(component=full, scope=scope)`.
- `_coref_cases_in_context` (s_linker13c.py:952) is the LLM call whose output schema extends for VAR-06. The existing consumer at L1000-1016 iterates `data.get("resolutions", [])`; the new code reads each `res.get("antecedent_via_alias", False)` and uses it in place of the regex-based `_has_strong_alias_mention(comp, ant_sent.text)` call.
- `doc_knowledge.aliases` is the central data structure touched by VAR-05 — 6 read sites in 13c (L174, L179, L389, L533-541, L569-573, L741, L1077). Each must be migrated to read off `AliasEntry` (D-34).
- The `_has_standalone_mention` callsite at L1014 (also in the coref antecedent-verification predicate) is **untouched** by both VAR-05 and VAR-06; its keep/replace decision is deferred to Phase 5 / EXT-01.
- `prompts_v2.py` is **not edited** in this phase (D-32b). All new prompt-schema text stays inline in `s_linker13e.py` and `s_linker13f.py`.

### Slip-Channel / Failure Modes to Pre-Watch
- **VAR-05 scope-mis-classification (D-39).** The LLM may emit `global` for everything (defeating the gate by re-broadcasting weak aliases) OR emit `local` for true global aliases (FN tail). The dual hard-tier protocol (D-35a) is the empirical guard. The side-by-side LLM-vs-structural log (D-34a) is the diagnostic signal.
- **VAR-05 `aliases` type-change downstream slip.** Each read site of `doc_knowledge.aliases` must be migrated (D-34); a missed site that does `aliases[term]` and expects a string will raise `AttributeError` or compare-against-string-fail silently. **Recommended planner action:** `grep -n 'aliases\[' src/llm_sad_sam/linkers/experimental/s_linker13e.py` after migration to confirm zero raw-dict-string reads remain.
- **VAR-06 `antecedent_via_alias` schema-conformance slip.** The LLM occasionally omits the new field. The consumer uses `res.get("antecedent_via_alias", False)` (D-33a) as a safe default — but a default of `False` means "require canonical-name antecedent". This is the conservative side (loses some TPs rather than admits FPs) and matches the existing structural behavior when `_has_strong_alias_mention` returned `False`.
- **Cache-stream timing on BBB.** Both 13e and 13f modify prompt text. Per D-39c, one-time cache miss is expected. The 6pp BBB carry-over (D-36) admits the [-0.04, -0.06] band as a documented limitation if it occurs. If 13e or 13f full-sweep BBB drops > 6pp, surface for user direction per D-38a.
- **VAR-05 dual-run inter-variance > variance band.** Per D-35a, if Run 1 and Run 2 disagree by > 0.02 on TM macro OR > 0.04 on BBB macro, the variant is rejected even if both individual runs pass auto-approve thresholds. This is a documented planner protocol — the third condition is the variance check itself.
- **Cumulative-removal chain integrity.** If 13e is rejected (D-31c fallback), 13f's parent reverts to 13c. The `rules_removed` claim then narrows to exactly `["_has_strong_alias_mention"]` (not `["_is_strong_alias", "_get_strong_alias_mappings", "_has_strong_alias_mention"]`). The planner MUST update the 13f docstring accordingly under the fallback branch.

</code_context>

<specifics>
## Specific Ideas

- **VAR-05 is the highest-risk variant in the entire 13-series.** Phase 3 retired VAR-04 because the LLM could not learn a project-specific structural classification (dotted-path Java package convention). VAR-05 asks the LLM to learn a structurally analogous classification (alias scope: global vs local), with the same risk profile. **The dual hard-tier protocol (D-35a) is the explicit early-warning system.** If 13e fails the dual run, log it the same way 13d was logged in Plan 03-01 SUMMARY — empirical evidence that the rule does not survive LLM substitution — and trigger the D-31c fallback for 13f.
- **VAR-06 is the lowest-risk variant in the chain.** The coref antecedent-via-alias call is structurally simpler than a taxonomy-classification call: the LLM is already picking the antecedent sentence and quoting it, so it implicitly knows whether the quote uses the canonical name or an alias. MEMORY.md §"V39 Coref Unification & Judge Analysis" notes Variant E coref is the cross-model Pareto winner — 0 FP on both Claude and GPT-5.2 — which suggests the coref prompt's representational capacity is healthy and a single output-bit addition is unlikely to perturb it.
- **The 6pp BBB tolerance remains standing policy, not a one-time exception.** Phase 2 set it, Phase 3 inherited it without question, Phase 4 inherits it without question (D-36). Phase 5's promotion artifact will record it as the final gate the winning variant cleared.
- **If 13e BBB lands within the original 2pp tolerance despite the dual-run schema extension on the alias prompt:** that is fresh evidence the BBB perturbation pattern documented in Phase 1 (Spike 001) and reconfirmed in Phase 2 (13c parity probe) is NOT driven by prompt-content changes — only by call-count or call-ordering changes. This is publishable empirical evidence for the methodology writeup (PROMO-04, Phase 5).
- **Conversely, if 13e BBB drifts > 2pp despite no new LLM call:** that is evidence the BBB perturbation IS driven by prompt-content changes (one-time cache miss on the longer prompt). Either reading is publishable.
- **The dual-comparator (vs 12c for GATE, vs parent for sanity) is now structurally established across Phases 2-4** — same pattern as Phase 2 D-12, Phase 3 D-24b, Phase 4 D-36b. Phase 5's ablation table presents both comparators in adjacent columns.

</specifics>

<deferred>
## Deferred Ideas

- **`_has_standalone_mention` keep/replace decision** — Phase 5 / EXT-01. `_has_standalone_mention` is consumed at `s_linker13c.py:1014` (coref antecedent verification, alongside the to-be-deleted `_has_strong_alias_mention`) and at the entity / seed pipelines. Spike 002 classified it RISKY (O(N×M) anchor collection). Phase 4 leaves it untouched in both 13e and 13f; the formal KEEP decision is logged in Phase 5 per PROMO-02.
- **`_classify_mention` regex** — VAR-04 RETIRED in Phase 3; the regex remains in 13c and is inherited unchanged in 13e and 13f. Phase 5's ablation table records VAR-04 as "13d — TM regression, retired."
- **Combining VAR-05 + VAR-06 into a single variant** — explicitly rejected by ROADMAP Phase 4's two-criterion structure and by the "one rule removal per variant" Key Decision (D-08, D-42). Keep separate.
- **Lenient enum coercion for the alias-scope field** (analogue of Phase 3 D-21b Spike 003 lenient pattern) — the planner may choose between STRICT (raise on unknown scope value) and LENIENT (fall back to `local`). **Recommended default: STRICT** for the same ablation-purity argument as D-21a (silent fallback to `local` would convert a prompt-conformance regression into an FN tail that looks like variance). The planner may switch to LENIENT during 13e implementation if a one-shot run shows the LLM occasionally emits a near-synonym (e.g., `"specific"` instead of `"local"`); document the choice in the 04-01 SUMMARY.
- **LaTeX rendering of the ablation table** — `tabulate` will emit it via `tablefmt="latex"`; this is a Phase 5 PROMO-03 deliverable. Phase 4 ships only the rows (markdown via `tabulate`) for 13e and 13f.
- **GPT-5.2 cross-model run of 13e / 13f** — EXT-03, out of scope. Claude Sonnet only per PROJECT.md constraint and MEMORY.md ("Always use Claude Sonnet … never opus").
- **Prompt-schema extraction into `prompts_v2.py`** — the new scope-taxonomy and antecedent-via-alias schema text could live in a shared module if Phase 5 wants to reuse it for the methodology writeup. Phase 4 keeps it inline per D-32b; the planner for Phase 5 may extract at that point.
- **Three-valued alias scope (global / mixed / local)** — out of scope. The existing `_is_strong_alias` is binary; the LLM emits a binary `scope`. A finer taxonomy is its own decision in a follow-up phase if Phase 5 motivates it.
- **Coref fold using a different schema field name** (e.g., `antecedent_alias_match: str | null`) — D-33c documents the string-form alternative. Default boolean per D-33; planner may switch.

### Reviewed Todos (not folded)
None — STATE.md "Pending Todos" is empty.

</deferred>

---

*Phase: 04-alias-scope-and-coref-fold*
*Context gathered: 2026-05-29 (auto mode — recommended defaults selected from Phase 1-3 precedent + ROADMAP Phase 4 + Spike 002 rules-audit; D-31..D-42)*
