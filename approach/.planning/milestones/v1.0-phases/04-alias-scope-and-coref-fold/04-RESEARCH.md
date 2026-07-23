# Phase 4 — Research Notes (Alias Scope & Coref Fold)

**Compiled:** 2026-05-29
**Mode:** Skeleton notes backing the two PLAN.md files (no `gsd-phase-researcher` agent — `--skip-verify` invocation).
**Scope:** VAR-05 (`s_linker13e.py`, alias scope field) + VAR-06 (`s_linker13f.py`, coref fold). Both deltas are schema extensions on **existing** LLM calls (no new calls).

---

## 1. Alias-discovery prompt shape today (`s_linker13c.py`)

The alias-discovery surface is **two consecutive LLM calls** inside `_learn_document_knowledge_enriched` (`s_linker13c.py:318-392`):

- **Prompt 1 — extraction** (`s_linker13c.py:323-337`): emits a flat-dict JSON schema
  ```
  {"abbreviations": {"short_form": "FullComponent"},
   "synonyms":      {"specific_alternative_name": "FullComponent"}}
  ```
  Consumer loop at L347-353 flattens both dicts into a single `all_mappings: dict[str, str]`, keeping only entries whose value matches a known component name.
- **Prompt 2 — judge** (`s_linker13c.py:358-373`): returns `{"approved": [terms]}`; survivors are written into `DocumentKnowledge.aliases[term] = comp_name` at L387-390.

Net post-judge state: `doc_knowledge.aliases: dict[str, str]` (term → canonical component name). **No scope information is captured today.**

VAR-05 extends Prompt 1's schema (D-32). Two acceptable shapes:
- **Option A (default per D-32):** record-per-alias list — `{"abbreviations": [{"term":..., "component":..., "scope":...}], "synonyms":[...]}`. Pairs with `AliasEntry` dataclass naturally.
- **Option B:** parallel `scopes` dict alongside the existing flat-dict shape.

The judge prompt (Prompt 2) does **not** need a schema change — scope is fixed at extraction time.

---

## 2. `_is_strong_alias` signature + callsites (rule under retirement, VAR-05)

```python
@staticmethod
def _is_strong_alias(term: str) -> bool:
```

Defined at `s_linker13c.py:254-275`. Five-rule structural classifier (multi-word OR hyphenated OR CamelCase OR all-caps `len ≥ 2` OR starts-with-capital → STRONG; else lowercase single-word → WEAK). Encodes the empirical observation that single-word lowercase aliases (`server`, `auth`, `cache`) collide with English vocabulary and leak globally if broadcast.

**Callsites (2, both in-class):**
- L281-282 — inside `_get_strong_alias_mappings` (list-comp filter).
- L1078 — inside `_has_strong_alias_mention` (per-alias predicate).

`_get_strong_alias_mappings` (`s_linker13c.py:277-282`) returns the strong-alias subset of `doc_knowledge.aliases` formatted as `"term=component"` strings. **Single callsite:** L741 (inside `_extract_entities_enriched`, passed into both parallel extraction passes as the `mappings` arg).

**VAR-05 cleanup (D-32d, D-34):**
- Delete `_is_strong_alias` (option (ii) per D-34b — cleanest `rules_removed` claim; reuse-once for the side-by-side diagnostic log via an inline test helper, then drop).
- Delete `_get_strong_alias_mappings`; replace L741 with `[f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items() if entry.scope == "global"]`.
- Migrate every `doc_knowledge.aliases` read site (6 in 13c, listed in D-34): L174/L179 (len reads — unchanged), L281 (deleted with the method), L389 (write: now `AliasEntry(component=comp, scope=...)`), L533-541 (`_build_component_profile` evidence formatter — read `.component`), L569-573 (`_classify_mention` alias-match formatter — read `.component`), L1077 (`_has_strong_alias_mention` — in 13e reads `.scope == "global"`; deleted in 13f).

The 13e dual-hard-tier mitigation (D-35a) is the explicit early-warning for taxonomy mis-classification — see §5.

---

## 3. `_has_strong_alias_mention` signature + callsites (rule under retirement, VAR-06)

```python
def _has_strong_alias_mention(self, comp_name, sentence_text):
```

Defined at `s_linker13c.py:1068-1081`. Iterates `doc_knowledge.aliases.items()`, filters by `target == comp_name AND _is_strong_alias(alias)`, then word-boundary regex-searches `sentence_text.lower()` for each surviving alias. Returns True on first match.

**Single callsite:** L1014-1016, inside `_coref_cases_in_context` antecedent verification:
```python
if not (self._has_standalone_mention(comp, ant_sent.text) or
        self._has_strong_alias_mention(comp, ant_sent.text)):
    continue
```

The check guards coref antecedent quality — without it, the LLM could pick an antecedent sentence whose only reference to the component is a weak alias collision (an ordinary English word that happens to match), turning a TP coref into an FP.

**VAR-06 fold (D-33, D-33a):** Extend `_coref_cases_in_context`'s output schema with `antecedent_via_alias: bool` (default per D-33; string-form `coref_alias_used: str | null` alternative per D-33c). Consumer becomes:
```python
if not (self._has_standalone_mention(comp, ant_sent.text) or
        res.get("antecedent_via_alias", False)):
    continue
```
The LLM already has full sentence text (±5 bidirectional, L967) and the canonical name; deciding "is the antecedent quote a canonical-name reference or an alias reference?" is structurally simpler than the VAR-04 taxonomy that failed at -19pp TM.

VAR-06 is single-hard-tier (D-35b): the LLM is being asked an easier question, on a surface the cross-model Pareto coref winner (Variant E) already handles cleanly (0 FP on Claude and GPT-5.2 per MEMORY.md §"V39 Coref Unification & Judge Analysis").

---

## 4. Dotted-path canary template (from Phase 3 lesson)

Phase 3's 13d hard-tier produced **33 entity-source FPs on Teammates**, all driven by dotted-path Java-package references (`ui.website`, `logic.api`, `storage.entity`, `logic.core`, `ui.controller`) that the LLM `_classify_mention` substitution emitted as "concrete" instead of "indirect." The structural regex in 12c/13c correctly tagged them as `indirect`.

VAR-05 doesn't touch `_classify_mention` (out of scope, D-19 / Phase 3 closed empty), but the LLM scope-taxonomy in `_learn_document_knowledge_enriched` (Prompt 1) could analogously emit an `alias` entry for a dotted-path fragment (e.g., the LLM "discovers" `ui.website` as an alias for `UI` with `scope: global`) — that path would leak into Tier 2 entity extraction via the strong-alias mappings list and re-create the 13d failure mode by a different route.

**Canary template for Plan 04-01 Task 2/3 verify steps:** after each 13e hard-tier run, count entity-source FPs whose `matched_text` contains a `.` token (`ui.website`, `logic.api`, etc.). Sourced from the per-dataset CSV under `results/ablation_results/s_linker13e_{teammates,bigbluebutton}_links.csv`:

```python
import csv
def dotted_path_fps(csv_path):
    n = 0
    examples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            # FP rows: gold == empty / "False", correctness column varies by schema
            is_fp = (row.get("is_tp", "true").lower() in ("false", "0", "")
                     and row.get("source", "") == "entity")
            mt = (row.get("matched_text") or "").lower()
            if is_fp and "." in mt and any(c.isalpha() for c in mt.split(".")[-1]):
                n += 1
                if len(examples) < 5:
                    examples.append(f"S{row.get('sentence')}: {row.get('component')} <- {mt!r}")
    return n, examples
```

**Threshold:** 12c TM has ~2-3 entity-source FPs total; 13c has ~3. **Flag if 13e TM dotted-path FP count > 10** (one-third of 13d's 33-FP catastrophe). This is the explicit canary the plan must run — diagnostic and informational, not a hard gate (GATE-05 dual-comparator already handles the F1 drop), but it surfaces *which* failure mode is firing if F1 regresses.

If the canary fires:
- Cause: LLM scope-discovery is emitting `alias` records for dotted-path tokens (`ui.website` → `UI` global).
- Mitigation: prompt-engineer the scope-taxonomy text to explicitly reject dotted-path fragments as aliases (D-32c rule expansion), or fall back to D-34b option (i) keeping `_is_strong_alias` as a post-filter on the LLM output.
- Default response per CONTEXT.md §"Specifics": log empirically and surface for user direction (do not auto-loosen the gate).

---

## 5. Dual-hard-tier rationale (variance band, widest blast radius)

**Why VAR-05 gets the dual run (D-35a) and VAR-06 doesn't (D-35b):**

VAR-05's modification is consumed at **3+ downstream sites** that pre-existed `_is_strong_alias`:
- Entity-extraction prompts (L739-741, `_extract_entities_enriched`) — strong aliases are the *only* aliases shown to the dual-pass extractor, gating which terms are matchable.
- Evidence-bundle alias-match formatter (L569-573, `_classify_mention`) — alias hit determines the human-readable mention type fed into validation.
- Coref antecedent verification (L1015, `_has_strong_alias_mention`) — strong aliases gate which sentences can serve as antecedents (still present in 13e; deleted in 13f).

If the LLM mis-classifies (a) `global` everywhere → strong-alias list inflates → weak ordinary-English words leak into entity extraction → FP cascade across all 5 datasets; or (b) `local` everywhere → strong-alias list deflates → TPs disappear from entity extraction → FN cascade. Both manifest only at Tier 2 / Tier 3, which is exactly the widest blast radius in the 13-series.

**Claude run-to-run variance (MEMORY.md §"LLM Variance"):**
> Same model gives DIFFERENT behavior across days (Phase 1 ambiguity, Phase 3 synonyms). This is NOT code change — affects entire phases, not individual links. Not fixable by temperature/seed.

A single hard-tier pass on VAR-05 cannot distinguish (i) classifier-mistake regression from (ii) variance-band noise. The dual run with **per-run GATE-05 + inter-run variance band** (D-35a: |Δ| ≤ 0.02 on TM, |Δ| ≤ 0.04 on BBB) gives three signals at once:
- Both runs pass → confident promotion (variance is small, classifier is stable).
- Both runs fail → confident reject (systematic taxonomy error; this is the VAR-04 failure pattern).
- Disagreement > variance band → reject as unreliable (we cannot ship a variant whose downstream F1 is determined by Claude's mood on a given day).

VAR-06's surface is narrower (one consumer site at L1014-1016) and asks the LLM an easier question (already-quoted-antecedent disambiguation, not a global taxonomy). Single hard-tier is sufficient per the precedent — this is the same protocol that shipped 13b, 13c, and 13d's hard-tier.

**Dual-run hygiene constraint (D-37c):** `results/phase_cache/s_linker13e/` is cleared between Run 1 and Run 2. Cache re-use would defeat the variance measurement — the pickle cache encodes prior LLM responses, so re-using it pins Run 2 to Run 1's exact outputs.

---

## 6. References (codebase + planning artifacts)

- `src/llm_sad_sam/linkers/experimental/s_linker13c.py` — copy source for 13e (1132 lines).
- `.planning/phases/04-alias-scope-and-coref-fold/04-CONTEXT.md` — D-31..D-42 locked decisions.
- `.planning/phases/02-ambiguity-cleanup/02-01-PLAN.md` + `02-02-PLAN.md` — sequential wave template.
- `.planning/phases/02-ambiguity-cleanup/02-01-SUMMARY.md` + `02-02-SUMMARY.md` — 13b + 13c outcomes (parent baseline numbers).
- `.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` — 13d -19pp TM cautionary tale (dotted-path FPs, mechanism, recovery paths).
- `.planning/phases/01-baseline-and-infrastructure/01-04-SUMMARY.md` — 12c baseline (MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973 / macro 0.9405).
- `BENCHMARK_TABOO.md` — universal-taboo list (D-40a hazards for alias-scope prompt; D-40b for coref fold).
- MEMORY.md §"LLM Variance" + §"V39 Coref Unification & Judge Analysis" + §"Phase 3 lessons" — backing for D-35a / D-39 / D-39b.

---

*Phase: 04-alias-scope-and-coref-fold*
*Notes compiled: 2026-05-29*
