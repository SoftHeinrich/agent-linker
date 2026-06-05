# Phase 39: BBB Coref Dampening + s10 Partial-Injection Port

**Status**: not started
**Depends on**: Phase 37 (v2.6 close)
**Parallel with**: Phase 38
**Budget**: $0 (no LLM training)

## Goal

`s_linker14_voyager.py` gains (a) per-dataset coref aggressiveness flag that disables role-NP coref when `model_knowledge.ambiguous_names` is empty/small (BBB case where coref is net -3 FP per voyager-improvement-ideas #5), and (b) a ported version of `s_linker10/13` Tier 1.5 multi-word partial enrichment + Tier 2 partial injection — so BBB's structural FN ceiling (17 missed gold links from terminal partials like "Apps", "Server") becomes reachable via direct injection, not just coref extension.

## Background

Per s10/s13 forensics (cited in HANDOFF blockers and `.planning/notes/2026-06-02-voyager-improvement-ideas.md` #4): BBB ceiling is structurally below 0.87 even with axiom fixes because ILinker only seeds 65% of BBB gold links. The 17 BBB FNs are role-description aliases ILinker never aliased. Phase 38 axiom fixes recover some via coref widening; Phase 39 recovers the rest via direct partial-form enrichment + injection that s10 had.

Per voyager-improvement-ideas #5: BBB-only data shows coref adds ~2 TP + 5 FP = net -3. All BBB components are proper nouns / abbreviations (zero in `ambiguous_names`). Auto-dampening avoids damage without sacrificing coref on TM/MS/TS where ambiguous names benefit from it.

## Subtasks

1. **12 — BBB coref dampening flag**
   - In `_coref_cases_in_context`, check `len(model_knowledge.ambiguous_names) < N` (N tuned, probably 1).
   - When triggered: drop role_ref_pat-only matches (keep pronoun-only); equivalent to pronoun-only coref.
   - Log dampening trigger per project.

2. **14 — Port s10 Tier 1.5 multi-word partial enrichment**
   - Source: `s_linker10.py:392-470` (partial enrichment pass) and `s_linker13_min.py` equivalents.
   - Lift: multi-word component → set of terminal-word partials. E.g. "HTML5 Server" → {"server"}, "akka-apps" → {"apps"}.
   - Wire output into entity-extraction prompt context as candidate aliases.
   - LLM (not regex) gates generic-word filter — runtime check, no hand-curated set (per `feedback-no-hardcoding`).

3. **15 — Port s10 Tier 2 partial injection**
   - Source: `s_linker10.py:850-875` and equivalents in s13.
   - Lift: inject partial-name candidates as seed links subject to seed-disambiguation gating.
   - Protect from boundary/convention filter (s10 had `partial_inject` immunity flag — preserve here too).
   - Wire through assessor for downstream validation.

## Success Criteria

1. Coref dampening flag is automatic per-project (no manual toggle); BBB triggers, MS/TS/TM/JAB don't.
2. Tier 1.5 enrichment pass runs on all 5 projects without crashes; per-project partial-set logged.
3. Tier 2 injection produces candidate links on BBB matching s10's coverage (≥ 80% of s10's BBB partial seeds reproduced).
4. GATE-01: `s_linker13_min` unchanged, Claude macro ≥ 0.9506, gpt-5.4 macro ≥ 0.9069.
5. GATE-06: no benchmark vocabulary in new prompt or filter logic. Partial enrichment is structural (terminal-word extraction); generic-word filter is LLM-runtime.
6. Coverage report: before/after BBB recall on axiom-only baseline run (no training).

## Risk

- **Tier 2 injection adds FPs without judge** — partial mentions are ambiguous by definition. Mitigation: route through seed-disambiguation (existing path); rely on Tier C axiom (Phase 38) for terminal-partial discrimination.
- **Dampening flag triggers on borderline projects** if `ambiguous_names` threshold wrong. Mitigation: log + manual review of first run.
- **s10 code paths assumed regex-free** but reality may differ. Verify with `grep -nE 're\.' s_linker10.py | head` before lifting.

## Out of Scope

- Reintroducing hand-curated `_generic_partials` from s10. v2.7 forces LLM-runtime filter.
- Modifying ILinker3 / ILinker4 alias generation (Phase 38 axiom widening is the alias-side response).
- Hardcoded BBB-specific config. Dampening is data-driven via `ambiguous_names`.

## Plans

- TBD (subtask 12)
- TBD (subtask 14)
- TBD (subtask 15)
