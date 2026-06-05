# Phase 38: Tier C Axiom + Coref Filter Widening

**Status**: not started
**Depends on**: Phase 37 (v2.6 close + GATE-06 'Persistence' fix)
**Parallel with**: Phase 39
**Budget**: $0 (no LLM training)

## Goal

The `prompts_v4_axiom.py` axiom layer gains TWO new bank slots (`COREF_TERMINAL_PARTIAL_RULES`, `SUB_COMPONENT_OF_RULES`) AND the coref runtime filter in `s_linker14_voyager.py:949` is widened so axiom changes actually reach the LLM batch. Both surface forms identified by P36 forensics (Pattern A modifier-stripped + Pattern B service-of) become reachable by coref. SEED_DISAMBIGUATION gerund-rejection rule added after a gold-link safety scan confirms no MS/TS regressions.

## Background

Per `.planning/todos/pending/2026-06-01-implement-refined-v3-axiom-diffs-feasibility-study.md`: axiom-only change has NO EFFECT. SCN sentences ("The server handles…") contain no pronoun → never match `PRONOUN_PATTERN` → never reach `_coref_cases_in_context` LLM batch. Filter widening via runtime `role_ref_pat` from component terminal words is the load-bearing change.

Per P36-FORENSICS.md: Pattern A ("server side" → HTML5 Server) and Pattern B ("MongoDB" → HTML5 Server) are distinct. Need two slots, not one.

## Subtasks

1. **10a — Axiom slots**
   - Add `COREF_TERMINAL_PARTIAL_RULES` slot to `prompts_v4_axiom.py` — modifier-stripped + section-context anchor. Skeleton + slot pattern (Voyager-compliant). Empty by default; loop-learnable.
   - Add `SUB_COMPONENT_OF_RULES` slot — internal-service → parent-component. Empty by default.
   - Wire both slots into `COREF_RULES` rendering path.

2. **10b — Code: widen coref filter**
   - Build runtime `role_ref_pat = re.compile(r'\bthe (' + '|'.join(comp_terminals) + r')\b', re.IGNORECASE)` from multi-word component terminal words.
   - Expand filter at `s_linker14_voyager.py:949`: union of PRONOUN_PATTERN-match OR role_ref_pat-match.
   - Update coref prompt header: "pronoun references" → "anaphoric references (pronouns and role-referential noun phrases)".
   - Update JSON template field: `"pronoun"` → `"reference"`.
   - Rename `pronoun_sents` / `pronoun_count` → `anaphoric_sents` / `anaphoric_count`.
   - Line 1004 gate unchanged (`has_standalone_mention` still correct for SCN antecedent sentences).

3. **10c — MS/TS gerund TP scan (SEED_DISAMBIGUATION prereq)**
   - Scan MS + TS gold links for "Providing X to Y" / "Connecting to Z" gerund TPs.
   - If any TP would be killed by gerund-rejection rule, refine rule or abort 10d.

4. **10d — SEED_DISAMBIGUATION diff** (conditional on 10c clean)
   - Edit SEED_DISAMBIGUATION_RULES OTHER clause: add "or description of the component's own capabilities without referencing an external participant".
   - Verify with TM S82/S83/S136/S159/S182 expected-rejects.

## Success Criteria

1. Two new axiom slots exist in `prompts_v4_axiom.py`, registered with the v5 training-loop slot index.
2. `_coref_cases_in_context` filter union runs on a 3-project test (MS, TS, BBB) without crashes; batch size logged.
3. P36 evidence sentences (S10, S12, S13, S26 BBB) reach the LLM batch — verified by log inspection.
4. GATE-01: `s_linker13_min` unchanged, Claude macro ≥ 0.9506, gpt-5.4 macro ≥ 0.9069.
5. GATE-06: zero benchmark vocabulary in new slot text or refactored prompt strings.
6. 10c gold-link scan documented; 10d only ships if zero MS/TS regression confirmed.

## Risk

- **Filter widening blows up batch size on TM** (long doc, many multi-word components). Mitigation: log batch sizes; cap if >30 sentences/batch.
- **`role_ref_pat` matches generic English** ("the server" outside arch context). Mitigation: pattern only built from component terminal words; matches outside Variant E ±5 window get dropped by existing gate at line 1004.
- **10d gerund rule fires on cross-component gerunds** if "external participant" check is too weak. Mitigation: 10c gold-scan first.

## Out of Scope

- Hand-curated `_generic_partials` set (s10's approach). v2.7 prefers runtime LLM check.
- CamelCase structural-unambiguous guard (Tier D — skipped per HANDOFF decision).
- Pre-seeding new slots with prompts_v2 bodies (Tier B forbidden per `feedback-voyager-no-prompt-rehydration`).

## Plans

- TBD (subtask 10a)
- TBD (subtask 10b)
- TBD (subtask 10c)
- TBD (subtask 10d, conditional)
