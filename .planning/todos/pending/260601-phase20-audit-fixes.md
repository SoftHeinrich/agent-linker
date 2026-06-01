---
id: 260601-phase20-audit-fixes
created: 2026-06-01
phase: 20
priority: high
blocks: Phase 21 run
---

# Phase 20 Code Audit Fixes

From code-quality-reviewer audit of Phase 20 changes (Gate A/B + axiom + coref).
Fix all issues before running Phase 21 probe.

## ISSUE #1 — Premature convergence / mislabeled committed_f1s (CRITICAL)
File: `scripts/voyager_train_tlr_v4_beta.py` lines 1047, 1076

`committed_f1s = train_f1s` stores pre-commit F1. `converged = committed_macro >= 0.90`
uses that pre-commit value → convergence fires on unmeasured bank state.
On pass 1 with empty prior_f1s, if axiom floor ≥ 0.90 → training terminates with 0 patterns.

Fix: gate convergence on `pass_num >= 2` minimum, or re-label as "L F1 this pass" and
only fire convergence when patterns were actually committed AND pass_num >= 2.

## ISSUE #2 — Gate A/B cross-project FM-ID collision (CRITICAL)
File: `scripts/voyager_train_tlr_v4_beta.py` lines 732-755, 832

D is per-project so proposals cite FMs from that project's O only. But `_gate_a_check`
validates against ALL o_jsons' FM IDs (FM-1 from project A and FM-1 from B both valid).
`fm_lookup` in Gate B is a plain dict → project B's FM-1 silently overwrites project A's.
Gate B may judge a proposal against the wrong failure mode entirely.

Fix: scope FM IDs per project, e.g. `f"{project}:FM-1"`, both in D's addresses_failure_modes
output (update D prompt) and in Gate A/B lookup construction.
OR: pass only the relevant project's o_json to Gate B per proposal.

## ISSUE #3 — `_bank_content_hash` ignores axiom file (CRITICAL)
File: `scripts/voyager_train_tlr_v4_beta.py` lines 165-171

Only hashes `slot_patterns`. After Phase 20 changed COREF_RULES + SEED_DISAMBIGUATION_RULES,
cached L results from old axioms will be reused — silently stale.

Fix: include hash of `prompts_v3_axiom.py` source text in the cache key.
E.g. `hashlib.md5(Path("src/.../prompts_v3_axiom.py").read_bytes()).hexdigest()[:8]`

## ISSUE #4 — `role_ref_pat` matches generic terminal words (HIGH)
File: `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` ~line 965

"service", "manager", "system", "controller" as multi-word terminal words → "the service"
fires on every generic sentence. V31 lessons: "Server" was a known partial-name landmine.
High recall noise risk on BBB/TM.

Fix: maintain a GENERIC_TERMINALS blocklist (common English architectural nouns that are
too broad: service, manager, system, controller, server, client, handler, provider, module,
layer, component, adapter, gateway, proxy). Skip these from comp_terminals.

## ISSUE #5 — `seen_proposal_titles` dedup never fires (MEDIUM)
File: `scripts/voyager_train_tlr_v4_beta.py` line 1001

D schema has no `title` field. `pat.get("title", str(pat))[:40]` always uses dict repr
(unique per proposal). Dedup is dead code.

Fix: use `(pat.get("slot","") + pat.get("rule_text","")[:60]).lower()` as dedup key.

## ISSUE #6 — Gate B bool strictness (MEDIUM)
File: `scripts/voyager_train_tlr_v4_beta.py` lines 852-854

`fixes is True` uses identity — fails if LLM returns string `"true"`.
`causes is False` fails silently too (string `"false"` is truthy → `causes is False` = False → accept; wrong).

Fix:
```python
def _to_bool(v): return v is True or v == "true" or v == "True"
accept = _to_bool(fixes) and not _to_bool(causes) and confidence in ("high", "medium")
```

## ISSUE #7 — Cross-project removals destructive (MEDIUM)
File: `scripts/voyager_train_tlr_v4_beta.py` lines 1006-1009, 1041-1046

Project A's D proposes removing `p_007`. This is merged and applied to ALL project banks.
Pattern IDs are per-bank so `p_007` in MS ≠ `p_007` in TS. Cross-project removals are
semantically wrong and potentially destructive.

Fix: scope removals per-project (don't merge removal lists; apply each project's D removals
only to that project's bank).

## Non-issues confirmed OK
- `anaphoric_sents` dedup: list comprehension with `or` is correct, no double-inclusion.
- `_cache_read/write` order of operations: safe.
- `predicted`/`gold` empty on cache hit: confirmed not read downstream (dead code risk only).
- Sampling limits `fps[:20]`, `fns[:10]`, `context_lines[:5]`: intentional, documented as kept.
