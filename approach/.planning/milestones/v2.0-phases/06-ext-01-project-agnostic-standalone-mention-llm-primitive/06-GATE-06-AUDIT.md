# Phase 6 — GATE-06 Generality Audit

**Status:** PRE-CLEARANCE (Plan 01 prompts)
**Updated:** 2026-05-30
**Final canonical audit:** TBD — written by Plan 04 after canonical sweep.

## Scope of this audit

Two prompt constants newly added to `src/llm_sad_sam/linkers/experimental/prompts_v2.py`:
- `STANDALONE_MENTION_RULES_PRE_FILTERED` (sub-variant a: regex pre-filter + LLM judge) — 889 bytes
- `STANDALONE_MENTION_RULES_LLM_ONLY` (sub-variant b: LLM-only, dotted-path in prompt) — 1245 bytes

Both constants live under the section header `# Tier 1 — Standalone-Mention Detection (EXT-01)`, appended after `COREF_RULES` (line 222) and before the `# Tier 2 — Seed Reference Disambiguation` section.

## (a) BENCHMARK_TABOO.md mechanical scan

Command (whole file — for context only, since the scope of this audit is the two new constants):

    grep -iE -o "(UserDBAdapter|AudioWatermarking|Reencoding|MediaManagement|Facade|MediaAccess|Packaging|FileStorage|TagWatermarking|UserManagement|DownloadLoadBalancer|ParallelWatermarking|WebUI|Registry|Persistence|Recommender|SlopeOneRecommender|OrderBasedRecommender|DummyRecommender|PopularityBasedRecommender|ImageProvider|PreprocessedSlopeOneRecommender|Logic|Storage|Common|Test Driver|GAE Datastore|Recording Service|kurento|WebRTC-SFU|HTML5 Server|HTML5 Client|Presentation Conversion|BBB web|Redis PubSub|FSESL|Redis DB|FreeSWITCH|globals|bibdatabase|bibentry|watermark|reencoding|datastore|recording|pubsub|html5|preferences|cascade|conversion|dedicated|adapter|processor|cache|registry|persistence|facade|server|database|recommender)" src/llm_sad_sam/linkers/experimental/prompts_v2.py

Whole-file stdout (sorted unique):

```
server
```

This single hit comes from the literal substring `obSERVER` inside the pre-existing `VALIDATION_RULES` example (`"observer pattern"`, line 204) — NOT from the two new constants, and not a benchmark term in any case. The Plan 01 GATE-06 scope is the two new constants only; the narrow-scope scan below is the operative check.

Narrow-scope command (only the two NEW constants — the operative GATE-06 check):

    awk '/STANDALONE_MENTION_RULES_PRE_FILTERED|STANDALONE_MENTION_RULES_LLM_ONLY/,/^"""$/' src/llm_sad_sam/linkers/experimental/prompts_v2.py | grep -iE "(UserDBAdapter|AudioWatermarking|Reencoding|MediaManagement|Facade|MediaAccess|Packaging|FileStorage|TagWatermarking|UserManagement|DownloadLoadBalancer|ParallelWatermarking|WebUI|Registry|Persistence|Recommender|SlopeOneRecommender|OrderBasedRecommender|DummyRecommender|PopularityBasedRecommender|ImageProvider|PreprocessedSlopeOneRecommender|Logic|Storage|Common|Test Driver|GAE Datastore|Recording Service|kurento|WebRTC-SFU|HTML5 Server|HTML5 Client|Presentation Conversion|BBB web|Redis PubSub|FSESL|Redis DB|FreeSWITCH|globals|bibdatabase|bibentry|watermark|reencoding|datastore|recording|pubsub|html5|preferences|cascade|conversion|dedicated|adapter|processor|cache|registry|persistence|facade|server|database|recommender)" || echo "NO HITS"

Recorded result:

```
NO HITS
```

Expected match count in the two NEW constants: 0. Recorded match count: 0. **PASS.**

### Substring-match artefact (documented for Plan 04)

The PLAN's `<verification>` block uses a non-word-bounded regex, which produces one false-positive substring match: `ui` matches inside `ASTBuilder` (`AST**Bui**lder`). This is NOT a benchmark term — `ASTBuilder` is on BENCHMARK_TABOO.md's confirmed-safe whitelist (line 62, Compiler design). The semantically correct check uses word boundaries:

    awk '/STANDALONE_MENTION_RULES_PRE_FILTERED|STANDALONE_MENTION_RULES_LLM_ONLY/,/^"""$/' src/llm_sad_sam/linkers/experimental/prompts_v2.py | grep -iwE "(logic|UI|client|storage|common|cache|registry|persistence|facade|recording|cascade|conversion|dedicated|adapter|processor|kurento|freeswitch|redis|bbb|html5|preferences|globals|watermark|reencoding|recommender|datastore)"

Stdout:

```
NO HITS (word-bounded)
```

Plan 04 should use `-iwE` (word-bounded) when re-running the canonical sweep to avoid false positives from substring overlaps.

### Words used in the new prompts (whitelist)

Cross-check against BENCHMARK_TABOO.md §"Safe SE Textbook Examples" (lines 60-68):

- `Parser` — Compiler design, safe.
- `ASTBuilder` — Compiler design (AST + Builder), safe.
- `compiler.parser` — qualified-identifier example; the words `compiler`, `parser` are safe (compiler-design lingua franca).
- `lexer` — Compiler design, safe.
- `FileSystem` — Operating systems, safe.
- `Disk` / `I/O` — Operating systems, safe.
- `tokens` / `grammar` / `class` / `extends` / `base class` — generic SE/programming, safe.

None of these words appear in any of the 5 benchmark projects' component, alias, or keyword lists (BENCHMARK_TABOO.md §MediaStore/TeaStore/Teammates/BigBlueButton/JabRef, lines 6-29) nor in the Universal Taboo list (lines 31-58).

## (b) Reviewer-defensibility check

For each example in `STANDALONE_MENTION_RULES_LLM_ONLY`, answer the test "would a reviewer believe this approach generalizes to a random new project?":

| Example | Project-agnostic? | Why |
|---------|-------------------|-----|
| "The Parser consumes tokens emitted by the lexer." (Rule 1, YES case) | Yes | Compiler-design lingua franca; appears in textbooks (e.g., Dragon Book). No tie to any benchmark project. |
| "The class compiler.parser.ASTBuilder extends the base class." (Rule 2, NO case) | Yes | Qualified-identifier pattern is universal in OO languages (Java, C#, Python). Not specific to TeaMMates / Java packages. |
| "Parser-style grammar" (Rule 3, NO case) | Yes | Hyphenated-modifier pattern is generic English. |
| "Disk I/O is handled by the FileSystem." (Rule 4, YES case) | Yes | OS-textbook example. No tie to any benchmark. |

For `STANDALONE_MENTION_RULES_PRE_FILTERED`: contains no domain examples — rules are abstract instructions only (the placeholder `X` is used in rule 3). Trivially project-agnostic.

## Decision

PASS for Plan 01 scope. The two prompts encode no project structure and use only confirmed-safe example domains. Final GATE-06 record (post-sweep, includes helper code from Plan 02) is appended by Plan 04.

## Open items handed to Plan 04

- Re-run the taboo scan after Plan 02 introduces `_in_dotted_or_hyphen_context_only` (helper code, not prompt — but GATE-06 also covers helpers).
- Append the canonical sweep's macro F1 and per-dataset F1 into a final "Generality vs Performance" section.
- TBD — written by Plan 04: final canonical audit (post-sweep).

---

## Plan 06-05 — Alias-aware prompt pre-clearance

**Status:** PRE-CLEARANCE (Plan 06-05 prompts — alias-aware quartet)
**Date:** 2026-05-30

### Scope

Four NEW prompt constants newly added to `src/llm_sad_sam/linkers/experimental/prompts_v2.py`:
- `STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE` — sub-variant `pre_alias`: regex pre-filter + LLM judge with KNOWN ALIASES block injected.
- `STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE` — sub-variant `sem_alias`: LLM-only + KNOWN ALIASES block injected.
- `STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE` — sub-variant `pre_full`: regex pre-filter + LLM judge with KNOWN ALIASES + RUNNING LINK MAP blocks injected.
- `STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE` — sub-variant `sem_full`: LLM-only + KNOWN ALIASES + RUNNING LINK MAP blocks injected.

All four live under the new section header `# Tier 1 — Standalone-Mention Detection (EXT-01) — Alias-Aware (Plan 06-05)`, appended AFTER the Plan 06-01 section and BEFORE `# Tier 2 — Seed Reference Disambiguation`.

### (a) BENCHMARK_TABOO.md mechanical scan (word-bounded, operative check)

Command:

    awk '/STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE|STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE|STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE|STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE/,/^"""$/' src/llm_sad_sam/linkers/experimental/prompts_v2.py | grep -iwE "(UserDBAdapter|AudioWatermarking|Reencoding|MediaManagement|Facade|MediaAccess|Packaging|FileStorage|TagWatermarking|UserManagement|DownloadLoadBalancer|ParallelWatermarking|WebUI|Registry|Persistence|Recommender|SlopeOneRecommender|OrderBasedRecommender|DummyRecommender|PopularityBasedRecommender|ImageProvider|PreprocessedSlopeOneRecommender|Logic|Storage|Common|Test Driver|GAE Datastore|Recording Service|kurento|WebRTC-SFU|HTML5 Server|HTML5 Client|Presentation Conversion|BBB web|Redis PubSub|FSESL|Redis DB|FreeSWITCH|globals|bibdatabase|bibentry|watermark|reencoding|datastore|recording|pubsub|html5|preferences|cascade|conversion|dedicated|adapter|processor|cache|registry|persistence|facade|server|database|recommender|event|socket|layer|UI|client|storage|common)" || echo "NO HITS"

Recorded stdout:

    NO HITS

**PASS.** Zero benchmark surface forms appear under word-bounded grep across the four new constants.

### (b) Reviewer-defensibility check

| Example sentence | Project-agnostic? | Why |
|---|---|---|
| "The Parser consumes tokens emitted by the lexer." (rule 1, YES) | Yes | Compiler-design textbook (Dragon Book); no overlap with any benchmark term. |
| alias-aware example: `SymTbl -> SymbolTable`, "SymTbl is consulted before scope resolution." (rule 2, YES) | Yes | Compiler design (SymbolTable); abbreviation `SymTbl` is generic and does not match any benchmark abbreviation (no overlap with `bbb-html5`, `FSESL`, `WebRTC-SFU`, etc.). |
| "The class compiler.parser.ASTBuilder extends the base class." (NO case for dotted identifier) | Yes | Qualified-identifier pattern universal in OO languages; ASTBuilder is on the BENCHMARK_TABOO safe whitelist (line 62). |
| "Parser-style grammar" (NO case for hyphenated compound) | Yes | Hyphenated-modifier pattern is generic English. |
| "Disk I/O is handled by the FileSystem." (YES, subject of architectural action) | Yes | OS-textbook example. |
| `S12: Scheduler` + "S13: It then assigns the task to an idle worker." (pronoun resolution in `*_FULL_KNOWLEDGE`) | Yes | Operating-systems textbook (Scheduler); generic English pronoun "It"; +-3 sentence locality window is a language-universal heuristic, not a project-specific value. |

For the two `*_FULL_KNOWLEDGE` constants: the additional RUNNING LINK MAP block uses only Scheduler (OS) as an example component name. The pronoun list ("it", "the component", "the service") is generic English.

### (c) Integration-shape generality (D-11)

The four prompts use TWO literal placeholder tokens (`{KNOWN_ALIASES_BLOCK}` and `{RUNNING_LINK_MAP_BLOCK}`) that the linker code in Plan 06-06 substitutes at call time with whatever upstream stages discovered. On a project with zero discovered aliases the alias block becomes the literal line `(none discovered)`; on a project with zero prior links the linkmap block becomes `(none yet)`. No project-specific term or pattern is baked into the prompt text — the prompts are project-agnostic data sinks for project-agnostic upstream outputs.

### Decision

PASS for Plan 06-05 scope. The four new alias-aware prompts encode no project structure and use only confirmed-safe example domains. Plan 06-06 will pre-clear the linker bodies (helper code) and Plan 06-08 records the final post-sweep canonical audit.

### Open items handed to Plan 06-06

- Re-run the mechanical scan on the four new `s_linker13g_{pre,sem}_{alias,full}.py` linker bodies after they are introduced (helper code is in GATE-06 scope).
- Verify no benchmark surface forms enter via the alias/linkmap injection helpers (the substitution code must echo upstream-discovered terms verbatim — it must not normalize, rewrite, or inject any hand-coded values).
