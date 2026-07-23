# Phase 6: EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive — Pattern Map

**Mapped:** 2026-05-30
**Files analyzed:** 5 (2 new linker files + 1 canonical promotion + 2 modified upstream files)
**Analogs found:** 5 / 5 — every file has a strong in-repo analog.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py` | linker (standalone variant) | event-driven DAG → request-response LLM batches | `src/llm_sad_sam/linkers/experimental/s_linker13.py` | **exact** (copy-fork, byte-equivalent skeleton; per-variant rule swap) |
| `src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py` | linker (standalone variant) | event-driven DAG → request-response LLM batches | `src/llm_sad_sam/linkers/experimental/s_linker13.py` | **exact** (same copy-fork pattern; differs only in prompt + dropped pre-filter) |
| `src/llm_sad_sam/linkers/experimental/s_linker13g.py` | linker (canonical, byte-copy of winner) | event-driven DAG → request-response LLM batches | `src/llm_sad_sam/linkers/experimental/s_linker13.py` | **exact** (mirrors the 13f→s_linker13 promotion pattern in the same file's docstring) |
| `src/llm_sad_sam/linkers/experimental/prompts_v2.py` (modified — append constants) | prompt-constants module | config / static data | `src/llm_sad_sam/linkers/experimental/prompts_v2.py:179-205` (`ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`) | **exact** (same module, same constant shape, same docstring convention) |
| `run_ablation.py` (modified — register 3 variants) | ablation runner / registry | config | `run_ablation.py:40-80` + `run_ablation.py:274-316` (existing 13a..13 entries) | **exact** (mechanical append; pattern repeated ~30× already) |

**Note:** No file extracts the new LLM primitive into a helper module. Per GATE-07 + PROJECT.md "one rule = one standalone variant file", every linker is copy-pasted standalone. The Spike-003 piggyback for the `mention_type` field is in-class only (no helper module).

---

## Pattern Assignments

### `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py` (linker, copy-fork of s_linker13)

**Analog:** `src/llm_sad_sam/linkers/experimental/s_linker13.py` (1198 lines, byte-equivalent skeleton).

**Step-by-step:** copy `s_linker13.py` byte-for-byte, then perform 5 targeted edits below. Do NOT inherit from `SLinker13`.

---

**Structured variant docstring** (`s_linker13.py:1-20` — copy and amend `RULES_REMOVED` / `KEEP`):

```python
"""S-Linker13g_pre: 13 - _has_standalone_mention via LLM with regex pre-filter for dotted-path.

REMOVED_FROM: s_linker12c (cumulative via 13a->13b->13c->13e->13f->13)
RULES_REMOVED: ["_split_component_name (13a partial)",
                "_is_structurally_unambiguous (13b)",
                "_is_ambiguous_name_component (13c)",
                "_is_strong_alias (13e)",
                "_get_strong_alias_mappings (13e)",
                "_has_strong_alias_mention (13f)",
                "_has_standalone_mention (13g; LLM primitive + regex pre-filter for dotted-path)"]
KEEP: ["_in_dotted_or_hyphen_context_only (Phase 7 / EXT-02 will remove this pre-filter)"]
"""
```

Then drop the `# KEEP DECISION (PROMO-02, Phase 5, 2026-05-29):` comment block at `s_linker13.py:22-35` (the rule is no longer kept).

---

**Class declaration + variant namespace** (`s_linker13.py:133-136` — change two strings):

```python
class SLinker13gPre:                           # was: SLinker13
    """LLM-driven SAD-SAM traceability — EXT-01 sub-variant (a): regex pre-filter + LLM judge."""

    _VARIANT_NAME = "s_linker13g_pre"          # was: "s_linker13"
```

**Critical:** `_VARIANT_NAME` MUST be unique per sub-variant. The assertion at `s_linker13.py:1165` catches missing-namespace bugs but NOT collision; choosing distinct strings is the only defense (Pitfall 2 in RESEARCH.md).

---

**Imports pattern** (`s_linker13.py:37-62` — unchanged, plus one new constant):

```python
from __future__ import annotations

import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink,
    ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import (
    Sentence, load_sentences, build_sent_map,
)
from llm_sad_sam.linkers.experimental.ilinker3 import ILinker3
from llm_sad_sam.linkers.experimental.prompts_v2 import (
    AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES,
    ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES,
    STANDALONE_MENTION_RULES_PRE_FILTERED,   # NEW for EXT-01 sub-variant (a)
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend
```

---

**Cite-evidence LLM-call pattern (REUSE — the canonical shape; 5 occurrences in s_linker13.py)** — `s_linker13.py:561-568`:

```python
# Retry-once + approve-biased fallback. The new _compute_standalone_mention_map MUST follow this shape.
for attempt in range(2):
    data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
    if data and data.get("disambiguations"):
        break
    if attempt == 0:
        print(f"    [{comp_name}] Empty response, retrying...")
if not data:
    verified.extend(valid_seeds)  # Keep all on failure (approve-biased)
    continue
```

Apply this shape verbatim for the new map-building call. **Approve-bias for EXT-01 specifically = mark all batch sentences as `standalone=True` on LLM failure** (matches the `verified.extend(valid_seeds)` recall-protection intent).

---

**Per-component batched-scan pattern (REUSE — analog batching topology)** — `s_linker13.py:748-803` (`_run_single_extraction_pass`):

```python
def _run_single_extraction_pass(self, sentences, comp_names, mappings,
                                 name_to_id, sent_map, pass_label=""):
    batch_size = 50
    candidates = {}

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]
        # ... print progress
        prompt = f"""Extract ALL references to software architecture components ...

COMPONENTS: {', '.join(comp_names)}
...
DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "..."}}]}}
JSON only:"""

        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
            if data and data.get("references"):
                break
            if attempt == 0:
                print(f"    {pass_label}Empty response, retrying batch...")

        if not data:
            continue

        for ref in data.get("references", []):
            # parse + validate + collect
            ...
    return candidates
```

**Mirror this exactly** for the new `_compute_standalone_mention_map(sentences, components) -> dict[(comp_name, snum), bool]`:
- `batch_size = 50` (proven on all 5 benchmark sizes)
- per-component outer loop (the analog iterates pass × batch; the new function iterates component × batch)
- `chr(10).join(f"S{s.number}: {s.text}" ...)` sentence-block format (matches every existing prompt in `s_linker13`)
- `N_INTEGER` placeholder in the JSON template (consistent with `s_linker13.py:772`, `1068` — also handled by `_parse_snum` at `s_linker13.py:1107-1117`)
- `_parse_snum` (REUSE — `s_linker13.py:1107-1117`) for parsing `"S42"` / `"42"` / `42` consistently.

---

**Dotted-path pre-filter helper (NEW, sub-variant a only)** — extract from `s_linker13.py:1136-1146` (the regex guards inside `_has_standalone_mention`):

```python
@staticmethod
def _in_dotted_or_hyphen_context_only(comp_name, text):
    """Return True iff EVERY occurrence of comp_name in text sits inside a
    dotted/hyphenated context (i.e., regex baseline would reject all matches).

    Pre-filter for sub-variant (a): if True, skip this sentence — do not pay
    LLM cost for a guaranteed-no candidate. EXT-02 (Phase 7) deletes this
    helper.
    """
    if not comp_name:
        return False
    is_single = ' ' not in comp_name
    if is_single:
        if comp_name[0].islower():
            pattern = rf'\b{re.escape(comp_name)}\b'
        else:
            cap_name = comp_name[0].upper() + comp_name[1:]
            pattern = rf'\b{re.escape(cap_name)}\b'
        flags = 0
    else:
        pattern = rf'\b{re.escape(comp_name)}\b'
        flags = re.IGNORECASE

    any_match = False
    for m in re.finditer(pattern, text, flags):
        any_match = True
        s, e = m.start(), m.end()
        # Reuse the same 4 guards from _has_standalone_mention
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return False  # found at least one clean occurrence
    return any_match  # True iff there were matches but all were dotted/hyphen
```

**Source:** byte-extracted from `s_linker13.py:1119-1147`. Do NOT fork the regex semantics. Document the helper as "pre-filter only; EXT-02 removes."

---

**New Tier-1 task: `_compute_standalone_mention_map`** (NEW — mirrors the batching pattern above + the cite-evidence shape):

```python
def _compute_standalone_mention_map(self, sentences, components):
    """Document-level: returns dict[(comp_name, snum) -> bool].

    Sub-variant (a): regex pre-filter strips dotted/hyphen-only contexts before
    the LLM sees the sentence; LLM judges only word-boundary-clean candidates.
    Sub-variant (b) overrides this method to skip the pre-filter entirely.
    """
    comp_names = self._get_comp_names(components)
    smap = {}  # (cname, snum) -> bool

    for cname in comp_names:
        # PRE-FILTER (sub-variant a only — sub-variant b drops this `if`)
        cand_sents = [
            s for s in sentences
            if cname.lower() in s.text.lower()
            and not self._in_dotted_or_hyphen_context_only(cname, s.text)
        ]
        if not cand_sents:
            continue

        for batch_start in range(0, len(cand_sents), 50):
            batch = cand_sents[batch_start:batch_start + 50]
            prompt = f"""{STANDALONE_MENTION_RULES_PRE_FILTERED}

COMPONENT: {cname}
SENTENCES:
{chr(10).join(f"S{s.number}: {s.text}" for s in batch)}

Return JSON: {{"results": [{{"component": "{cname}", "sentence": N_INTEGER, "standalone": true}}]}}
JSON only:"""
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("results"):
                    break
                if attempt == 0:
                    print(f"    Standalone-mention [{cname}] retry...")
            if not data:
                # Approve-biased fallback (matches s_linker13.py:567-568 pattern)
                for s in batch:
                    smap[(cname, s.number)] = True
                continue
            for r in data.get("results", []):
                snum = self._parse_snum(r.get("sentence"))
                if snum is not None:
                    smap[(cname, s.number)] = bool(r.get("standalone", True))
    return smap
```

---

**DAG tier integration (REUSE `_run_parallel` shape)** — extend `s_linker13.py:240-244`:

```python
# Before (s_linker13.py:240-244):
acq = self._run_parallel({
    "model": lambda: self._analyze_model(components),
    "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
    "seed": lambda: self._run_seed(sentences, components),
})

# After (s_linker13g_pre):
acq = self._run_parallel({
    "model": lambda: self._analyze_model(components),
    "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
    "seed": lambda: self._run_seed(sentences, components),
    "standalone_map": lambda: self._compute_standalone_mention_map(sentences, components),  # NEW
})
# ... right after the existing acq assignments at s_linker13.py:246-248:
self._standalone_map = acq["standalone_map"]
```

**Checkpoint persistence (REUSE)** — add `"standalone_map"` to `_save_phase` in the Tier-1 block at `s_linker13.py:259-263`:

```python
self._save_phase(text_path, "standalone_map", {
    "standalone_map": self._standalone_map,
})
```

Open Question #1 in RESEARCH.md confirms this should be a **separate** top-level phase (so D-02 diff stage can be re-run from pickle without re-running the rest of Tier-1).

---

**Drop-in replacement at the 6 call sites** — replace every `self._has_standalone_mention(comp_name, sent_or_text)` with `self._has_standalone_mention_llm(comp_name, snum)`:

```python
def _has_standalone_mention_llm(self, comp_name, snum):
    """Lookup against the precomputed map. No regex, no LLM call."""
    return self._standalone_map.get((comp_name, snum), False)
```

| Line in s_linker13.py | Before | After |
|-----------------------|--------|-------|
| `510` (seed_val anchor) | `if self._has_standalone_mention(comp_name, s.text):` | `if self._has_standalone_mention_llm(comp_name, s.number):` |
| `623` (via `_classify_mention`) | `if self._has_standalone_mention(comp_name, text):` | Replaced via Spike-003 piggyback — see next subsection |
| `675` (bundle anchor) | `if self._has_standalone_mention(comp_name, s.text):` | `if self._has_standalone_mention_llm(comp_name, s.number):` |
| `880` (`has_exact_case`) | `has_exact_case = self._has_standalone_mention(c.component_name, sent.text)` | `has_exact_case = self._has_standalone_mention_llm(c.component_name, sent.number)` |
| `895` (generic-filter anchor) | `if self._has_standalone_mention(comp_name, s.text):` | `if self._has_standalone_mention_llm(comp_name, s.number):` |
| `1095` (coref antecedent) | `if not (self._has_standalone_mention(comp, ant_sent.text) or res.get(...)):` | `if not (self._has_standalone_mention_llm(comp, ant_sent.number) or res.get(...)):` |

**Delete** the static method `_has_standalone_mention` at `s_linker13.py:1119-1147` entirely.

---

**Spike-003 piggyback (REUSE pattern for call site #2)** — the `_classify_mention` regex switch at `s_linker13.py:617-649` is replaced via the existing entity-extraction prompt. Pattern source: `.planning/spikes/003-llm-mention-classifier/spike.py:34-52`:

```python
MENTION_TYPES = {"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"}

def format_mention(mention_type: str, alias_used: str | None = None) -> str:
    if mention_type == "proper_case":
        return "proper case, standalone"
    if mention_type == "lowercase":
        return "lowercase mention"
    if mention_type == "dotted_path":
        return "lowercase, inside dotted path"
    if mention_type == "via_alias":
        if alias_used:
            return f'via known alias "{alias_used}"'
        return "via known alias"
    return "indirect/unclear match"
```

**Caveat (Pitfall 6 in RESEARCH.md):** Spike-003 is an **API-shape pattern reference**, NOT proof of safety — the 13d failure was on the LLM-emission side, not the consumer side. For sub-variant (b), planner must keep `mention_type` emission **conservative** (default to `"indirect"` on parse failure) and let the Tier-1 map carry the standalone signal.

**Recommended approach for EXT-01:** Use `self._standalone_map` for the `proper_case` branch of `_classify_mention` (the line that calls `_has_standalone_mention`); leave the other branches (lowercase, alias, dotted-path) on regex inside `_classify_mention` for now. Reason: the lowercase / alias branches are NOT part of the rule being removed in EXT-01 — they remain regex-driven in s_linker13.

---

**Per-variant checkpoint-dir assertion (REUSE — DO NOT TOUCH)** — `s_linker13.py:1159-1170`:

```python
def _checkpoint_dir(self, text_path):
    cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
    ds = os.path.splitext(os.path.basename(text_path))[0]
    d = os.path.join(cache_dir, self._VARIANT_NAME, ds)
    # D-07: fail-fast if the directory does not embed the variant name.
    assert self._VARIANT_NAME in d, (
        f"_checkpoint_dir must contain _VARIANT_NAME "
        f"('{self._VARIANT_NAME}' not in '{d}')"
    )
    os.makedirs(d, exist_ok=True)
    return d
```

This file does NOT need edits beyond keeping it intact — the `_VARIANT_NAME` value change at the class declaration is enough.

---

### `src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py` (linker, copy-fork of s_linker13)

**Analog:** `src/llm_sad_sam/linkers/experimental/s_linker13.py` (same starting point as `_pre`).

Diffs from `s_linker13g_pre.py`:

1. **`_VARIANT_NAME`** = `"s_linker13g_sem"`.
2. **Class name** = `SLinker13gSem`.
3. **Imports:** import `STANDALONE_MENTION_RULES_LLM_ONLY` instead of `STANDALONE_MENTION_RULES_PRE_FILTERED`.
4. **`_compute_standalone_mention_map`** — DROP the `_in_dotted_or_hyphen_context_only` pre-filter call. Pass ALL sentences whose lowercased text contains `cname.lower()` to the LLM:

   ```python
   cand_sents = [s for s in sentences if cname.lower() in s.text.lower()]
   ```

5. **`_in_dotted_or_hyphen_context_only` static method** — NOT included. The dotted-path concept is taught in the prompt (`STANDALONE_MENTION_RULES_LLM_ONLY`), not enforced in code. This is the sub-variant (b) bet.
6. **Docstring `KEEP` list** = `[]` (the pre-filter is not kept; full LLM-only).

Everything else (DAG integration, 6 call site swaps, `_has_standalone_mention_llm` lookup, deletion of `_has_standalone_mention`) is byte-identical to sub-variant (a).

**Pitfall 1 warning (13d redux):** This is the highest-risk cell in the matrix. The prompt MUST contain the dotted-identifier negative example (Example 2 from RESEARCH.md "Prompt Domain Selection"). Skipping it likely reproduces the -19pp TeaMMates regression that retired VAR-04.

---

### `src/llm_sad_sam/linkers/experimental/s_linker13g.py` (canonical, byte-copy of winner)

**Analog:** the byte-copy promotion pattern in `src/llm_sad_sam/linkers/experimental/s_linker13.py:1-20` (which itself was a byte-copy of `s_linker13f`).

**Step-by-step:**

1. `cp src/llm_sad_sam/linkers/experimental/s_linker13g_<winner>.py src/llm_sad_sam/linkers/experimental/s_linker13g.py`
2. Search-and-replace `_VARIANT_NAME = "s_linker13g_<winner>"` → `"s_linker13g"`. Verify: `grep _VARIANT_NAME src/llm_sad_sam/linkers/experimental/s_linker13g.py` returns exactly one line with `"s_linker13g"` (Pitfall 2 in RESEARCH.md).
3. Search-and-replace `class SLinker13g<Winner>:` → `class SLinker13g:`.
4. Rewrite docstring header per `s_linker13.py:1-20`:

```python
"""S-Linker13g: canonical promotion of s_linker13g_<winner> (Phase 6, 2026-XX-XX).

REMOVED_FROM: s_linker12c (cumulative via 13a->13b->13c->13e->13f->13->13g_<winner>)
RULES_REMOVED: ["_split_component_name (13a partial)",
                "_is_structurally_unambiguous (13b)",
                "_is_ambiguous_name_component (13c)",
                "_is_strong_alias (13e)",
                "_get_strong_alias_mappings (13e)",
                "_has_strong_alias_mention (13f)",
                "_has_standalone_mention (13g; LLM primitive)"]
KEEP: [<winner-specific kept rules, if any (sub-variant a keeps the pre-filter; sub-variant b keeps nothing)>]

s_linker13g is the canonical EXT-01 deliverable. Full-sweep macro F1 = <X.XXXX>
(results/ablation_results/ablation_<TS>.json). See .planning/phases/06-*/06-SUMMARY.md
for the EXT-01 study narrative + the tagged ## EXT-01 cost/quality signal block (D-06).
"""
```

5. Wipe checkpoint dir before final canonical sweep:
   ```bash
   rm -rf results/phase_cache/s_linker13g/
   ```
   (Pitfall 2 in RESEARCH.md — prevents cross-variant pickle contamination.)

---

### `src/llm_sad_sam/linkers/experimental/prompts_v2.py` (modify — append 2 new constants)

**Analog:** existing constants `ENTITY_EXTRACTION_RULES` (`prompts_v2.py:179-191`), `VALIDATION_RULES` (`prompts_v2.py:194-205`), and the section-header convention (`prompts_v2.py:10-11`, `67-69`, `175-177`).

**Required structure** (same as `ENTITY_EXTRACTION_RULES` shape — block of natural-language rules + a JSON schema line at the end):

```python
# Append AFTER prompts_v2.py:222 (end of COREF_RULES block) and BEFORE the
# SEED_DISAMBIGUATION_RULES section. Use the established `# ═══` section header.

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Standalone-Mention Detection (EXT-01)
# ═══════════════════════════════════════════════════════════════════════════════

STANDALONE_MENTION_RULES_PRE_FILTERED = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence contains a standalone reference to the named component (the name appears as a subject, object, or named participant — not only as an ordinary English word).

RULES:
1. YES when the component name appears as a standalone token — as the subject of an architectural action, in a list of components, or named as a participant.
2. NO when the name is used only as an ordinary English word with its dictionary meaning, with no architectural intent.
3. YES when the name is configured, queried, or named as the target of an interaction (e.g., "data is stored in X", "via X", "through X").
4. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


STANDALONE_MENTION_RULES_LLM_ONLY = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence makes a standalone reference to the named component; NO if the name appears only as part of a longer code identifier or as an ordinary English word.

RULES:
1. YES when the component name appears as a standalone token, including as a subject, object, or in a list of components.
   Example: "The Parser consumes tokens emitted by the lexer." → YES for Parser.
2. NO when the name appears only inside a qualified or dotted identifier.
   Example: "The class compiler.parser.ASTBuilder extends the base class." → NO for Parser; Parser is a path segment, not a standalone reference.
3. NO when the name participates only in a hyphenated compound that denotes a different entity.
   Example: "Parser-style grammar" → NO for Parser.
4. YES when the name is the subject of an architectural action — performs work, provides a service, is configured, receives input.
   Example: "Disk I/O is handled by the FileSystem." → YES for FileSystem.
5. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""
```

**GATE-06 audit checklist** (REUSE — pattern from `prompts_v2.py:1-8` module docstring: "All examples use safe SE textbook domains ... Zero benchmark-derived terms."):

- All example terms (`Parser`, `ASTBuilder`, `compiler`, `lexer`, `FileSystem`) cross-checked against `BENCHMARK_TABOO.md`. Safe (per BENCHMARK_TABOO.md "Safe SE Textbook Examples" lines 62-68).
- No use of `logic`, `UI`, `client`, `storage`, `common`, `model`, `database`, `cache`, `registry`, `auth`, `server`, `persistence`, `facade`, `recording`, `cascade`, `validation`, `internal`, `adapter`, `order`, `processor`, `event`, `socket`, `layer`, `preferences`, `config` (the Universal Taboo list, Pitfall 4 in RESEARCH.md).

**Module docstring update (optional, recommended):** update `prompts_v2.py:4` to mention the new domain coverage (compiler, OS, e-commerce, graphics — already covered).

---

### `run_ablation.py` (modify — append 3 entries to each list)

**Analog:** existing `s_linker13a..13` entries at `run_ablation.py:74-80` (CANONICAL_VARIANTS) and `run_ablation.py:274-316` (VARIANT_SPECS).

**Edit 1 — append to `CANONICAL_VARIANTS` list** (after `run_ablation.py:79`, the `"s_linker13"` line):

```python
CANONICAL_VARIANTS = [
    ...
    "s_linker13f",
    "s_linker13",      # canonical promotion of 13f (Phase 5)
    "s_linker13g_pre",   # NEW — EXT-01 sub-variant (a): regex pre-filter + LLM judge
    "s_linker13g_sem",   # NEW — EXT-01 sub-variant (b): LLM-only, dotted-path in prompt
    "s_linker13g",       # NEW — canonical promotion of winning sub-variant (Phase 6)
]
```

**Edit 2 — append to `VARIANT_SPECS` dict** (after `run_ablation.py:316`, the `s_linker13` entry):

```python
VARIANT_SPECS = {
    ...
    "s_linker13": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13",
        class_name="SLinker13",
        description="S-Linker13: canonical promotion of s_linker13f (Phase 5) — 6 rules removed cumulatively from 12c",
        canonical=True,
    ),
    "s_linker13g_pre": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_pre",
        class_name="SLinker13gPre",
        description="S-Linker13g-pre: 13 - _has_standalone_mention via LLM with regex pre-filter for dotted-path (EXT-01 sub-variant a)",
    ),
    "s_linker13g_sem": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_sem",
        class_name="SLinker13gSem",
        description="S-Linker13g-sem: 13 - _has_standalone_mention via LLM-only (dotted-path encoded in prompt) (EXT-01 sub-variant b)",
    ),
    "s_linker13g": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g",
        class_name="SLinker13g",
        description="S-Linker13g: canonical promotion of winning EXT-01 sub-variant (Phase 6) — 7 rules removed cumulatively from 12c",
        canonical=True,
    ),
}
```

**Verification (GATE-07 enforcement):**

```bash
python run_ablation.py --list-variants | grep s_linker13g
# Expect 3 lines: s_linker13g_pre, s_linker13g_sem, s_linker13g
```

Pitfall 3 in RESEARCH.md: forgetting either list causes silent half-registration. Always edit both in the same commit.

---

## Shared Patterns

### Cite-Evidence LLM Call (project-wide; 5 occurrences in s_linker13.py)
**Sources:** `s_linker13.py:561-568, 775-783, 932-940, 1016-1021, 1072-1078`.
**Apply to:** Every LLM call in the new variants — specifically `_compute_standalone_mention_map`.
```python
for attempt in range(2):
    data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
    if data and data.get("<key>"):
        break
    if attempt == 0:
        print(f"    <context>: empty response, retrying...")
if not data:
    <approve_biased_fallback>  # Keep all on failure (recall protection)
    continue
```

### Approve-Biased Fallback (project-wide)
**Sources:** `s_linker13.py:567-568` (seed val), `s_linker13.py:783` (extraction), `s_linker13.py:939` (generic filter).
**Apply to:** `_compute_standalone_mention_map` on LLM failure — mark all sentences in the failed batch as `standalone=True`. Matches the recall-protection intent.

### DAG Parallel Task Execution (project-wide)
**Source:** `s_linker13.py:190-210` (`_run_parallel`).
**Apply to:** Tier-1 standalone-map task is a 4th lambda in the existing `_run_parallel({"model": ..., "doc_knowledge": ..., "seed": ..., "standalone_map": ...})` call at `s_linker13.py:240-244`. No new orchestration code — just extend the dict.

### Per-Variant Checkpoint Namespacing (GATE D-07)
**Source:** `s_linker13.py:1159-1170` (`_checkpoint_dir` + assertion at line 1165).
**Apply to:** All three new linker files. `_VARIANT_NAME` distinct per sub-variant (`s_linker13g_pre` / `s_linker13g_sem` / `s_linker13g`); the byte-copy promotion to canonical MUST update `_VARIANT_NAME` (Pitfall 2 in RESEARCH.md).

### Structured Variant Docstring (GATE-07)
**Source:** `s_linker13.py:1-20` (the `REMOVED_FROM` / `RULES_REMOVED` / `KEEP` shape).
**Apply to:** All three new linker files. Sub-variant (a) `KEEP` lists the kept pre-filter; sub-variant (b) `KEEP = []`. Canonical `s_linker13g.py` inherits the winner's `KEEP` list.

### Prompt-Constant Section Header
**Source:** `prompts_v2.py:10-12`, `67-69`, `142-144`, `175-177`, `208-210`, `225-227`.
**Apply to:** New `STANDALONE_MENTION_RULES_*` constants get their own `# ═══ Tier 1 — Standalone-Mention Detection (EXT-01) ═══` section header.

### Sentence-Number Parsing (`_parse_snum`)
**Source:** `s_linker13.py:1107-1117`.
**Apply to:** Parsing `r.get("sentence")` from the standalone-mention LLM response (handles `"S42"` / `"42"` / `42`).

### Sentence-Block Prompt Format
**Source:** `s_linker13.py:769` (extraction), `s_linker13.py:1050` (coref), `s_linker13.py:917` (generic filter).
**Apply to:** All new prompts — `chr(10).join(f"S{s.number}: {s.text}" for s in batch)` is the universal sentence-block format. Use `N_INTEGER` as the integer placeholder in JSON templates.

### Variant Registration in Both Lists (GATE-07)
**Source:** `run_ablation.py:40-80` (CANONICAL_VARIANTS) + `run_ablation.py:274-316` (VARIANT_SPECS).
**Apply to:** All three new variants get 1 list entry + 1 dict entry. Canonical `s_linker13g` gets `canonical=True` in its spec dict (mirrors line 315 for `s_linker13`).

---

## No Analog Found

| File | Reason |
|------|--------|
| (none) | All planned files have direct in-repo analogs. |

The "diff-stage harness" mentioned in RESEARCH.md (D-02 anchor-collection diff) is a planner-discretion file (potentially `--diff-stage` flag in `run_ablation.py`, per Open Question #3 in RESEARCH.md). If the planner creates it as a separate script, its closest analog is `run_ablation.py`'s per-variant load + LLM-cached-execution path (`run_ablation.py:364-379` ILinker3Adapter pattern). Defer this judgement to the planner.

---

## Metadata

**Analog search scope:** `src/llm_sad_sam/linkers/experimental/` (12 files), `src/llm_sad_sam/core/` (data types + loader), `src/llm_sad_sam/`, `run_ablation.py`, `.planning/spikes/003-llm-mention-classifier/`.

**Files scanned:** 7 primary analogs read in detail (s_linker13.py, prompts_v2.py, run_ablation.py, spike.py, 06-CONTEXT.md, 06-RESEARCH.md, 06-DISCUSSION-LOG.md).

**Strong-match early-stop:** All 5 file slots have exact-quality analogs in the same module/path; no broader search needed. The s_linker13 → s_linker13g copy-fork pattern is the dominant analog and is reused 3× (sub-variant a, sub-variant b, canonical promotion).

**Pattern extraction date:** 2026-05-30.

**Verified-against-source claims:**
- Six call sites for `_has_standalone_mention`: lines 510, 623, 675, 880, 895, 1095 — verified via direct read of s_linker13.py.
- `_has_standalone_mention` body: lines 1119-1147 — verified.
- `_run_parallel` shape: lines 190-210 — verified.
- `_VARIANT_NAME` assertion: line 1165 — verified.
- Cite-evidence pattern occurrences: 5 (lines 561, 775, 932, 1016, 1072) — verified via direct reads.
- `prompts_v2.py` constants location: lines 14-247 — verified end-to-end (247 lines total).
- `run_ablation.py` CANONICAL_VARIANTS + VARIANT_SPECS shape: lines 40-80 + 274-317 — verified.
- Spike-003 `format_mention` shape: spike.py:37-52 — verified.
