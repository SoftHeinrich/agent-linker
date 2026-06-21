# Phase 50: EXTRACT - Research

**Researched:** 2026-06-21
**Domain:** Deterministic pickle→JSON extraction of frozen `s_linker20_union` per-run phase caches (no LLM, no network)
**Confidence:** HIGH — every claim below was verified by loading the actual on-disk pickles this session across all 30 cells (2 backends × 3 runs × 5 projects).

## Summary

Phase 50 is a pure, mechanical, read-only transformation: load five `*.pkl` checkpoints per (backend × run × project) cell, and re-serialize their contents into one neutral, stdlib-loadable JSON per cell so the downstream eval bundle never needs the linker classes or pickle. I loaded every pickle in the matrix and confirmed the exact Python types, the `(sentence, component_id)` tuple-key convention, the decision-dict schema (`approved/p1/p2/path/stage` for entity; `approved/path` for coref), the knowledge layer (`ModelKnowledge.ambiguous_names: set[str]`, `DocumentKnowledge.aliases: dict[str, AliasEntry]`), and the `final`/`final_provenance` shape. The on-disk matrix is **complete — 30/30 cells present, all five pkls each, no missing cells**. The N=6 "full" runs (`results/v2.6.5/full_s_linker20_union_run{1..6}/`) are **empty directories with no phase_cache → out of scope**. The sonnet backend subdir is **`claude/`** (not `anthropic/`); gpt is `openai/`.

The single most important finding for faithfulness (EXTRACT-03): the authoritative final-link set must be re-derived from the **validated *lists*** (`layer3.validated` + `layer4.coref_validated`, dedup by `(s,c)` with entity priority) — **NOT** from the `decisions`/`coref_decisions` dicts. In 8 of 30 cells `coref_raw` contains duplicate `(s,c)` keys, and because `coref_decisions` is a dict keyed by `(s,c)` it collapses duplicates (last-write-wins) while `coref_validated` is a list that preserves the first-approved occurrence. I verified that final-set == the run's own `*_links.csv` (on the `(sentence, component_id, source)` triple) for all 30 cells, whereas decision-dict re-derivation diverges by one link in `gpt/run2/teammates` and `gpt/run3/teammates`. The neutral JSON must therefore carry `final.links` (and the validated lists) as the authoritative set, and treat the decision dicts as ablation/audit metadata only.

**Primary recommendation:** Write a single stdlib + repo-import extraction script that (1) `import llm_sad_sam.linkers.experimental.s_linker20_union` so pickle can resolve `AliasEntry`, (2) walks the fixed 30-cell matrix, (3) serializes each cell to `<extract_root>/<backend>/<run>/<project>.json` using the schema in this doc — representing every `(s,c)` tuple-keyed dict as an *ordered list of records* (lossless, never collapse coref lists), sets as sorted lists, and dataclasses via `asdict`, and (4) prints a per-cell PASS/FAIL that asserts the JSON's `final.links` set equals the cell's `*_links.csv` set.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Unpickle frozen caches | Extraction script (in-repo, classes importable) | — | EXTRACT-01: must run inside `agent-linker` because pickle needs `AliasEntry`/`*Knowledge`/`*Link` classes resolvable |
| Lossless re-serialization to JSON | Extraction script (stdlib `json`) | — | EXTRACT-02: every ablation field captured with JSON-native types only |
| Final-set faithfulness oracle | Extraction script vs. run's `*_links.csv` | `ablation_*.json` (counts cross-check) | EXTRACT-03: set-equality must be machine-checked per cell |
| Downstream re-derivation (RQ3/RQ4) | Stdlib consumer in `../working/` (later phases) | — | The JSON is the contract; consumer must never import linker classes |

## Locked Design Constraints (from ROADMAP / REQUIREMENTS / GATES)

> No `CONTEXT.md` exists for this phase yet (dir is empty). These constraints are effectively locked by the milestone docs and standing gates and the planner MUST honor them verbatim.

### Locked Decisions
- **Source of truth is `s_linker20_union`** per-run phase caches — **not s19**, **not** `s_linker20.py`. [CITED: .planning/REQUIREMENTS.md L7-9]
- **In scope:** both backends (gpt + sonnet), all N runs present (run1–run3), all 5 projects. [CITED: ROADMAP Phase 50 success criterion 1]
- **Read-only over caches + a new extraction script + its output JSON.** Phase 50 must NOT modify `s_linker20_union.py` or any linker (GATE-01). [CITED: STATE.md Standing Gates]
- **No LLM, no network, deterministic, re-runnable.** [CITED: EXTRACT-01 / ROADMAP success criterion 4]
- **No benchmark-derived vocabulary** in the new extraction code (GATE-06) — the script must be a pure structural transform; do not hardcode component names, aliases, or project-specific strings. [CITED: STATE.md GATE-06]

### Claude's Discretion (for the planner)
- Output directory location and JSON file layout (recommended below: `<extract_root>/<backend>/<run>/<project>.json`).
- Whether the friendly backend tag is `gpt`/`sonnet` or the on-disk `openai`/`claude` (recommend storing both in `meta`).
- Exact serialization of `(s,c)` tuple keys (recommend: ordered list of records).

### Deferred Ideas (OUT OF SCOPE for Phase 50)
- No-Knowledge ablation generation (Phase 51 / NOKNOW).
- Metric core, RQ3/RQ4 computation, SUMMARY.md, bundle packaging (Phases 52–55).
- N=6 full runs — **not extractable** (empty, no phase_cache); not required by any v2.6.6 requirement.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXTRACT-01 | Script run inside `agent-linker` dumps every per-run phase_cache (`layer1`–`layer4` + `final`) into neutral stdlib-loadable JSON — both backends, all N runs, all 5 projects. | Verified import surface (below): `import llm_sad_sam.linkers.experimental.s_linker20_union` suffices to unpickle all layers. Verified complete 30-cell matrix and exact on-disk paths/backend subdirs. |
| EXTRACT-02 | Extract captures entity `candidates`/`validated`/`decisions` (incl. `p1`/`p2`), coref `coref_raw`/`coref_validated`/`coref_decisions`, knowledge (`model_knowledge`+`doc_knowledge`), and `final` links + per-link `source`/provenance. | Field-by-field schema tables below map every required field to its pickle location + type, all loaded & confirmed. Proposed neutral JSON schema covers each. |
| EXTRACT-03 | Final-link set re-derived from each extract == that run's own `*_links.csv` / `ablation_*.json`, per project × run × backend. | Verified set-equality on `(sentence, component_id, source)` for all 30 cells using `final.links`. Documented the dup-key landmine and the correct (list-based) re-derivation. CSV/JSON oracle formats captured. |
</phase_requirements>

## On-Disk Matrix (verified 2026-06-21)

**Backend subdir mapping** (from `SLinker20Union._backend_tag()` → `LLMBackend.value`): gpt → `openai`, sonnet → `claude`. [VERIFIED: directory listing + final.pkl `backend` field == "openai"/"claude"]

| Backend (friendly) | Cache root | Backend subdir | Runs present | Projects | Cells |
|--------------------|-----------|----------------|--------------|----------|-------|
| gpt | `results/v2.6.5_s20union/gpt/run{1,2,3}/phase_cache/s_linker20_union/openai/<proj>/` | `openai` | run1, run2, run3 | 5 | 15 |
| sonnet | `results/v2.6.5_s20union_sonnet/run{1,2,3}/phase_cache/s_linker20_union/claude/<proj>/` | `claude` | run1, run2, run3 | 5 | 15 |

- **Projects (exact dir names):** `bigbluebutton`, `jabref`, `mediastore`, `teammates`, `teastore`.
- **Per cell:** exactly 5 files — `layer1.pkl`, `layer2.pkl`, `layer3.pkl`, `layer4.pkl`, `final.pkl`.
- **Completeness:** 30/30 cells present with all 5 pkls. **No missing cells.** [VERIFIED: full `os.path.exists` sweep this session]
- **Oracle files (per cell):** `results/.../<run>/<proj>/s_linker20_union_<proj>_links.csv` and `results/.../<run>/<proj>/ablation_<TIMESTAMP>.json` (+ a `.done` marker). The ablation filename is timestamped → glob `ablation_*.json` (exactly one per cell).
- **N=6 full runs OUT OF SCOPE:** `results/v2.6.5/full_s_linker20_union_run{1..6}/` are **empty directories — no `phase_cache/`, no `*.pkl`**. Nothing to extract; not referenced by any requirement. [VERIFIED: `find ... -name '*.pkl'` returns nothing]
- **Note:** the gpt cache root has a `gpt/` level (`results/v2.6.5_s20union/gpt/...`) while the sonnet root does not (`results/v2.6.5_s20union_sonnet/run.../...`). The matrix walker must encode these two roots explicitly, not assume a symmetric template.

## Minimal Import / Dependency Surface

**Pickle protocol:** 4 (readable by any Python ≥3.4 stdlib `pickle`). [VERIFIED: opcode byte]

**To `pickle.load` the layers, these classes must be importable** (pickle stores fully-qualified paths):

| Class | Module | Appears in |
|-------|--------|-----------|
| `ModelKnowledge`, `DocumentKnowledge` | `llm_sad_sam.core.data_types_v2` | layer1 |
| `CandidateLink` | `llm_sad_sam.core.data_types_v2` | layer2, layer3 |
| `SadSamLink` | `llm_sad_sam.core.data_types_v2` | layer4, final |
| `AliasEntry` (frozen dataclass) | `llm_sad_sam.linkers.experimental.s_linker20_union` | layer1 `doc_knowledge.aliases` **values** |

**Simplest correct import line for the script:**
```python
import llm_sad_sam.linkers.experimental.s_linker20_union  # noqa: F401  (registers AliasEntry; transitively imports data_types_v2)
import pickle, json, csv, glob, os
```
- `EvidenceBundle` and `MentionType` are **NOT** pickled: `evidence_bundles` are stored via `dataclasses.asdict()` (plain dicts) and `mention_type` is stored as the enum **`.value` string**. So no enum/EvidenceBundle import is required. [VERIFIED]
- Importing `s_linker20_union` is **safe at module-import time** — it imports `LLMClient`/`LLMBackend` but only *instantiates* them inside `SLinker20Union.__init__`, which the extraction script never calls. No network/LLM occurs on import. [VERIFIED: linker imported cleanly this session, no side effects]
- The repo is importable from its root (`python3 -c "import llm_sad_sam"` resolves to `src/llm_sad_sam/__init__.py`). The script must run with the repo env active (editable install `pip install -e ".[dev,openai]"`, or `src/` on `PYTHONPATH`). No third-party packages are needed beyond the repo itself. [VERIFIED]

## Field-by-Field Schema (all VERIFIED by loading pickles this session)

Every top-level pickle is a **plain `dict`** (not a dataclass). Tuple keys are `(sentence_number: int, component_id: str)`, written `(s,c)` below. `component_id` is an ECORE-style id string, e.g. `_st2Y0HDrEeSqnN80MQ2uGw`.

### layer1.pkl — Knowledge layer
| Key | Python type | Notes |
|-----|-------------|-------|
| `model_knowledge` | `ModelKnowledge` dataclass | field `ambiguous_names: set[str]` (e.g. `{'Cache','Packaging','Facade','DB'}`) — **set → serialize as sorted list** |
| `doc_knowledge` | `DocumentKnowledge` dataclass | field `aliases: dict[str, AliasEntry]`; **the dataclass's type hint says `dict[str,str]` but the runtime values are `AliasEntry(component:str, scope:str)`** where `scope ∈ {"global","local"}`. Legacy fields `abbreviations`/`synonyms`/`partial_references` are always empty `{}`. |
| `elapsed_s` | float | phase-1 wall time |
| `n_sentences` | int | document sentence count |
| `n_components` | int | model component count |

### layer2.pkl — Entity extraction (raw, pre-validation)
| Key | Python type | Notes |
|-----|-------------|-------|
| `framing_c` | `dict[(s,c), CandidateLink]` | union of the two extraction passes (the candidate pool) |
| `framing_c_pass1` | `dict[(s,c), CandidateLink]` | pass-1 only |
| `framing_c_pass2` | `dict[(s,c), CandidateLink]` | pass-2 only |
| `elapsed_s` | float | |

`CandidateLink` fields (dataclass, `asdict`-able): `sentence_number:int, sentence_text:str, component_name:str, component_id:str, matched_text:str, source:str("entity"), mention_type:str("indirect" — default, not re-classified at this stage), alias_used:str|None`.

> **layer2 is not strictly required by EXTRACT-02** (which names candidates/validated/decisions — those live in layer3). It is the raw extraction provenance. Recommend extracting it anyway (cheap, useful for audit/RQ4 "entity-only" provenance), but the planner may treat it as optional.

### layer3.pkl — Entity validation (the entity ablation core)
| Key | Python type | Notes |
|-----|-------------|-------|
| `candidates` | `list[CandidateLink]` | `= list(framing_c.values())`; **unique `(s,c)` keys** (came from a dict) |
| `validated` | `list[CandidateLink]` | subset of candidates that passed two-pass; **authoritative entity contribution to final** |
| `decisions` | `dict[(s,c), dict]` | per-candidate verdict; keys: `approved:bool, p1:bool, p2:bool, path:str, stage:str` |
| `evidence_bundles` | `dict[(s,c), dict]` | plain dicts (already `asdict`); fields: `source, matched_span, mention_type, preceding_text, anchor_sentences:list[str], is_ambiguous:bool, extraction_rationale` |
| `elapsed_s` | float | |

- `decisions[*].approved == (p1 and p2)`. Observed `(p1,p2,approved)` combos across all 30 cells: `(T,T,T)=909`, `(F,F,F)=180`, `(F,T,F)=47`, `(T,F,F)=26`. [VERIFIED]
- `decisions[*].path ∈ {"entity_twopass" (approved), "entity_twopass_reject"}`. `stage == "entity_twopass"` always. [VERIFIED — only these two paths exist]
- **Entity decisions are faithful 1:1 with candidates** (no dup keys), so `validated`-keyset == `{k for k,v in decisions if v.approved}` for all 30 cells. [VERIFIED]

### layer4.pkl — Coreference (the coref/citation ablation core)
| Key | Python type | Notes |
|-----|-------------|-------|
| `coref_raw` | `list[SadSamLink]` | discovered coref links (source="coreference"); **CAN contain duplicate `(s,c)` keys** (see landmine) |
| `coref_validated` | `list[SadSamLink]` | subset passing single-pass coref validation; **authoritative coref contribution to final** |
| `coref_metadata` | `dict[(s,c), dict]` | fields: `reference:str, antecedent_sentence:int, antecedent_text:str, antecedent_via_alias:bool, raw_resolution:dict` (full LLM JSON — bulky, audit-only) |
| `coref_decisions` | `dict[(s,c), dict]` | fields: `approved:bool, path:str`; **dict — collapses dup `(s,c)` keys** |
| `elapsed_s` | float | |

- `coref_decisions[*].path ∈ {"coref_validated" (approved), "coref_rejected"}`. [VERIFIED — only these two in practice]
- Two *defensive* paths exist in code but never fired in this matrix: `"coref_no_sentence_keep"` with `invariant_violation:True` (a coref link to a missing sentence — kept for recall). The extractor should serialize whatever keys are present (don't hardcode the schema) so these are captured if ever present. [VERIFIED: absent across all 30 cells; CITED: s_linker20_union.py `_validate_coref_links`]

### final.pkl — Merged result + provenance
| Key | Python type | Notes |
|-----|-------------|-------|
| `final` | `list[SadSamLink]` | **the authoritative final link set** (dedup-by-`(s,c)`, entity-priority) |
| `final_provenance` | `dict[(s,c), dict]` | fields: `from_coref:bool, source:str, entity_decision:dict|None, coref_decision:dict|None, coref_meta:dict|None` |
| `phase_metrics` | `dict[str, dict]` | per-phase `{calls,elapsed_s,tokens,errors}` + `_total{elapsed_s,llm_calls}`; LLM-cost audit only |
| `backend` | str | `"openai"` or `"claude"` (the backend tag, **not** the model name) |
| `elapsed_s` | float | total wall time |

`SadSamLink` fields: `sentence_number:int, component_id:str, component_name:str, confidence:float(always 1.0), source:str("entity"|"coreference")`.

## Final-Link Derivation Logic

From `SLinker20Union.link()` Phase 6 (lines 571–599), confirmed against pickles:

```
entity_links = [SadSamLink(s, c_id, c_name, source="entity") for each in layer3.validated]
all_links    = entity_links + layer4.coref_validated          # entity FIRST
final = dedup(all_links) keeping the FIRST occurrence per (sentence_number, component_id)
```

So on a `(s,c)` collision between an entity and a coref link, **entity wins the `source` tag**. The `final_provenance[(s,c)]` records `from_coref`, the surviving `source`, and (where present) the entity/coref decision + coref metadata for that key.

**Authoritative re-derivation for a stdlib consumer (no linker classes):**
```
final_set = {}                      # ordered: entity first
for rec in extract.entity.validated:   final_set.setdefault((rec.s, rec.c), "entity")
for rec in extract.coref.validated:    final_set.setdefault((rec.s, rec.c), "coreference")
# final_set now equals extract.final.links as a (s, c, source) set
```

> **DO NOT** re-derive the Full final set from the `decisions`/`coref_decisions` dicts — see landmine #1. The validated **lists** (or `final.links` directly) are authoritative. The decision dicts are for *ablation* (toggling validators in RQ3) and *audit*, not for reconstructing the Full set.

## Faithfulness Oracle (EXTRACT-03)

### `s_linker20_union_<proj>_links.csv`
Header: `sentence,component_id,component_name,confidence,source`
- `sentence`: int; `component_id`: ECORE id str; `component_name`: str; `confidence`: `"1.00"` (literal, always 1.0); `source ∈ {"entity","coreference"}`.
- One row per final link, in `final` list order. [VERIFIED on mediastore + 30-cell set sweep]

### `ablation_<TIMESTAMP>.json`
Shape: `{ "<project>": { "s_linker20_union": { ... } } }` with keys:
`variant, P, R, F1, tp, fp, fn, n_links, time, sources{entity:int, coreference:int}, fp_by_source{}, fp_details[], fn_details[{sentence,component,name_in_text,transarc_had}]`.
- `n_links == len(final)`; `sources` is the per-`source` count of final links. Useful **secondary** cross-checks: `n_links` == extract final count, and `sources.entity`/`sources.coreference` == per-source counts in the extract. (P/R/F1/tp/fp/fn are gold-standard metrics — **not** needed for EXTRACT-03 faithfulness, which is set-equality against the model's own output, not the gold.)

### Exact set-equality definition for EXTRACT-03 (recommended)
Per cell, assert:
```
{ (l.sentence, l.component_id, l.source) for l in extract.final.links }
  ==
{ (int(row.sentence), row.component_id, row.source) for row in csv }
```
This **passed for all 30 cells** this session. The weaker `(sentence, component_id)`-only equality also passed everywhere; including `source` is a strictly stronger, free check. Optionally also assert `len(extract.final.links) == ablation.n_links` and per-source counts match.

## Proposed Neutral JSON Schema

**Layout:** one file per cell → `<extract_root>/<backend>/<run>/<project>.json`, backend ∈ {`gpt`,`sonnet`}. (`<extract_root>` is the planner's choice; e.g. `results/v2.6.6_extracts/` in-repo, later vendored into `../working/` by Phase 52. Keep it OUT of any linker dir to respect GATE-01.)

**Serialization rules (lossless, JSON-native only):**
- `(s,c)` tuple-keyed dict → **ordered list of records**, each record carrying `"s":int, "c":str` plus the value fields. (Avoids tuple-as-key; preserves order; robust even if an id contained a delimiter.)
- `set[str]` → sorted `list[str]`.
- `AliasEntry` → `{"component":..., "scope":...}`.
- dataclasses (`SadSamLink`, `CandidateLink`) → `asdict`, but rename `sentence_number`→`s`, `component_id`→`c` is optional; keeping full names is fine. Be consistent.
- `coref.raw` and `coref.validated` → **lists** (NEVER collapse to a keyed map — preserves duplicate `(s,c)` occurrences).
- `final.links` → list in original `final` order (authoritative set).
- No floats are non-finite (confidence always 1.0; `elapsed_s` finite) — no NaN/inf handling needed, but the script should still `json.dump(..., allow_nan=False)` to fail loudly if that ever changes.

**Example (abridged, real mediastore/gpt/run1 values):**
```json
{
  "meta": {
    "backend": "gpt", "backend_subdir": "openai", "backend_tag": "openai",
    "run": "run1", "project": "mediastore", "variant": "s_linker20_union",
    "n_sentences": 37, "n_components": 12,
    "elapsed_s": {"layer1": 6.6, "layer2": 9.1, "layer3": 12.4, "layer4": 8.2, "final": 38.7}
  },
  "knowledge": {
    "model_knowledge": {"ambiguous_names": ["Cache", "DB", "Facade", "Packaging"]},
    "doc_knowledge": {
      "aliases": [
        {"term": "ReEncoder", "component": "Reencoding", "scope": "global"},
        {"term": "Database", "component": "DB", "scope": "global"}
      ]
    }
  },
  "entity": {
    "candidates": [
      {"s": 1, "c": "_st2Y0HDrEeSqnN80MQ2uGw", "component_name": "Facade",
       "matched_text": "Facade component", "source": "entity",
       "mention_type": "indirect", "alias_used": null,
       "sentence_text": "One of the main components ... the Facade component ..."}
    ],
    "validated": [
      {"s": 1, "c": "_st2Y0HDrEeSqnN80MQ2uGw", "component_name": "Facade", "source": "entity"}
    ],
    "decisions": [
      {"s": 1, "c": "_st2Y0HDrEeSqnN80MQ2uGw", "approved": true, "p1": true, "p2": true,
       "path": "entity_twopass", "stage": "entity_twopass"}
    ],
    "evidence_bundles": [
      {"s": 1, "c": "_st2Y0HDrEeSqnN80MQ2uGw", "source": "entity",
       "matched_span": "Facade component", "mention_type": "proper case, standalone",
       "preceding_text": "", "anchor_sentences": ["S3: ...", "S6: ..."],
       "is_ambiguous": true, "extraction_rationale": "Framing C extraction"}
    ]
  },
  "coref": {
    "raw": [
      {"s": 9, "c": "_p_EeYHDrEeSqnN80MQ2uGw", "component_name": "MediaManagement", "source": "coreference"}
    ],
    "validated": [
      {"s": 9, "c": "_p_EeYHDrEeSqnN80MQ2uGw", "component_name": "MediaManagement", "source": "coreference"}
    ],
    "decisions": [
      {"s": 9, "c": "_p_EeYHDrEeSqnN80MQ2uGw", "approved": true, "path": "coref_validated"}
    ],
    "metadata": [
      {"s": 9, "c": "_p_EeYHDrEeSqnN80MQ2uGw", "reference": "it",
       "antecedent_sentence": 8,
       "antecedent_text": "The MediaManagement component coordinates the communication ...",
       "antecedent_via_alias": false}
    ]
  },
  "final": {
    "links": [
      {"s": 1, "c": "_st2Y0HDrEeSqnN80MQ2uGw", "component_name": "Facade",
       "confidence": 1.0, "source": "entity"}
    ],
    "provenance": [
      {"s": 1, "c": "_st2Y0HDrEeSqnN80MQ2uGw", "from_coref": false, "source": "entity",
       "entity_decision": {"approved": true, "p1": true, "p2": true,
                           "path": "entity_twopass", "stage": "entity_twopass"},
       "coref_decision": null, "coref_meta": null}
    ]
  },
  "audit": {
    "phase_metrics": {"_total": {"elapsed_s": 38.7, "llm_calls": 17}}
  }
}
```

Notes:
- `raw_resolution` (the full LLM JSON inside `coref_metadata`) is intentionally dropped from the example for brevity; the planner may include it under `coref.metadata[*].raw_resolution` for completeness, or omit it to keep files lean. The four fields shown are sufficient for OUTPUT-01's per-link audit.
- `entity.validated` is shown as keyed-with-name records; storing full `CandidateLink` detail there instead is equally fine (self-contained). Whatever is chosen, keep it consistent across cells.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reading pickles | A custom unpickler / `find_class` shim | stdlib `pickle.load` after `import s_linker20_union` | Classes are importable; protocol 4 is fully supported |
| `(s,c)` key serialization | A bespoke `"s|c"` string-join scheme parsed downstream | ordered list of `{"s","c",...}` records | Avoids delimiter ambiguity, preserves order, trivially JSON-native |
| Final-set reconstruction | Re-running merge logic from `decisions` | Read `final.links` (authoritative) and/or `validated` lists | Decision dicts are lossy on dup coref keys (landmine #1) |
| Set-equality oracle | Re-implementing F1/gold matching | Compare against the run's own `*_links.csv` set | EXTRACT-03 is model-output faithfulness, not gold metrics |

**Key insight:** Phase 50 has zero algorithmic content — the only "logic" is faithful structure-preserving serialization. Every bug risk is a *serialization* bug (collapsing a list into a dict, dropping a set element, mangling a tuple key), which is exactly what the per-cell faithfulness assertion catches.

## Common Pitfalls

### Pitfall 1 (LANDMINE): Re-deriving final from decision dicts
**What goes wrong:** Reconstructing the Full final set from `coref_decisions` approved flags drops one link in `gpt/run2/teammates` and `gpt/run3/teammates`.
**Why:** `coref_raw` has duplicate `(s,c)` keys in 8/30 cells; `coref_decisions` is a dict (last-write-wins, collapses dups) while `coref_validated` is a list (preserves first-approved). Concrete case: `(96, _3LCnIK...)` in teammates appears twice — first approved (kept in `coref_validated`), second rejected (overwrites the dict to `approved:False`).
**How to avoid:** Treat `final.links` and the validated **lists** as authoritative. Serialize `coref.raw`/`coref.validated` as lists, never as keyed maps.
**Warning signs:** EXTRACT-03 faithfulness FAIL by exactly 1 link on a teammates/bigbluebutton cell.

### Pitfall 2: Asymmetric cache roots
**What goes wrong:** A single path template misses sonnet (or invents a `gpt/` level for sonnet).
**Why:** gpt root = `results/v2.6.5_s20union/gpt/run.../phase_cache/.../openai/...`; sonnet root = `results/v2.6.5_s20union_sonnet/run.../phase_cache/.../claude/...` — different depth and different leaf subdir.
**How to avoid:** Encode the two `(root, backend_subdir, friendly_tag)` triples explicitly.

### Pitfall 3: Expecting `anthropic/` for sonnet
**What goes wrong:** Looking for `anthropic/` finds nothing.
**Why:** The backend tag is `LLMBackend.CLAUDE.value == "claude"`.
**How to avoid:** Sonnet subdir is `claude/`. [VERIFIED]

### Pitfall 4: Treating N=6 full runs as extractable
**What goes wrong:** Script errors or emits empty cells for `full_s_linker20_union_run{1..6}`.
**Why:** Those dirs are empty — no `phase_cache`.
**How to avoid:** They are out of scope; do not include them in the matrix.

### Pitfall 5: `AliasEntry` import omitted
**What goes wrong:** `layer1` unpickle raises `AttributeError: Can't get attribute 'AliasEntry'`.
**Why:** `doc_knowledge.aliases` values are `AliasEntry` instances defined in `s_linker20_union`.
**How to avoid:** `import llm_sad_sam.linkers.experimental.s_linker20_union` before any `pickle.load`.

### Pitfall 6: `DocumentKnowledge` type-hint vs runtime mismatch
**What goes wrong:** Assuming `aliases` maps `str→str` (per the dataclass hint) and serializing the value as a bare string loses `scope`.
**Why:** The linker writes `AliasEntry(component, scope)` into the `dict[str,str]`-hinted field at runtime.
**How to avoid:** Serialize each alias value as `{"component","scope"}`.

## Package Legitimacy Audit

**No external packages are installed by this phase.** The extraction script uses only Python stdlib (`pickle`, `json`, `csv`, `glob`/`pathlib`, `os`) plus the repo's own already-installed importable modules (`llm_sad_sam.*`). No registry install, no `npm`/`pip` add, no postinstall surface. Slopcheck/registry verification is **N/A** — nothing new is introduced. GATE-06 (no benchmark-derived vocabulary) is satisfied by keeping the script a pure structural transform with no hardcoded component/alias/project strings (project names come from directory enumeration, which is structural, not vocabulary).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 + stdlib `pickle`/`json`/`csv` | All extraction | ✓ | proto-4 readable | — |
| `llm_sad_sam` repo importable (editable install or `src/` on path) | Unpickling (`AliasEntry`, `*Knowledge`, `*Link`) | ✓ | `src/llm_sad_sam/__init__.py` resolves | — |
| Frozen pkl caches (30 cells) | EXTRACT-01/02 | ✓ | 30/30 present | — |
| Oracle `*_links.csv` + `ablation_*.json` (30 cells) | EXTRACT-03 | ✓ | present in each cell dir | `ablation_*.json` only as secondary cross-check |

**Missing dependencies:** none. **No network / no LLM** required (and forbidden by EXTRACT-01).

## Validation Architecture

> `workflow.nyquist_validation` is `false` in `.planning/config.json`, so formal Nyquist sampling is not mandated. Including natural test/verification points per the phase objective, since this phase has an exact, machine-checkable oracle.

### Natural verification points (the phase is self-verifying)
| Check | What it asserts | Command shape | Hooks requirement |
|-------|-----------------|---------------|-------------------|
| Matrix completeness | All 30 cells have 5 pkls + a CSV | enumerate matrix, `os.path.exists` | EXTRACT-01 criterion 1 |
| Per-cell faithfulness | `set(extract.final.links on (s,c,source)) == set(csv)` | the script's own printed PASS/FAIL line per cell; exit non-zero if any FAIL | EXTRACT-03 |
| Count cross-check | `len(final) == ablation.n_links`; per-source counts match `ablation.sources` | inside the same per-cell check | EXTRACT-03 (secondary) |
| JSON-loadable without linker | `python3 -c "import json; json.load(open(f))"` in a **clean** interpreter (no `llm_sad_sam` import) succeeds for every emitted file | post-extraction smoke | EXTRACT-01 "neutral, stdlib-loadable" |
| Determinism | Re-running the script yields byte-identical JSON (sort keys, fixed list order) | `diff` two runs | ROADMAP criterion 4 / PARITY gate |
| Lossless dup preservation | `len(coref.raw) == len(layer4.coref_raw)` and `len(coref.validated) == len(layer4.coref_validated)` (no dict-collapse) | inside extractor | Landmine #1 guard |

**Recommended test fixture:** `mediastore/gpt/run1` (smallest, fully traced above) for a golden snapshot, plus `teammates/gpt/run2` as the **dup-key regression fixture** (this is where decision-dict re-derivation diverges — a faithful extractor must still PASS here).

### Wave 0 gaps
- No existing test harness for the extraction script (it doesn't exist yet). The script's built-in per-cell PASS/FAIL printout IS the primary verification artifact; a thin `pytest` wrapper around it on the two fixtures above is optional but cheap.

## Architecture Patterns

### Recommended script structure
```
scripts/extract_s20union_caches.py   # new file; or a one-off under tools/ — planner's call
  ├─ MATRIX = [(root, backend_subdir, friendly) for gpt & sonnet]
  ├─ load_cell(cache_dir) -> {layer1..4, final}      # pickle.load x5
  ├─ to_neutral(cell, meta) -> dict                  # the serialization rules
  ├─ faithfulness(neutral, csv_path) -> (ok, detail) # set-equality oracle
  └─ main(): walk matrix → write JSON → print PASS/FAIL → exit(nonzero if any FAIL)
```

### Pattern: tuple-key dict → record list
```python
def keyed_to_records(d):
    # d: dict[(s,c), value-dict] ; preserve insertion order (Py3.7+ dicts ordered)
    return [{"s": s, "c": c, **v} for (s, c), v in d.items()]
```

### Anti-patterns to avoid
- **Collapsing coref lists into dicts** (loses dup `(s,c)` → breaks faithfulness).
- **Re-implementing the merge** instead of reading `final.links`.
- **Writing the script anywhere it could be mistaken for a linker change** — keep it outside `src/llm_sad_sam/linkers/` to keep GATE-01 obviously satisfied.

## State of the Art

Not applicable — this is an internal, project-specific extraction task over a frozen artifact. No external libraries, frameworks, or evolving APIs are involved. The "current approach" is fixed by the on-disk pickle format produced by `s_linker20_union._save_phase`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The friendly model behind `openai` is `gpt-5.4` and behind `claude` is sonnet. The phase_cache stores only the backend tag (`openai`/`claude`), not the model string; model identity is inferred from the directory convention (`.../gpt/...` vs `..._sonnet/...`) and `.env`/llm_logs. | On-Disk Matrix / final.backend | Low — only affects a `meta.model` label, not the extracted data or faithfulness. If exact model string is required, read it from a `*_calls.json` in the cell's `llm_logs/` (the trace records `model`). |
| A2 | `<extract_root>` location is the planner's choice; recommended `results/v2.6.6_extracts/` in-repo, vendored to `../working/` in Phase 52. | Proposed JSON schema | Low — purely organizational; no requirement pins the path. |
| A3 | layer2 (`framing_c*`) is optional for EXTRACT-02 (which names only candidates/validated/decisions). Recommend including it for audit completeness. | layer2 schema | Low — including extra data cannot violate a requirement; excluding it is also defensible. |

## Open Questions

1. **Extract output directory + whether Phase 50 writes directly into `../working/`'s vendored-extracts dir or a staging dir under `results/`.**
   - What we know: BUNDLE-01 (Phase 55) requires `../working/` to vendor the neutral extracts; Phase 52 stands up `../working/`.
   - What's unclear: whether Phase 50 emits straight into the (not-yet-existing) `../working/` tree or into a repo-local staging dir that Phase 52 copies from.
   - Recommendation: emit to a repo-local `results/v2.6.6_extracts/<backend>/<run>/<project>.json`; let Phase 52 vendor it. Keeps Phase 50 self-contained and GATE-01-clean.

2. **Should `coref.metadata[*].raw_resolution` (the full LLM JSON) be retained?**
   - What we know: OUTPUT-01's per-link audit needs `reference`, antecedent fields, `p1/p2/approved` — all available without `raw_resolution`.
   - What's unclear: whether downstream wants the raw LLM blob.
   - Recommendation: omit `raw_resolution` by default (keeps files lean); add a `--with-raw` flag if ever needed.

3. **Model-name label in `meta`** (see Assumption A1) — confirm whether RQ summaries need the exact model string vs. the friendly `gpt`/`sonnet` tag. If exact, source it from `llm_logs/*_calls.json` `model` field.

## Sources

### Primary (HIGH confidence — verified by direct pickle loads this session)
- `results/v2.6.5_s20union/gpt/run{1,2,3}/...` and `results/v2.6.5_s20union_sonnet/run{1,2,3}/...` — loaded all 5 layers across all 30 cells; ran full faithfulness + matrix-completeness + dup-key sweeps.
- `src/llm_sad_sam/linkers/experimental/s_linker20_union.py` — `link()` Phase 1–6, `_save_phase`, `_backend_tag`, `_validate_with_evidence`, `_validate_coref_links`, `AliasEntry`/`EvidenceBundle`/`MentionType` definitions.
- `src/llm_sad_sam/core/data_types_v2.py` — `SadSamLink`, `CandidateLink`, `ModelKnowledge`, `DocumentKnowledge` field definitions.
- `.planning/ROADMAP.md` (Phase 50 + data-reality block), `.planning/REQUIREMENTS.md` (EXTRACT-01/02/03, gates, scope), `.planning/STATE.md` (milestone context, standing gates), `CLAUDE.md` (retained surface).

### Secondary / Tertiary
- None — no external/web sources were needed; the domain is a frozen in-repo artifact.

## Metadata

**Confidence breakdown:**
- Schema (all 5 layers): HIGH — loaded and field-inspected every layer; cross-checked across both backends.
- Matrix completeness: HIGH — exhaustive `os.path.exists` over 30 cells.
- Faithfulness oracle + set-equality: HIGH — verified `final == csv` for all 30 cells; isolated the dup-key divergence to the decision-dict path.
- Import surface: HIGH — confirmed clean import + identified `AliasEntry` as the only linker-module-local pickled class.
- Output JSON schema: MEDIUM-HIGH — design is sound and lossless; exact field naming/location is the planner's to finalize.

**Research date:** 2026-06-21
**Valid until:** Stable indefinitely — inputs are frozen artifacts. Re-verify only if the s20_union caches are regenerated or the matrix is extended (e.g., more runs/backends added).
