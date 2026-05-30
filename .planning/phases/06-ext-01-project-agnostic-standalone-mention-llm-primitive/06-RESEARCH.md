# Phase 6: EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive — Research

**Researched:** 2026-05-30
**Domain:** LLM-driven traceability linker — replacement of a regex word-boundary primitive with a project-agnostic LLM primitive (cite-evidence pattern, no benchmark-specific structure)
**Confidence:** HIGH (every claim either verified against repo files or pulled from CONTEXT.md)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01 — Do not pre-lock the semantic scope.** Run an empirical study comparing three candidate primitives:
- **Literal**: LLM mirrors regex semantics — "is `<comp_name>` a surface token here, not embedded in another identifier?"
- **Semantic**: LLM judges "does this sentence reference the architectural component, not just contain the word?"
- **Hybrid**: single call emits `{surface_mention: bool, architectural_ref: bool}`; caller chooses signal per call site.

**D-02 — Study method = offline anchor-collection diff → finalist sweep.** Replay `s_linker13`'s anchor-collection logic with each primitive. Compute per-(component, dataset) diff vs the regex baseline. Drop catastrophic variants. Surviving 1–2 variants get the full 5-project sweep. **No full-pipeline run during diff stage.**

**D-03 — Winner decided by macro F1 only.** Highest macro F1 from the finalist sweep wins. Ties broken by GATE-06 cleanliness (fewer/cleaner prompt examples, smaller call-graph footprint). Cost is NOT part of the winner-decision rule — captured separately as Phase 8 input (see D-06).

**D-04 — Phase 6 preserves dotted-path skip behavior** — EXT-02 (Phase 7) removes the guard. Phase 6 ships two sub-variants:
- **(a) Regex pre-filter + LLM judge** — keep dotted/hyphen regex guards as cheap pre-filter; LLM judges only sentences that survive.
- **(b) LLM-only with dotted-path encoded in prompt semantics** — no regex; prompt teaches "token embedded in compound dotted identifier is not a standalone mention" using safe-domain examples.
- Both compete on same diff → finalist sweep used for D-02. Semantic-scope (D-01) and dotted-path (D-04) evaluated as ONE matrix (3 × 2 = up to 6 cells; collapse equivalents).

**D-05 — Naming & promotion mirrors v1.0 13f→s_linker13 pattern.** Build candidates as siblings (e.g. `s_linker13g_pre.py`, `s_linker13g_sem.py`); byte-copy winner to canonical `s_linker13g.py`. Loser stays as rejected artifact for ablation table. Canonical file gets structured docstring (`REMOVED_FROM` / `RULES_REMOVED`) + registration in `CANONICAL_VARIANTS` + `VARIANT_SPECS` per GATE-07.

**D-06 — Capture EXT-01 cost/quality signal** as a structured tagged block in `06-SUMMARY.md`. Minimum content:
- LLM call-count delta vs `s_linker13` (per dataset, totals)
- Wall-clock latency delta vs `s_linker13` (per dataset)
- Per-dataset and macro F1 delta vs `s_linker13`
- Notes on stackable-vs-unify topology
- Tagged section header `## EXT-01 cost/quality signal (Phase 8 input)` so Phase 8 can grep it.

### Claude's Discretion (planner decides, constrained by D-02/D-06)

- **API shape & call topology** of the new LLM primitive (per-(comp, sent) call vs per-component batch vs Spike-003 piggyback vs document-level enrichment map). Must serve D-02 diff-ability and D-06 cost signal.
- **Fallback policy** on LLM failure (default: approve-bias per existing `s_linker13._run_seed_validation` pattern unless reason to deviate).
- **Anchor-section vs `has_exact_case`-flag split** — current primitive serves both; new primitive may unify or split.
- **Prompt-example domains** — must come from BENCHMARK_TABOO.md §"Safe SE Textbook Examples".

### Deferred Ideas (OUT OF SCOPE)

- **EXT-02** (drop dotted-path guard) — Phase 7, gated on Phase 6 pass.
- **Stack-vs-unify decision** — Phase 8, fed by D-06 signal.
- **GPT-5.2 cross-model evaluation** — Phase 9 (CROSS-01..03).
- **EXT-04** (emit-biased boundary on alias-discovery, BBB variance band) — v2.1+.
- **Cost optimization** beyond D-06 minimum metric set — user explicitly set "no LLM budget limit"; replaceability + generality trump cost.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **EXT-01** | `_has_standalone_mention` replaced by a project-agnostic LLM primitive in a new linker variant (`s_linker13g.py` or similar). Relaxed cost budget; replacement must not encode project-specific structure. Dual floor met (GATE-01). | Sections "Standard Stack", "Architecture Patterns", "Replacement Site Inventory", "Common Pitfalls" (Pitfall 1: 13d TM regression is the controlling negative result), "Don't Hand-Roll", "Empirical Matrix Operationalization" |
| **GATE-01** (standing) | Macro F1 ≥ 0.93 AND BBB ≤ 6pp below `s_linker12c` BBB (0.8073) AND each other dataset ≤ 2pp below `s_linker12c` per-dataset baseline. | Section "Dual-Floor Thresholds (numeric)" — verified per-dataset numbers from `ablation_20260404_000505.json`. |
| **GATE-05** (standing) | Hard-tier-first dev loop: regress >1pp on TM or BBB vs parent → no full sweep, re-work. | Section "Architecture Patterns → Dev Loop". |
| **GATE-06** (standing, NEW v2.0) | Generality audit per phase in SUMMARY.md: (a) BENCHMARK_TABOO.md scan AND (b) reviewer-defensibility check. Zero hardcoded project-tailored values in prompts OR logic. | Section "Don't Hand-Roll" + "Common Pitfalls (Pitfall 4: prompt-example leakage)" + "Prompt Domain Selection". |
| **GATE-07** (standing) | Promoted variant registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS` of `run_ablation.py`; standalone file (no inheritance); structured docstring with `REMOVED_FROM` / `RULES_REMOVED`. | Section "Replacement Site Inventory" + "Architecture Patterns → Standalone Variant File". |

</phase_requirements>

---

## Summary

EXT-01 replaces a 28-line regex primitive (`_has_standalone_mention`, s_linker13.py:1120-1147) that has six call sites in `s_linker13`. The primitive answers "is this sentence a confirmed standalone reference to component `<comp_name>`?" and is used to (a) build anchor sections in two LLM prompts (`_run_seed_validation`, `_validate_with_evidence` generic-filter, `_build_evidence_bundle`), (b) drive the `_classify_mention` mention-type string emitted into evidence bundles, (c) gate a `has_exact_case` flag in the generic-filter pre-pass, and (d) verify coref antecedents.

The phase ships **two sibling candidate variants** that compete via an **offline anchor-collection diff stage** before any full sweep. The competing axes are (D-01) semantic scope of the LLM judgement and (D-04) how the dotted-path skip is encoded — a 3×2 matrix collapsed to up to 4 distinct cells. Survivors of the diff stage get the full 5-project sweep; macro F1 picks the winner; the loser becomes a rejected-ablation artifact (`s_linker13g_<loser>.py`).

The dominant risk is the **13d TM regression** (v1.0 negative result, retired VAR-04, -19pp on TeaMMates from 33 dotted-path FPs when LLM-emitted mention enum replaced the regex). EXT-01's "project-agnostic" mandate (GATE-06) **forbids** the v1.0 escape hatch (keeping the dotted-path regex elsewhere); D-04 sub-variant (a) keeps it as an explicit pre-filter the prompt does not see, sub-variant (b) tries to teach the LLM the dotted-path concept via safe-SE examples. The diff-stage's purpose is to catch a (b)-style 13d-class catastrophe before paying for a full sweep.

**Primary recommendation:** Use the **Spike-003-piggyback API shape** (extend `_extract_entities_enriched`'s existing prompt with a `standalone_mention` boolean per emitted reference — zero net LLM cost) as the **default primary topology for the LLM judge**, but compute the **document-level standalone-mention map** as a **separate, dedicated prompt** for the **anchor-collection** call sites. Anchor collection currently runs per-component over all sentences (3 of 6 call sites); piggybacking it onto entity extraction collapses 3 call sites to zero new calls. The dedicated anchor-map call is needed because anchor collection scans sentences NOT in the seed set (i.e., sentences the extractor may not have surfaced candidates for). One per-component batched call (50 sents/batch like extraction) for anchor population + piggybacked per-candidate `standalone_mention` field for the other 3 call sites is the lowest-cost call topology that remains diff-able against the regex baseline (D-02) and produces a clean D-06 signal.

---

## Architectural Responsibility Map

The phase's "capabilities" are the 6 distinct callers of `_has_standalone_mention`. Tier ownership in this codebase is the linker variant file (single Python class, no inheritance); below maps each caller to its **role in the pipeline**, not to a different process tier.

| Capability | Pipeline Tier | Owner Method (s_linker13.py) | Rationale |
|------------|---------------|------------------------------|-----------|
| Anchor section for seed disambiguation prompt | Tier 2 — Link Recovery (seed_val task) | `_run_seed_validation` (lines 480–592, anchor loop at 505–525) | Builds "KNOWN REFERENCES" block; LLM uses anchors to calibrate per-component disambiguation. Per-component batched scan over all non-seed sentences. |
| Anchor section for generic-mention filter prompt | Tier 2 — Link Recovery (entity task → pre-pass) | `_validate_with_evidence` generic-filter (lines 871–956, anchor loop 893–898) | Same logic, different prompt: pre-pass for ambiguous-named components with only lowercase candidates. |
| Anchor list inside `EvidenceBundle.anchor_sentences` | Tier 2 — Link Recovery (entity task → bundle build) | `_build_evidence_bundle` (lines 651–694, anchor loop 671–678) | Up to 5 confirmed full-name mentions attached to every candidate's evidence trail. Consumed by 2-pass validation prompts. |
| Mention-type string ("proper case, standalone" / ...) | Tier 2 — Link Recovery (entity task → bundle build, also Tier 2 seed_val case formatting) | `_classify_mention` (lines 617–649) called from `_build_evidence_bundle` (line 666) AND `_run_seed_validation` (line 537) | Human-readable mention type → printed into LLM prompts as `Mention: <string>` line. Spike 003 already showed this is replaceable by an enum emitted by the extractor. |
| `has_exact_case` flag for generic-filter routing | Tier 2 — Link Recovery (entity task → pre-pass) | `_validate_with_evidence` (line 880) | Per-candidate boolean: if False AND lowercase match AND component is ambiguous → route to LLM generic filter. |
| Coref antecedent verification | Tier 2 — Link Recovery (coref task → post-LLM check) | `_run_coreference` → `_coref_cases_in_context` (line 1095) | After LLM emits a resolution, the antecedent sentence is verified to contain a standalone mention of the resolved component (OR antecedent_via_alias flag). |

**Tier ownership decision:** All 6 callers stay inside the same variant class (`SLinker13g`) — no extraction to a helper module, no inheritance, per the project's "one rule = one standalone variant file" convention (PROJECT.md Key Decisions, GATE-07).

---

## Standard Stack

### Core (already in the codebase — reuse, do NOT reinvent)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `LLMClient` (project-internal) | repo HEAD | `query(prompt, timeout)` → str, `extract_json(text)` → dict | Cite-evidence pattern: structured JSON + approve-biased fallback. Used everywhere in `s_linker13` (verified at lines 562, 933, 1017, 1073). [VERIFIED: repo grep] |
| `prompts_v2` module | repo HEAD (247 lines) | Holds prompt constants (`ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, `SEED_DISAMBIGUATION_RULES`, etc.) | Project convention: prompt strings live in `prompts_v2.py`, NOT in linker files. CONTEXT.md confirms new primitive's prompt belongs here. [VERIFIED: grep + CLAUDE.md] |
| `concurrent.futures.ThreadPoolExecutor` | stdlib | `_run_parallel` for DAG tier execution | `s_linker13._run_parallel` (lines 190–210) already used for Tier 1 (3-way parallel) and Tier 2 (3-way parallel). New LLM call (if separate from extraction) plugs in here. [VERIFIED: s_linker13.py:190-210] |
| `data_types_v2` | repo HEAD | `SadSamLink`, `CandidateLink`, `ModelKnowledge`, `DocumentKnowledge` | Don't touch — retained upstream per PROJECT.md "Changes to retained upstream components (`ilinker*`, `prompts_v2`, `data_types_v2`) unless required by a rule removal" — adding new optional fields to `CandidateLink` (e.g., `standalone_mention: bool | None = None`) IS allowed if a sub-variant needs it. [CITED: PROJECT.md line 53] |
| `run_ablation.py` registration | repo HEAD | `CANONICAL_VARIANTS` list + `VARIANT_SPECS` dict | GATE-07 enforcement point. New entries land at lines 40–80 (list) and lines 274–316 (dict) following the `s_linker13a..13f` precedent. [VERIFIED: run_ablation.py:40-80, 274-316] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ILinker3` (project-internal) | repo HEAD | Seed extraction (Tier 1) | Already constructed in `SLinker13.__init__` line 181. EXT-01 does NOT touch seeds — only Tier 2 validation paths use `_has_standalone_mention`. |
| `pcm_parser_v2.parse_pcm_repository` | repo HEAD | Component list from .repository file | Unchanged. |
| `document_loader_v2.load_sentences` / `build_sent_map` | repo HEAD | Sentence list + `{snum: Sentence}` map | Unchanged. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Approve-biased JSON fallback | Strict JSON validation + reject-on-fail | Would break recall on flaky LLM responses. Existing pattern at s_linker13.py:567-568 (`verified.extend(valid_seeds) # Keep all on failure`) and 939 (`remaining.extend(cands) # On failure, keep all`) is unanimous across `s_linker13`. CONTEXT.md D-Discretion: default to approve-bias unless reason to deviate. [VERIFIED] |
| Per-(comp, sent) call (one LLM call per pair) | Worst-case O(N×M) calls; for BBB (~250 sentences × ~12 components) = ~3000 calls per dataset. | Eliminated by Spike 002 RISKY classification (this is the exact reason `_has_standalone_mention` was KEPT in v1.0). User now has "no LLM budget" but a 3000-call topology would dominate D-06 signal and bias Phase 8 against EXT-01. [CITED: Spike 002 AUDIT.md line 12] |
| Document-level full-enrichment map (one call returns `{(comp, snum): bool}` for ALL pairs) | Single prompt of size O(N×M) tokens | Prompt-size blowup for BBB (~250 sents × full comp list); LLM context window pressure; cannot be diff-checked sentence-by-sentence against regex baseline as easily. |
| Per-component batched scan (one call per component, batch of N sentences) | Mirrors extraction batching (50 sents/batch) | Natural fit for **anchor collection** call sites (which scan all sentences for one component). For the OTHER 3 call sites (mention-type, has_exact_case, coref antecedent), a piggyback on entity extraction is cheaper. **Recommended hybrid topology.** |

### Installation

No new dependencies. All stack is already in `pyproject.toml` (`pip install -e ".[dev,openai]"` per CLAUDE.md).

---

## Architecture Patterns

### System Architecture (current `s_linker13` DAG, with EXT-01 replacement sites marked)

```
                    text_path, model_path
                            │
                            ▼
      ┌─────────────────────────────────────────┐
      │ Tier 1: Knowledge Acquisition (parallel)│
      │  ┌──────────┬──────────────┬──────────┐ │
      │  │ model    │ doc_knowledge│ seed     │ │
      │  │ analyze  │ enriched     │ (ILinker3)│ │
      │  └──────────┴──────────────┴──────────┘ │
      └─────────────────────────────────────────┘
                            │
                            ▼ (ModelKnowledge, DocumentKnowledge, raw_seed_links)
      ┌─────────────────────────────────────────┐
      │ Tier 2: Link Recovery (parallel)        │
      │                                         │
      │  seed_val ◄── [EXT-01 site #1]          │
      │   └─ anchor section per component       │
      │   └─ _classify_mention per seed         │
      │                                         │
      │  entity ─── _extract_entities_enriched  │
      │   │         (2 parallel passes)         │
      │   ├─ _build_evidence_bundle             │
      │   │   ├─ _classify_mention ◄── [#2]     │
      │   │   └─ anchor list (≤5) ◄── [#3]      │
      │   └─ _validate_with_evidence            │
      │       ├─ generic-filter pre-pass:       │
      │       │   has_exact_case ◄── [#4]       │
      │       │   anchor section ◄── [#5]       │
      │       └─ 2-pass intersection vote       │
      │                                         │
      │  coref ── _coref_cases_in_context       │
      │   └─ antecedent check ◄── [#6]          │
      └─────────────────────────────────────────┘
                            │
                            ▼
                  Tier 3: Output union
                  (seed_links ∪ validated ∪ coref_links)
```

Six call sites: `[#1]` anchor seed-val, `[#2]` mention-type in bundle, `[#3]` anchor in bundle, `[#4]` exact-case flag, `[#5]` anchor generic-filter, `[#6]` coref antecedent verify.

### Recommended Project Structure (Phase 6 additions)

```
src/llm_sad_sam/linkers/experimental/
├── s_linker13.py                # parent (unchanged)
├── s_linker13g_pre.py           # sub-variant (a) — regex pre-filter + LLM judge
├── s_linker13g_sem.py           # sub-variant (b) — LLM-only with dotted-path in prompt
└── s_linker13g.py               # canonical (byte-copy of winner, post-sweep)
└── prompts_v2.py                # new prompt constant(s) for the primitive

.planning/phases/06-*/
├── 06-CONTEXT.md                # (exists)
├── 06-RESEARCH.md               # this file
├── 06-PLAN.md(s)                # (planner output)
├── 06-DIFF-MATRIX.md            # NEW — D-02 diff-stage results, threshold derivation, drop decisions
├── 06-SUMMARY.md                # NEW — includes tagged "## EXT-01 cost/quality signal (Phase 8 input)" block (D-06)
└── 06-GATE-06-AUDIT.md          # NEW — BENCHMARK_TABOO scan + reviewer-defensibility checklist

results/ablation_results/
├── ablation_<ts>_diff.json      # anchor-set Jaccard / sym-diff per (variant, dataset, component)
└── ablation_<ts>_<variant>.json # full 5-project sweep, one per surviving variant
```

### Pattern 1: Cite-Evidence LLM Call (project standard, already used 5× in s_linker13)

**What:** Structured JSON output + retry-once + approve-biased fallback.
**When to use:** Every LLM call in this codebase.
**Example (from s_linker13.py:561-568, the canonical shape):**
```python
# Source: s_linker13.py:561-568 (verified)
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

### Pattern 2: Standalone Variant File (project standard, GATE-07)

**What:** Each rule removal lands as a copy-pasted full file (no inheritance from `SLinker13`).
**When to use:** Always — user explicitly prefers duplicated standalone files over inheritance chains (CLAUDE.md + PROJECT.md Key Decisions).
**Required structure:**
```python
"""S-Linker13g_<sub>: <one-line description>.

REMOVED_FROM: s_linker13 (cumulative via 13a->...->13f->13)
RULES_REMOVED: ["_split_component_name (13a)", ..., "_has_strong_alias_mention (13f)",
                "_has_standalone_mention (13g)"]
KEEP: []  # all structural rules now removed (EXT-02 still owns the dotted-path guard removal)
"""

class SLinker13gPre:  # or SLinker13gSem
    _VARIANT_NAME = "s_linker13g_pre"  # MUST appear in _checkpoint_dir (assertion at s_linker13.py:1165)
    ...
```

### Pattern 3: Per-Variant Checkpoint Namespacing (GATE D-07-derived)

**What:** `_checkpoint_dir` MUST embed `_VARIANT_NAME` so checkpoints from different variants do not cross-contaminate.
**Source:** s_linker13.py:1159-1170 — assertion at line 1165 fails fast if `_VARIANT_NAME` not in path.
**Implication for EXT-01:** Sub-variants `s_linker13g_pre` and `s_linker13g_sem` MUST have distinct `_VARIANT_NAME` values, AND the byte-copy promotion to `s_linker13g.py` MUST update `_VARIANT_NAME = "s_linker13g"`. Forgetting this is a silent-corruption bug.

### Pattern 4: DAG Tier Parallel Execution

**What:** `_run_parallel({"name": lambda: fn(), ...})` executes named tasks concurrently and re-raises on first failure (s_linker13.py:190-210).
**When to use:** Only at tier boundaries (Tier 1 has 3-way parallel: model/doc_knowledge/seed; Tier 2 has 3-way parallel: seed_val/entity/coref).
**Implication for EXT-01:** If the primitive runs as a **dedicated document-level pass** (recommended for anchor collection), it belongs in **Tier 1** (it computes a `(comp, snum) -> standalone_mention` map consumed by all Tier 2 tasks), so it parallelizes with model/doc_knowledge/seed. Add it as a 4th parallel task in the `_run_parallel` call at s_linker13.py:240-244. If it is purely piggybacked on entity extraction (Spike-003 shape), no new Tier-1 step needed.

### Dev Loop (GATE-05 enforcement)

1. Implement sub-variant (a) — `s_linker13g_pre.py`.
2. Run **hard-tier only** sweep (teammates + bigbluebutton) via `python run_ablation.py --variant s_linker13g_pre --datasets teammates,bigbluebutton`.
3. Check vs `s_linker13` parent: if BBB regress > 1pp OR TM regress > 1pp → **rework before full sweep** (GATE-05).
4. Hard-tier clean → full 5-project sweep.
5. Repeat for sub-variant (b).
6. Run D-02 diff stage (offline, no LLM) on the **anchor-collection step only** for both variants → drop catastrophic.
7. Survivors get full sweep (already done if hard-tier passed).
8. Pick winner by macro F1; ties → GATE-06 cleanliness.
9. Byte-copy winner to `s_linker13g.py`, register in `run_ablation.py`, run final sweep under canonical name.

### Anti-Patterns to Avoid

- **Inheritance from `SLinker13`:** Forbidden. Copy the full file. Sibling competition requires byte-comparable files.
- **Editing `s_linker13.py`:** Forbidden — it is the canonical v1.0 deliverable (macro F1 0.9509 baseline reference). Edits would invalidate the baseline.
- **Sharing a checkpoint cache between sub-variants:** Will silently corrupt because `_extract_entities_enriched` (and friends) cache by phase name keyed under `_VARIANT_NAME`. Distinct `_VARIANT_NAME` per sub-variant is mandatory (s_linker13.py:1165 assertion).
- **Putting prompt strings in the linker file:** Project convention puts them in `prompts_v2.py`. Inline prompts (like the disambiguation prompt at s_linker13.py:547-559, the generic filter at 914-930, the coref prompt at 1053-1070, the validation pass at 1003-1014) are EXCEPTIONS for case-specific assembly; CONSTANTS (rules, schemas, few-shot blocks) go in `prompts_v2`. Follow the existing split.
- **Reusing the 13d (Spike-003) extractor schema unchanged on TeaMMates:** This is the central pitfall — see "Common Pitfalls → Pitfall 1".

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON-from-LLM parsing | `json.loads(text)` + regex extraction | `LLMClient.extract_json(self.llm.query(prompt))` | Repo standard. Handles fenced code blocks, prose preamble, trailing text. Verified everywhere in s_linker13. |
| Retry on empty response | Bespoke retry loop | The `for attempt in range(2): ... if attempt == 0: print(...retrying...)` cite-evidence loop | Project pattern. Five canonical occurrences in s_linker13 (lines 561, 775, 932, 1016, 1072). |
| Approve-biased fallback | `raise` / discard / skip | `if not data: keep_all_on_failure(); continue` | Recall protection. Project-wide unanimous pattern. |
| Anchor-collection batching | Bespoke per-component batch loop with token-budgeting | Mirror `_run_single_extraction_pass` shape (s_linker13.py:748-803, batch_size=50) — already battle-tested for the 5 benchmark sizes | Avoids re-deriving batch boundaries; LLM-side prompt-size behavior is known to work. |
| DAG task orchestration | `asyncio` / bespoke threads | `_run_parallel({name: fn})` at s_linker13.py:190-210 | Project standard. Auto-cancels on first failure, single `with ThreadPoolExecutor`. |
| Variant registration | Bespoke `__init_subclass__` / decorator | Append to `CANONICAL_VARIANTS` list AND `VARIANT_SPECS` dict in `run_ablation.py` | GATE-07. Both edits are mandatory — `CANONICAL_VARIANTS` controls `--list-variants`; `VARIANT_SPECS` carries the module path + class name. |
| Mention-type formatter | New regex switch | Spike-003 `format_mention(enum, alias_used)` function (`.planning/spikes/003-llm-mention-classifier/spike.py:37-52`) | Already validated. 6 enum branches → string lookup. Zero regex. |
| Dotted-path detection (sub-variant a) | New regex | Reuse the **existing** `_has_standalone_mention` dotted/hyphen guards at s_linker13.py:1138-1145 (lines 4 of the function) — extract them into a small helper `_in_dotted_or_hyphen_context(text, span)`; the LLM judge then sees ONLY pre-filtered sentences | Don't fork the regex semantics — copy the bytes, isolate, document as "pre-filter only; EXT-02 will remove". |

**Key insight:** **EXT-01 is a prompt + plumbing problem, NOT an LLM-research problem.** Every infrastructure piece exists. The 80% of risk is in (a) the prompt domain choice (GATE-06) and (b) whether sub-variant (b) can teach the dotted-path concept without a 13d-style regression. The 20% remaining is wiring the new primitive into 6 call sites without introducing cross-variant checkpoint contamination.

---

## Empirical Matrix Operationalization

### The 3 × 2 Matrix and Cell Equivalences

| | (a) Regex pre-filter + LLM judge | (b) LLM-only, dotted-path in prompt |
|---|---|---|
| **Literal** (LLM mirrors regex semantics) | Cell A1 — minimal LLM responsibility; sub-variant (a) inherits dotted-path from pre-filter | Cell A2 — LLM does BOTH word-boundary and dotted-path; weakest cell on TeaMMates risk |
| **Semantic** (LLM judges architectural reference, not surface) | Cell B1 | Cell B2 |
| **Hybrid** (LLM emits both `surface_mention` and `architectural_ref`; caller picks per-site) | Cell C1 | Cell C2 |

**Suggested collapses (planner may revise based on diff results):**

- **A1 ≈ A2:** Literal scope means "mirror regex semantics." Sub-variant (a) already has the regex; the LLM in A1 only adds the word-boundary check, not the dotted-path check. In A2, the LLM does both. The diff vs regex is small in both cells but A2 is **strictly more 13d-risky**. Recommend **collapse: drop A2 from the matrix**, keep A1 as the conservative cell.
- **B1, C1, B2, C2 are distinct** and should all be probed in the diff stage. The Hybrid emits BOTH signals so its diff vs regex on the surface-mention axis should be close to regex; its architectural-reference signal is what differentiates it.

**Recommended cell set: A1, B1, B2, C1, C2 (5 cells).** Planner may further collapse C1/C2 if hybrid's surface-mention output is byte-identical to literal across the diff probe.

### The "Catastrophic Diff" Threshold (operationalize for D-02)

D-02 leaves this for research. Recommended operationalization, computed offline (no full LLM pipeline run):

1. **Run the candidate primitive's anchor-collection step only** on each of the 5 datasets, for each candidate cell. Output: per-(component, dataset) **anchor sentence set** = `S_variant[comp][ds]`.
2. **Run regex baseline anchor-collection** (i.e., `_has_standalone_mention` from `s_linker13` unchanged) on the same datasets: `S_regex[comp][ds]`.
3. **Per-(component, dataset) Jaccard:** `J = |S_variant ∩ S_regex| / |S_variant ∪ S_regex|`.
4. **Per-(component, dataset) symmetric-difference size:** `D = |S_variant △ S_regex|`.
5. **Roll-up metrics per (variant, dataset):**
   - `min_jaccard_per_comp` — worst component on the dataset
   - `mean_jaccard_weighted_by_|S_regex|` — overall similarity weighted by anchor count
   - `count_components_with_J < 0.5` — count of components where variant diverges hard
6. **Catastrophic-diff drop rule (recommended threshold):**
   - **Drop if:** `min_jaccard_per_comp < 0.3` on TM OR BBB (the hard tier), **OR** any (comp, ds) has `D > 10 sentences` (gross over-collection or under-collection), **OR** `count_components_with_J < 0.5` > 25% of components on TM or BBB.
   - **Rationale for these numbers:** anchor-collection caps at 5 sentences per component in `s_linker13` (lines 512–513, 677). A Jaccard < 0.3 means fewer than 1.5 of 5 anchors agree — the LLM is anchoring on different evidence than the regex, which is the 13d-class failure signature on a non-LLM-visible axis. D > 10 means total flux > 2× the cap. The percentile rule catches systematic mismatch even when individual mismatches are small.
7. **Tie-break / sanity check:** If two cells survive but one has 5× the LLM call cost of the other for similar Jaccard, the **D-06 cost signal** (NOT D-03 winner rule, which is macro F1 only) flags the high-cost one as "not stackable" in SUMMARY.md.

**Why offline diff is essential here:** Full-pipeline sweeps consume hours of LLM time per cell. The 13d regression was visible at the **entity-extraction output level** — but the 13d call site that broke was the mention-type field, not the anchor set. EXT-01 reverses this: the **anchor-collection** call sites dominate the LLM cost (3 of 6 sites), so the anchor-set diff is the lowest-cost probe that captures the highest-blast-radius failure. **Critically: a low-Jaccard cell can still pass macro F1** if the LLM's "wrong" anchors happen to be more architecturally informative than the regex's. The diff stage drops only **catastrophically** divergent cells, not merely **differently** divergent cells. This is why the drop threshold is conservative (0.3 Jaccard floor on hard tier), not "any divergence."

### Stage 2 — Finalist Full Sweep

- Run survivors (1-2 cells) on full 5-project sweep via `python run_ablation.py --variant s_linker13g_<X>`.
- Required artifacts per surviving variant: `results/ablation_results/ablation_<ts>.json` with per-dataset P/R/F1, FP/FN detail, source breakdown (seed/entity/coreference).
- Cross-check vs **Dual-Floor Thresholds** below.

### Dual-Floor Thresholds (numeric — verified from `ablation_20260404_000505.json`)

`s_linker12c` per-dataset F1 baseline (the floor GATE-01 compares against):

| Dataset | s_linker12c F1 | EXT-01 floor (per-dataset) |
|---------|---------------:|---------------------------:|
| mediastore | 1.0000 | ≥ 0.9800 (≤ 2pp below) |
| teastore | 0.9811 | ≥ 0.9611 (≤ 2pp below) |
| teammates | 0.9464 | ≥ 0.9264 (≤ 2pp below) |
| bigbluebutton | 0.8073 | ≥ 0.7473 (≤ 6pp below, BBB tolerance) |
| jabref | 1.0000 | ≥ 0.9800 (≤ 2pp below) |

`s_linker13` (canonical v1.0 final, macro F1 = 0.9509) per-dataset reference (`ablation_20260529_215932.json`, variant `s_linker13f` which is byte-equivalent to `s_linker13`):

| Dataset | s_linker13 F1 | GATE-05 hard-tier dev-loop floor (vs parent) |
|---------|--------------:|--------------------------------------------:|
| mediastore | 0.9841 | — (not hard tier) |
| teastore | 1.0000 | — (not hard tier) |
| teammates | 0.9474 | ≥ 0.9374 (no regress > 1pp) |
| bigbluebutton | 0.8990* | ≥ 0.8890 (no regress > 1pp) |
| jabref | 0.9730 | — (not hard tier) |

\* BBB number from `s_linker13f` final-sweep cell; canonical `s_linker13`'s sweep is byte-equivalent per PROJECT.md.

**Aggregate (macro) requirements:**
- macro F1 ≥ 0.93 (GATE-01)
- Hard-tier GATE-05: TM Δ vs s_linker13 ≥ -0.01 AND BBB Δ vs s_linker13 ≥ -0.06 (hard-tier-first, dev-loop)

---

## Replacement Site Inventory

Exhaustive list of every line referencing `_has_standalone_mention` in `s_linker13.py` (and any helpers/closures the replacement must subsume). Sourced from `grep -n "_has_standalone_mention" s_linker13.py`.

| Call Site | Line | Context | Replacement Topology Hint |
|-----------|------|---------|---------------------------|
| #1 — seed_val anchor section | 510 | `if self._has_standalone_mention(comp_name, s.text):` inside per-component loop building "KNOWN REFERENCES" block | Per-component batched call OR document-level enrichment map. NOT piggyback-able (these are sentences NOT in seed set — extractor may have skipped them). |
| #2 — bundle mention_type | 623 (via `_classify_mention` at 617-649) | First branch of `_classify_mention`: if standalone → returns `"proper case, standalone"` | Piggyback on entity extraction (Spike 003 enum). Zero net cost. |
| #3 — bundle anchor list | 675 | `if self._has_standalone_mention(comp_name, s.text):` inside `_build_evidence_bundle` anchor loop (≤5 cap) | Same map as #1. |
| #4 — has_exact_case flag | 880 | `has_exact_case = self._has_standalone_mention(c.component_name, sent.text)` in `_validate_with_evidence` generic-filter pre-pass | Piggyback (same map as extraction; per-candidate boolean). |
| #5 — generic-filter anchor | 895 | `if self._has_standalone_mention(comp_name, s.text):` inside generic-filter prompt assembly | Same map as #1. |
| #6 — coref antecedent verify | 1095 | `if not (self._has_standalone_mention(comp, ant_sent.text) or res.get("antecedent_via_alias", False)):` in `_coref_cases_in_context` | Same map as #1, OR explicit per-(comp, ant_snum) lookup. Verification only — LLM has already proposed the resolution. |

**Recommended topology (Claude's Discretion call):**

1. **One new document-level Tier-1 task** — `_compute_standalone_mention_map(sentences, components) -> dict[(comp_name, snum), bool]`. Runs in parallel with `model`, `doc_knowledge`, `seed` in the existing `_run_parallel` at lines 240-244. Per-component batched (50 sents/batch like extraction). Serves call sites #1, #3, #4, #5, #6.
2. **Extend `_extract_entities_enriched` prompt** with the Spike-003-style `mention_type` enum field on each emitted reference. Replaces `_classify_mention` (call site #2) entirely. Zero net new calls.

**Total new LLM call topology:**
- New cost: ~1 batched call per component for the map (BBB: ~12 components × ceil(250/50)=5 batches → ~60 calls per dataset).
- Removed cost: `_classify_mention` had no LLM cost (it was regex), so no offset.
- **Net delta:** +60 calls on BBB (worst case), +30 on TM, less elsewhere. **D-06 baseline number.**

Sub-variant (a) vs (b) topology difference: (a) pre-filters the input sentence list to the per-component batched call to exclude dotted-path-only mentions before paying LLM cost; (b) passes all sentences and asks the LLM to identify dotted-path-only as a NOT-standalone case in the same call. Call-count is identical; (a) sees ~10–20% fewer tokens per batch; (b) prompt is longer (carries the dotted-path teaching).

---

## Prompt Domain Selection (GATE-06 input)

Per BENCHMARK_TABOO.md §"Safe SE Textbook Examples" — must use these domains for prompt examples. Recommended pick for the new prompt's didactic examples:

| Domain | Concept demonstrated | Example sentence (illustrative — planner to finalize) |
|--------|----------------------|-------------------------------------------------------|
| **Compiler design** | Standalone proper-case mention | "The **Parser** consumes tokens emitted by the lexer." → standalone reference to `Parser` component |
| **Compiler design** | Token embedded in dotted identifier (the EXT-02 case sub-variant (b) must teach) | "The class `compiler.parser.ASTBuilder` extends the base class." → `Parser` is NOT a standalone mention here (it appears inside a qualified identifier) |
| **Operating systems** | Generic-English collision (architectural-vs-ordinary, the Semantic-cell case) | "The system schedules tasks using a **Scheduler** with a priority queue." → reference to `Scheduler`. Contrast: "The user must schedule a backup." → NOT a reference even though 'schedule' is the word root. |
| **Operating systems** | Token at sentence boundary with punctuation (a regex-corner case the LLM should also pass) | "Disk I/O is handled by the **FileSystem**." → standalone (trailing period is sentence end, not in-identifier dot). |

**GATE-06 audit rule:** No example may use any term from BENCHMARK_TABOO.md. The 4 illustrative examples above use `Parser`, `ASTBuilder`, `Scheduler`, `FileSystem` — all on the explicit safe list (BENCHMARK_TABOO.md lines 62-68).

**Why these specific 4 examples:**
- Example 1 establishes positive-case (the simplest "yes, standalone" pattern).
- Example 2 is the dotted-path negative — **critical** for sub-variant (b) to defuse the 13d failure mode. Without this teaching example, sub-variant (b) is highly likely to regress on TM (which has documentation like "logic, ui.website, ui.controller represent an MVC pattern" — exactly the 13d failure sentence at s_linker13.py:1102 FP).
- Example 3 differentiates Literal scope from Semantic scope (the D-01 axis). The Semantic cell must reject "schedule a backup"; the Literal cell must accept it (because the surface token matches).
- Example 4 covers the punctuation/boundary edge — provides robustness against regex-corner-case loss without project-specific teaching.

---

## Runtime State Inventory

> Phase 6 is a code change to a single linker variant plus registration. Includes one runtime-cache namespace addition.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Per-variant checkpoint pickles in `$PHASE_CACHE_DIR` (default `./results/phase_cache/<_VARIANT_NAME>/<dataset>/`) — created on first run of each sub-variant. | No action: new variants get new dirs; `_VARIANT_NAME` assertion at s_linker13.py:1165 prevents collision. |
| Live service config | None — no external services. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | `CLAUDE_MODEL`, `OPENAI_MODEL_NAME`, `PHASE_CACHE_DIR`, `LLM_LOG_DIR` — already read by `SLinker13.__init__` (lines 170-171, 1160). | None — names unchanged. |
| Build artifacts | None (`pip install -e ".[dev,openai]"` uses source layout, no compiled artifacts). | None. |

**Variant registration is NOT a runtime state issue** — it is a source edit to `run_ablation.py` (`CANONICAL_VARIANTS` list at lines 40-80; `VARIANT_SPECS` dict at lines 274-316). Two entries per sub-variant (intermediate) and one for the canonical (post-promotion).

---

## Common Pitfalls

### Pitfall 1: The 13d TM Regression (THE controlling negative result)

**What goes wrong:** LLM-emitted mention type / standalone signal on TeaMMates produces ~33 FPs because the LLM cannot reproduce the dotted-path Java-package convention from training data alone. Net macro F1 regression -19pp. v1.0 VAR-04 was **retired empirically** as a publishable negative result over this exact failure mode.

**Why it happens:** TeaMMates documentation contains sentences like `"logic, ui.website, ui.controller represent an application of Model-View-Controller pattern."` (verified s_linker13f run: s_linker13.py:1102 FP detail, sentence 22, component `Logic`). The regex correctly rejects this as a dotted-path context; the LLM in a piggyback-shape extraction prompt sees a comma-separated list of identifiers and judges each as a "reference to the component." 33 such cases on TM alone → -19pp.

**How to avoid for EXT-01:**
- **Sub-variant (a) inherits the regex** as a pre-filter — structurally cannot reproduce 13d.
- **Sub-variant (b) MUST include the dotted-identifier example** in its prompt (Example 2 above). The diff-stage D-02 threshold (`min_jaccard < 0.3` on TM) is calibrated to catch this exact failure mode at the anchor-set level **before** paying for a full sweep.
- **Diff-stage hard rule:** If sub-variant (b) shows `D > 10` on TM at the per-component level for components named after generic English (`Logic`, `Storage`, `Common`, `UI`) — drop it. These are the 13d-class failures.

**Warning signs:**
- Anchor count on TM increases by > 30% vs regex baseline → over-collection signature.
- Per-(comp, ds) diff log shows new anchors from sentences containing `.` between identifiers.
- Full-sweep TM F1 < 0.93 → 13d redux.

**Source:** PROJECT.md line 24 ("Spike 003 LLM mention classifier integration attempted — REJECTED ... 33 entity-source FPs on TeaMMates → −18.8 pp regression."), v1.0-ROADMAP.md line 28, METHODOLOGY.md §4 "The 13d Failure Mode".

### Pitfall 2: Checkpoint Cache Cross-Contamination

**What goes wrong:** Two sub-variants share `_VARIANT_NAME`, or the byte-copy to canonical `s_linker13g.py` forgets to update `_VARIANT_NAME` → checkpoint reads stale pickles from the other variant → bogus "ablation" numbers.

**Why it happens:** `_checkpoint_dir` namespacing at s_linker13.py:1159 only asserts `_VARIANT_NAME in d` — it does NOT detect duplicates. The assertion catches "missing" but not "collision."

**How to avoid:**
- Choose distinct, descriptive `_VARIANT_NAME` per sub-variant: `s_linker13g_pre`, `s_linker13g_sem`.
- After byte-copy to canonical: search-and-replace `_VARIANT_NAME = "s_linker13g_<X>"` → `"s_linker13g"` is the ONLY required edit. Verify with `grep _VARIANT_NAME src/llm_sad_sam/linkers/experimental/s_linker13g.py`.
- Wipe `results/phase_cache/s_linker13g*/` before the canonical final sweep to ensure clean numbers.

**Warning signs:** Two variants producing byte-identical FP/FN detail dicts. Final-sweep numbers matching a sibling's intermediate-sweep numbers exactly.

### Pitfall 3: Forgetting GATE-07 Dual Registration

**What goes wrong:** Variant added to `VARIANT_SPECS` but not `CANONICAL_VARIANTS` (or vice-versa) → `run_ablation.py --list-variants` shows partial registration, sweep runs but does not appear in canonical artifacts.

**Why it happens:** Two-list pattern is easy to half-implement.

**How to avoid:**
- Always edit both lists in the same commit.
- Canonical `s_linker13g` MUST have `canonical=True` in its `VARIANT_SPECS` entry (mirror line 315 for `s_linker13`).
- Verify with `python run_ablation.py --list-variants | grep s_linker13g`.

### Pitfall 4: Prompt-Example Leakage (GATE-06 failure)

**What goes wrong:** Prompt examples use words from BENCHMARK_TABOO.md → audit fails → variant rejected even at macro F1 ≥ 0.93.

**Why it happens:** "Universal Taboo" list at BENCHMARK_TABOO.md lines 31-58 contains many common SE words (`logic`, `UI`, `client`, `storage`, `common`, `model`, `database`, `cache`, `registry`, `auth`, `server`, `persistence`, `facade`, `recording`, `cascade`, `validation`, `internal`, `adapter`, `order`, `processor`, `event`, `socket`, `layer`, `preferences`, `config`). Easy to use accidentally.

**How to avoid:**
- Pre-clear all prompt-example words against BENCHMARK_TABOO.md before commit. Recommended pre-commit check: `grep -i -E "(logic|UI|client|storage|common|model|database|cache|registry|...)" prompts_v2.py | grep -v "^#"`.
- Use the 4 examples in "Prompt Domain Selection" above as the starting set (already audited).
- GATE-06 audit MUST be recorded in `06-SUMMARY.md` and `06-GATE-06-AUDIT.md`.

### Pitfall 5: Diff Stage Picks the Wrong Loser

**What goes wrong:** Sub-variant (b) shows higher anchor-set divergence than (a) on BBB (because semantic-scope re-anchors on architecturally-meaningful sentences regex misses), gets dropped. But (b) might have produced higher macro F1 if it had reached the full sweep.

**Why it happens:** D-02's drop-rule is a **conservative low-pass filter** on divergence; the principle is "kill catastrophes, not differences." Setting the threshold too tight (e.g., `min_jaccard < 0.7`) is the failure.

**How to avoid:**
- Keep the recommended threshold (`min_jaccard < 0.3` on hard tier; `D > 10` per-component) — these are calibrated to catch 13d-class failures (anchor count doubling from dotted-path over-collection), not gentle divergence.
- If diff-stage flags both (a) and (b) for drop, **escalate to user** — D-02 is a filter, not a decider; D-03 picks the winner from survivors, and a no-survivor outcome is a research-design failure to surface (planner must flag this scenario in PLAN).

### Pitfall 6: Mistaking Spike-003 Validation for End-to-End Pipeline Validation

**What goes wrong:** Spike 003 validated only the **consumer-side enum-to-string formatter** (`format_mention`), not that the LLM actually emits correct enums in pipeline conditions. The 13d FAILURE was at the LLM-emission side, not the formatter side.

**Why it happens:** Spike-003 README claims "VALIDATED" → easy to assume the whole approach is safe.

**How to avoid:**
- Treat Spike 003 only as the **API-shape pattern reference**, NOT as evidence that piggybacking works for EXT-01 specifically. v1.0 13d already proved piggyback can break on TM.
- For sub-variant (b) (the riskier cell), test on **TM in isolation** during the hard-tier dev loop BEFORE running the full sweep (GATE-05 explicit rule).

---

## Code Examples

### Example A — New prompt constant in `prompts_v2.py` (sub-variant b — semantic + LLM-only dotted-path)

```python
# Source: NEW for EXT-01, modeled on prompts_v2.py:179-191 (ENTITY_EXTRACTION_RULES)
# All examples drawn from BENCHMARK_TABOO.md §"Safe SE Textbook Examples"

STANDALONE_MENTION_RULES_LLM_ONLY = """STANDALONE-MENTION DETECTION — answer YES if the sentence makes a standalone, on-its-own reference to the component named `<comp>`; NO if the name appears only as part of a longer code identifier, in an unrelated technical phrase, or as an ordinary English word.

RULES:
1. YES when the component name appears as a standalone token, including as a subject, object, or in a list of components — e.g., "The Parser consumes tokens emitted by the lexer." → YES for Parser.
2. NO when the name appears only inside a qualified or dotted identifier — e.g., "compiler.parser.ASTBuilder extends the base class." → NO for Parser; "Parser" is a path segment, not a standalone reference.
3. NO when the name participates only in a hyphenated compound where the compound denotes a different entity — e.g., "Parser-style grammar" → NO for Parser.
4. YES when the name is the subject of an architectural action — performs work, provides a service, is configured, receives input.
5. When uncertain between a surface mention and a generic English use, favor YES — the downstream validator filters generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""
```

### Example B — New Tier-1 task: document-level standalone-mention map

```python
# Source: NEW for EXT-01 — sub-variant (a) shape (LLM only sees pre-filtered sentences).
# Modeled on _run_single_extraction_pass batching at s_linker13.py:748-803.

def _compute_standalone_mention_map(self, sentences, components):
    """Document-level: (comp_name, snum) -> bool.

    Sub-variant (a): regex pre-filter strips dotted/hyphen contexts before the LLM
    sees the sentence; LLM judges only word-boundary-clean candidates.
    """
    comp_names = self._get_comp_names(components)
    smap = {}  # (cname, snum) -> bool

    for cname in comp_names:
        # PRE-FILTER (sub-variant a only — sub-variant b skips this and lets the LLM decide)
        cand_sents = [
            s for s in sentences
            if not self._in_dotted_or_hyphen_context_only(cname, s.text)  # NEW helper
            and cname.lower() in s.text.lower()  # cheap pre-narrowing
        ]
        if not cand_sents:
            continue

        for batch_start in range(0, len(cand_sents), 50):
            batch = cand_sents[batch_start:batch_start + 50]
            prompt = f"""{STANDALONE_MENTION_RULES_PRE_FILTERED}

COMPONENT: {cname}
SENTENCES:
{chr(10).join(f"S{s.number}: {s.text}" for s in batch)}
"""
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("results"):
                    break
                if attempt == 0:
                    print(f"    Standalone-mention [{cname}] retry...")
            if not data:
                # Approve-biased fallback: assume all candidates ARE standalone mentions
                for s in batch:
                    smap[(cname, s.number)] = True
                continue
            for r in data.get("results", []):
                snum = self._parse_snum(r.get("sentence"))
                if snum is not None:
                    smap[(cname, snum)] = bool(r.get("standalone", True))

    return smap
```

### Example C — Consumer shape (replaces 6 call sites)

```python
# Source: NEW for EXT-01 — drop-in replacement at each of the 6 call sites.
# self._standalone_map is the dict computed by _compute_standalone_mention_map and
# attached to self by the Tier-1 task (see Pattern 4 above).

def _has_standalone_mention_llm(self, comp_name, snum):
    """Lookup against the precomputed map. No regex, no LLM call."""
    return self._standalone_map.get((comp_name, snum), False)
```

Each of the 6 original sites becomes (for example, the s_linker13.py:510 case):
```python
# Before:
if self._has_standalone_mention(comp_name, s.text):
# After:
if self._has_standalone_mention_llm(comp_name, s.number):
```

### Example D — Registration in `run_ablation.py`

```python
# Source: NEW edits to run_ablation.py — appended to lines 40-80 and 274-316.

# In CANONICAL_VARIANTS list (after "s_linker13"):
    "s_linker13g_pre",
    "s_linker13g_sem",
    "s_linker13g",   # canonical promotion of winning sub-variant (Phase 6)

# In VARIANT_SPECS dict (after the "s_linker13" entry):
    "s_linker13g_pre": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_pre",
        class_name="SLinker13gPre",
        description="S-Linker13g-pre: 13 - _has_standalone_mention via LLM with regex pre-filter for dotted-path",
    ),
    "s_linker13g_sem": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_sem",
        class_name="SLinker13gSem",
        description="S-Linker13g-sem: 13 - _has_standalone_mention via LLM-only (dotted-path encoded in prompt)",
    ),
    "s_linker13g": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g",
        class_name="SLinker13g",
        description="S-Linker13g: canonical promotion of winning EXT-01 sub-variant (Phase 6) — 7 rules removed cumulatively from 12c",
        canonical=True,
    ),
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Regex `_has_standalone_mention` with dotted/hyphen guards (s_linker12c through s_linker13) | LLM primitive — sub-variant (a) keeps regex as pre-filter; sub-variant (b) goes LLM-only with prompt-encoded dotted-path concept | Phase 6 (this phase) | EXT-01 is the FINAL structural-rule removal in the v2.0 thesis chain. Success → only parsers/formatters remain as non-LLM. |
| Spike-003 piggyback as proven-safe pattern | Spike-003 piggyback validated only the consumer side; the LLM-emission side was empirically retired in v1.0 (13d/VAR-04) for TM | v1.0 Phase 3 close-out (2026-05-29) | EXT-01 must NOT assume Spike 003 is a free safe choice. The piggyback shape is still **the recommended call topology** for the mention-type field (call site #2 only) because that field was the Spike-003 enum verified; but the dotted-path-bearing call sites (#1, #3, #5, #6) must NOT rely solely on piggyback. |
| GATE-04 (v1.0 BENCHMARK_TABOO scan) | GATE-06 (v2.0): BENCHMARK_TABOO scan + reviewer-defensibility check | v2.0 kickoff (2026-05-30) | Mechanical scan is insufficient — the audit now also asks "would a reviewer believe this prompt generalizes to a random new project?" Phase-6 SUMMARY.md must answer both halves. |

**Deprecated/outdated:**
- Spike-003 README's `VERDICT: VALIDATED ✓` — outdated framing. The v1.0 outcome is the authoritative state: piggyback is **API-shape valid**, NOT **end-to-end safe**. Treat as pattern reference only.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The "catastrophic diff" threshold (`min_jaccard < 0.3` on hard tier; `D > 10` per-component) calibrates well to catching 13d-class failures. | Empirical Matrix Operationalization | If too loose: a (b)-variant with mid-Jaccard but high TM-FP count slips through → expensive full-sweep failure. If too tight: a high-F1 cell gets dropped from the matrix → suboptimal winner. Mitigation: planner runs the threshold rule against the s_linker13d historical diff (if recoverable) as a sanity check; planner may calibrate downward if 13d-class signal was visible at higher Jaccard. |
| A2 | The recommended hybrid topology (Tier-1 standalone-mention map + Spike-003-piggyback for mention_type field) is the lowest-cost call shape that satisfies D-02 + D-06. | Primary recommendation | If wrong: a per-(comp, sent) shape might be cheaper than expected on small datasets, or per-component batch might fragment too much. Mitigation: the D-06 cost signal is captured deterministically by the call log; planner can switch topology between sub-variants if the cost numbers suggest a re-design. The matrix axes (D-01, D-04) are orthogonal to topology, so topology change doesn't invalidate the diff stage. |
| A3 | `s_linker12c` per-dataset F1 numbers from `ablation_20260404_000505.json` are the GATE-01 baseline (no later 12c re-run overrides). | Dual-Floor Thresholds | If a more recent 12c sweep exists with different numbers, the per-dataset floors shift. Mitigation: planner confirms with `grep -l s_linker12c results/ablation_results/*.json | tail -1` and re-checks before locking the gate values into PLAN. |
| A4 | The PROJECT.md "do not change retained upstream components" rule permits adding a new optional field to `CandidateLink` (for the Spike-003-style `standalone_mention` boolean) on the grounds that it is "required by a rule removal." | Standard Stack table | If wrong: cannot use the Spike-003 piggyback shape without a workaround (e.g., maintain the map separately). Mitigation: planner can sidestep by keeping the standalone-mention signal entirely in `self._standalone_map` (a dict), without touching `CandidateLink`. Recommended fallback. |
| A5 | The "5-anchor cap" in `s_linker13` (lines 512, 677) is the natural unit of comparison for the Jaccard threshold (i.e., 5 anchors → 1.5 of 5 agreement is "0.3 Jaccard"). | Empirical Matrix Operationalization | The cap is a UI-cosmetics decision (anchor section in prompt), not a semantic decision. If planner relaxes the cap in the LLM-emitted map, Jaccard reasoning shifts. Mitigation: compute Jaccard on the **full** unfiltered anchor set per (comp, ds), not the cap-truncated 5. The cap is applied downstream during prompt assembly only. |

---

## Open Questions

1. **Where should the precomputed `_standalone_map` be persisted?**
   - What we know: `_save_phase` writes pickles to `_checkpoint_dir`; existing phases (`layer1`, `entity_candidates`, etc.) are well-structured.
   - What's unclear: Is it a Tier-1 sub-phase (`"standalone_map"`) or attached to `"layer1"`?
   - Recommendation: New top-level phase name `"standalone_map"` so it can be invalidated independently (D-02 diff stage can be re-run from pickle without re-running other Tier-1 tasks).

2. **Should the diff stage also probe the `_classify_mention` output (Spike-003 enum), or only the boolean `_has_standalone_mention` output?**
   - What we know: `_classify_mention` returns a string with 4 distinct values; the bool is one component.
   - What's unclear: A cell could agree on the bool but disagree on the mention-type string (e.g., regex says "lowercase, inside dotted path"; LLM says "indirect/unclear"). Whether that flux is relevant for D-02 is judgement.
   - Recommendation: probe **bool only** for D-02 (matches the rule the phase removes). Mention-type-string flux is a separate concern, observable in the final sweep's FP detail.

3. **Does the diff stage need its own dedicated LLM call topology, or can it reuse the variant's production topology?**
   - What we know: D-02 says "no full-pipeline run during diff stage."
   - What's unclear: Does "no full-pipeline" prohibit running just `_compute_standalone_mention_map` in isolation?
   - Recommendation: Run `_compute_standalone_mention_map` standalone (no Tier 2/3). This is the most faithful reproduction of the variant's behavior at zero pipeline-wide cost. Add a `--diff-stage` flag to `run_ablation.py` that loads cached Tier-1 outputs and runs only the new map computation.

4. **If sub-variant (a) wins the matrix, does Phase 7 (EXT-02) actually delete the same regex this phase preserves?**
   - What we know: D-04 says sub-variant (a) keeps regex as "cheap pre-filter; EXT-02 then has a clean target (the pre-filter) to drop."
   - What's unclear: If sub-variant (b) loses precisely because of dotted-path failures, then deleting the (a)-winner's pre-filter in EXT-02 will recreate exactly the (b)-failure → EXT-02 fails → milestone proceeds without EXT-02.
   - Recommendation: Planner should flag in PLAN that "Phase 7 success is conditional on EITHER sub-variant (b) being the Phase 6 winner OR sub-variant (a) being the winner but tolerating regex removal." If (a) wins decisively on macro F1, Phase 7 is **already** a high-risk gamble. Suggest planner add an explicit "Phase 7 risk note" line to the SUMMARY's Phase 8 hand-off block (D-06 mentions stackable-vs-unify; planner may also note Phase 7 risk).

---

## Sources

### Primary (HIGH confidence — verified via file reads)

- `s_linker13.py` lines 1-247 (header/setup), 480-694 (seed_val + bundle), 711-803 (entity pipeline core), 805-839 (extract_enriched), 841-999 (validate_with_evidence), 1031-1101 (coref), 1107-1170 (helpers + checkpoint dir) — full call site mapping.
- `prompts_v2.py` lines 14-247 — existing prompt constants pattern.
- `run_ablation.py` lines 40-80 (CANONICAL_VARIANTS), 270-317 (VARIANT_SPECS) — GATE-07 registration shape.
- `.planning/spikes/002-rules-audit/AUDIT.md` lines 12, 32-44, 53 — RISKY classification + O(N×M) cost + recommended removal order.
- `.planning/spikes/003-llm-mention-classifier/spike.py` + README — piggyback API shape, formatter contract, 6 enum branches.
- `BENCHMARK_TABOO.md` lines 6-68 — taboo terms + safe SE textbook examples.
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/PROJECT.md`, `.planning/STATE.md` — gates, requirements, milestone scope.
- `.planning/phases/06-*/06-CONTEXT.md` — locked decisions D-01..D-06.
- `results/ablation_results/ablation_20260404_000505.json` — `s_linker12c` per-dataset baseline numbers (MS=1.0, TS=0.9811, TM=0.9464, BBB=0.8073, JAB=1.0).
- `results/ablation_results/ablation_20260529_215932.json` — `s_linker13f` (= canonical `s_linker13`) per-dataset numbers (MS=0.9841, TS=1.0000, TM=0.9474, JAB=0.9730).
- `MILESTONES.md` line 36 — `s_linker13f` macro 0.9509.
- `.planning/milestones/v1.0-ROADMAP.md` line 28 — VAR-04 closed empty + -19pp TM regression.

### Secondary (MEDIUM confidence — derived from primaries)

- The 4 illustrative prompt examples in "Prompt Domain Selection" — chosen from BENCHMARK_TABOO.md safe list but specific phrasing is a research suggestion, NOT verified empirically.
- The recommended diff threshold (`min_jaccard < 0.3` on hard tier, `D > 10` per-component) — derived from the 5-anchor cap and the known 13d failure shape, but NOT calibrated against a 13d historical diff (which may not exist as a recoverable artifact).

### Tertiary (none — no WebSearch used; all data is in-repo)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library/pattern verified against repo source.
- Architecture: HIGH — DAG tier structure read directly from `s_linker13.link()` at lines 216-310.
- Pitfalls: HIGH — Pitfall 1 (13d) is the documented authoritative v1.0 negative result; Pitfalls 2-6 are derived from explicit assertions and project conventions in the source.
- Empirical matrix operationalization: MEDIUM — the threshold numbers are reasoned-from-anchor-cap, not measured against the 13d historical diff. Planner should treat them as a starting calibration, not a locked specification.

**Research date:** 2026-05-30
**Valid until:** 2026-06-13 (14 days — repo changes daily during a milestone; planner should re-read `s_linker13.py` if any newer 12c sweep has landed).

---

*Phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive*
*Research output for `/gsd-plan-phase 6` consumer.*
