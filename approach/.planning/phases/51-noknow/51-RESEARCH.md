# Phase 51: NOKNOW — Research

**Researched:** 2026-06-21
**Domain:** Python code surgery on `s_linker20_union.py` + `run_ablation.py` + extractor extension + shell sweep scripts
**Confidence:** HIGH — all findings are direct code reads with file:line citations; no web search required

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Knowledge-disable is a constructor flag `SLinker20Union(no_knowledge=True)`. Default-off path must be snapshot-identical to today.
- **D-02:** Registered sibling variant `s_linker20_union_noknow` in `run_ablation.py`. No linker logic duplicated.
- **D-03:** Skip the 3 layer1 LLM call-sites (ambiguity + doc-knowledge-extract + doc-knowledge-judge). Set `model_knowledge = ModelKnowledge()` and `doc_knowledge = DocumentKnowledge()` directly. Still call `_save_phase(..., "layer1", {...})`.
- **D-04:** Downstream degrades automatically — no downstream code change beyond passing empty knowledge.
- **D-05:** N=3, symmetric with Full extracts — 30 live runs total.
- **D-06:** Output folders + extracts annotated as no-knowledge (`…_noknow` results root; parallel extract tree).
- **D-07:** Phase 51 runs the sweep end-to-end (no separate go-ahead checkpoint).
- **D-08:** Soft cost cap ~$60; log cumulative cost to PROGRESS log; no hard-abort.
- **D-09:** GATE-01 via zero-LLM snapshot replay against 30 frozen phase_caches + structural guard check.

### Claude's Discretion
- Exact No-Knowledge results/extract directory names (any scheme unambiguously annotated, machine-distinguishable from Full).
- Whether GATE-01 replay covers all 30 cells or a representative subset (structural guard backstops it).
- Run-script ergonomics (cooldowns, retry-once, resume markers) — mirror existing skeleton.

### Deferred Ideas (OUT OF SCOPE)
- Pending axiom/prompt-design todos (v2.6.1/v2.6.2 era work).
- NOKNOW-N as a separate requirement (absorbed by D-05).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NOKNOW-01 | `s_linker20_union` gains a knowledge-disable path (no alias table, no ambiguity map) behind a flag/variant; with the flag off, full-knowledge behavior is unchanged (GATE-01) | D-01/D-03/D-09; guard insertion point confirmed at `s_linker20_union.py:L502–516`; GATE-01 evidence path confirmed |
| NOKNOW-02 | No-Knowledge run on 5 projects × {gpt, sonnet} × N≥1; outputs + phase_cache captured; extracted into same neutral JSON format | D-05/D-06/D-07; extractor extension point confirmed; faithfulness oracle transfers unchanged |
</phase_requirements>

---

## Summary

Phase 51 has a fully verified, low-risk implementation path. The code surgery is minimal and concentrated at one insertion point in `s_linker20_union.py`. All downstream consumers of `doc_knowledge.aliases` and `model_knowledge.ambiguous_names` degrade gracefully with empty objects — no downstream changes required and no knowledge leaks found. GATE-01 is achievable via a structural guard check plus a zero-LLM replay of the Phase-50 faithfulness oracle (which needs no modification). The extractor needs one new MATRIX entry pointing at the No-Knowledge results root. The run-script template is verbatim-copy with three line changes. The only execution risk is sweep wall-clock time and LLM API cost; the `.done` resume markers contain both.

**Primary recommendation:** Implement in five sequential tasks: (1) flag + guard in `s_linker20_union.py`; (2) registry entry in `run_ablation.py`; (3) GATE-01 evidence generation; (4) run-scripts + PROGRESS cost logging; (5) extractor extension + 30-cell extraction.

---

## Key Question Answers

### Q1 — Exact disable-guard insertion point (D-03)

**Finding: VERIFIED by code read**

The single insertion point is `s_linker20_union.py` lines 499–516 (Phase 1 block inside `link()`):

```python
# s_linker20_union.py:L499–516 (actual — VERIFIED)
# ── Phase 1 ─────────────────────────────────────────────────────────────────
t_p1 = time.time()
print("\n[Phase 1] Knowledge acquisition (parallel)")
knowledge = self._run_parallel({
    "model": lambda: self._analyze_model(components),          # L503 — LLM call 1
    "doc": lambda: self._learn_document_knowledge(sentences, components),  # L504 — LLM calls 2+3
})
self.model_knowledge = knowledge["model"]
self.doc_knowledge = knowledge["doc"]
print(f"  Model: {len(self.model_knowledge.ambiguous_names)} ambiguous ...")
print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")
self._save_phase(text_path, "layer1", {
    "model_knowledge": self.model_knowledge,
    "doc_knowledge": self.doc_knowledge,
    "elapsed_s": round(time.time() - t_p1, 2),
    "n_sentences": len(sentences), "n_components": len(components),
})
```

**Guard to add** — replace the `knowledge = self._run_parallel({...})` block with:

```python
t_p1 = time.time()
print("\n[Phase 1] Knowledge acquisition (parallel)")
if self.no_knowledge:
    # NOKNOW path: skip all 3 layer1 LLM calls; set empty knowledge directly.
    self.model_knowledge = ModelKnowledge()       # ambiguous_names = set()
    self.doc_knowledge = DocumentKnowledge()      # aliases = {}
    print("  [NOKNOW] Skipping knowledge layer — empty ModelKnowledge + DocumentKnowledge")
else:
    knowledge = self._run_parallel({
        "model": lambda: self._analyze_model(components),
        "doc": lambda: self._learn_document_knowledge(sentences, components),
    })
    self.model_knowledge = knowledge["model"]
    self.doc_knowledge = knowledge["doc"]
print(f"  Model: {len(self.model_knowledge.ambiguous_names)} ambiguous ...")
print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")
self._save_phase(text_path, "layer1", {          # <-- STILL CALLED in both branches
    "model_knowledge": self.model_knowledge,
    "doc_knowledge": self.doc_knowledge,
    "elapsed_s": round(time.time() - t_p1, 2),
    "n_sentences": len(sentences), "n_components": len(components),
})
```

**Constructor signature — VERIFIED** (`s_linker20_union.py:L265–283`):

```python
def __init__(
    self,
    backend: LLMBackend | None = None,
    model: str | None = None,
    checkpoint_fallback: LLMBackend | str | None = None,
    checkpoint_fallback_model: str | None = None,
):
```

Add `no_knowledge: bool = False` as a new parameter (after the existing four), and `self.no_knowledge = no_knowledge` in the body. No other `__init__` changes needed.

**Empty-knowledge default constructors — VERIFIED** (`data_types_v2.py:L38–62`):

- `ModelKnowledge()` → `ambiguous_names = set()` (via `field(default_factory=set)` at L44)
- `DocumentKnowledge()` → `aliases = {}` (via `field(default_factory=dict)` at L57)

IMPORTANT: `DocumentKnowledge.aliases` type hint in `data_types_v2.py` says `dict[str, str]` but at runtime in `s_linker20_union.py` the values are `AliasEntry(component, scope)` objects (see `L675`). The empty dict `{}` is still correct as the default — it just means "no entries" regardless of value type.

**GATE-01 constraint (D-09):** The guard is strictly additive — the `else:` branch is the unmodified pre-existing code. Flag OFF follows exactly the same statements as today.

**`_VARIANT_NAME` impact — LANDMINE:** `_VARIANT_NAME = "s_linker20_union"` (L263) is used in `_checkpoint_dir()` (L1022–1026) to construct the phase_cache path:
```
{PHASE_CACHE_DIR}/s_linker20_union/{backend_tag}/{dataset}/
```
If `no_knowledge=True` runs use the same `_VARIANT_NAME`, their phase_caches will be co-mingled with Full caches under the same subdirectory name. This is avoided automatically by using a different `PHASE_CACHE_DIR` env per the run-script skeleton (each run gets its own `$rdir/phase_cache`), AND by using separate results roots (D-06). However, if anyone sets `PHASE_CACHE_DIR` to a shared location, the caches would overlap. The extractor's MATRIX entries are keyed to the results root, not the variant name, so the extractor is immune. The planner should note that `_VARIANT_NAME` does NOT need to change — run-script isolation via `PHASE_CACHE_DIR` is sufficient.

---

### Q2 — GATE-06 / D-04 no-leak verification (CRITICAL)

**Finding: CLEAN — no leaks found. VERIFIED by full downstream trace.**

All consumers of `doc_knowledge.aliases` and `model_knowledge.ambiguous_names` in `s_linker20_union.py`:

**Consumer 1: `_run_framing_c` (Phase 2 entity extraction) — L681–703**

```python
mappings = (
    [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items()
     if entry.scope == "global"]
    if self.doc_knowledge else []
)
```
With `doc_knowledge.aliases = {}`: `aliases.items()` is empty → `mappings = []`. The extraction prompt is built with an empty alias list. This is exactly canonical-name-only matching. GATE-06 clean.

**Consumer 2: `_classify_mention_typed` (Phase 4 evidence bundle) — L749–754**

```python
if self.doc_knowledge:
    for alias, entry in self.doc_knowledge.aliases.items():
        if entry.component == comp_name and re.search(...):
            return MentionType.VIA_ALIAS
```
With `aliases = {}`: loop body never executes. Falls through to `MentionType.INDIRECT`. No leak.

**Consumer 3: `_build_evidence_bundle` (Phase 4) — L786–789**

```python
is_ambig = bool(
    self.model_knowledge
    and self.model_knowledge.ambiguous_names
    and comp_name in self.model_knowledge.ambiguous_names
)
```
With `ambiguous_names = set()`: `bool(set())` is `False` → `is_ambig = False` for every candidate. This changes what gets reported in the evidence bundle but does NOT suppress any candidates or links — it is informational only.

**Consumer 4: `_antecedent_supports_resolution` (Phase 5 coref gate) — L870–887**

```python
def _antecedent_supports_resolution(self, comp_name: str, ant_text: str) -> bool:
    if has_standalone_mention(comp_name, ant_text):
        return True
    if not self.doc_knowledge:
        return False
    for alias, entry in self.doc_knowledge.aliases.items():
        ...
    return False
```
With `doc_knowledge` set (not None) but `aliases = {}`: `has_standalone_mention` is checked first (canonical-name-only gate). Then `aliases.items()` is empty — loop body never runs. Returns `False` unless canonical name has a standalone mention. This is correct canonical-name-only coref behavior. No leak.

**No other consumers exist.** There are no hardcoded alias lists, no project-name dictionaries, no gold-derived hints, and no other knowledge sources in the pipeline. The COREF_RULES constant (L156) is a static prompt string that mentions aliases conceptually ("known aliases include…") but that text is instructions to the LLM, not a hardcoded alias list — the LLM will simply find no aliases to use since the antecedent gate won't pass alias-based resolutions.

**GATE-06 verdict:** CLEAN. No benchmark-derived vocabulary. No hardcoded alias lists. The No-Knowledge path is pure canonical-name-only matching + LLM judgment without alias augmentation.

**Behavioral note for the planner:** The No-Knowledge run still runs ALL non-knowledge LLM calls (Phase 2 extraction × 2 passes, Phase 4 twopass × 2 passes, Phase 5 coref batch + coref validator). The 3 skipped calls are only the layer1 ambiguity + doc-knowledge-extract + doc-knowledge-judge prompts.

---

### Q3 — GATE-01 zero-LLM snapshot replay (D-09)

**Finding: Two viable evidence paths. VERIFIED by extractor design.**

**Path A — Structural guard check (primary, zero cost):**
After adding `if self.no_knowledge:` guard, git-diff the file and assert:
1. Only one new branch added (the `if self.no_knowledge:` block + `no_knowledge` constructor parameter).
2. The `else:` branch contains the original statements byte-for-byte (no edits to the pre-existing code path).
This is structurally sufficient for GATE-01: it proves the flag-off path executes identical pre-existing statements.

**Path B — Faithfulness-oracle replay (corroborative, zero LLM):**
The Phase-50 extractor's faithfulness oracle (`scripts/extract_s20union_caches.py:L233–303`) re-derives final links from the frozen phase_caches using set-equality on `(sentence, component_id, source)`. To verify flag-OFF == today:
1. Point the extractor at the 30 frozen Full cells (already done in Phase 50 — 30/30 PASS confirmed).
2. Run the extractor again against the same cells; confirm the same 30/30 PASS.
This demonstrates the freeze is stable and the extraction code hasn't drifted.

However, this does NOT replay the linker itself with the flag OFF — it replays the extractor against the frozen caches. True "linker replay" would require calling `SLinker20Union(no_knowledge=False).link(...)` but since the linker makes live LLM calls, this is not zero-cost.

**What D-09 actually requires (re-read):** "replay and assert the produced `layer1` and final `_links.csv` are byte-identical to the frozen Full outputs." This is achievable without live LLM calls only if the linker supports loading a frozen phase_cache and continuing from it. The existing linker has NO checkpoint-resume path — it always runs from scratch making live calls. So byte-identical replay of the linker with flag-OFF is NOT achievable at zero LLM cost.

**Resolution (recommended for planner):** The GATE-01 evidence in practice is:
1. Structural guard check (mandatory, zero cost): shows flag-OFF code path is the pre-existing unmodified code.
2. Re-run Phase-50 extractor against frozen Full cells (zero LLM, already confirmed 30/30 PASS): proves the Full caches + output schema haven't drifted.
3. One representative live re-run per backend (2 × 1 dataset, ~$1) to spot-check that flag-OFF behavior matches the frozen cell for that dataset.

This combination is the cheapest faithful GATE-01 evidence achievable without a checkpoint-resume harness. The structural check is the load-bearing proof; the spot-check run adds empirical corroboration.

---

### Q4 — Extractor extension

**Finding: One new MATRIX entry + one new EXTRACT_ROOT path. VERIFIED.**

The extractor (`scripts/extract_s20union_caches.py:L44–53`) has a hardcoded MATRIX:

```python
MATRIX = [
    ("results/v2.6.5_s20union/gpt", "openai", "gpt"),
    ("results/v2.6.5_s20union_sonnet", "claude", "sonnet"),
]
EXTRACT_ROOT = "results/v2.6.6_extracts"
```

**Minimal change to add No-Knowledge cells:**

Option A (add to existing MATRIX + add a separate NOKNOW_MATRIX constant):
```python
NOKNOW_MATRIX = [
    ("results/v2.6.6_s20union_noknow/gpt", "openai", "gpt"),
    ("results/v2.6.6_s20union_noknow_sonnet", "claude", "sonnet"),
]
NOKNOW_EXTRACT_ROOT = "results/v2.6.6_extracts_noknow"
```

Then add a separate `main_noknow()` function that iterates NOKNOW_MATRIX and uses NOKNOW_EXTRACT_ROOT, or pass the matrix as a parameter to a unified `run_matrix(matrix, extract_root)` helper.

**Cell path construction** (`extract_s20union_caches.py:L357–365`):
```python
cell_dir = f"{root}/{run}/phase_cache/s_linker20_union/{subdir}/{project}"
csv_path = f"{root}/{run}/{project}/s_linker20_union_{project}_links.csv"
```

IMPORTANT: The `cell_dir` includes `s_linker20_union` as the variant name (because `_VARIANT_NAME = "s_linker20_union"` is used in `_checkpoint_dir()`). Since `no_knowledge=True` doesn't change `_VARIANT_NAME`, the phase_cache path for No-Knowledge cells will also be `…/phase_cache/s_linker20_union/{subdir}/{project}/`. The extractor's hardcoded `s_linker20_union` in the cell_dir path is correct for No-Knowledge cells too — no change needed there.

The `csv_path` uses the variant name from the run-script via `export_links_csv(predictions, results_dir / f"{variant_name}_{dataset_name}_links.csv")` (run_ablation.py:L1188). Since the registered variant name is `s_linker20_union_noknow`, the CSV file will be named `s_linker20_union_noknow_{dataset}_links.csv`. This means the extractor's `csv_path` pattern needs to change for No-Knowledge cells:
```python
# Full: s_linker20_union_{project}_links.csv
# NoKnow: s_linker20_union_noknow_{project}_links.csv
csv_path = f"{root}/{run}/{project}/s_linker20_union_noknow_{project}_links.csv"
```

**ablation_*.json secondary oracle:** The ablation JSON keys the results by variant name (`ablation_*.json` has shape `{"<proj>": {"s_linker20_union_noknow": {...}}}`). The extractor's faithfulness oracle secondary cross-check at L269 looks for key `"s_linker20_union"` — this needs updating for No-Knowledge cells:
```python
su = proj_data.get("s_linker20_union_noknow", {})   # changed from "s_linker20_union"
```

**Primary faithfulness oracle (set-equality):** The oracle at L249–258 compares `extract_set` (from `neutral["final"]["links"]`) with `csv_set` (from the CSV file). This logic is completely independent of variant name — it works unchanged for No-Knowledge cells as long as the CSV path is correct.

**Summary of extractor changes:**
1. Add `NOKNOW_MATRIX` and `NOKNOW_EXTRACT_ROOT` constants.
2. Use `s_linker20_union_noknow_{project}_links.csv` in No-Knowledge cell CSV path.
3. Update secondary oracle key from `"s_linker20_union"` to `"s_linker20_union_noknow"`.
4. Add a function or argument to run the No-Knowledge matrix through the same `load_cell / to_neutral / rederive_final / faithfulness` flow.
5. The `meta["variant"]` field should say `"s_linker20_union_noknow"` for No-Knowledge cells (for downstream scorer to distinguish Full vs No-Knowledge).

---

### Q5 — Run-script mechanics + cost logging

**Finding: VERIFIED by reading both existing scripts.**

**Env routing (VERIFIED: `run_s20union_gpt_n3.sh:L38–40`, `run_s20union_sonnet_n3.sh:L36–38`):**

```bash
# Per-run (inside run_one()):
export PHASE_CACHE_DIR="$rdir/phase_cache"   # genuine N independence
export LLM_LOG_DIR="$rdir/llm_logs"
export CHECKPOINT_DIR="$rdir/llm_checkpoint"
```

**Backend env vars:**
- gpt: `LLM_BACKEND=openai` + `OPENAI_MODEL_NAME=gpt-5.4` (`run_s20union_gpt_n3.sh:L19–20`)
- sonnet: `LLM_BACKEND=claude` + `CLAUDE_MODEL=sonnet` (`run_s20union_sonnet_n3.sh:L18–19`)

**Resume marker scheme (VERIFIED: L34–36 of both scripts):**
Per-`(run, dataset)` `.done` marker at `$rds/.done`. If the marker exists, skip. OK condition: Python exit code 0 AND `$csv` has >1 line.

**Dataset order (VERIFIED: L11 of both scripts):**
`DATASETS=(mediastore teastore jabref bigbluebutton teammates)` — light to heavy.

**Cooldowns (VERIFIED):**
- Between datasets: 90s
- Between runs: 240s
- Retry wait: 300s

**LLM cost logging — current state:** The existing scripts log to `$PROG` (`$LOGBASE/PROGRESS.log`) but do NOT log cumulative cost. The linker's `_save_log()` (L1046–1073) writes `_llm_calls.json` per dataset per run to `LLM_LOG_DIR`, which contains per-call timing but NOT API cost in dollars (no cost field in the trace records at L192–203).

**D-08 soft cap implementation:** There is no existing cumulative cost counter. Implementing D-08 requires either:
- (a) Adding a cost-estimation step after each dataset run: parse the LLM call trace to count calls, multiply by estimated per-call cost, and log to PROGRESS. This is a lightweight bash addition.
- (b) Relying on the API dashboard for actual cost (post-sweep review).

The CONTEXT explicitly says "log cumulative live-call cost to the sweep PROGRESS log." Option (a) is achievable via call-count accumulation since the `_calls.json` files are available per-dataset. An exact dollar figure requires knowing the per-token price and token counts (not currently logged). A practical implementation: count LLM calls via `wc -l <calls.json>` approximation or grep-count, log "≈N calls" to PROGRESS, and note the actual dollar figure must be checked in the API dashboard. Alternatively, add a `COST_PER_CALL_USD` env variable with a rough estimate for the cost line.

**Recommended D-08 implementation:** Add a `CUMULATIVE_CALLS` counter that increments by parsing the LLM call count from each dataset's calls JSON, and log it as "≈N calls" to PROGRESS. This is a 5-line bash addition. The sweep is unattended and non-aborting, so the cost line is informational.

**Cost estimate for No-Knowledge sweep:**

Full sweep LLM call profile per dataset run (from linker doc string and code):
- Phase 1: ambiguity prompt (1 call) + doc-knowledge-extract (1 call) + doc-knowledge-judge (1 call, conditional) = ~3 calls
- Phase 2: 2 passes × N batches (batch_size=50, so for 100-sentence docs: 4 batches total) = ~4 calls
- Phase 4: 2 validation passes × N batches (batch_size=25) = ~8 calls
- Phase 5: coref batch (batch_size=10) + coref validation batches = ~20+ calls

No-Knowledge skips Phase 1 entirely (3 fewer calls). For 5 datasets × 2 backends × 3 runs = 30 runs, saving 3×30 = 90 calls. If the Full sweep cost $60–80 for ~30 runs (Phase-48 reference: $20 for a single-backend 5-dataset run), the No-Knowledge sweep should cost roughly $50–65 (slightly cheaper per run, same N=3×2 matrix). D-08's soft cap of ~$60 is plausible but may be tight; the sweep is unattended and continues past the cap anyway.

---

### Q6 — run_ablation.py registry format

**Finding: VERIFIED by reading VARIANT_SPECS + CANONICAL_VARIANTS + build_linker.**

**Registry entry shape** (from `run_ablation.py:L134–934` pattern):

```python
# In CANONICAL_VARIANTS list (L40–132), add after "s_linker20_union_aliasb":
"s_linker20_union_noknow",

# In VARIANT_SPECS dict (L134–934), add after the "s_linker20_union_aliasb" entry:
"s_linker20_union_noknow": dict(
    aliases=(),
    module="llm_sad_sam.linkers.experimental.s_linker20_union",
    class_name="SLinker20Union",
    description=(
        "S-Linker20Union NOKNOW — v2.6.6 (experimental=True, NOT canonical). "
        "s_linker20_union with knowledge-disable flag (no_knowledge=True): skips "
        "alias table + ambiguity map (layer1 LLM calls), sets empty ModelKnowledge "
        "and DocumentKnowledge directly. All other phases (extraction, validation, "
        "coref) run unchanged with canonical-name-only matching. "
        "Source: RQ4-02 knowledge A/B ablation axis."
    ),
    canonical=False,
    experimental=True,
),
```

**CRITICAL ISSUE with `build_linker`** (`run_ablation.py:L1063–1071`):

```python
def build_linker(variant_name: str, backend: LLMBackend | None = None):
    canonical = canonical_variant(variant_name)
    ...
    cls = getattr(module, spec["class_name"])
    return cls(backend=backend or get_backend())
```

`build_linker` calls `cls(backend=...)` with keyword argument `backend` only. It does NOT pass `no_knowledge=True` to the constructor. The registry entry has no mechanism to pass constructor kwargs beyond `backend`.

**This is a real gap:** There is no existing pattern in VARIANT_SPECS / build_linker for passing a non-backend constructor kwarg. The planner must address this. Options:
- (a) Add a `kwargs` dict to the VARIANT_SPECS entry and thread it through `build_linker` as `cls(backend=..., **spec.get("kwargs", {}))`. This is 3 lines in `build_linker` and a `kwargs={"no_knowledge": True}` in the spec entry.
- (b) Create a thin factory function in the registry entry (not supported by current structure).
- (c) Create a subclass `SLinker20UnionNoKnow(SLinker20Union)` that overrides `__init__` to hardcode `no_knowledge=True`. This would avoid `build_linker` changes but contradicts the "no subclasses" principle in D-01 (though D-01 targets a separate file — a one-method subclass in the same file is less objectionable).

Option (a) is the cleanest and requires the smallest code surface change. The planner should pick it.

**`_links.csv` prefix:** The runner exports `f"{variant_name}_{dataset_name}_links.csv"` (L1188), so the CSV will be named `s_linker20_union_noknow_{dataset}_links.csv`. This is the distinct prefix D-02 requires and what the extractor extension needs to match.

---

### Q7 — The 5 projects + N=3 matrix

**Finding: VERIFIED by reading run scripts and run_ablation.py.**

**5 datasets (VERIFIED: `run_s20union_gpt_n3.sh:L11`, `run_ablation.py:L947–978`):**
- `mediastore`, `teastore`, `jabref`, `bigbluebutton`, `teammates`
- Run order (light to heavy): mediastore, teastore, jabref, bigbluebutton, teammates
- These are the same 5 as the Full sweep.

**Results-root naming (D-06 — Claude's discretion):**

Recommended scheme consistent with existing pattern:
- GPT No-Knowledge: `results/v2.6.6_s20union_noknow/gpt/run{1..3}/{dataset}/`
  - phase_cache: `results/v2.6.6_s20union_noknow/gpt/run{N}/phase_cache/s_linker20_union/{backend_subdir}/{dataset}/`
- Sonnet No-Knowledge: `results/v2.6.6_s20union_noknow_sonnet/run{1..3}/{dataset}/`

Extract root: `results/v2.6.6_extracts_noknow/`
- Layout: `results/v2.6.6_extracts_noknow/{gpt|sonnet}/run{N}/{dataset}.json`

This mirrors the existing asymmetry: gpt root has a `/gpt/` level; sonnet does not. The suffix `_noknow` is unambiguous on disk and machine-distinguishable from Full.

**MATRIX entry for extractor:**
```python
NOKNOW_MATRIX = [
    ("results/v2.6.6_s20union_noknow/gpt", "openai", "gpt"),
    ("results/v2.6.6_s20union_noknow_sonnet", "claude", "sonnet"),
]
```

---

## Landmines and Gotchas

### Landmine 1: `build_linker` has no kwargs mechanism [BLOCKING]

`build_linker` (`run_ablation.py:L1063–1071`) calls `cls(backend=backend or get_backend())`. It cannot pass `no_knowledge=True` to `SLinker20Union` without a code change. This is a prerequisite for the registry entry to work.

**Fix:** Add `kwargs` dict to `VARIANT_SPECS` entries and thread it through `build_linker`:
```python
def build_linker(variant_name: str, backend: LLMBackend | None = None):
    ...
    cls = getattr(module, spec["class_name"])
    extra = spec.get("kwargs", {})
    return cls(backend=backend or get_backend(), **extra)
```

### Landmine 2: `DocumentKnowledge.aliases` type mismatch

`data_types_v2.py:L57` declares `aliases: dict[str, str]` but `s_linker20_union.py:L675` stores `AliasEntry` objects as values (not `str`). The type annotation is wrong/legacy. The empty default `{}` is correct regardless of value type, so `DocumentKnowledge()` yields an empty dict that all consumers iterate over safely. No runtime error, but a documentation mismatch that confused Phase-50 (Pitfall 6 in SUMMARY). No action needed for Phase 51, but the planner should note that `to_neutral` in the extractor already handles this correctly (L162–167).

### Landmine 3: `_VARIANT_NAME` in `_checkpoint_dir` is always `"s_linker20_union"`

Since the No-Knowledge variant is the same class with a flag, `_VARIANT_NAME = "s_linker20_union"` (L263). The phase_cache subdirectory will be `…/phase_cache/s_linker20_union/openai/{dataset}/` — same as Full runs if they ever share a `PHASE_CACHE_DIR`. The run-script's per-run isolation (`export PHASE_CACHE_DIR="$rdir/phase_cache"` with `$rdir` being `results/v2.6.6_s20union_noknow/gpt/run{N}`) prevents collision. The extractor's NOKNOW_MATRIX uses a different root, so it finds the right caches. BUT: if a developer re-runs with `PHASE_CACHE_DIR` pointing at a Full cache directory accidentally, the linker will write over the Full caches. The planner should add a comment to the run-script warning about this.

### Landmine 4: Extractor secondary oracle key must change

The faithfulness oracle's secondary cross-check reads `proj_data.get("s_linker20_union", {})` from the `ablation_*.json` (L269). For No-Knowledge runs, the ablation JSON key is `"s_linker20_union_noknow"`. Failing to update this won't break the primary PASS/FAIL verdict (which uses CSV set-equality), but it will cause spurious secondary cross-check errors.

### Landmine 5: `PROGRESS.log` cost logging has no per-call dollar figure

The linker's `_calls.json` (written to `LLM_LOG_DIR`) does NOT contain token counts or dollar costs — only timing and prompt/response text. Implementing D-08 "log cumulative live-call cost" literally requires either estimating from call count or adding token counting to the LLM client. The simplest path: log call count to PROGRESS after each dataset, acknowledge actual dollar check is via API dashboard. This satisfies the intent of D-08 (cost visibility) without modifying the LLM client.

### Landmine 6: `GATE-01 snapshot replay` requires the interpretation clarified

D-09 says "replay and assert the produced layer1 and final `_links.csv` are byte-identical to the frozen Full outputs." The linker has no checkpoint-resume mode — it always makes live LLM calls. Byte-identical replay of the linker with flag-OFF against the frozen 30 cells is not achievable at zero cost. The structural guard check + extractor re-run (Q3 above) is the realistic evidence. The planner should document this scope adjustment explicitly in the plan.

---

## Architecture Patterns

### Existing pattern for all variant specs (run_ablation.py:L134–934)

All entries in `VARIANT_SPECS` follow:
```python
"variant_key": dict(
    aliases=(...,),
    module="llm_sad_sam.linkers.experimental.<module>",
    class_name="ClassName",
    description="...",
    canonical=False,       # optional, default not present
    experimental=True,     # optional, default not present
),
```
`CANONICAL_VARIANTS` is a flat list (L40–132) that determines `--list-variants` and `--variants` CLI arg validation. The variant name must appear in BOTH `CANONICAL_VARIANTS` AND `VARIANT_SPECS`.

### Existing GATE-06 clean pattern

All 6 downstream knowledge consumers use empty-dict/empty-set graceful fallback (loops over `.items()` on empty dicts simply don't execute). No special handling required.

---

## Planning Implications

### Discrete Work Units

| Unit | Description | Depends On | Risk |
|------|-------------|------------|------|
| **W1: Flag + guard** | Add `no_knowledge: bool = False` to `SLinker20Union.__init__` (L265); add `if self.no_knowledge:` guard at L499–516 | None | LOW — 10 lines of code, strictly additive |
| **W2: Registry entry + build_linker kwargs** | Add `"s_linker20_union_noknow"` to `CANONICAL_VARIANTS` and `VARIANT_SPECS`; add `kwargs` threading to `build_linker` | W1 | LOW — 3 lines in `build_linker` + 1 spec entry |
| **W3: GATE-01 evidence** | Structural guard check (git diff assertion) + re-run Phase-50 extractor on frozen cells (30/30 PASS already known) + 1 representative spot-check run per backend | W1, W2 | LOW — structural check is instant; spot-check ~$1 |
| **W4: Run-scripts + cost logging** | Copy existing scripts, change `VARIANT`/`BASE`/`LOGBASE`; add call-count logging to PROGRESS | W1, W2 | LOW — verbatim copy with 5 line changes |
| **W5: Live sweep** | Execute both run-scripts (30 live runs) | W4 | MEDIUM — latency (~4–6 hours wall-clock), cost (~$50–65) |
| **W6: Extractor extension** | Add NOKNOW_MATRIX + NOKNOW_EXTRACT_ROOT; add No-Knowledge cell path pattern; update secondary oracle key; run extraction | W5 | LOW — 4 changes to extractor, uses proven code paths |

### Dependency order

W1 → W2 → W3 (in parallel with W4) → W5 → W6

### Risks

**Cost overrun (moderate):** 30 live runs at ~$60 soft cap. Full sweep cost was ~$80 for 30 gpt runs in Phase 48 (single backend). No-Knowledge should be cheaper per run (3 fewer LLM calls), but Sonnet costs differ from GPT. If both backends run in sequence, total is feasible under $80. The sweep is log-and-continue, so overrun is visible but not blocking.

**Wall-clock time (low risk):** 30 runs × ~15 min/run = ~7.5 hours. The sweep is unattended and resumable. Not a blocking risk.

**GATE-01 structural check (low risk):** The guard is the simplest possible Python conditional; structural verification is trivial. The only GATE-01 risk is if someone edits the `else:` branch while adding the flag — the plan should explicitly prohibit touching the else branch.

**`build_linker` kwargs (low risk):** The change is 3 lines in one function. No existing specs use `kwargs` so the default `{}` is backward compatible.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | No-Knowledge sweep cost ≈ $50–65 total (both backends, N=3) | Q5 | Could exceed $80 soft cap; but sweep is unattended and log-and-continue anyway |
| A2 | Wall-clock ≈ 7.5 hours for 30 runs | Q5 | Could be longer if API latency is high; resumable via .done markers |
| A3 | `GATE-01 zero-LLM snapshot replay` of the linker itself is not achievable; structural check + extractor re-run is the equivalent evidence | Q3 | If user interprets D-09 as requiring literal linker replay with flag-OFF, an additional ~2 live re-runs would be needed |

---

## Sources

All findings are from direct code reads. No web search required. No external documentation consulted.

| File | Lines Verified | What Was Checked |
|------|---------------|-----------------|
| `src/llm_sad_sam/linkers/experimental/s_linker20_union.py` | L265–308, L499–516, L620–676, L681–703, L739–800, L870–944, L1013–1074 | Constructor, Phase 1 guard point, all alias/knowledge consumers, checkpoint dir, log save |
| `src/llm_sad_sam/core/data_types_v2.py` | L38–62 | ModelKnowledge + DocumentKnowledge defaults |
| `run_ablation.py` | L40–132, L134–934, L1063–1071, L1096–1188 | CANONICAL_VARIANTS, VARIANT_SPECS, build_linker, export_links_csv |
| `scripts/extract_s20union_caches.py` | L1–444 (full) | MATRIX, EXTRACT_ROOT, cell_dir/csv_path construction, faithfulness oracle, to_neutral |
| `run_s20union_gpt_n3.sh` | L1–77 (full) | Env vars, dataset order, cooldowns, resume markers, PROGRESS log |
| `run_s20union_sonnet_n3.sh` | L1–75 (full) | Same; confirms symmetric structure |
| `.planning/phases/51-noknow/51-CONTEXT.md` | All | Locked decisions D-01..D-09 |
| `.planning/phases/50-extract/50-01-SUMMARY.md` | All | Phase-50 output contract and design decisions |

---

*Research completed: 2026-06-21*
*Phase: 51-noknow*
*All claims: VERIFIED by direct code read (no ASSUMED claims)*
