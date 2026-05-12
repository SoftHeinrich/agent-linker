# Phase 1: Baseline and Infrastructure - Research

**Researched:** 2026-04-25
**Domain:** Python LLM pipeline infrastructure — checkpoint refactor + variant namespacing + baseline capture + first rule-removal variant
**Confidence:** HIGH (all code referents verified by direct file inspection; no external library claims beyond what Phase 0 research already covered)

---

## Summary

Phase 1 does four independent things and one paperwork item:

1. **Baseline capture (INFRA-01):** Run `s_linker12c` once on the 5-project sweep via the existing `run_ablation.py`. The runner already writes `results/ablation_results/ablation_YYYYMMDD_HHMMSS.json` with per-dataset P/R/F1/TP/FP/FN plus FP/FN source breakdowns — no runner changes needed for baseline capture. Just invoke it and archive the JSON.
2. **diskcache migration (INFRA-03):** Replace 5 private methods (`_prompt_hash`, `_checkpoint_path`, `_load_cached_response`, `_save_cached_response`, `_query_checkpoint`) in `src/llm_sad_sam/llm_client.py` with `diskcache.Cache`. SHA-256 prompt hash stays as the cache key. External surface (`enable_checkpoint`, `query`, `LLMBackend.CHECKPOINT`) unchanged. Add `diskcache>=5.6.1` and `tabulate>=0.9.0` to `pyproject.toml`. No `anthropic` dep — D-01 strikes it.
3. **Per-variant checkpoint namespacing (INFRA-05):** Every `s_linker*` variant has its **own** `_checkpoint_dir()` method containing a hardcoded `"s_linker12c"` (or `"s_linker11c"`, etc.) literal. This is a **separate** cache from the LLM-response cache in llm_client.py (it is the per-phase `*.pkl` pipeline cache in `./results/phase_cache/<variant>/<dataset>/`). The fix: declare `_VARIANT_NAME` as a class-level constant and derive the directory from it. Add a runtime assertion (D-07) that `_checkpoint_dir` contains `_VARIANT_NAME`.
4. **s_linker13a (VAR-01):** Standalone copy of `s_linker12c.py` that (a) deletes `_split_component_name`, (b) replaces the structural candidate-gate in `_enrich_trailing_words` with Spike 001's `fully_llm_driven(knowledge, sentences, components, llm_call)` pattern, (c) sets `_VARIANT_NAME = "s_linker13a"`, (d) is registered in both `CANONICAL_VARIANTS` and `VARIANT_SPECS` in `run_ablation.py`. Gate: hard-tier (teammates + BBB) first; if >1pp regression on either, rework before running the full sweep.
5. **Doc update prerequisite (D-01a):** REQUIREMENTS.md must mark INFRA-02 and INFRA-04 struck; ROADMAP.md §Phase 1 must remove success criterion #2 and drop INFRA-02/INFRA-04 from the requirements list. This is a Phase 1 prerequisite, not a follow-up.

**Primary recommendation:** Plan this as 5 separate tasks (doc-update, baseline, diskcache, _VARIANT_NAME, s_linker13a) in this order. Tasks 2-4 are parallelizable after task 1; task 5 depends only on task 4 (it needs the `_VARIANT_NAME` pattern established so 13a can follow it immediately).

---

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** INFRA-02 and INFRA-04 (SDK migration) are STRUCK. Backend stays on `claude -p` subprocess via `LLMBackend.CLAUDE`. Not deferred — dropped. Reason: Claude CLI exposes no temperature flag and no caller-controlled cache headers, so the tied success criteria cannot be met via CLI. [VERIFIED: `src/llm_sad_sam/llm_client.py` `_query_claude` at L838-898 confirms no `--temperature` or `--cache-control` CLI arguments; only `--model`, `--output-format`, `--dangerously-skip-permissions`, `--resume`, and the prompt are passed.]
- **D-01a:** REQUIREMENTS.md and ROADMAP.md must be updated before Phase 1 plans are drafted — planner surfaces as prerequisite task.
- **D-02:** Single-run baseline for both 12c and 13a. No N-run median. GATE-05 hard-tier-first still applies.
- **D-03:** `_VARIANT_NAME` class constant; `_checkpoint_dir` derived from it; no hardcoded `"s_linker12c"` outside 12c.
- **D-04:** `run_ablation.py` identifies variants by `_VARIANT_NAME`/class, never by string literal.
- **D-05:** Replace 5 SHA-256 methods in `llm_client.py` with `diskcache.Cache`. Key = same SHA-256 of prompt text. External `LLMBackend.CHECKPOINT` API unchanged. No migration of existing on-disk caches.
- **D-06:** `tabulate>=0.9.0` is added to `pyproject.toml` only; not exercised until Phase 5 (PROMO-03).
- **D-07:** Runtime assertion in checkpoint init: if `_checkpoint_dir` doesn't contain `_VARIANT_NAME`, raise immediately. Fail-fast.

### Claude's Discretion
- 13a file structure — full standalone copy of `s_linker12c.py` with `_split_component_name` deleted and `_enrich_trailing_words` replaced per Spike 001's `fully_llm_driven(...)` signature.
- Evidence guardrail strictness — use the light guardrail validated in Spike 001 tests 2+3 (reject alias if word absent from cited sentence; reject if cited sentence contains full component name). No fallback to structural splitter.
- Prompt constant placement in 13a — inline or from `prompts_v2.py`, planner decides. Taboo audit required either way (GATE-04).
- `CANONICAL_VARIANTS` and `VARIANT_SPECS` registration order in `run_ablation.py` — append at end of existing list.

### Deferred Ideas (OUT OF SCOPE)
- Temperature=0.0 enforcement and prompt-caching header assertions (originally INFRA-02) — dropped, not deferred.
- Back-compat migration of existing SHA checkpoint files into diskcache — not needed under single-run baseline.
- Phase-output cache layer above the LLM-response cache.
- `_has_standalone_mention` LLM replacement — deferred to Phase 5 keep-decision + EXT-01.

---

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFRA-01 | Reproducible 12c baseline (per-dataset + macro F1, FP/FN table, JSON in `results/ablation_results/`) | `run_ablation.py` already writes `ablation_YYYYMMDD_HHMMSS.json` with every field named in the requirement (see "Standard Stack" -> Existing runner output schema). No new code needed. |
| INFRA-03 | `diskcache>=5.6.1` and `tabulate>=0.9.0` in `pyproject.toml` | "Standard Stack" table. diskcache replaces 5 private methods in `llm_client.py` (§Code Examples); tabulate is dep-only in Phase 1 per D-06. |
| INFRA-05 | Each variant's `_checkpoint_dir` namespaced per variant | §Architecture Patterns -> `_VARIANT_NAME` pattern. Current state: every variant has its own `_checkpoint_dir()` method with a hardcoded string literal (see "Existing Code Insights"). |
| VAR-01 | `s_linker13a.py` with Spike 001 integrated; `_split_component_name` removed; `_enrich_trailing_words` replaced by LLM+guardrail | §Code Examples -> Spike 001 `fully_llm_driven` signature; §Pitfalls -> Pitfall 1 spike-to-pipeline invalidation. Dual-floor gate = GATE-01. |

INFRA-02 and INFRA-04 are **struck** per D-01 and must be removed from REQUIREMENTS.md before planning begins (prerequisite task).

---

## Architectural Responsibility Map

This phase modifies infrastructure but does not alter the 3-tier DAG. Capability mapping is about **which module owns which fix**:

| Capability | Primary Module | Secondary Module | Rationale |
|------------|----------------|------------------|-----------|
| Baseline capture / metrics | `run_ablation.py` | — | Existing runner already owns dataset loop, metric computation, JSON output |
| LLM-response cache (SHA-256 → JSON) | `src/llm_sad_sam/llm_client.py` | `pyproject.toml` | Cache lives in `LLMClient._query_checkpoint`; diskcache is the new backend library |
| Per-phase pipeline pickle cache (`./results/phase_cache/`) | `src/llm_sad_sam/linkers/experimental/s_linker12c.py` (and all `s_linker*` variants) | — | Each variant has its own `_checkpoint_dir()` method; `_VARIANT_NAME` constant lives at class level |
| Variant registration | `run_ablation.py` | — | `CANONICAL_VARIANTS` list + `VARIANT_SPECS` dict |
| Rule replacement (13a) | `src/llm_sad_sam/linkers/experimental/s_linker13a.py` (new) | `prompts_v2.py` (optional, for new prompt constant) | Standalone variant file — no shared infra modification |
| Requirement/roadmap upkeep | `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` | `.planning/STATE.md` | Doc-only prerequisite (D-01a) |

**Key boundary note:** "checkpoint" refers to **two independent subsystems** in this repo. Mixing them in a plan is the single biggest footgun:

- **LLM-response cache** = `LLMBackend.CHECKPOINT` in `llm_client.py`. Keyed by `SHA256(prompt_text)`. Default dir: `./results/llm_checkpoint/` (from `CHECKPOINT_DIR` env var). This is the INFRA-03 target.
- **Per-phase pipeline pickle cache** = `_checkpoint_dir` method on each `SLinker*` class. Keyed by `<variant_name>/<dataset_basename>/<phase>.pkl`. Default dir: `./results/phase_cache/` (from `PHASE_CACHE_DIR` env var). This is the INFRA-05 target.

The two caches are **not related by code** and do not share a directory tree. D-05 modifies the first; D-03/D-04/D-07 modify the second.

---

## Standard Stack

### Core (already present — do not re-add)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `click` | >=8.1.0 | CLI framework | Already in pyproject.toml [VERIFIED: pyproject.toml L13] |
| `lxml` | >=5.0.0 | PCM XML parsing | Already in pyproject.toml [VERIFIED: pyproject.toml L14] |
| `rapidfuzz` | >=3.0.0 | Fuzzy matching | Already in pyproject.toml [VERIFIED: pyproject.toml L15] |
| `pytest` | >=8.0.0 | Test runner | Already in pyproject.toml dev extra [VERIFIED: pyproject.toml L19] |
| Python stdlib `subprocess`, `json`, `hashlib` | stdlib | Claude CLI invocation, SHA-256 hash | Current `llm_client.py` approach — unchanged [VERIFIED: `llm_client.py` L9, 718-722] |

### New (add to pyproject.toml)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `diskcache` | >=5.6.1 | SQLite-backed prompt-response cache | Replaces the 5 SHA-256 methods in `llm_client.py` (D-05). Concurrent-safe, zero extra config. |
| `tabulate` | >=0.9.0 | Markdown/LaTeX table formatting | Dep-only in Phase 1 per D-06; exercised in Phase 5 PROMO-03. |

### NOT added (D-01 strikes)
| Library | Reason |
|---------|--------|
| `anthropic` | D-01: backend stays on `claude -p` subprocess. The STACK.md research (`.planning/research/STACK.md` L24-38) recommended this but the user decision supersedes. Do not add to pyproject.toml. |

**Installation:**
```bash
pip install -e ".[dev,openai]"
# diskcache, tabulate will be installed as core deps
```

**Version verification:**
```bash
npm view diskcache version  # N/A — Python package; use pip
pip index versions diskcache  # Confirms >=5.6.1 available
pip index versions tabulate   # Confirms >=0.9.0 available
```
[ASSUMED] Versions cited are from Phase 0 STACK.md research (2026-04-21). No reason to expect regression; planner should re-verify against pypi during plan execution (`pip index versions <pkg>`) if the cached artifact shows staleness.

---

## Architecture Patterns

### System Flow — What Phase 1 Touches

```
┌───────────────────────────────────────────────────────────────────────────┐
│  REQUIREMENTS.md + ROADMAP.md  (D-01a prerequisite)                       │
│  Strike INFRA-02/INFRA-04; remove success criterion #2                    │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
                                ▼  (unblocks all downstream)
┌─────────────────────────┐   ┌──────────────────────────────────────────┐
│  run_ablation.py        │   │  pyproject.toml                          │
│  (no code change for    │   │  + diskcache>=5.6.1                      │
│   baseline capture;     │   │  + tabulate>=0.9.0                       │
│   append 13a entry)     │   │  (no anthropic per D-01)                 │
└─────────────┬───────────┘   └───────────────────┬──────────────────────┘
              │                                   │
              │                                   ▼
              │      ┌──────────────────────────────────────────────────┐
              │      │  src/llm_sad_sam/llm_client.py                   │
              │      │  D-05: Replace 5 methods                         │
              │      │  - _prompt_hash (keep — SHA-256 unchanged)       │
              │      │  - _checkpoint_path → delete                     │
              │      │  - _load_cached_response → cache.get(key)        │
              │      │  - _save_cached_response → cache[key] = ...      │
              │      │  - _query_checkpoint → uses diskcache.Cache      │
              │      └───────────────────────┬──────────────────────────┘
              │                              │
              ▼                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│  src/llm_sad_sam/linkers/experimental/s_linker12c.py                   │
│  D-03/D-04/D-07: _VARIANT_NAME pattern                                 │
│    - Add `_VARIANT_NAME = "s_linker12c"` class constant                │
│    - `_checkpoint_dir` uses self._VARIANT_NAME (no hardcoded string)   │
│    - `_save_log` uses self._VARIANT_NAME                               │
│    - Add runtime assertion in __init__ or _checkpoint_dir              │
└───────────────────────────────┬────────────────────────────────────────┘
                                │  copy-then-modify
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  src/llm_sad_sam/linkers/experimental/s_linker13a.py  (new)            │
│  VAR-01: Standalone variant                                            │
│    - _VARIANT_NAME = "s_linker13a"                                     │
│    - Delete _split_component_name (staticmethod at L292-298)           │
│    - Replace _enrich_trailing_words body with Spike 001 pattern        │
│    - Append to CANONICAL_VARIANTS + VARIANT_SPECS in run_ablation.py   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Baseline + 13a evaluation (hard-tier first, then 5-project sweep)    │
│  Output: results/ablation_results/ablation_YYYYMMDD_HHMMSS.json        │
└────────────────────────────────────────────────────────────────────────┘
```

### Pattern 1: `_VARIANT_NAME` Class Constant (INFRA-05)

**What:** Each `SLinker*` class declares a class-level constant `_VARIANT_NAME` holding its canonical string identity. `_checkpoint_dir` and `_save_log` derive paths from it.

**When to use:** Every variant file (including the in-place refactor of `s_linker12c.py`) and the new `s_linker13a.py`.

**Before (current s_linker12c.py L1181-1210) [VERIFIED: direct read]:**
```python
def _checkpoint_dir(self, text_path):
    cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
    ds = os.path.splitext(os.path.basename(text_path))[0]
    d = os.path.join(cache_dir, "s_linker12c", ds)    # HARDCODED
    os.makedirs(d, exist_ok=True)
    return d

def _save_log(self, text_path):
    log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
    os.makedirs(log_dir, exist_ok=True)
    ds = os.path.splitext(os.path.basename(text_path))[0]
    path = os.path.join(log_dir, f"s_linker12c_{ds}_{...}.json")  # HARDCODED
```

**After (recommended):**
```python
class SLinker12c:
    _VARIANT_NAME = "s_linker12c"   # class constant

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, ds)
        # Runtime assertion (D-07):
        assert self._VARIANT_NAME in d, (
            f"_checkpoint_dir must contain _VARIANT_NAME "
            f"('{self._VARIANT_NAME}' not in '{d}')"
        )
        os.makedirs(d, exist_ok=True)
        return d

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        path = os.path.join(log_dir,
            f"{self._VARIANT_NAME}_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        ...
```

**Design note on D-07:** CONTEXT.md says the assertion goes "in the checkpoint constructor". s_linker12c's `_checkpoint_dir` is per-call (takes `text_path`), not a constructor. Placing the assertion inside `_checkpoint_dir()` itself — as shown — satisfies D-07's intent (fail-fast if the directory doesn't include `_VARIANT_NAME`) and is the obvious site given the existing method shape. An alternative is to add `__init_subclass__` that walks the method body for the literal — that is brittle and not recommended. Use the in-method assertion.

**Note on scope:** D-07 explicitly covers "every variant that touches the CHECKPOINT backend". The per-phase pickle cache and the LLM-response CHECKPOINT are different subsystems (§Architectural Responsibility Map). The planner should add the assertion to `_checkpoint_dir` in **every** variant file touched in Phase 1 — which is `s_linker12c.py` (refactor) and `s_linker13a.py` (new). Pre-12c variants are not touched this phase (their hardcoded strings are irrelevant to the Phase 1 scope because those variants aren't being re-run; if run later, the planner in that phase can decide whether to retrofit).

### Pattern 2: diskcache drop-in replacement (INFRA-03)

**What:** Replace file-based SHA-256 cache with `diskcache.Cache`. Key scheme (SHA-256 of prompt text) unchanged so existing caches are theoretically readable — but D-05 states old caches are NOT migrated under the single-run baseline.

**When to use:** Once, in `llm_client.py`, for the `LLMBackend.CHECKPOINT` code path.

**Before (5 methods — [VERIFIED: `llm_client.py` L718-805]):**
- `_prompt_hash(prompt)` → SHA-256 hex of prompt (static)
- `_checkpoint_path(prompt)` → `_checkpoint_dir / f"{hash}.json"`
- `_load_cached_response(path)` → read JSON, rebuild `LLMResponse`
- `_save_cached_response(path, response)` → write JSON
- `_query_checkpoint(prompt, timeout, max_retries)` → cache lookup; delegate to fallback; save

**After (sketch; planner owns final shape):**
```python
import diskcache

class LLMClient:
    def __init__(self, ...):
        ...
        self._cache: Optional[diskcache.Cache] = None
        if self.backend == LLMBackend.CHECKPOINT:
            self._cache = diskcache.Cache(str(self._checkpoint_dir))
            # _checkpoint_dir still resolved the same way (CHECKPOINT_DIR env
            # var, default ./results/llm_checkpoint) — only storage layer changes

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        # KEEP — same SHA-256 key scheme per D-05
        import hashlib
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _query_checkpoint(self, prompt, timeout, max_retries):
        key = self._prompt_hash(prompt)
        cached_dict = self._cache.get(key)
        if cached_dict is not None:
            response = _llm_response_from_dict(cached_dict)
            response.latency_ms = 0
            if self.enable_logging:
                self._log_request(prompt, response, 0)
            return response
        # Cache miss — delegate to fallback (unchanged logic)
        original = self.backend
        self.backend = self._checkpoint_fallback
        try:
            response = self.query(prompt, timeout=timeout, max_retries=max_retries)
        finally:
            self.backend = original
        if response.success:
            self._cache[key] = _llm_response_to_dict(response)
        return response
```

`_checkpoint_path`, `_load_cached_response`, `_save_cached_response` are deleted. `_llm_response_{to,from}_dict` helpers replace the JSON shape conversions (they handle the nested `token_usage` dict; see `_save_cached_response` at L757-772 and `_load_cached_response` at L728-755 for the current dict shape).

**Also update `enable_checkpoint` (L591-618):** the method reassigns `self._checkpoint_dir` and currently does `self._checkpoint_dir.mkdir(parents=True, exist_ok=True)`. After migration, it must close any existing `self._cache` and open a new `diskcache.Cache(str(new_dir))`.

### Pattern 3: Spike 001 integration for `s_linker13a` (VAR-01)

**What:** Replace the structural candidate-gate loop in `_enrich_trailing_words` (s_linker12c.py L420-482) with a single LLM call that does discovery + verification in one pass, guarded by an evidence-sentence post-check.

**When to use:** Inside `s_linker13a.py` only. `s_linker12c.py` is read-only except for the `_VARIANT_NAME` refactor.

**Before — the structural gate (`s_linker12c.py` L427-443 [VERIFIED]):**
```python
existing_lower = {a.lower() for a in knowledge.aliases}
candidates = []
for comp in components:
    parts = self._split_component_name(comp.name)          # <-- to be deleted
    if len(parts) < 2:
        continue
    last_word = parts[-1]
    last_lower = last_word.lower()
    if any(c.name != comp.name and c.name.lower().endswith(last_lower)
           for c in components):
        continue
    if last_lower in existing_lower:
        continue
    full_lower = comp.name.lower()
    if any(last_lower in s.text.lower() and full_lower not in s.text.lower()
           for s in sentences):
        candidates.append((last_word, comp.name))
# ... then LLM verify batch over `candidates` ...
```

**After (Spike 001 pattern, adapted — [CITED: `.planning/spikes/001-llm-trailing-words/spike.py` L121-153]):**
```python
def _enrich_trailing_words(self, knowledge, sentences, components):
    prompt = LLM_ONLY_PROMPT.format(
        components=", ".join(c.name for c in components),
        document="\n".join(f"S{s.number}: {s.text}" for s in sentences),
    )
    data = self.llm.extract_json(self.llm.query(prompt, timeout=300)) or {}

    comp_set = {c.name for c in components}
    sent_map = {s.number: s.text for s in sentences}
    existing_lower = {a.lower() for a in knowledge.aliases}

    for entry in data.get("aliases", []):
        alias = entry.get("alias", "").strip()
        comp = entry.get("component", "").strip()
        ev = entry.get("evidence_sentence")
        if not alias or comp not in comp_set:
            continue
        if alias.lower() in existing_lower:
            continue
        ev_text = sent_map.get(ev, "")
        if not ev_text:
            continue
        # Light guardrail (Spike 001 tests 2+3)
        if alias.lower() not in ev_text.lower():
            continue
        if comp.lower() in ev_text.lower():
            continue
        knowledge.aliases[alias] = comp
        print(f"    Alias (trailing-word, LLM): {alias} -> {comp}")
```

The `_split_component_name` static method at L292-298 is **deleted** (its only caller is `_enrich_trailing_words` — verified by grep: 2 hits, both in `_enrich_trailing_words`).

**Prompt constant placement:** CONTEXT.md lets planner decide — inline in `s_linker13a.py` or export from `prompts_v2.py`. Recommendation: **inline** to stay consistent with the standalone-file convention (MEMORY.md: "User prefers standalone linker files (duplicate code intentionally, not inheritance chains)"). A named constant `LLM_ONLY_TRAILING_WORD_PROMPT` at module top, verbatim from Spike 001's `LLM_ONLY_PROMPT` after a taboo audit.

**GATE-04 taboo audit for `LLM_ONLY_PROMPT`:** Spike 001's prompt uses "OrderProcessor" as the only concrete example (L43-45). "Order" is on BENCHMARK_TABOO.md Universal Taboo (TeaStore: `OrderBasedRecommender`). **This is leakage and must be replaced** before promotion. Safe substitutes from BENCHMARK_TABOO.md safe-domain list: `TaskScheduler`, `FileLexer`, `InvoiceHandler`. Recommended replacement: `TaskScheduler` / `Scheduler` (the example sentence becomes "Component `TaskScheduler`. Document says '...the Scheduler validates each item...'."). The rest of Spike 001's test fixtures use `TaskDispatcher`, `AuthService`, `QueueBroker`, `MediaPlayer` — of these, `AuthService` contains "Auth" (TeaStore taboo), `MediaPlayer` contains "Media" (MediaStore keyword), and `Dispatcher`/`Broker` are the safe-domain examples from BENCHMARK_TABOO.md. Tests are not in the prompt — but if the planner copies spike test names into prompt examples, audit them. **Action for plan:** GATE-04 checklist item specifically calls out these substitutions.

### Pattern 4: Registering `s_linker13a` in `run_ablation.py`

**What:** Append two entries. No other runner changes.

**Where (run_ablation.py L40-73 [VERIFIED]):** `CANONICAL_VARIANTS` is an ordered list; `s_linker12e` is currently the last entry. D-04 says append at end.

```python
CANONICAL_VARIANTS = [
    ...
    "s_linker12e",
    "s_linker13a",   # append
]

VARIANT_SPECS = {
    ...
    "s_linker12e": dict(...),
    "s_linker13a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13a",
        class_name="SLinker13a",
        description="S-Linker13a: 12c - _split_component_name (Spike 001 LLM trailing-word)",
    ),
}
```

**D-04 re-read:** "passes/consumes variant identity through `_VARIANT_NAME` (or reads it off the class), never by string literal." The current `VARIANT_SPECS` keys are string literals (`"s_linker13a"`). This is not a conflict — D-04 is about _within-pipeline_ identity (e.g., variant asking itself "what is my name?"). The runner's `VARIANT_SPECS` dict keys are external catalog identifiers. Confirm with the planner that appending string keys to `VARIANT_SPECS` is compatible with D-04's intent; the spirit is "no hardcoded `'s_linker12c'` drifting through variant code paths," not "no strings anywhere in the harness."

---

### Anti-Patterns to Avoid

- **Adding `anthropic` to pyproject.toml anyway** — D-01 explicitly strikes it. Even adding it as an unused dep creates confusion and signals "SDK migration coming" when user said "not coming."
- **Migrating existing SHA-256 cache files into diskcache** — D-05 says no. Single-run baseline regenerates them.
- **Putting the D-07 assertion in `LLMClient.__init__` or `enable_checkpoint`** — wrong target. The per-phase pickle cache (INFRA-05 scope) lives in `s_linker*.py`'s `_checkpoint_dir` method, not in `LLMClient`. The LLM-response CHECKPOINT dir in `LLMClient` is namespaced by env var (`CHECKPOINT_DIR`), not by variant name, and has no `_VARIANT_NAME` to check against. The assertion belongs in `SLinker*._checkpoint_dir` methods.
- **Modifying `prompts_v2.py` to add Spike 001's prompt** — Phase 1 scope lets planner decide; the cleaner, standalone-file-convention choice is to inline in `s_linker13a.py`. If planner chooses `prompts_v2.py`, the taboo audit must run on the new constant before committing.
- **"Quickly" patching pre-12c variants to use `_VARIANT_NAME`** — out of scope. The 20+ other `s_linker*.py` files still have hardcoded strings (grep shows them all). Phase 1 only namespaces 12c (in place) and 13a (new). Retrofitting pre-12c variants is unnecessary risk and unrelated to the phase goal.
- **Running the full 5-project sweep on 13a before hard-tier gate** — violates GATE-05 and D-02. Hard tier = teammates + bigbluebutton first; >1pp regression on either → rework.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Concurrent-safe disk cache | File locking around JSON files | `diskcache.Cache` | Already the D-05 decision; SQLite backend handles concurrency; zero config. |
| Prompt hashing | Anything other than SHA-256 | stdlib `hashlib.sha256` (existing `_prompt_hash` stays) | D-05 explicitly preserves the SHA-256 key scheme. |
| Trailing-word structural gate | Regex splitter + candidate filter + uniqueness check | Spike 001 `fully_llm_driven` pattern (single LLM call + evidence guardrail) | Spike 001 VALIDATED; 4 tests pass including parity. Rebuilding the structural gate defeats VAR-01. |
| Baseline metric computation | Any custom scoring loop | `run_ablation.py`'s existing `eval_metrics()` + `run_variant()` | The runner already produces per-dataset P/R/F1/TP/FP/FN, source breakdowns, FP/FN details, CSV export, JSON summary. Reuse verbatim. |
| Variant registry | Any bespoke config file or plugin system | `CANONICAL_VARIANTS` list + `VARIANT_SPECS` dict | Project pattern (run_ablation.py L40-267). Append-only per D-04. |

---

## Runtime State Inventory

Phase 1 is code/config/doc changes — not a rename, refactor-rename, or data migration. Runtime state categories:

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | **Existing pickle files in `./results/phase_cache/s_linker12c/<dataset>/*.pkl`** from prior 12c runs. These are keyed by the hardcoded `"s_linker12c"` path — after the D-03 refactor the path is still `s_linker12c` (because `_VARIANT_NAME = "s_linker12c"`), so existing pickles remain valid. **Existing JSON files in `./results/llm_checkpoint/*.json`** from prior CHECKPOINT backend use. These are NOT migrated per D-05; diskcache will create a fresh SQLite db and the JSONs become orphaned. | No data migration required. Old pickles still work for 12c; old CHECKPOINT JSONs are abandoned (D-05 explicit); 13a generates its own `./results/phase_cache/s_linker13a/*` tree. |
| Live service config | None — no external services. | None. |
| OS-registered state | None — no OS-level registrations. | None. |
| Secrets / env vars | `PHASE_CACHE_DIR`, `LLM_LOG_DIR`, `CHECKPOINT_DIR`, `CHECKPOINT_FALLBACK`, `CHECKPOINT_FALLBACK_MODEL`, `OPENAI_API_KEY`, `CLAUDE_MODEL`, `OPENAI_MODEL_NAME`, `LLM_BACKEND`, `LLM_SESSION_DIR`, `LLM_LOG_DIR` — none renamed. | None — env var names are untouched. |
| Build artifacts | `llm_sad_sam_agent.egg-info/` from editable install. Re-running `pip install -e ".[dev,openai]"` after pyproject changes is required for diskcache + tabulate to be importable. | Plan task for diskcache migration must include `pip install -e ".[dev,openai]"` in its actions so the new deps resolve. |

**Canonical question:** *After every file in the repo is updated, what runtime systems still have the old string cached, stored, or registered?*

**Answer:** The `./results/llm_checkpoint/*.json` files from prior CHECKPOINT backend runs are the only stale artifacts. D-05 chooses to abandon them (no migration). The plan should NOT include a step to delete them — they're harmless and reverting to the file cache is theoretically possible if diskcache breaks (abandoning them is non-destructive). Document in task notes: "pre-migration JSON cache files in `./results/llm_checkpoint/` are orphaned and ignored by diskcache; not deleted."

---

## Common Pitfalls

Most items below are drawn from `.planning/research/PITFALLS.md`. Only those that actively bite Phase 1 are reproduced; full catalog lives in the research doc.

### Pitfall 1: Spike-to-Pipeline Invalidation (VAR-01 specific)
**What goes wrong:** Spike 001's 4 tests pass on 4-component curated fixtures. In-pipeline, `_enrich_trailing_words` runs on 5 projects with 8-25 components each, full document bodies, and existing `knowledge.aliases` already populated with discovered abbreviations/synonyms. The LLM may return aliases that collide with existing entries (silently ignored), hallucinate aliases whose "evidence_sentence" number doesn't exist in the document, or produce 20+ aliases on a long document (larger than spike fixtures covered).
**Why it happens:** Scale + heterogeneous input not covered by isolated spike tests.
**How to avoid:** Hard-tier first (teammates + bigbluebutton), then full sweep. GATE-01 dual floor: macro F1 ≥ 93% AND no dataset >2pp below 12c per-dataset baseline. If hard tier regresses >1pp, rework before full sweep (GATE-05).
**Warning signs:** Spike test passes but first full-benchmark run shows ≥1pp regression on mediastore or jabref (the datasets NOT in the hard tier). LLM response logs show truncated JSON or sentence numbers > `max(sentence.number)`.

### Pitfall 2: Benchmark Leakage in LLM_ONLY_PROMPT (GATE-04)
**What goes wrong:** Spike 001's prompt example uses "OrderProcessor" → "Order" is on BENCHMARK_TABOO.md Universal Taboo (TeaStore). Other spike test-fixture names ("AuthService", "MediaPlayer") contain taboo tokens ("Auth", "Media"). If copied into the production prompt, this is benchmark leakage.
**Why it happens:** Spike fixtures were designed for isolated testing, not for production prompt audit.
**How to avoid:** Replace "OrderProcessor" with a safe-domain name from BENCHMARK_TABOO.md: `TaskScheduler`, `FileLexer`, or `InvoiceHandler`. Recommended: `TaskScheduler`/`Scheduler` (aligns with compiler/OS domain already used in other prompts). Plan task must include an explicit taboo audit step (GATE-04).
**Warning signs:** `grep -f BENCHMARK_TABOO.md s_linker13a.py` returns hits.

### Pitfall 3: Two-Caches Confusion (INFRA-03 vs INFRA-05)
**What goes wrong:** Planner conflates "checkpoint" meanings. Task for diskcache migration accidentally edits `s_linker12c._checkpoint_dir` (pickle cache). Task for `_VARIANT_NAME` namespacing accidentally edits `LLMClient._checkpoint_dir` (LLM-response cache).
**Why it happens:** Both subsystems use the word "checkpoint" and both have a `_checkpoint_dir` attribute. They share no code.
**How to avoid:** Plan tasks name them distinctly. Suggested task names:
  - "INFRA-03: Migrate LLM-response cache in llm_client.py to diskcache"
  - "INFRA-05: Namespace per-phase pickle cache via `_VARIANT_NAME` in s_linker12c"
**Warning signs:** A diff to `llm_client.py` that mentions `_VARIANT_NAME`, or a diff to `s_linker12c.py` that imports `diskcache`.

### Pitfall 4: Variance Masking Single-Run Regression (13a hard-tier gate)
**What goes wrong:** Single-run baseline (D-02) means no averaging. Claude has ~1-3 link run-to-run variance. If 13a holds within 0.3pp of 12c on teammates/BBB, the "no regression" call may be noise.
**Why it happens:** Memory entry "LLM Variance (Critical Finding)" — same model, different day, different Phase 1/3 behavior.
**How to avoid:** The 1pp hard-tier gate (GATE-05) already accounts for this; 1pp > typical variance envelope. But if a regression is marginal (0.5-1.0pp), plan the follow-up explicitly: either accept (document as marginal, not promotion-blocking) or re-run once to confirm (outside phase 1's "single run" rule, because the rule is for *capture*, not *gate evaluation*).
**Warning signs:** 13a within 0.3pp of 12c on one hard-tier dataset and +1.5pp on the other — reconcile before calling it a clean pass.

### Pitfall 5: `prompts_v2.py` Accidentally Modified (Shared-State Contamination)
**What goes wrong:** Planner places Spike 001's `LLM_ONLY_TRAILING_WORD_PROMPT` in `prompts_v2.py`. `s_linker12c`'s behavior doesn't change (it doesn't import the new constant) but the import contract of `prompts_v2.py` has silently widened.
**Why it happens:** "Shared helpers" convenience bias.
**How to avoid:** Inline the prompt in `s_linker13a.py` — matches the standalone-file convention. If planner chooses `prompts_v2.py`, the file diff must be audited to confirm zero changes to existing constants (only additions).
**Warning signs:** `git diff prompts_v2.py` shows changes to existing constants (not just additions).

---

## Code Examples

### Existing runner output schema (INFRA-01 is satisfied by this — no code changes)
[VERIFIED: `run_ablation.py` L530-544, L678-681]

Per-variant result dict already written to JSON:
```python
{
    "variant": "s_linker12c",
    "P": 0.952, "R": 0.933, "F1": 0.942,
    "tp": 45, "fp": 2, "fn": 3,
    "n_links": 47,
    "time": 123.4,
    "sources": {"seed": 30, "entity": 12, "coref": 5},    # source breakdown
    "fp_by_source": {"seed": 1, "entity": 1},
    "fp_details": [{"sentence": 42, "component": "Foo", ...}],
    "fn_details": [{"sentence": 17, "component": "Bar", ...}],
}
```

JSON path: `results/ablation_results/ablation_{YYYYMMDD}_{HHMMSS}.json` (L678-680).
Per-variant/dataset CSV also exported: `results/ablation_results/{variant}_{dataset}_links.csv` (L521).

To capture the baseline:
```bash
python run_ablation.py --variants s_linker12c --datasets mediastore teastore teammates bigbluebutton jabref
```

### Current `_query_checkpoint` (to be migrated)
[VERIFIED: `llm_client.py` L774-805]

```python
def _query_checkpoint(self, prompt: str, timeout: int, max_retries: int) -> LLMResponse:
    cache_path = self._checkpoint_path(prompt)    # → delete
    cached = self._load_cached_response(cache_path)   # → cache.get(key)
    if cached is not None:
        cached.latency_ms = 0
        if self.enable_logging:
            self._log_request(prompt, cached, 0)
        return cached
    original_backend = self.backend
    self.backend = self._checkpoint_fallback
    try:
        response = self.query(prompt, timeout=timeout, max_retries=max_retries)
    finally:
        self.backend = original_backend
    if response.success:
        self._save_cached_response(cache_path, response)   # → cache[key] = ...
    return response
```

### Spike 001 `fully_llm_driven` signature
[CITED: `.planning/spikes/001-llm-trailing-words/spike.py` L121-153]

Signature: `fully_llm_driven(knowledge, sentences, components, llm_call) -> knowledge`

The `llm_call` parameter has the same contract as `self.llm.extract_json(self.llm.query(...))` → returns a parsed JSON dict or `None`. In the pipeline integration, the adapter is:

```python
def _llm_call(prompt):
    return self.llm.extract_json(self.llm.query(prompt, timeout=300)) or {}
```

### Baseline + 13a evaluation commands

```bash
# INFRA-01: capture 12c baseline (full sweep)
python run_ablation.py --variants s_linker12c

# VAR-01: 13a hard-tier gate (GATE-05)
python run_ablation.py --variants s_linker13a --datasets teammates bigbluebutton

# If hard tier passes (>=12c - 1pp on both): full sweep
python run_ablation.py --variants s_linker12c s_linker13a

# The single-command sweep runs 12c + 13a side-by-side for ablation comparison
```

---

## State of the Art

Phase 0 STACK.md research already surveyed this domain. Relevant currency checks:

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `claude -p` subprocess | Direct Anthropic SDK (`client.messages.create` / `.parse`) | SDK GA late 2025 | **IGNORED per D-01** — user keeps subprocess path. |
| SHA-256 JSON file cache | `diskcache.Cache` (SQLite-backed) | — | **D-05 migration** — zero interface change, concurrent-safe. |
| Ad-hoc markdown reports | `tabulate` Markdown + LaTeX export | — | **Phase 5 only** — Phase 1 just adds dep. |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `diskcache>=5.6.1` and `tabulate>=0.9.0` are still current versions on PyPI | Standard Stack | LOW. Phase 0 STACK.md researched 2026-04-21; 4-day gap. Planner should `pip index versions` during execution, but no functional risk — these are mature, stable libraries. |
| A2 | D-07's assertion location (`_checkpoint_dir` method body, not `__init__`) matches D-07's intent | Pattern 1 | RESOLVED 2026-05-08 by user — in-method placement (inside `_checkpoint_dir()` body) is correct. See Open Questions Q1. |
| A3 | "Every variant that touches the CHECKPOINT backend" (D-07 wording) means the per-phase pickle cache, not the LLM-response CHECKPOINT | Pattern 1, Anti-patterns | MEDIUM. "CHECKPOINT backend" literally matches `LLMBackend.CHECKPOINT` (LLM-response cache). But the LLM-response cache dir is env-var controlled, not variant-named — there's nothing `_VARIANT_NAME`-shaped to assert against there. So the pragmatic reading is "per-phase pickle cache in variant files". **Recommend user confirmation.** |
| A4 | Spike 001's `LLM_ONLY_PROMPT` with "OrderProcessor" example would fail GATE-04 taboo audit | Pitfall 2, Pattern 3 | LOW. BENCHMARK_TABOO.md line 53 lists "order" as Universal Taboo (TeaStore `OrderBasedRecommender`). Replacement is mechanical. |
| A5 | Appending to `VARIANT_SPECS` dict with string key `"s_linker13a"` is compatible with D-04's "never by string literal" constraint | Pattern 4 | LOW. D-04's spirit reads as "no hardcoded variant identity *inside variant code paths* or runtime logic"; the harness catalog being keyed by strings is external. Still worth confirming with user. |
| A6 | Old `./results/llm_checkpoint/*.json` files are safe to abandon (orphaned but harmless) | Runtime State Inventory | LOW. D-05 explicitly says no migration; diskcache writes to the same directory with different file names (SQLite db files), so they coexist without collision. |

---

## Open Questions (RESOLVED)

1. **D-07 assertion site — `__init__` or `_checkpoint_dir()` method?**
   - What we know: CONTEXT.md says "in the checkpoint constructor"; `_checkpoint_dir` is a per-call method taking `text_path`, not a constructor; asserting in `_checkpoint_dir` fails fast at first use.
   - What's unclear: Does "constructor" literally mean `__init__` (requiring storing `_checkpoint_dir_root` at construction and asserting there), or "the init path of the checkpoint subsystem" (= the `_checkpoint_dir` method is a reasonable reading)?
   - Recommendation: Planner proposes in-method assertion; planner-checker confirms with user if unclear. Either reading is cheap to implement — the difference is just where the failure message surfaces.
   - **RESOLVED 2026-05-08:** User confirms in-method placement. The assertion lives inside the `_checkpoint_dir()` method body (which is the per-call site for the per-variant pickle cache). The `LLMBackend.CHECKPOINT` path in `LLMClient` is namespaced separately by env var and is not the target of D-07. Plan 03's existing placement is correct.

2. **Prompt constant placement — inline vs `prompts_v2.py`?**
   - What we know: CONTEXT.md marks as Claude's discretion. MEMORY.md says standalone-file preference.
   - What's unclear: Is adding a new constant to `prompts_v2.py` considered a violation of "standalone"? Other 12x variants import from `prompts_v2.py` currently.
   - Recommendation: Inline in `s_linker13a.py` as `LLM_ONLY_TRAILING_WORD_PROMPT` module constant — safest under standalone convention.
   - **RESOLVED 2026-05-08:** Inline in `s_linker13a.py` (Plan 05's existing choice — Claude's discretion per CONTEXT.md). The taboo-audited `LLM_ONLY_TRAILING_WORD_PROMPT` constant lives in the variant file itself, matching the standalone-file convention.

3. **Existing JSON cache files in `./results/llm_checkpoint/` — delete or leave?**
   - What we know: D-05 says no migration. Abandoning them is safe (coexist with diskcache SQLite files).
   - What's unclear: Hygiene preference.
   - Recommendation: Leave them in place. Document as orphaned in plan notes. Removing them is a separate cleanup task unrelated to INFRA-03.
   - **RESOLVED 2026-05-08:** Leave as-is per D-05 ("existing on-disk caches not migrated"). No deletion task added; orphaned JSONs are documented as harmless coexisting artifacts.

4. **Does Phase 1 include any retrofit of pre-12c variants (s_linker3..12b) to use `_VARIANT_NAME`?**
   - What we know: CONTEXT.md §domain says "no hardcoded `s_linker12c` string anywhere outside the 12c file." Grep finds 20+ variants with hardcoded strings.
   - What's unclear: Scope interpretation. "Outside 12c" could mean (a) specifically "outside the 12c file" (so other variants can keep their hardcoded names) or (b) "across the entire codebase" (requiring retrofit of every variant).
   - Recommendation: Reading (a) matches the phase's stated goal (cleanup in preparation for 13a, which is what needs to NOT have `s_linker12c` in it). Pre-12c variants are frozen experiments — retrofit is busy work. Planner confirms with user during discuss.
   - **RESOLVED 2026-05-08:** Reading (a) — only 12c and 13a are touched in Phase 1. No retrofit of pre-12c variants. They remain frozen experiments with their hardcoded strings; if any are re-run later, the planner in that phase decides whether to retrofit.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python 3.11+ | All | [ASSUMED ✓] | — | — |
| `pip install -e` | diskcache/tabulate install | [ASSUMED ✓] | — | — |
| `claude` CLI | `_query_claude` subprocess backend | [ASSUMED ✓ — already the project's primary execution path] | — | `codex` or `openai` backends available via `LLM_BACKEND` env var |
| `diskcache` | INFRA-03 | ✗ (new dep) | to be installed >=5.6.1 | None — D-05 requires it |
| `tabulate` | Phase 5 (dep-only in Phase 1) | ✗ (new dep) | to be installed >=0.9.0 | None |
| Benchmark datasets (mediastore/teastore/teammates/bigbluebutton/jabref) | INFRA-01, VAR-01 | [ASSUMED ✓ — path `../ardoco/core/tests-base/src/main/resources/benchmark/` per `run_ablation.py` L277] | — | If missing, baseline/evaluation blocked |
| TransArc CSVs at `/mnt/hostshare/ardoco-home/cli-results/` | run_ablation passes `transarc_csv` to `linker.link()`; s_linker12c uses it in seed extraction? | [ASSUMED ✓] | — | Runner guards with `paths["transarc_sam"].exists()` at L650 — missing files tolerated |

**Missing dependencies with no fallback:** None (new Python deps install via pip).

**Missing dependencies with fallback:** Backend fallback is the `CHECKPOINT`→claude path in `llm_client.py`; not relevant to Phase 1 execution because D-01 keeps `claude -p`.

**Planner action:** Add a sanity-check task at the start of plan execution that runs `pip install -e ".[dev,openai]"` after pyproject.toml is updated, then `python -c "import diskcache; import tabulate"` to confirm installs succeeded.

---

## Validation Architecture

`.planning/config.json` doesn't exist in the workspace — workflow.nyquist_validation key absent. Treating as enabled per research policy.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` L32-34 |
| Quick run command | `pytest -x` |
| Full suite command | `pytest` |

Current tests directory: `tests/` (declared in pyproject, contents not inspected — existence [ASSUMED ✓]).

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | 12c baseline JSON is produced with per-dataset and macro F1 | integration (manual invocation — full sweep takes 5-15 min) | `python run_ablation.py --variants s_linker12c` | ✅ (runner exists) |
| INFRA-03 | `_query_checkpoint` returns cached response on hit and calls fallback on miss | unit | `pytest tests/test_llm_client.py::test_checkpoint_hit_miss -x` | ❌ Wave 0 — stub test needed |
| INFRA-03 | SHA-256 key scheme unchanged (same prompt produces same key across old/new impl) | unit | `pytest tests/test_llm_client.py::test_prompt_hash_stable -x` | ❌ Wave 0 |
| INFRA-05 | `_checkpoint_dir` contains `_VARIANT_NAME` at runtime | unit (assertion-driven) | `pytest tests/test_slinker12c_namespacing.py -x` | ❌ Wave 0 |
| INFRA-05 | `_VARIANT_NAME` differs between s_linker12c and s_linker13a (cache non-collision) | unit | `pytest tests/test_variant_name_unique.py -x` | ❌ Wave 0 |
| VAR-01 | `_split_component_name` not present in s_linker13a | unit (grep-style) | `pytest tests/test_slinker13a_structure.py::test_no_split_component_name -x` | ❌ Wave 0 |
| VAR-01 | `_enrich_trailing_words` signature matches Spike 001 pattern (single LLM call, evidence guardrail) | unit | `pytest tests/test_slinker13a_structure.py::test_enrich_trailing_words_pattern -x` | ❌ Wave 0 |
| VAR-01 | Dual floor passed on full 5-project sweep | integration (manual) | `python run_ablation.py --variants s_linker12c s_linker13a` → manual F1 check | ✅ (runner) |
| VAR-01 | Hard tier gate (GATE-05) passed | integration (manual) | `python run_ablation.py --variants s_linker13a --datasets teammates bigbluebutton` | ✅ (runner) |

### Sampling Rate
- **Per task commit:** `pytest -x` on the test files added by the task
- **Per wave merge:** `pytest` full suite
- **Phase gate:** Full suite green + baseline/13a JSONs archived

### Wave 0 Gaps
- [ ] `tests/test_llm_client.py` — diskcache migration unit tests (cache hit/miss, key scheme stability)
- [ ] `tests/test_slinker12c_namespacing.py` — D-07 assertion verified, `_VARIANT_NAME` constant present
- [ ] `tests/test_slinker13a_structure.py` — VAR-01 structural assertions (no `_split_component_name`, Spike 001 pattern in place)
- [ ] `tests/conftest.py` — may need shared fixtures for mocked `LLMClient` (check if one exists)
- [ ] Framework install: [ASSUMED ✓] pytest already in pyproject dev extra

---

## Project Constraints (from CLAUDE.md)

### From `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/CLAUDE.md`
- Retained runtime files are listed explicitly — `s_linker.py` through `s_linker11a.py` are retained. `s_linker12c.py` is NOT in the retained list (the file quoted is outdated vs. the actual codebase — the project has evolved since CLAUDE.md was written). The active target is `s_linker12c.py` as confirmed by CONTEXT.md §canonical_refs and the current `run_ablation.py` `CANONICAL_VARIANTS` list including `s_linker12a..12e`.
- "Default model policy remains Claude Sonnet unless there is an explicit reason to change it" — matches D-01 (keep `claude -p`).
- Standalone linker files, duplicated code intentionally — matches Pattern 3 recommendation for `s_linker13a.py`.

### From `/mnt/hostshare/ardoco-home/CLAUDE.md` (parent repo)
- Java 21 project guidance (Spotless, JSpecify) — irrelevant to this Phase 1 which is Python-only.
- "**Never hardcode word lists derived from benchmark datasets** (e.g., component names, project-specific terms, synonym mappings)" — reinforces GATE-04 / BENCHMARK_TABOO.md audit.
- "Prompt examples must be abstract — never use names resembling benchmark components. Use generic placeholders like X, Y, or describe patterns abstractly." — reinforces Pitfall 2 fix (replace OrderProcessor → TaskScheduler).

---

## Security Domain

`.planning/config.json` does not exist; `security_enforcement` key absent. Phase 1 is a code-refactor + new-file-addition phase with no user input surface, no network exposure, no authentication, no persistence of user data. ASVS categories V2/V3/V4 do not apply. V5 (Input Validation) applies only to the LLM JSON parsing — `llm_client.py::extract_json` (L994-1028) already uses stdlib `json.loads` with try/except; no new surface added in Phase 1. V6 (Cryptography) — SHA-256 use is preserved as a cache key (not a security control); no cryptographic operations added.

No STRIDE threat expansion relative to 12c baseline. Security domain is a no-op for this phase.

---

## Sources

### Primary (HIGH confidence — direct inspection)
- `src/llm_sad_sam/llm_client.py` (1096 lines) — 5 methods to replace (L718-805), `enable_checkpoint` (L591-618), checkpoint backend resolution (L127-142, L212-232), `_query_claude` CLI args (L838-898 — confirms no temperature/cache flags)
- `src/llm_sad_sam/linkers/experimental/s_linker12c.py` (1211 lines) — `_checkpoint_dir` (L1181-1186), `_save_log` (L1204-1211), `_enrich_trailing_words` (L420-482), `_split_component_name` (L292-298), `__init__` (L89-110)
- `run_ablation.py` (687 lines) — `CANONICAL_VARIANTS` (L40-73), `VARIANT_SPECS` (L75-267), `run_variant` metrics schema (L461-544), JSON output (L678-681)
- `pyproject.toml` (34 lines) — current deps (L11-15), dev extra (L18-20), openai extra (L22-24), pytest config (L32-34)
- `BENCHMARK_TABOO.md` (69 lines) — Universal Taboo section includes "order" (L53); safe-domain substitutes (L60-68)
- `.planning/spikes/001-llm-trailing-words/spike.py` — `fully_llm_driven` signature (L121-153), `LLM_ONLY_PROMPT` with OrderProcessor example (L43-62), tests 2+3 evidence guardrail pattern (L185-217)
- `.planning/spikes/002-rules-audit/AUDIT.md` — `_split_component_name` ESSENTIAL-but-removable via Spike 001
- `.planning/phases/01-baseline-and-infrastructure/01-CONTEXT.md` — all decisions D-01..D-07
- `.planning/REQUIREMENTS.md` — INFRA-01..05, VAR-01, GATE-01..06
- `.planning/ROADMAP.md` — §Phase 1 success criteria
- `.planning/research/ARCHITECTURE.md` (422 lines) — 13-series variant layout, build order
- `.planning/research/STACK.md` (223 lines) — diskcache/tabulate versions (A1 source)
- `.planning/research/PITFALLS.md` (350 lines) — Pitfalls 1, 2, 7, 8 reused here

### Secondary (MEDIUM confidence)
- Auto-memory (MEMORY.md excerpt in system context) — Claude Sonnet preference, standalone-file convention, no benchmark-leakage rule

### Tertiary (LOW confidence)
- None — Phase 1 relied entirely on direct inspection and user decisions.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions traceable to Phase 0 STACK.md which verified against PyPI; no new dependencies introduced beyond those already researched.
- Architecture: HIGH — all code referents verified by direct file read; the two-caches distinction confirmed by grep across the source tree.
- Pitfalls: HIGH — reused from `.planning/research/PITFALLS.md` (already Phase 0-validated); Pitfall 2 (taboo audit for OrderProcessor) newly identified by this research via direct grep of Spike 001 against BENCHMARK_TABOO.md.
- Requirements-to-code mapping: HIGH — all 5 private methods in `llm_client.py` (D-05), `_checkpoint_dir` in `s_linker12c.py` (D-03), `_enrich_trailing_words` + `_split_component_name` (VAR-01) verified at exact line numbers.
- D-07 assertion placement: RESOLVED 2026-05-08 — user confirmed in-method placement (see Open Questions Q1).

**Research date:** 2026-04-25
**Valid until:** 2026-05-25 (30 days — code references stable, pyproject pins unchanged)
