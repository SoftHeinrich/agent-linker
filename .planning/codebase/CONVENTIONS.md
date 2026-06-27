# Conventions

## Code Style

### Type Annotations & Language Features
- **Python 3.11+** required
- Full use of PEP 604 union syntax: `Type1 | Type2` (not `Union[Type1, Type2]`)
- Use `from __future__ import annotations` at file top
- Comprehensive type hints on all function signatures and dataclass fields
- Dataclasses used extensively for structured data (e.g., `EvidenceBundle`, `ExtractedLink`, `ModelKnowledge`)

### Naming Conventions
- **Snake_case** for variables, functions, module names
- **CamelCase** for classes
- **SCREAMING_SNAKE_CASE** for module-level prompt/rule constants
- Private methods prefixed with `_` (single underscore)
- Component names preserve their architectural case (e.g., `PaymentGateway`, `RenderEngine`)

### Line Length & Formatting
- Soft 88-character target (pragmatic, not strict)
- Long list comprehensions and f-strings broken across multiple lines
- Multi-line prompt constants use triple-quoted strings with explicit formatting
- Comments use `#` with space before text (PEP 8)

### Docstrings & Comments
- Module-level docstring at file start (required), follows PEP 257
- Class docstrings: one-line summary (followed by blank line if needed) — example:
  ```python
  class SLinker12c:
      """LLM-driven SAD-SAM traceability with structural guardrails."""
  ```
- Method docstrings: concise; use `Args:` and `Returns:` sections when needed
- Internal section comments use visual separators: `# ═══════════════════════════════`
- Inline comments rare; code is self-documenting via naming

### Static Methods & Class Design
- Heavily use `@staticmethod` for pure utilities (no instance state needed)
- Example: `_is_structurally_unambiguous()`, `_split_component_name()`
- Private helper methods grouped in logical sections with `# ──` comments

---

## Naming Scheme: Linker Versioning

### Integer Variant Lineage
Linkers follow a strict numeric progression:
- **ILinker** family: `ilinker1.py`, `ilinker2.py`, `ilinker3.py` — explicit mention extractors
- **S-Linker** family: `s_linker.py` (alias for s1), then `s_linker2.py` through `s_linker12e.py`
  - Represents cumulative refinement via phase ablations, heuristic changes, and LLM prompt tuning

Each version is **independent and intentionally duplicate-heavy** (not inherited). Versions are kept side-by-side for ablation studies and regression detection.

### Letter Suffixes for Ablations
**Single letter** appended to integer: `s_linker9a.py`, `s_linker9b.py`, `s_linker9c.py`, etc.
- Indicates **experimental variant** of the base version (e.g., 9a/9b/9c are all variants of 9)
- Letters used for one-off ablations that don't graduate to the next integer
- Example progression: S9 base → S9a (try removing heuristic X) → S9b (try alternative Y)

### API Versioning: `_v2` Suffix
- `data_types_v2.py`, `document_loader_v2.py`, `prompts_v2.py` — **major API overhaul versions**
- Used when data structures or function signatures change incompatibly from `v1`
- ILinker/early S-linker use `data_types.py` and `document_loader.py` (v1)
- Later S-linker (S10+) standardized on `_v2` stack
- **Important**: `_v2` is NOT a timeline marker, but compatibility marker

### Prompt Versioning
- Prompt constants organized by tier: `AMBIGUITY_*` (Tier 1), `DOC_KNOWLEDGE_*` (Tier 1), `ENTITY_EXTRACTION_*` (Tier 2), etc.
- Each constant is immutable and referenced by multiple linkers
- **prompts.py** — legacy prompt set (used by ILinker, early S-Linker)
- **prompts_v2.py** — current unified set, no dead code, only actively-used constants

---

## Patterns

### Self-Contained Linker Classes
Each linker (S9, S10, S12c) is a **complete, standalone Python file**:
- No inheritance from parent linker classes
- No shared base-class abstraction
- Duplicated code between versions is **intentional** (enables independent ablation)
- All imports are local; no `from .s_linker9 import X` cross-references
- Reason: reproducibility and low-coupling for experimental forks

### Phased Pipeline Methods
Linkers implement a **tiered DAG pipeline** with explicit phase methods:

```python
def link(self, text_path, model_path, **_kwargs):
    """Main entry point — orchestrates Tier 1 → Tier 2 → Tier 3."""
    
    # Tier 1: Knowledge Acquisition (parallel)
    acq = self._run_parallel({
        "model": lambda: self._analyze_model(components),
        "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
        "seed": lambda: self._run_seed(sentences, components),
    })
    
    # Tier 2: Link Recovery (parallel)
    rec = self._run_parallel({
        "seed_val": lambda: self._run_seed_validation(...),
        "entity": lambda: self._run_entity_pipeline(...),
        "coref": lambda: self._run_coreference(...),
    })
    
    # Tier 3: Consolidation (sequential)
    final = self._consolidate(seed_links, entity_links, coref_links)
    
    return final
```

**Helper methods follow `_build_*` or `_run_*` naming**:
- `_build_comp_block()` — construct data structures (non-LLM)
- `_run_entity_pipeline()` — execute a pipeline phase (may call LLM)
- `_analyze_model()`, `_learn_document_knowledge_*()` — specific Tier 1 tasks
- `_run_*` methods are side-effect aware (print, log, save checkpoints)

### Parallel Execution
Tier 1 and Tier 2 tasks execute in parallel via `ThreadPoolExecutor`:

```python
@staticmethod
def _run_parallel(tasks):
    """Run named tasks concurrently, wait for all. Returns {name: result}.
    
    On first failure, cancels remaining futures and re-raises.
    """
    if len(tasks) == 1:
        name, fn = next(iter(tasks.items()))
        return {name: fn()}
    
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        try:
            for fut in as_completed(futures):
                name = futures[fut]
                results[name] = fut.result()
        except Exception:
            for other in futures:
                other.cancel()
            raise
    return results
```

### LLM Call Wrapping & Retry Loop
All LLM interactions follow this pattern:

```python
for attempt in range(2):
    data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
    if data:
        break
    if attempt == 0:
        print("    [Task]: empty response, retrying...")

if data:
    # Process result
else:
    # Fallback (often empty dict)
    data = {}
```

**Key characteristics**:
- Two-attempt retry (zero backoff, immediate re-run)
- Print-based notification on first failure
- `self.llm.extract_json()` wrapper extracts JSON from response text
- Fallback to empty dict on double failure (no exception raised)
- **No structured logger** — print() is the logging mechanism throughout

### JSON Extraction Pattern
LLM responses are expected to contain JSON blocks:

```python
prompt = f"""Your task here...

Return JSON:
{{
  "field1": ["value1", "value2"],
  "field2": boolean
}}

JSON only:"""

response = self.llm.query(prompt, timeout=100)
data = self.llm.extract_json(response)  # Parses JSON from response text
```

**Conventions**:
- Prompt instructs "JSON only:" at end to signal tight formatting
- JSON response must be a single dict (not array) at top level
- `extract_json()` handles error cases (returns empty dict on parse failure)

### Checkpoint Saving (Phase-Level)
Some linkers (S10, S30c) save pickle checkpoints after each Tier:

```python
self._save_phase(text_path, "layer1", {
    "model_knowledge": self.model_knowledge,
    "doc_knowledge": self.doc_knowledge,
    "raw_seed_links": raw_seed_links,
})
```

**Used for**:
- Single-phase ablation studies (replay from checkpoint, skip LLM)
- Resumption support on long runs
- Test fixture data

---

## Error Handling

### Retry Loop
Standard two-attempt pattern (no exponential backoff):

```python
for attempt in range(2):
    result = self.llm.extract_json(self.llm.query(prompt))
    if result:
        break
    if attempt == 0:
        print("    [Context]: empty response, retrying...")
```

**Behavior**:
- First attempt fails silently; prints message; retries immediately
- Second attempt result (successful or empty) is used as-is
- No exception raised; empty dict returned on double failure
- Caller must check truthiness of result (`if data:` or `if not data: data = {}`)

### Fallback Patterns
When LLM returns empty or unparseable:
1. **Extraction phase**: Return empty list `[]` — no candidates added
2. **Validation phase**: Return empty set — candidates are not validated
3. **Alias discovery**: Return empty dict or set — no aliases added
4. **Coref**: Return empty list — no coref links generated

**Rationale**: Better to skip problematic links than crash; downstream phases are robust to missing inputs.

### No Exception-Based Control
- Exceptions are rare (only for fatal errors like file not found, YAML parsing)
- **Not used** for expected failures (empty LLM response, parse failure, timeout)
- Allows graceful degradation when LLM is flaky or under load

### Progress Logging (Print-Based)
All progress and diagnostics via `print()`:

```python
print(f"Loaded {len(components)} components, {len(sentences)} sentences")
print("\n[Tier 1] Knowledge Acquisition (parallel)")
print(f"  Model: {len(ambig)} ambiguous (of {len(components)} components)")
print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")
```

**Characteristics**:
- Human-readable, unstructured format
- Indentation indicates nesting (two-space indent for sub-messages)
- Timing and statistics printed at key checkpoints
- No file logging in production linkers (only in CLI/runner)

---

## Prompt Construction

### F-String Interpolation
All prompts built via f-strings with embedded variables:

```python
prompt = f"""Classify these component names.

NAMES: {', '.join(names)}

RULES:
{AMBIGUITY_RULES}

Return JSON:
{{
  "architectural": [...],
  "ambiguous": [...]
}}

JSON only:"""
```

### Prompt Constant Organization
All reusable prompt text stored as module-level constants in `prompts_v2.py`:

```python
AMBIGUITY_FEW_SHOT = """
EXAMPLE 1:
NAMES: Lexer, Parser, CodeGenerator, ...
→ architectural: ["Lexer", "Parser", ...]
→ ambiguous: ["Core", "Util", "Base"]
..."""

AMBIGUITY_RULES = """RULES:
1. ARCHITECTURAL: Names that refer to a specific role...
2. AMBIGUOUS: Single words that writers regularly use generically..."""
```

**Ordering principle**: Constants organized by pipeline tier (1 → 2 → 3), then by task within tier.

### JSON Schema Instructions
Prompts specify exact JSON structure with placeholder example:

```python
prompt = f"""...
Return JSON:
{{
  "field1": ["value1", "value2"],
  "field2": boolean,
  "field3": {{"key": "value"}}
}}
JSON only:"""
```

**Conventions**:
- Placeholder uses realistic types and names (not X, Y, Z)
- "JSON only:" at end signals tight formatting
- No prose after JSON schema

### Few-Shot Examples
Examples embedded in prompt constants use **safe SE textbook domains**:
- Compiler: Lexer, Parser, CodeGenerator, Optimizer, SymbolTable, AST
- Operating systems: Scheduler, Dispatcher, MemoryManager, ProcessTable
- Networking: Router, Multiplexer, PacketHandler
- Graphics: RenderEngine, SceneGraph, Pipeline
- E-commerce: PaymentGateway, InvoiceHandler, ShoppingCart, InventoryTracker

**Never use benchmark domain terms** (see BENCHMARK_TABOO.md).

---

## Data Leakage Rules

### Hardcoded Word Lists Are Prohibited
**CRITICAL**: No hardcoded component names, aliases, or project-specific abbreviations derived from benchmark datasets.

**Examples of violations**:
```python
# ❌ BANNED — violates data leakage rule
GENERIC_WORDS = {"cache", "database", "server", "client", "registry", "auth"}
KNOWN_COMPONENTS = ["Lexer", "Parser", "Scheduler"]  # If from benchmark
BENCHMARK_ALIASES = {"DB": "Database", "UI": "UserInterface"}
```

**What IS allowed** — common English stopwords:
```python
# ✅ ALLOWED — universal English, not benchmark-derived
PRONOUNS = ["it", "they", "this", "that", "its", "their"]
GENERIC_ENGLISH = ["the", "a", "an", "is", "are", "be"]
```

### All Domain Knowledge Must Be Dynamic
- **Abbreviations, synonyms, generic words**: discovered at runtime via LLM
- **Aliases**: extracted from document using `DOC_KNOWLEDGE_EXTRACTION_RULES`
- **Component ambiguity**: classified dynamically in Tier 1 via LLM
- **Prompt examples**: sourced from safe SE textbook domains, not benchmark

### Audit Checklist
Before committing changes to prompts or data constants:
1. No benchmark component names (check BENCHMARK_TABOO.md)
2. No benchmark aliases (cascade, recorder, authentication, etc.)
3. No benchmark keywords (watermark, recommender, kurento, FreeSWITCH, etc.)
4. All examples from safe textbook domains
5. Abbreviation/synonym discovery happens at runtime only

See `/mnt/hostshare/ardoco-home/llm-sad-sam-v45/BENCHMARK_TABOO.md` for complete prohibited term list.
