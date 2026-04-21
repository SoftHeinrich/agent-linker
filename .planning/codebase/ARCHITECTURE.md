# Architecture

LLM-based traceability link recovery (TLR) framework for connecting software architecture documentation (SAD) with architecture models (SAM). The codebase implements 40+ experimental linker variants across two major families (ILinker, S-Linker) designed to recover trace links between documentation sentences and architecture components.

## Overall Pattern: Phased Pipeline with Shared Blackboard State

Each linker is a **self-contained phased pipeline** that processes inputs and produces `SadSamLink[]` output. Linkers share state via a **blackboard pattern** (documented in `DocumentKnowledge` and `ModelKnowledge` dataclasses).

```
Input (documentation text, PCM model) 
  ↓
Linker.link(text_path, model_path) 
  ↓
Phase 1, Phase 2, ... (typically 8-12 phases)
  ↓
Output: list[SadSamLink] → CSV/pickle/checkpoint
```

**Phase Structure** (S-Linker12c example):

1. **Tier 1 (Parallel Knowledge Acquisition)**
   - Model Analysis: LLM classifies component names as architectural vs ambiguous
   - Document Knowledge Learning: LLM discovers abbreviations, synonyms, trailing-word forms
   - Seed Extraction: ILinker3-based baseline via two LLM passes (explicit mentions only)

2. **Tier 2 (Parallel Link Recovery)**
   - Seed Validation: Per-component disambiguation (single LLM pass)
   - Entity Pipeline: Dual-pass extraction + consensus + evidence-aware validation
   - Coreference: Pronoun resolution with ±5-sentence context window

3. **Tier 3 (Consolidation)**
   - Priority-ordered deduplication (seed > entity > coref)
   - Convention-aware boundary filtering
   - Final CSV/pickle output

**Key Design Principle**: LLM-driven decisions (extraction, validation, judgment) with lightweight structural guardrails (CamelCase detection, dotted-path exclusion, pronoun pattern matching, alias strength classification).

## Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Entry Points                                                    │
│ - run_ablation.py (CLI orchestrator, variant loading, metrics)  │
│ - pytest test files (unit/integration tests)                    │
│ - compare_s12c_vs_transarc.py (variant comparison script)       │
└─────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Linker Implementations                                          │
│ src/llm_sad_sam/linkers/experimental/{s_linker,ilinker}*.py    │
│ - SLinker (s_linker.py through s_linker12e.py)                 │
│ - ILinker (ilinker1.py, ilinker2.py, ilinker3.py)              │
│ Each linker: one Python class with link(text, model) method     │
└─────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Orchestrator Services                                           │
│ - LLMClient (llm_client.py): backend abstraction                │
│ - Prompt constants (prompts.py, prompts_v2.py)                 │
│ - Model parser (pcm_parser_v2.py): extract components          │
│ - Document loader (core/document_loader_v2.py): load sentences │
└─────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Core Types & State                                              │
│ - SadSamLink: (sentence_number, component_id, component_name)  │
│ - DocumentKnowledge: aliases, abbreviations, synonyms          │
│ - ModelKnowledge: ambiguous_names, architectural_names         │
│ - Sentence: (number, text)                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Input Loading**
   - Documentation text file (one sentence per line) → `load_sentences()`
   - PCM model `.repository` XML file → `parse_pcm_repository()`
   - Both routed through linker constructor

2. **Pipeline Execution**
   - Linker.link() instantiates shared state (`DocumentKnowledge`, `ModelKnowledge`)
   - Each phase reads/writes to shared state; candidate links accumulated in lists
   - LLM calls via `LLMClient.query()` with prompt templates

3. **Output Generation**
   - List of `SadSamLink` objects with:
     - `sentence_number` (1-indexed)
     - `component_id` (from PCM model)
     - `component_name` (from PCM model)
     - `confidence` (typically 1.0 for final links)
     - `source` (e.g., "seed", "entity", "coref")
   - Serialized to:
     - **CSV**: `{variant}_{dataset}_links.csv` in `results/ablation_results/`
     - **Pickle**: `{variant}_{dataset}.pkl` for checkpoint/resumption
     - **Log**: LLM traces in `results/llm_logs/`

## Key Abstractions

### Data Types

**`SadSamLink`** (`core/data_types_v2.py`):
```python
@dataclass
class SadSamLink:
    sentence_number: int
    component_id: str
    component_name: str
    confidence: float = 1.0
    source: str = ""
```
- Single trace link (documentation → architecture)
- Hashable by `(sentence_number, component_id)` tuple for deduplication
- Source field tracks origin (seed extraction, entity pipeline, coreference)

**`DocumentKnowledge`** (shared mutable state):
```python
@dataclass
class DocumentKnowledge:
    aliases: dict[str, str]          # alternative_name → canonical_name
    abbreviations: dict[str, str]    # AST → AbstractSyntaxTree (legacy)
    synonyms: dict[str, str]         # dispatcher → task dispatcher (legacy)
    partial_references: dict[str, str]  # server → html5server (legacy)
```
- Built during Phase 2/3 (document analysis)
- Used by entity extraction, validation, coreference to expand matching
- S-Linker12c+ consolidates to single `aliases` dict

**`ModelKnowledge`** (shared read-only state):
```python
@dataclass
class ModelKnowledge:
    ambiguous_names: set[str]  # generic names (Core, Util, Base)
```
- Built during Phase 1 (model analysis)
- Used by extraction/validation to identify generic mentions
- Example: ["Core", "Util", "Base"] vs architectural ["Lexer", "Parser"]

**`Sentence`** (`core/document_loader_v2.py`):
```python
@dataclass
class Sentence:
    number: int     # 1-indexed
    text: str       # full sentence text
```

**`ArchitectureComponent`** (`pcm_parser_v2.py`):
```python
@dataclass
class ArchitectureComponent:
    id: str         # PCM model ID (e.g., "_Gp6H...")
    name: str       # entity name (e.g., "Dispatcher")
```

### Services

**`LLMClient`** (`llm_client.py`):
- Unified backend abstraction (Claude, OpenAI, Codex, Checkpoint)
- Methods: `query(system_prompt, user_message)` → `LLMResponse`
- Conversation mode: `start_conversation()`, `query_conversation()`, `end_conversation()`
- Logging: per-query logs in `results/llm_logs/`
- Token tracking: cumulative usage across all instances

**`LLMBackend` (enum)**:
- `CLAUDE`: Local Claude Code CLI (`claude -p`)
- `OPENAI`: OpenAI API (GPT-5.2 default)
- `CODEX`: OpenAI Codex CLI (legacy)
- `CHECKPOINT`: Cached responses from prior runs

**Prompt Modules**:
- `prompts.py` (v1 API): Older ILinker/early S-Linker prompts
- `prompts_v2.py` (v2 API): S-Linker11+ prompts, cleaner organization
  - Constants: `AMBIGUITY_FEW_SHOT`, `VALIDATION_RULES`, `COREF_RULES`, etc.
  - Grouped by tier (Tier 1 model analysis, Tier 2 extraction, Tier 3 judgment)

**Model Parser** (`pcm_parser_v2.py`):
- XPath iteration over PCM `.repository` XML
- Filters: `BasicComponent`, `CompositeComponent`
- Returns list of `ArchitectureComponent` objects

**Document Loader** (`core/document_loader_v2.py`):
- Line-by-line sentence loading (assumes one sentence per line)
- Builds sentence map: `sentence_number → Sentence`
- Used by all linkers uniformly

## Entry Points

### Primary Orchestrator: `run_ablation.py`

CLI runner supporting 40+ linker variants:

```bash
python run_ablation.py                    # Run all retained variants on all datasets
python run_ablation.py --variant i2       # Run ILinker2 only
python run_ablation.py --dataset mediastore  # Run all variants on MediaStore
python run_ablation.py --list-variants    # List all available variants
```

**Workflow**:
1. Parse `--variant`, `--dataset`, `--benchmark-dir` flags
2. For each (variant, dataset) pair:
   - Dynamically load linker class (e.g., `llm_sad_sam.linkers.experimental.s_linker12c.SLinker12c`)
   - Instantiate and call `linker.link(text_path, model_path)`
   - Serialize results to CSV + pickle + log
3. Compute metrics: precision, recall, F1 against gold standards
4. Summarize results table

**Canonical Variants** (from `CANONICAL_VARIANTS` list):
- ILinker: `i1`, `i2`, `i3`
- S-Linker: `s_linker` (v1) through `s_linker12e`
- Ablations: `s_linker7a`, `s_linker7b`, `s_linker9a`–`9e`, `s_linker12a`, `s_linker12b`

### Test Suite: `tests/test_*.py`

```bash
pytest                              # Run all tests
pytest tests/test_12e_variance.py   # Run single test file
```

Test files:
- `test_12e_variance.py`: S-Linker12e run-to-run consistency (LLM variance)
- `test_checkpoint_migration.py`: Cross-linker checkpoint loading
- `test_mention_type_ablation.py`: Evidence bundle mention-type analysis
- `test_voting_analysis.py`: Phase 2 voting strategy analysis
- `test_voting_asymmetry.py`: Asymmetric extraction behavior

### Comparison Scripts: `compare_s12c_vs_transarc.py`

Ad-hoc comparison of two linkers on all datasets:
```bash
python compare_s12c_vs_transarc.py
```
Outputs side-by-side metrics CSV + FP/FN analysis.

## Phase Structure (Typical S-Linker Variants)

### Tier 1: Knowledge Acquisition (Parallel)

**Phase 1 — Model Analysis** (`_analyze_model()`):
- LLM classifies component names as architectural (specific) vs ambiguous (generic)
- Input: list of component names from PCM
- Output: `model_knowledge.ambiguous_names` set
- Example:
  ```
  Input: ["Lexer", "Parser", "Core", "Util", "Base", "AST"]
  Output: {"Core", "Util", "Base"}
  ```

**Phase 2 — Document Knowledge Learning** (`_learn_document_knowledge()`):
- LLM discovers abbreviations, synonyms, trailing-word forms in text
- Input: full documentation + component names
- Output: `doc_knowledge.aliases` dict (alternative_name → canonical_name)
- Example:
  ```
  aliases = {
    "AST": "AbstractSyntaxTree",
    "dispatcher": "Dispatcher",
    "task dispatcher": "Dispatcher"
  }
  ```

**Phase 3 — Seed Extraction** (`_seed_extraction()`):
- Reuses ILinker3 (two-pass explicit mention extractor)
- Pass A: Extraction-framed (find all mentions)
- Pass B: Actor-framed (what is each sentence about?)
- Merge: exact matches from either pass; synonyms/partials only if both agree
- Output: initial `SadSamLink[]` with source="seed"

### Tier 2: Link Recovery (Parallel)

**Phase 4 — Seed Validation** (`_validate_seed_links()`):
- Per-component disambiguation (LLM judges each seed link)
- Input: seed link + sentence text + component name
- Decision rules: architectural role vs code-level vs different entity
- Output: approved/rejected seed links

**Phase 5–6 — Entity Pipeline** (`_entity_extraction()` + `_validate_entities()`):
- Dual-pass LLM extraction (two independent passes to find explicit mentions)
- Consensus: require both passes to agree (intersection voting)
- Evidence-aware validation: LLM assesses mention type (proper case, lowercase, alias, etc.)
- Output: validated `SadSamLink[]` with source="entity"

**Phase 7 — Coreference Resolution** (`_coreference()`):
- Pronoun resolution (it, they, this, that, its, their)
- Context window: ±5 sentences from pronoun mention
- Antecedent linking: LLM determines which component the pronoun refers to
- Output: `SadSamLink[]` with source="coref"

### Tier 3: Consolidation

**Phase 8 — Deduplication & Boundary Filtering** (`_deduplicate_and_filter()`):
- Priority: seed > entity > coref
- Remove duplicates by `(sentence_number, component_id)` key
- Convention-aware boundary filter (LLM judges) catches FPs:
  - Technology-named components (Redis PubSub, kurento)
  - Standalone lowercase mentions (unlikely to be components)
  - Embedded in dotted-path notation

**Phase 9 — Final Judge** (`_judge_links()` in some variants):
- Four-rule judge over borderline links:
  1. Explicit Reference: sentence explicitly uses component name
  2. System-Level Perspective: describes component's role/behavior
  3. Primary Focus: component is main subject of sentence
  4. Component-Specific Usage: name not used generically

### Variant-Specific Phases

**S-Linker12c (ICSE/Clean)**:
- Alias stratification (strong global aliases vs weak local)
- Unified `DocumentKnowledge.aliases` dict (consolidates abbreviations/synonyms/partials)
- Evidence bundles track mention type (proper case, lowercase, alias, etc.)
- Intersection voting for seed/entity deduplication

**S-Linker7** (Simplified):
- Skips Phases 8c (convention filter) and 10 (FN recovery)
- Direct deduplication on Phase 5+6+7 output

**S-Linker10a** (Ablation):
- Replaces LLM word-usage enrichment with fixed threshold (count >= 3)

**ILinker2** (Baseline):
- Two-pass (Pass A extraction, Pass B actor-framed)
- No contextual reasoning or coreference
- Used as seed extraction component in S-Linker variants

**ILinker3** (v2-Stack):
- Wrapper around v2 data types for compatibility with S-Linker12+
- Same two-pass logic as ILinker2

## Example Data Flow: S-Linker12c + MediaStore

```
Input:
  documentation: "The Dispatcher routes requests to handlers. 
                  Each handler processes specific message types."
  model: ["Dispatcher", "Handler", "Core", "Util"]

Phase 1 — Model Analysis:
  ambiguous_names = {"Core", "Util"}

Phase 2 — Document Knowledge Learning:
  aliases = {"handler": "Handler", "requests": "request"}

Phase 3 — Seed Extraction (ILinker3):
  - Pass A: finds "Dispatcher", "handlers", "Handler", "types"
  - Pass B: subject is Dispatcher; finds "Dispatcher", "handler"
  - Consensus: ["Dispatcher", "Handler"]
  → SadSamLink(sentence_number=1, component_id="_ABC", component_name="Dispatcher")
  → SadSamLink(sentence_number=1, component_id="_XYZ", component_name="Handler")

Phase 4 — Seed Validation:
  "Dispatcher routes requests" → COMPONENT (architectural role)
  "handler processes message types" → COMPONENT (processes → action)

Phase 5–6 — Entity Pipeline:
  Dual extraction finds same links; validation passes.
  Evidence bundle: mention_type="proper case, standalone"

Phase 7 — Coreference:
  "Each handler processes..." → "handler" refers to Handler component
  → SadSamLink(sentence_number=2, component_id="_XYZ", component_name="Handler")

Phase 8 — Deduplication:
  Seen links: {(1, Dispatcher), (1, Handler), (2, Handler)}
  No duplicates; skip boundary filter (all proper-case).

Output:
  [
    SadSamLink(1, "Dispatcher", confidence=1.0, source="seed"),
    SadSamLink(1, "Handler", confidence=1.0, source="seed"),
    SadSamLink(2, "Handler", confidence=1.0, source="coref")
  ]
```

## Configuration & Environment

**Environment Variables** (loaded from `.env`):
- `CLAUDE_MODEL` (default: "sonnet") — Claude model for linker instantiation
- `OPENAI_API_KEY` — OpenAI API key for GPT-5.2 backend
- `OPENAI_MODEL_NAME` (default: "gpt-5.2") — OpenAI model
- `LLM_BACKEND` (default: "claude") — Backend: claude, openai, codex, checkpoint
- `LLM_LOG_DIR` (default: `results/llm_logs`) — Where to save LLM request/response logs
- `BENCHMARK_DIR` (default: parent directory) — Location of benchmark datasets

**Linker Instantiation**:
```python
from llm_sad_sam.linkers.experimental.s_linker12c import SLinker12c
linker = SLinker12c(
    backend=LLMBackend.CLAUDE,
    model="sonnet",
    checkpoint_fallback=LLMBackend.OPENAI
)
links = linker.link(text_path="doc.txt", model_path="model.repository")
```

## Design Decisions & Constraints

1. **No Dataset Leakage**: Prompts use only safe SE textbook domains (compiler, OS, e-commerce, graphics). All benchmark component names (MediaStore, TeaStore, etc.) taboo in prompts. See `BENCHMARK_TABOO.md`.

2. **Standalone Linker Files**: Duplicate code intentionally (not inheritance chains). Each linker is self-contained for experimental independence.

3. **Default Backend**: Claude Sonnet (per MEMORY.md). GPT-5.2 optional for cross-model evaluation (3.9pp gap on v32).

4. **Deterministic Serialization**: All output to CSV + pickle for reproducibility and checkpoint migration.

5. **Blackboard State Pattern**: Shared mutable state (`DocumentKnowledge`, `ModelKnowledge`) reduces parameter passing; clear phase separation maintains readability.
