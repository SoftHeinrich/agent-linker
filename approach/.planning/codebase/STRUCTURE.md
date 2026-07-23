# Structure

## Top-level Layout

```
/mnt/hostshare/ardoco-home/llm-sad-sam-v45/
├── run_ablation.py              Main CLI orchestrator (40+ variants)
├── pyproject.toml               Package metadata + dependencies
├── CLAUDE.md                    Project instructions (no dataset leakage, standalone linkers)
├── BENCHMARK_TABOO.md           Benchmark component names forbidden in prompts
├── .env                         Environment config (CLAUDE_MODEL, OPENAI_API_KEY, etc.)
├── .planning/
│   ├── codebase/                ← Codebase maps (ARCHITECTURE.md, STRUCTURE.md)
│   └── spikes/                  Earlier research spikes
├── src/llm_sad_sam/             Core package + linker implementations
├── tests/                       pytest test suite
├── results/                     Output artifacts (CSV, pickle, LLM logs)
├── archive/                     Archived linker variants (pre-retained, archived in archive/README.md)
├── doc/                         Documentation notes
└── paper/                       Paper drafts / writing
```

## src/llm_sad_sam/

Core package structure:

```
src/llm_sad_sam/
├── __init__.py
├── llm_client.py                LLMClient abstraction (Claude, OpenAI, Codex, Checkpoint)
├── pcm_parser.py                Legacy PCM parser (v1 API)
├── pcm_parser_v2.py             Clean PCM parser (ArchitectureComponent dataclass)
├── core/
│   ├── __init__.py              Public exports (SadSamLink, DocumentKnowledge, etc.)
│   ├── data_types.py            v1 data types (legacy, 6.6 KB, with dead fields)
│   ├── data_types_v2.py         v2 data types (lean, 1.8 KB, S-Linker11+ standard)
│   ├── document_loader.py       v1 document loader (legacy)
│   ├── document_loader_v2.py    v2 document loader (Sentence dataclass, load_sentences)
│   ├── model_analyzer.py        Legacy model analysis utilities
│   └── __pycache__/
└── linkers/
    ├── __init__.py
    └── experimental/
        ├── __init__.py          Linker version registry
        ├── ilinker1.py          Three-pass precision cascade (13 KB)
        ├── ilinker2.py          Two-pass explicit extractor (10 KB)
        ├── ilinker3.py          v2-stack adapter wrapper (8.3 KB)
        ├── s_linker.py          S-Linker v1 base DAG pipeline (102 KB)
        ├── s_linker2.py through s_linker11e.py
        │                        S-Linker variants 2–11e (45–71 KB each)
        ├── s_linker12a.py       Alias stratification + no partial injection
        ├── s_linker12b.py       ICSE candidate (alias + evidence bundles)
        ├── s_linker12c.py       Clean production (57 KB, Mar 27)
        ├── s_linker12d.py       Trailing-word enrichment variant
        ├── s_linker12e.py       Trailing-word + seed disambiguation variant
        ├── prompts.py           v1 prompt constants (17.8 KB, older)
        ├── prompts_v2.py        v2 prompt constants (15.4 KB, clean, S-Linker11+)
        ├── archive -> ../../../archive/linkers  Symlink to archived variants
        └── __pycache__/
```

## Core Data Types & Utilities

### `src/llm_sad_sam/core/data_types_v2.py` (1.8 KB, S-Linker11+ standard)

Lean dataclasses for S-Linker11 and later:

```python
@dataclass
class SadSamLink:
    """Trace link: (sentence_number, component_id)."""
    sentence_number: int
    component_id: str
    component_name: str
    confidence: float = 1.0
    source: str = ""

@dataclass
class CandidateLink:
    """Candidate before validation."""
    sentence_number: int
    sentence_text: str
    component_name: str
    component_id: str
    matched_text: str
    source: str = ""

@dataclass
class ModelKnowledge:
    """Model state: ambiguous component names."""
    ambiguous_names: set[str] = field(default_factory=set)

@dataclass
class DocumentKnowledge:
    """Document state: aliases (abbreviations, synonyms, trailing-word)."""
    aliases: dict[str, str] = field(default_factory=dict)
    # Legacy fields for pre-12c compat
    abbreviations: dict[str, str] = field(default_factory=dict)
    synonyms: dict[str, str] = field(default_factory=dict)
    partial_references: dict[str, str] = field(default_factory=dict)
```

### `src/llm_sad_sam/core/data_types.py` (6.6 KB, legacy for pre-S11 linkers)

Extended data types with dead fields (`impl_to_abstract`, `shared_vocabulary`, `impl_indicators`) — kept for backward compat.

### `src/llm_sad_sam/core/document_loader_v2.py` (1.1 KB)

Clean document loading:

```python
@dataclass
class Sentence:
    """One sentence from documentation."""
    number: int  # 1-indexed
    text: str

def load_sentences(doc_path: str) -> list[Sentence]:
    """Load sentences (one per line)."""
    ...

def build_sent_map(sentences: list[Sentence]) -> dict[int, Sentence]:
    """Build number → Sentence mapping."""
    ...
```

### `src/llm_sad_sam/pcm_parser_v2.py` (1.5 KB)

Clean PCM parser:

```python
@dataclass
class ArchitectureComponent:
    """PCM model component."""
    id: str      # PCM ID (e.g., "_Gp6H8...")
    name: str    # entity name (e.g., "Dispatcher")

def parse_pcm_repository(model_path: str | Path) -> list[ArchitectureComponent]:
    """Extract components from .repository XML."""
    ...
```

## Linker Implementations

### ILinker Family

**ILinker1** (`ilinker1.py`, 13 KB):
- Three-pass precision cascade (extraction, actor, debate)
- Legacy variant, retained for ablation
- Used in early V26–V30 experiments

**ILinker2** (`ilinker2.py`, 10 KB):
- Two-pass explicit extractor (Pass A: extraction-framed, Pass B: actor-framed)
- High precision (~95%), lower recall (~76%)
- Seed baseline for S-Linker variants; output shape compatible with TransArc CSV
- Memory note: 86.5–87.9% macro F1 standalone; 92.7–95.7% with S-Linker pipeline

**ILinker3** (`ilinker3.py`, 8.3 KB):
- Adapter wrapper around ILinker2 using v2 data types
- Used as seed extractor in S-Linker11+ via `_seed_extraction()`
- Ensures data type compatibility between ILinker2 (v1) and S-Linker (v2)

### S-Linker Family

**S-Linker (v1)** (`s_linker.py`, 102 KB):
- Base DAG pipeline: 3-tier architecture (knowledge, link recovery, consolidation)
- Parallel execution of independent phases (ThreadPoolExecutor)
- V39 architecture (Mar 14, 2026): 94.5% macro F1
- Foundational design for all S-Linker 2–12 variants

**S-Linker2–S-Linker6** (`s_linker2.py`–`s_linker6.py`, 45–109 KB each):
- Incremental refinements post-V39
- S-Linker3: unified coref + keep_coref flag, 94.0% macro F1
- S-Linker6: simplified (no subprocess learning, no targeted recovery), 94.9% macro F1
- Each variant represents a single experimental change (2–5pp sensitivity)

**S-Linker7 variants** (`s_linker7.py`, `s_linker7a.py`, `s_linker7b.py`, ~46 KB each):
- Removes convention filter + FN recovery (ICSE simplification)
- S-Linker7a: partials through validation (Pareto win), 93.1% macro F1

**S-Linker8–S-Linker9e** (`s_linker8.py`–`s_linker9e.py`, ~39–44 KB each):
- Phase 3 ablations on prompts + code heuristics
- S-Linker8: truncation fix + CamelCase rescue, 93.1% macro F1
- S-Linker9a–9e: Heuristic removal experiments (CamelCase, synonym injection)
- S-Linker9: 3 safe removals, 93.9% macro F1

**S-Linker10** and **S-Linker10a** (~43–45 KB):
- Alias-context validation + evidence-stratified voting
- prompts_v2.py integration (clean constants)
- S-Linker10a: count>=3 enrichment threshold (ablation variant)
- S-Linker10 (Mar 19): 95.9% macro F1 (best on current suite)

**S-Linker11 variants** (`s_linker11.py`, `s_linker11a.py`–`s_linker11e.py`, ~47–71 KB):
- Alias stratification experiments (strong global vs weak local)
- S-Linker11b: evidence bundles + structured debate, 95.0% macro F1
- S-Linker11c: evidence bundles + debate on rejects, 71 KB
- S-Linker11d: no partial injection (ablation)
- S-Linker11e: evidence bundles in validation (no debate)
- Unified coref (Variant E) + deterministic antecedent validation

**S-Linker12 variants** (`s_linker12a.py`, `s_linker12b.py`, `s_linker12c.py`–`s_linker12e.py`, ~47–60 KB):
- S-Linker12a: alias stratification + no partial injection
- S-Linker12b: ICSE candidate (alias + evidence bundles)
- **S-Linker12c** (Mar 27, 53 KB): **Current production version**
  - Clean unified aliases + parallel Tier 1
  - Structural guardrails (CamelCase, dotted-path, pronoun pattern)
  - LLM-driven decisions with lightweight overrides
  - 94.8% macro F1 (MS 95.4%, TS 94.5%, TM 93.9%, BBB 89.9%, JAB 97.3%)
  - Used in ICSE submission
- S-Linker12d: Trailing-word enrichment variant (seed disambiguation prompt)
- S-Linker12e: Trailing-word + seed disambiguation variant (47.8 KB, Apr 4)

## Prompt Modules

### `src/llm_sad_sam/linkers/experimental/prompts.py` (17.8 KB)

v1 API — older prompt constants for ILinker and early S-Linker:
- Unstructured, verbose
- Mixed examples from various domains (some with benchmark risk)
- Used by ILinker1, ILinker2, S-Linker–S-Linker10a

**Key constants**:
- `EXTRACTION_PROMPT`, `ACTOR_PROMPT`, `DEBATE_PROMPT`
- `VALIDATION_PROMPT`, `JUDGE_PROMPT`
- Example-based few-shots with custom wording

### `src/llm_sad_sam/linkers/experimental/prompts_v2.py` (15.4 KB)

v2 API — clean, audited prompt constants for S-Linker11+:
- Organized by tier (Tier 1 model analysis, Tier 2 extraction, Tier 3 judgment)
- All examples from safe SE domains (compiler, OS, e-commerce, graphics)
- Zero benchmark component names (passed taboo audit Feb 26)
- Clean constants: `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, etc.

**Tier 1 (Model Analysis)**:
- `AMBIGUITY_FEW_SHOT`: Examples of architectural vs ambiguous names
- `AMBIGUITY_RULES`: Definition and examples

**Tier 2 (Link Recovery)**:
- `ENTITY_EXTRACTION_RULES`: How to find explicit mentions
- `VALIDATION_RULES`: When to accept a candidate link
- `COREF_RULES`: Pronoun resolution guidelines

**Tier 3 (Consolidation)**:
- `DOC_KNOWLEDGE_JUDGE_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`: Alias judgment
- Removed dead constants (CONVENTION_GUIDE, WORD_USAGE_PROMPT)

## LLM Client (`src/llm_sad_sam/llm_client.py`, 45 KB)

Unified backend abstraction:

```python
class LLMBackend(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    CODEX = "codex"
    CHECKPOINT = "checkpoint"

class LLMClient:
    def __init__(backend: LLMBackend, model: str, temperature: float, ...):
        ...

    def query(system_prompt: str, user_message: str) -> LLMResponse:
        """Single stateless query."""

    def start_conversation() -> None: ...
    def query_conversation(message: str) -> LLMResponse: ...
    def end_conversation() -> None: ...
```

**Features**:
- Per-query JSON logging (request, response, tokens, latency)
- Cumulative token tracking across all instances
- Conversation mode for multi-turn reasoning
- Temperature control (for OpenAI)
- Checkpoint fallback (cached responses from prior runs)

**Backends**:
- **Claude CLI**: `claude -p` subprocess with JSON parsing
- **OpenAI API**: Direct API calls with `openai` SDK (requires OPENAI_API_KEY)
- **Codex CLI**: Legacy Codex interface
- **Checkpoint**: Cached LLM responses from `results/llm_checkpoint/`

## Tests (`tests/test_*.py`)

Pytest test suite for variant validation and ablation:

### `test_12e_variance.py` (16 KB)
- **Purpose**: Measure run-to-run consistency of S-Linker12e (LLM variance impact)
- **Pattern**: Run 3–5 trials on single dataset, compare link sets
- **Output**: Variance statistics (stdev, max-min) per dataset

### `test_checkpoint_migration.py` (11.2 KB)
- **Purpose**: Verify cross-linker checkpoint loading compatibility
- **Pattern**: Run S-Linker12b, checkpoint after phase N, resume in S-Linker12c
- **Validates**: Phase checkpoints serializable/deserializable across variants

### `test_mention_type_ablation.py` (11.4 KB)
- **Purpose**: Analyze evidence bundle mention-type classification
- **Pattern**: Extract phase state, categorize mentions (proper case, lowercase, alias)
- **Validates**: Mention type discovery correctness (critical for evidence-stratified voting)

### `test_voting_analysis.py` (12.1 KB)
- **Purpose**: Analyze Phase 2 entity extraction voting (dual-pass consensus)
- **Pattern**: Compare Pass A vs Pass B independently, then intersection
- **Validates**: Voting strategy effectiveness (false positives filtered by intersection)

### `test_voting_asymmetry.py` (10.7 KB)
- **Purpose**: Detect asymmetric extraction behavior (pass order dependency)
- **Pattern**: Run linker with Pass A→B vs B→A (if supported), measure link set difference
- **Validates**: Voting order independence (important for reproducibility)

## Results Artifacts

### `results/ablation_results/`

Output of `run_ablation.py` runs:

```
results/ablation_results/
├── {variant}_{dataset}_links.csv      CSV output: sentence_number, component_id, component_name
├── {variant}_{dataset}.pkl            Pickle checkpoint: full linker state (phases, candidates)
├── {variant}_{dataset}.log            (optional) detailed phase execution log
└── summary_{timestamp}.csv            Aggregate metrics table (P/R/F1 across all variants/datasets)
```

**CSV Format** (example):
```
sentence_number,component_id,component_name,confidence,source
1,_Gp6H8E,Dispatcher,1.0,seed
1,_X2yK9Q,RequestHandler,1.0,seed
2,_X2yK9Q,RequestHandler,1.0,coref
```

### `results/llm_logs/`

Per-query LLM interaction logs (323 MB total):

```
results/llm_logs/
├── llm_{timestamp}_{variant}_{dataset}_phase1_000.json
├── llm_{timestamp}_{variant}_{dataset}_phase2_000.json
├── llm_{timestamp}_{variant}_{dataset}_phase3_001.json
└── ...
```

Each log file:
```json
{
  "timestamp": "2026-04-20T22:43:15.123Z",
  "variant": "s_linker12c",
  "dataset": "mediastore",
  "phase": 2,
  "batch": 0,
  "system_prompt": "Discover abbreviations...",
  "user_message": "DOCUMENT:\nSentence 1: The Dispatcher...",
  "response": "{\"abbreviations\": {...}}",
  "tokens": {"prompt": 512, "completion": 256, "total": 768},
  "latency_ms": 2341,
  "model": "claude-sonnet"
}
```

### `results/phase_cache/` and variants

Intermediate checkpoint storage:

```
results/phase_cache/
└── {variant}_{dataset}/
    ├── phase1_model_knowledge.pkl
    ├── phase2_document_knowledge.pkl
    ├── phase3_seed_links.pkl
    ├── phase4_validated_seed.pkl
    ├── phase5_entity_candidates.pkl
    └── ...
```

Used for:
- Checkpoint resumption (`--resume-from-phase` flag)
- Single-phase ablation (skip LLM, load cached input state)
- Cross-linker checkpoint migration tests

## Archive & Documentation

### `archive/`

Older linker families (deprecated, retained for historical reference):

```
archive/
├── linkers/                 40+ archived variants (archiv01–archiv43)
├── test_scripts/            Ad-hoc test scripts from development
├── README.md                Metadata on archived versions
└── ...
```

All archived variants follow naming: `archiv{NN}.py`. Entry point via `run_ablation.py` does NOT include these — only CANONICAL_VARIANTS are runnable.

### `doc/`

Development documentation notes.

### `paper/`

Paper draft artifacts (LaTeX, figures).

### `.planning/`

```
.planning/
├── codebase/                ← You are here (ARCHITECTURE.md, STRUCTURE.md)
├── spikes/                  Earlier research spike documentation
└── ...
```

## Naming Conventions

### Linker Versions

**Format**: `{linker_family}_{version}{optional_ablation}`

Examples:
- `s_linker` = S-Linker v1 (base)
- `s_linker2` = S-Linker v2 (first refinement)
- `s_linker10` = S-Linker v10 (10 refinements)
- `s_linker11a` / `s_linker11b` = S-Linker v11 ablation variants (A, B)
- `s_linker12c` = S-Linker v12 primary variant (C)
- `ilinker1`, `ilinker2`, `ilinker3` = ILinker variants

**Numbering**:
- Major versions: s_linker N (1–12)
- Ablations: s_linker Na, s_linker Nb, s_linker Nc (lowercase a–e for variants)
- Minor refinements: incremental major version bump

### Data Type Versions

**Suffix `_v2`**:
- `data_types_v2.py`: Lean S-Linker11+ standard (2KB)
- `data_types.py`: Legacy v1 (6KB, with dead fields)
- `document_loader_v2.py`: Clean S-Linker11+ standard (1KB)
- `document_loader.py`: Legacy v1 (6KB)
- `pcm_parser_v2.py`: Clean S-Linker11+ standard (1.5KB)
- `pcm_parser.py`: Legacy v1 (2KB)

**Prompt Versions**:
- `prompts.py`: v1 (old, unstructured)
- `prompts_v2.py`: v2 (new, organized by tier, audited)

### Output Files

**Pattern**: `{variant}_{dataset}_{extension}`

Examples:
- `s_linker12c_mediastore_links.csv` — final output
- `s_linker12c_mediastore.pkl` — checkpoint
- `s_linker12c_mediastore.log` — execution log

### Benchmark Datasets

5 projects (from parent directory `llm-sad-sam-v45/../ardoco/core/tests-base/src/main/resources/benchmark/`):
- `mediastore` (MS)
- `teastore` (TS)
- `teammates` (TM)
- `bigbluebutton` (BBB)
- `jabref` (JAB)

Each dataset: `doc.txt`, `model.repository`, `goldstandard_sad_sam_*.csv`
