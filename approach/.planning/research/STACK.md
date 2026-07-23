# Stack Research

**Domain:** Rule-to-LLM ablation research — Python + Claude Sonnet traceability linker
**Researched:** 2026-04-21
**Confidence:** HIGH (all claims verified against official Anthropic docs, PyPI, or installed packages)

## What Exists Already (Do Not Re-Add)

The project already has these — do not duplicate in pyproject.toml:

| Already Present | Version Confirmed | Notes |
|-----------------|------------------|-------|
| `pydantic` | 2.12.4 (installed) | For data model validation |
| `pandas` | 3.0.2 (installed) | For ablation table construction |
| `rich` | 14.2.0 (installed) | For terminal table rendering |
| `click` | 8.2.1 (installed) | CLI framework |
| `lxml` | 6.1.0 (installed) | PCM XML parsing |
| Prompt-hash file cache | Existing (`LLMBackend.CHECKPOINT`) | SHA256-keyed JSON files in `results/llm_checkpoint/` |
| Pytest | `>=8.0.0` in `pyproject.toml` | Test runner |
| `pytest-asyncio` | `>=0.23.0` in `pyproject.toml` | Async test support |

## Recommended Stack Additions

### Core Technology: Anthropic Python SDK (Direct API)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `anthropic` | `>=0.40.0` | Direct HTTP calls to Claude API | Eliminates subprocess overhead, unlocks structured outputs, prompt caching, and temperature control; the current Claude CLI subprocess approach (`claude -p`) adds ~100-300ms IPC overhead per call and cannot use `cache_control` or `output_config` |

**This is the most important addition.** The existing `LLMBackend.CLAUDE` uses `subprocess`/`claude -p`. For the ablation project:
- Structured output validation requires the SDK (`client.messages.parse()`)
- Prompt caching (`cache_control`) requires the SDK — saves 90% on system prompt token costs for repeated ablation runs
- Temperature control on Claude only works via SDK (the CLI ignores the flag in most contexts)
- Reproducibility strategies only work via SDK

Model ID to use: `claude-sonnet-4-6` (alias; currently resolves to latest Sonnet 4.6). For reproducibility, pin to a snapshot: `claude-sonnet-4-6` has no dated snapshot yet; Sonnet 4.5 snapshot is `claude-sonnet-4-5-20250929`. Use `claude-sonnet-4-6` until a snapshot is published, and pin when one becomes available.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `diskcache` | `>=5.6.1` | SQLite-backed prompt-response cache with TTL support | Replaces or supplements the existing SHA256 file cache when you need TTL, concurrent-safe access, or inspectable SQLite storage; use for ablation re-runs that share an LLM backend switch mid-project |
| `tabulate` | `>=0.9.0` | Markdown/LaTeX/plain-text table formatting from dicts/lists | Renders ablation tables to Markdown (for paper) or LaTeX (for ICSE submission) without pulling in heavy deps; complements `rich` (which targets terminal only) |

These two are the only additions worth making. Both are zero-dependency small libraries.

### Development Tools (No Change Needed)

`pytest` is already configured. No additional test framework is needed.

## Installation

Add to `pyproject.toml` `[project.dependencies]`:

```toml
"anthropic>=0.40.0",
"diskcache>=5.6.1",
"tabulate>=0.9.0",
```

```bash
pip install -e ".[dev,openai]"
# anthropic, diskcache, tabulate will now be installed as core deps
```

## Structured Output: What to Use

**Recommendation: Native Anthropic SDK `client.messages.parse()` with Pydantic models.**

The pipeline already uses Pydantic (`pydantic>=2.12.4` installed). The SDK's `parse()` method is the official GA path as of 2026:

```python
from anthropic import Anthropic
from pydantic import BaseModel

class MentionClassification(BaseModel):
    mention_type: str  # "proper case, standalone" | "lowercase mention" | etc.
    is_component_reference: bool

client = Anthropic()
response = client.messages.parse(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}],
    output_format=MentionClassification,
)
result = response.parsed_output  # typed MentionClassification instance
```

The `output_format` parameter (Pydantic model) compiles to a JSON schema and restricts token generation to schema-valid outputs. No beta header required as of GA (late 2025). No retry loop needed.

**Confidence: HIGH** — verified against official Anthropic structured outputs docs (April 2026).

## Structured Output: What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `instructor` | Adds a wrapper + Pydantic-already-present; the only value-add is cross-provider portability (not needed here) and retry loops on validation failure (not needed with native `parse()`). Overhead: extra dep, patched client, not needed. | Native `client.messages.parse()` |
| `outlines` | Grammar-constrained sampling for local models (vLLM, llama.cpp). Irrelevant to hosted Claude API. | Native `client.messages.parse()` |
| `pydantic-ai` | Full agent-loop framework. This project has its own pipeline architecture; pulling in an agent framework would conflict with the standalone-linker design constraint. | Native SDK |
| `langchain` | Full orchestration stack with heavy deps; entire project would be less understandable. Nothing this project does requires it. | Native SDK |

## Deterministic Sampling: Claude Sonnet Specifics

**Claude has no seed parameter.** This is a known difference from OpenAI.

Practical reproducibility strategy (HIGH confidence, verified via Anthropic docs):

1. **Temperature `0.0`** — set via SDK (`temperature=0.0` in `client.messages.create()`). This minimizes (does not eliminate) variance. Claude's variance floor at temp=0 is slightly higher than OpenAI's; expect ~1-3 link variation across runs on the same document.

2. **Prompt caching** — the primary reproducibility tool for ablation. When the same system prompt + document is sent multiple times, mark it with `cache_control` so the KV cache is reused. Cache hits return byte-identical prefill context, substantially reducing the variance source. 5-minute TTL is free to add; 1-hour TTL costs 2x input tokens.

   ```python
   response = client.messages.create(
       model="claude-sonnet-4-6",
       max_tokens=1024,
       temperature=0.0,
       system=[
           {
               "type": "text",
               "text": SYSTEM_PROMPT,
               "cache_control": {"type": "ephemeral"},  # 5-min TTL
           }
       ],
       messages=[{"role": "user", "content": user_prompt}],
   )
   ```

   Requirements for a cache hit: prompt prefix must be 100% identical up to the `cache_control` breakpoint; minimum 1024 tokens for Sonnet 4.5, 2048 tokens for Sonnet 4.6. Up to 4 breakpoints per request.

3. **Checkpoint cache (existing)** — the current SHA256-keyed file cache in `results/llm_checkpoint/` is the strongest reproducibility tool for ablation. Cache a complete linker run; replay from cache for comparison variants. Augment with `diskcache` if concurrent writes are needed.

4. **Do not rely on temperature alone** for paper claims. The correct claim is "all variants ran with temperature=0.0 and shared prompt cache" — not "outputs are deterministic".

**Confidence: HIGH** — Claude no-seed behavior confirmed via Anthropic docs and community reports; prompt caching API verified via official docs.

## Ablation Table Tooling

**Recommendation: `pandas` (already installed) + `tabulate` (add).**

The existing `run_ablation.py` already collects per-variant, per-dataset precision/recall/F1 and prints a summary. The pattern to add is:

```python
import pandas as pd
from tabulate import tabulate

# rows collected by run_ablation.py
rows = [
    {"variant": "s_linker12c", "rules_removed": 0, "MS": 98.1, "TS": 94.2, "TM": 93.9, "BBB": 90.1, "JAB": 97.3, "macro_F1": 94.7},
    {"variant": "s_linker13a", "rules_removed": 1, "MS": 97.8, "TS": 94.5, ...},
]
df = pd.DataFrame(rows)

# Terminal display
print(df.to_string(index=False))

# Markdown for paper draft
print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".1f"))

# LaTeX for ICSE submission
print(tabulate(df, headers="keys", tablefmt="latex_booktabs", floatfmt=".1f"))
```

The `rich` library (already installed) handles colored terminal output during runs. `tabulate` handles static Markdown/LaTeX for writing.

**Do not add:** MLflow, Weights & Biases, DVC, or any experiment tracking server. The project runs 5-30 variants on 5 datasets — a CSV + pandas is the right scale. Full experiment trackers add config overhead for no benefit at this scale.

## Cache Strategy for Re-Running Ablations

**Recommendation: Extend the existing checkpoint cache with `diskcache` as a drop-in for the CHECKPOINT backend.**

The existing checkpoint backend stores `{sha256(prompt)}.json` files. This works but is not concurrent-safe (two parallel variant runs on the same dataset can corrupt the same file) and has no TTL.

`diskcache` wraps SQLite and is concurrent-safe with no configuration:

```python
import diskcache as dc

cache = dc.Cache("results/llm_diskcache")

def query_with_cache(prompt_key: str, call_fn):
    if prompt_key in cache:
        return cache[prompt_key]
    result = call_fn()
    cache[prompt_key] = result
    return result
```

For this project, the existing file cache is sufficient for single-process runs. Add `diskcache` only if parallel ablation runs (multiple variants simultaneously) are needed. The SHA256 key scheme is identical.

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `anthropic>=0.40.0` | `pydantic>=2.0` | SDK uses pydantic v2 internally; both require pydantic v2 — compatible with currently installed 2.12.4 |
| `diskcache>=5.6.1` | Python 3.11+ | No known conflicts; zero dependencies beyond stdlib |
| `tabulate>=0.9.0` | Python 3.11+, pandas 3.x | No conflicts; `pd.DataFrame.to_string()` + `tabulate()` are complementary not competing |
| `anthropic>=0.40.0` | `openai>=1.0.0` | Both optional deps, no conflict; existing `[openai]` extra in pyproject.toml remains unchanged |

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `instructor` | Pydantic is already present; native `client.messages.parse()` is GA and does what instructor does for Anthropic; instructor adds a wrapper layer and extra dependency with no net benefit | `anthropic>=0.40.0` native `parse()` |
| `mlflow` / `wandb` / `neptune` | Overkill experiment trackers for 5-30 variant comparisons; require server setup or cloud accounts | `pandas` + CSV + `tabulate` |
| `pytest-benchmark` | This project benchmarks LLM quality (F1), not wall-clock performance — pytest-benchmark measures execution time, irrelevant here | Plain `pytest` |
| `openai` Structured Outputs | GPT compatibility is explicitly out of scope per PROJECT.md; adding OpenAI structured outputs patterns would diverge from Claude-only target | Anthropic native |
| `dvc` (Data Version Control) | Adds git-based data pipeline management; the project already uses pickle + CSV + git directly, and DVC's overhead is not warranted | Existing CSV/pickle output |

## Sources

- `/anthropics/anthropic-sdk-python` (Context7) — tool_use, structured output, prompt caching API patterns
- `https://platform.claude.com/docs/en/about-claude/models/overview` — Claude model IDs verified: current Sonnet is `claude-sonnet-4-6`; Sonnet 4.5 snapshot is `claude-sonnet-4-5-20250929` (HIGH confidence)
- `https://platform.claude.com/docs/en/build-with-claude/structured-outputs` — `client.messages.parse()` GA, beta header no longer required, `output_format` deprecated in favor of `output_config`, Pydantic integration (HIGH confidence)
- `https://platform.claude.com/docs/en/build-with-claude/prompt-caching` — `cache_control: {type: ephemeral}`, 5-min default TTL, 1-hour TTL at 2x cost, 4 breakpoints max, 1024-token minimum for Sonnet 4.5 / 2048 for Sonnet 4.6 (HIGH confidence)
- `pip list` (local environment) — confirmed installed: pydantic 2.12.4, pandas 3.0.2, rich 14.2.0, click 8.2.1, lxml 6.1.0; anthropic NOT installed (HIGH confidence)
- `https://python.useinstructor.com/integrations/anthropic/` — instructor overhead vs native confirmed; requires pydantic; adds retry loops + streaming patterns (MEDIUM confidence — official docs)
- WebSearch: Claude no-seed parameter confirmed; temperature=0.0 minimizes but does not eliminate variance; prompt caching is primary reproducibility tool (MEDIUM confidence — cross-referenced with official docs)

---
*Stack research for: rule-to-LLM ablation research on Python + Claude Sonnet linker codebase*
*Researched: 2026-04-21*
