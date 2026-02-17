# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llm-sad-sam-v45** is a Python tool that uses LLMs to recover trace links between Software Architecture Documents (SAD) and Software Architecture Models (SAM). It refines baseline TransArc links through a 10-phase LLM-driven pipeline with discourse-aware coreference resolution and implicit reference detection.

Part of the ARDoCo research ecosystem. See the parent directory's `CLAUDE.md` for broader project context.

## Build & Run Commands

```bash
# Install in development mode
pip install -e ".[dev,openai]"

# Run tests
pytest

# Run a single test
pytest tests/test_foo.py::test_bar

# Run the CLI (entry point defined but not yet implemented)
llm-sad-sam [options]
```

Python 3.11+ required. Dependencies: click, lxml, rapidfuzz. Optional: openai SDK.

## Architecture

### 10-Phase Pipeline

The linker (`TransArcRefinedLinkerV45`) processes documents through sequential phases, each building on prior results:

| Phase | Purpose | Output |
|-------|---------|--------|
| 0 | Document profiling (complexity, pronoun ratio) | `DocumentProfile` |
| 1 | Model structure analysis | `ModelKnowledge` |
| 2 | Pattern learning with LLM debate | `LearnedPatterns` |
| 3 | Document knowledge with judge validation | `DocumentKnowledge` |
| 4 | TransArc baseline link processing | Initial links |
| 5 | LLM entity extraction | Candidate links |
| 6 | Self-consistency validation (two-pass voting) | Validated links |
| 7 | Discourse-aware coreference resolution | Coref links |
| 8 | Implicit reference detection | Implicit links |
| 9 | Agent-as-Judge review | Approved links |
| 10 | Adaptive false-negative recovery | Final links |

Entry point: `TransArcRefinedLinkerV45.link(text_path, model_path, transarc_csv=None) → list[SadSamLink]`

### Module Layout

- **`core/data_types.py`** — All shared dataclasses (`SadSamLink`, `CandidateLink`, `DocumentProfile`, `ModelKnowledge`, `DocumentKnowledge`, `LearnedPatterns`, `DiscourseContext`, etc.) and enums (`LinkSource`)
- **`core/document_loader.py`** — Loads documentation sentences (one per line), TransArc baseline CSVs, paragraph detection, context windowing
- **`core/model_analyzer.py`** — Parses PCM models via `pcm_parser`, classifies components (architectural vs ambiguous) using LLM, extracts implementation patterns and shared vocabulary
- **`pcm_parser.py`** — XML parsing of Palladio Component Model `.repository` files using lxml
- **`llm_client.py`** — LLM abstraction with three backends and two modes (see below)
- **`linkers/experimental/transarc_refined_linker_v45.py`** — The V45 linker implementation (~1073 lines)

### LLM Client

Supports three backends selected via `LLM_BACKEND` env var:
- **`claude`** (default) — Calls `claude -p --output-format json --dangerously-skip-permissions` as subprocess
- **`openai`** — Uses OpenAI SDK (temperature 0.1, max 4096 tokens, exponential backoff retry)
- **`codex`** — Calls `codex exec` as subprocess

Two operation modes:
1. **Stateless**: `query(prompt)` — independent calls
2. **Conversation**: `start_conversation()` → `query_conversation()` → `end_conversation()` — maintains message history, trims at 50K tokens

All queries are logged to JSONL files with token usage tracking.

### Key Design Patterns

- **Adaptive thresholds**: Document profile (learned in Phase 0) determines strictness level and per-category confidence thresholds for coref, validation, FN recovery, and disambiguation
- **Multi-pass voting**: Phases 6 and 10 use two independent LLM passes that vote on uncertain candidates
- **Debate + Judge**: Phase 2 uses debate (two LLM passes cross-validate patterns); Phase 3 uses a judge to validate document knowledge
- **Discourse tracking**: `DiscourseContext` maintains recent mentions, paragraph topic, and active entity per sentence for Phases 7-8
- **Batch processing**: Coreference (12/batch), implicit refs (10/batch), entity extraction (100 sentences/batch), validation (25/batch)

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_BACKEND` | Backend: "claude", "openai", "codex" | "claude" |
| `OPENAI_API_KEY` | Required for OpenAI backend | — |
| `OPENAI_MODEL_NAME` | OpenAI model | "gpt-5.2" |
| `CLAUDE_MODEL` | Claude CLI model override | — |
| `LLM_LOG_DIR` | Query log directory | "./results/llm_logs" |
| `LLM_SESSION_DIR` | Working directory for CLI subprocesses | — |

## LLM Model

Always use **Claude Sonnet** as the LLM model unless explicitly told otherwise. Never use Opus — it is too expensive for the pipeline's many LLM calls. Ensure `CLAUDE_MODEL` is set to `"sonnet"` (via `os.environ` or env var) before any linker runs.

## Evaluation

This project is evaluated against ARDoCo's gold standards. See the parent `CLAUDE.md` for evaluation details (the `metrics_output/evaluate.py` script, gold standard enrollment for code links, expected values from TLR test files).
