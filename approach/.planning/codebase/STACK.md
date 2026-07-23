# Stack

## Language & Runtime

- **Python 3.11+** (from `pyproject.toml: requires-python = ">=3.11"`)
- CPython (standard Python runtime)
- Subprocess integration with external LLM CLIs (Claude Code CLI, Codex)
- No special runtime config; relies on environment variables for backend selection and model tuning

## Package Manager

- **pip** with `setuptools>=61.0` build backend
- **pyproject.toml** layout:
  - Build system: setuptools with `build-meta`
  - Project metadata: version 0.1.0, name `llm-sad-sam-agent`
  - Package discovery: `src/llm_sad_sam` layout
  - CLI entry point: `llm-sad-sam` command (maps to `llm_sad_sam.cli:main`)

- **Optional dependencies** (`[project.optional-dependencies]`):
  - `dev`: pytest>=8.0.0, pytest-asyncio>=0.23.0
  - `openai`: openai>=1.0.0 (for GPT-5.2 backend; optional, installed with `pip install -e ".[dev,openai]"`)

## Core Dependencies

Core dependencies (from `pyproject.toml: dependencies`):
- **click>=8.1.0** — CLI framework (for `llm-sad-sam` CLI command parsing)
- **lxml>=5.0.0** — XML parsing (for PCM repository parsing; `.repository` files are XML)
- **rapidfuzz>=3.0.0** — Fuzzy string matching (for entity name similarity in linking)

LLM client libraries (not pinned in dependencies, imported conditionally):
- **openai>=1.0.0** — Optional; enables `LLMBackend.OPENAI` (GPT-5.2 API calls) when installed
- **anthropic SDK** — Not listed; Claude backend uses subprocess CLI instead (`claude -p` command)
- **httpx** — HTTP client (bundled with openai SDK; used for connection pooling to OpenAI API)

NLP/Text processing:
- No built-in NLP library (e.g., no NLTK/spaCy); text splitting and sentence-level analysis handled manually
- Document parsing: custom `DocumentLoader` (in `src/llm_sad_sam/core/document_loader_v2.py`) reads plain-text files line-by-line as sentences

Parsers:
- **lxml** — Parses PCM `.repository` (XML) architecture model files via `src/llm_sad_sam/pcm_parser_v2.py`
- Custom parsers: `DocumentLoader`, `PCMParser` (internal)

## Dev Dependencies

- **pytest>=8.0.0** — Test runner; test discovery from `tests/test_*.py`
- **pytest-asyncio>=0.23.0** — Async test support (though primary code is sync)

## Configuration

**Environment Variables** (from `llm_client.py` and `run_ablation.py`):

### LLM Backend & Model Selection
- `LLM_BACKEND` — Backend choice: `"claude"` (default), `"openai"`, `"codex"`, `"checkpoint"`
- `CLAUDE_MODEL` — Claude model variant; defaults to `"sonnet"` (usually Claude 3.5 Sonnet)
- `OPENAI_MODEL_NAME` — OpenAI model; defaults to `"gpt-5.2"`
- `OPENAI_API_KEY` — Required for `LLMBackend.OPENAI`; no default
- `OPENAI_SERVICE_TIER` — Optional; sets OpenAI service tier (e.g., "default" or "pro")

### Checkpoint Backend (Cache-to-Fallback)
- `CHECKPOINT_DIR` — Cache directory for checkpoint responses; defaults to `./results/llm_checkpoint`
- `CHECKPOINT_FALLBACK` — Fallback backend if checkpoint misses; defaults to `"claude"`
- `CHECKPOINT_FALLBACK_MODEL` — Model name for fallback; e.g., `"gpt-5.2"`, `"sonnet"`

### Logging & Session Management
- `LLM_LOG_DIR` — Directory for LLM request/response logs (JSONL + summary); defaults to `./results/llm_logs`
- `LLM_SESSION_DIR` — Working directory for CLI subprocess sessions; defaults to `~/.llm-sad-sam/sessions`

### .env File
- **Location**: `run_ablation.py` checks for `.env` in project root
- **Format**: `KEY=VALUE` (one per line; comments with `#` ignored)
- **Auto-loaded**: `load_dotenv()` parses `.env` into `os.environ` at runner startup
- **Example (from current .env)**: Contains `OPENAI_API_KEY` (hardcoded in git — not ideal for production)

## Build/Run Commands

From `CLAUDE.md` and `README.md`:

**Installation**:
```bash
pip install -e ".[dev,openai]"
```
Installs package in editable mode with dev and OpenAI dependencies.

**Ablation Study (Main Entry)**:
```bash
# Default (retained variant s_linker11a, all 5 datasets)
python run_ablation.py

# Specific datasets
python run_ablation.py --datasets mediastore teastore

# Specific variants
python run_ablation.py --variants i1 i2 i3 s_linker s_linker11a

# List available
python run_ablation.py --list-variants
python run_ablation.py --list-datasets
```

**Backend Override**:
```bash
# Checkpoint with fallback
LLM_BACKEND=checkpoint python run_ablation.py --datasets mediastore

# OpenAI backend
LLM_BACKEND=openai OPENAI_API_KEY=sk-... python run_ablation.py --datasets mediastore

# Custom models
CLAUDE_MODEL=opus CLAUDE_MODEL_NAME=gpt-5.2 python run_ablation.py
```

**Testing**:
```bash
pytest
```
Runs all tests in `tests/` matching `test_*.py`; uses pytest.ini from `pyproject.toml`.

## Notes

**Version Pins & Constraints**:
- Python 3.11+ enforced (no upper bound specified; compatible with 3.12+)
- No version pins on core deps (click, lxml, rapidfuzz); allows flexible upgrades
- openai SDK >=1.0.0 required (major API change; <1.0 incompatible)

**Unusual Choices**:
- **Claude via subprocess CLI** — LLMClient calls `claude -p --output-format json` instead of using Anthropic Python SDK. Enables conversation state tracking via `--resume` flag but adds subprocess overhead.
- **Optional OpenAI SDK** — Installed conditionally; code gracefully falls back if `from openai import OpenAI` fails.
- **Custom text/document parsing** — No built-in NLP tooling; phrase/sentence analysis done via LLM prompts, not traditional tokenizers.
- **Checkpoint caching** — Prompt-hash-based file cache (`./results/llm_checkpoint/{sha256}.json`) to avoid duplicate LLM calls during ablation studies; fallback delegation keeps retry logic intact.
