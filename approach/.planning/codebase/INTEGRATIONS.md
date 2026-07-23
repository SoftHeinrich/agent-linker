# Integrations

## LLM Backends

**Three supported backends** (enum `LLMBackend` in `src/llm_sad_sam/llm_client.py`):

### 1. Claude (Default)
- **Type**: Subprocess CLI invocation
- **Command**: `claude -p --output-format json --dangerously-skip-permissions [--model MODEL] PROMPT`
- **Model**: Controlled by `CLAUDE_MODEL` env var (defaults to `"sonnet"`, typically Claude 3.5 Sonnet)
- **Features**:
  - Stateless queries via `.query(prompt)`
  - Conversation mode with `--resume` for session persistence (tracking `_claude_resume_id`)
  - Output parsed from JSON stream or single JSON object
  - Fallback to raw stdout if JSON parse fails
- **Working Directory**: Subprocess runs in `~/.llm-sad-sam/sessions/{session_id}` to isolate CLI state
- **Env Var**: `LLM_BACKEND=claude` (default if unset)

### 2. OpenAI (GPT-5.2)
- **Type**: REST API via `openai` SDK (installed with `pip install -e ".[openai]"`)
- **Model**: `OPENAI_MODEL_NAME` env var (defaults to `"gpt-5.2"`)
- **API Key**: `OPENAI_API_KEY` (required; no fallback)
- **Features**:
  - System prompt: "You are a helpful assistant that analyzes software architecture documents..."
  - Stateless: `.query(prompt)` → single API call
  - Conversation mode: full message history sent each call (auto-trim if >50k tokens)
  - Seed=42 for reproducibility; temperature configurable (default 0.1)
  - Token usage tracked: `prompt_tokens`, `completion_tokens`, `total_tokens`
  - Connection pooling via httpx (max 20 connections, 5 keepalive)
  - Optional service tier via `OPENAI_SERVICE_TIER` env var
- **Retry Logic**: Exponential backoff (2s, 3s, 5s) for transient errors (timeout, rate limit, 5xx)
- **Env Var**: `LLM_BACKEND=openai`

### 3. Checkpoint (Cache-to-Fallback)
- **Type**: Prompt-hash cache with fallback delegation
- **Cache Key**: SHA-256 hash of prompt text → `{CHECKPOINT_DIR}/{hash}.json`
- **Cache Hit**: Returns instantly (latency_ms=0); no LLM call
- **Cache Miss**: Delegates to fallback backend (Claude or OpenAI), saves response to cache, returns result
- **Fallback Config**: `CHECKPOINT_FALLBACK` + `CHECKPOINT_FALLBACK_MODEL` env vars
- **Use Case**: Ablation studies with repeated prompts across linker variants
- **Env Var**: `LLM_BACKEND=checkpoint`

### 4. Codex (Legacy)
- **Type**: Subprocess CLI invocation (not actively used in current variants)
- **Command**: `codex exec --skip-git-repo-check --json PROMPT`
- **Status**: Supported but superseded by Claude backend

**Backend Selection**:
- Command-line: `run_ablation.py` parses `LLM_BACKEND` env var in `get_backend()`
- Linker constructors: Accept `backend: LLMBackend | None` parameter (defaults to `LLMBackend.CLAUDE`)
- Fallback: If `LLM_BACKEND` unset and no param, defaults to `LLMBackend.CLAUDE`

## External Services

**No persistent external services** in current code. All integrations are:

### Claude CLI
- **Scope**: Local subprocess call (not a service)
- **Requirement**: `claude` command must be in PATH
- **Auth**: Relies on local Claude CLI authentication (via `~/.config/claude` or equivalent)
- **Failure**: Returns error "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"

### OpenAI REST API
- **Endpoint**: Default OpenAI API (`https://api.openai.com/v1/chat/completions`)
- **Auth**: `OPENAI_API_KEY` header
- **Scope**: Stateless HTTP POST per query; no persistent connection
- **Retry**: Built-in exponential backoff for transient failures

### No MCP / Microservices
- No references to MCP servers or external microservices in current codebase
- ARDoCo (parent project) mentions Stanford CoreNLP microservice for text preprocessing, but this linker does not call it

## Data Inputs

**Benchmark Datasets** (5 projects, located in sibling `../ardoco/core/tests-base/src/main/resources/benchmark/`):

### Per-Dataset Structure (from `run_ablation.py:DATASETS`):
```
{dataset}/
  text_YEAR/            SAD (software architecture documentation)
  model_YEAR/           SAM (architecture model)
  goldstandards/        Gold standard trace links (ground truth)
```

**Specific Paths**:

1. **mediastore**
   - Text: `benchmark/mediastore/text_2016/mediastore.txt` (plain text, one sentence per line)
   - Model: `benchmark/mediastore/model_2016/pcm/ms.repository` (XML, Palladio Component Model)
   - Gold: `benchmark/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv` (columns: `modelElementID`, `sentence`)

2. **teastore**
   - Text: `benchmark/teastore/text_2020/teastore.txt`
   - Model: `benchmark/teastore/model_2020/pcm/teastore.repository`
   - Gold: `benchmark/teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv`

3. **teammates**
   - Text: `benchmark/teammates/text_2021/teammates.txt`
   - Model: `benchmark/teammates/model_2021/pcm/teammates.repository`
   - Gold: `benchmark/teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv`

4. **bigbluebutton**
   - Text: `benchmark/bigbluebutton/text_2021/bigbluebutton.txt`
   - Model: `benchmark/bigbluebutton/model_2021/pcm/bbb.repository`
   - Gold: `benchmark/bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv`

5. **jabref**
   - Text: `benchmark/jabref/text_2021/jabref.txt`
   - Model: `benchmark/jabref/model_2021/pcm/jabref.repository`
   - Gold: `benchmark/jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv`

**TransArc Baseline** (optional):
- Path: `CLI_RESULTS / {dataset}-sad-sam / sadSamTlr_{dataset}.csv` (e.g., `../cli-results/mediastore-sad-sam/sadSamTlr_mediastore.csv`)
- Provides baseline performance for comparison (if file exists; skipped if missing)
- Not generated by this project; comes from external ARDoCo CLI run

**Data Loading** (from `src/llm_sad_sam/core/document_loader_v2.py`):
- **Text**: Plain text files; each line = one sentence (tokenized by `load_sentences(text_path)`)
- **Model**: XML `.repository` files; parsed by `pcm_parser_v2.parse_pcm_repository(model_path)` to extract components and their IDs
- **Gold**: CSV with headers `modelElementID`, `sentence` (1-indexed sentence numbers); loaded into `set[tuple[int, str]]` by `load_gold_sam(path)`

## Data Outputs

**Results Directory Structure** (defaults to `results/`):

```
results/
  ablation_results/
    {variant}_{dataset}_links.csv     per-variant per-dataset predictions (columns: sentence, component_id, component_name, confidence, source)
    ablation_{timestamp}.json         comprehensive ablation summary (all metrics, FP/FN details, timing)
  
  llm_logs/
    llm_session_{session_id}.log      per-session summary (one line per request)
    llm_requests_{session_id}.jsonl   detailed request/response log (full prompt preview, model, latency, token usage)
    usage_summary_{session_id}.json   token usage totals (prompt, completion, total tokens; session + cumulative)
  
  llm_checkpoint/
    {sha256}.json                     cached LLM response (prompt-hash keyed); JSON with text, success, error, model, token_usage
  
  phase_cache/                         (used internally by some linker variants for checkpoint resumption)
```

**Ablation Output Format** (`ablation_{timestamp}.json`):
```json
{
  "dataset_name": {
    "variant_name": {
      "variant": "s_linker12c",
      "P": 0.95,                      // precision
      "R": 0.93,                      // recall
      "F1": 0.94,                     // F1 score
      "tp": 120,                      // true positives
      "fp": 6,                        // false positives
      "fn": 8,                        // false negatives
      "n_links": 126,
      "time": 45.2,                   // seconds
      "sources": { "seed": 85, "entity": 30, "coref": 11 },  // link origin counts
      "fp_by_source": { "seed": 2, "coref": 4 },
      "fp_details": [{ "sentence": 42, "component": "AuthService", "source": "coref", "confidence": 0.87, "text": "..." }],
      "fn_details": [{ "sentence": 35, "component": "Database", "name_in_text": true, "transarc_had": false }]
    }
  }
}
```

**Per-Dataset CSV** (`{variant}_{dataset}_links.csv`):
```
sentence,component_id,component_name,confidence,source
12,comp_001,AuthService,0.95,seed
45,comp_012,MessageBroker,0.82,entity
```

## Auth/Secrets

**API Keys & Credentials**:

### OPENAI_API_KEY
- **Required for**: `LLMBackend.OPENAI` (optional; only needed if using OpenAI)
- **Format**: Bearer token (e.g., `sk-proj-...`)
- **Source**: Environment variable (no default)
- **Storage**: `.env` file (currently checked into git with a valid key — **SECURITY RISK** for public repos; should use `secrets.local` or CI/CD secrets)
- **Fallback**: If unset with OpenAI backend, raises `ValueError("OPENAI_API_KEY environment variable not set")`

### CLAUDE_MODEL / OPENAI_MODEL_NAME
- **Purpose**: Model selection, not authentication
- **Defaults**: `CLAUDE_MODEL="sonnet"`, `OPENAI_MODEL_NAME="gpt-5.2"`
- **Overridable**: Via env var or constructor parameter

### Claude CLI Auth
- **Mechanism**: Local CLI authentication (managed by `claude` command itself)
- **Storage**: Typically `~/.config/claude` or platform-specific auth cache
- **Requirement**: User must have run `claude login` or similar before first use
- **Failure**: Subprocess returns non-zero exit code or "unauthorized" error

**Token Usage Tracking**:
- **OpenAI**: Tokens returned in `response.usage` (prompt_tokens, completion_tokens)
- **Claude**: No token count exposed by CLI; estimated as `len(text) // 4` for logging purposes
- **Accumulation**: Class-level `LLMClient._cumulative_usage` tracks across all instances and sessions
- **Output**: Written to `usage_summary_{session_id}.json` and console via `print_usage_summary()`

**Logging & Observability**:
- **JSONL Request Log**: Every query logged to `llm_requests_{session_id}.jsonl` with full details (prompt preview, response preview, latency, token usage, error message)
- **Timestamps**: UTC ISO format for correlation
- **Session Isolation**: Each `LLMClient()` instance gets unique `session_id` (YYYYmmdd_HHMMSS)
- **Log Rotation**: No automatic cleanup; user responsible for archival
