# Phase 09 — GATE-06 Harness Audit

**Phase:** 09-cross-gpt-5-2-cross-model-validation
**Plan:** 09-01
**Audited:** 2026-05-31
**Auditor:** Plan 09-01 executor
**Purpose:** Establish that the harness layer (invocation entry point + model-adapter
shim + linker-module env defaults) carries no benchmark-derived branching, no
per-project special cases, and no per-model conditional logic before any
`gpt-5.4` LLM call is made (D-07, D-08, CROSS-01 success-criterion-4).

---

## 1. Scope

Three files are in scope. Body code of `s_linker13.py` was audited under
v1.0 Phase 5 PROMO and is not re-audited here; only the module-level env
defaults are revisited because they are part of the cross-model invocation
surface.

| File | Role | Audited Region |
|------|------|----------------|
| `src/llm_sad_sam/llm_client.py` | Model-adapter shim (backend dispatch, OpenAI/Claude request paths) | Whole file (full body audit) |
| `run_ablation.py` | Invocation entry point (backend resolver, DATASETS dict, model env defaults) | Whole file (full body audit) |
| `src/llm_sad_sam/linkers/experimental/s_linker13.py` | COMBINE artifact — module-level env defaults only | Lines 136–171 (variant-name constant + `__init__` env defaults). Body code was audited under v1.0 Phase 5 PROMO. |

Out of scope (already covered or different layer):

- `src/llm_sad_sam/linkers/experimental/prompts.py`, `prompts_v2.py` — prompt
  text, covered by Phase 6 `06-GATE-06-AUDIT.md` and Phase 8 unit re-audit.
- s_linker13.py body — covered under v1.0 Phase 5 PROMO.

---

## 2. Method

Four mechanical grep checks were executed against the in-scope files. Raw
output is pasted verbatim in §3 ("Evidence — Grep Output"). Each hit is then
classified ACCEPTABLE / VIOLATION in §4 ("Findings").

| # | Check | Command shape | Purpose |
|---|-------|---------------|---------|
| (a) | BENCHMARK_TABOO scan | `grep -nEi '<taboo-words>' <files>` | Surface any benchmark component name, alias, or universal-taboo word that has slipped into harness code. |
| (b) | Per-model conditional scan | `grep -nE 'gpt-?5\.[24]\|sonnet\|opus\|claude.*-[0-9]' <files>` | Surface every model id mentioned in code; flag if used to gate control flow (vs. resolving / composing a default). |
| (c) | Per-project conditional scan | `grep -nEi 'mediastore\|teastore\|teammates\|bigbluebutton\|jabref' <files>` | Surface every project mention; flag if used outside the data-plumbing layer (e.g., in model dispatch, OpenAI/Claude request body). |
| (d) | New prompt-file scan | `ls src/llm_sad_sam/linkers/experimental/prompts*.py` | Confirm only the two existing prompt files exist; flag any new `prompts_gpt*.py`, `prompts_openai*.py`, `prompts_v3.py`, or similar GPT-only variant. |

VIOLATION classes per BENCHMARK_TABOO.md §"Tailored Code Anti-Patterns" and D-04:

- Per-component regex tables / per-component synonym maps in harness code.
- Per-project model overrides (e.g., `if dataset == "bigbluebutton": model = ...`).
- Any benchmark surface form embedded in OpenAI/Claude request bodies (system prompt, user prompt prefix, error message, retry-classification keyword).
- Any new prompt-constant file targeting `gpt-5.4`.

---

## 3. Evidence — Grep Output

### 3a. BENCHMARK_TABOO scan — specific component names + ambiguous English overlap

```
$ grep -nEi '\b(UserDBAdapter|AudioWatermarking|Reencoding|MediaManagement|Facade|MediaAccess|Packaging|FileStorage|TagWatermarking|UserManagement|DownloadLoadBalancer|ParallelWatermarking|WebUI|Registry|Persistence|Recommender|SlopeOneRecommender|OrderBasedRecommender|DummyRecommender|PopularityBasedRecommender|ImageProvider|PreprocessedSlopeOneRecommender|GAE|kurento|WebRTC-SFU|HTML5|FSESL|FreeSWITCH|bbb|bibdatabase|bibentry|watermark|watermarking|reencoding|slope|datastore|pubsub|globals|preferences|cascade|dedicated)\b' src/llm_sad_sam/llm_client.py run_ablation.py src/llm_sad_sam/linkers/experimental/s_linker13.py
run_ablation.py:93:        description="ILinker1 three-pass precision cascade",
run_ablation.py:393:        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
src/llm_sad_sam/llm_client.py:155:        # Dedicated working directory for CLI subprocesses (avoids cwd side-effects)
```

```
$ grep -nEi '\b(watermark|recommender|persistence|registry|facade|adapter|recording|conversion|kurento|freeswitch|redis|html5|gui|cli|preferences|bibdatabase|bibentry)\b' src/llm_sad_sam/llm_client.py run_ablation.py src/llm_sad_sam/linkers/experimental/s_linker13.py
run_ablation.py:103:        adapter="ilinker3",
run_ablation.py:104:        description="ILinker3 v2-stack extractor adapter",
run_ablation.py:370:CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")
src/llm_sad_sam/llm_client.py:4:- codex: OpenAI Codex CLI (codex exec)
src/llm_sad_sam/llm_client.py:5:- claude: Local Claude Code CLI (claude -p)
src/llm_sad_sam/llm_client.py:82:                   For Claude CLI: passed as --model flag. Defaults to CLAUDE_MODEL env var (unset = CLI default).
src/llm_sad_sam/llm_client.py:155:        # Dedicated working directory for CLI subprocesses (avoids cwd side-effects)
src/llm_sad_sam/llm_client.py:165:        self._claude_resume_id: Optional[str] = None  # For Claude CLI --resume
src/llm_sad_sam/llm_client.py:799:        """Query using Codex CLI."""
src/llm_sad_sam/llm_client.py:830:        """Query using Claude Code CLI."""
src/llm_sad_sam/llm_client.py:836:            # Strip CLAUDECODE env var so nested CLI calls work
src/llm_sad_sam/llm_client.py:887:            return LLMResponse(text="", success=False, error="Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
```

### 3b. Per-model conditional scan

```
$ grep -nE 'gpt-?5\.[24]|sonnet|opus|claude.*-[0-9]' src/llm_sad_sam/llm_client.py run_ablation.py src/llm_sad_sam/linkers/experimental/s_linker13.py
run_ablation.py:436:os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")
run_ablation.py:437:os.environ.setdefault("CLAUDE_MODEL", "sonnet")
run_ablation.py:443:        return f"claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
run_ablation.py:445:        return f"openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.2')})"
run_ablation.py:449:            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
run_ablation.py:453:        if fallback_model in {"claude", "sonnet"} or fallback_model.startswith("claude"):
run_ablation.py:454:            model = os.environ.get("CLAUDE_MODEL", "sonnet")
run_ablation.py:455:            if fallback_model not in {"claude", "sonnet"}:
run_ablation.py:460:            return f"checkpoint -> openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.2')})"
run_ablation.py:463:        return f"checkpoint -> claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
src/llm_sad_sam/linkers/experimental/s_linker13.py:170:        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
src/llm_sad_sam/linkers/experimental/s_linker13.py:171:        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")
src/llm_sad_sam/llm_client.py:81:            model: Model name. For OpenAI: defaults to OPENAI_MODEL_NAME env var or "gpt-5.2".
src/llm_sad_sam/llm_client.py:90:                                       Examples: "sonnet", "gpt", "gpt-5.2".
src/llm_sad_sam/llm_client.py:112:        self.openai_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
src/llm_sad_sam/llm_client.py:113:        self.claude_model = os.environ.get("CLAUDE_MODEL", "sonnet")
src/llm_sad_sam/llm_client.py:200:            return LLMBackend.OPENAI, os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
src/llm_sad_sam/llm_client.py:202:            return LLMBackend.OPENAI, os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
src/llm_sad_sam/llm_client.py:206:        if lowered == "sonnet":
src/llm_sad_sam/llm_client.py:207:            return LLMBackend.CLAUDE, "sonnet"
src/llm_sad_sam/llm_client.py:209:            return LLMBackend.CLAUDE, os.environ.get("CLAUDE_MODEL", "sonnet")
```

### 3c. Per-project conditional scan

```
$ grep -nEi 'mediastore|teastore|teammates|bigbluebutton|jabref' src/llm_sad_sam/llm_client.py run_ablation.py src/llm_sad_sam/linkers/experimental/s_linker13.py
run_ablation.py:373:    "mediastore": {
run_ablation.py:374:        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
run_ablation.py:375:        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
run_ablation.py:376:        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
run_ablation.py:377:        "transarc_sam": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
run_ablation.py:379:    "teastore": {
run_ablation.py:380:        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
run_ablation.py:381:        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
run_ablation.py:382:        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
run_ablation.py:383:        "transarc_sam": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
run_ablation.py:385:    "teammates": {
run_ablation.py:386:        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
run_ablation.py:387:        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
run_ablation.py:388:        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
run_ablation.py:389:        "transarc_sam": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
run_ablation.py:391:    "bigbluebutton": {
run_ablation.py:392:        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
run_ablation.py:393:        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
run_ablation.py:394:        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
run_ablation.py:395:        "transarc_sam": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
run_ablation.py:397:    "jabref": {
run_ablation.py:398:        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
run_ablation.py:399:        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
run_ablation.py:400:        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
run_ablation.py:401:        "transarc_sam": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
```

### 3d. New prompt-file scan

```
$ ls src/llm_sad_sam/linkers/experimental/prompts*.py
src/llm_sad_sam/linkers/experimental/prompts.py
src/llm_sad_sam/linkers/experimental/prompts_v2.py
```

---

## 4. Findings

### 4a. BENCHMARK_TABOO scan

| File:Line | Hit | Classification | Reason |
|-----------|-----|----------------|--------|
| `run_ablation.py:93` | `"ILinker1 three-pass precision cascade"` (variant description string) | ACCEPTABLE | English word "cascade" used in a human-readable variant description; not a control-flow branch and not an LLM-visible string. No benchmark term smuggled. |
| `run_ablation.py:103` | `adapter="ilinker3"` (key in `VARIANT_SPECS` dict) | ACCEPTABLE | Internal variant-routing key naming the i3 adapter shim; English word "adapter" used in the generic software-engineering sense (wrapper/adapter pattern), not the MediaStore `UserDBAdapter` component. Not LLM-visible. |
| `run_ablation.py:104` | `"ILinker3 v2-stack extractor adapter"` (description string) | ACCEPTABLE | Same as above — generic SE-pattern usage in a description string. |
| `run_ablation.py:370` | `CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")` | ACCEPTABLE | "CLI" = Command-Line Interface (tool category), not the JabRef `cli` component. Path constant. |
| `run_ablation.py:393` | `"bigbluebutton/model_2021/pcm/bbb.repository"` (DATASETS path) | ACCEPTABLE | `bbb.repository` is the on-disk filename of the BigBlueButton PCM model. Pure data plumbing — see §4c. Not used in model dispatch or LLM request body. |
| `src/llm_sad_sam/llm_client.py:4` | `# - codex: OpenAI Codex CLI (codex exec)` (module docstring) | ACCEPTABLE | "CLI" = Command-Line Interface. Docstring only. |
| `src/llm_sad_sam/llm_client.py:5` | `# - claude: Local Claude Code CLI (claude -p)` | ACCEPTABLE | Same — "CLI" tool category. |
| `src/llm_sad_sam/llm_client.py:82` | `"For Claude CLI: passed as --model flag..."` (docstring) | ACCEPTABLE | Same — "CLI" tool category. |
| `src/llm_sad_sam/llm_client.py:155` | `# Dedicated working directory for CLI subprocesses` (code comment) | ACCEPTABLE | English word "dedicated" used adjectivally in a code comment; not LLM-visible and not branching. |
| `src/llm_sad_sam/llm_client.py:165` | `self._claude_resume_id: Optional[str] = None  # For Claude CLI --resume` | ACCEPTABLE | "CLI" tool category in comment. |
| `src/llm_sad_sam/llm_client.py:799` | `"""Query using Codex CLI."""` (docstring) | ACCEPTABLE | "CLI" tool category. |
| `src/llm_sad_sam/llm_client.py:830` | `"""Query using Claude Code CLI."""` (docstring) | ACCEPTABLE | "CLI" tool category. |
| `src/llm_sad_sam/llm_client.py:836` | `# Strip CLAUDECODE env var so nested CLI calls work` | ACCEPTABLE | "CLI" tool category. |
| `src/llm_sad_sam/llm_client.py:887` | error string `"Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"` | ACCEPTABLE | "CLI" tool category in an error message; surface form is "Claude CLI" (Anthropic tool name), not the JabRef `cli` component. Not embedded in any LLM prompt body. |

No taboo hit gates control flow, appears in an LLM request body, or encodes a per-(component, dataset) casing/synonym map. **No VIOLATION**.

### 4b. Per-model conditional scan

| File:Line | Hit | Classification | Reason |
|-----------|-----|----------------|--------|
| `run_ablation.py:436` | `os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")` | ACCEPTABLE | Default site — uses `setdefault`, so the env override (`OPENAI_MODEL_NAME=gpt-5.4` in Plan 09-02) wins. This is the documented gpt-5.4 override mechanism (D-01). No hardcoded gpt-5.2 dispatch logic. |
| `run_ablation.py:437` | `os.environ.setdefault("CLAUDE_MODEL", "sonnet")` | ACCEPTABLE | Same pattern for Claude; `setdefault` honours external env. |
| `run_ablation.py:443` | `f"claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"` | ACCEPTABLE | Human-readable backend descriptor (print line). No control flow. |
| `run_ablation.py:445` | `f"openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.2')})"` | ACCEPTABLE | Same — descriptor string. With `OPENAI_MODEL_NAME=gpt-5.4` set, this prints `openai (gpt-5.4)`. |
| `run_ablation.py:449–463` | `describe_backend_target` checkpoint fallback descriptor | ACCEPTABLE | String-formatting for a print statement. Branches on `fallback_model` to pick which env var to display; same `setdefault`-honouring pattern. No gpt-5.2-vs-5.4 dispatch. |
| `src/llm_sad_sam/linkers/experimental/s_linker13.py:170–171` | `os.environ.setdefault("CLAUDE_MODEL", "sonnet")` / `os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")` | ACCEPTABLE | Module-level env defaults inside `__init__`. `setdefault` ensures Plan 09-02's `OPENAI_MODEL_NAME=gpt-5.4` (set before import or in shell env) is respected. This is the third place where the gpt-5.4 override threads through; all three places use the same `setdefault` pattern, no hardcoded gpt-5.2. |
| `src/llm_sad_sam/llm_client.py:81, 82, 90` | docstrings mentioning `gpt-5.2`, `sonnet` | ACCEPTABLE | Docstring text only. Not executed. |
| `src/llm_sad_sam/llm_client.py:112` | `self.openai_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")` | ACCEPTABLE | Resolution of effective OpenAI model id. Reads env (Plan 09-02's `gpt-5.4` wins). The fallback string `"gpt-5.2"` is only used when env is unset. |
| `src/llm_sad_sam/llm_client.py:113` | `self.claude_model = os.environ.get("CLAUDE_MODEL", "sonnet")` | ACCEPTABLE | Same pattern, Claude side. |
| `src/llm_sad_sam/llm_client.py:200, 202` | `_infer_backend_from_model` — `"gpt"` / `"openai"` short-forms resolve to `LLMBackend.OPENAI` with effective env model | ACCEPTABLE | Backend inference; maps the human-friendly selector "gpt" to "whatever `OPENAI_MODEL_NAME` says". Cross-model run never invokes this with `"gpt"` short-form because Plan 09-02 sets the env directly. |
| `src/llm_sad_sam/llm_client.py:203` (`startswith("gpt")` branch) | resolves any `"gpt*"` model id to OpenAI route | ACCEPTABLE | Generic prefix dispatch — `gpt-5.4`, `gpt-5.2`, future `gpt-6` all route the same way. Treats `gpt-5.4` as a generic OpenAI model id. **No per-model special-casing.** |
| `src/llm_sad_sam/llm_client.py:206–211` | Claude-side parallel: `"sonnet"`, `"claude"`, `"claude*"` prefix dispatch | ACCEPTABLE | Same generic prefix pattern, Anthropic side. |

The OpenAI request body (`_query_openai`, lines 911–983 inspected) uses `self.openai_model` directly (line 926: `model=self.openai_model`) and contains exactly one system prompt string (lines 929–931: *"You are a helpful assistant that analyzes software architecture documents and extracts trace links between documentation and architecture models. Always respond with valid JSON when asked."*). The system prompt is model-agnostic and project-agnostic — no benchmark term, no component name, no per-project conditional. Retry logic (lines 967–979) classifies errors by generic keywords (`timeout`, `rate_limit`, `connection`, etc.), no benchmark-derived keyword.

The Claude request body (`_query_claude`, lines 829–889) uses `self.claude_model` via the CLI `--model` flag and forwards `prompt` verbatim. No system prompt is injected on the Claude path. No benchmark-derived branching.

`temperature=self.temperature` (default 0.1) and `seed=42` are constants applied uniformly — not per-model, not per-project. **No VIOLATION**.

### 4c. Per-project conditional scan

All 25 hits sit in the `DATASETS` dict (`run_ablation.py` lines 372–403), which is pure data plumbing: dataset name → (text, model, gold_sam, transarc_sam) file paths. Iteration over this dict in `main()` (lines 726–766) is uniform — every dataset is processed identically. No dataset name appears in any branching expression elsewhere in the audited files. No dataset name appears in any LLM prompt body or in any backend dispatch. **No VIOLATION**.

### 4d. New prompt-file scan

Only `prompts.py` and `prompts_v2.py` exist. No `prompts_gpt.py`, `prompts_openai.py`, `prompts_v3.py`, or any other GPT-only prompt-constant file is present. D-08 satisfied. **No VIOLATION**.

---

## 5. Verdict

Verdict: CLEAN — no benchmark-derived branching, no per-project special cases, no new GPT-only prompt files. Cross-model sweep on gpt-5.4 may proceed.

Justification: The harness layer's three Python files contain no per-component regex tables, no per-project model overrides, no benchmark surface forms in LLM request bodies, and no new GPT-only prompt files. Every model id mention is either (i) a `setdefault` fallback honouring external env, (ii) a human-readable descriptor string for print output, or (iii) a generic prefix dispatch (`startswith("gpt")` / `startswith("claude")`) that treats `gpt-5.4` as a generic OpenAI model id with no per-version special casing. Every project name mention is data plumbing in the `DATASETS` dict, iterated uniformly. The OpenAI system prompt is a single project-agnostic instruction with no benchmark term.

Plan 09-02 (BBB probe) and Plan 09-03 (full sweep + report) are unblocked.

---

## 6. GATE-06 Cross-References

- **Phase 6 `06-GATE-06-AUDIT.md`** — prompt-level audit of `prompts.py` / `prompts_v2.py` and linker-body prompt constants. Still valid; no new prompts were combined in Phase 8 (no-op closure).
- **Phase 8 `08-SUMMARY.md` §"GATE-06 unit re-audit"** — confirmed no new prompts merged; CROSS-02 collapse to CROSS-01 means no second linker arm to re-audit.
- **This audit (`09-GATE-06-AUDIT.md`)** — extends the prompt-level GATE-06 chain with a harness-level audit; together they cover the full cross-model invocation surface.
