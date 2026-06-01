# Phase 11: Research — Context

**Gathered:** 2026-05-31
**Status:** Ready for planning
**Mode:** Research phase — smart discuss compressed (scope is bounded by REQUIREMENTS PROMPT-05 + user-supplied scope expansion).

<domain>
## Phase Boundary

Produce `.planning/research/PROMPT-HARNESS-SURVEY.md` — a short, concrete survey of prompt-minimization harness techniques targeted at `s_linker13_clean`'s prompt surface (via `prompts_v2`). The survey must directly inform Phase 12's rule-trim strategy: which prompts to attack first, which techniques are GATE-06 compatible, and what rule-count reduction each technique can realistically deliver.

This is a literature + tooling review, not implementation. No source-tree edits. No LLM ablation runs.

</domain>

<decisions>
## Implementation Decisions

### Survey Scope (locked — user-supplied expansion)
The survey must cover three families, not just classical prompting:

1. **Classical prompt-minimization techniques** (REQUIREMENTS PROMPT-05 baseline):
   - Chain-of-Thought (CoT) — explicit vs implicit reasoning blocks; how it affects rule count.
   - ReAct (Reason + Act) — interleaved thought/action; relevance to multi-pass linker.
   - Self-consistency (multi-sample voting) — does the existing intersect/union voting in s_linker13 already cover this?
   - Rubric distillation — collapsing explicit rules into a single rubric the LLM internalizes.
   - Plan-then-execute / deliberation patterns — decompose the prompt into planning + execution phases.
   - Any deliberative-token or "let's think step by step" minimal-rule variants.

2. **Open-source coding-tool harnessing** (user-supplied):
   - `opencode` (https://github.com/sst/opencode and related) — how its prompts are structured, what rules it relies on vs delegates to the model, what guard layers it uses.
   - `codex` (OpenAI Codex CLI / the OSS variants) — same analysis.
   - Look for transferable patterns: minimal system prompts, tool-use schemas as rule substitutes, structured-output coercion vs free-text + post-validation.
   - Identify which patterns are GATE-06 safe (project-agnostic) and could replace rules in `s_linker13_clean`.

3. **Recent agentic-reasoning papers (last ~1 month, ~April–May 2026)** (user-supplied):
   - Find breaking/recent agentic reasoning papers (arXiv, ACL/EMNLP/ICLR preprints) on rule-minimization or harness simplification.
   - Cross-reference with public implementations / repos.
   - Score each by transferability to a traceability-linking pipeline like s_linker13.

### Output Contract
`.planning/research/PROMPT-HARNESS-SURVEY.md` must include:
- Per-technique entry with: (a) one-paragraph description, (b) primary citation/source/URL, (c) GATE-06 compatibility verdict + reasoning, (d) estimated rule-count reduction in the relevant `prompts_v2` prompt(s) (low/medium/high or explicit number), (e) fit-to-s_linker13 score (1–5) with one-sentence justification.
- ≥ 3 technique entries (REQUIREMENTS minimum); aim for 5–8 to cover the three scope families.
- Concluding section: **Recommended trim order for Phase 12** — concrete priority list (which prompt, which technique, why first).
- One-paragraph "what we explicitly did NOT find / open questions" so Phase 12 doesn't over-promise.

### Survey Constraints
- **Brevity**: short and concrete, not a literature review essay. Per-technique entry ~150-300 words. Total survey ≤ ~3000 words.
- **Concreteness**: every claim about applicability must reference a specific prompt in `prompts_v2` (e.g. "applies to `EXTRACTION_PROMPT`, would collapse rules 2–5 into rubric R-A").
- **GATE-06 strict**: zero benchmark-derived examples. All survey examples must come from textbook SE/Python/JS contexts (parsers, schedulers, queues, etc.).
- **Honest negative results**: if a technique looks attractive but is structurally incompatible (e.g. requires bit-deterministic LLM), say so explicitly.

### Tools Available to the Researcher
- WebSearch + WebFetch for paper search and open-source repo inspection.
- Read access to `src/llm_sad_sam/linkers/experimental/prompts_v2.py` to enumerate the prompt surface that needs trimming.
- Read access to `src/llm_sad_sam/linkers/experimental/s_linker13.py` and `s_linker13_clean.py` to understand prompt invocation patterns.
- Read access to `.planning/PROJECT.md`, `STATE.md`, `REQUIREMENTS.md`, and prior phase SUMMARYs for historical context (V32 lessons, V35 simplification failures, etc.).

### Out of Scope
- Implementing any prompt change (Phase 12 territory).
- Running LLM evaluations against the survey hypotheses (Phase 12 territory).
- Surveys of model providers beyond what's already gated (Claude Sonnet + gpt-5.4 are fixed by GATE-01).

</decisions>

<code_context>
## Existing Code Insights

### Prompt Surface to Trim (from `prompts_v2.py`)
The Phase 11 survey targets the prompts that `s_linker13_clean` actually invokes via `prompts_v2`. The researcher should enumerate these as the first step (read `prompts_v2.py`, list every prompt template + estimate current rule count).

### Historical Context — What Already Failed (memory-derived, V35 series)
Reference [MEMORY V35 entries](/home/dev/.claude/projects/-mnt-hostshare-ardoco-home-llm-sad-sam-v45/memory/MEMORY.md). All six prompt-simplification proposals tested in V35 regressed Claude Sonnet macro F1 (-2.4pp to -7.1pp). Lessons:
- Claude prompts are at a local optimum — concrete output examples HURT (bias sentence distribution).
- Example-driven rubric loses edge-case coverage vs rule-based.
- "Information density" of verbose prompts is exploited by Claude.

The Phase 11 survey must engage with this — propose techniques that have a *theoretical* reason to break the V35 ceiling (e.g. self-consistency aggregation across simpler prompts, rubric distillation with explicit edge-case retention, plan-then-execute decoupling).

### Project Conventions
- Default Claude Sonnet (not opus).
- GATE-06 strict (no benchmark leakage in prompt examples).
- v2.0 frozen files untouched (`prompts_v2.py` itself is frozen; survey only reads it).

</code_context>

<specifics>
## Specific Ideas

- **Pipe through opencode + codex repos directly** — read their `prompts/` or system-prompt files where available.
- For recent papers: prefer arXiv preprints with public implementations (GitHub repo linked). Avoid speculative-only papers.
- The "fit-to-s_linker13 score" should anchor on the V35 ceiling: a technique scoring 4–5 must explain why it could escape the V35 trap that simple simplification fell into.
- Possible cross-cutting theme to watch for: **the difference between "remove rules" (lose coverage → V35) and "restructure rules" (compress without losing coverage)** — this distinction may be the survey's central organizing axis.

</specifics>

<deferred>
## Deferred Ideas

None — survey is bounded by REQUIREMENTS PROMPT-05 + the user's scope expansion (opencode/codex + recent agentic papers).

</deferred>
