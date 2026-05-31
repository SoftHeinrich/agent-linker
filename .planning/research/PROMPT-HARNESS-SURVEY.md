# Phase 11 — Prompt-Minimization Harness Survey

**Produced:** 2026-05-31
**Phase:** 11 (v2.1 Research, PROMPT-05)
**Status:** Final
**Total word count:** ~3,400

---

## 0. Prompt Surface in `prompts_v2.py` (Trim Target)

Enumeration of every prompt template in `prompts_v2.py`, with rule counts derived from numbered/bulleted items inside each template, and an indication of whether `s_linker13_clean.py` (Phase 10 artifact) actually imports it. The unused constants below ARE the dead weight Phase 12 can drop from `prompts_v3.py` with zero ablation cost.

| # | Constant | Lines | Numbered rules | Active in `s_linker13_clean`? | Pipeline tier (consumer) |
|---|----------|-------|---------------|------------------------------|--------------------------|
| 1 | `AMBIGUITY_FEW_SHOT` | 14–47 | 4 worked examples + rationales | YES | Tier 1 — model-side ambiguity classifier |
| 2 | `AMBIGUITY_RULES` | 50–64 | 2 buckets + 2 sub-categories + "The test" + "Key" | YES | Tier 1 — same call as #1 |
| 3 | `DOC_KNOWLEDGE_EXTRACTION_RULES` | 71–84 | 2 (ABBREVIATIONS, SYNONYMS) + APPROVE/REJECT pair | YES | Tier 1 — alias discovery extraction |
| 4 | `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | 87–121 | 7 worked examples (5 APPROVE / 2 REJECT) | YES | Tier 1 — alias judge |
| 5 | `DOC_KNOWLEDGE_JUDGE_RULES` | 124–139 | 3 numbered rules + IMPORTANT closing directive | YES | Tier 1 — same call as #4 |
| 6 | `WORD_USAGE_PROMPT` | 146–172 | 3 conceptual rules + JSON contract | **NO** (legacy ≤ 12c) | Legacy — drop from `prompts_v3` |
| 7 | `ENTITY_EXTRACTION_RULES` | 179–191 | 6 include + 2 exclude + favor-inclusion directive | YES | Tier 2 — entity extraction |
| 8 | `VALIDATION_RULES` | 194–205 | 3 APPROVE + 3 REJECT | YES | Tier 2 — entity validation |
| 9 | `COREF_RULES` | 212–222 | 5 numbered rules + worked example | YES | Tier 2 — coreference |
| 10 | `STANDALONE_MENTION_RULES_PRE_FILTERED` | 229–238 | 4 numbered rules + JSON contract | **NO** (EXT-01, deferred) | Tier 1 EXT-01 |
| 11 | `STANDALONE_MENTION_RULES_LLM_ONLY` | 241–255 | 5 numbered rules + JSON contract | **NO** | Tier 1 EXT-01 |
| 12 | `STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE` | 271–286 | 5 numbered rules + alias block + JSON | **NO** | Tier 1 EXT-01 alias-aware |
| 13 | `STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE` | 289–310 | 6 numbered rules + alias block + JSON | **NO** | Tier 1 EXT-01 alias-aware |
| 14 | `STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE` | 313–334 | 6 numbered rules + 2 knowledge blocks + JSON | **NO** | Tier 1 EXT-01 full-knowledge |
| 15 | `STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE` | 337–365 | 7 numbered rules + 2 knowledge blocks + JSON | **NO** | Tier 1 EXT-01 full-knowledge |
| 16 | `SEED_DISAMBIGUATION_RULES` | 372–390 | 1 APPROVE clause + 5 REJECT clauses | YES (lifted as classvar inside `s_linker13_clean`, line 143) | Tier 2 — seed disambiguation |

**Phase-12 trivial wins (no LLM ablation needed):** constants #6 and #10–#15 are unused by `s_linker13_clean`. They survived in `prompts_v2.py` for back-compat with frozen siblings; `prompts_v3.py` can omit all of them. That is a **~150-line, 36-rule deletion** before any technique below fires.

**Phase-12 contested surface (the ablation target):** constants #1, #2, #3, #4, #5, #7, #8, #9, #16. Combined ~37 numbered rules / sub-rules + 11 worked examples across nine prompts. THIS is where every technique in §2 must apply.

---

## 1. The V35 Ceiling — Setup

Six prompt-simplification proposals (V35, V35a–c, V35-combined) were tested on `s_linker13` ancestors in March 2026. All regressed Claude Sonnet macro F1 by 2.4–7.1 pp. The lessons logged in MEMORY were sharp: (a) example-driven rubrics that replace explicit rule lists lose edge-case coverage; (b) concrete JSON output examples bias the sentence-number distribution; (c) Claude exploits the **information density** of verbose rule lists in a way that aggressive shortening destroys.

A March 2026 arXiv paper, *Prompt Complexity Dilutes Structured Reasoning* (Jo, 2603.13351), independently reports the mirror finding from the other direction: a STAR reasoning framework that scored 100 % on a 10-line prompt collapsed to 0–30 % once embedded in a 60-line production prompt — not because the framework was wrong, but because **competing directives reversed reasoning order** ("Lead with specifics" forced conclusion-first output before reasoning could fire). The paper's central recommendation is: treat reasoning-vs-conclusion order as a first-class prompt-design variable; restructure rather than remove. This is the survey's organizing axis. Phase 12 must not "make prompts shorter"; it must **redistribute rule mass without lowering the per-rule information density Claude is currently exploiting**.

---

## 2. Technique Catalog

### Technique 1: Concise Chain-of-Thought (CCoT) — explicit "think step-by-step + be concise" directive (Family 1)

- **Description.** Renze & Guven (arXiv 2401.05618) show that instructing the model to "think step-by-step **and be concise**" with a few-shot example of a concise solution cuts response length 48.7 % while keeping accuracy flat on MCQA. On arithmetic GPT-3.5 regressed 27.7 %; on procedural classification it did not regress. The technique reduces *reasoning-trace* length, not rule-list length, so it leaves rule mass untouched.
- **Primary citation.** Renze & Guven, *The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models*, arXiv 2401.05618.
- **GATE-06 compatibility.** Yes. CCoT is a prompt instruction, not project-specific content.
- **Rule-count reduction in `prompts_v2.py` prompts.** Low. CCoT shortens model output, not the prompt's rule block. Applied to #9 (`COREF_RULES`) and #16 (`SEED_DISAMBIGUATION_RULES`) it might enable a 1–2-rule trim by absorbing the "favor inclusion / when uncertain" tie-breaker into a single "think briefly, then default" directive.
- **Fit-to-`s_linker13` score.** **2/5.** CCoT addresses verbosity in the model's reply, but `s_linker13_clean` already enforces JSON-only output for every prompt. Token cost is not the constraint (CLAUDE.md "no LLM budget limit"). The V35 ceiling is about *prompt-side* information density; CCoT does not address it.
- **V35-ceiling argument.** None — CCoT does not change rule mass and so cannot break the ceiling.

### Technique 2: Plan-then-Execute decomposition (Family 1)

- **Description.** Plan-then-Execute (P-t-E) separates strategic planning from tactical execution: the model first emits a plan, optionally is gated, then executes step-by-step. Compared to a ReAct loop it trades reactivity for predictability and cost control. In `s_linker13_clean` the natural mapping is at the alias-judge prompt (#4 + #5), where the current pattern is "look at all candidate aliases at once and emit an APPROVE/REJECT batch". A P-t-E variant would (a) first generate a one-line classification rubric per candidate, then (b) apply it. Architecturally similar to what the Tier-1 doc-knowledge prompt is already doing with `DOC_KNOWLEDGE_EXTRACTION_RULES` → `DOC_KNOWLEDGE_JUDGE_RULES`.
- **Primary citation.** SurePrompts, *Plan-and-Execute Prompting: Decompose First, Then Act* (2026); arXiv 2509.08646 *Architecting Resilient LLM Agents (Plan-then-Execute)*.
- **GATE-06 compatibility.** Yes. Structural pattern, no project content.
- **Rule-count reduction in `prompts_v2.py`.** Medium. If P-t-E is applied to #4 (`DOC_KNOWLEDGE_JUDGE_EXAMPLES`, 7 examples) by splitting into (a) emit-a-rubric (3 abstract rules) then (b) apply-rubric (2 rules), the example block can be cut to 3 examples without losing edge cases. Estimated 30–40 % reduction in token mass on prompts #3 + #4 + #5 combined.
- **Fit-to-`s_linker13` score.** **3/5.** The s_linker pipeline is already a multi-stage pipeline; adding *another* layer of decomposition inside a single LLM call is partly redundant with the existing stage decomposition. The win is non-zero for the alias judge specifically because that prompt currently asks the model to internalize 7 worked examples in one shot.
- **V35-ceiling argument.** Plausible. P-t-E does not REMOVE rules — it redistributes them across two inferences inside the same LLM call. If the planning step emits a project-internal rubric and the execution step references it, no edge-case coverage is lost. This is structurally distinct from V35a's "example-driven CONVENTION_GUIDE" failure mode (which removed rule mass and lost coverage).

### Technique 3: Rubric distillation with explicit edge-case retention (Family 1)

- **Description.** Collapse an explicit rule list into a compact rubric the model internalizes, but retain edge cases as a separate "calibration" block that the model can attend to without consuming reasoning bandwidth on every call. Inspired by LLMLingua-2 (arXiv 2403.12968) which formalizes prompt compression as token classification: keep the rule list, but compress its surface representation while guaranteeing semantic faithfulness.
- **Primary citation.** Pan et al., *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*, arXiv 2403.12968.
- **GATE-06 compatibility.** Conditional. LLMLingua-2 itself trains on task-agnostic compression and is safe. The hand-rolled variant — writing a single rubric paragraph that *replaces* a numbered rule list — is exactly what V35a did and failed at. The safe form of this technique is **lossless rewriting**: same rules, denser surface form, identical decision boundary.
- **Rule-count reduction in `prompts_v2.py`.** Medium-high if applied losslessly. Prompts #2 (`AMBIGUITY_RULES`) and #5 (`DOC_KNOWLEDGE_JUDGE_RULES`) duplicate intent across nested categories ("Category A / Category B / The test / Key"). A lossless rewrite could collapse the four sub-blocks into two without losing the Connector / Controller / Wrapper distinction.
- **Fit-to-`s_linker13` score.** **4/5.** This is the most defensible Phase 12 starting point because it directly attacks structural redundancy (same rule restated four ways) while keeping every coverage case the V35 failures lost. The V35 mistake was *replacing* rules with examples; lossless rubric distillation does the opposite — it merges examples and rules into the same surface form.
- **V35-ceiling argument.** Strong. The V35 failure mechanism was rule deletion ("example-driven loses edge cases"). Lossless rubric distillation does not delete; it consolidates. The 2026 *Prompt Complexity Dilutes* paper supports this — competing directives, not rule mass per se, are what degrade reasoning. Merging an "APPROVE if X" rule and an "REJECT if NOT X" rule into a single compound clause removes the competing-directive surface without removing coverage.

### Technique 4: Self-consistency (multi-sample + voting) on top of the existing intersect-union scheme (Family 1)

- **Description.** Sample the same prompt K times at non-zero temperature, then vote. The recent Self-Consistent Structured Generation paper (SemEval-2026 Task 3) reports significant gains at K = 15. The intersect/union voting already in `s_linker13_clean` is *cross-pass* voting (different prompts agreeing). Self-consistency would add *intra-pass* voting (same prompt, multiple samples).
- **Primary citation.** SemEval-2026 Task 3 SCSG (arXiv 2603.01788); Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (foundational, 2022).
- **GATE-06 compatibility.** Yes. Sampling discipline is project-agnostic.
- **Rule-count reduction in `prompts_v2.py`.** Zero per se. Self-consistency does not change prompts. But it ENABLES rule-trimming: a trimmed prompt that has higher run-to-run variance can be made *deterministic enough* by K-sample voting. This is the unlock for techniques 3 and 5.
- **Fit-to-`s_linker13` score.** **3/5.** Already partially present (intersect voting). The marginal value is enabling more aggressive prompt trims downstream. The cost is K× LLM calls — the project explicitly accepts this ("no LLM budget limit").
- **V35-ceiling argument.** Indirect. Self-consistency does not by itself break V35, but it provides *insurance* for trims that would otherwise add variance. V35a-style trims could be re-examined if K=5 sampling absorbs the noise they introduced. Worth running as a control variable in Phase 12 ablations.

### Technique 5: Schema-driven harness (lift rules into tool/output schemas) — Family 2 (opencode/codex pattern)

- **Description.** `opencode` (sst/opencode) uses a hybrid system-prompt architecture: provider-specific identity prompt + environment block + AGENTS.md instructions + structured-output enforcement via tool schemas. Rules are split between (a) a small base prompt with imperative directives ("ALWAYS / NEVER / USE") and (b) tool schemas that constrain output shape *without* requiring textual rules. Crucially, opencode's guidance is "keep AGENTS.md to 20–30 lines" — explicitly anti-verbose. By contrast, OpenAI's `codex` CLI prompting guide (developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide) is the opposite: comprehensive rule structures, explicit behavioral instructions, schemas as supporting role. The two leading 2026 open-source coding agents disagree on this axis. Transferable insight: `s_linker13_clean`'s prompts already emit structured JSON; some rule clauses ("Return JSON: …", "JSON only", APPROVE/REJECT enums) duplicate constraints the JSON schema could enforce.
- **Primary citations.** [sst/opencode system prompts (DeepWiki)](https://deepwiki.com/sst/opencode/4.3-system-prompts-and-context); [opencode prompt assembly gist](https://gist.github.com/rmk40/cde7a98c1c90614a27478216cc01551f); [Codex Prompting Guide (developers.openai.com)](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide); [openai/codex repo](https://github.com/openai/codex).
- **GATE-06 compatibility.** Yes. Schema-driven output enforcement is project-agnostic.
- **Rule-count reduction in `prompts_v2.py`.** Low-medium. Removing the trailing `Return JSON: …` line from every prompt and instead enforcing a strict JSON schema via the LLM client's structured-output mode would shave ~1 rule per prompt (~8 lines total across active prompts). Bigger wins are unavailable because the `s_linker13_clean` rule lists are SEMANTIC ("when uncertain favor inclusion"), not SHAPE constraints — schemas can't replace semantic rules.
- **Fit-to-`s_linker13` score.** **2/5.** The opencode pattern is built around *tool use* (function calls, file system ops); `s_linker13_clean` makes pure classification calls with no tool surface. The transferable subset is small (output-shape enforcement). Codex's heavily-ruled approach is closer to the s_linker style and is independent evidence that for *non-agentic* code reasoning tasks, rule lists outperform minimal prompts — which DIRECTLY validates the V35 ceiling finding.
- **V35-ceiling argument.** Weak. Schema-driven enforcement cannot replace the semantic rules that V35 tried to trim. The Codex evidence is more important than the opencode evidence here: it tells us "for classification-style coding tasks, the leading OpenAI guide explicitly recommends comprehensive rule structures, not minimal prompts".

### Technique 6: Extended-thinking / adaptive thinking budget (Family 1 + Family 3)

- **Description.** Claude Sonnet 4.5+ supports an `effort` (formerly `budget_tokens`) parameter that allocates hidden deliberation tokens before the response. Anthropic's 2026 guidance: use thinking when the cost of a wrong answer is > 5× the call cost (code generation, architecture decisions); skip for summarization/classification/extraction. The `s_linker13` validation, alias-judge, and coref calls all sit on the "expensive-error" side. By giving the model an explicit deliberation budget instead of asking it to "weigh" things inline, the prompt itself can be trimmed of meta-deliberation directives ("When in doubt, APPROVE", "Favor inclusion over exclusion").
- **Primary citations.** [Anthropic *Building with extended thinking*](https://platform.claude.com/docs/en/build-with-claude/extended-thinking); [Anthropic *Effort* docs](https://platform.claude.com/docs/en/build-with-claude/effort).
- **GATE-06 compatibility.** Yes. Backend parameter, not project content.
- **Rule-count reduction in `prompts_v2.py`.** Medium. The "When in doubt / Favor inclusion / IMPORTANT" tie-breaker directives appear in prompts #3, #5, #7, #8, #16 — at least 5 trimmable lines. These directives exist precisely because the model is being asked to weigh options without explicit deliberation budget. With extended thinking on, that weighing happens in hidden tokens.
- **Fit-to-`s_linker13` score.** **4/5.** Concrete, applies to multiple prompts, breaks the V35 ceiling through a model-capability mechanism (deliberation in hidden tokens) rather than a prompt-shortening one. Risk: extended thinking on Claude Sonnet has price implications, but the user's policy accepts unlimited LLM budget.
- **V35-ceiling argument.** Strong. The V35 failure was Claude losing the *information* it needed to weigh edge cases. Extended thinking gives it dedicated bandwidth to weigh; removing the "favor X" directive from the surface prompt should be net-safe IF the model is given thinking budget to come to the same conclusion. Empirically testable in Phase 12.

### Technique 7: Adaptive specificity — keep detail where smaller-task subprompts need it (Family 3)

- **Description.** *DETAIL Matters* (Jo et al., arXiv 2512.02246, Dec 2025) reports that prompt specificity improves accuracy especially for procedural tasks. The recommendation is **adaptive prompting strategies** — keep detail where the model is doing step-by-step reasoning, abstract where it is doing classification. Applied to `s_linker13_clean`: prompts that demand reasoning (#4 `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, #9 `COREF_RULES`) should keep rule-level detail; prompts that demand classification (#16 `SEED_DISAMBIGUATION_RULES`, #1+#2 ambiguity) might tolerate denser rubrics. This is the OPPOSITE of "trim every prompt equally" — and Phase 12 should treat the trim budget as non-uniform.
- **Primary citation.** Jo et al., *DETAIL Matters: Measuring the Impact of Prompt Specificity on Reasoning in Large Language Models*, arXiv 2512.02246.
- **GATE-06 compatibility.** Yes. Adaptive-specificity discipline is methodological.
- **Rule-count reduction in `prompts_v2.py`.** Variable by prompt — that is the point. Estimated 0–25 % per prompt depending on classification-vs-reasoning load.
- **Fit-to-`s_linker13` score.** **4/5.** Provides a *trim-order discipline* for Phase 12 rather than a single edit. Pairs naturally with Technique 3 (lossless rubric distillation) and Technique 6 (extended thinking).
- **V35-ceiling argument.** Strong. V35 trimmed all six proposals at once and all regressed. DETAIL Matters predicts this is exactly what would happen — different prompts need different specificity levels. Phase 12's per-prompt ablation methodology (PROMPT-02) is the right shape; this technique justifies it theoretically.

### Technique 8: Prompt Complexity Dilution Order Restructure (Family 3 — March 2026)

- **Description.** *Prompt Complexity Dilutes Structured Reasoning* (Jo, arXiv 2603.13351, March 2026) identifies the precise failure mode that explains V35: **competing directives reverse reasoning-vs-conclusion order**. A STAR framework that hit 100 % alone collapsed to 0–30 % inside a 60-line production prompt because directives like "Lead with specifics" forced answer-first output. The paper's recommendation is not "shorten the prompt" but "ensure reasoning precedes conclusion in the directive ordering". Applied to `s_linker13_clean`: scan every active prompt for directives that demand a JSON verdict before reasoning. The `DOC_KNOWLEDGE_JUDGE_RULES` "AUTO-APPROVE these — they are always valid mappings" clause is a candidate — it short-circuits reasoning. The `VALIDATION_RULES` APPROVE/REJECT lists are similarly verdict-first. Restructuring these so the model emits a brief consideration before the verdict (or using the extended-thinking budget from #6) may unlock further trims that V35 thought were impossible.
- **Primary citation.** Jo, *Prompt Complexity Dilutes Structured Reasoning: A Follow-Up Study on the Car Wash Problem*, arXiv 2603.13351, March 2026.
- **GATE-06 compatibility.** Yes. Methodological.
- **Rule-count reduction in `prompts_v2.py`.** Indirect. The technique does not delete rules; it reorders directives. Cumulatively it may unlock 10–20 % deeper trims from techniques 3 and 6.
- **Fit-to-`s_linker13` score.** **5/5.** This is the recent paper most directly relevant to v2.1. It explains *why* V35 failed and gives a concrete prescription. It justifies Phase 12 spending ablation budget on **directive-ordering** experiments, not just rule-deletion experiments.
- **V35-ceiling argument.** This paper IS the V35-ceiling explanation. The mechanism is documented: long prompts with competing directives undermine structured reasoning frameworks even when the frameworks are correct in isolation. The escape route is to redesign directive ordering, not to delete rules.

---

## 3. Open-Source Coding-Tool Patterns

The two leading 2026 open-source coding agents disagree on rule density in informative ways.

**`sst/opencode`** uses a layered system-prompt assembly (provider header → provider-specific prompt → environment block → AGENTS.md → agent-specific block → user override) with the explicit guidance to keep AGENTS.md at 20–30 lines. Rules are short, imperative ("ALWAYS / NEVER / USE"). Tool schemas carry significant behavioral weight. Reasoning is delegated to the provider's `reasoningEffort` / extended-thinking parameter rather than inline directives.

**`openai/codex`** does the opposite. The Codex Prompting Guide ([developers.openai.com](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide)) explicitly recommends **comprehensive rule structures**: General, Autonomy and Persistence, Code Implementation, Editing constraints, etc. Tool schemas are documentation-only; rules are the primary control lever. This is closer to the `s_linker13` style and is independent industry evidence that for structured classification-style code tasks, rule-heavy prompts continue to outperform minimal ones.

**Transferable patterns for `s_linker13_clean`:**

| Pattern | Source | Transferable? | Why |
|---|---|---|---|
| Provider-specific prompt selection (Claude vs GPT branches) | opencode | **Yes — already deferred (ADAPTER-01)** | Out of v2.1 scope but relevant. |
| Tool/output-schema enforcement replacing "Return JSON: …" lines | opencode + structured-output APIs | **Conditional** | Low-rule-count win (Technique 5). |
| Reasoning delegated to `effort` parameter | opencode | **Yes** | This is Technique 6. |
| Heavy rule lists with imperative directives | codex | **Already in use** | The Codex guide validates the current `s_linker13` design choice. |
| AGENTS.md-style external instruction file | both | **No** | s_linker prompts are per-call, not per-session. |
| "Skills" system for on-demand task instructions | opencode | **No** | s_linker has no user/runtime dispatch surface. |

Overall: opencode's minimalism is a poor fit because `s_linker13` is not an agent; codex's rule-heaviness validates the current direction, with two narrow exceptions (Techniques 5 and 6).

---

## 4. Recent Agentic-Reasoning Findings (~April–May 2026)

Two recent papers directly relevant; one large survey for background; one negative finding worth recording.

**Most relevant — *Prompt Complexity Dilutes Structured Reasoning* (Jo, arXiv 2603.13351, March 2026).** Already covered as Technique 8. This is the V35 explanation: competing directives in long prompts collapse reasoning frameworks even when the frameworks are sound. Recommendation: reorder directives so reasoning precedes conclusions.

**Counterweight — *DETAIL Matters* (Jo et al., arXiv 2512.02246, Dec 2025 / cross-references 2026).** Already covered as Technique 7. Specificity improves accuracy especially for procedural reasoning tasks. Implication: do not uniformly trim. Per-prompt ablation (PROMPT-02 already mandates this).

**Background — *Agentic Reasoning for Large Language Models* (Tianxin Wei et al., arXiv 2601.12538).** Comprehensive survey of agentic reasoning across three layers (foundational, self-evolving, multi-agent). Distinguishes in-context reasoning (test-time orchestration) from post-training reasoning (RL/SFT). Not directly actionable for `s_linker13` because s_linker is single-prompt classification, not agentic. Worth citing once for context.

**Negative — Self-Consistent Structured Generation (SCSG, SemEval-2026 Task 3, arXiv 2603.01788).** Reports K=15 self-consistency gains on dimensional ABSA. Transferable mechanism (Technique 4) but the gains are reported on a noisier task than s_linker. Unverified whether s_linker's already-deterministic Claude Sonnet calls have enough run-to-run variance to benefit from K-sample voting. Flagged for empirical test only.

No "breaking" recent paper invalidates the V35 findings. The recent literature CONFIRMS V35: prompt complexity matters; per-prompt adaptive strategies are right; restructuring beats deletion.

---

## 5. Recommended Trim Order for Phase 12

Phase 12 (`prompts_v3.py` + per-prompt ablations) should attack the prompts in this order. Each step has a hypothesis, an expected risk, and a fallback.

| Order | Prompt(s) | Technique to apply | Why first | Expected risk |
|---|---|---|---|---|
| **0** | #6 `WORD_USAGE_PROMPT`, #10–#15 standalone-mention variants | **Drop entirely from `prompts_v3.py`** | Already unused by `s_linker13_clean`. ~36 rules / ~150 lines deleted with zero F1 risk. Free win that improves reviewer-defensibility (smaller surface to audit for GATE-06). | None. Verify via GATE-02 frozen-compat test that frozen siblings continue importing from `prompts_v2.py` (which stays untouched). |
| **1** | #5 `DOC_KNOWLEDGE_JUDGE_RULES` + #4 `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | **Technique 3 (lossless rubric distillation)** + **Technique 8 (directive reorder)** | Highest rule mass (3 numbered rules + 7 worked examples + AUTO-APPROVE clause + IMPORTANT closer). The "AUTO-APPROVE / APPROVE if / REJECT only if" three-tier structure has known V31/V32 calibration history — losslessly merging the AUTO-APPROVE list with the APPROVE clause into a single positive-bias rubric, while keeping all 7 examples, should preserve the empirical calibration. Reorder so the "When in doubt, APPROVE" clause comes BEFORE the verdict format (per arXiv 2603.13351). | Medium. The doc-knowledge judge is a load-bearing precision filter (per V31 phase-contribution analysis). Trims here MUST clear GATE-01 cross-model. |
| **2** | #7 `ENTITY_EXTRACTION_RULES` + #8 `VALIDATION_RULES` | **Technique 3 (lossless rubric distillation)** | The 6 include + 2 exclude + 3 APPROVE + 3 REJECT structure across two prompts contains structural overlap (rule 1 of EXTRACTION mirrors APPROVE-clause-1 of VALIDATION). Phase 12 candidate: merge into a single shared "architectural-participant" rubric block and import in both prompts. Estimated 4-rule reduction without coverage loss. | Medium. Validation is the last gate before final links — false rejects are permanent. Run isolated single-prompt ablation, not joint. |
| **3** | #2 `AMBIGUITY_RULES` | **Technique 3 (lossless rubric distillation)** | The "Category A / Category B / The test / Key:" structure is four restatements of the same boundary. Lossless merge candidate. The 4 worked examples in #1 carry the calibration weight and should NOT be touched (per V35a lesson — example-driven simplification of ambiguity classification regressed). | Low. Ambiguity classifier is upstream of multiple downstream tiers — regressions cascade. But the *rules* block is structurally redundant, while the *examples* block is where the model picks up calibration. |
| **4** | #9 `COREF_RULES`, #16 `SEED_DISAMBIGUATION_RULES` | **Technique 6 (extended thinking budget)** + targeted "when uncertain" trim | These prompts each carry one or more "When uncertain, choose X" tie-breaker directives. With extended-thinking enabled the model can deliberate in hidden tokens. Remove the surface "when uncertain" directive AS A SEPARATE ablation variant, keeping the rest of the rule list intact. Per Technique 6, this is theoretically motivated. | Medium-high. Coref is precision-critical (V31 phase-contribution shows +1.0–1.5 pp). Extended thinking is a backend parameter change — must validate both gates carefully. |
| **5** | All active prompts | **Technique 5 (output-schema enforcement)** | Remove the trailing `Return JSON: …` line from each active prompt and lift the constraint into the LLM-client structured-output enforcement (if supported by both Claude and gpt-5.4 backends). ~1-line × 8 prompts. Reviewer-defensibility win. | Low. Requires verifying that gpt-5.4 supports the same structured-output schema discipline as Claude. If asymmetric, defer. |
| **6** | Self-consistency layer (orthogonal) | **Technique 4** | If steps 1–5 introduce run-to-run variance that pushes a variant just below GATE-01, add K=3 sampling at non-zero temperature to recover. Treat as a recovery tool, not a default. | Negligible — purely additive. |

**Out of priority order:** Technique 1 (CCoT) does not justify Phase 12 budget; technique 2 (P-t-E) is candidate for Phase 12 ONLY if step 1 alone fails to trim #4 + #5 enough — otherwise defer to v2.2+.

---

## 6. Open Questions / Explicit Negative Results

**What was investigated but rejected:**

- **Naive prompt shortening (V35 baseline).** All six V35 proposals regressed Claude. Confirmed by *Prompt Complexity Dilutes Structured Reasoning* mechanism. Will not be repeated.
- **Example-driven CONVENTION_GUIDE replacement (V35a).** Regressed −2.5 pp. The 4 examples in `AMBIGUITY_FEW_SHOT` (#1) and the 7 examples in `DOC_KNOWLEDGE_JUDGE_EXAMPLES` (#4) should be preserved unless trimmed with an explicit lossless mechanism.
- **Concrete JSON output examples in prompts (V35c).** Regressed −7.1 pp because they bias the sentence-number distribution. Technique 5 (schema enforcement at API layer) is the safe alternative.
- **AGENTS.md-style external file for s_linker rules (opencode pattern).** Not transferable — s_linker is per-call classification, not multi-turn agent.

**What was searched for but not findable:**

- **Recent (April–May 2026) paper specifically on rule-list compression for classification prompts.** Closest match is *DETAIL Matters* (Dec 2025) and *Prompt Complexity Dilutes* (March 2026). No April–May 2026 paper found that directly addresses the trade-off; the latter is the most recent relevant evidence.
- **Empirical evidence on extended-thinking budget vs prompt length trade-off for Claude Sonnet 4.5+** specifically on classification tasks (as opposed to math/code). The Anthropic guidance gives heuristics but no controlled benchmark. Technique 6's V35-escape argument is theoretically motivated but **empirically unverified for s_linker's task profile** — Phase 12 should treat this as the primary research question, not a settled prescription.
- **Quantified rule-count vs F1 curves for Claude Sonnet on traceability-style tasks.** No public dataset. Phase 12's per-prompt ablation will produce the first such curve.

**Open question for Phase 12 to settle:**

> Does enabling extended thinking on Claude Sonnet 4.5+ allow `s_linker13_clean` to absorb the same prompt trims that V35 failed at, by moving the "weighing" work from prompt surface to hidden tokens — without regressing gpt-5.4 (which uses a different reasoning mechanism)?

This is the highest-leverage empirical question the survey identified. It is not answerable from the literature alone.

---

## Sources

### Primary (HIGH confidence)
- [Anthropic Extended Thinking docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) — Technique 6.
- [Anthropic Effort docs](https://platform.claude.com/docs/en/build-with-claude/effort) — Technique 6.
- [sst/opencode (DeepWiki system-prompts section)](https://deepwiki.com/sst/opencode/4.3-system-prompts-and-context) — Family 2.
- [openai/codex GitHub](https://github.com/openai/codex) — Family 2.
- [Codex Prompting Guide (developers.openai.com)](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide) — Family 2.
- `src/llm_sad_sam/linkers/experimental/prompts_v2.py` — surface enumeration in §0.
- `src/llm_sad_sam/linkers/experimental/s_linker13_clean.py` — active-prompt verification in §0.

### Secondary (MEDIUM confidence — peer-reviewed / arXiv preprints with named authors)
- Jo, *Prompt Complexity Dilutes Structured Reasoning: A Follow-Up Study on the Car Wash Problem*, [arXiv 2603.13351](https://arxiv.org/abs/2603.13351) — Technique 8, March 2026.
- Jo et al., *DETAIL Matters: Measuring the Impact of Prompt Specificity on Reasoning in Large Language Models*, [arXiv 2512.02246](https://arxiv.org/abs/2512.02246) — Technique 7, Dec 2025.
- Renze & Guven, *The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models*, [arXiv 2401.05618](https://arxiv.org/abs/2401.05618) — Technique 1.
- Pan et al., *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*, [arXiv 2403.12968](https://arxiv.org/abs/2403.12968) — Technique 3.
- *Agentic Reasoning for LLMs* survey, [arXiv 2601.12538](https://arxiv.org/abs/2601.12538) — §4 background.
- *Architecting Resilient LLM Agents (Plan-then-Execute)*, [arXiv 2509.08646](https://arxiv.org/abs/2509.08646) — Technique 2.

### Tertiary (LOW confidence — secondary/aggregator sources; cited for direction only)
- SurePrompts, *Plan-and-Execute Prompting* — Technique 2 framing.
- Adaline, *What is ReAct Prompting in 2025?* — context for ReAct displacement by P-t-E.
- SemEval-2026 SCSG report (arXiv 2603.01788) — Technique 4, unverified transfer.

### Project / historical
- `MEMORY.md` V35 entries — V35a/b/c regression evidence (the ceiling this survey is engineered to escape).
- `BENCHMARK_TABOO.md` — GATE-06 enforcement; all examples in this survey use textbook SE contexts (parsers, lexers, schedulers, dispatchers, brokers, code generators, payment-gateway / invoice-handler, render-engine / scene-graph).

---

## Metadata

**Confidence breakdown:**
- Prompt-surface enumeration (§0): HIGH — verified by direct file read + grep of `s_linker13_clean` and `s_linker13`.
- V35 ceiling framing (§1): HIGH — V35 results are in project MEMORY and reconfirmed by arXiv 2603.13351.
- Technique catalog (§2): MEDIUM — primary citations verified; fit-to-s_linker scores reflect MEMORY-grounded theoretical arguments, not empirical s_linker data (which is Phase 12's job to produce).
- Open-source patterns (§3): HIGH — three independent sources (DeepWiki, gist, OpenAI cookbook) cross-verified.
- Recent papers (§4): MEDIUM — March 2026 paper directly relevant; April–May 2026 surface searched but no closer match found.
- Trim order (§5): MEDIUM-HIGH — order is theoretically motivated; absolute pp gains await Phase 12 ablation.

**Research date:** 2026-05-31
**Valid until:** 2026-06-30 (30 days for stable techniques; new agentic-reasoning papers land weekly so re-survey if Phase 12 stretches beyond June).
