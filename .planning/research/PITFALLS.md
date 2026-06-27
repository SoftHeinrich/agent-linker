# Pitfalls Research

**Domain:** Rule-to-LLM ablation on s_linker12c (SAD-SAM traceability)
**Researched:** 2026-04-21
**Confidence:** HIGH (all findings grounded in this project's own experiment history)

---

## Critical Pitfalls

### Pitfall 1: Spike-to-Pipeline Invalidation (Isolated Validation Does Not Transfer)

**What goes wrong:**
A spike validates that an LLM replacement produces byte-identical output on representative examples, then the in-pipeline integration surfaces new failures — different call volumes, different context shapes, partial-response edge cases, or latency-induced timeouts — that were invisible in isolated tests.

**Why it happens:**
Spikes run against a handful of curated sentences. The pipeline runs against 100-300 sentences across 5 heterogeneous projects. Scale exposes:
- JSON parsing failures that only occur at high batch counts (truncated responses)
- Component names that were absent from spike examples but appear in-pipeline
- Context-window interactions between adjacent pipeline phases (e.g., doc_knowledge aliases feeding into extraction prompts with unexpected alias lengths)

PROJECT.md explicitly mandates: "Re-validate spikes in-pipeline" because "spike validation was isolated; pipeline integration can surface new failure modes."

**How to avoid:**
After each spike integration, run the full 5-project benchmark, not just the hard-tier-first shortcut. Treat hard-tier-first as the development signal, but never promote a variant on hard-tier results alone. The full sweep is the gate.

**Warning signs:**
- Spike test passes but first full-benchmark run shows ≥1pp regression on a dataset not in the hard tier (mediastore or jabref)
- LLM response logs show `"Empty response, retrying..."` at higher rates than 12c baseline

**Phase to address:**
Every integration phase (13a, 13b, ...). Each variant must re-run all 5 datasets before promotion.

---

### Pitfall 2: Ablation False Signal from Hard-Tier-First Scheduling

**What goes wrong:**
A rule removal looks regression-free on teammates and bigbluebutton (the hard tier used during development). When the full 5-project sweep runs, mediastore or jabref regress, pulling macro F1 below the 93% floor.

**Why it happens:**
The dataset schedule is hard-tier-first for cheap development signal. Teammates/BBB are the most rule-sensitive, but mediastore and jabref have different component naming patterns (e.g., jabref has single-lowercase-word components like `gui`, `cli`, `model` that exercise the ambiguity-classification path heavily; mediastore has multi-word names that exercise `_split_component_name` and trailing-word enrichment). A rule that doesn't matter for teammates can be load-bearing for jabref.

Memory entry S-Linker7 is the canonical example: removing the boundary/convention filter looked acceptable on the hard tier but cost -2.6pp macro F1 overall. Similarly, V35b showed MS collapsing -9.9pp while BBB/JAB actually improved.

**How to avoid:**
Hard-tier-first is a development accelerator only. Establish the rule: a variant is not a candidate for promotion until all 5 datasets are run. Add a per-dataset F1 table row to the ablation log for every candidate variant before the promotion decision.

**Warning signs:**
- "BBB and TM hold, let's promote" reasoning without the full 5-project numbers
- Variant achieves exactly 93.0% macro by having two datasets regress while three improve — this is floor gaming, not real quality (see Pitfall 11 below)

**Phase to address:**
Promotion decision for every variant (13a through final 13). The project's ablation table requirement (PROJECT.md) enforces this if every row has per-dataset columns.

---

### Pitfall 3: Benchmark Leakage via Prompt Rewrites

**What goes wrong:**
When a structural rule is rewritten as an LLM prompt, the developer writes natural-sounding examples to illustrate the rule. Those examples accidentally use component names, aliases, or keywords from the 5 benchmark projects. The pipeline then performs better (or differently) specifically on those projects, invalidating the evaluation.

**Why it happens:**
BENCHMARK_TABOO.md lists dozens of taboo terms that span ordinary English (logic, storage, client, model, server, cache, auth, adapter, order, event, socket, layer, facade, persistence, preferences). These words appear naturally in software architecture examples. A developer writing "e.g., the 'Logic' layer handles business rules" has introduced a TeaStore/Teammates collision. History: leakage was found and fixed in V31/V32 CONVENTION_GUIDE (cascade, dedicated, preferences, config, internal, Redis PubSub, kurento, FreeSWITCH, HTML5 Server, Recording Service). It was introduced during routine prompt improvement, not malicious intent.

**How to avoid:**
Every new or modified prompt must be cross-checked against BENCHMARK_TABOO.md before committing. Use the confirmed-safe domains from the taboo file: Compiler (Lexer, Parser, AST, CodeGenerator), OS (Scheduler, MemoryManager, FileSystem, Dispatcher), Networking (Router, Multiplexer, PacketHandler), E-commerce (ShoppingCart, PaymentGateway, InvoiceHandler), Game engine (RenderEngine, PhysicsSimulator). Avoid even the safe-domain words that appear in the Universal Taboo section (logic, storage, client, model, auth, cache, server, facade, adapter, order, event, socket, layer, processor).

**Warning signs:**
- Any prompt example that uses a word from BENCHMARK_TABOO.md's Universal Taboo section
- Prompt examples that contain more than 3 words matching architecture component vocabulary
- Unexpected F1 improvement on a specific dataset after a prompt rewrite that should have been neutral

**Phase to address:**
Every phase that modifies a prompt string. Add a checklist item to the variant creation protocol: "grep new prompt examples against BENCHMARK_TABOO.md".

---

### Pitfall 4: The Claude Prompt Local-Optimum Effect (V35 Lesson)

**What goes wrong:**
A prompt rewrite that appears simpler, cleaner, or more principled regresses Claude's F1 by 2-7pp. The verbose, detailed prompts in 12c are not over-engineered — they encode information density that Claude leverages. Reducing them to their essence loses calibration signal.

**Why it happens:**
Claude Sonnet operates contextually/pragmatically, not literally. The verbose prompts in 12c carry:
- 6 calibration examples in the judge (each teaching a specific edge case)
- Detailed multi-step reasoning guides
- Enumerated edge-case rules (CamelCase vs generic, partial-name, technology-as-component)

V35 experiments (all 6 prompt simplification proposals combined) produced ~92.4% macro F1 (-2.4pp). V35c (concrete JSON output examples) was worst at -7.1pp avg because it biased sentence-number distribution. V35a (example-driven CONVENTION_GUIDE) lost edge-case coverage. V30a (code overrides moved to prompt as "could be" signals) lost 1.5pp and collapsed TS to 86.2%.

Corollary: when Spike 003 validates that LLM enum emission "matches regex byte-identically" on a small sample, this does NOT mean the in-pipeline prompt can be simplified. The match is on examples; the information density question is whether the prompt is calibrated for the full distribution.

**How to avoid:**
When translating a structural rule into a prompt, transplant the full semantics of the rule, not a summary. If the rule has edge cases (CamelCase guard, dotted-path guard, single-lowercase guard), the prompt must have corresponding clauses. Do not simplify for aesthetics. Test the full 5-project sweep before concluding that a simplified prompt holds.

**Warning signs:**
- A new prompt is shorter than 8 lines where the original rule had 3+ branches
- "This is cleaner" as the sole rationale for a prompt change
- Spike validation matches byte-identically on 10 examples but in-pipeline benchmark shows unexpected per-dataset variance

**Phase to address:**
Phases removing `_classify_mention` (Spike 003 integration) and `_is_strong_alias` (scope field migration). Both involve translating multi-branch rules into prompts.

---

### Pitfall 5: `_classify_mention` String Coupling to Downstream Prompts

**What goes wrong:**
`_classify_mention` returns specific strings: `"proper case, standalone"`, `"lowercase mention"`, `"via known alias X"`, `"lowercase, inside dotted path"`. These strings appear verbatim inside the seed disambiguation prompt and the evidence bundle formatter (`_format_evidence`). If the LLM replacement (Spike 003 enum emission) produces even slightly different strings, the downstream prompts receive unrecognized mention-type values and may misinterpret evidence quality — without raising any error.

**Why it happens:**
The 4 regex branches produce human-readable strings that were designed to be informative to the downstream LLM judge. When the output is consumed by another LLM prompt rather than machine logic, string mismatch is silent. The judge's behavior changes subtly for borderline cases.

Specifically: the seed disambiguation prompt in `_run_seed_validation` constructs `match_ctx = self._classify_mention(comp_name, sent.text)` and embeds it as `"Mention: {match_ctx}"`. The validation pass in `_validate_with_evidence` embeds `mention={bundle.mention_type}` in the evidence block. Both prompts calibrate their LLM decision on these strings.

**How to avoid:**
When implementing the Spike 003 integration, define an enum class with fixed string values that exactly match the 4 current return strings. The LLM replacement must produce one of these 4 enum members. Add an assertion or enum coercion in `_build_evidence_bundle` and `_run_seed_validation` so that any out-of-enum value raises immediately rather than silently degrading.

**Warning signs:**
- LLM replacement produces strings like `"proper-case standalone"` (hyphen) or `"alias match"` instead of the exact current strings
- Seed validation rejection rates change after Spike 003 integration without any seed disambiguation prompt change

**Phase to address:**
Phase integrating Spike 003 (`_classify_mention` replacement). Must include exact-string contract test.

---

### Pitfall 6: Alias Scope Schema Bleeding Benchmark Patterns

**What goes wrong:**
When the alias discovery prompt is extended with a `scope: global|local` field (to retire `_is_strong_alias`), the LLM is asked to classify alias strength on the actual benchmark component names and their aliases. The LLM's scope classification may be calibrated on benchmark-specific patterns (e.g., consistently classifying single-word lowercase aliases as `local` for TeaStore because those are the ones that cause trouble), effectively embedding benchmark-specific knowledge via the model's learned pattern, not the prompt text.

**Why it happens:**
Unlike benchmark leakage via prompt examples (Pitfall 3), this is leakage via the LLM's in-context behavior. The scope classification prompt sees the actual component names ("gui", "cli", "model" for JabRef; "Auth", "Persistence" for TeaStore) and reasons about alias strength in their context. The LLM may generalize correctly, or it may produce scope assignments that are tuned to these 5 projects.

**How to avoid:**
The scope classification rule must be evaluated by asking: "would this scope decision generalize to a sixth project we have never seen?" The prevention is to use the same structural criteria that `_is_strong_alias` uses (multi-word, CamelCase, all-caps, starts-with-capital, single-lowercase-weak) as explicit prompt rules, not just ask the LLM to "judge strength". The LLM can then classify by applying these structural rules, making the scope decision generalizable by construction.

**Warning signs:**
- LLM-assigned scope differs from `_is_strong_alias` output on aliases you can manually check
- After the scope migration, F1 improves on TeaStore specifically (scope classification over-fitted to TeaStore's alias patterns)
- In-context scope assignments are inconsistent across two independent prompt calls on the same input

**Phase to address:**
Phase integrating `_is_strong_alias` + `_get_strong_alias_mappings` retirement.

---

### Pitfall 7: Variance Masking Real Regressions (Phase 3 Synonym Discovery Drift)

**What goes wrong:**
A variant run shows macro F1 within the ≥93% floor. A second run of the same variant (same code, same model, different wall-clock day) produces meaningfully different results because Phase 3 (doc knowledge synonym discovery) generates a different synonym set, which feeds into extraction alias injection, which changes the candidate set, which changes the intersection voting outcome.

**Why it happens:**
Memory entry "LLM Variance (Critical Finding)": "Same model gives DIFFERENT behavior across days (Phase 1 ambiguity, Phase 3 synonyms). This is NOT code change — affects entire phases, not individual links. V29 results vary by ~2-3pp across runs due to this." Claude's run-to-run variance is lower than GPT's but not zero, and it is concentrated in exactly the phases that are most sensitive: alias discovery (Phase 3) and ambiguity classification (Phase 1).

When a rule removal changes which components enter Phase 3, or changes how alias injection happens, it can amplify run-to-run variance even if the mean F1 is unchanged.

**How to avoid:**
For variants that touch Phase 1 (`_classify_components`, `_is_structurally_unambiguous` removal) or Phase 3 (alias discovery, `_is_strong_alias` retirement), run each variant at least twice on the hard tier before claiming a regression-or-no-regression conclusion. Log the synonym sets produced by Phase 3 in both runs and compare. If the synonym sets diverge, the run-to-run variance is the driver, not the rule removal.

The 12c checkpoint system (`_save_phase`) allows freezing the Phase 3 output and replaying downstream phases deterministically — use this for ablation.

**Warning signs:**
- Two runs of the same variant produce different link counts on the same dataset
- Phase 3 logs show different alias counts across runs (e.g., 4 aliases one day, 7 the next)
- A variant appears to hold ≥93% on run 1, then drops to 91% on run 2 with no code change

**Phase to address:**
Every phase that modifies the alias discovery path or the ambiguity classification path. Checkpoint replay (freeze layer1 pkl, vary only downstream) is the correct ablation methodology.

---

### Pitfall 8: Checkpoint Staleness Across Variant Swaps

**What goes wrong:**
A developer runs variant A to generate `layer1.pkl` checkpoints, then runs variant B starting from `resume_from_phase="layer2"` — loading variant A's checkpoints into variant B's downstream logic. The result is a chimera: variant B's code applied to variant A's knowledge acquisition output. The F1 reading is invalid for both variants.

**Why it happens:**
12c saves pickle checkpoints keyed to the linker class name and dataset name (`s_linker12c/{dataset}/layer1.pkl`). If variant A is also named `s_linker12c` (or if the variant is tested by modifying the live 12c file), and variant B loads the same path, cross-contamination is silent. The checkpoint directory naming (`_checkpoint_dir`) uses the class name, so a new standalone file with class `SLinker13a` uses a different checkpoint namespace — but only if the developer creates a proper standalone file.

**How to avoid:**
Each variant must be a standalone file with a distinct class name (enforced by the user's stated preference for standalone files, not inheritance chains). The class name determines the checkpoint directory. Never load a checkpoint from a different variant class. Before a benchmark run, clear or verify the checkpoint directory contains artifacts from the current variant, not a previous one.

**Warning signs:**
- A variant run completes suspiciously quickly on "full pipeline" with no LLM calls logged for Tier 1 phases
- The `layer1.pkl` timestamp is older than the current variant file's modification time
- The printed alias count matches the previous variant's count rather than the expected new behavior

**Phase to address:**
Every promotion milestone. Add a pre-run protocol: verify checkpoint directory name matches the current variant class name, or delete the directory to force a clean run.

---

### Pitfall 9: `_has_standalone_mention` RISKY Replacement (Latency Bomb)

**What goes wrong:**
An attempt to replace `_has_standalone_mention` with an LLM-based call creates an O(sentences × components) prompt budget. At 200 sentences × 15 components, this is 3,000 LLM calls just for anchor collection in a single dataset run. The pipeline becomes unusable, and the latency overwhelms any F1 benefit.

**Why it happens:**
AUDIT.md classifies `_has_standalone_mention` as RISKY: "Called O(N*M) in anchor collection. LLM replacement would be a massive prompt per sentence-pair." It is called in `_build_evidence_bundle` (once per candidate, iterating all sentences), in `_run_seed_validation` (once per component, iterating all sentences for anchors), and in `_coref_cases_in_context` (antecedent verification). These are not latency-isolated paths.

Unlike the other REPLACEABLE helpers, this function is a boundary primitive (word-boundary regex with dotted-path and hyphen guards) not a content heuristic. Its correctness is structural, not semantic.

**How to avoid:**
Do not attempt to replace `_has_standalone_mention` with an LLM call. AUDIT.md's recommendation is clear: KEEP as boundary primitive. If simplification is desired, narrow the dotted-path guard (let the LLM mention classifier handle dotted-path classification) but keep the word-boundary regex intact.

**Warning signs:**
- Any plan to "replace `_has_standalone_mention` with an LLM prompt"
- A benchmark run that shows 10x longer wall-clock time vs 12c baseline

**Phase to address:**
Not a phase — a permanent constraint. Capture in PROJECT.md Key Decisions as "KEEP `_has_standalone_mention`: word-boundary primitive, O(N×M) call sites, LLM replacement not feasible."

---

### Pitfall 10: Promotion Discipline — Silent Bug-Fix Side Effects

**What goes wrong:**
Variant A fixes a bug as a side effect of a rule removal (e.g., removing `_is_structurally_unambiguous` also eliminates an off-by-one in its CamelCase detection). Variant B is then promoted as "13b on top of 13a" and inherits the bug fix. When the ablation table is read, 13b's F1 improvement appears attributable to its own rule removal, but part of the gain was actually the bug fix in 13a. If someone later tries to apply only 13b's change to a clean 12c base, they cannot reproduce the result.

**Why it happens:**
The project uses sequential variants (12c → 13a → 13b → ...) where each variant is the previous plus one rule removal. A bug fix in 13a becomes an invisible dependency for 13b. This is the same problem as dependency accumulation in git cherry-pick sequences.

**How to avoid:**
When creating a variant, explicitly document in the file header: "This variant is 12c + removal of X. No other intentional changes." Run a diff against the parent variant and audit every difference. If an unexpected change is present, it must be either reverted or documented as a deliberate fix and recorded in the ablation table with a separate row.

**Warning signs:**
- Diff between variant N and variant N-1 shows changes beyond the targeted rule removal
- A variant's F1 improvement is larger than the expected contribution of the removed rule (per prior ablation data)

**Phase to address:**
Every promotion milestone. The ablation table row for each variant must list "rules changed" and "other intentional changes" as separate columns.

---

### Pitfall 11: F1 Floor Gaming (Selective Dataset Reporting)

**What goes wrong:**
A variant achieves exactly ≥93% macro F1 by having two datasets regress 3-4pp while three datasets improve slightly. The macro average barely clears the floor, but the individual per-dataset regressions are masked.

**Why it happens:**
Macro F1 averages across 5 datasets. If the floor check is only on the macro number, a variant that wins on easy datasets (jabref/mediastore) while losing on hard datasets (teammates/bigbluebutton) can still pass the gate. This is especially likely for variants that simplify rule logic: structural rules tend to be load-bearing specifically on ambiguous-name datasets.

**How to avoid:**
The promotion criterion must include both:
1. Macro F1 ≥ 93%
2. No single dataset below its 12c per-dataset F1 by more than 2pp

Per-dataset F1 must be reported in the ablation table. A macro-only floor is not sufficient.

**Warning signs:**
- Macro F1 is exactly at the threshold (93.0-93.2%)
- One dataset (typically teammates or bigbluebutton) shows a notable drop while others are up
- Developer reports macro F1 without the full per-dataset table

**Phase to address:**
Promotion decision for every variant. Define the per-dataset floor policy in PROJECT.md before the first promotion.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hard-tier-first evaluation only | Faster iteration (2 datasets vs 5) | False promotions, discovered late | Development iteration only; never for promotion |
| Using previous variant's checkpoints | Skip Tier 1 LLM calls (minutes saved) | Chimera evaluation, invalid F1 | Only when Tier 1 is provably unchanged (class name + checkpoint timestamp verified) |
| Spike isolated validation as pipeline validation | Saves one full benchmark run | Spike failures surface in integration, not spike | Never — spikes are hypotheses, pipeline is the gate |
| Simplifying prompts for readability | Cleaner code | Claude F1 regression (-2 to -7pp, V35 evidence) | Never unless simplification preserves all edge-case clauses |
| Macro F1 floor only (no per-dataset check) | Simpler promotion gate | Floor gaming, hidden per-dataset regressions | Never — always require per-dataset table |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Spike 001 (trailing-word LLM) into `_enrich_trailing_words` | Spike validated on curated candidates; pipeline candidates include multi-component overlap cases the spike did not cover | Re-run full 5-project sweep; check that `_split_component_name` removal doesn't break the uniqueness guard in the candidate generation loop |
| Spike 003 (mention-type enum) into `_build_evidence_bundle` | LLM emits string variant not matching the 4 exact current strings; silent mismatch in downstream prompts | Define enum with exact string values; add coercion with ValueError on unknown values |
| `scope: global\|local` alias schema into `_get_strong_alias_mappings` | Prompt scope assignment diverges from `_is_strong_alias` structural logic; aliases that were previously weak become global | Log a side-by-side comparison of LLM scope vs `_is_strong_alias` on the same alias set before retiring the code function |
| Coref prompt fold of `_has_strong_alias_mention` signal | Coref prompt already runs on antecedent evidence; adding alias signal may cause over-approval of coref for aliased components | Measure coref TP/FP before and after on all 5 datasets; Variant E is the Pareto winner precisely because it is conservative |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| LLM call inside O(N×M) loop | Pipeline wall-clock time 10-100x baseline; API rate limit errors | Never add LLM calls inside `_has_standalone_mention`, anchor-collection loops, or sentence-iteration with component cross-product | Any attempt to replace `_has_standalone_mention` |
| Parallel Tier 1 with slow alias discovery | One slow alias-discovery call blocks the entire Tier 1 `_run_parallel` barrier | Monitor per-task timings; alias discovery timeout is 300s — if it fires, the pipeline stalls | Documents > 300 sentences or component lists > 20 |
| Checkpoint accumulation | Disk fills with per-variant, per-dataset `.pkl` files from development runs | Checkpoint directory is named by class name; each new standalone variant file gets its own namespace | After 10+ variants in development, manual cleanup needed |

---

## "Looks Done But Isn't" Checklist

- [ ] **Spike integration:** Spike validated on isolated examples — verify full 5-project benchmark run shows no regression before marking "done"
- [ ] **Prompt rewrite:** New prompt looks clean — verify every clause from the original rule (including edge cases: CamelCase guard, dotted-path guard, single-lowercase guard) is represented
- [ ] **Taboo audit:** Any new prompt example — run grep against BENCHMARK_TABOO.md Universal Taboo section before committing
- [ ] **Variant file:** Created standalone `s_linkerXX.py` — verify class name is unique and distinct from checkpoint directories of parent variants
- [ ] **Promotion decision:** Macro F1 ≥ 93% — verify per-dataset table shows no single dataset regressed > 2pp from 12c baseline
- [ ] **Ablation table row:** F1 numbers recorded — verify "rules changed" and "other intentional changes" columns are filled separately
- [ ] **Checkpoint replay:** Running from resume point — verify checkpoint directory class-name prefix matches current variant, not a prior one

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Spike-to-pipeline failure discovered post-integration | MEDIUM | Freeze layer1 checkpoint from 12c; replay downstream phases with new variant to isolate failure to specific phase |
| Benchmark leakage discovered post-promotion | HIGH | Identify taboo term, replace with safe-domain equivalent from BENCHMARK_TABOO.md, re-run full 5-project sweep, update ablation table |
| Claude prompt local-optimum regression | MEDIUM | Revert to 12c prompt for that phase; add edge-case clauses one at a time, running a quick hard-tier check after each addition |
| Checkpoint staleness chimera | LOW | Delete checkpoint directory for the variant, re-run from scratch |
| `_classify_mention` string mismatch | LOW | Add enum coercion with exact string constants; catch and log mismatches before they reach the downstream prompt |
| Variance masking regression | MEDIUM | Freeze layer1.pkl; replay Tier 2/3 three times from the same frozen checkpoint to measure variance contribution vs rule-removal contribution |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Spike-to-pipeline invalidation | Every integration phase | Full 5-project F1 run completed and logged |
| Hard-tier false signal | Every promotion decision | Per-dataset F1 table present in ablation log row |
| Benchmark leakage via prompts | Every phase touching prompt strings | grep new examples against BENCHMARK_TABOO.md; diff shows zero taboo terms |
| Claude prompt local-optimum effect | Phases integrating Spike 003, alias scope | Prompt diff retains all edge-case clauses from original rule |
| `_classify_mention` string coupling | Spike 003 integration phase | Enum class defined; coercion assertion in place; edge-case unit test |
| Alias scope schema bleeding | `_is_strong_alias` retirement phase | Side-by-side log of LLM scope vs structural classification on all aliases |
| Variance masking regressions | Phases touching Phase 1 or Phase 3 (alias discovery) | Run variant twice; compare Phase 3 alias logs across runs |
| Checkpoint staleness | Every variant creation | Checkpoint directory name verified against current class name |
| `_has_standalone_mention` replacement | Permanent — not a phase | Constraint documented in PROJECT.md Key Decisions |
| Promotion discipline / silent bug-fix side effects | Every promotion milestone | Variant diff reviewed for unintended changes; ablation table has "other changes" column |
| F1 floor gaming | Every promotion decision | Per-dataset floor check: no dataset > 2pp below 12c baseline |

---

## Sources

- Project experiment memory: MEMORY.md (V35 series, V29 series, S-Linker7, S-Linker3-10)
- Spike 002 rules audit: `.planning/spikes/002-rules-audit/AUDIT.md`
- Benchmark taboo list: `BENCHMARK_TABOO.md`
- Source code: `src/llm_sad_sam/linkers/experimental/s_linker12c.py` (L259-298, L632-686, L791-820, L1118-1161)
- Project requirements: `.planning/PROJECT.md`

---
*Pitfalls research for: rule-to-LLM ablation on s_linker12c*
*Researched: 2026-04-21*
