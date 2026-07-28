---
spike: 016
name: s24-general-tool-discovery
type: comparison
validates: "Given saved S24 checkpoint floors, when duplicated lexical applicability rules and participant-specific morphology are replaced by tool-owned discovery plus a contrastive judge, then S24 retains clean recovery with less hand-written policy."
verdict: VALIDATED
related: [013-s24-lexical-entity-normalization, 014-s24-discourse-scope-participants, 015-s24-minimal-tool-orchestration]
tags: [s24, simplicity, discovery, hardcoding, controller, checkpoint]
---

# Spike 016: S24 general tool discovery

## What This Validates

Can S24 remove the hand-written reference vocabulary, participant terminal-token
rule, suffix blacklist, manual pluralization, and special server/host rubric
without losing the checkpoint-backed recovery that justified those policies?

Three comparisons run in risk order:

1. **tool-owned availability** — remove the controller's lexical evidence
   inventory and let each tool discover whether it has candidates;
2. **general participant discovery** — ask a small generic proposer for
   shortened/contextual mentions instead of constructing terminal handles;
3. **minimal semantic contract** — decide target versus strongest alternative
   without enumerating host/type/technology failure classes.

## Research

No external dependency is needed. The comparison reuses the repository's
candidate/link data types, exact-quote grounding, saved phase floors, and OpenAI
backend. The credible approaches are:

| Approach | Advantage | Risk | Status |
| --- | --- | --- | --- |
| Expand regex/suffix lists | Cheap | More language- and benchmark-shaped policy | Rejected |
| Broad catalog-token overlap | No LLM discovery call; complete checkpoint reach | Broad candidate set needs a semantic gate | Chosen |
| Tool-owned LLM discovery | General contract; handles morphology/context | Poor recall and precision in the pilot | Rejected |
| Remove all grounding code | Fewest rules | Allows invented targets/claims | Rejected |

Exact catalog membership, exact source quotes, allowed anchors, deduplication,
and finite tool execution remain safety invariants rather than semantic
heuristics.

## Promotion Gates

- identical saved-floor comparison on BigBlueButton and TeamMates;
- participant contribution at least 10 TP / 0 FP;
- no loss of any checkpoint participant TP solely because discovery missed it;
- coreference capability remains reachable on every checkpoint where its tool
  produced candidates;
- controller prompt contains no lexical evidence inventory or cited-evidence
  field;
- production checkpoint has zero participant-source FP;
- fresh paired five-project E2E does not regress macro or pooled F1/F2 versus
  canonical S21;
- canonical S21 remains byte-stable and no benchmark vocabulary enters code.

## How to Run

```bash
../.venv/bin/python pilot/test_s24_orchestrator.py
../.venv/bin/python pilot/test_s24_simple_orchestrator.py
../.venv/bin/python pilot/test_s24_general_discovery.py
../.venv/bin/python pilot/test_s24_discourse_scope.py
../.venv/bin/python pilot/test_s24_lexical_entity.py

OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
  ../.venv/bin/python pilot/s24_general_discovery_pilot.py \
  --checkpoint-root \
    ../results/s24_simple_production_checkpoint_v2_contract_20260725 \
  --datasets bigbluebutton teammates \
  --discovery overlap \
  --judge minimal \
  --results-dir \
    ../results/s24_general_discovery_overlap_blind_shared_v11_repeat_20260728

OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=\
../results/s24_general_discovery_e2e_v1_20260728/phase_states \
LLM_LOG_DIR=../results/s24_general_discovery_e2e_v1_20260728/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/s24_general_discovery_e2e_v1_20260728
```

## Investigation Trail

1. Spike 015 proved that compact prompts preserve performance, but its
   controller still duplicates tool applicability through a fixed reference
   lexicon and participant morphology.
2. The saved production checkpoint contains the raw candidates and decisions
   needed to compare discovery independently of extraction/validation
   randomness.
3. Saved traces showed that coreference produced candidates on all five
   projects (11, 42, 16, 24, and 12). The controller's fixed reference-word
   detector therefore excluded no call in the evaluated corpus; it only
   duplicated applicability logic.
4. A generic LLM proposer was not viable. Its first run proposed 7 candidates,
   recalled 1/12 checkpoint participant TPs, and ended at 1 TP / 4 FP. An
   exhaustive retry proposed 18, recalled 2/12, and ended at 3 TP / 11 FP.
5. A catalog-overlap generator instead proposed every uniquely owned runtime
   catalog-token continuation. It reached all 12 checkpoint TPs without a
   terminal-token rule, suffix blacklist, pluralizer, or project vocabulary.
6. A one-stage judge was unstable because seeing the target biased the model
   toward identity. Splitting the decision fixed the causal problem: first
   classify the expression's denotation without a target, then compare only
   participant expressions with the target and its runtime anchors.
7. Three target-blind runs produced 12/0, 10/0, and 10/0. The promoted shared
   identity review repeated at 11/0 and 11/0. A production checkpoint on both
   projects contributed 15 TP / 0 FP.

## Results

**VALIDATED.**

The production implementation removes:

- the fixed reference regex and controller-side evidence inventory;
- project-profile evidence examples and controller citations;
- terminal-token-only participant handles;
- the participant suffix blacklist and manual pluralization;
- enumerated server/host/type/technology failure classes.

The controller now sees only three general capability descriptions, remaining
actions, and counts from completed tools. It still chooses the order across
three turns; completion remains structural after every bounded tool has run.
Each tool discovers an empty or non-empty candidate set for itself.

The remaining deterministic checker is intentionally narrower. It enforces
runtime catalog ownership and evidence integrity, not benchmark semantics:

- tokenize runtime component names and document words;
- propose a word when it continues a catalog token owned by exactly one
  component;
- exclude exact identities/current links and qualified identifier fragments;
- require an exact source substring and a listed runtime target anchor;
- deduplicate and bound evidence windows/batches.

These are still hard-coded operational policies—this spike minimizes
hard-coding; it does not claim zero code-level constraints. In particular,
prefix continuation is a general morphology approximation, and the ±2/±4
windows, three-anchor limit, and 25-case batch are fixed resource bounds.

Focused production checkpoint:

| Project | Participant TP | Participant FP | Focused prompt tokens |
| --- | ---: | ---: | ---: |
| BigBlueButton | 12 | 0 | 5,797 |
| TeamMates | 3 | 0 | 4,084 |
| Total | 15 | 0 | 9,881 |

This generalization has a measured cost. Focused tokens rise from spike 015's
6,632 to 9,881 (+49.0%), and each participant-bearing project uses two more
LLM calls because denotation and identity are separated. The controller itself
is smaller at 497 input tokens per project; the extra cost is entirely in the
broader participant tool. The production file falls from 654 to 599 lines
(-55, -8.4%).

Fresh paired five-project E2E:

| Variant | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S21 | 170 | 15 | 25 | 92.67 | 91.63 | 89.47 | 88.08 |
| general S24 | 182 | 8 | 13 | 96.07 | 95.40 | 94.55 | 93.81 |

S24 improves every promotion aggregate: +3.40 macro F1, +3.77 macro F2,
+5.08 pooled F1, and +5.73 pooled F2. The new participant path contributes
14 TP / 0 FP (11 BigBlueButton, 3 TeamMates) and contributes no false
positives on the other three projects. Fresh-run stochasticity leaves this
below spike 015's unusually strong 185/4/10 run, but the identical-floor
repeats and the fresh S21 comparison both pass the stated gates.
