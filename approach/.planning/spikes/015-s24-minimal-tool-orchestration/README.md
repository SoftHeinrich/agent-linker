---
spike: 015
name: s24-minimal-tool-orchestration
type: comparison
validates: "Given the validated S24 tool ownership, when the controller sees compact evidence signals and the role judge sees local evidence packs, then S24 preserves its clean role contribution and aggregate gates with materially smaller, more general prompts."
verdict: VALIDATED
related: [009-s24-replacement-orchestrator, 013-s24-lexical-entity-normalization, 014-s24-discourse-scope-participants]
tags: [s24, simplicity, controller, prompts, orchestration]
---

# Spike 015: S24 minimal-tool orchestration

## What This Validates

Can the active S24 be made conceptually smaller without becoming a fixed
pipeline?

The intended architecture has three simple tools:

- **identity** — named component evidence;
- **reference** — references to previously introduced components;
- **participant** — generic participant nouns grounded by context.

The controller remains multi-turn. It sees only evidence signals, completed
tool outcomes, and remaining actions. Deterministic probes omit tools with no
evidence; the controller orders the evidence-bearing tools and completion is
automatic when none remain. It never emits link decisions.

## Research

No external library is required. This is an architectural comparison against
the current S24 implementation.

| Surface | Current | Minimal comparison |
| --- | --- | --- |
| Controller input | Full document, complete profile, history, current links | Tool evidence counts/examples and compact outcomes |
| Controller schema | Action, quotes, obligation, reason | Action, cited evidence IDs, reason |
| Participant judge | Full document, all anchors, six evidence fields | Local context, nearest anchors, claim/anchor/alternative |
| Tool scheduling | Full-profile availability | Compact probes detect tools; controller orders them |
| Runtime variants | One active S24 | One active S24; pilot subclass is never registered |

## Design Contract

1. Tools own evidence modes, not benchmark cases.
2. The controller receives no gold data and makes no link decisions.
3. Controller evidence is derived from runtime catalog, aliases, text, and
   deterministic candidate ownership; evidence-free tools are omitted.
4. Each state-transforming tool can run at most once.
5. The participant judge receives the source, nearby context, and nearest
   verified target anchors rather than the full document.
6. Approval requires an exact target anchor and exact source claim, plus the
   strongest alternative referent.
7. Canonical S21 remains byte-stable.

## Promotion Gates

- saved-floor participant replay retains at least 10 TP / 0 FP across
  BigBlueButton and TeamMates;
- controller remains multi-turn and selects only tools with runtime evidence;
- controller plus participant prompt tokens fall by at least 50% on the
  focused projects;
- production checkpoint has zero participant-source FP;
- fresh paired five-project E2E does not regress either macro or pooled F1/F2
  versus S21;
- the active S24 has fewer prompt fields, no unused identity-review method,
  and no second runnable S24 variant.

## How to Run

```bash
../.venv/bin/python pilot/test_s24_orchestrator.py
../.venv/bin/python pilot/test_s24_simple_orchestrator.py

OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
  ../.venv/bin/python pilot/s24_simple_orchestrator_pilot.py \
  --mode replay \
  --datasets bigbluebutton teammates \
  --baseline-dir \
    ../results/s24_simple_production_checkpoint_v1_20260725 \
  --results-dir \
    ../results/s24_simple_quote_host_replay_v2_contract_20260725

OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=\
../results/s24_simple_e2e_v1_contract_20260725/phase_states \
LLM_LOG_DIR=../results/s24_simple_e2e_v1_contract_20260725/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/s24_simple_e2e_v1_contract_20260725
```

## Investigation Trail

1. The spike-014 controller consumed 22,262 input tokens across three
   TeamMates turns and 15,725 across three BigBlueButton turns.
2. The current participant judge consumed another 6,594 TeamMates and 4,703
   BigBlueButton input tokens because it resends each complete document.
3. The active S24 file is 725 lines and still contains an unused generic
   identity-review method left behind by the superseded role judge.
4. Replay v1 deduplicated the evidence table but recovered only 9 TP / 1 FP.
   Sharing the bounded evidence table across cases produced 10 TP / 0 FP.
5. Full pilot v1 allowed an LLM `finalize` action. It immediately stopped
   BigBlueButton at zero links, proving that autonomous stopping was unsafe.
   Full pilot v2 listed only evidence-bearing tools and completed
   automatically after they were consumed. The controller still chose the
   order on each turn.
6. The first promoted checkpoint exposed quoted claims: the model returned
   exact substrings wrapped in delimiters, so the lexical gate rejected every
   BigBlueButton participant. Stripping only surrounding quote delimiters
   recovered the evidence without weakening substring grounding.
7. A general hardware-host boundary and explicit output contract rejected the
   remaining false positive and prevented ellipsized claims or unlisted
   anchors.

## Results

**VALIDATED.**

Focused production checkpoint:

| Project | Participant TP | Participant FP | Focused prompt tokens |
| --- | ---: | ---: | ---: |
| BigBlueButton | 9 | 0 | 4,372 |
| TeamMates | 3 | 0 | 2,260 |
| Total | 12 | 0 | 6,632 |

The prior controller plus participant surface used 49,284 tokens on the same
projects. The promoted surface uses 6,632, an 86.5% reduction. The production
file fell from 725 to 654 lines, the unused identity-review path was removed,
and the pilot now aliases the sole production class rather than defining
another runnable linker.

Full pilot v2 also demonstrated genuine turn-by-turn scheduling:
BigBlueButton selected identity → participant → reference, while TeamMates
selected identity → reference → participant. Evidence-free participant tools
were absent on MediaStore, TeaStore, and JabRef in the final E2E.

Fresh paired five-project E2E:

| Variant | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S21 | 175 | 16 | 20 | 93.71 | 93.29 | 90.67 | 90.11 |
| compact S24 | 185 | 4 | 10 | 97.13 | 96.47 | 96.35 | 95.46 |

S24 improves every aggregate (+3.42 macro F1, +3.18 macro F2, +5.68 pooled
F1, +5.35 pooled F2) and its participant source contributes 13 TP / 0 FP.
TeaStore and JabRef remain perfect. Raw scores and link CSVs are preserved in
`../results/s24_simple_e2e_v1_contract_20260725/`.
