# V29 Experiment Details (Feb 21)

## Goal
Replace V26a's advocate-prosecutor-jury deliberation for TransArc FPs with a decomposed contrastive judge using NeurIPS/ICLR prompt techniques.

## V26a Baseline: 95.4% macro F1
- MS 98.4% (FP=1), TS 100% (FP=0), TM 94.2% (FP=7), BBB 89.6% (FP=12), JAB 94.7% (FP=2)
- 13 stubborn TransArc FPs: TM S17/S188→E2E, S22→Logic, S125→Storage, S173→Test Driver; BBB S5/S74→WebRTC-SFU, S18/S68→HTML5 Server, S60→FreeSWITCH; JAB S5→logic, S7→preferences; MS S37→Reencoding
- Gate `_is_ambiguous_name_component()` too narrow: only single-word, non-CamelCase, non-uppercase

## V29 Run 1: ALL TransArc + 3-question decomposed + intersection voting → 77.7%
- MS 90.3%, TS 76.0%, TM 89.4%, BBB 76.5%, JAB 56.0%
- **Catastrophic TP loss**: JAB killed 11 TPs (lowercase names), BBB killed 20/24 TA links
- Root cause: Q1 (Subject match), Q2 (not technology/tool), Q3 (architectural discourse) too strict
- Q2 "not a technology" fundamentally wrong — components ARE named after technologies (FreeSWITCH, kurento)
- Conjunction of 3 sub-questions = very high rejection rate

## V29 Run 2: Same prompt, union voting → 74.1%
- MS 92.1%, TS 84.0%, TM 59.8%, BBB 73.2%, JAB 61.5%
- **Even worse** — proves the prompt itself is broken, not the voting strategy
- TM: 36/55 TransArc links rejected even with union (both passes reject generic-word components)
- BBB: 20/24 rejected — FreeSWITCH, Redis PubSub, kurento all fail "not technology" check

## V29 Run 3: Widened gate + softened 2-question prompt + intersection + safe examples → 94.2%
- MS 98.4% (FP=1), TS 93.1% (FP=4), TM 90.4% (FP=6), BBB 88.9% (FP=13), JAB 100% (FP=0)
- **Key changes**:
  1. Restored ambiguous gate but WIDENED: catches ALL Phase 1 ambiguous names (any format) + names with generic word parts
  2. Softened prompt to 2 questions: Q1-Reference (generic vs component?) + Q2-Architectural (describes role?)
  3. Removed "not technology/tool" rejection (P2 from run 1)
  4. Added 5 few-shot examples from safe SE textbook domains (Scheduler, Optimizer, Renderer, Parser)
  5. Intersection voting (both passes must approve) — safe because only ambiguous names judged
- **JAB 100%** — killed both FPs (S7→preferences, S5→logic) with zero TP loss
- **TM 90.4%** — killed 5 FPs but also lost 5 TPs (Logic/Storage intersection too aggressive)
- **BBB 88.9%** — only 2 Apps links went to judge; FreeSWITCH/WebRTC-SFU bypass as non-ambiguous
- **TS 93.1%** — 4 partial_inject FPs (LLM variance in earlier phases, not judge-caused)

## Why V29-gate is -1.2pp below V26a
1. **TS regression** (100→93.1%): 4 partial_inject FPs from LLM variance, not judge
2. **TM regression** (94.2→90.4%): Intersection voting killed 5 valid Logic/Storage TPs that V26a's union-voted deliberation would keep
3. **BBB similar** (89.6→88.9%): 5 transarc FPs survive because FreeSWITCH/WebRTC-SFU not ambiguous
4. **JAB improvement** (94.7→100%): Widened gate + decomposed judge killed both FPs perfectly

## Potential Next Steps
- Switch TM to union voting (approve if either pass approves) to recover 5 TPs — but may re-admit FPs
- Expand ambiguous gate to also catch technology-named components (FreeSWITCH, WebRTC-SFU) for BBB
- Combine V29 gate with V26a deliberation as fallback for borderline cases
