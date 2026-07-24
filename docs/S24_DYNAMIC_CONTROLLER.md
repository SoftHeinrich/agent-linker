# S24 dynamic controller

## Decision

Promote `SLinker24Dynamic` as the profile-aware S24 experiment. It keeps the
complete S21 output as its floor, prepares a runtime project profile, selects
one bounded recovery phase, observes its validation funnel, and then decides
whether to stop or call the remaining phase.

```text
unchanged S21 floor
        |
document + component + floor + candidate profiles
        |
structured controller decision
        |
one bounded phase
        |
validator funnel feedback
        +---- stop
        `---- reconsider independent remaining phase
        |
deduplicating union with the S21 floor
```

The controller selects workflow only. It cannot propose or approve trace links.

## Runtime profile

The controller receives:

- document size and floor-linked sentence coverage;
- complete runtime component catalog, Phase-1 ambiguity labels, per-component
  floor-link counts, and components without floor links;
- Phase-1-approved aliases and scopes;
- S21 floor source mix;
- concrete, grounded candidate sketches for each eligible recovery phase.

No project identifier, gold link, benchmark score, or benchmark-specific prompt
vocabulary is present.

## Decision and feedback

Every controller step returns a compact assessment:

- documentation regime;
- component-catalog risk;
- strongest remaining evidence channel;
- expected marginal gain;
- false-positive risk;
- one action and a runtime-evidence-based reason.

`alias_phase4` reports proposed candidates, distinct targets, Phase-4 pass-one
and pass-two approvals, agreement, disagreements, and accepted targets.
`anchored_reference` reports eligible, resolver-approved, validator-approved,
and accepted targets. This feedback is included in the next controller step.

The final pilot demonstrates actual adaptation: one project first selected
anchored recovery, observed 0 approvals from 27 candidates, then selected the
independent alias phase and recovered one validated link.

## Oracle-analysis boundary

At the user's direction, gold was inspected before the final controller prompt
was designed. The analysis was used to derive generic evidence regimes and the
feedback contract. It established that raw candidate count is a poor routing
signal: the observed anchor pools ranged from 1/1 to 2/27 and 7/11 gold.

Gold never enters runtime. However, because all five projects informed design,
the fixed-floor result is oracle-informed method development, not held-out
generalization evidence. An unseen-project replication remains necessary.

## Fixed-floor results

The first frozen oracle-informed protocol passed:

| Measure | Result |
| --- | ---: |
| Marginal TP / FP | 5 / 0 |
| Marginal precision | 100% |
| S21 macro F1 | 93.34% |
| Dynamic S24 macro F1 | 94.68% |
| Delta | +1.34pp |
| Recovery phases executed | 4 |
| Run-all phase count | 6 |
| Phase executions saved | 2 |

After adding the generic zero-yield feedback rule, the controller produced four
distinct workflows and the adaptive two-step route:

| Project | Ordered workflow | Marginal TP / FP |
| --- | --- | ---: |
| mediastore | alias | 1 / 0 |
| teastore | anchor → stop | 1 / 0 |
| teammates | anchor (0 accepted) → alias | 1 / 0 |
| bigbluebutton | anchor | 1 / 0 |
| jabref | no-op | 0 / 0 |
| **Aggregate** | **4 workflows** | **4 / 0** |

The feedback iteration retained 100% marginal precision and improved macro F1
from 93.34% to 94.48% (+1.14pp), while executing five phases instead of six.
The difference from the first passing run is anchored-resolver variance.

## Live end-to-end result

A normal Codex-backed Mediastore run freshly executed all S21 phases and then
the dynamic controller:

- internal S21 floor: 30 TP, 1 FP, 1 FN, F1 96.77%;
- workflow: `alias_phase4`;
- marginal addition: 1 TP, 0 FP;
- final: 31 TP, 1 FP, 0 FN, F1 98.41%;
- same-run delta: +1.64pp F1.

## Commands

```bash
cd approach
../.venv/bin/python pilot/test_s24_dynamic.py
../.venv/bin/python pilot/s24_dynamic_controller_pilot.py
LLM_BACKEND=codex ../.venv/bin/python run_ablation.py \
  --variants s_linker24_dynamic --datasets mediastore
```
