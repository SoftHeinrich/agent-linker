# S24 replacement orchestrator

## Decision

`SLinker24Orchestrator` replaces S21's fixed `link()` workflow. It subclasses
S21 only to reuse phase implementations; it never calls `SLinker21.link` and
does not preserve or post-process an S21 final result.

```text
raw document + component model
          |
model-profile + document-profile tools
          |
controller state
          +-- entity pipeline
          +-- coreference pipeline
          +-- semantic coverage audit
          `-- finalize
```

The controller chooses and orders phase tools. Candidate generation and
validation remain inside tools.

## Why this is agentic

After every tool call the controller receives concrete candidates, accepted
links, validator outcomes, document/reference style, component ambiguity, and
runtime aliases. It then chooses the next available capability or finalizes.

Autonomy is bounded by an acyclic capability registry: once a tool transforms
the state it is removed. There is no retry counter, score threshold, or maximum
step number.

The current five-project replay converged on the same tool sequence for every
project. This is reported honestly: the architecture permits project-specific
routes, but this registry did not produce route diversity on these documents.
Project-specific profiles, decisions, evidence, and outputs still differed.
Because the audit overlaps the other evidence modes and has no observable cost,
calling every remaining capability is rational under the current contract.

## No-magic boundary

Runtime has no:

- S21 floor or final-result union;
- benchmark score or gold input;
- project vocabulary;
- candidate-count or confidence threshold;
- prefix length, context window, frequency cutoff, or special component family;
- controller authority to create or approve links.

The audit enforces only structural validity before the existing validator:
the target must be a runtime catalog member and the quoted words must occur in
the source sentence.

## Error-analysis trail

An appeal-only replacement was tested first. A generic semantic appeal produced
7 TP / 12 FP; explicit referent identity and claim ownership improved that to
5 TP / 6 FP, still below existing S24. It was also the wrong architecture because
it refined rejected S21 candidates.

The replacement workflow instead operates from raw project inputs. A unified
coverage audit outperformed a split identity/reference audit. The split created
two workflow orders but reduced F2 and added false positives; route diversity
was therefore not manufactured at the expense of quality.

## Fixed-phase replay

Saved S21 phase checkpoints were treated as deterministic recordings of reusable
knowledge/entity/coreference tool calls. The controller assembled its own state;
S21 final output was loaded only after execution as a comparison.

| System | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S21 | 168 | 5 | 27 | 93.34% | 90.83% | 91.30% | 88.14% |
| Dynamic augmentation | 173 | 5 | 22 | 94.68% | 92.80% | 92.76% | 90.29% |
| Replacement orchestrator | 181 | 12 | 14 | 94.50% | 94.75% | 93.30% | 93.01% |

With F2 as the primary objective, replacement beats S21 by:

- +3.92pp macro F2;
- +4.87pp pooled F2;
- +1.16pp macro F1;
- +2.00pp pooled F1.

It exceeds the prior dynamic augmentation by 1.95pp macro F2 and 2.72pp pooled
F2. Macro F1 is 0.17pp lower, while pooled F1 is 0.54pp higher. The lower
precision and higher recall remain visible in reporting.

## End-to-end smoke

A fresh Codex Mediastore run executed the replacement workflow from raw inputs:

- workflow: profile → entity → coreference → coverage audit → finalize;
- final: 31 TP, 1 FP, 0 FN;
- F1: 98.41%;
- F2: 99.36%.

The audit returned no accepted additions because the fresh entity phase already
covered every gold link. This validates full execution and safe no-op behavior,
not an all-project same-backend replication.

## Reporting

`run_ablation.py` now emits per-project F1/F2 and both macro and pooled summary
rows. F2 is primary for this variant; F1, precision, FP, and FN remain mandatory
context.

## Commands

```bash
cd approach
../.venv/bin/python pilot/test_s24_orchestrator.py
../.venv/bin/python pilot/s24_replacement_orchestrator_pilot.py
LLM_BACKEND=codex ../.venv/bin/python run_ablation.py \
  --variants s_linker24_orchestrator --datasets mediastore
```
