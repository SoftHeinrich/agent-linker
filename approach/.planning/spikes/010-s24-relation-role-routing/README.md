---
spike: 010
name: s24-relation-role-routing
type: standard
validates: "Given fresh project inputs, a grounded controller selects non-overlapping phase tools from the document/component profile, produces project-specific workflows, and exceeds fresh same-backend S21 macro and pooled F2 without runtime numeric gates or project vocabulary."
verdict: VALIDATED
related: [005-upstream-candidate-gap, 009-s24-replacement-orchestrator]
tags: [s24, replacement, dynamic-workflow, relation-role, fresh-run]
---

# Spike 010: S24 relation/role routing

## Verdict

Validated with fresh OpenAI API calls using `gpt-5.6-terra` and
`reasoning_effort=none`.

The replacement controller beat fresh S21 on both primary metrics:

- macro F2: 92.70% -> 93.95% (+1.24 percentage points);
- pooled F2: 89.69% -> 92.21% (+2.52 percentage points).

It also produced three distinct workflows across five projects. Macro F1 fell
from 93.19% to 90.51%, so this is specifically an F2/recall improvement rather
than an across-the-board precision improvement.

## Final Design

S24 does not call `SLinker21.link()` and has no protected S21 floor. It reuses
three phase capabilities:

1. `entity_pipeline` for exact names and approved aliases;
2. `coreference_pipeline` for pronouns and anaphoric references;
3. `relation_role_resolution` for project-specific shortened handles derived
   from the runtime component catalog.

The controller sees the document, component profile, available capabilities,
current links, and normalized prior-tool feedback. It may select each available
tool once or finalize. Every non-final action must retain at least one exact
document quote; paraphrased quotes are discarded and a wholly ungrounded action
fails closed.

Role capability discovery is structural:

- a space-separated compound name exposes its terminal word;
- a hyphenated compound name exposes its identifier segments;
- a handle must belong to only one catalog component;
- only exact standalone occurrences match;
- synthesized plurals and substrings inside hyphenated identifiers do not
  match.

There is no project vocabulary, score threshold, route count, step count, or
benchmark-specific branch. The runtime catalog and document provide all
project-specific information.

The broad `coverage_audit` from spike 009 is intentionally absent. It overlaps
the three reference modes and caused the controller to rationally run it on
every project.

## Edge-First Protocol

Gold labels were used offline only to select sentences containing prior FNs or
FPs. Model knowledge, document knowledge, entity extraction, validation, and
controller decisions were fresh OpenAI calls. Gold was never included in a
model prompt.

The final all-project edge checkpoint:

| State | TP | FP | FN | Macro F2 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh entity-only | 16 | 10 | 12 | 58.74% | 57.97% |
| Controller + role tool | 24 | 10 | 4 | 66.84% | 82.19% |

Both `finalize` and `relation_role_resolution` occurred. The role tool added
8 TP and 0 FP.

## Fresh E2E Result

Result:
`results/s24_role_e2e_v3_noaudit_gpt56terra_20260724/ablation_20260724_123747.json`

| Project | S21 F2 | S24 F2 | S24 TP / FP / FN | Final S24 workflow |
| --- | ---: | ---: | ---: | --- |
| MediaStore | 94.16% | 94.77% | 29 / 0 / 2 | entity -> coreference -> finalize |
| TeaStore | 100.00% | 95.74% | 27 / 6 / 0 | entity -> coreference -> finalize |
| TEAMMATES | 89.29% | 88.44% | 52 / 14 / 5 | entity -> role -> coreference -> finalize |
| BigBlueButton | 80.00% | 90.77% | 61 / 27 / 1 | entity -> coreference -> role -> finalize |
| JabRef | 100.00% | 100.00% | 18 / 0 / 0 | entity -> coreference -> finalize |

| Aggregate | S21 | S24 | Delta |
| --- | ---: | ---: | ---: |
| TP / FP / FN | 174 / 16 / 21 | 187 / 47 / 8 | +13 / +31 / -13 |
| Pooled recall | 89.23% | 95.90% | +6.67 pp |
| Macro F1 | 93.19% | 90.51% | -2.68 pp |
| Macro F2 | 92.70% | 93.95% | +1.24 pp |
| Pooled F2 | 89.69% | 92.21% | +2.52 pp |
| LLM calls | 98 | 112 | +14 |

## Workflow Diversity

Route diversity is observed, not merely project-specific prose attached to one
shared route:

- MediaStore, TeaStore, and JabRef had no applicable unresolved handle route
  and finalized after entity plus coreference.
- TEAMMATES exposed catalog-handle evidence before the controller selected
  coreference.
- BigBlueButton exposed handle evidence after entity and coreference feedback.

Thus there are three distinct ordered workflows. Controller assessments,
evidence quotes, available-tool profiles, and tool outputs are also
project-specific.

## Causal Error Analysis

The full E2E with `coverage_audit` failed before the final design:

- S24 macro F2 90.2%, pooled F2 89.6%, 86 FP;
- the audit was called on all five projects;
- it contributed 56 FP.

Removing it was a semantic-ownership correction, not a numeric gate. A saved
counterfactual excluding only audit-sourced links scored macro F2 95.12% and
pooled F2 93.78%, motivating the narrowed registry. The subsequent fresh run
confirmed improvement.

Final S24 has eight FNs:

- MediaStore: `MediaAccess` S25 and `DB` S33;
- TEAMMATES: `UI` S4 and `Logic` S7, S8, S88, S185;
- BigBlueButton: plural `clients` -> `HTML5 Client` at S19.

These sit outside the exact standalone-handle contract. Adding plural synthesis
was explicitly piloted and rejected because it produced `tests -> Test Driver`.

The 47 FPs decompose into 35 from the reused entity phase and 12 from the role
tool. Role errors are concentrated in generic terminal words in TEAMMATES and
BigBlueButton. The independent fresh S24 entity calls also overgenerated more
than the independent S21 calls, showing material no-reasoning run variance.
This is the main precision limitation and explains why F1 falls while F2 rises.

## Investigation Trail

Preserved failed checkpoints establish the design progression:

1. A local free-text resolver plus overlapping judges missed true edge cases
   and accepted internal-process roles.
2. A learned full-document lexicon mapped descriptive concepts such as
   `center`, `core`, and `outer layers`, adding only FPs.
3. Catalog-unique handles found the correct BBB candidates, but the reused
   S21 specificity validator contradicted the generic-handle premise.
4. A short participation judge passed once and reversed every BBB decision on
   the next fresh run.
5. Deterministic exact handles removed judge variance; eliminating synthesized
   plurals removed the TEAMMATES edge FP.
6. The first full E2E exposed the overlapping coverage audit as the dominant
   FP source. Removing that capability yielded the validated final result.
7. A controller quote-grounding crash was fixed by retaining exact quotes and
   discarding only invalid ones; wholly ungrounded actions still fail.

## Reproduction

```bash
cd approach
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/<fresh-state-dir> \
LLM_LOG_DIR=../results/<fresh-log-dir> \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teastore teammates bigbluebutton jabref \
  --results-dir ../results/<fresh-results>
```
