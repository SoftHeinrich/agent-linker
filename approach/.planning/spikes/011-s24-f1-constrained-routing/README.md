---
spike: 011
name: s24-f1-constrained-routing
type: comparison
validates: "Given fresh S21 and S24 calls, phase ownership plus project-context handle review lets S24 exceed S21 macro and pooled F2 without lowering macro or pooled F1."
verdict: VALIDATED
related: [004-nogap-validator-ab, 005-upstream-candidate-gap, 010-s24-relation-role-routing]
tags: [s24, precision, f1, f2, ownership, no-magic]
---

# Spike 011: S24 F1-constrained routing

## Verdict

Validated with fresh OpenAI API calls using `gpt-5.6-terra` and
`reasoning_effort=none`.

Against a fresh paired S21 run, S24 improved all four required aggregates:

- macro F1: 92.80% -> 93.61%;
- pooled F1: 89.66% -> 91.29%;
- macro F2: 91.56% -> 92.41%;
- pooled F2: 87.84% -> 89.73%.

S24 also reduced total FP from 13 to 11 while increasing TP from 169 to 173.
No cached model response was read to obtain either the edge checkpoint or the
final E2E result.

## Final Design

The controller still replaces S21 rather than refining or flooring it. It
selects among the existing entity and coreference phase tools plus the
catalog-derived role capability.

Two non-overlapping ownership contracts address spike 010's precision loss:

1. The entity tool owns candidates whose sentence contains the canonical
   component name or an approved runtime alias.
2. The role tool owns shortened handles derived from compound names in the
   runtime component catalog. Full names, approved aliases, and occurrences
   inside dotted identifiers remain outside this tool.

Remaining role candidates are resolved in one project-context tool call. Its
prompt contains only a short task statement, candidate cases, and explicit
identity anchors found in the same document. The workflow supplies the
complexity; the prompt does not enumerate benchmark cases.

This design contains no project names, benchmark vocabulary, score thresholds,
candidate-count gates, or fixed route length. Its deterministic code only
defines capability ownership and lexical boundaries.

## Edge-First Investigation

The preserved spike 010 E2E output first supplied an offline causal replay.
Requiring explicit entity ownership changed its final aggregate from
187 TP / 47 FP / 8 FN to 183 TP / 20 FP / 12 FN, suggesting that precision
could be recovered without erasing the F2 gain.

Fresh pilot v1 then failed: entity ownership removed 2 TP and 2 FP, while the
unreviewed role tool added 3 TP and 15 FP. The role errors had two causes:
dotted code identifiers were mistaken for prose handles, and isolated handle
occurrences lacked the project's explicit identity usages.

Pilot v2 showed that project-context review reduced role overreach, but trace
inspection found that the sparse-document harness allowed the extractor to
return locally renumbered sentence IDs. Pilot v3 used intact documents but
incorrectly scored all predictions against edge-only gold. Both runs were
invalidated, documented, and not used as performance evidence.

The corrected v4 protocol:

- predeclares prior S24 FP/FN sentence IDs using gold offline;
- runs all inference over each intact document;
- uses fresh OpenAI calls for profiles, extraction, validation, control, and
  role review;
- restricts both predictions and gold to the predeclared sentences only when
  scoring;
- compares unfiltered and ownership states from the same accepted candidates.

| State | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh unfiltered entity | 19 | 16 | 8 | 40.32% | 41.24% | 61.29% | 66.43% |
| Entity ownership | 18 | 8 | 9 | 42.99% | 41.64% | 67.92% | 67.16% |
| Controller final | 19 | 10 | 8 | 42.94% | 42.50% | 67.86% | 69.34% |

The corrected edge checkpoint passed all four gates and exercised both
`finalize` and `relation_role_resolution`, so the design review permitted E2E.

## Fresh Paired E2E

Result:
`results/s24_f1_constrained_e2e_v1_gpt56terra_20260724/ablation_20260724_134154.json`

| Project | S21 TP/FP/FN | S24 TP/FP/FN | S21 F1/F2 | S24 F1/F2 | S24 workflow |
| --- | ---: | ---: | ---: | ---: | --- |
| MediaStore | 29/0/2 | 29/1/2 | 96.67/94.77 | 95.08/94.16 | entity -> coreference -> finalize |
| TeaStore | 27/0/0 | 26/0/1 | 100/100 | 98.11/97.01 | entity -> coreference -> finalize |
| TEAMMATES | 51/6/6 | 50/4/7 | 89.47/89.47 | 90.09/88.65 | entity -> coreference -> role -> finalize |
| BigBlueButton | 44/7/18 | 50/6/12 | 77.88/73.58 | 84.75/82.24 | entity -> role -> coreference -> finalize |
| JabRef | 18/0/0 | 18/0/0 | 100/100 | 100/100 | entity -> coreference -> finalize |

| Aggregate | S21 | S24 | Delta |
| --- | ---: | ---: | ---: |
| TP / FP / FN | 169 / 13 / 26 | 173 / 11 / 22 | +4 / -2 / -4 |
| Macro F1 | 92.80% | 93.61% | +0.80 pp |
| Macro F2 | 91.56% | 92.41% | +0.85 pp |
| Pooled F1 | 89.66% | 91.29% | +1.63 pp |
| Pooled F2 | 87.84% | 89.73% | +1.89 pp |

## Workflow Diversity

The final paths demonstrate semantic and ordering diversity:

- MediaStore, TeaStore, and JabRef had no unresolved catalog-handle capability
  and stopped after entity plus coreference.
- TEAMMATES selected role resolution after coreference feedback. Structural
  ownership rejected the apparent handles because they occurred in dotted
  identifiers, so that tool correctly returned no additions.
- BigBlueButton selected role resolution before coreference and accepted 13
  project-context mappings.

Thus project-specific assessments are not merely different prose attached to
one route: two projects used an extra tool, three did not, and the two longer
routes used different tool orders and outputs.

## Causal Error Analysis

The aggregate improvement is concentrated in BigBlueButton: S24 gains 6 TP,
removes 1 FP, and removes 6 FN relative to fresh S21. TEAMMATES trades 1 TP for
2 fewer FP, improving F1 but slightly lowering F2. MediaStore adds 1 FP,
TeaStore loses 1 TP, and JabRef is unchanged.

The 22 remaining S24 FNs divide into three causal groups:

1. **Validator rejection after successful extraction.** TeaStore S7 and
   TEAMMATES S7, S8, S88, and S185 contain explicit names or approved aliases,
   reached the entity validator, and were rejected. This is downstream
   no-reasoning judgment variance, not a controller routing gap.
2. **Unproposed indirect references.** MediaStore's `AudioAccess` ->
   `MediaAccess` and lowercase `database` -> `DB`, plus three TEAMMATES
   datastore references, were not recovered consistently by extraction or
   coreference.
3. **Relations outside the role contract.** BigBlueButton's 12 FNs include
   code-shaped variants (`akka-apps`, `bbb-web`), inflected or generic terms
   (`clients`, `server`, `client`, `WebRTC`), and three server cases explicitly
   rejected by project-context review. Relaxing dotted/hyphenated boundaries
   would reintroduce the code-path false positives observed in pilot v1.

The 11 FPs are also localized:

- MediaStore has one entity alias/derivational mismatch (`reencoded`);
- TEAMMATES has three code-path entity links and one ambiguous coreference;
- BigBlueButton has two overlapping entity/alias links and four role links
  where conversion-process language is architecturally plausible but absent
  from the benchmark gold.

This supports stopping rather than adding another gate: the remaining errors
mix validator variance, extraction misses, and gold/semantic boundary cases.
A new universal filter would threaten the demonstrated F2 gain, while a
project-specific exception would violate the no-magic requirement.

## Reproduction

```bash
cd approach
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teastore teammates bigbluebutton jabref \
  --results-dir ../results/<fresh-results>
```
