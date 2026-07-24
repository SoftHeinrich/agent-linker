---
spike: 014
name: s24-discourse-scope-participants
type: comparison
validates: "Given the exact saved S24 non-role floor, when a structured discourse-scope resolver replaces relation-role review, then it adds at least three net true positives at 95% marginal precision while improving both BigBlueButton F1 and F2."
verdict: INVALIDATED
related: [010-s24-relation-role-routing, 011-s24-f1-constrained-routing, 013-s24-lexical-entity-normalization]
tags: [s24, discourse, participant-resolution, role-ownership, replacement]
---

# Spike 014: S24 discourse-scope participants

## What This Validates

Can a discourse-grounded participant resolver replace the current
`relation_role_resolution` tool and recover generic or inflected component
references without restoring the old tool's false positives?

The pilot is replacement-based:

1. load the exact saved S24 link set from spike 013;
2. remove all `s24_relation_role` links;
3. derive a new candidate set from unique terminal-role nouns;
4. review candidates with full-document discourse evidence;
5. compare the replacement against the original result on the same floor.

## Research

No external library is needed. Prior spikes already establish the relevant
comparison:

| Approach | Strength | Failure mode |
| --- | --- | --- |
| Exact handle review | Bounded and catalog-grounded | Lacks discourse scope; current BBB role output is 4 TP / 2 FP |
| Broad semantic audit | High recall | Overlapping ownership and severe FP growth |
| Discourse-scope replacement | Uses the same bounded candidate ownership with explicit section/anchor/claim evidence | May still confuse deployment roles with component identity |

Chosen approach: replace the role judge while retaining structural candidate
ownership. Do not add another overlapping tool.

## Design Contract

- Candidate nouns come only from unique terminal tokens of compound runtime
  catalog names.
- Singular and regular plural occurrences are eligible.
- Full names, approved aliases, orthographic identities, dotted paths,
  hyphenated identifiers, and already-linked pairs are excluded.
- Every approval must provide:
  - an exact document section or discourse anchor;
  - an exact identity-anchor quote from a verified sentence for the target;
  - an exact local bridge quote establishing the discourse chain;
  - an exact architectural-claim quote from the candidate sentence;
  - the participant role and strongest competing referent.
- The highlighted noun must itself participate in a finite architectural
  claim; modifier-only workflow, process, stage, artifact, and technology
  names are rejected.
- Missing or non-verbatim evidence fails closed.

## Promotion Gate

- replacement role output gains at least three TP over the old role output;
- marginal role precision is at least 0.95;
- role FP do not exceed the old role FP;
- BigBlueButton replacement F1 and F2 both exceed the exact saved baseline;
- no benchmark vocabulary, project identity, score, or candidate count enters
  runtime logic or prompts.

## How to Run

```bash
../.venv/bin/python pilot/test_s24_discourse_scope.py

OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
  ../.venv/bin/python pilot/s24_discourse_scope_pilot.py \
  --datasets bigbluebutton \
  --baseline-dir ../results/s24_lexical_entity_e2e_v1_20260724 \
  --results-dir ../results/s24_discourse_scope_pilot_v1_20260724
```

## Investigation Trail

1. The exact saved BigBlueButton role output contains 4 TP and 2 FP. A valid
   replacement must retain that utility and add at least three net TP.
2. Pilot v1 used a strict component-identity interpretation. It reproduced the
   old role output exactly: 4 TP / 2 FP, with no aggregate change.
3. Pilot v2 admitted deployed instances, endpoints, owned state, and
   user-facing instances inside an anchored discourse scope. It reached
   6 TP / 2 FP and improved BigBlueButton F1 from 87.60% to 89.43%, but
   marginal precision remained only 75%.
4. Pilot v3 required the noun to be a semantic participant in a finite claim
   and required a verbatim local bridge. It rejected both old role false
   positives and retained the two new true positives: 6 TP / 0 FP.
5. The remaining reachable endpoint case was still rejected because the
   document explicitly names a broader server referent. Overriding that
   grounded distinction would be benchmark-label tuning, not a safe generic
   discourse rule.
6. The best result therefore missed the predeclared true-positive gate by one.
   No production code was changed and no five-project E2E was run.

## Results

**INVALIDATED — do not promote.**

| Run | Role TP / FP | Final TP / FP / FN | F1 | F2 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Saved S24 | 4 / 2 | 53 / 6 / 9 | 87.60% | 86.32% | reference |
| v1 strict identity | 4 / 2 | 53 / 6 / 9 | 87.60% | 86.32% | fail |
| v2 instance-aware | 6 / 2 | 55 / 6 / 7 | 89.43% | 89.00% | fail precision |
| v3 participant + bridge | 6 / 0 | 55 / 4 / 7 | 90.91% | 89.58% | fail reach |

The v3 contract is materially cleaner than the current role judge, but the
experiment was explicitly designed to require at least three additional true
positives before replacing a working tool. Its two-TP improvement is
insufficient for promotion in this study.

No additional high-value FN tool is recommended from this slice. A future
attempt may revisit deterministic section topology only if a catalog-wide
reach audit demonstrates more than this single unresolved endpoint; it should
remain part of the same role-ownership path, not become an overlapping tool.
