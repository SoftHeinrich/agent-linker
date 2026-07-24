---
spike: 013
name: s24-lexical-entity-normalization
type: comparison
validates: "Given catalog names and document surface forms, when exact unique lexical signatures augment entity candidates, then the entity validator retains S24's identifier true positives without a separate controller tool or identifier prompt."
verdict: VALIDATED
related: [011-s24-f1-constrained-routing, 012-s24-targeted-fn-tools]
tags: [s24, lexical-normalization, entity-ownership, identifiers, icse]
---

# Spike 013: S24 lexical entity normalization

## What This Validates

Can exact orthographic normalization become part of entity identity, replacing
the standalone catalog-identifier tool while retaining its clean recoveries?

Promotion requires:

- every accepted lexical addition is a gold link;
- the two known exact-identifier recoveries remain reachable;
- dotted paths, partial names, plurals, synonyms, and catalog-ambiguous
  signatures remain ineligible;
- no identifier-specific LLM review or controller action remains;
- a fresh paired five-project E2E still improves macro and pooled F2 over S21
  without lowering either F1 aggregate.

## Research

No external library is required. The comparison is architectural:

| Approach | Advantage | Problem |
| --- | --- | --- |
| Separate identifier tool | Explicit trace and isolated review | Fragments entity identity and adds controller/prompt surface |
| Fuzzy lexical recovery | May reach more residual FNs | Prior pilot admitted hyphenated prose false positives |
| Exact catalog signature inside entity | Generic, deterministic candidate ownership; reuses entity validation | Cannot recover semantically divergent aliases or contextual references |

Chosen approach: exact catalog signatures inside the entity pipeline.

## Design Contract

1. Normalize Unicode with NFKC.
2. Split catalog names and document spans on whitespace, hyphen, underscore,
   CamelCase, acronym, and digit boundaries.
3. Require equality of the complete case-folded token tuple.
4. Require that tuple to identify exactly one runtime catalog component.
5. Reject qualified dotted paths and partial surrounding identifiers.
6. Do not synthesize prefixes, plurals, synonyms, abbreviations, or edits.
7. Add only candidates missing from ordinary entity extraction, then use the
   unchanged entity evidence validator.

## How to Run

```bash
../.venv/bin/python pilot/test_s24_lexical_entity.py

../.venv/bin/python pilot/s24_lexical_entity_pilot.py \
  --backend codex \
  --datasets bigbluebutton \
  --results-dir ../results/s24_lexical_entity_pilot_20260724
```

## What to Expect

The deterministic contracts must pass first. The live pilot must report at
least the previously established exact-identifier true positives with zero
lexical-source false positives before promotion and full E2E.

## Investigation Trail

1. Same-run counterfactual removal of the identifier tool changes aggregate
   S24 from 178/7/17 to 176/7/19. The tool's causal contribution is exactly two
   true positives and no false positives.
2. Catalog-wide deterministic reach analysis found orthographic variants for
   `BBB web` and `ImageProvider`. The latter are already covered by ordinary
   entity extraction in the preserved run; only the two `bbb-web` occurrences
   are residual-FN opportunities.
3. The other residual FNs are not lexical equivalents: they require divergent
   aliases, contextual roles, protocol-to-component realization, plural
   reference, or reversal of an existing validator rejection. Exact
   normalization cannot safely claim them.
4. The first live command failed before inference because the host exposes the
   credential as `OAI_KEY`, not `OPENAI_API_KEY`. The project memory now records
   the process-local mapping; no credential value was persisted.
5. A fresh BigBlueButton OpenAI pilot using `gpt-5.6-terra` passed with two
   lexical TP, zero lexical FP, and 26 calls. Both `bbb-web` occurrences passed
   the unchanged entity validator. The workflow contained entity, coreference,
   and relation-role phases only.
6. The promoted five-project paired E2E passed all four aggregate gates. The
   lexical source again contributed two TP and zero FP.

## High-value next-tool design

Only one additional capability has enough same-run FN reach to justify a
future pilot:

### Discourse-scope participant resolution

This should replace and generalize `relation_role_resolution`, not become an
overlapping fifth tool.

**Owned evidence mode:** generic or inflected participant nouns such as
`server`, `client`, and `clients` whose identity is established by an explicit
component introduction and maintained within a document section or local
discourse chain.

**Runtime evidence:**

- section/heading boundaries from the input document;
- exact prior component-name or approved-alias anchors;
- the candidate noun and complete source sentence;
- competing catalog components exposing the same role noun;
- accepted/rejected entity and coreference feedback.

**Decision contract:**

1. Build a section-local referent ledger from explicit identity anchors.
2. Propose only when one active referent owns the noun in that scope.
3. Require a judge to identify the grammatical participant, cite the anchor,
   rule out the strongest competing referent, and cite the architectural claim.
4. Exclude full names, aliases, orthographic variants, pronouns, code paths,
   and technology/protocol mentions; those remain owned elsewhere.
5. Fail closed at section changes, competing active referents, generic
   deployment language, or missing exact quotes.

**Current maximum reach:** seven of the fifteen residual FNs, all in
BigBlueButton: five `HTML5 Server` and two `HTML5 Client` references. Four were
already proposed by the old handle tool but rejected, so a future pilot must
compare the new discourse evidence against the old judge rather than restoring
gold labels.

**Promotion gate:** at least three marginal TP, marginal precision at least
0.95, no regression in either F1 aggregate, and fewer role-source FP than the
current tool on a same-candidate replay.

No other new tool currently clears the value/ownership bar:

- exact `Logic` cases are extractor/validator disagreements; spike 008 showed
  that broad appeals restore too many FP;
- `AudioAccess`, `database`, and `distributed datastore` need semantic alias
  induction, whose repeated pilot was unstable;
- `WebRTC` to `WebRTC-SFU` crosses a protocol/component boundary and is
  genuinely debatable rather than a safe identity recovery.

## Results

**VALIDATED.**

Fresh focused pilot:

- lexical additions: 2 TP / 0 FP;
- BigBlueButton final: 52 TP / 6 FP / 10 FN;
- workflow: entity → coreference → relation-role → finalize;
- 26 LLM calls, one fewer than the prior identifier-tool trace.

Fresh paired five-project E2E:

| Variant | TP / FP / FN | Macro F1 | Pooled F1 | Macro F2 | Pooled F2 |
| --- | --- | ---: | ---: | ---: | ---: |
| S21 | 174 / 26 / 21 | 90.95% | 88.10% | 91.93% | 88.78% |
| S24 lexical entity | 180 / 11 / 15 | 95.29% | 93.26% | 94.75% | 92.69% |
| Delta | +6 / -15 / -6 | +4.34 pp | +5.16 pp | +2.82 pp | +3.91 pp |

The full E2E used 112 S24 calls. Orthographic normalization added no
identifier-specific call: its candidates were batched into the existing entity
validator.
