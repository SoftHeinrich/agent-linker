# S24: anchored reference recovery

## Goal

Increase S21 recall without reopening the broad candidate stream that caused the
S23 precision collapse. S24 is a **subclass**, not a rewrite: S21's Phase 1–6
floor, its entity gate, and its accepted links remain unchanged.

The fresh GPT-5.4 S21 control misses 25 gold links: 14 in BigBlueButton, 8 in
teammates, and 3 in mediastore. The high-value gap is not ordinary named-entity
extraction. It is a reference-resolution gap:

| Gap | Examples | Count in the GPT-5.4 control | S24 treatment |
| --- | --- | ---: | --- |
| Local role/alias reference | `HTML5 client/server`, `bbb-html5`, `DataStorage`, `AudioAccess` | about 15–16 | Target |
| Unique structural shorthand | `WebRTC` → `WebRTC-SFU` | 2 | Target |
| Generic ambiguous term | `logic`, `storage`, `datastore` | 8 | Defer |
| Diagram/meta caption | BBB S81 presentation-conversion diagram | 1 | Defer |

The deferred classes are intentional. The same generic word is sometimes a gold
reference and sometimes an FP, so accepting it globally would recreate the S23
router's precision problem.

## Minimal design

`SLinker24(SLinker21)` adds one narrow recovery pass after S21 Phase 5 and
before merge. The first implementation deliberately starts with the two
least-ambiguous eligibility forms—structural Client/Server siblings and unique
technical prefixes—before considering Phase-1 aliases:

1. Form a recovery question only for a sentence containing either a role word
   locally anchored to one structural sibling family or a unique token-prefix
   shorthand for exactly one component (`WebRTC` for `WebRTC-SFU`).
2. Eligibility additionally requires a nearby earlier sentence explicitly naming
   the exact candidate component. A prefix is eligible only if it maps to exactly
   one runtime component.
3. The resolver must return an exact referring phrase, exactly one component (or
   an explicit abstention), and the anchor sentence that establishes the referent.
   It does **not** read the whole catalog and does not propose arbitrary links.
4. A dedicated anchored-reference validator then checks that exact phrase against
   the supplied anchor. It explicitly rejects hardware/generic uses, another
   same-prefix component, package paths, and diagram captions. Merge only those
   approved additions with the unchanged S21 floor.

The critical invariant is:

```text
S24 additions = anchored eligibility AND resolver decision AND anchored-reference validator
```

This is simpler than S23: one bounded resolver, one bounded validator, no separate
general blocks proposer, no action router, no second entity validator policy, and
no change to S21's floor.

## Generic, runtime-only eligibility

Eligibility is derived from the runtime model/document, never benchmark names:

- **Alias anchor:** an alias comes from Phase 1 and its component is established in
  a nearby sentence.
- **Sibling anchor:** component names form a structural family after removing one
  qualifier token (for example, a shared `HTML5` base with `Client`/`Server`);
  the resolver receives only that family and local context.
- **Unique prefix:** a standalone source token is a token-boundary prefix of one
  and only one component name, and the sentence makes an architectural claim.

The resolver may abstain. Ambiguous generic vocabulary with no local anchor is out
of scope for S24. This prevents the `frontends/backends` and generic-role flood
seen in S23 from becoming a whole-catalog routing task.

## Why this is the best next experiment

The existing sibling-disambiguation probe reports BBB extraction recall
47/62 → 55/62 with no loss in the sibling-family gold population. The error
analysis identifies the same client/server/WebRTC family as the dominant
high-confidence recall gap. S23 showed that recovery can be real, but its broad
proposer/router created 22 router FPs in teammates+BBB. S24 keeps the recovery
mechanism and removes the broad candidate source.

This is a hypothesis, not an expected score. The existing S21 two-pass entity gate
is deliberately not weakened. S24 uses a separate validator because anchored
shorthand is neither a direct entity match nor necessarily a pronoun-style coreference.

## First live result: negative (N=1)

The first implementation was evaluated on all five projects with OpenAI GPT-5.4,
enforced Flex tier, and explicit `reasoning_effort=none`. The independent
`mini-src` macro score was P 97.31%, R 90.55%, F1 93.73% (170 TP, 7 FP, 25 FN;
pooled F2 88.82%). See
[`results/s24_gpt54_openai_flex_noreasoning_20260723/RESULTS.md`](../results/s24_gpt54_openai_flex_noreasoning_20260723/RESULTS.md)
for per-project scores and request provenance.

Crucially, the marginal S24 result was **zero accepted additions**. The anchored
resolver approved some candidates, but the inherited S21 coreference gate rejected
every one. Since S21 was freshly rerun and LLM outputs are stochastic, its changed
floor cannot be read as an S24 score delta. The valid conclusion is narrower: the
current strict coreference gate is not an adequate acceptance test for these direct
anchored role/prefix references. The next S24 run replaces it with a dedicated
anchored-reference validator and narrows eligibility to explicit local anchors.

## Dedicated-validator result: promising marginal evidence (N=1)

The corrected implementation evaluated 40 eligible candidates, of which 6 were
resolver-approved and 3 passed the dedicated validator. All three additions were
gold links (BBB `Apps` S52, BBB `FreeSWITCH` S66, and teammates `Logic` S88), with
zero marginal FP. This is the signal the inherited coreference gate had suppressed.
The full rerun's macro score is not a meaningful delta because its independently
sampled S21 floor, especially BBB, was much lower. The complete evidence is in
[`results/s24_anchored_validator_gpt54_openai_flex_noreasoning_20260723/RESULTS.md`](../results/s24_anchored_validator_gpt54_openai_flex_noreasoning_20260723/RESULTS.md).

The next required step is two more identically configured runs: promotion requires
at least one clean TP in two of three runs and no more than one marginal FP per run.

## Evaluation sequence

1. **Deterministic eligibility audit.** On the frozen benchmark, report eligible
   sentences, components, and gold coverage before any LLM decision. Confirm that
   no generic-only `logic`/`storage`/`datastore` sentence is eligible.
2. **Resolver/gate trace.** Run S21 and S24 side-by-side with a fixed model and
   preserve every `(phrase, anchor, component, resolver, coref-gate)` record.
   Score marginal additions independently of the re-run S21 floor.
3. **N=3 all-five-project evaluation.** Compare S21 and S24 on macro P/R/F1/F2,
   total and source-level FP, recall additions, calls, latency, and failure rate.

Promotion criteria: S24 must add at least one clean TP in two of three runs, have
no more than one marginal FP per run, and not reduce the S21 floor. Otherwise keep
S21 and retain S24 as a negative result.

## Non-goals

- no S21 prompt or gate replacement;
- no broad all-sentence / all-component proposer;
- no global threshold or benchmark-specific word list;
- no `verify1p_all`-style removal of P2;
- no diagram-caption recovery in the first iteration.
