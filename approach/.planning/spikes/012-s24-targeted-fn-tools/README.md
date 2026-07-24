---
spike: 012
name: s24-targeted-fn-tools
type: comparison
validates: "Given categorized residual FNs, when the controller gains non-overlapping evidence-completion tools, then fresh S24 improves macro and pooled F2 without regressing either S24 F1 aggregate."
verdict: VALIDATED
related: [005-upstream-candidate-gap, 008-s24-semantic-appeal, 011-s24-f1-constrained-routing]
tags: [s24, false-negative, dynamic-workflow, aliases, identifiers, no-magic]
---

# Spike 012: S24 targeted FN tools

## What This Validates

Residual false negatives must first be separated into clear recoveries,
context-dependent cases, and genuinely debatable benchmark boundaries. A new
tool is eligible only if its evidence mode does not retry a prior rejection or
overlap an existing capability.

The evaluation gate is stricter than the S21 comparison:

- macro and pooled F1 must not regress from the fresh S24 pilot baseline;
- macro and pooled F2 must improve over that same baseline;
- runtime code contains no benchmark vocabulary, score thresholds, fixed
  candidate counts, or project branches.

## Initial FN Classification

The fresh spike 011 S24 result has 22 FNs:

| Class | Count | Interpretation |
| --- | ---: | --- |
| Existing entity candidate rejected | 5 | Clear in gold, but broad appeal is unsafe |
| Existing role candidate rejected | 3 | Context-dependent server references |
| Approved alias occurrence never proposed | 4 | Clear recoverable coverage gap |
| Standalone code-shaped identity never proposed | 4 | Mostly clear, distinct evidence mode |
| Generic participant/reference never proposed | 6 | Debatable or high-context boundary |

Restoring all rejected entity candidates would yield only 5 TP / 17 FP.
Restoring all rejected role candidates would yield 3 TP / 7 FP. Spike 008
already invalidated broad semantic appeal, so rejected candidates are not
retried.

### Clear recoveries

- exact approved aliases omitted by extraction: one database reference and
  three datastore references;
- standalone code-shaped identities such as a CamelCase alternative name or a
  hyphenated application identifier;
- distinctive explicit names rejected by the validator are genuine misses,
  but have no clean new ownership contract and are therefore not targeted.

### Context-dependent

- generic `server` or `client` after an explicit component introduction;
- plural participant nouns;
- protocol names that may denote either a technology or a component.

### Genuinely debatable

- protocol use versus the component implementing that protocol;
- deployment/server language versus a specific server component;
- code identifiers that may identify an implementation namespace rather than
  the architectural component itself.

Gold defines scoring, but these cases should not drive runtime exceptions.

## Candidate Tools

1. **Approved-alias coverage** enumerates exact standalone occurrences of
   runtime-approved aliases that the fresh entity tool never proposed. One
   project-context review resolves occurrence meaning.
2. **Identifier identity** owns standalone CamelCase or hyphenated identifiers
   excluded from prose tools. Dotted qualified paths remain excluded.
3. **Section referent propagation** is deferred unless the first two fail. It
   overlaps coreference and role resolution and is the highest-risk option.

## Investigation Trail

Every inference run used the OpenAI API with `gpt-5.6-terra` and reasoning
effort `none`. No cached model response was used.

1. **Alias coverage, v1:** passed one fresh run (1 TP / 0 FP), but did not
   survive repetition. In the combined v3 pilot it added 0 TP / 4 FP by
   interpreting ordinary occurrences as architectural aliases. Rejected as an
   unstable ownership boundary.
2. **Fuzzy code-shaped identifier, v2:** added 1 TP / 2 FP. Hyphenated prose
   such as `re-encoding` and `order-based` leaked into the candidate set.
   Rejected.
3. **Exact full-token identifier, v3/v4:** require a standalone CamelCase or
   hyphenated expression whose complete normalized token sequence equals one
   runtime catalog name. Exclude canonical spellings, approved aliases,
   qualified dotted paths, and links already handled by another tool. The
   isolated v4 pilot added 2 TP / 0 FP and improved macro/pooled F1 by
   0.38/0.56 pp and F2 by 0.54/0.84 pp.
4. **Production checkpoint v1:** found an orchestration bug: the controller
   could finalize while an evidence-backed tool remained. The fix is a general
   evidence-scope completion invariant. Project profiling decides which tools
   exist; the controller orders all applicable tools and cannot silently drop
   one. This is neither a count gate nor a project rule.
5. **Production checkpoint v2:** on the only project exposing exact identifier
   evidence, the production tool added 2 TP / 0 FP. The route was
   `entity -> coreference -> role -> identifier -> finalize`.
6. **Fresh paired E2E:** all five projects were rerun from scratch against
   S21. Both F1 non-regression gates and both F2 improvement gates passed.

## Final Workflows

The profile, rather than a fixed phase list, determines the available
capabilities. The controller then orders the evidence-backed set:

| Project | Controller workflow |
| --- | --- |
| MediaStore | entity -> coreference -> finalize |
| TeaStore | entity -> coreference -> finalize |
| TEAMMATES | entity -> role -> coreference -> finalize |
| BigBlueButton | entity -> coreference -> role -> identifier -> finalize |
| JabRef | entity -> coreference -> finalize |

There are three distinct paths. Only BigBlueButton exposed catalog-identifier
evidence, so only that project invoked the new tool. TEAMMATES placed role
before coreference, while BigBlueButton placed coreference before role; order
is controller-selected from unresolved evidence, not prescribed per project.

## Residual FN Causal Analysis

The final fresh S24 run has 17 FNs:

| Class | Count | Examples and disposition |
| --- | ---: | --- |
| Explicit entity candidate rejected | 5 | One UI spelling and four explicit UI/logic names. Clear gold misses, but retrying the same evidence would violate tool ownership; broad restoration was previously 5 TP / 17 FP. |
| Role candidate rejected | 3 | Generic `server` mapped to a particular server component. Some are plausible, but the target depends on local deployment context; broad restoration was 3 TP / 7 FP. |
| Uncovered alias or divergent identifier | 3 | A database alias, a datastore reference, and `AudioAccess` for `MediaAccess`. The first two are clear gold links but alias review was unstable; the last changes a semantic token and deliberately falls outside exact identity. |
| Generic participant/reference | 4 | `clients`, `server`, and `client`. These are contextual architectural references, not stable identity evidence. |
| Protocol/component boundary | 2 | `WebRTC` scored as the `WebRTC-SFU` component. Genuine debatable boundary: the sentence can name a protocol without naming its implementation component. |

The remaining FNs therefore do not justify another target tool yet. A future
capability would need independent evidence for discourse referents or
protocol-to-implementation realization; merely appealing rejected candidates
would duplicate existing tools and predictably lose precision.

## Results

Fresh paired aggregate:

| Variant | TP / FP / FN | Macro F1 | Pooled F1 | Macro F2 | Pooled F2 |
| --- | --- | ---: | ---: | ---: | ---: |
| S21 | 176 / 24 / 19 | 91.44% | 89.11% | 92.31% | 89.80% |
| S24 | 178 / 7 / 17 | 95.36% | 93.68% | 94.12% | 92.23% |
| Delta | +2 / -17 / -2 | +3.92 pp | +4.57 pp | +1.81 pp | +2.43 pp |

The exact identifier tool accounts for two true-positive additions and no
tool-attributed false positives. The design passes because it adds one
non-overlapping evidence capability and one general completion invariant. It
contains no benchmark vocabulary, score/count threshold, fixed route, or
project-specific branch.
