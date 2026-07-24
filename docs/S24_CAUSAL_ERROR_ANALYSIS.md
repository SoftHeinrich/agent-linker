# S24 causal error analysis

## Route diversity

Both all-project replacement replays selected:

```text
entity_pipeline → coreference_pipeline → coverage_audit → finalize
```

This is route convergence, not identical project state. Documents, catalogs,
aliases, ambiguity, candidates, validator outcomes, controller assessments, and
accepted links differed by project. Those differences did not change the chosen
capability sequence.

The cause is now isolated:

1. Entity and coreference are described as complementary evidence modes.
2. Coverage audit is a catch-all over both identity and contextual reference.
3. Before calling audit, the controller cannot prove that audit has no useful
   candidate.
4. Tool calls have no represented downside or capability obligation that can be
   discharged without executing them.

Given that contract, “run every remaining complementary tool” is a rational
policy. The controller is dynamically narrating state, but the action space
induces a fixed workflow.

## Feedback-parity experiment

The first fixed-phase replay exposed only accepted entity/coreference links to
the controller. Production exposes candidates and validator outcomes too.

Passing the raw production-equivalent feedback failed on the third controller
decision with `Argument list too long`: repeated source sentences and detailed
decisions were recursively embedded in history.

The corrected representation retains full outputs as evidence while passing the
controller normalized accepted/rejected `(sentence, component)` facts. The
replay completed, but still selected the same route on all projects. Therefore:

- compact feedback is required for scale and truthful state;
- missing feedback was not the cause of route convergence;
- the controller/tool contract, especially catch-all audit, is the cause.

The feedback-parity replay produced 175 TP / 12 FP / 20 FN, macro F2 93.23% and
pooled F2 90.49%. Its different audit links are a fresh-model sampling effect,
not evidence that feedback changed the route.

## Residual false negatives

The 20 remaining gold links decompose by the furthest phase reached:

| Cause boundary | Count | Interpretation |
| --- | ---: | --- |
| Never proposed by entity, coreference, or audit | 15 | Missing evidence mode |
| Proposed but rejected | 5 | Validator semantics |

The 15 proposal failures fall into four general evidence modes:

- **Relational participant under-linking.** A component is an endpoint, server,
  client, callee, datastore, or protocol-mediated participant rather than the
  grammatical claim owner.
- **Multi-target incompleteness.** A sentence relates several architectural
  components but extraction emits only the subject or most salient target.
- **Technology/role metonymy.** A runtime technology or role phrase denotes the
  modeled component in context.
- **Structural discourse.** A heading, caption, or section-continuation sentence
  carries component evidence without a conventional subject-predicate claim.

The five validator failures show two mismatches:

- negated or contrastive statements are rejected even though they specify an
  architectural boundary;
- explicit relational participation is rejected when the component is not the
  sentence's primary claim owner.

## Measured false positives

The feedback-parity result contains seven audit additions absent from gold:

- one database participant in a query relation;
- three package/component or component-pronoun claims;
- one extracted integration component;
- one named conversion process;
- one approved architectural alias.

Several are semantically defensible and resemble nearby gold-positive forms.
They should remain measured false positives—evaluation must respect the supplied
gold—but they indicate annotation incompleteness or inconsistency rather than
one uniform semantic failure.

The true audit error modes are:

- package/API ownership can be confused with a modeled component claim;
- an approved alias can be treated as sufficient without checking what fact the
  whole sentence asserts;
- a relational mention can be valid semantically but outside the gold's local
  annotation policy.

## Ideal controller preparation

Gold is used only offline to discover the right generic decomposition. An ideal
runtime controller should receive:

1. a document/component profile describing reference modes, ambiguity, aliases,
   section structure, and multi-component relations;
2. a coverage ledger keyed by `(sentence, component)` with proposed, accepted,
   rejected, and unresolved states;
3. non-overlapping tool capabilities expressed as semantic obligations;
4. compact feedback that says which obligations a tool discharged and which
   remain unresolved;
5. authority only over workflow, never over link acceptance.

The reasoning pattern should be:

```text
profile evidence modes
→ identify unmet semantic obligations
→ call the tool owning one obligation
→ update the coverage ledger from tool feedback
→ finalize when no observable obligation remains
```

This is more meaningful than asking whether a broad catch-all tool might still
find something; that question almost always favors another call.

## Participation hypothesis

The highest-leverage correction is inside candidate generation:

- replace narrow **claim ownership** with **architectural participation**;
- enumerate every participating catalog component in a sentence;
- recognize negated/contrastive boundaries and relational objects;
- allow descriptive headings/captions when they denote an architectural flow;
- keep exact catalog membership and exact-quote grounding;
- retain the independent existing validator.

This is a generic semantic contract derived from residual classes. It introduces
no benchmark vocabulary, score threshold, context window, prefix rule, or
project-specific gate.

## Participation result

The all-project replay produced:

| System | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S21 | 168 | 5 | 27 | 93.34% | 90.83% | 91.30% | 88.14% |
| Participation audit | 181 | 12 | 14 | 94.50% | 94.75% | 93.30% | 93.01% |

The hypothesis recovered the predicted causal classes:

- all three residual MediaStore links, including both participants in one
  contrastive sentence;
- two datastore-participation links previously missed in TeamMates;
- a negated FreeSWITCH integration boundary;
- an HTML5 client relation;
- both Presentation Conversion heading/caption links.

The remaining 14 false negatives are now concentrated in relation targets whose
surface form is a generic server/client/protocol role, plus four Logic references
and one contrastive WebUI validation failure.

Measured precision fell because the broader contract also accepted seven
non-gold additions beyond S21. Macro and pooled F1 nevertheless improved over
S21, and the recall-weighted F2 gains were substantially larger.

The route remained unchanged across projects. Candidate semantics and F2
performance are validated; a genuinely project-specific action policy still
requires a non-overlapping capability/obligation design.
