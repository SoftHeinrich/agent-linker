# ALinker/Core: elegance that survives the ledger

## The result this document exists to record

The obvious beautiful architecture — read the document once, propose everything,
resolve each sentence in one contrastive call, three stages and one contract —
**is refuted by experiments this branch already ran.** It was designed, built
(`alinker_core.py`), and then checked against `approach/CLAUDE.md`. Four of its
five collapses have already been priced, and they lose:

| collapse in the first Core design | already run as | measured cost |
| --- | --- | --- |
| READ merges alias proposal and alias judging into one call | s26 | F1 94.27 vs 96.4 |
| — the gentler version, judging kept separate but folded into reading | s60 | TP −5.0, FP +11.2, **F1 −2.7** |
| RESOLVE replaces the two focused full-name judging calls with one | s36 | **F1 −0.7** (p=0.01), FP +3.5 |
| RESOLVE shows every candidate its target, deleting target-blindness | s25 design law | **−5.5 gold** |
| the alias table no longer suppresses partial-name candidates | s46 | **F1 −1.5** (p=0.00), FP +6.5 |

And the standing finding that covers all of them:

> **Every consolidation of two LLM decisions into one call raises recall and
> lowers precision. Twelve variants, five instances, no exception.**

One collapse in the design is *not* refuted: PROPOSE and RESOLVE stay separate
calls, so Core never lets a proposer approve its own list (s31's failure). And
one element is genuinely untested by any round: **contrastive resolution**, where
competing components for one sentence are ruled on together. Every merge the
ledger refuted was of the form "fold judging into extraction". None of them asked
one judge about two candidates side by side.

Estimated cumulative cost of the refuted collapses is 4–6 macro F1. No
equivalence margin worth declaring covers that. **The first design is withdrawn.**

## What that means for "beautiful"

The head's complexity is not accidental. It is twelve refuted merges deep, and
the ledger is its justification. So the elegance target has to move: not *fewer
decisions*, but *less code and one contract for the same decisions*.

That distinction is the whole design now.

- **Decision topology is load-bearing and stays fixed at seven decision points.**
  Alias proposal, alias judging, reference extraction, two-pass full-name
  judging, target-blind denotation, coreference resolution, coreference judging.
  Each is separately measured; each merge is separately refused.
- **Code surface is accidental and collapses.** Seven decisions are currently
  expressed in 1,359 lines, 61 functions, three evidence formats, two enums and
  21 prompt literals. None of that arrangement was measured. It is the part no
  experiment defends, and therefore the part free to change.

The claim becomes provable rather than probable: a refactor that renders
byte-identical prompts and produces identical candidate sets is equivalent *by
construction*, and can be certified with **zero API budget** at level 1 of this
branch's own measurement policy.

## The corrected architecture

One contract, seven declared steps, deterministic layer unchanged.

```
Claim(sentence, component, span, anchor, origin)   -- the single data type

  propose_aliases   -> alias candidates          (LLM)
  judge_aliases     -> alias table               (LLM, separate: s26/s60)
  extract_refs      -> Claims                    (LLM)
  judge_named_p1    -> verdicts  \  two focused calls, not one: s36
  judge_named_p2    -> verdicts  /
  classify_denote   -> verdicts                  (LLM, target withheld: design law)
  resolve_coref     -> Claims with anchors       (LLM)
  judge_coref       -> verdicts                  (LLM, strict default preserved)
```

`Claim` replaces `EvidenceBundle` and the three per-linker evidence formats. One
well-formedness predicate replaces the substring check and the structural
antecedent constraint, stated once instead of once per linker. The name relation
is already one relation at four settings (s65 proved that an identity over all
3,697 pairs), so it needs no further work — it needs only to stop being spread
across six methods and two enums.

What must **not** change, each because a round says so:

- the alias judge stays a separate call (s26, s60);
- the full-name judge stays two focused calls asking different questions — "independence
  comes from asking a different question, not from resampling the same one" (s36, s38);
- the denotation judge is not shown the component (design law, −5.5 gold);
- the alias table keeps its second job, suppressing partial-name candidates (s46);
- each judge keeps its asymmetric default — lenient where it approves, strict
  where it rejects. The typed round (s86) showed that closing a rubric into a
  named verdict set deletes the default, and the default is what the asymmetry
  was carrying: −16.6 gold on terra, FP +34.0 on luna.

That last point is the sharpest constraint on any "elegant" rewrite: **typed
verdict enums are already refuted here.** A declarative rewrite that renders
every judge as `Literal["approve","reject"]` reintroduces a measured regression.

## The one addition worth testing

Contrastive resolution, proposed as a **new decision point, not a merge**.

The motivation is measured and is not in the ledger: 68% of the head's false
positives put a sentence on the wrong one of two components that share a name
word (`HTML5 Client`/`HTML5 Server`, `Redis DB`/`Redis PubSub`), and a judge
shown one candidate at a time structurally cannot see the other. An oracle that
resolved those competitions perfectly, over the candidates the head already
produces, moves macro F1 from 0.933 to **0.957**.

Shape that respects the ledger: after the existing judges have ruled, take only
the sentences where two or more *surviving* claims name components that share a
name word or that the reader flagged confusable, and ask one additional call
which the sentence is about. It adds a question rather than folding one away, it
runs on a small population, and it cannot starve a later stage because nothing
comes after it. By `pilot/composition_check.py`'s rule it is a stage arm that
*is* the pipeline answer, so it can be priced on recorded checkpoints before any
E2E is bought.

Risk to state: this shows the judge both targets, which is the shape the design
law prices at −5.5 gold for the denotation step. The difference is that
denotation asks *what does this expression denote* — a question the target
biases — while this asks *which of these two* — a question that requires both.
That difference is a hypothesis, and it is what the arm would test.

## Simplicity, measured

Countable from the repository, both systems on equal footing. These are the
columns reviewers find persuasive; they are all code-surface, not topology.

| | head (`s_linker88` + `helper_v3`) | Core (refactor target) |
| --- | ---: | ---: |
| lines of code, live path | 1,359 | 439 |
| functions | 61 | 17 |
| prompt literals (excluding docstrings) | 21 | 6 |
| authored prompt characters | 5,047 | 2,190 |
| evidence formats | 3 | 1 |
| surface-form enums | 2 | 0 |
| tunable constants | 16 | 6 |
| **LLM decision points** | **7** | **7 — unchanged, deliberately** |

The last row is the point. Every other row is the contribution; that row is the
constraint that makes the contribution safe to claim.

Also worth reporting separately, because it is a *superiority* claim rather than
an equivalence one: the deterministic scan alone recovers **89.7%** of gold, and
in the collapsed prototype 30% of claims settled with no model call at all. If
the refactor preserves the topology, call count is unchanged; if the contrastive
arm is adopted, it rises by roughly one call per project.

## How to certify it, at zero API cost

This branch's measurement policy escalates and stops at the first level that
decides. A pure refactor decides at level 1.

1. **Byte-identical prompt rendering.** Assert that each of the seven re-declared
   prompt builders renders character-for-character identically to the head's, on
   every case in the recorded checkpoints. The policy already requires this of
   stage pilots; here it is the whole proof.
2. **Identical candidate sets.** Replay both deterministic layers over all 3,697
   (name, sentence) pairs and assert equality, as `pilot/rule_audit.py --only A2`
   did for the four-rule unification (s65: 49/49 invariant checks, no E2E owed).
3. If 1 and 2 both hold, the refactor is an identity and **no runs are owed.**
   s65 and s85 both closed this way. That is a far stronger result than an
   equivalence test — it is equality, not indistinguishability.

Only the contrastive arm, being a real behavioural change, needs measurement, and
then only at level 2 (stage pilot on recorded inputs) followed by level 3
(`composition_check.py`) before any E2E is considered.

## If an E2E is ever owed

Should the contrastive arm reach E2E, the protocol is the branch's own, with the
equivalence machinery layered on:

1. The unit of independence is the **project**, n = 5 — not 195 links. Links in
   one document share its vocabulary, its author and one run's idiosyncrasies.
2. **Six paired runs is the bar**, per this branch's own standing finding: s44
   read F1 −0.0 (p=1.00) over three runs and −0.9 (p=0.05) over six.
3. Both arms in the same invocation. Never compare across invocation sets —
   s49's FP mean read 10.7, 11.7, 12.5, 14.5 and 16.8 across five sets in one day.
4. Pair by project, not by run index; declare a margin before running, justified
   from the measured noise floor (the head's recorded runs span 0.928–0.937, so
   ±0.03 macro F1 is a defensible bound).
5. Report TOST against that margin *and* a Bayesian ROPE/HDI, plus
   leave-one-project-out sensitivity. At n = 5, "consistent with equivalence but
   underpowered to confirm" is a legitimate and more credible outcome than a
   confident TOST pass.
6. Run every arm on **both models**. The typed and compaction rounds each refused
   arms on the second model that the first accepted; a cut that holds on the
   stricter model says nothing about the laxer one.

## Status

- `approach/src/llm_sad_sam/linkers/experimental/alinker_core.py` — the collapsed
  prototype. It runs end to end, and it is **retained as the priced alternative,
  not as a candidate head**, in the same spirit as `s_linker76` and `s_linker79`.
  Its measured floor with no judging at all is macro F1 0.706 at recall 0.897.
- Registered as variant `core` in `approach/run_ablation.py`.
- The refactor described above, and the contrastive arm, are not yet built.

## The finding worth keeping

The ledger's twelve refuted merges are all of one shape: folding a judgment into
the call that produced the thing being judged. That is a real law and this round
confirms it a thirteenth time, on paper, before spending anything.

But it is a law about **decision topology**, and it was silently being read as a
law about code. It is not. Seven decisions do not require 1,359 lines, three
evidence formats or two enums, and nothing in the ledger defends that
arrangement. The elegance available here is real; it is just one level down from
where it looked.
