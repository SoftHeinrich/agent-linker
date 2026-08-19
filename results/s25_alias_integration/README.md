# Is the alias module the extractor in disguise? — 2026-08-11

Both modules are the same model reading the same document against the same
component list. The alias pass asks "what surface forms name component X"; the
extractor asks "which sentences reference X" and already reports the surface it
matched. So: can one replace or absorb the other?

Audit: `approach/pilot/alias_integration_audit.py` (no LLM call, off the promoted
run's checkpoints and traces). Arms: `approach/pilot/alias_integration_pilots.py`
(five runs per side, all five projects, permutation-tested, TP/FP/F1/F2).

## Answer

**Structurally yes, empirically pointless.** The two are the same question and
merging them is *exactly* neutral — but every simplification the equivalence
suggests fails end-to-end, and the ordering between them is forced.

## What the audit established

| Question | Finding |
|---|---|
Is the table load-bearing? | **Yes.** 29 full-name links across five projects are admitted only via an alias, **23 of them gold**; 22 coreference antecedents pass only via an alias (14 gold); 20 partial-name candidates are suppressed by one |
Can the table be projected from the extractor's `matched_text`? | **No.** A derived table recovers **41%** (12/29) of the discovered aliases and adds **28** spurious surfaces — `It`, `it`, `client`, `This component`, `re-encoded`, `GAE server`, `components of BigBlueButton`. `matched_text` is a *span*, not a *name* |
Who reads the table, and when? | Six consumers. **Exactly one runs before extraction** (the `KNOWN ALIASES` line); the other five — the contract filter, the mention-type classifier, the partial-name suppressor, the identity anchors, the antecedent gate — all run after it and could read a table the extractor produced |

That last row is what makes a merge structurally possible at all.

## What the arms measured

| Arm | TP | FP | F1 | F2 | Verdict |
|---|---|---|---|---|---|
Remove `KNOWN ALIASES` from the extraction prompt | **−5.2 (p=0.02)** | −4.0 (p=0.01) | −0.7 (p=0.27) | **−2.0 (p=0.02)** | the one pre-extraction consumer is load-bearing; **the ordering is forced** |
Remove the alias judge | +4.6 (p=0.04) | +1.2 (p=0.33) | +1.1 (p=0.09) | **+1.9 (p=0.04)** | looked like a win on the stage, **reverted end-to-end** (below) |
**Fold alias discovery into extraction** — one prompt per batch returning references *and* aliases, table accumulated across batches and fed forward | +0.2 (p=1.00) | +0.8 (p=0.38) | −0.1 (p=0.85) | **±0.0 (p=0.98)** | **exactly neutral**: saves a stage, loses the document-wide view, and the two cancel |

## Why the merge is neutral rather than better

The unified extractor builds a *larger and noisier* table than the document-wide
pass, because a batch of 50 sentences cannot see a definition that appears
elsewhere and there is no judge over the result. Its tables contain real aliases
the separate pass also found (`bbb-web`, `KMS`, `akka-apps`, `fsels`,
`Logic API`) alongside noise the separate pass rejects (`client`, `core`,
`other layers`, `Web browser`, `conversion fallback`, `Storage level`). The saved
stage and the lost precision cancel to within noise.

## The judge removal: fifth of the same trap

| | with the judge | without |
|---|---|---|
| TP | 180.8 (six-run reference band) | 179.0 |
| FP | 4.8 | **8.7** |
| macro F1 | 96.42 ± 0.42 | **94.57** |
| macro F2 | 95.38 ± 0.58 | 94.30 |

An unjudged table is larger, so the full-name stage admits more candidates and
looks better *on that stage*. Composed, the extra aliases admit false positives
that the earlier-wins union locks in and that never reach the two stricter
linkers. Reverted.

**This is now the fourth independent change** — the contract filter, the mention
classifier restructure, the bundle de-duplication, and the alias judge — where a
stage-level arm pointed one way and the composed pipeline the other, always on
precision. In a cascade whose stages subtract from one another, single-stage
ablation is not a valid estimator of the composed effect. Treat it as a screening
tool only.

## What to say in the paper

The knowledge module and the extractor ask the same question and are kept
separate for one measured reason: the extractor needs the table before it runs
(−5.2 TP, −2.0 F2 without it), and a document-wide pass with a judge builds a
better table than per-batch discovery can. State the equivalence as a design-space
result — the separation is *not* structurally necessary, and merging costs
nothing — rather than as a claim that two stages are required.
