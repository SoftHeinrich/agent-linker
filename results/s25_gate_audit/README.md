# s25 code-driven gates: inventory and LLM-handover pilots — 2026-08-10

Question behind this round: how much of the workflow is decided by hand-written
code rather than by the LLM, and can any of it be handed over without losing F2?

`approach/pilot/gate_audit.py` inventories every code-driven decision off the
promoted run's checkpoints and per-call traces (no LLM call).
`approach/pilot/gate_pilots.py` runs the handover arms — five runs per side, all
five projects, permutation-tested, scored on TP, FP, F1 **and F2**.

## The three kinds of code-driven decision

Not all code is the same kind of liability, and the paper should not defend them
the same way.

| Kind | What it is | Reviewer exposure |
|---|---|---|
| **Sanity** | the model named a sentence or component that does not exist, or quoted a span absent from the sentence | none — 3 of 228 extractor references were caught this way; nobody asks about it |
| **Grounding** | the model's own quote is checked against the text it claims to quote | one sentence to state; verifies the model against itself, not against a domain rule |
| **Heuristic** | a hand-written linguistic rule that decides admissibility | this is the exposure: it makes the approach read as rule-driven |

Three heuristics were priced. Only one could be handed over.

## Inventory (`audit.json`)

| Gate | Kind | Measured |
|---|---|---|
| `_keep_stated_names` — stated-name contract filter | heuristic | rejects **22 of 228** extractor proposals, **9 of them gold** |
| `_name_word_candidates` — partial-name proposer | heuristic | 57 proposals, 17 gold-reachable, 11 accepted and all gold; on teammates 28 proposals contain 1 gold pair |
| `_antecedent_states_name` — coreference antecedent test | heuristic | blocks **20 of 133** reported resolutions, **7 of them gold** — the largest code-driven rejection |
| identity-judge evidence conditions (claim substring, anchor membership, non-empty alternative) | grounding | voided **0** keeps across all five projects |
| denotation step ("participant" vs "associated") | LLM decision | rejects 38 of 57 proposals, only **1** of them gold |
| extraction sanity (unknown component / sentence / absent span) | sanity | 3 of 228 |

## Handover results

| Change | Report | TP | FP | F1 | F2 | Outcome |
|---|---|---|---|---|---|---|
| Drop the contract filter; extractor proposes, judge decides | `gate_contract_filter.json` | +2.8 (p=0.01) | +3.6 (p=0.01) | −0.0 (p=0.86) | +0.9 (p=0.01) | adopted on the stage, **reverted end-to-end** |
| Drop the antecedent gate; strict judge decides alone | `gate_antecedent.json` | −1.2 (p=0.03) | +12.0 (p=0.01) | −3.2 (p=0.01) | −1.7 (p=0.01) | rejected |
| LLM proposes partial-name references instead of the prefix rule | `gate_partial_proposer.json` | −7.0 (p=0.01) | +0.6 (p=1.00) | −6.6 (p=0.01) | −4.4 (p=0.01) | rejected |

### The contract filter: a stage arm that pointed the wrong way

On its own stage, dropping the filter is F2-positive. Composed into the pipeline
it is not, and this is the one place in this whole investigation where a
single-stage arm mispredicted the end-to-end effect. Three five-project runs
either way (`results/s25_nogate_e2e_r{1,2,3}_20260810` vs
`results/s25_simplified_e2e_r{1,2,3}_20260810`):

| | with the filter | without |
|---|---|---|
| macro F1 | **96.8** | 94.4 |
| macro F2 | **95.9** | 94.9 |
| TP | 182.3 | 182.0 |
| FP | **4.3** | 17.3 |

Recall is identical and false positives quadruple. The reason is that this stage
feeds `_unlinked`: a false positive admitted here is locked into the union by the
earlier-wins merge *and* removes the pair from the two later linkers, whose
rubrics are stricter. A stage arm cannot see either effect.

The filter is therefore kept, and the rule this episode establishes is that an
adopted arm is confirmed end-to-end before it stays. A single confirming run
after the revert lands at macro F1 96.5 / F2 95.6 / FP 5
(`results/s25_reverted_confirm_20260810`), inside the three-run band above.

The two rejected ones are now defensible rather than merely present. The
antecedent gate is worth 12 false positives. The partial-name proposer reaches 11
gold links where an LLM asked the same question directly reaches 4 — and it was a
single generic prompt, not a prompt search, so this bounds the swap rather than
settling it forever.

## What the workflow decides in code, after this round

- **one** heuristic proposer (partial-name word test), justified at 11 vs 4 gold
  links against its LLM replacement;
- **two** heuristic gates: the coreference antecedent test (F2 −1.7 if removed)
  and the stated-name contract filter (FP 4.3 → 17.3 if removed);
- one deterministic proposer for spelling variants (2 gold links extraction never
  proposed);
- one computed evidence field (mention type, worth 6.6 gold links);
- grounding checks on the partial-name judge's own quotes, plus sanity checks.

Everything else — which sentences name a component, which of those links are
real, which coreference resolutions hold, which partial references denote a
participant, which alias is real — is an LLM decision.
