# The concept round — one principle instead of five leftovers

After the general round, `s_linker74` has no rule a reviewer can call corpus-fitted. What
it still has are **concepts** that look inelegant even though every piece is measured:

| | wart |
|---|---|
| C1 | a regex classifier computes a five-valued feature for the LLM about text the LLM is already reading (`_classify_mention_typed` → `mention=...` in the evidence line) |
| C2 | `unique_owner` survives only because the denotation judge is deliberately kept ignorant of the target |
| C3 | the full-name judge asks two hand-chosen facets and ANDs them |
| C4 | the alias table has two opposite jobs — it admits full-name candidates and suppresses partial-name ones |
| C5 | stage order does epistemic work that is never stated as a claim (`_union` is earlier-wins) |

C1 was the one to attack: it is the last hand-built feature in the workflow, and three of
its five values are already restated in English in the same prompt (`STRICTER_CLAUSE` is
the case distinction, `QUALIFIED_CLAUSE` the inside-an-identifier one). The fold law from
the previous rounds — *a gate folds into a judge's prompt exactly when that judge is shown
what the gate reads* — predicts it should fold: four of the five values are computed from
the sentence alone, which the judge has in front of it.

## The arm (`pilot/concept_pilots.py --pilot conceptlabel`, n = 3, checkpoint replay)

| arm | TP | FP |
|---|---|---|
| s74 — regex computes the label | 146.0 | 4.0 |
| label gone entirely | **−10.7** (p = 0.10) | −0.3 (p = 1.00) |
| judge states the realization itself | **−6.7** (p = 0.10) | +1.7 (p = 0.10) |

**The prediction was wrong, and the way it was wrong is the result.** The label is worth
10.7 true positives. Asking the judge to derive the same four facts itself — from the same
sentence, in the same call, before its verdict — recovers only 4.0 of them and buys 1.7
false positives.

## Why: information the judge *can* derive is not information it will derive impartially

When the deterministic layer says `mention=proper case, standalone`, the judge receives a
fact from a disinterested party and has to reconcile its verdict with it. When the judge
derives that fact itself, it derives it *in service of* the verdict it is already forming.
That is the same effect `s_linker25` measured from the other direction: showing the
denotation judge the target component made it confirm identity rather than test it, at a
cost of 5.5 gold links per run.

So the fold law needs its second clause, and with it the six fold results stop being six
results and become one:

> **Facts stay in code. Weighings go in the prompt.**
>
> The deterministic layer supplies *facts about a case*; the LLM supplies *judgment about
> the case*. A clause that tells the judge **how to weigh** what it sees folds. A
> statement of **what is true of the case** does not — not because the judge cannot see
> it, but because the judge is not disinterested about it.

Every result on the branch obeys it:

| moved into the prompt | kind | outcome |
|---|---|---|
| `skip_qualified` | weighing | folded — TP −0.4 (p = 0.44) |
| `skip_stricter` | weighing | folded — **TP +4.0, FP ±0.0** |
| the mention label, self-reported | fact | **−6.7 TP** |
| the mention label, removed | fact | **−10.7 TP** |
| `unique_owner` (`foldowner`) | fact | **−8.4 TP** |
| the target, revealed to the denotation judge (s25) | fact | **−5.5 gold** |

Two folds, four refusals, and the split falls exactly on fact-versus-weighing with no
exceptions.

## What this does to the other four warts

- **C2 is not a separate workaround.** Target-blindness and `unique_owner` are the same
  principle: ownership is a fact, so it stays in code, and the judge is kept blind to the
  target precisely because a self-reported fact about the target is contaminated by the
  verdict. It stops being "a gate we could not remove" and becomes "a fact the fact-finder
  supplies".
- **C4 dissolves.** The alias table is a *fact* — the names this document establishes —
  and a fact is a fact for every consumer. Two stages using it in opposite directions is
  not a dual role, it is two judgments resting on one finding.
- **C5 becomes statable.** Earlier-wins is the composition rule that pairs with "the
  looser the form a linker scans, the stricter the judge behind it": a linker that fires
  only on tight evidence holds the strongest fact, so its admission is final.
- **C3 stands on the existing law** from twelve variants — every consolidation of two LLM
  decisions into one call raises recall and lowers precision; splitting buys precision.

## Outcome

**No code changed.** All three arms were rejected, so `s_linker74` stands as measured
(macro F1 95.60, F2 96.66, n = 3). The deliverable is the principle: what was a list of
five leftovers is now one design statement that predicts every fold result on the branch —
including the two it would have got wrong before this round.
