# Knowledge-informed blocks proposer — which knowledge helps, and how much

Run date: 2026-07-03. Harness: `pilot/knowledge_proposer_compare.py`
(`OPENAI_SERVICE_TIER=default`, gpt-5.4 reasoning-off). Metric = gold-candidate
recall CEILING (grounded (sentence,component) pairs that are gold / all gold),
plus candidate VOLUME (precision proxy for the downstream gate) and the
noise-robust crux: recovery of gold that s21 Framing-C catches but blind blocks
misses (`s21-only`). s21's Phase-1 knowledge + Framing-C keys captured in one run.

## Question

The blind blocks proposer is a recall ceiling but NOT a superset of s21's
Framing-C entity pass (bbb loses ~2 gold Framing-C uniquely catches). Framing-C
is **alias-injected** and blocks is knowledge-blind. Does injecting s21's own
Phase-1 knowledge — global **aliases** (`term -> Component`) and/or
**ambiguous-name** cautions — close the gap? And does porting Framing-C's actual
extraction RULE, or borrowing from Artemis's prompt, add anything?

## Results (4 datasets)

| dataset | gold | aliases | ambig | s21 FC recall | blind | **alias** | ambig | both | alias recovers s21-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mediastore | 31 | 8 | 4 | 0.871 | 0.968 | **1.000** | 0.806 | 1.000 | **+1** (S20) |
| teastore | 27 | 10 | 4 | 0.778 | 1.000 | **1.000** | 0.889 | 1.000 | 0 (blind saturates) |
| teammates | 57 | 11 | 5 | 0.895 | 1.000 | 0.965 | 0.965 | 0.965 | 0 (blind saturates) |
| bigbluebutton | 62 | 14 | 0 | 0.677 | 0.774 | 0.694 | — | — | **+3** (HTML5 Server, S19-21) |

Candidate VOLUME under the ambiguous-name caution (precision proxy):

| dataset | blind cands | ambig cands | Δ |
|---|---:|---:|---|
| teammates | 160 | 107 | **−33%** |
| mediastore | 34 | 31 | −9% |
| teastore | 46 | 42 | −9% |

Framing-C RULE port (bbb, `fc` = proposer with s21's `ENTITY_EXTRACTION_RULES`):
`fc` recovered **0** of the 3 HTML5 Server links and *raised* volume 62→70 (the
"Favor inclusion" bias → more FP, not the targeted gold). `fc_alias` recovered all
3 — via the alias map, not the rule.

> **Measurement weather.** Aggregate recall is a single stochastic sample per cell
> (gpt-5.4 temp 0.1); on bbb `blind` sampled 0.774 here vs 0.742 in an earlier run,
> and the empty-`ambiguous` `ambig` cell (prompt identical to blind) sampled 0.694
> — a ±3-link noise band. So conclusions are drawn from the **noise-robust**
> signals (which specific s21-only links are recovered; candidate-volume direction),
> NOT from ranking configs on aggregate recall.

## Findings

1. **Alias injection is the fix — targeted and safe.** It recovers exactly the
   alias-mediated gold blind blocks misses (bbb +3 HTML5 Server, mediastore +1),
   reproducibly, and never drops a targeted link. Where blind already saturates
   recall (teastore/teammates) it neither helps nor meaningfully hurts. With alias,
   **blocks recall ≥ Framing-C on all four** (mediastore 1.000>0.871, teastore
   1.000>0.778, teammates 0.965>0.895, bbb ≥0.694>0.677 *and* recovers the
   uniques) → blocks+alias is a recall **superset** of Framing-C.

2. **The bbb gap was an alias-MAP gap, not a rule-strength gap.** Porting
   Framing-C's `ENTITY_EXTRACTION_RULES` verbatim (`fc`) does NOT recover the
   HTML5 Server links; only the runtime alias map (`bbb-html5`/`HTML5 server` →
   HTML5 Server, distinguished from HTML5 Client) does. "Give the proposer
   Framing-C's full strength" resolves to **give it the alias map** — which is
   knowledge, already available from Phase 1 — not the prompt wording.

3. **Ambiguous-name caution is a precision lever that costs recall — do NOT put it
   in a recall-first proposer.** It cuts candidate volume (teammates −33%) but
   drops real gold (mediastore 0.968→0.806, teastore 1.000→0.889): the "only link
   when it really means the component" instruction makes the model drop
   borderline-correct references. The downstream s21 two-pass gate + router already
   supply precision, so paying recall for precision at the proposer is the wrong
   trade here. (It could matter only for a REPLACE architecture with no router.)

4. **Artemis's prompt is largely non-transferable.** Artemis is **open-vocabulary
   NER** — it is not given the catalog, so most of its prompt (functional-suffix
   heuristics, "what counts as a component", exclude domain entities) solves the
   "is this a real component" problem that catalog **grounding already solves** for
   the proposer (part of why Artemis F1 0.836 < s21 0.936). Its transferable ideas
   — alias/abbreviation awareness, code-path exclusion, reverse-pronoun-only —
   are already present (grounding + alias injection + prev-sentence context +
   CODEPATH mode). Nothing to port.

## Best solution

**blocks proposer + Phase-1 global alias-map injection, no ambiguous caution,
keep the proposer's own reference rule (not Framing-C's, not Artemis's).**

- Aliases: the runtime knowledge s21 already computes; the single lever that turns
  blocks into a recall superset of Framing-C. GATE-06 safe (doc-derived runtime
  input s21 already consumes).
- Ambiguous caution: OFF for the augmentation proposer (net recall loss; precision
  is the gate's job). Revisit only for a router-less REPLACE variant.
- Framing-C rule / Artemis prompt: no benefit; skip.

## End-to-end (alias wiring shipped into s23_verify)

Alias injection was promoted into the shipped proposer (`build_batch_prompt` /
`propose_batch` gained an additive `aliases=` arg; no-alias path byte-unchanged)
and wired into `SLinker23._propose` (+ ctx / extract) via `_global_aliases()`,
which pulls s21's Phase-1 global aliases. E2E run (2026-07-03, default tier, all 5
datasets, s21 floor + alias-wired s23_verify in the SAME run for weather control):

| dataset | s21 floor F1 (R, FP) | s23_verify+alias F1 (R, FP) |
|---|---|---|
| mediastore | 94.9 (90.3, 0) | 94.9 (90.3, 0) |
| teastore | 96.2 (92.6, 0) | 89.3 (92.6, **4**) |
| teammates | 89.1 (86.0, 4) | 91.7 (87.7, 2) |
| bigbluebutton | 83.6 (74.2, 2) | 88.1 (83.9, 4) |
| jabref | 100 (0) | 100 (0) |
| **Macro** | **92.8 (FP 6)** | **92.8 (FP 10)** |

Macro F2: 0.902 → **0.916 (+1.4pp)**. But the raw table is misleading — read the
**marginal** effect, not the columns:

**Two confounds, both handled by the router's own accounting:**
1. *Double-s21-run.* `s23_verify` re-runs its OWN `super().link()` floor — a
   separate stochastic sample from the standalone `s_linker21` column. So most
   per-dataset F1 swings (bbb +4.5, teastore −6.9) are s21-FLOOR run-to-run noise,
   NOT the augmentation.
2. The floor-independent signal is the router line (`N proposed → M gate-approved
   additions`) and the `llmrouter` FP source:

| dataset | proposed → added | llmrouter FP | true marginal effect |
|---|---|---|---|
| mediastore | 35 → 0 | 0 | nothing |
| teastore | 37 → 5 | **4** | +4 FP, ~0 TP (precision leak) |
| teammates | 103 → 1 | 0 | +1 clean (FP2 are floor coref/entity) |
| bigbluebutton | 64 → 1 | 0 | +1 clean (FP4 are floor entity/coref) |
| jabref | 20 → 0 | 0 | nothing |

**Honest e2e verdict.** The alias-informed augmentation's TRUE contribution is
~2 clean TP (teammates, bbb) against 4 FP (teastore). It did NOT move macro F1
(92.8 flat), same "buys F2 not F1" pattern as blind s23_verify — the recall the
alias map surfaces at the extraction ceiling is largely absorbed by the (fresh)
s21 floor or filtered by the gate, so little reaches the final set.

**The teastore leak is diagnostic and actionable.** teastore's global aliases
include generic single words (`UI → WebUI`, `front-end → WebUI`); injecting them
plausibly drives WebUI over-linking (5 llmrouter links added, 4 FP; the specific
per-link components are not logged, so this is the probable — not confirmed —
mechanism). **Not all aliases are safe: broad generic global aliases inflate FP.**

**Invariant held everywhere** — no dataset regressed below its own floor
(mediastore/jabref untouched; every s23_verify ≥ its floor), so the wiring is safe.

### Refined recommendation
- **Keep** the alias wiring: additive, GATE-safe, raises the extraction recall
  ceiling at zero regression risk (gate + invariant absorb the downside).
- **But filter the injected aliases** to convert it to a net e2e win: drop generic
  single-word global aliases (`UI`, `front-end`) that inflate FP; keep specific /
  multi-word ones (`HTML5 server`, `image provider`, `persistence provider`).
  Directly targets the teastore leak without losing the bbb/mediastore specific-
  alias recovery. (This is where the ambiguous-name set could gate the generic
  aliases — the one place the two knobs compose.)
- **De-noise before any headline claim**: single run + double-s21 confound means
  the ±numbers are soft; the clean signal is the `llmrouter`-source FP/TP, and it
  says the current unfiltered alias set is roughly break-even end-to-end.

## Generic-alias filter — implemented, but the teastore-leak hypothesis was WRONG

`filter_generic_aliases` (proposer.py) was added and wired in (max_df=5): multi-word
alias terms kept, single-word terms kept only if they occur as a standalone token in
≤ max_df sentences. Verified on synthetic data it drops genuinely-generic terms
(`UI`/`front-end` at df≥6) and keeps specific/rare ones (`KMS`, `HTML5 server`).

**But the diagnostic on the ACTUAL teastore alias set shows the filter drops
nothing there**, so the hypothesis that generic aliases caused the teastore FP leak
is **not supported**:

| teastore global alias | words | doc-freq | filter |
|---|---:|---:|---|
| `UI → WebUI` | 1 | **5** | KEEP (== threshold) |
| `WebUi → WebUI` | 1 | 4 | KEEP |
| `PersistenceProvider → Persistence` | 1 | 1 | KEEP |
| `WebUI service`, `Image Provider`, `Auth service`, … | 2–3 | 1–4 | KEEP (multi-word) |

Two corrections to the earlier e2e read:
1. `front-end → WebUI` was **not in this run's alias set at all** — Phase-1 aliases
   are LLM-discovered and stochastic; the set differs run to run. The "UI/front-end
   over-linking" mechanism was *probable, not confirmed*, and is now **falsified for
   teastore**: the discovered aliases there are mostly specific/multi-word.
2. The filtered re-run moved teastore `llmrouter` FP only 4→3 (bbb: 0 FP, but on a
   low-recall floor sample) — a within-noise move on a floor that itself resampled.
   With the filter dropping nothing on teastore, that delta is floor/sample noise,
   not the filter.

**Revised conclusion.** The teastore FP leak is **not a generic-alias problem** — it
is the augmentation over-proposing on a precision-sensitive dataset and a few
candidates slipping past the gate regardless of aliases (the same teastore
sensitivity seen with `s23_union`, RESULTS.md). The frequency filter is kept as a
harmless, principled safety net (it *will* drop generic aliases where they occur, and
is GATE-06 safe) but it is **low-impact on these benchmarks** and does **not** fix
the teastore leak. Lowering max_df to 4 to force `UI` out would be threshold-fitting
to one dataset (rejected).

Net: alias injection remains an extraction-ceiling recall win that is **~break-even
end-to-end** (gain absorbed by the fresh floor / filtered by the gate); the residual
teastore precision cost is an augmentation-over-proposal issue, addressable only by a
stronger per-candidate precision control on that dataset — not by alias filtering.

## Concrete next steps

- [ ] Promote alias injection into the shipped proposer as an additive opt-in:
      `build_batch_prompt(..., aliases=None)` + `propose_batch(..., aliases=None)`
      (default None = current behavior, GATE-safe). Wire `s_linker23._propose`
      (and the extract variants) to pass `self.doc_knowledge.aliases`.
- [ ] Re-measure `s23_verify` end-to-end F1/F2 with the alias-informed proposer —
      extraction recall is only a ceiling; confirm the recovered alias links
      survive the gate and lift F2 without new FP.
- [ ] For the REPLACE question: re-run `extraction_replace_compare` with
      blocks+alias; it should now dominate Framing-C on recall everywhere,
      removing the bbb −2 that blocked a clean replace.
- [ ] De-noise: the aggregate-recall estimates are N=1; a 3-sample average on the
      key configs would tighten the ranking (targeted-recovery findings already hold).
```
