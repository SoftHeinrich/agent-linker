# Can the s23 augmentation run *simultaneously* with s21 instead of as post-processing?

Investigation date: 2026-07-03. Scope: the blind augmentation variants
(`s_linker23`, `s_linker23_verify`) and what "run alongside s21" can and cannot
mean. Grounded in the actual data dependencies, not the `link()` call order.

## TL;DR

- **Logically, the augmentation is NOT downstream of s21's linking decisions.** It
  only *looks* like post-processing because `SLinker23.link()` calls
  `super().link()` and then `_augment()` in sequence. The augmentation reads
  **none** of s21's Phase 2/4/5/6 outputs.
- Its only real inputs are the **raw doc+model** (available at t=0) and s21's
  **Phase-1 knowledge** (`model_knowledge.ambiguous_names`, used only as a context
  flag in the evidence bundle). Plus `base_final` — used **only** for the final
  dedup, i.e. at the merge barrier, not to *decide* anything.
- So there are **two different "simultaneous" designs, and they are not the same**:
  - **(A) Symmetric co-extraction into one shared gate** — union the two extractors
    and re-judge everyone together. *Already built and already lost:*
    `SLinker23Union`/`SLinker23Replace` regressed (macro FP 4 → 21, teastore
    P 100 → 74). Losing the floor is what costs the precision.
  - **(B) Concurrent execution, asymmetric (floor-priority) merge** — run the
    proposer→router→gate branch **in parallel with** s21's Phase 2–6, rejoin at a
    barrier, keep s21's accepted links protected. This is a **pure latency +
    framing win over `s23_verify`, with identical outputs and the same
    non-regression guarantee.** This is the recommended next variant.
- Reframing as "concurrent co-extraction" **removes the *post-processing*
  critique** and is an honest description of the mechanism. It does **not** make it
  a more accurate linker — `s23_verify` is still F1-parity/+F2 vs s21. Concurrency
  buys wall-clock and a cleaner story, not accuracy.

## The dependency graph (verified, not assumed)

What each step of the `s23_verify` augmentation actually reads:

| step | reads from s21? | reads what | available when |
|---|---|---|---|
| `_propose` (blocks proposer) | no | `sentences`, `names`, `prev_of` | **t=0** (raw inputs) |
| build `Candidate`s | no | `name_to_id`, `sent_map` | **t=0** |
| `DocModelAgenticRouter.route` | no | the candidates + a generic rubric | after propose |
| `_build_evidence_bundle` | **Phase 1 only** | `self.model_knowledge.ambiguous_names` (→ `is_ambiguous` flag); everything else (`mention_type`, `anchors`, `preceding_text`) from `sent_map`+candidate | **after Phase 1** |
| `_validate_with_evidence` (gate) | no | candidates + bundles + `components` | after bundles |
| final merge / dedup | **`base_final`** | the set `{(sent, comp)}` s21 accepted | **barrier** (needs both streams) |

The one non-obvious tie is `is_ambiguous`, which needs `model_knowledge` from
Phase 1. That is a *context flag*, not a decision, and in `no_knowledge` mode
`ambiguous_names` is empty, so the tie vanishes entirely (full fork from t=0).

Resulting DAG (blind variants):

```
                          ┌──> s21 Phase 2 → 3 → 4 → 5 → 6 ──> base_final ──┐
   inputs ──> Phase 1 ────┤                                                 ├─(barrier)─> floor-priority merge ──> final
   (doc,model) knowledge  └──> proposer → router → gate(uses P1 flag) ──────┘
```

The augmentation branch **forks right after Phase 1** and runs fully concurrently
with s21's Phases 2–6. s21 already ships the executor for exactly this:
`SLinker21._run_parallel` (ThreadPoolExecutor, used for Phase-1 model+doc and
Framing-C pass1+pass2). Adding the augment branch is one more task in that pool.

`s_linker23_ctx` is the **only** variant with a true data dependency on s21's
*final* output — its proposer is conditioned on `base_final` (`ALREADY LINKED:`).
It cannot fork early and must stay post-hoc. (It is also the invalidated variant,
so this costs nothing.)

## Why the symmetric "one shared gate" version is not the answer

The tempting simultaneous design is: run both extractors, union the candidates,
push the union through **one** validation gate, score the result. That is exactly
`SLinker23Union` (and `Replace` is the degenerate "only the new extractor"). The
pilot already measured it:

| design | macro F1 | total FP | note |
|---|---:|---:|---|
| s21 (floor) | 93.3 | 4 | protected |
| s23_union (one shared gate) | 88.9 | 21 | **teastore P 100→74** |
| **s23_verify (floor + separate gate)** | **92.7** | 9 | floor preserved |

Mechanism: in the symmetric framing s21's *own* validated links are no longer
protected — they get re-judged in the same pass as the speculative blocks
candidates, the gate is stochastic and precision-sensitive, and the extra volume
admits false positives. **The asymmetric floor is doing real work:** it keeps s21's
accepted links untouched and only puts the *new* candidates at risk. "Run
simultaneously" must not be allowed to mean "dissolve the floor."

### Corollary: don't fuse the gate into shared batches

A cheaper-looking idea is to run **one** Phase-4 validation over `Framing-C ∪ blocks`
(tagging provenance), instead of validating the two streams in two passes. This
**breaks the byte-level non-regression guarantee**: s21's two-pass validator
batches cases (25/batch for entity, 8 in the router gate), and which cases share a
batch is part of the input. A Framing-C candidate that standalone-s21 accepted
could flip if it is re-batched next to blocks candidates. To *prove* s23 ≥ s21 you
must validate s21's Framing-C stream in **exactly s21's batches** and validate the
blocks candidates **separately**. That is `s23_verify` — just with the two
validations overlapped in time. So: concurrent-but-separate ✅, fused-shared-batch ❌.

## Recommended next variant — `s23_verify` executed concurrently (design B)

Same class, same gate, same outputs (modulo the LLM's inherent run-to-run noise),
same gate-floor invariant — only the *scheduling* changes. Sketch:

```python
class SLinker23Concurrent(SLinker23Verify):
    _VARIANT_NAME = "s_linker23_concurrent"

    def link(self, text_path, model_path, **kwargs):
        # Fork: s21 pipeline and the augment branch run in one _run_parallel.
        # Both re-derive raw inputs; the augment branch waits only on Phase-1
        # knowledge (or drops the is_ambiguous flag / uses no_knowledge to fork at t=0).
        streams = self._run_parallel({
            "base":    lambda: super(SLinker23, self).link(text_path, model_path, **kwargs),
            "augment": lambda: self._augment_branch(text_path, model_path, **kwargs),
        })
        return self._floor_merge(streams["base"], streams["augment"])   # barrier + priority
```

Properties:
- **Wall-clock ≈ max(s21, augment)** instead of `s21 + augment`. The augment branch
  (≈10 proposer calls + router + one two-pass gate over the VALIDATE set) overlaps
  s21's Phases 2–6 (its extraction + two-pass gate + coref). Roughly halves the
  added latency `s23_verify` currently pays.
- **Identical guarantee.** The merge is still floor-priority: s21's accepted links
  are copied verbatim; only gate-approved new candidates are appended. Non-regression
  holds exactly as today; `test_s23_gate_floor.py` still covers it unchanged.
- **Cost is unchanged** (same LLM calls, just overlapped) — this does *not* address
  the token-cost critique, only the latency one.
- **Coupling to fix:** the augment branch reads `self.model_knowledge`, currently a
  side effect of `super().link()`'s Phase 1. To run concurrently, either (i) fork
  after a shared Phase-1 step, (ii) recompute `model_knowledge` in the branch
  (one cheap `_analyze_model` call), or (iii) drop `is_ambiguous` (context only).
  Option (i) is cleanest and keeps outputs identical.

## The framing question, answered honestly

> can it run simultaneously with s21, instead of post-processing? or can it be
> framed like that?

Yes to both, with one honest boundary:

1. **"Instead of post-processing" — legitimate.** For `s23`/`s23_verify` the
   augmentation is not logically downstream of s21's decisions; it is a **second
   doc→model extractor** that shares only the raw inputs and Phase-1 knowledge. The
   paper can present the linker as an **ensemble of two extractors feeding a shared
   validation discipline, with the s21 stream given merge priority to guarantee
   non-regression** — which is both accurate and drops my earlier "it's just
   post-processing" critique.
2. **"Simultaneously" — yes, as design (B), not design (A).** Concurrent execution
   with a floor-priority barrier merge is safe and free; symmetric co-extraction
   into one re-judging gate is the thing that already regressed.
3. **What it does *not* buy: accuracy.** `s23_verify` is F1-parity/+F2 vs s21.
   Reframing it as concurrent co-extraction changes the *story* and the *wall-clock*,
   not the score. Don't let the cleaner framing be read as a stronger result.

## Related question — can the proposer *replace* s21's entity pass?

Mechanically yes (`SLinker23Replace` already overrides `_run_framing_c`), but the
mechanism says not as a straight swap. s21's entity pass is **alias-injected**
(`_run_framing_c` feeds Phase-1 global aliases as `mappings` into the extraction
prompt) and 2-pass-union; the blocks proposer is **knowledge-blind** and
single-pass. Consequence (pilot `extraction_replace_compare`):

- blocks has the higher recall *ceiling* (per-item framing beats s21's flat
  2-pass union: teammates 1.000 vs 0.825, bbb 0.742 vs 0.677), **but is not a
  superset** — on bbb it loses 2 gold Framing-C uniquely catches, most likely
  **alias-mediated mentions** the knowledge-blind proposer can't resolve.
- End-to-end, replace is a net loss: ~2× candidate volume into the same stochastic
  gate → more FP, precision falls. Verdict stands: **union/augment, don't replace**;
  and integrate the union through the router (`s23_verify`), not one raw gate.

**Next variant to unblock replace:** a **knowledge-conditioned blocks proposer** —
inject the same Phase-1 aliases (and ambiguous-name flags) into `build_batch_prompt`
that `_run_framing_c` already passes. Targets the exact bbb −2 alias gap. If it
turns blocks into a true superset of Framing-C, replace becomes credible as a
single-pass, per-item, alias-aware extractor (GATE-06-safe: aliases are runtime
doc-derived input s21 already consumes). Measure at the extraction-recall ceiling
first (does it recover the −2?), then end-to-end F1.

## Concrete next steps

- [ ] Register `s_linker23_concurrent` (design B): `SLinker23Verify` with a
      fork-after-Phase-1 `link()` + floor-priority barrier merge. Assert
      output-set equality vs `s23_verify` on a clean-weather run (should match
      modulo LLM noise), and time both to quantify the latency win.
- [ ] Keep `s23_union`/`s23_replace` as the recorded negative for design (A) — they
      already answer "can you just run one shared gate over the union?" (no).
- [ ] Paper framing: describe the mechanism as concurrent two-extractor co-linking
      with a priority merge, not as a post-hoc augmentation pass.
- [ ] Not recommended: fused shared-batch gate (breaks the byte-level floor proof).
```
