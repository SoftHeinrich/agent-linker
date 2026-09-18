# CLAUDE.md

This is the **s126-only consolidation** of what used to be a branch carrying
every explored linker family (i1/i2, s_linker through s_linker126, the
router, the S23 verification family, and the s26-s125 exploration rounds
these notes originally recorded round by round). Only the paper arm,
`s_linker126`, and the modules its own validation suite depends on remain.
The full history — every archived module, every round's pilot script, every
per-variant result and report — is preserved on
`origin/archive/master-pre-s126-consolidation`; `git log` there for the
complete round-by-round ledger this file used to carry inline.

## What s126 is

`s_linker126` (paper arm) = `s_linker125` (no antecedent shortlist) plus
greedy whole-name span ownership, discard of each residual multi-component
surface, and an exact-antecedent contract on coreference resolutions: a
resolution whose cited antecedent sentence does not write the component's
name *as a name* is not put to the judge. It reuses `_written_as`, already
computed for every union-judging case, so the contract adds no new
computation. `s_linker124`'s antecedent-shortlist mark is dead code once the
shortlist itself is gone.

**Lineage** (each step's own round is in the archived ledger, not here):
s92a -> s109/s110 -> s120 -> s121 -> s122 -> s123 -> s124 -> s125 -> **s126**.

**Promoted to the paper arm on 2026-09-16, by explicit author decision on
simplicity, overriding the component-weighted doc-code gate's refusal**
(terra and luna each read a mixed doc-code signal against the in-invocation
`s123` control — some metrics better, some worse, 3/3 sign-agreement on
both directions depending on backend and re-run; see
`evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv` for the numbers this
branch actually has on disk). The promotion is justified on what the change
removes — a whole mechanism (`_named_before`, its prompt line, and the
paragraph vouching for it, up to 198 lines and 18,327 B off the resolver on
one project) for a four-line predicate — not on a measured win; the
precedent for adopting a cut at a measured non-win is `s_linker78` (see the
archived ledger).

`s_linker126.py` is a **STANDALONE file**: the `s122 -> s123 -> s125 -> s126`
subclass chain is flattened into it, verified byte-identical to the
pre-flatten subclassed form on all five projects at the time of flattening
(`pilot/test_s126.py`, and `pilot/test_s126_standalone.py` while it still
existed — see below). It imports no other `s_linkerNNN` module at runtime —
only `core/`, `pcm_parser{,_v2}.py`, `llm_client.py`, `helper_v3.py`,
`linker_infra.py`.

The ancestor modules this file was flattened from (`s_linker122`,
`s_linker123`, `s_linker125`) and the retired `s_linker25`, along with the
scripts that existed only to compare s126 against them
(`pilot/test_s126_standalone.py`, `pilot/s127_greedy_merge_audit.py`,
`pilot/design_audit.py`), were removed in a second consolidation pass: that
comparison already ran and passed, is recorded here and in `git log`, and
does not need live ancestor code to keep re-verifying. See
`origin/archive/master-pre-s126-consolidation` to resurrect any of it.

## Active Surface

- `run_ablation.py` — ablation runner; registry now holds only `s_linker126`
  and `s_linker126_noknow` (RQ4 knowledge A/B, same module, `no_knowledge=True`).
  `python run_ablation.py --list-variants` prints both.
- `s_linker126.py` — the paper arm, standalone (see above).
- `core/`, `llm_client.py`, `pcm_parser{,_v2}.py`, `helper_v3.py` — shared
  runtime.
- `linker_infra.py` — the linker plumbing, functions and one wrapper class,
  never a mixin: `TracingLLMClient`, `ask_json` (the JSON call path), the
  checkpoint/log/metrics writers, the batching and the log's views. A
  variant keeps each method under its own name with a one-line body, so a
  self-contained file stays readable without an MRO. Do not put a prompt, a
  rule constant or a scan in here — that is the variant's own file, by
  policy.
- `pilot/` — only the s126 validation chain remains: `test_s126.py`
  (24 checks, self-contained against `SLinker126` alone),
  `coref_exact_pilots.py` (trimmed to the fixture loader `test_s126.py`
  uses — its stage-pilot arms comparing s126 against the retired
  `s_linker123` were archived with it), `reading_pilots.py` (trimmed to the
  benchmark/gold-loading helpers this chain still uses), `score_runs.py`,
  `ab_stats.py`, and the two E2E runners `run_s126_e2e.sh` /
  `run_s126_e2e_noknow.sh`.

The router-pilot investigation and every other retired round's pilot/study
material live only in `origin/archive/master-pre-s126-consolidation` now.

## Build & Run

```bash
pip install -e ".[openai]"
python run_ablation.py --list-variants
python run_ablation.py --variants s_linker126 --datasets mediastore
```

The host provides the OpenAI credential as **`OAI_KEY`**, not
`OPENAI_API_KEY`. There is no `OPENAI_API_KEY` in the environment; every
OpenAI-backed command must map `OAI_KEY` into it inline, in the process
environment only:

```bash
OPENAI_API_KEY="$OAI_KEY" python run_ablation.py ...
```

Full five-project E2E form (the standard paired benchmark run):

```bash
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/<run>/phase_states \
LLM_LOG_DIR=../results/<run>/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker126 \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/<run>
```

Never write either credential value to `.env`, logs, results, or tracked
files.

## Measurement Policy — API budget first

**Standing instruction: do not spend a paired end-to-end batch to answer a
question a checkpoint can answer.** An E2E batch is ~25-35 minutes per
invocation and, at six runs with three or four arms, hours of API. Escalate
in this order and stop at the first level that decides:

1. **Deterministic, no LLM calls.** Replay the predicate against recorded
   checkpoints and call logs.
2. **Stage pilot on fixed recorded inputs.** Replay one stage with both
   wordings against the same checkpoint inputs, N samples a side. Assert
   first that the re-declared prompt builders render byte-identically to the
   variant's own.
3. **A composition check.** If the pairs a change adds or removes are not
   pairs a later stage would otherwise propose, and are not in the final
   link set, the stage arm IS the pipeline answer and an E2E would measure
   model drift instead of the change. Structurally vacuous for any change to
   the LAST linker (coreference), since nothing downstream can be starved.
4. **E2E, and only to finalize.** Never compare across invocation sets: arms
   are comparable only when they ran inside the same invocation, so every
   arm a claim rests on goes in the same batch.

**Read the pooled row of a late-stage arm as a summary, never as the
verdict.** A change to one stage (e.g. the coreference resolver) can only
move that stage's own row; a pooled macro statistic across all stages can
manufacture a verdict none of the parts carry by summing independent
sampling noise from stages the change never touched. This is exactly what
happened when s126's own E2E batch was first read (see the promotion note
above) — read a late-stage arm's own `source`-decomposed row, not the pooled
one.

**Six paired runs is the bar; three can manufacture a neutral as easily as a
regression.** At n=3 the two-sided sign-flip floor is p=0.25.

## Design Law — facts stay in code, weighings go in the prompt

The deterministic layer supplies **facts about a case**; the LLM supplies
**judgment about the case**. A clause that tells a judge *how to weigh* what
it sees can be moved out of code into that judge's prompt. A statement of
*what is true of the case* cannot — not because the judge cannot see it, but
because the judge is not disinterested about it. Before proposing any
relocation between code and prompt, classify it fact-or-weighing first.

A discovered fact (another judge's verdict, resampled every run) may OPEN a
case for a downstream judge and must not be used to CLOSE one — s126's own
antecedent-form predicate rests only on a GIVEN fact (catalog plus document,
identical every run) for exactly this reason.

## Standing Gates

- **GATE-01**: the reported arm stays byte-stable —
  `src/llm_sad_sam/linkers/experimental/s_linker126.py` above all. New work
  subclasses or forks it; edits to shared files (`__init__.py`,
  `run_ablation.py`) are purely additive.
- **GATE-06**: no benchmark-derived vocabulary introduced in any new code —
  prompts/rubrics stay generic English; the runtime catalog (component
  names, code identifiers) is the only project-specific input.
- **GATE-07 (the general round)**: every prompt clause and every code gate
  must stand on one of three grounds — a general rule (logic, or a
  distinction that holds for any text), general SE practice (a property of
  software as written anywhere), or prior work this branch or the
  literature already measured. A clause that names a surface form or a
  syntax whose frequency is a fact about these five documents is
  inadmissible however well it scores. `pilot/prompt_defensibility.py`
  scored the whole authored surface against it before it was archived with
  the round that used it most; s126's authored text carries forward from
  that lineage unchanged in the relevant judging clauses.

## Notes

- The full round-by-round ledger (s26 through s125's design rounds, every
  refused arm, every measured gate, the complete rationale for each
  lineage step) lived in this file before the consolidation. It is
  preserved verbatim in `git log`/`git show` on
  `origin/archive/master-pre-s126-consolidation`, at the commit tagged
  "s126 promoted to the PAPER ARM" and its ancestors.
- Default benchmarking backend is set in `.env` (`LLM_BACKEND=openai`,
  `gpt-5.4`). `.env` is untracked.
