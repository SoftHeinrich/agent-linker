# CLAUDE.md

This is the **router branch**: the experimental linker repo, extended beyond the
prior s20U trim with a second-route (doc->code) infra and a bounded-autonomy
agentic augmentation variant. The full history (all other linker families,
planning docs, logs, results, archives, tests) lives on `master`.

**Branch relationships (verified by `git ls-tree`, not assumed):**

| Branch | Diverges from `router` at | Has `s_linker21.py`? | Has `router_direct.py` / `agentic_router.py` / `proposer.py` / `s_linker21_agentrouter.py`? |
|---|---|---|---|
| `master` | `58d0d7f` (full history, pre-s20U-trim) | No — still on `s_linker20_union.py` | No |
| `s20U` | `9e40ac3` (s_linker21 inlined as canonical, s20U trim point) | Yes | No |
| `router` (this branch) | — | Yes | **Yes — only here** |

The entire code-routing surface (direct sentence→code linking + the bounded-autonomy
agentic router) is **`router`-branch-only**. It has not been merged/ported to `s20U`
or `master` — do not assume it is reachable by checking out either of those branches.

**Two distinct "router" concepts — do not conflate them:**

| | `DocCodeSentenceRouter` (`router_direct.py`) | `DocModelAgenticRouter` (`agentic_router.py`) |
|---|---|---|
| Task | DOC→CODE | DOC→MODEL (sentence→component) |
| Granularity | Per SENTENCE | Per CANDIDATE (sentence, component) |
| Decision | ARCH vs CODE — should this sentence go through direct code-linking at all | VALIDATE / CODE / REJECT — is this candidate a real link, a code-path mention, or neither |
| Used by `s_linker21_agentrouter.py`? | No — superseded there | Yes — its CODE action is the escape hatch into `DirectCodeLinker`/`DirectLinkJudge` |

`DocCodeSentenceRouter` remains standalone reusable infra (not currently wired into
any linker); `DocModelAgenticRouter` is what `SLinker21AgentRouter` actually uses.

## Active Surface

Every round below has a report directory under `../results/`; this guide carries the
verdict and the number, and those READMEs carry the narrative. `python run_ablation.py
--list-variants` prints what actually resolves.

**Infra and canonical artifacts**

- `run_ablation.py` — ablation runner; benchmark inputs from the sibling `../ardoco`
  repo, or from `ALINKER_BENCHMARK`. Variant registry lives here.
- `s_linker21.py` — **CANONICAL** paper Full linker (`SLinker21`), standalone.
  **GATE-01: byte-stable.** New work subclasses or forks; never edit.
- `router_direct.py` — doc→code infra: `CodeUnit`/`load_code_units`/`CodeIndex` (parses
  a `.acm` model), `DirectCodeLinker`, `DocCodeSentenceRouter` (per-sentence ARCH/CODE),
  `DirectLinkJudge`. Reusable, not a linker; not wired into any current variant.
- `{agentic_router,proposer}.py` — `GroundedTypedProposer` and `DocModelAgenticRouter`
  (per-candidate VALIDATE/CODE/REJECT for doc→model, `StrictGate`). Reusable infra.
- `s_linker21_agentrouter.py` — bounded-autonomy augmentation over `SLinker21`; a
  gate-floor invariant means it cannot regress below s21. ~1pp F1 below the non-agentic
  named+routed target (verified gold-incompleteness, not error).
- `s_linker24_role_orchestrator.py` — the retained S24 multi-turn controller.
  182 TP / 8 FP / 13 FN, macro F1/F2 96.07/95.40.
- `s_linker25.py` — the paper variant of the S24 design, standalone: three linkers in
  fixed **name-evidence order** (full-name → partial-name → coreference), no controller.
  **The reference band, N=6: macro F1 96.4 ± 0.4, F2 95.4 ± 0.6, TP/FP 180.8 / 4.8.**
  Everything from s26 on is measured against this design or a descendant of it.
- `s_linker26.py` … `s_linker89.py` — the rounds below. All `experimental=True`; the
  head is the highest-numbered one the compaction round confirms.
- `core/`, `llm_client.py`, `pcm_parser{,_v2}.py`, `helper_v3.py`, `ilinker3.py` —
  shared runtime.
- `pilot/` — every audit and arm cited below. Deterministic audits first
  (`rule_audit.py`, `bind_audit.py`, `partial_audit.py`, `stage_diff.py`,
  `composition_check.py`, `prompt_defensibility.py`, `lemma_swap_pilot.py`), then stage
  pilots (`*_pilots.py`), then `score_runs.py` for whole-run scoring and the paired
  permutation test.

The router-pilot investigation that produced `agentic_router.py`, `proposer.py` and
`s_linker21_agentrouter.py` is archived at `.planning/archive/router-pilot-260701/`.

### The name relation — the design rationale in one table

Four hand-written lexical rules turned out to be **one relation at four settings**,
verified as an identity over all 3697 (name, sentence) pairs (`pilot/rule_audit.py
--only A2`). Two dimensions: *fidelity* (how exactly the characters reproduce the name)
and *extent* (the whole name, or one word of it). Yield over the five projects:

| fidelity / extent | pairs | gold | gold per pair |
|---|---|---|---|
| `AS_SPELLED` whole name | 112 | 107 | **0.955** |
| `ANY_CASE` whole name | 172 | 133 | 0.773 |
| `ANY_SPELLING` whole name | 176 | 137 | 0.778 |
| `ANY_WORD` one word | 281 | 161 | 0.573 |

**The looser the form a linker scans, the stricter the judge behind it** — the full-name
linker scans the tight rows and judges leniently, the partial-name linker scans the
loosest row and judges target-blind, the coreference linker reaches what no row reaches
and rejects when uncertain. Two cells do not nest (`Image Provider` is reached by
`ANY_SPELLING` and not `ANY_WORD`; `redis pubsub` the reverse), so compound splitting is
a different normalization and the linker takes the **union**. **Nothing in the
deterministic layer admits a link** — 0 of 18 predicates, `--only A1`; every scan
produces a case for a judge.

### The variant ledger (s26–s74)

Verdict key: **✓** adopted, **✗** refuted, **=** parity/neutral, **?** undecided.
Reports under `../results/`; each row's detail is in the round README named at the end
of its block.

**Architecture — can the two document-reading questions be merged?** (`s25_architecture_exploration/`)

| | change | verdict | headline |
|---|---|---|---|
| s26 | merge alias discovery into the batched reading, drop the judge | ✗ | F1 94.27 / F2 93.47 vs 96.4 / 95.4 |
| s27 | one call, whole document, both questions | ✗ | F1 91.70; accuracy tracks document length (jabref 13 sents 100.0, teammates 198 **84.1**) |
| s28 | s26 minus the partial-name suppression | ✗ | 93.89 — recovers nothing |
| s29 / s30 | lexical grounding check / judging folded into extraction | ✗ | F1 90.07 / 90.40; both collapse MediaStore recall to 61.3% |
| s31 | review folded into the proposing call | ✗ | TP 178.7 but FP 9.7 — a proposer approves its own list |
| s32/33/34 | judge's rubric carried in the extraction calls, any / majority / unanimous | = on F2 | F2 95.01 / 94.97 / 95.20, TP 181.3–181.7, but FP 13.0 / 13.0 / 10.7 |
| s35 | the carried review asked *before* the document | ✗ | FP 8.3 (best of the line) at TP 162.0 |

**The result:** the two questions have **opposite optimal granularities** — references
degrade with passage length, alias definitions are stated once and used far away, so
names need the whole document. s25's two stages *are* those two granularities. The
knowledge module is **necessary rather than chosen**, with six implemented alternatives
pricing it.

**Judging arrangement** (`s38_audit/`)

| | change | verdict | headline |
|---|---|---|---|
| s36 | the two focused full-name calls merged into one | ✗ | n=6: F1 **-0.7** (p=0.01), FP +3.5 (p=0.01), F2 ±0.0, 79 calls vs 89 |
| s37 | + a committed quote per criterion | ✗ on F1 | n=6: F2 +0.08 (p=0.81), **F1 -0.77 (p=0.017)**, TP 182.2, FP 8.8 |
| s38 | one merged prompt sampled twice, verdicts ANDed | = | nothing significant — but the audit shows the samples split on **1.0 of 174.7 candidates (0.6%)**, so s38 is s36 plus a redundant call |
| s39 / s40 | a second alias judge (usage as well as validity) | ✗ | dominated on four projects; F1 93.5 |

**The mechanism, and the sharpest independence result on the branch:** s25's two focused
calls disagree on **4.7 of 172.3** candidates (2.7%, 1.0 gold / 3.7 not) while s38's two
samples of one prompt disagree on 1.0 of 174.7 (0.6%). The 3.7 false positives the
disagreements remove are exactly the margin s25 leads s36 by. **Independence comes from
asking a different question, not from resampling the same one.**

**The mention label** (`s25_complexity_audit/`, `s43`/`s44` E2E dirs)

| | change | verdict | headline |
|---|---|---|---|
| s42 | s36 + the three-value label | = | TP ±0.0 (p=1.00), F1 -0.1 — free **on that base** |
| s43 | s25 + the three-value label | ✗ | **F1 -1.3, F2 -1.3**, both at the n=3 floor |
| s44 | merge only the case grading (deletes the last case-sensitivity rule) | ✗ | n=6: **F1 -0.9 (p=0.05)** — and the first three runs read F1 -0.0 (**p=1.00**) |

**Resource bounds and the alias table's second job**

| | change | verdict | headline |
|---|---|---|---|
| s45 | `COREFERENCE_BATCH = JUDGE_BATCH` | ✓ parity | n=6: F1 -0.2 (p=0.52), F2 -0.0 (p=0.91), **65.3 calls vs 88.8 (-26%)** |
| s46 | the alias table no longer suppresses partial-name candidates | ✗ | n=6: **FP +6.5 (p=0.01), F1 -1.5 (p=0.00)** — and freeing 16 candidates *cost* 2.0 TP |
| s76 | the same batch unification on the s75 base | ✗ | TP -7.0, **F2 -1.8**, 65 calls vs 89 — base-dependence, not a contradiction of s45 |

**Mechanism removal and code merges** (`s4748_e2e_*`, `s49_composed_e2e_*`)

| | change | verdict | headline |
|---|---|---|---|
| s47 | delete the partial-name linker's grounded identity review | ✓ | n=6: TP +6.2, FP +6.8 (both p=0.00), F1 +0.2, **F2 +1.3 (p=0.01)** |
| s48 | eight condition copies in five shapes → three named predicates; three never-firing conjuncts deleted | ✓ free | composition **-0.2 (p=0.59)** — the arms' link sets differ less between arms than within them |
| s49 | s47 + s48 composed | ✓ | TP +5.0, F2 +0.9 (p=0.03), 87.2 calls vs 89.2 — the two are independent and additive |
| s65 | the four lexical rules restated as one relation at four settings | ✓ identity | 49/49 invariant checks; **no E2E owed** — candidate sets are equal |

**The prompt round — ablating the hand-written English** (`s5051_prompt_ablation/`,
`s5253_prompt_bisect/`, `s5455_prompt_families/`, `prompt_stage_pilots/`)

Ten rule constants, 4022 B, in 88 calls per five-project run — **6.4% of what the
workflow sends.** All arms are vs s49, six paired runs each.

| | generalized | rule text | macro F1 | p |
|---|---|---|---|---|
| s50 | the coreference resolution rule only | -27% | -0.2 | 0.71 |
| **s55** | **the whole coreference family** | **-31%** | **-0.0** | **0.90** |
| s54 | coreference + knowledge | -34% | -1.1 | 0.00 |
| s52 | coreference + full-name | -41% | -2.1 | 0.00 |
| s51 | all nine of ten constants | -44% | -2.4 | 0.00 |
| s53 | all nine, the indicted clause restored | -44% | -2.5 | 0.00 |

Then at clause level, on fixed recorded inputs (minutes, not hours): s56 (delete the
coreference preamble) **TP -16.2**, s58 (generalize the extraction rule) **FP +20.2** —
two hypotheses refuted before an E2E was paid for. `P1_FOCUS` generalizes for nothing
while `LAYERED_ENTITY_RULES` costs 2.4 FP on the same candidates in the same call.

- **s59 is what survives** — the coreference family + `P1_FOCUS` + the alias judge
  rubric. Rule text -26%, instruction bytes per run -34%. E2E: **TP +1.5 (p=0.05), F2
  +0.5 (p=0.03)**.
- **Both load-bearing families are on the admitting side.** A rejecting stage that
  over-rejects is caught by recall it never had; **an admitting stage that over-admits
  has no downstream that can tell.**
- **A prompt clause is removable when something downstream rejects by default.**
  Coreference rules sit in front of a gate that rejects when uncertain — they go.
  `ENTITY_EXTRACTION_RULES` sits in front of a judge that approves by default — it stays.

**The merged-alias round** (`merged_alias_design/`)

s60 folds alias proposal into the reading and keeps the judge separate — the one
arrangement the s26–s34 line never tried. The **alias side improves** (stage FP -16.6,
p=0.01) and the pipeline loses: **TP -5.0, FP +11.2, F1 -2.7** (all p ≤ 0.01), with
13.5 of the 14 extra FP landing on the **partial-name** linker. Deterministic cause, no
LLM call: adding the single term `GAE` to s60's table takes teammates from 40 candidates
to 30 with no gold lost. Per project the loss lands exactly where the partial-name
linker runs (teammates 91.40 → 83.56, bigbluebutton 93.10 → 89.02; the other three
unchanged). **The merge saves one call per project and costs 2.7 macro F1.**

s61 adds `ALIAS_EXCLUSION_RULES` to the merged reading's judge, because that reading
leaks identifier fragments the dedicated proposer never did. Measured reach on this
benchmark: **zero.** Kept as design integrity, not as a performance claim.

**The partial-name round** (`partial_name_round/`)

- **The judge is not the bottleneck; the proposer is.** The denotation judge runs at 95%
  recall / 83% precision over the gold candidates, so a *perfect* judge would be +1.0 TP
  / -3.5 FP. All headroom is upstream.
- **Of the 22.8 candidates the stage declines, 15.0 are not a loss** — every one is
  recovered by the coreference linker. Split declines by deterministic cause *and* check
  the final link set before calling any of them a hole.
- s62 (bound the ownership prefix to English inflections) **✓ TP +2.3 (p=0.00) at its
  own source**, neutral at the stages it cannot reach.
- s63 (fix `"" in "-_"`, which treats 378 sentence-initial spans as inside a qualified
  identifier) **✗ FP +3.8 at the source.** The defect is load-bearing on this benchmark
  and is **retained and documented — a validity threat the paper must state, not a design
  choice.** It is the one place where "measured" and "defensible" point apart.
- s64 (a deterministic `AS_SPELLED` net for pairs the extractor never proposed) **✓ TP
  +1.2 (p=0.01).** Case is the whole design: the same scan case-insensitively is 31.3
  pairs at 0.06 gold each against 1.2 at 0.86.

**The bind round — could the prompts do the deterministic layer's work?** (`bind_round/`)

- s66 (`_keep_stated_names` deleted, its contract stated in the extraction prompt) —
  **✓ holds**, F1 -0.2 (p=0.76), nine paired runs across two sets. Buys rule count, not
  calls.
- s67 (relocate the two tight scans as well) — **✗ TP -4.0 (p=0.03), F2 -1.1.**
- s68 (drop the label's qualified-path value) — **? undecided.** The macro read TP -5.0
  and four fifths of that sat in an extraction call the change cannot touch; restricted
  to the candidates both arms proposed it is TP -1.0 (p=0.30).
- **Telling the model to scan does not make it scan** — third measurement: a clause
  asking for exactly what the tight scans scan recovers **none** of the 3.6 TP their
  deletion costs.
- **Four predicates have no prompt form** (`_iter_batches`, `_window`, `_unlinked`,
  `_union`): control flow over calls, not statements about text. **The floor on "how many
  hand-written rules" is above zero and it is structural.**

**The general round — GATE-07 applied** (`general_round/`)

s70 scores 1700 of 3645 authored bytes admissible. The bar caught exactly **two** spans
in the whole surface. s71/s72 (replace the rubric's four numbered conditions with one
principle) cost ~0.8 F1 (94.80 / 94.94); s73 (remove "a heading, or a list") cost
**2.7 TP** — general documentation practice, not a corpus shape; s74 (remove `x.y or
x.y.z` from the judging rubric) is parity (95.60 vs 95.74). **A general clause is not a
drop-in for a specific one**, and **a clause is only general relative to the judge that
reads it** — moving `QUALIFIED_CLAUSE` into the coreference rubric costs 3.0 TP because
that stage's cases contain no name for a clause about identifiers to be about.

### Standing findings

Rules earned by the rounds above and not superseded. The gates, the measurement policy
and the design law have their own sections below.

- **Every consolidation of two LLM decisions into one call raises recall and lowers
  precision.** Twelve variants, five instances, no exception. Splitting buys precision,
  merging buys recall; s25 sits at the precision corner where F1 rewards it, and F2 does
  not distinguish s25 from s36 or s34.
- **A judging step must be separate, semantic, lenient, independent of what it judges,
  and undivided.** Five properties, each measured by a variant that dropped it (s29–s35).
  The dedicated call is the only arrangement that is simultaneously undivided,
  context-free and lenient.
- **The alias table has two jobs and both are load-bearing.** It *admits* full-name
  candidates (23 gold links) and *suppresses* partial-name ones, so **table size trades
  recall between two linkers** and no single-stage arm can see it. Measured four ways
  (s26 diagnosis, s46 at F1 -1.5, the partial-name round, s60 at F1 -2.7). It is broad
  enough to hold document-introduced short forms like `GAE` and narrow enough to exclude
  ordinary name words like `Server`, and only a document-wide pass told to reject terms
  whose ordinary English use dominates produces that shape.
- **A stage arm screens candidates; it does not decide them.** Nine instances of a
  stage-level arm pointing opposite to the composed pipeline, always on precision. The
  mechanism is `_unlinked`: a link admitted early is locked into the union *and stolen
  from the later, stricter linkers*. `pilot/composition_check.py` tests the precondition
  deterministically — when it reads 0 pairs, the stage arm **is** the pipeline answer.
- **Six paired runs is the bar; three can manufacture a neutral as easily as a
  regression.** s44 read F1 -0.0 (p=1.00) over its first three runs and F1 -0.9 (p=0.05)
  over six.
- **A trace-derived equivalence is a hypothesis, not a licence.** Equal aggregate
  behaviour per label value does not make a distinction inert — rewriting the field
  changes the prompt for every case that carries it. s53 was directionally right about
  which clause was implicated and still wrong about the mechanism.
- **A surface attribution is not a causal one.** "This link came in through a term only
  this arm has" is not "this term caused it" — every alias is fed to the extraction
  prompt, and the table is not stable enough to attribute from (byte-identical knowledge
  prompts still build tables differing by 2.8 terms per run).
- **A clause is not independently priceable.** Two changes that each lose ground can gain
  it together when one changes what the other's population contains (s77 alone F2 -1.8;
  inside s78, positive). Measured in both directions.
- **A fact's value depends on what else the code is doing.** The mention label is worth
  -10.7 TP with the gates in place and *recovers* precision once they are gone.
- **A value chosen by unification is defensible; a value chosen by search is not.**
- **Never compare across invocation sets.** Absolute levels drift — s49's FP mean read
  10.7, 11.7, 12.5, 14.5 and 16.8 across five sets in one day, one run hitting 33. Arms
  are comparable only when they ran in the same invocation.
- **The harness null is not zero.** `s_linker{49,59,65,66,75}_null.py` are byte-identical
  copies of their base differing only in the checkpoint namespace. `s_linker49_null` read
  TP -4.8 / F1 -0.7 / F2 -1.2 against the code it copies, sign consistent in 6 of 6 runs;
  `s_linker75_null` read F1 -1.58 / FP +10.7. Sampling is not pinned; two runs of one program are two draws. The
  floor is measured and recorded — see the measurement policy for why new batches no
  longer carry a null arm.

## Build & Run

```bash
pip install -e ".[openai]"
python run_ablation.py --list-variants
python run_ablation.py --variants s_linker21 --datasets mediastore
python run_ablation.py --variants s_linker21_agentrouter --datasets mediastore
```

The host provides the OpenAI credential as **`OAI_KEY`**, not `OPENAI_API_KEY`.
There is no `OPENAI_API_KEY` in the environment; every OpenAI-backed command
must map `OAI_KEY` into it inline, in the process environment only:

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
  --variants s_linker21 s_linker25 \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/<run>
```

Never write either credential value to `.env`, logs, results, or tracked files.

## Measurement Policy — API budget first

**Standing instruction: do not spend a paired end-to-end batch to answer a question a
checkpoint can answer.** An E2E batch is ~25-35 minutes per invocation and, at six runs
with three or four arms, hours of API. Most questions on this branch have been settled
for minutes. Escalate in this order and stop at the first level that decides:

1. **Deterministic, no LLM calls.** Replay the predicate against recorded checkpoints and
   call logs (`pilot/rule_audit.py`, `bind_audit.py`, `unlinked_audit.py`,
   `partial_audit.py`, `stage_diff.py`, `lemma_swap_pilot.py`). This settles identities
   (`_unlinked` removes nothing), reach (a scan frees 12.0 pairs and 0.0 gold), and yields
   (gold per pair by fidelity). Two of `s_linker69`'s four changes never needed a call,
   and `s_linker85` replaced the whole morphology rule on this level alone — 3697 pairs
   compared, 2 spans different, no runs bought.
2. **Stage pilot on fixed recorded inputs.** Replay ONE stage with both wordings against
   the same checkpoint inputs, N samples a side (`pilot/prompt_stage_pilots.py`,
   `bind_pilots.py`, `fold_pilots.py`). Minutes, not hours. Always assert first that the
   re-declared prompt builders render byte-identically to the variant's own.
3. **`pilot/composition_check.py` (or the equivalent inline check).** If the pairs the
   change adds or removes are not pairs a later stage would otherwise propose, and are
   not in the final link set, **the stage arm IS the pipeline answer** and an E2E would
   measure model drift instead of the change. Structurally vacuous for any change to the
   LAST linker (coreference), since nothing downstream can be starved.
4. **E2E, and only to finalize.** Pay for runs when the composition risk is non-zero, and
   then only for the change that carries it — not for the whole variant. Never compare
   across invocation sets: arms are comparable only when they ran inside the same
   invocation, so every arm a claim rests on goes in the same batch.

**No in-set null arm.** Earlier rounds carried a byte-identical copy of the base
(`s49_null`, `s59_null`, `s65_null`, `s66_null`, `s75_null`) to size the harness noise a
delta had to clear. That floor is now measured — six rounds of it, and it is quiet except
where a checkpoint-namespace difference makes it loud (the finetune round's `s75_null`
reads F1 -1.58 / FP +10.7 against its own control, `../results/finetune_round/README.md`).
**Do not add one to new batches.** It is a whole arm — a third of a two-arm invocation, a
quarter of a three-arm one — spent re-measuring a constant, and the measurement policy
above says not to pay E2E for a settled question. Read new deltas against the recorded
floor and against N>=3 paired runs with a sign-flip permutation test, which is what
separates a real effect from the +/-55-link run-to-run swing anyway. If a *new* claim
turns on the floor itself, re-measure it once and record it here rather than carrying it
in every batch.

**Do not pair-run arms that a checkpoint replay separates.** Adding an arm to an
invocation multiplies its cost by the number of arms; an arm that a stage pilot already
answers does not belong in the batch. When a batch is running only to raise n on an
already-decided arm, stop it.

**Read an arm on the `source` its change can reach.** A macro F1 over a multi-arm
invocation mixes one stage's effect with two stages of sampling — `s_linker68`'s macro
read TP -5.0 while four fifths of that gap sat in an extraction call whose prompt the
change did not touch (`pilot/source_stats.py` and the per-stage decomposition in
`../results/bind_round/README.md`).

## Design Law — facts stay in code, weighings go in the prompt

The deterministic layer supplies **facts about a case**; the LLM supplies **judgment
about the case**. A clause that tells a judge *how to weigh* what it sees can be moved
out of code into that judge's prompt. A statement of *what is true of the case* cannot —
not because the judge cannot see it, but because the judge is not disinterested about it.

| moved into the prompt | kind | outcome |
|---|---|---|
| `skip_qualified` | weighing | folded — TP −0.4 (p = 0.44) |
| `skip_stricter` | weighing | folded — **TP +4.0, FP ±0.0** |
| the mention label, self-reported by the judge | fact | **−6.7 TP** |
| the mention label, removed | fact | **−10.7 TP** |
| `unique_owner` (`fold_pilots.py --pilot foldowner`) | fact | **−8.4 TP** |
| the target, shown to the denotation judge (`s_linker25`) | fact | **−5.5 gold** |

Two folds, four refusals, no exceptions. This supersedes the earlier fold law ("a gate
folds when the judge is shown what the gate reads") — that rule predicted the mention
label would fold, and it does not: four of its five values are computable from the
sentence the judge is holding, and asking the judge to compute them still costs 6.7 true
positives. **Information the judge *can* derive is not information the judge will derive
impartially.** Before proposing any relocation, classify it fact-or-weighing first; the
arm is only worth paying for on the weighing side.
Details and what it does to the other conceptual leftovers:
`../results/concept_round/README.md`.

## Standing Gates

- **GATE-01**: canonical/paper artifacts stay byte-stable —
  `src/llm_sad_sam/linkers/experimental/s_linker21.py` above all. New variants
  subclass it; edits to shared files (`__init__.py`, `run_ablation.py`) are
  purely additive (new export line, new registry entry).
- **GATE-06**: no benchmark-derived vocabulary introduced in any new code —
  prompts/rubrics stay generic English; the runtime catalog (component names,
  code identifiers) is the only project-specific input.
- **GATE-07 (the general round)**: every prompt clause and every code gate must stand
  on one of three grounds — a **general rule** (logic, or a distinction that holds for
  any text: use/mention, reference, negation, ambiguity), **general SE practice** (a
  property of software as written anywhere: qualified names compose), or **prior work**
  this branch or the literature already measured. A clause that names a surface form or
  a syntax whose frequency is a fact about these five documents is inadmissible however
  well it scores. GATE-06 forbids benchmark *vocabulary*; GATE-07 forbids benchmark
  *shapes*, which is the weaker thing a reviewer will actually catch.
  `pilot/prompt_defensibility.py` scores the whole authored surface against it
  (`s_linker70`: 1700 of 3645 bytes admissible).
  **The bar catches shapes peculiar to a corpus, not the structure every document of the
  genre has** — applying it too widely cost 2.7 TP per run (`s_linker73` removed "a
  heading, or a list", which is general documentation practice). In the whole authored
  surface it caught exactly **two** spans: the judging rubric's `x.y or x.y.z`
  (removed in `s_linker74`, F1 95.60 against `s70`'s 95.74 — parity) and the alias
  prompt's, kept by measurement.
  **Three lessons from applying it** (`../results/general_round/README.md`):
  a general clause is not a drop-in for a specific one — the alias prompt's
  `X.Y or X.Y.Z` sentence admits **0** identifier fragments and so does every general
  replacement, yet replacing it grows the alias table from 24.0 to 36.7 terms per run,
  so **its measurable effect is not the effect it states**; a clause is only
  general *relative to the judge that reads it* — moving `QUALIFIED_CLAUSE` into the
  coreference rubric costs TP 3.0 because that stage's cases contain no name for a
  clause about identifiers to be about; and **restructuring a rubric is not the same
  edit as degeneralizing one** — replacing the four numbered reject-conditions with a
  single principle reads TP +0.7 / FP -1.3 on a fixed candidate set and costs ~0.8 F1
  composed (`s_linker71` 94.80 at n=6, `s_linker72` 94.94), so the enumeration stays
  and only the span that names a shape changes.

### The finetune round (s75, s75_null, s76) — every remaining corpus-shaped span

`s_linker74` had removed the one span GATE-07 caught in the judging path and left four:
the same distinction restated in three bespoke wordings (`ENTITY_EXTRACTION_RULES`,
`P1_FOCUS`, `LAYERED_COREF_RULES`) plus `ALIAS_EXCLUSION_RULES`, which still spelled
`X.Y or X.Y.Z`. The round's budget was set in advance at **2 pp of macro F1 to remove
finetuning**. Report: `../results/finetune_round/README.md`; arms:
`pilot/finetune_pilots.py`; invariants: `pilot/test_s75_nofinetune.py` (36 checks);
runner: `pilot/run_s75_e2e.sh`.

- **Stage arms, three a side, replayed on s74's own checkpoints.** Extraction, general
  clause instead of the code-path one: TP +0.7 (p = 1.00), FP -6.0 (p = 0.20).
  Coreference, phrase removed and nothing added: TP +4.7, FP +3.7. P1's tail dropped
  **with `QUALIFIED_CLAUSE` added**: TP -0.7 (p = 0.90), FP -1.3 (p = 0.40); P1's tail
  dropped **with nothing added**: **TP +2.3 (p = 0.20), FP ±0.0 (p = 1.00)**. **A clause
  belongs once per prompt**: the full-name rubric already states the ground inside
  reject-condition (1), so adding the clause there is a restatement and reads worse than
  removing the tail alone. The extraction prompt has no enumeration, so there it is added.
- **The alias syntax's defence does not reproduce, and the round says so.** The general
  round kept `ALIAS_EXCLUSION_RULES` because both general rewordings grew the judged alias
  table from 24.0 to ~37 terms per run. Re-measured against s74's checkpoints
  (`--pilot aliascomp`), the syntax arm itself reads **35.7** against the general arm's
  39.3 (FP +3.7, p = 0.90) — the gap is an invocation-set level, not the clause.
  **What the clause does buy is reported rather than dropped**: 0 identifier fragments
  admitted in 15 project-runs against 6 in one of fifteen. Compensating by flipping the
  alias judge's tie-break to REJECT — the branch's own "looser proposer, stricter judge"
  law — neither shrinks the table (37.7) nor keeps fragments out (13), so it is **not**
  adopted: an unnecessary change is not a defensible one.
- **`LAYERED_ENTITY_RULES` is byte-identical to s74's and is now re-grounded rather than
  rewritten.** Its enumeration carries precision (s71/s72: ~0.8 F1 without it) and its
  approve-shapes carry recall (s73: exactly 2.7 TP in each of three runs), and neither is
  corpus-shaped — an enumeration is a rubric structure and headings and lists are general
  documentation practice. The `prompt_defensibility.py` annotation for it was stale from
  s70 and is corrected in place, with the measurement as the ground.
- **The score the round exists for** (`pilot/prompt_defensibility.py --variant
  s_linker75`, no LLM calls): **3412 of 3412 authored bytes admissible — general 2866,
  se-practice 299, prior-work 247, corpus 0**, against s70's 1700 of 3645. GATE-06 also
  re-checked: none of the 67 benchmark component names appears anywhere in the authored
  text.
- **`s_linker76` — the last tuned number.** `COREFERENCE_BATCH = 10` was the only resource
  bound with no counterpart (the module also states 50 and 25) and the module's largest
  cost: **40.0 of 91.7 calls per five-project run**. `s_linker45` measured the same
  unification on the s25 base at parity over six paired runs (F1 -0.2 p = 0.52, F2 -0.0
  p = 0.91, 65.3 calls against 88.8); s76 carries that result into this line. Chosen by
  unification, not by search. **Priced and NOT adopted**
  (`../results/s76_e2e_r{1,2,3}_20260819`, three paired runs): TP **-7.0**, FP -4.7, macro
  F1 -0.7, **macro F2 -1.8** (every p at the n=3 floor), at **65 calls against 89 (-27%)**.
  `s_linker45` measured the identical unification on the s25 base at parity over six runs
  (macro F2 -0.0, p = 0.91), so this is another base-dependence result: s75's coreference
  stage sits behind three linkers that subtract from it, and a wider resolution batch
  changes which cases share a prompt. The cost is inside a 2 pp F2 budget but it is spent
  on call count and taken out of recall, so the head keeps `COREFERENCE_BATCH = 10` and
  s76 stands as the priced alternative.

- **The non-prompt surface audited on the same terms** (deterministic, no LLM calls):
  `INFLECTIONS` is general English morphology and **5 of its 9 endings never fire on any
  of the 3697 (name, sentence) pairs** — a list fitted to this benchmark would contain
  only the four that do, so its being larger than the benchmark needs is the evidence it
  was not fitted. **Superseded from `s_linker85` on**, which deletes the list rather than
  defending it (see the morphology round below); the argument above is what the finetune
  round could say while the list was still there. `CONTEXT_SENTENCES` and `ANCHOR_LIMIT` are one value, not two;
  `EXTRACTION_BATCH` is grounded in s27's passage-length effect and `JUDGE_BATCH` in the
  measured neutrality of batching. **A value chosen by unification is defensible; a value
  chosen by search is not** — which is why s76 sets the coreference batch to a number the
  module already states rather than sweeping for the best one.
- **End to end, three paired runs in one invocation set**
  (`../results/s75_e2e_r{1,2,3}_20260819`, arms s75 / s75_null / s74):

  | arm | TP | FP | macro F1 | macro F2 | calls | F1 range |
  |---|---|---|---|---|---|---|
  | `s_linker74` (control) | 182.7 | 15.0 | 94.42 | 94.46 | 90 | 2.51 |
  | `s_linker75_null` | 184.0 | 25.7 | 92.84 | 94.27 | 90 | 1.54 |
  | **`s_linker75`** | 182.7 | 22.7 | **93.59** | **94.49** | 90 | **0.84** |

  **The null in this set is loud — F1 -1.58 and FP +10.7 against the control from a
  checkpoint-namespace difference — so the null is the reference.** s75 against it:
  TP -1.3 (p = 0.60), **FP -3.0 (p = 0.30)**, **macro F1 +0.7 (p = 0.30)**, macro F2 +0.2
  (p = 0.70), composition +0.0 (p = 0.50) — QUALITY-NEUTRAL. s75 against s74: **TP ±0.0
  (p = 1.00)**, **macro F2 ±0.0 (p = 1.00)**, macro F1 -0.8 (p = 0.40), FP +7.7 (p = 0.10,
  the n=3 floor) — and the null moved FP by +10.7 against the same control, more than the
  arm did. **Removing every finetuned span costs at most 0.8 macro F1 and nothing on
  recall or F2**, against a budget of 2 pp. s75 also has the tightest run spread of the
  three arms. Caveat: arm order is s75, null, s74 and s74 leads in all three runs; this
  batch did not pay for the order reversal the prompt round used
  (`../results/nullrev_e2e_*`), so the s75-vs-null row is the one to quote.

### The elegance round (s77, s78, s79, s80) — structure priced on F2, budget 3 pp

The finetune round removed the fitted English; this one asks the same question of the
structure, at the measure the paper leads with. Four arms, each the previous plus one cut,
**one invocation set** with an in-set null (`pilot/run_elegance_e2e.sh`,
`../results/elegance_e2e_r{1,2,3}_20260819`; report in
`../results/finetune_round/README.md`).

| arm | cut | TP | FP | macro F1 | macro F2 | calls |
|---|---|---|---|---|---|---|
| `s_linker75` | control | 181.0 | 22.0 | 92.99 | 93.68 | 89 |
| `s_linker75_null` | in-set null | 181.3 | 25.0 | 92.10 | 93.31 | 89 |
| `s_linker77` | `SCANS` 3 rows → **1** (the two tight rows relocated) | 177.3 | 25.0 | 91.25 | 91.92 | 87 |
| **`s_linker78`** | **+ rubric's 4 numbered conditions → one principle** | **184.3** | **22.0** | **93.15** | **94.41** | 89 |
| `s_linker79` | + the last two options (**no gate anywhere**) | 182.0 | 39.0 | 89.66 | 92.26 | 98 |
| `s_linker80` | + the computed mention label (**nothing computed**) | 180.7 | 32.7 | 90.59 | 92.30 | 98 |

- **`s_linker78` is the head.** Against the control: **TP +3.3 (p = 0.10), FP ±0.0
  (p = 1.00), macro F1 +0.2 (p = 0.90), macro F2 +0.7 (p = 0.20)**, null at F2 −0.37. It
  removes more structure than any variant on this branch and is not worse than what it
  removes it from: **one `SCANS` row, no enumeration in any prompt, 3365 of 3365 authored
  bytes admissible.**
- **The two cuts are complements, and this is the round's methodological result.** `s78`
  contains `s77`'s cut, yet `s77` alone reads F2 −1.8. Relocating the tight scans makes the
  extraction call propose the incidental mentions they used to guarantee; the *enumerated*
  rubric rejects those (conditions (1) and (4)) and the one-principle rubric approves them.
  The enumeration was carrying precision against candidates the scans were not producing —
  which is why `s71`/`s72`, which kept the scans, measured its removal as a −0.8 F1 loss.
  **A clause is not independently priceable: two changes that each lose ground can gain it
  together when one changes what the other's population contains.**
- **The frontier, priced and not adopted.** `s_linker79` (no deterministic gate at all) is
  F2 −1.4 for **FP +17.0**, so `unique_owner` and `skip_when_named` are worth ~17 spurious
  links between them. `s_linker80` (nothing computed either) is F2 −1.4 at FP +10.7 — i.e.
  **removing the mention label on top of the gates recovers precision relative to s79**,
  where the concept round priced the label at −10.7 TP with the gates in place. **A fact's
  value depends on what else the code is doing.** Both are inside the 3 pp F2 budget; both
  are refused because `s78` is better on every measure at nearly the same simplicity.

## Notes

- The variant registry in `run_ablation.py` still lists many older
  non-retained variants from earlier branches (their modules were removed);
  only the `s_linker21*`, `s_linker20_union*`, and other still-present-module
  entries actually resolve to runnable code here.
- Default benchmarking backend is set in `.env` (`LLM_BACKEND=openai`,
  `gpt-5.4`). `.env` is untracked.

### The morphology round (s85) — the last authored word list, deleted not defended

The finetune round could only *defend* `INFLECTIONS`: nine English endings, stripped off
the sentence token, general morphology rather than benchmark vocabulary, and larger than
the benchmark needs. That is an argument, and a reviewer asking "why those nine" still has
no answer beyond "they are English". This round removes the question instead. `s_linker85`
composes `s_linker83`'s coreference judge with WordNet's lemmatizer over noun and verb
readings, **applied to both sides** — the sentence token and the name's word are the same
word when any reading of one equals any reading of the other. Tooling:
`pilot/lemma_swap_pilot.py` (E1 identity, E2 the rules not taken, E3 the ending
histogram), no LLM calls.

- **Priced at level 1 of the measurement policy, and no E2E is owed.** Both modules' own
  `_name_spans` and `_scan`, run over every (name, sentence) pair of all five projects:
  **3697 pairs compared, the spans differ on 2; partial-name candidates 109 → 110, of
  which gold 28 → 28; 0 lost (0 gold), 1 added (0 gold)**. A one-candidate,
  zero-gold delta is far inside the run-to-run band this pipeline moves in, so paying for
  paired runs would have measured model drift and reported it as a result.
- **The one disagreement is the mechanism.** bigbluebutton S49/S50, `recorded` against
  `Recording Service`. An ending list strips endings off the *sentence token*, so a name
  whose own word is already inflected — `Recording` — can never reach the sentence's
  `recorded`. **Symmetry is the entire gain, and it is the reason both sides are
  lemmatized**: the one-sided arm (lemmatize the token, compare to the name's word as
  written) reads 109 candidates and loses exactly that pair.
- **A context-sensitive lemmatizer is worse, and this is the round's transferable
  result.** spaCy `en_core_web_sm`, POS-disambiguated in the sentence, reads 103
  candidates and **loses 7 including 1 gold**: it takes `testing` in the sentence as a
  verb and lemmatizes it to `test`, while the same word inside a component's name is a
  noun and stays `testing`, so the two sides stop matching. **Making the deterministic
  layer depend on a tagger's reading of a sentence buys a defect** — the layer's job is to
  state facts about a case, and a POS tag is already a judgment.
- **Why WordNet can be trusted here and a bigger lexicon could not.** It is a lexicon with
  an identity fallback: a word it does not know comes back unchanged, so the domain tokens
  this scan actually runs on (`webrtc`, `freeswitch`) are compared by their own surface
  and nothing is invented for them. That is also why the swap cannot buy much — no
  dictionary carries the vocabulary the partial-name linker mostly sees.
- **What was refused: pruning.** Over the population the scan reaches, only four of the
  nine endings ever fire — `""` 114, `ing` 20, `s` 14, `ed` 1; `es`, `d`, `ings`, `er`,
  `ers` reach nothing (`--only E3`, reproducing the finetune round's count). **Deleting
  the five dead ones would have been fitting the list to the benchmark (GATE-07)** — the
  objection this round exists to answer, not to earn. Deleting the list is the answer;
  trimming it is the same objection, smaller.
- **Accounting.** The module carries **no authored word list at all**, and its GATE-07
  score is unchanged because a word list was never authored *prompt* text. The cost is one
  dependency — `nltk` plus the `wordnet` corpus, added to `pyproject.toml` and to
  `scripts/bootstrap-approach.sh`, since the corpus is data and not a pip dependency.
  **The trade is a nine-item hand-written list for a 155k-lemma general English resource**,
  which is smaller to defend and larger to audit; it is worth stating in the paper as a
  choice rather than a cleanup.
- The head lineage carries it: `s_linker86` and `s_linker87` are forks of `s_linker85` and
  neither declares `INFLECTIONS`.

### The typed round (s86) — one contradiction, one clause, and a closed set of verdicts

The goal was compaction that holds on **both** models. Three questions, asked in the
measurement policy's order; report `../results/typed_round/README.md`, arms
`pilot/typed_prompt_pilots.py`, statistics `pilot/typed_round_stats.py`, deterministic
screen `pilot/entity_prompt_audit.py`, invariants `pilot/test_s86_nofocus.py` (75
checks), runner `pilot/run_typed_e2e.sh`.

- **The full-name judging prompt contradicts itself, and the audit says which half
  wins.** `LAYERED_ENTITY_RULES` says a mention that says nothing further "still counts
  as a valid link"; the builder then asks for the architectural claim and says to decide
  "based on that claim". Over the recorded runs, `claim = "none"` was rejected **45/45 on
  terra (s85), 45/45 (s82), 23/23 on luna** — 105 of 105. The lenient sentence is inert,
  and the two ways of resolving the contradiction were both measured: deleting it
  (`nodead`) is neutral on both models, and honouring it (`typedlenient`, approving
  `NO_CLAIM`) costs 5.0 gold per run on terra.
- **Typed verdicts were asked of all three judges and refused at every one.** The module
  already has one typed judge (the denotation step answers `participant`/`associated`),
  so the question was whether the other rubrics could be a closed set of named verdicts
  instead of prose. Full-name: gold 151.3 → 134.7 (p = 0.10) on terra, −8.7 on luna;
  approving `NO_CLAIM` instead: −5.0 terra; restating the default as well: −8.3 terra,
  −7.0 luna. Coreference: terra F1 −1.2; **with the default restated, terra-neutral
  (F1 −0.0) and luna-fatal (FP +34.0, F1 −3.8)**. Alias: table 27.0 → 31.3 terms,
  F1 −1.4. **Mechanism, one sentence: typing a rubric deletes its default, and the
  default is what each judge's asymmetry was carrying** — the lenient gate lost recall
  (three reject types and no "approve by default" invites reaching for one), the strict
  gate lost strictness (three reject types instead of "when uncertain, reject" makes a
  merely-plausible resolution reachable). A typed rubric is also **not smaller**: +66
  chars per call at the coreference judge, +272 at the alias judge. **Had the round
  stopped at terra it would have adopted the typed coreference judge.**
- **The morphology clause stays, and the audit's attribution of its cost was wrong.**
  "count a name written with different spacing, hyphenation or compound joining as that
  name" is the only instruction admitting a candidate whose sentence writes no name at
  `ANY_CASE`. That population is 3.3 pairs/run on terra (2.3 gold) and 12.0 on luna (2.3
  gold, 9.7 spurious), which reads like a luna liability. Removing the clause removed
  none of it: luna stage spurious went **up** (10.0 → 12.0) while gold fell 5.0 (macro
  F2 −1.9); terra gold −3.3. The extractor proposes those pairs with or without a licence
  to; what the clause buys is the hyphenation cases. **A surface attribution is not a
  causal one** — s53's lesson from a new direction.
- **`s_linker86` is what holds: `s_linker85` minus `VALIDATION_FOCUS`, and nothing
  else.** The focus line asked for architectural participation and referential
  specificity; `LAYERED_ENTITY_RULES` makes the first its approve-condition and
  `STRICTER_CLAUSE` is about nothing but the second. Authored rule text **3485 → 3242 B
  (−7.0%)**, 244 B out of every full-name judging call. Stage arm, three runs a side,
  every arm judging the same extraction pass: terra TP 182.0 → 183.0 (F2 −0.0, p = 0.80),
  luna TP 174.7 → 175.7 (F1 +0.1 p = 0.90, F2 +0.3 p = 0.60). Composition risk off the
  checkpoints: 0.7 added pairs/run that a later stage also proposes, 0.0 removed pairs in
  the final link set — non-zero, so E2E was paid for; small, so at n = 3.
- **End to end, three paired runs per model in the same invocations**
  (`../results/typed_e2e_{terra,luna}_r{1,2,3}_20260821`): terra TP 184.3 against 180.7,
  FP 18.3 against 18.7, macro F1 94.65 against 94.11, macro F2 95.19 against 94.53;
  luna TP 179.0 against 177.7, macro F1 89.02 against 88.75, macro F2 91.65 against
  91.22. **QUALITY-NEUTRAL on both models on all four statistics** (every p >= 0.20),
  composition +0.1 (p = 0.50) terra and -4.6 (p = 1.00) luna, and every point estimate
  in s86's favour. 243 B of instruction removed for no measurable change.
- **`s_linker87` is the round's head: the same cut, made twice.** `COREF_RULES` opened
  by asking the resolver the question its own prompt preamble already asks -- and the
  preamble also carries the input-format contract, which is why s56 measured deleting
  the whole thing at TP -16.2. This deletes the restatement and keeps the contract, the
  untried half. It is where the bytes are: the resolver is **40 of the ~82 calls a
  five-project run makes**, so 163 B off it is ~6.5 kB of instruction per run against
  244 B x ~8.7 calls for s86's cut. Stage arm over the resolver *and* the strict judge
  behind it: terra composed TP +1.7, macro F1 -0.2 (p = 0.80), F2 +0.2; luna TP +/-0.0
  (p = 1.00), F1 +0.2, F2 +0.3. E2E, three paired runs per model against s86:
  terra TP 186.0 vs 182.3, FP 26.3 vs 34.0, macro F1 93.23 vs 92.00, F2 95.03 vs 93.90;
  luna TP 183.7 vs 181.3, FP 51.0 vs 46.7, macro F1 89.74 vs 89.44, F2 92.89 vs 92.03.
  **QUALITY-NEUTRAL on both, every p >= 0.20**, F1 and F2 favouring s87 on both, and its
  run spread the tighter of the two arms in both invocations (0.22 vs 1.55 terra, 0.71
  vs 1.62 luna); composition +2.8 (p = 0.40) terra and +4.1 (p = 0.10, at the floor)
  luna. **Authored rule text 3485 -> 3079 B (-11.7%) for two deleted restatements.**
- **The frontier, priced and refused: the strict judge's focus line.** The argument that
  removed the lenient judge's focus applies verbatim to `COREF_VALIDATION_FOCUS`, and it
  does not survive the second model: terra TP +/-0.0 (p = 1.00), macro F1 -0.3; luna
  **FP +6.3 (p = 0.10, the floor)**, F1 -0.4. **Third instance of one asymmetry: at the
  lenient gate a restatement is redundant, at the strict gate it is reinforcement.** The
  typed coreference rubric, the same rubric with its default restated, and this deletion
  all weaken the same framing, all cost luna precision (+34.0, +34.0, +6.3 FP) and all
  read neutral on terra. **A prompt cut that holds on the stricter model says nothing
  about the laxer one** -- which is the round's reason for running every arm twice.
- **`nodead` and `nofocus` are each neutral and negative together** (terra `compact`
  F1 −1.3, luna −0.45 at FP +6.0). Once the focus is gone the inert sentence stops being
  inert, because the focus was carrying the participation requirement the claim-first
  instruction leans on. **A clause is not independently priceable** — s78's result in the
  other direction — so the round removes one clause, not two, and the dead sentence
  stays, documented as dead.

### The compaction round (s88, s89) — the prompt is mostly not rules

The goal was the typed round's, sharpened: compact **every** long prompt and hold on
both models. It starts by measuring what a prompt is made of, and that measurement
redirects the whole round. Report `../results/compaction_round/README.md`; deterministic
screens `pilot/clause_audit.py` and `pilot/judge_prompt_bytes.py`; arms
`pilot/compaction_pilots.py`; statistics `pilot/compaction_round_stats.py`; composition
gate `pilot/composition_from_kept.py`; invariants `pilot/test_s88_anchors.py` (35 checks).

- **Authored rules are 5.3% of a full-name judging call and 4.3% of a resolver call.**
  What is big is repetition: **27.9%** of the judging call is anchor sentences it has
  already printed (a batch is 25 cases and several concern one component), and **25.4%**
  of the resolver call is `SENTENCES` rows for sentences the same call prints inline as a
  TARGET. Every earlier prompt round spent itself on the 5%.
- **`s_linker88` writes each component's anchors once per call** — the union of what
  every case for it in the batch would show, so no case is shown less — and points the
  later cases at the first. **No English changes at all.** Stage arm, every arm judging
  the same extraction pass: terra TP +0.7 (p = 0.80), FP -1.3 (0.70), F1 +0.4 (0.60),
  F2 +0.3 (0.50); luna TP +0.3 (1.00), FP -1.7 (0.80), F1 +0.3 (0.60), F2 +0.4 (0.50);
  judging bytes 148 199 -> 106 708 (terra, -27%) and 161 699 -> 116 122 (luna, -28%).
  Composition risk 1.3 pairs/run, so E2E is owed and was paid for.
- **Lossy and lossless compaction of the same 27% have opposite signs on the laxer
  model.** Showing later cases the FIRST case's anchor list (`anchorref`) is terra-neutral
  and luna **stage spurious +6.7 (p = 0.10)**; the union form is luna **FP -1.7**. Only 19
  of 121 same-component case pairs have equal lists, so the first form withholds about one
  anchor in five. **The invariants test caught that, not the stage arm** — a stage arm
  reports gold and spurious, and a judge shown four of its five anchors still answers.
  *Write the equivalence test before adopting a compaction, not after.*
- **Two clauses refused from the checkpoints at zero API cost**: the strict judge's
  leniency guard (terra changes 4 verdicts in 442; luna 28, **25 of them gold
  approvals**) and the alias enumeration's third item (1.3 / 0.7 aliases a run).
- **`nodenotqual` is the round's second surface-is-not-cause instance**: the denotation
  prompt's `QUALIFIED_CLAUSE` speaks about 2.0 candidates a run on both models, 0 gold,
  and deleting it costs **15 spurious partial-name links a run** (composed F1 -1.9).
  `noartifact` is the third: the enumerated ground is cited 1.0-1.7 times a run with 0
  gold, terra reads its deletion neutral, and luna loses **6.7 gold resolutions a run**.
- **The round's largest open finding, recorded and not acted on**: half the resolver's
  output is for sentences that *write* the component's name (96.0 judged cases a run on
  terra, 51.6%), which `LAYERED_COREF_RULES` opens by saying is not a coreference link.
  Fixing it means *adding* a clause, and 53 of terra's 58 approvals in that population
  are gold, so it is a separate question from compaction.
- **The resolver split on the second model.** Terra read both cuts fine
  (`notargetrows` stage gold +8.7 / spurious -2.3 for **-23.8%** of the resolver prompt,
  `nocasectx` F1 +0.1 for -8.6%). Luna refused `notargetrows` and kept `nocasectx`, so
  only the per-case range line — the smaller cut — is in the head. **A prompt cut that
  holds on one model says nothing about the other, in either direction**: this round
  refused four arms on the second model after the first accepted them.
- **`s_linker89` = s88 + the resolver's range line gone, and it is the head.**
  `pilot/test_s89_compact.py` (15 checks) pins the deletion to exactly the `CONTEXT`
  lines on all five projects, 324 B a call / 12 961 B a run. End to end, three paired
  runs a side, both arms in every invocation: **terra** TP -0.3 (p = 1.00), FP -1.7
  (1.00), F1 +0.4 (0.60), F2 +0.1 (1.00); **luna** TP +2.3 (0.80), FP +2.0 (0.70),
  F1 +0.3 (0.90), F2 +0.8 (0.70). **QUALITY-NEUTRAL on both, smallest p 0.60.**
- **The luna FP number the s88 batch flagged did not reproduce.** There it was +10.3
  (p = 0.20) against a stage read of -1.7; in the set that decides the head it is +2.0
  (p = 0.70). Different invocation sets, so this is not a trend — only that the sign is
  not reproduced where it mattered. Two prompt families compacted, **no authored rule
  text removed at all** (3079 B, unchanged from s87).

### The reading round (s91) — the two proposal stages merged, judging untouched

The head asks the document two questions in two LLM stages: the named-reference
extractor and the coreference resolver. This round asks whether they are one question
at two reference forms. Report: `../results/reading_round/README.md`; invariants:
`pilot/test_s91_reading.py` (47 checks); variant: `s_linker91`, a subclass of
`SLinker90` overriding **only** `_extract_named_mentions`, `_resolve_references`,
`_read_document`, `_prompt_reading` and `link`.

- **This is the cell the s26-s35 line never tried.** Every merge that line refused folds
  *alias discovery* or *judging* into extraction (s26/s60 alias, s29/s30/s31/s32-35
  judging, s36/s38 the two full-name judging calls). `grep -i coref` over the ledger
  returns no merge at all. The merged reading keeps the property every refused merge
  broke: **a proposer still never approves its own list.**
- **The two proposers already overlap on half their output** (recorded runs, no LLM
  calls): extractor 32.2 pairs/run at 0.905 precision, resolver 37.7 at 0.614, **17.5
  proposed by both at 0.947** — 54% of the extractor's pairs and 46% of the resolver's.
  The union discards the duplicates, so the resolver spends much of its 8 calls a project
  re-deriving pairs the extractor already had. **This reproduces the compaction round's
  largest open finding** (51.6% of resolver output is for sentences that write the name)
  from a second direction; that round left it because fixing it meant *adding* a clause,
  and merging the questions removes it by construction.
- **Anchors are local, so one 50-sentence block suffices.** Over 414 recorded
  resolutions the antecedent is a median 2 sentences back (mean 2.7, max 14), and only
  **1.0%** fall outside a fixed 50-sentence block against 21% outside a 10-sentence one.
  The reading keeps `EXTRACTION_BATCH`, which s27 already grounds, and carries a
  per-component note of the last sentence that named it for the 1% and the boundaries.
  `COREFERENCE_BATCH` becomes unused, which is the resource bound s76 could only remove
  by paying TP -7.0 for it.
- **The routing shift is 2.1 pairs per project-run.** A claim whose sentence *states* a
  name routes to the lenient judge instead of the strict one -- by the same relation the
  head uses, not by the model's choice of field. Of the resolver's 19.5 name-stating
  pairs a run, 17.5 are already on the named route via the extractor; only 2.1 actually
  change judge, carrying 0.6 gold, and the strict judge already keeps 0.6 of them.
  **Inside the recorded null floor** (FP 10.7, TP 4.8).
- **Cost:** ~16.8 LLM calls a project to ~8.8 (extraction + resolution 9.8 to ~1.8),
  with authored rule text unchanged at 3079 B -- the reading prompt composes
  `ENTITY_EXTRACTION_RULES`, `QUALIFIED_CLAUSE` and `COREF_RULES` verbatim, so GATE-07's
  accounting does not move.
- **Level 1 is done and level 2 is owed.** The invariants pin what is structurally
  unchanged; what they cannot answer is whether *one* prompt asking both questions
  proposes what *two* prompts asking them separately propose. That is a stage pilot on
  fixed recorded inputs, N samples a side, before anything composed is bought -- and
  s76 is the standing warning that cutting the resolver's call count has been refused
  once already on a neighbouring base.
- **The ladder, built before it was needed.** A merge that loses should not end the
  round, so each predicted failure mode has a rung behind it, all registered:
  **`s_linker92`** puts the head's ordering *inside* the merged call (a named section,
  then refer-backs resolved against the list that same call produced) -- the answer if
  merging costs precision, which is the direction the standing finding predicts;
  **union over k readings** is the answer if the merged reading is unstable, and it is
  the only rung with a measured effect already (a majority vote over three runs of the
  current head reads micro F1 0.913 against 0.901 for one run); **`s_linker93`** keeps
  both calls and asks the resolver only about sentences that write no name --
  **8.0 -> 4.5 resolver calls a project-run (-44%)** for **0.7 gold a run at risk**,
  and it is the model-robust rung because its prompts are the head's *byte-identically*
  when the target set is unrestricted (`pilot/test_s9293_ladder.py`, 21/21).
  s93 is also the design law applied rather than argued: which sentences write a name
  is a fact about the case, `_states_a_name` already computes it, and the compaction
  round's open finding only needed a clause because the fact was not being used.

### The reading round's verdict (s94-s100) — two proposers are not one

Six structurally different merges of the **two proposal stages** were built and
measured as stage arms, five documents, three samples a side, every arm against
control in its own invocation, on terra and the decisive arm again on luna.
All six lose, and they lose the same links: bigbluebutton's gold on the 12
sentences that reference more than one component (26 links). Control finds 19.6 of
them on terra and 19.3 on luna; no merged arm exceeds 14.7, under any batch size
(50 or 10), with or without the resolver's per-case obligation, its context table,
an explicit instruction to report several components, resampling and union, or a
conditioned gleaning pass.

**The mechanism: two proposal stages are two looks at the same sentence that
cannot see each other.** The extractor reports the component whose name is
written; the resolver, which never sees that answer, independently names a second
component; the union carries both. Rung I (`s_linker100`) proves it by failing
downward — conditioning the second look on the first's output, GraphRAG-style
gleaning, added **zero pairs in two of three samples** — because conditioning is
the opposite of blindness.

Consequences for the ledger:

* Merging two **proposers** *lowers* recall and *raises* precision (terra gold
  −1.1 to −2.2, spurious −5.5 to −12.5; luna gold −1.7, spurious −14.8), the
  inverse of the standing finding. **The standing finding is scoped to merging a
  proposer into a judge**, which is what all twelve of its variants did.
* The head's two proposal stages are load-bearing. The duplication the round
  targeted was never in the authored text — `ENTITY_EXTRACTION_RULES` and
  `COREF_RULES` are already shared constants — it was in the call count, and the
  call count buys the blindness.
* The proposers' 54%/46% overlap does not imply low marginal value: the
  non-overlap is exactly the multi-participant links.
* `s_linker93` (narrow resolver) carries a correctness defect, found before
  adoption: 12 gold links (6.2%) sit on sentences naming only *some other*
  component, unreachable by a per-sentence `_nameless` filter. Filtering per
  component instead collapses the filter. **Do not adopt.**

Full write-up and per-arm numbers: `results/reading_round/README.md`.

### The regex round (s92a–s92f) — the entity extraction pass, replaced by a scan

`ENTITY_EXTRACTION_RULES` states a surface test and defers every weighing to the gate
one stage later ("whether the mention carries an architectural claim is decided
later"). A contract with no judgement in it is a regex, and it is one this branch
already states: the whole-name row of the surface-realization relation. Report:
`../results/regex_round/README.md`; level 1 `pilot/regex_extract_audit.py`; level 2
`pilot/regex_proposer_pilots.py`; statistics `pilot/regex_round_stats.py`; invariants
`pilot/test_s92abcd_regex.py` (2316 checks, no calls).

- **Level 1 settled the proposer question at zero API cost**, off 30 recorded runs of
  the s89–s92 extractor (15 terra, 15 luna) × 5 projects. Per five-project run against
  195 gold: LLM extraction 175.3 pairs / 150.1 gold; the scan at `ANY_CASE` over the
  catalog **and the run's own aliases** 221.9 / 158.3, missing 2.4 of the extractor's
  gold and adding 10.6. **Ceiling +7.8 net gold a run.** The audit reproduces the
  branch's own name-relation table exactly (no-alias rows 172/133 and 176/137), so the
  scan is the relation the module already implements, not a new rule.
- **The alias table is the load-bearing input.** Catalog-only, the scan loses 25.6 of
  the extractor's gold. This round replaces the extraction pass, not the knowledge
  stage — a fourth measurement of the alias table's two jobs.
- **`s_linker92a` is the head of the round**: the extraction call deleted, no
  deterministic machinery added at all. Stage arm, four arms in one invocation per
  model, three runs a side, composed with the same run's untouched other two stages —
  **terra** TP 180.3 → 186.7, macro F1 91.98 → 92.43, **macro F2 93.09 → 95.12**;
  **luna** TP 180.0 → 190.7, macro F1 90.46 → 89.61, **macro F2 92.57 → 94.35**.
  **F2 up on both, F1 neutral on both** (+0.4 / −0.9, neither significant), at
  **−7.0 of ~84 calls a run** and one whole prompt constant removed. E2E not yet paid
  for: composition risk is non-zero, so this is a stage result, not the head.
- **Three variants built to repair predicted failures, all refused because the judge
  already does their job.** `s_linker92b` (do not propose a name written only inside a
  dotted identifier — 21.0 pairs a run, 0 gold): the gate rejects them itself, 21/21
  terra and 12/19 luna with no gold among the approvals, so **`QUALIFIED_CLAUSE`
  works and the design law holds even when the folded weighing's population grows
  tenfold**. `s_linker92c` (the deleted prompt's morphology clause as a second
  fidelity): +0.8 gold a run for ~25 lines. `s_linker92d` (both fidelities unioned, as
  the relation table prescribes): +1.2 gold, best bracket, most code, and every pair it
  adds is already linked by another route. **Which whole-name fidelity the scan uses is
  worth ~1 gold pair a run; whether it is a scan at all is worth ~8.**
- **The residue is `STRICTER_CLAUSE`'s population, and the repair is a thinking
  template, not a rule.** What the gate leaks is lowercased ordinary words that
  coincide with a name, and generic terms the alias stage bound. Restating the clause
  is refused (s86: a restatement at the lenient gate is redundant), so both repairs
  change only the order the reply is written in — `s_linker106`'s mechanism at a
  different question — and both render the strict branch byte for byte.
  **`s_linker92e` (quote the surface first) is REFUTED**: stage gold 152.0 → 147.7 on
  terra, FP 59.0 → 70.7 on luna. **`s_linker92f` (list the readings that surface could
  have, name the one it has, then decide) is real on terra**: macro F1 93.07, the best
  of the round, at FP 26.3 — *below* the control's 27.3, i.e. it takes the scan's whole
  added-FP cost back out; on luna it cuts the added FP (59.0 → 51.3) at 6.0 TP.
  **Echoing what you see is not deliberating about it**: e and f differ only in whether
  the model writes down the surface or weighs what it could be, and only the second
  moves anything. Nothing enumerates the readings for the model — that is what keeps it
  a template and not a clause.
- **End to end, three runs per model** (`pilot/run_regex_e2e.sh`,
  `../results/regex_e2e_{terra,luna}_r{1,2,3}_20260822`; **one arm — the control is
  byte-unchanged and its 0821 runs are reused, so this comparison is cross-set by
  decision and the in-set claim stays with the stage arm**): **terra QUALITY-NEUTRAL
  on all four statistics** — TP 178.3 → 181.0 (p = 0.40), FP 27.3 → 32.3 (0.40), macro
  F1 92.14 → 91.36 (0.30), macro F2 93.22 → 93.10 (0.80) — at **75.3 calls against
  83.2 (−9.5%)**. **luna reproduces the stage arm**: TP 177.3 → 188.7 (p = 0.10),
  **macro F2 91.45 → 93.30 (+1.9, p = 0.10)**, macro F1 −1.1 (0.40), FP 45.0 → 71.7,
  79.0 calls against 85.2.
- **The E2E is 2.1 pp of terra F2 below the stage arm, and the per-source
  decomposition says why.** By `source` over the per-variant link CSVs, terra's
  **`full_name` stage is TP +4.4 at FP +1.0** — the change is clean where it can
  reach. `partial_name` gives back 4.0 TP (mostly relabelling — `_union` tags a pair
  both linkers propose by the earlier one) and adds 4.3 FP of its own, at a stage this
  change does not touch and whose judge runs at ~0.6 precision. **A stage arm that
  composes with recorded downstream stages cannot see the downstream stage's own
  variance**; that is the fifth instance of the composition caveat and the first where
  it costs the arm rather than flattering it. On luna the effect is at the full-name
  gate itself (TP +12.6 at FP +17.0), which is the stage arm's 0.736 approve rate on
  the added pairs, end to end.
- **The false-negative decomposition inverts the branch's standing error shape**
  (`pilot/regex_fn_analysis.py`, no calls). Labelling every missed gold pair by the
  furthest it got across all three linkers, per five-project run: `fn/unproposed`
  (nothing proposed it) **4.7 → 0.0 on terra and 7.7 → 0.3 on luna**, so after the
  swap essentially **every remaining false negative reached a judge**. The standing
  finding — "95% of false negatives never reach a judge; the proposer is the
  bottleneck" — no longer describes this pipeline. **It is now the gate.** And every
  pair in that closed bucket was reachable at the tightest row measured: the
  `@ one-word` and `@ no surface` sub-rows of `fn/unproposed` are **0.0 in every
  column**, so what the LLM extractor lost was never morphology or context — it was
  sentences that literally write the name. What is left is judging and it concentrates:
  `HTML5 Server` in bigbluebutton is 3 of luna's 4 residual FNs a run and 3 of terra's
  8, declined every run by the partial-name denotation judge — the same
  sibling-confusion mechanism the error analysis found on the precision side, now on
  the recall side. **This makes a contrastive discriminator over the existing candidate
  set the live prize, not a better proposer.**
- **Why it regresses where it does** (`pilot/regex_regression_analysis.py`, no calls).
  Splitting the two arms' symmetric difference by the stage it sits at: **on terra the
  changed stage is TP +6.3 at FP +1.0 and the whole net regression is at stages the
  change does not touch** (`partial_name` −4.7 TP / +4.3 FP, a ~0.6-precision judge in
  a different invocation set). Three mechanisms are the scan's own:
  (1) **lowercase surfaces** — 6.7 of 7.7 added full-name FPs on terra and 24.6 of 27.3
  on luna are lowercase ("database"→`DB` via an alias, "common"→`Common`,
  "e2e"→`E2E`, "logic"→`Logic`). **The extractor was applying use/mention judgement at
  proposal time and nobody wrote it down**; the scan delegates it to `STRICTER_CLAUSE`
  at a gate that approves by default, which is also why the two models differ threefold
  here. **The precision cost of the swap is the implicit judgement the extraction call
  was doing.** (2) **A hard dependency on the alias table where the extractor had a
  soft one** — mediastore loses 3.0 TP a run because its sentences write `DataStorage`
  and this batch's knowledge stage did not discover that alias, while the control's
  did; the same three-term table also fires "database"→`DB` three times, so **one alias
  table costs that project 3 TP and 3 FP at once**, which is all of its −7.08 F2.
  (3) **Name nesting between siblings** — a catalog name matched inside a longer name
  of a *different* component (1.0 FP a run), which no clause in the module speaks about.
- **The `s_linker92d` refusal did not survive the E2E, and the reason is
  transferable.** Level 1 priced the fidelity axis at +0.8 gold a run on the *recorded
  control's* alias tables and the round refused the union arm on that basis. On the
  alias tables the E2E arm actually ran with it is **+2.0 gold on terra for +0.3
  non-gold pairs** — the hyphen-joined writings of space-separated alias names that
  `ANY_CASE` cannot reach. **Which spellings an alias table contains is exactly what
  varies run to run, so pricing a fidelity against one recorded table under-measures
  it.** `s_linker92d` is re-opened; `s_linker92b` and `s_linker92c` stand refused.
- **The false-negative accounting, asked directly.** Of the 44.9 gold pairs a run the
  extractor never proposed, the scan proposes 10.6; of the other 34.3, **30.1 are
  already linked** by the partial-name and coreference linkers. Against the pipeline's
  actual 14.4 false negatives a run the scan reaches **10.2 (71%)**; of the 4.2
  residue, 3.2 are already proposed by the partial-name scan (a judging question) and
  **0.5 a run is out of reach of any lexical scan at any fidelity or extent**.
  **Replacing the extractor with a scan removes 71% of the remaining false negatives
  and moves what is left off the proposer.**

### The consolidation round (s109, s110) — the two rounds composed, four refusals for free

The reading round and the regex round ran on different bases and were never measured
against each other. This round composes them and answers every question at **level 1**,
off six recorded runs of `s_linker92a`, no LLM calls spent. Report:
`../results/consolidation_round/README.md`; audit `pilot/consolidation_audit.py`;
invariants `pilot/test_s109_nesting.py` (129 checks).

- **The third blind proposer is redundant in front of a scan, so `s_linker101` is
  retired as a base.** Of the 10.3 gold a run it adds over `s_linker90`, the scan
  proposes **7.0 (68%)** for no call; the remainder is 3.3 pairs against a TP floor of
  4.8. It costs ~4 calls a project and took luna's FP from 43 to 106. `s_linker107` is
  rebased onto the head as `s_linker110`; `s_linker108` is dropped.
- **`s_linker93`'s narrowing is refused a second time, now on the scan base.** The
  filter saves 44% of resolver cases (378 sentences a run to 212) and costs **3.2 gold
  a run, 2.5 of it the defect the reading round named** — a sentence that names X and
  refers back to Y. The scan rescues the rest through the named route; it cannot rescue
  those. *Measured per (sentence, component) pair first, which read 0.7 gold: the filter
  is per sentence, and the wrong predicate flattered it fourfold.*
- **`s_linker109` is the head, and it is one refusal in `_scan`.** The partial-name
  scan proposes a pair on one word of a name; if **every** writing of that word sits
  inside a span where the sentence writes *another* component's whole name, the pair is
  that component's. terra **-5.2 FP a run**, luna **-10.8**, **0.0 gold in twelve runs of
  twelve** (six recorded before the round, six the E2E's control added after), no call
  added or removed. The refusal fires on **exactly 12 candidates in every
  run of both models** — it reads the catalog and the document and nothing sampled, so
  it is the only arm here with no run-to-run band. **No E2E owed**: 0.0 of the removed
  links are proposed by the coreference linker, so `_unlinked` frees nothing
  re-proposable (level 3, the `s_linker85` precedent).
- **The judge could not have been asked instead, and that is the point.** The
  denotation judge is target-blind by design — its case carries the expression and the
  sentence, never the component — so it answers `participant` correctly for a
  participant that is a *different* component. Showing it the target is the design law's
  own −5.5 gold refusal (s25). **The distinction is a fact about the case whose judge
  cannot be shown it, so code is not the better place for it but the only one.**
- **A discovered fact may open a case and may not close one.** The first version of the
  predicate consulted N(c) — catalog names *and* the run's aliases, as every scan here
  does — and **cost 3 gold links in one luna run**, each where that run's table bound a
  term to the sibling of the component the gold names. Scans may use the alias table
  because a scan only *admits* a case for a judge; this predicate **ends** one, so it
  rests only on given input. The alias table varies ~2.8 terms a run and would otherwise
  make one stage's sampling a silent refusal in another's.
- **The sibling confusion is not one expression judged twice.** 3.2 shared
  `(sentence, quoted claim)` groups a run holding **0.0 TP and 0.8 FP** — a chooser over
  identical cases has nothing to choose. **A contrastive discriminator as a new stage is
  priced and not built**: 8.3 FP and 2.3 FN a run sit in a sibling group another member
  owns, so its ceiling is −8.3 FP / +2.3 TP, **below the recorded FP floor of 10.7** —
  an E2E cannot see it and only a stage pilot on fixed candidates could.
- **The transferable result: who enumerates the alternatives.** Four arms asked one
  structural question — enumerate the alternatives, then commit — and agree only under
  one reading:

  | where | who enumerates | result |
  |---|---|---|
  | resolver (`s_linker106`) | the model | spurious **+6.6** |
  | resolver (`s_linker107`) | code | spurious **−10.0** |
  | lenient gate (`s_linker92e`) | nobody — echo the surface | **refuted** |
  | lenient gate (`s_linker92f`) | the model | **best terra macro F1** at FP below control |

  **The alternative set is a fact when the case contains it and a weighing when it does
  not.** Which components the sentences above name is a fact `_states_a_name` computes;
  which readings a lowercased word could have is in no table. This is the design law
  applied to the *alternative set* rather than to the rule, and it is what makes s106
  and s92f agree instead of contradict.
- **`s_linker110` holds at level 2 on both models.** s107's shortlist rebased onto the
  head, three samples x five projects, both arms in the same invocation per model:
  **terra spurious 16.9 -> 12.3 at gold 36.7 -> 36.5; luna 38.4 -> 23.1 at 36.4 -> 35.9.**
  Spurious down on both at a gold cost of 0.2 and 0.5 — luna's -15.3 above the FP floor of
  10.7, terra's -4.7 inside it. Over the resolver's own windows the list carries 1.8-4.5
  of a catalog's 6-14 components a case, which is what separates it from `s_linker102`'s
  mostly-negative checkbox. **Level 4, three paired runs a model, both arms in every
  invocation** (`../results/consolidation_e2e_{terra,luna}_r{1,2,3}_20260825`,
  `pilot/score_runs.py`): **terra QUALITY-CHANGING in the arm's favour on all four** --
  TP 181.7 -> 186.3, FP 34.0 -> 26.0, macro F1 91.92 -> **93.85**, macro F2 93.93 ->
  **95.51**, every p at the n=3 floor and every run ahead of every control run; **luna
  QUALITY-NEUTRAL on all four with every point estimate favourable** (TP +0.3 p=1.00,
  FP -5.3 p=0.60, F1 +0.6 p=0.70, F2 +0.2 p=0.90). Calls 75.0 -> 73.0 and 78.3 -> 75.7.
- **It repairs what the regex round conceded.** That round's terra E2E read macro F1
  **-0.8** while luna carried the F2 gain, because the scan bought recall and paid
  precision at `partial_name`, a stage it did not touch. These two changes are precision
  at exactly the stages the scan disturbed, and terra now reads **+1.9 F1 / +1.6 F2**.
  **bigbluebutton is ahead in six runs of six on both models** -- the project whose
  catalog carries the sibling names -- at FP 13.3 -> 10.3 (terra) and **25.7 -> 8.7**
  (luna). On luna's teammates the arm moves both ways, inside a control that itself
  ranges 72.7-86.2 F1 across three runs. **`s_linker110` is the head.**
- **`s_linker110.py` is a STANDALONE file** (2026-09-02). Being the reported arm,
  it carries the whole workflow and no linker base class -- `s_linker92`'s pipeline
  inlined, plus `s_linker92a`'s scan proposer, `s_linker109`'s nesting refusal and
  its own resolver prompt, each marked `HEAD DELTA 1/2/3`. The branch policy is one
  self-contained file per reported variant, because the paper's supplement is the
  file (`.planning/research/ARCHITECTURE.md`). `s_linker92`, `s_linker92a` and
  `s_linker109` are untouched and remain the arms this ledger records; `s_linker111`,
  `s_linker112`, `s_linker114` and `s_linker110_onecall` still subclass `SLinker110`.
  `pilot/test_s110_shortlist.py` (224 checks, no calls) re-checks the inlining block
  by block against all three sources and the composed behaviour against them over
  five projects, under an empty alias table and a populated one.
- **RQ4's floor arm is `s_linker110_onecall`, alone.** `s_linker110_noevidence` and
  `s_linker110_nocoderef` are killed -- modules, invariant tests, registrations and
  `pilot/run_noevidence_e2e.sh`, the batch runner, all gone. Nothing in the paper read
  either arm: the RQ4 floor table reads `s_linker110` and `s_linker110_onecall` only
  (`evaluation/mini-rq34/rq4_floor.py`), so it is unaffected. The recorded
  `results/noevidence_e2e_*_20260902` runs DO stay and are still read -- that batch ran
  `s_linker110` as its in-set control, and `RQ4_FLOOR_HEAD_TMPL` points the floor
  table's control at it, which is why those directories carry a dead arm's name.
  Recover the arm or its runner from git history (`e160a76f^`) if it is ever wanted.

### The uniform-schema round (s116–s119) — a reply schema carries the verdict's default

`s_linker114` expressed the three judges as one loop over three `JudgeSkill`
declarations, which put their differences in a table and made the next question
askable: two judges reply `{"validations":[{case, claim[, objection], approve}]}` and
the third replies `{"judgments":[{case, denotation, claim}]}`. Can all three write one
thing? Report: `../results/uniform_round/README.md`; level 1
`pilot/objection_audit.py`; invariants `pilot/test_uniform_schema_arms.py` (104
prompts, no calls); arms `pilot/nextgen_pilots.py --gate {lenient,sortal}` driven by
`pilot/run_uniform_round.sh`; statistics `pilot/judge_round_stats.py`.

- **No, and the sortal gate is where it fails.** `s_linker119` — that gate replying in
  the other two's key, order and boolean — is the worst arm of the round on **both**
  models: net (`3*gold − spurious`) **−9.0 terra / −16.0 luna** against the in-set null,
  gold −4.7 / −7.7, both at the n = 3 floor. It keeps 17.3 and 14.7 links where the null
  keeps 27.0 and 29.3, at precision 0.904 and 0.773. **It became a stricter judge, not a
  worse one.**
- **The typed round's mechanism, running backwards.** That round found typing a rubric
  deletes its default; this finds **untyping one imports a different default.**
  `participant`/`associated` and `approve`/`reject` are not two spellings of one
  question — the enum keeps only a positive classification, the boolean is the lenient
  gate's vocabulary, and that gate's default is the opposite of this one's. The judge
  round's polarity clause predicted it: this stream is 0.31 / 0.19 gold, the dirtiest of
  the three, so its default has to be reject-by-default.
- **The field set is nearly free to unify and buys nothing.** `objection` at the sortal
  gate (`s_linker118`): net ±0.0 / −0.3, every p = 1.00. Priced first with no calls
  (`pilot/objection_audit.py`): the strict gate's ground is 78 / 85 chars — 5 / 22 on
  approvals, 112 / 104 on rejections — and the two gates that would gain it judge 300.3
  / 305.7 cases a run, so a uniform schema is **+5.9k / +6.5k completion tokens a run
  against 28.6k, i.e. +20% / +23%. Uniformity is not a token saving**; it has to pay in
  verdicts.
- **`objection` at the lenient gate is the round's frontier, refused by the sign-flip
  rule.** terra −2.7 gold to save 5.7 spurious (0.47 FP per gold, a loss under F2); luna
  −7.3 gold to save **26.3** (3.6 per gold, net **+4.3**, precision 0.817 → 0.948). Same
  direction on both models, different exchange rate — which is `s_linker111`'s trade at a
  much better rate. **Unlike `s111` it also stabilises the gate it changes**: luna's
  lenient gate moves 22, 18, 4 links between identical samples under the null and 3, 6, 5
  under the arm (`s111` was 2–5× *less* stable). **Asking for the ground is not the same
  kind of change as asking for the readings** — one adds a field the rubric already
  licenses, the other adds a step the model resamples. The whole luna gain is teammates,
  the one project whose lenient stream is half spurious.
- **Field order is refused in both directions.** `s_linker112` (sortal takes the lenient
  order) flipped sign between models; `s_linker117` (lenient takes the sortal order) is
  **−5.0 / −6.3 net with gold down on both**, measured at the gate with 150 gold a run and
  five contributing projects. `s_linker48`'s separation — the committed quote *before* the
  verdict is what pays — now has a measurement at a gate that can carry it.
- **What is unifiable is in code, not on the wire.** `s_linker114` now declares each
  skill's verdict as `verdict_field` + `verdict_values` (`None` = the boolean contract),
  so the enum and boolean parsers are one function and each polarity is one expression.
- **A refactor's equivalence test must exercise the polarity it preserves, not only the
  default.** The first `test_s114_skills.py` stubbed `_ask` to answer nothing, so nothing
  was ever kept: its 142/142 covered the prompts and the reject path, and its kept-set
  assertion compared two empty sets — while the variant returned `approved: True` on kept
  denotation rows where the head returns `False` and corrects it downstream. With a second
  stub that answers every case, alternating the verdict: **284/284 batches and 1444 kept
  rows identical.**
- **The in-set null earned its slot in one invocation.** `skills` (s114, byte-identical
  by test) reads net **−9.3 against the control it copies** on luna's sortal gate — so
  every delta here is read against it, and the measurement policy's "no in-set null"
  guidance is about *E2E batches*, not about a five-call stage gate where the null is one
  arm of four.
- **The head does not move. `s_linker110` stands.** No arm composed, no E2E owed.
