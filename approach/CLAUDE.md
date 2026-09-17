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
- `s_linker26.py` … `s_linker126.py` — the rounds below. All `experimental=True`.
  **THE PAPER ARM IS `s_linker126`** (greedy whole-name ownership, unresolved name
  ambiguity discarded, no competitors field, NO antecedent shortlist in the resolver,
  the contract that shortlist used to assert enforced as a code predicate), promoted
  on **2026-09-16 by explicit author decision on simplicity, overriding the
  component-weighted doc-code gate's refusal rather than waiting for it to clear.**
  `s_linker124` was the head for one day and the doc-code gate refused it (dc F1 -1.67,
  dc worst F1 -2.33, both 3/3 WORSE).
  **What the override costs, stated rather than hidden.** The reported E2E batch
  (`results/greedymerge_e2e_{terra,luna}_r{1,2,3}_20260916v2`, six paired flex-tier
  runs against the in-invocation `s_linker123` control, rerun on 2026-09-16 to also
  carry the `antecedent_form_rejected` decision-logging fix so RQ3's NoCitation
  ablation reads correctly) reads doc-model link F1/F2 deltas over the in-set
  `s123gctl` control of terra `+1.06`/`+1.85` and luna `+0.72`/`-0.07`. The
  component-weighted doc-code gate (`studies/compare_arms.py`,
  `evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv`) reads a mixed signal, not a
  clearance: **terra** doc-code file F2 `+0.90` (3/3 BETTER) alongside doc-code
  worst-component F1 `-2.43` (3/3 WORSE); **luna** doc-code file F1 `-0.43` (3/3
  WORSE), every other tracked doc-code/CMR metric on both backends INSIDE NOISE.
  This supersedes the numbers an earlier same-day entry cited from the pre-rerun
  batch (terra dc F2 +2.30, luna WORSE 3/3 on dc F1/F2/worst-F1) — that batch's
  actual output links were fine, but its RQ3 logging gap meant a rerun was needed
  regardless, and the fresh independent sample reads differently: some of luna's
  clean-worse signals softened into noise, while terra picked up a clean-worse
  signal (doc-code worst-component F1) that the original batch did not show. Read
  this as this project's own noise warning demonstrated, not resolved by rerunning
  once more.
  **The promotion is on simplicity, not on this result**: it deletes a whole
  mechanism (`_named_before`, its prompt line, and the paragraph vouching for it --
  up to 198 lines and 18 327 B off the resolver a run on teammates) and adds a
  four-line predicate that reuses `_written_as`, already computed for every judging
  case. The precedent for adopting on simplicity at a measured non-win is
  `s_linker78`; the precedent for a one-model (here, one-metric) result not
  surviving a rerun is the unscoped s122 clause and the typed coreference judge.
  **`s_linker126` is now the standalone file** (2026-09-16), same policy as
  `s_linker120`/`s_linker110`/`s_linker122` before it: the `s122 -> s123 -> s125 ->
  s126` subclass chain is flattened into one file, no sibling `s_linkerNNN` import,
  verified by `pilot/test_s126.py` (27 checks) plus a byte-identity check of every
  resolved method/prompt constant against the pre-flatten subclassed form on all
  five projects. `s_linker123` and `s_linker124` remain subclasses -- neither
  cleared a gate, so neither gets the one-file treatment.
  Three head-moves before this one landed in one day and none of them cleared the
  reporting gate: `s_linker122` is link-level neutral at 21.5% less judging but
  reads WORSE on terra's doc-code metrics 3/3; `s_linker123` and the shortlist mark
  composed into `s_linker124` were each priced at the LINK-LEVEL grain
  (`score_runs.py`) and **were never put to the gate at all** -- the component-weighted
  doc-code read, needing `rq12.py`-scored E2E run directories that did not exist for
  them. **A cut that is free at the grain you are not reporting is not free**: s122
  is the standing instance, and s124's whole delta is *which components* a handful of
  links land on, which is precisely what the link-level grain cannot see.
  The ledger below is chronological, so an earlier round's "X is the head" sentence is
  true of its own date and superseded by the next round that moves it:
  s92a -> s109/s110 -> s120 -> s121 -> s122 -> s123 -> s124 -> s125 -> s126 (PAPER ARM).
- `core/`, `llm_client.py`, `pcm_parser{,_v2}.py`, `helper_v3.py`, `ilinker3.py` —
  shared runtime.
- `linkers/experimental/linker_infra.py` — the linker plumbing, **functions and one
  wrapper class, never a mixin**: `TracingLLMClient`, `ask_json` (the JSON call path),
  the checkpoint/log/metrics writers, the batching and the log's views. Eleven blocks
  that were byte-identical in 72 linker modules and that **no decision rule reads**.
  A variant keeps each method under its own name with a one-line body, so a
  self-contained file stays readable without an MRO. Do not put a prompt, a rule
  constant or a scan in here — that is the variant's own file, by policy.
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
  `pilot/test_s110_shortlist.py` (244 checks, no calls) re-checks the inlining block
  by block against all three sources and the composed behaviour against them over
  five projects, under an empty alias table and a populated one.
- **The policy is about the approach, not the plumbing** (2026-09-03). Eleven blocks
  of `s_linker110.py` were byte-identical in **72 linker modules** and none is read by
  a decision rule -- the tracing wrapper, the JSON call path (`_ask`), the checkpoint
  and log writers, the per-phase metrics, the batching, the log's views. They are now
  `linker_infra`, called from the methods that held them. **Functions, not a mixin**:
  `SLinker110.__mro__` is still `(SLinker110, object)`, so the invariant and the
  supplement claim both hold, and every prompt, rule constant, scan, judge and the
  union rule stay in the file. `link()` is **byte-identical over all five projects**
  against the pre-refactor file (`pilot/test_infra_refactor_e2e.py`, 51 checks; each
  helper against `s_linker92`'s untouched block in `pilot/test_linker_infra.py`, 107
  checks; 11/11 injected divergences caught). The eleven blocks leave
  `test_s110_shortlist.py`'s byte comparison and gain a stronger check -- that each is
  a delegation to its named helper, no longer than what it replaced -- so the suite
  goes 224 -> 235, not 224 -> 213. Report:
  `../results/linker_infra_refactor/README.md`. **No E2E owed; the head does not
  move.**
- **`MentionType` is pruned to three members in the head** (2026-09-04). `s_linker92`
  declares five; two of them cannot reach a prompt from this file. `LOWERCASE_PROSE` is
  blanked by `RETAINED_MENTION_TYPES` -- **216 of 1326 evidence lines** over the six
  consolidation E2E runs are its cases and every one printed nothing -- and `INDIRECT`
  cannot be produced at all, because `s_linker110` builds evidence bundles only for the
  full-name proposer's candidates and that proposer emits a pair only when the
  classifier's own predicate matched. Verified at level 1: **530 candidates over five
  projects x three alias tables (empty, a recorded run's, one lowercase term per
  component), 0 classified `None`**. The case `LOWERCASE_PROSE` named now returns
  `PROPER_STANDALONE`, which no caller distinguishes from it, and the unreachable
  fallback returns `None`, which `_retained_mention_label` already blanks. **No prompt
  byte moves and no E2E is owed** -- `_retained_mention_label` agrees with
  `s_linker92`'s over **every (component, sentence) pair** of all five projects under an
  empty alias table and a populated one, which is the check that replaces byte-equality
  for that one block in `test_s110_shortlist.py` (235 -> 244; the byte comparison still
  runs on every other block). This is the **only** block of the file exempt from
  "not a marked delta, therefore `s_linker92`'s bytes", and the exemption is named in
  both the module docstring and the test's `touched` set.
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

### The similarity-merge audit — one SWATTR-style proposer for both name scans

Can the full-name and partial-name scans be one string-similarity relation, SWATTR's or
a superset of it, judged once by a judge allowed to refuse? Level 1 only, no LLM calls.
Report: `../results/simmerge_audit/README.md`; audit `pilot/simmerge_audit.py` (M1–M7,
16 relation checks, both scans replayed against six recorded runs, 60/60 identical).

- **The superset exists and is refused on its exchange rate.** ArDoCo's relation
  (`splitLengthTest` then equality / levenshtein ≤ min(1, 0.9·min|w|) / Jaro-Winkler
  ≥ 0.90, against `nameParts` = the camel-split name plus the identifier) reaches
  **603 pairs carrying 183 of 195 gold** against the head union's **296 carrying 180**.
  That is **+308 cases (+104%), +12 judge calls (+86%), for +3 gold** — and **all three
  are already found by the coreference linker in 6 recorded runs of 6, on both models**.
  Marginal gold against the pipeline that exists: **zero**.
- **The fuzzy rows are empty, and the strictness axis extends cleanly.** Gold per pair by
  code-computed form over the merged set: exact whole name **0.716** (215 pairs), lemma
  word **0.321** (81), *fuzzy whole name* **0.000** (9), *fuzzy name part only* **0.010**
  (299). The relation's extra mass is 71× less gold-dense than the row it generalizes.
- **A fuzzy relation is not a superset by itself.** It loses exactly one head pair —
  bigbluebutton S49 `Recording Service` against *recorded*, the pair `s_linker85` adopted
  WordNet for. A merged proposer must be `similarity ∪ lemma-word`, so the dependency it
  was meant to replace stays.
- **SWATTR's published output is inside our candidate set: 187 of 188 pairs.** Its one
  outsider is mediastore S37 `Reencoding` against *re-encoding* (the `ANY_SPELLING` row
  s82 deleted) and is **not gold**; the head's scans propose **32 gold pairs SWATTR never
  emits**.
- **What *is* free is merging the scans, not the relation.** One stream of 296 cases
  costs the same 14 judging calls as two streams of 215 and 81 (2/1/6/4/1 a project
  either way). **What must not be merged is the judging**: the two streams are judged at
  opposite defaults — keep rate **0.756 terra / 0.922 luna** at the lenient gate against
  **0.336 / 0.303** at the target-blind one — because their gold densities differ 2.2×,
  and `s_linker119` already priced collapsing two defaults into one schema at net
  **−9.0 / −16.0**. The registered arm is therefore *one scan emitting a `form` field
  that selects the rubric inside the call*, which is the design law applied to the rubric
  rather than to the candidate. Unbuilt.

### The union round (s120) — one judge, one rule, evidence computed from the match

Can the head's two name judges be one? Not two rubrics addressed by a code fact, but
**one rule stating what a trace link is and how to read each piece of evidence**, with
every candidate in one case format. Level 1 `pilot/unijudge_audit.py`; level 2
`pilot/union_pilots.py` (both arms in one invocation, fixed candidates, alias table
pinned), statistics `pilot/union_stats.py`, error analysis `pilot/union_diff.py`, level 3
`pilot/union_composition.py`; arm `s_linker120` (`unijudge`), its thirteen iterations as
data in `union_iterations.py`, invariants `pilot/test_s120_union.py` (2593 checks),
defensibility `pilot/union_defensibility.py` (25 checks). Report:
`../results/union_round/README.md`.

- **Adopted at the stage, on both models.** terra (n=5, 25 paired units): gold **+0.2
  (p = 1.00)**, spurious **−12.6 (p = 0.000)**, net **+13.2 (p = 0.025)**, precision
  0.878 → 0.939. luna (n=3): gold **+0.3 (p = 1.00)**, spurious **−17.7 (p = 0.008)**,
  net **+18.7 (p = 0.011)**, precision 0.789 → 0.859. **Same 14 judging calls, one
  prompt instead of two.** Composition is clean on terra (0 gold pairs removed that
  nothing re-proposes) and 2 distinct pairs on luna, below the TP floor of 4.8.
- **The level-1 finding that motivated it held up.** Routing by stream is coarser than
  the facts in code: whole-name/capitalized is 101 cases at base 0.980, whole-name/
  lowercase 71 at 0.479 carrying **13.8 FP a run** under "approve by default", and
  word-only/lowercase 72 at 0.306 carrying 1.8 under the strictest gate. The union's
  whole-name row is where its win comes from on both models (terra gold +0.36/unit at
  p = 0.007 with spurious −0.60 at p = 0.002; luna −1.3 gold at **−21.0 spurious**).
- **An evidence field restrains when it is stated and misleads when it is weighted.**
  Iteration 1 stated the alternative set as a ground for rejecting: **−7.6 gold** on a
  bucket that is 0.765 gold. Iteration 6 deleted the same field from the case:
  **+26.4 spurious** at +0.7 gold. Same fact, opposite errors, a number on each side of
  the design law.
- **The company a case keeps is part of its evidence.** Three rewrites of the rule left
  luna's word-only row at 9.7–12.3 gold against a control's ~21. Grouping those cases by
  what the match computed moved it to 15.3 with no prompt change; letting the call carry
  only what its batch has (no catalog, the head's denotation contract) finished it at
  22.0. **No sentence of any rule moved that row as far as the batch boundary did.**
- **`s_linker25`'s refusal is about the question, not the target.** Blinding the
  word-only case recovered 1.4 of 6.4 gold on terra and nothing on luna; what that stream
  loses to is being asked an identity question in any of the ways a merged prompt can ask
  one. And **a row-free prompt cannot carry a per-row reply contract**: iteration 10 asked
  those cases for `denotation`, luna kept answering `approve`, and the row kept 1.0 of 81.
- **Defensible by construction, checked mechanically.** Every clause of the rule that
  states a criterion is a verbatim slice of a head constant, verified against the constant
  it came from; `LAYERED_ENTITY_RULES`' "Approve the link by default" is deliberately not
  carried and its absence is asserted. 0 of 63 catalog words, 0 dotted identifiers, 0
  document-shape words, 0 corpus-grounded sentences. The judge punches on two things: the
  architectural claim, and what an expression denotes where no name is written.
- **End to end, three paired runs a model, both arms in every invocation, arm order
  alternating by run** (`pilot/run_union_e2e.sh`,
  `../results/union_e2e_{terra,luna}_r{1,2,3}_20260911`, `pilot/score_runs.py`):

  | model | arm | TP | FP | macro F1 | macro F2 | calls |
  |---|---|---|---|---|---|---|
  | terra | `s_linker110` | 183.7 | 27.3 | 92.91 | 94.58 | 74.0 |
  | terra | **`s_linker120`** | **189.0** | **22.3** | **95.16** | **96.54** | 73.7 |
  | luna | `s_linker110` | 189.0 | 64.3 | 88.77 | 93.48 | 74.0 |
  | luna | **`s_linker120`** | **190.7** | **60.7** | **90.03** | **94.21** | 74.7 |

  **terra QUALITY-CHANGING in the arm's favour on all four** (TP +5.3, FP -5.0, F1 +2.3,
  F2 +2.0, every p at the n=3 floor, every arm run ahead of every control run on every
  statistic); **luna QUALITY-NEUTRAL with every point estimate favourable**. Same shape
  the head itself was adopted on, at the same call count. In-set through the paper's own
  engines (`studies/compare_arms.py s120 --base s110ctl`): terra BETTER 3/3 on all six
  moving metrics, luna BETTER on both doc-code metrics.
- **The union rejects more and costs less.** Two judges reject **146.0** distinct false
  positives a run against the head's three rejecting 143.7, at **6.0** true links lost
  outright against 8.7. Merging the two name judges removed a rejection the head was
  making twice; it did not trade recall for precision.
- **MediaStore is repaired, and it was the paper's one honest-failure project.** The head
  reads 0.954 doc-model F1 there; the union reads **1.000**. That project's gold hangs on
  `FileStorage` being written "the DataStorage" -- three sentences the head's coreference
  judge rejected *because they name a component explicitly*. Under one rule they are name
  cases. **The union beats ArTEMiS on all five projects at both grains**, which the head
  did not.
- **`s_linker120` IS THE HEAD (2026-09-11) and the paper reports it.** It is a
  STANDALONE file by the branch's one-file-per-reported-variant policy, checked against
  its ancestor method by method (`pilot/test_s120_standalone.py`, 85 checks: 38 methods
  byte-identical, 3 rewritten and declared, 9 replaced by the union, every rule constant
  and every other prompt identical -- including the coreference judging prompt against
  `s_linker110._prompt_validation(..., strict=True)`).
- **Two shape changes in the RQ engines**, both per-arm and both leaving `s110`'s CSVs
  byte-identical (`gen_csv_to_temp.py`): RQ3 reads **two judges** (`rq34.py` `PHASE_SETS`
  is arm-keyed), and RQ4 still prices **three forms** -- the links carry the stage label
  their scan gave them, so `FORM_SETS` splits the name phase by `source`. **After the
  union, "how many judges" and "how many forms" are no longer the same question**, which
  is exactly what the round set out to separate. No one-call floor was built on this arm;
  `rq_tables.py` drops that table and prints the absence rather than borrowing s110's.

### The label-and-rule round — what the union judge is shown, and in whose words

Four arms against `s_linker120`'s one judge, all at level 2 on fixed recorded candidates
(296 cases, 180 gold, 14 calls, identical across every arm and sample), three samples a
side, every arm in the same invocation. Report: `../results/labelrule_round/README.md`;
arms `pilot/union_pilots.py --arms union alllabels aliasmute nomention v14n`; statistics
`pilot/union_stats.py`; level-0 guard `pilot/union_render_snapshot.py`.

| arm | change | gold | spurious | net | p(net) |
|---|---|---|---|---|---|
| `alllabels` | print every `MentionType` the classifier computes | ±0.0 (p=1.00) | +1.3 | −1.3 | 0.750 |
| `aliasmute` | drop `VIA_ALIAS`, which restates the case's `writes` line | ±0.0 (p=1.00) | +1.0 | −1.0 | 0.875 |
| `v14n` | every carried criterion clause paraphrased | +0.3 (p=1.00) | +6.0 | −5.0 | 0.336 |
| `v14mention` | only `MENTION_COUNTS` paraphrased | **−5.3 (p=0.062)** | −2.3 | **−13.7** | 0.062 |
| `v14ground` | only `POSITIVE_GROUND` paraphrased | −2.7 | −2.0 | −6.0 | 0.250 |
| `v14ref` | only `QUALIFIED_CLAUSE` + `ACTS_ON` | −0.7 | +1.3 | −3.3 | 0.188 |
| `v14def` | only the definition (no GATE-07 cost) | −1.0 | −2.3 | −0.7 | 0.938 |
| `v14stricter` | only `STRICTER_CLAUSE` | −0.3 | −1.7 | +0.7 | 0.906 |
| `v14n` (in the ablation set) | all four, re-run | ±0.0 | +0.3 | −0.3 | 1.000 |
| `nomention` | print no computed label at all | −1.7 | +10.0 | **−15.0** | **0.031** |

- **Showing every label is free to build and buys nothing.** One frozenset; it adds a
  `mention=` line to **144 of 296 cases, 131 of them gold**, and moves gold **0.00 a unit
  at p = 1.000**. The census says why: the three labels left out are a function of the
  case's own `naming` row and its capitalization, so a judge holding the sentence reads
  them off the sentence (`proper case, standalone` 108 cases at 0.963 gold,
  `lowercase mention` 36 at 0.750, `indirect/unclear match` 81 at 0.321), and
  `STRICTER_CLAUSE` already says what capitalization is worth. **The retained set was
  chosen by an argument about re-derivability and the argument holds.**
- **The field as a whole is load-bearing, and that is the round's only significant
  result.** `nomention` is net −15.0 at p = 0.031, all of it at the whole-name row
  (spurious +2.00 a unit) — the 28 `lowercase, inside qualified name` cases, a 0.071-gold
  bucket that `QUALIFIED_CLAUSE` speaks to in every prompt and does not catch.
  **A clause stating the criterion is not a substitute for a fact saying this case is an
  instance of it** — the design law from the side usually taken for granted, and a fifth
  measurement of the mention label after s42/s43/s44, s80 and the concept round.
- **Naturalizing the rule loses, and a tie would also have lost.** `v14n` holds the
  definition, field lines, format contract, demand, reply, fields and every flag, and
  paraphrases only the four carried criterion clauses into plain general English. Net
  −5.0, gold +0.3 (p = 1.00) with the whole point estimate in spurious, concentrated on
  the word-only row (+1.07 a unit). **Quotation is not a style choice here**: it is what
  lets `union_defensibility.py` check each clause against the ancestor constant it was
  sliced from, so a paraphrase must be scored as authored text against GATE-07 and the
  arm had to win to be worth adopting. **First clean measurement of quotation against
  paraphrase on this branch** — v1 → v2 moved the same way and is unreadable for it,
  having moved the alternative set in the same step.
- **`VIA_ALIAS` is a genuine redundancy and is kept anyway.** All 43 alias cases print
  the same fact twice (`writes=a short form ...` and `mention=via known alias`); removing
  it is gold-neutral at net −1.0 (p = 0.875). Point estimate unfavourable, nothing bought
  — **an unnecessary change is not a defensible one** (the finetune round's rule).
- **The naturalization ablation: the clause that pays is the obvious one to reword.**
  Five arms, each `v13` with exactly one paragraph paraphrased, plus `v14n` re-run in the
  same invocation (7 arms, 294 calls). `MENTION_COUNTS` alone — "a mention that says
  nothing further still counts" rewritten as "an architectural mention is enough" — is
  **gold -5.3, net -13.7 (p = 0.062)**, the worst arm of either batch. The two read as
  synonyms and are not: **the original lowers a bar and the paraphrase restates it**, so
  the judge reimports the criterion the sentence exists to relax. It lands where that
  predicts, on the word-only row (kept 29.0 -> 22.7, gold 22.0 -> 18.3), the cases with
  the least surface to go on. `POSITIVE_GROUND` is the same effect at half size (-6.0);
  `STRICTER_CLAUSE` is the only safe one (net +0.7) because it is a **test, not a
  licence**, so restating it does not move what it licenses; `QUALIFIED_CLAUSE`+`ACTS_ON`
  is the only arm costing precision rather than recall.
- **The parts do not sum to the whole — third instance, third direction.** `v14mention`
  alone is -13.7 and `v14ground` alone -6.0, yet all four paraphrased together is **-0.3,
  every p = 1.000**. s77/s78 had two losers composing to a winner; this has two losers
  composing to a wash. **A clause is not independently priceable**, and a rule read whole
  has a register that moving every paragraph into does not equal moving each one alone.
  The same `v14n` also read -5.0 in the label batch and -0.3 here: two invocation sets,
  both real, which is why the composite was re-run in-set rather than compared across.
- **`s_linker120.py` stops being a diff against its ancestor (2026-09-13).** The
  byte-identity rule in `test_s120_standalone.py` T2 was keeping the file readable only
  as a delta: `HEAD DELTA 1/2/3` banners naming **`s_linker110`'s** derivation from
  `s_linker92` (archaeology two variants back, in the current paper supplement, with no
  key in its own docstring); seven 1:1 `linker_infra` wrappers plus `_named_spans`; a
  five-method proposer chain and a four-method label chain; and **`_writes_name`, which
  IS `_find_exact_form`** behind a `SKIP_QUALIFIED` flag the head declares `False` — one
  predicate under two names, kept apart only because the ancestor kept them apart. All 18
  are inlined and declared in `INLINED`, `SKIP_QUALIFIED` is deleted, and the file is
  **51 methods -> 33**: the proposer is one `_name_candidates`, the label one
  `_mention_label`, the judge `_judge_union`'s four blocks, evidence computed once per
  candidate instead of twice.
- **`pilot/method_dup_audit.py` — the duplicate check, generalised.** Two passes, no
  calls: structural (each method's AST, docstrings dropped and parameters renamed
  positionally, so a rename cannot hide a copy) and behavioural (every 2-argument
  method run over all 3697 (sentence, name) pairs, and any two that agree everywhere
  reported). It is validated against the bug it was written for: run on `s_linker110`
  it reports **`_find_exact_form == _writes_name`**; run on `s_linker120` it reports
  none. **A byte-identical-copy policy hides this class of duplicate by construction**
  -- the two names were kept apart only because the ancestor kept them apart -- so a
  standalone file wants a check that does not care what anything is called. `s_linker110`
  keeps its duplicate: it is a recorded control and does not move.
- **`pilot/call_chain_audit.py` — depth, not length, is what a standalone file costs a
  reader.** It prints the longest chain of self-calls from `link()` and the methods that
  are **pure hops** (one caller, and a body that belongs in it). `s_linker110` reads
  **9 hops**; `s_linker120` read 9 too until `_run_linker` (a two-entry dispatch, now a
  dict in `link()`) and `_named_before` (built in the loop that prints it) came out —
  **7 hops, 31 methods**. Not every hop is waste: a prompt builder is one f-string and
  belongs alone, and a primitive with several callers is depth worth paying for.
  `_run_validation_pass` is kept on those grounds — folding it saves one hop and makes a
  75-line method.
- **What the byte comparison claimed is now claimed by behaviour, and more strictly.**
  T6 runs the ancestor's own `_extract_named_mentions` and `_scan` beside this file's
  `_name_candidates` and compares every candidate — pair, component, source label and the
  **matched surface**, which is what a rewritten span loop moves first — over five
  projects under both alias settings. Byte identity never said what the bytes did. The
  suite goes **85 -> 133 checks**. Apply the same rule to any future standalone: pin what
  the file DOES against its ancestor, not what it reads like.
- **`pilot/union_render_snapshot.py` is the equivalence test, written before the
  refactor.** It hashes every case, every prompt and every judged decision — under a stub
  that answers both contracts and alternates the verdict, per the uniform round's lesson
  — over 5 projects x 2 alias tables x 14 iterations: **280 renderings, all identical**.
  Use it for any future change to this file that is supposed to be a refactor.
- **The head does not move. `s_linker120` stands, at `v13`, with the label set it had.**
  No E2E owed: every arm is refused or neutral-and-not-adopted.

### s121 + s122 — the judge's arrangements out, then the anchor block out (2026-09-14)

- **`s_linker121` is `s_linker120` with the judge's three call-level arrangements
  removed** and its method set cut from 49 to 34: an indirection that only forwarded a
  call, or named one step of a caller that had exactly one, is not structure.
  `pilot/test_s121_standalone.py` checks self-containment and reachability rather than a
  method-by-method copy, because the file is no longer a copy.
- **The scan's one refusal is GONE** (`s_linker109`'s nesting predicate: a one-word pair
  whose word is written only inside another component's whole name). It fires on **one
  of five projects, drops 12 pairs, and 0 of them are gold**, so it could never gain a
  link; the judge rejects those same pairs on its own in **140 of 144 case-samples across
  two models**. Removed on the simplicity argument, at a cost of +1 judging call on
  bigbluebutton. The deterministic layer now only OPENS cases — nothing in it ends one —
  which is what `SKIP_QUALIFIED`'s second clause had always claimed. The predicate is
  owned by `pilot/s121_ablations.py` so the round stays reproducible after the head moved.
- **`s_linker122` is `s_linker121` with the union judge's `anchors` evidence removed.**
  The block is up to five whole sentences a case and the largest single thing in a
  judging call: the name judging goes **199,466 -> 156,519 characters over the five
  projects (-21.5%) at the same 15 calls**. STANDALONE, by the one-file-per-reported-
  variant policy, checked against its ancestor method by method
  (`pilot/test_s122_standalone.py`, 99 checks, no calls: **28 methods byte-identical, 4
  declared changes**, every rule constant identical except the two that carry the anchors
  line, and every judging prompt on all five projects equal to the ancestor's bytes minus
  the anchors plus the clause).
- **What the anchors were holding, and the one clause that replaces them.** A case's
  `writes` line can say the sentence writes "a short form the document established for
  it" — a claim the case makes about itself, which the anchor block was the only thing in
  the call able to bear on. `pilot/anchor_diff.py` located the leak: of the 33
  sample-counts an anchor-free arm keeps and the head does not, 30 are on cases whose
  block was printed, **22 are the alias row and 18 of those are one component**. In its
  place, 125 bytes: *"Where the sentence does not write the name in full, that a surface
  can name this component is not evidence that it does here."*
- **THE SCOPE IS LOAD-BEARING, and it was learned the expensive way.** A 73-byte
  unscoped version shipped first and the E2E refused it on a sign flip (terra better,
  luna **TP -10.3** at unchanged precision). `pilot/noanchor_fn.py` named luna's loss
  exactly: **seven of the eight lost links are one teammates sentence**, a bare
  enumeration of component names, lost in **3 runs of 3**, every one `writes=whole name`.
  Unscoped, the clause fires on whole-name bare mentions and contradicts the rule's own
  `MENTION_COUNTS`. Scoped, luna's whole-name row goes **-2.07 -> -0.13 a unit** and its
  run-level gold loss **-8.7 -> -2.3**, at no cost on terra (spurious -2.3 -> -3.7).
- **A FACT CANNOT DO A WEIGHING'S JOB** — measured, and the round's transferable result.
  A clause-free repair was built and refused: reword the alias evidence so it claims only
  what the match computed ("a short form listed for it elsewhere in the document"), zero
  added prompt bytes, and it **cannot reach a whole-name case** because a whole-name case
  never renders that row. It recovers S1 on both models (luna whole-name -2.07 -> -0.47)
  and **reopens the row the anchors were holding**: terra spurious **+7.3 a run
  (p = 0.047)**, luna +7.0. Changing what the evidence says moved what the judge knows
  about one field; it did not move the threshold the block was setting. The design law —
  facts in code, weighings in the prompt — also runs the other way: a fact may not be
  asked to stand in for a weighing.
- **The stage understated the change in BOTH directions**, which is the round's
  methodological result. On the unscoped arm the stage read terra neutral and the
  composed run read FP -8.0 / F2 +0.9; the stage read luna -5.0 gold and the composed run
  read TP -10.3, roughly double. Luna's composition statistic is **+24.1 (p = 0.10)**
  against terra's +1.1. The standing caveat is that a stage arm flatters a change by
  hiding composition; here it understated one twice.
- **End to end, three paired runs a model, both arms in every invocation, arm order
  alternating by run** (`STAMP=20260914scoped pilot/run_noanchor_e2e.sh`,
  `../results/noanchor_e2e_{terra,luna}_r{1,2,3}_20260914scoped`, `pilot/score_runs.py`):

  | model | arm | TP | FP | macro F1 | macro F2 | calls | F1 range |
  |---|---|---|---|---|---|---|---|
  | terra | `s_linker121` | 182.0 | 23.7 | 93.23 | 94.52 | 72.3 | 0.98 |
  | terra | `s_linker122` | 181.7 | 23.3 | 92.90 | 94.21 | 72.7 | 1.43 |
  | luna | `s_linker121` | 182.3 | 47.0 | 90.07 | 93.03 | 75.3 | 2.71 |
  | luna | `s_linker122` | 181.0 | 46.3 | 89.50 | 92.28 | 74.0 | **1.11** |

  **QUALITY-NEUTRAL on BOTH models** — terra TP -0.3 (p = 1.00), FP -0.3 (1.00), macro
  F1 -0.3 (0.70), macro F2 -0.3 (0.50); luna TP -1.3 (0.70), FP -0.7 (1.00), macro F1
  -0.6 (0.70), macro F2 -0.7 (0.40). **The sign flip is gone**: the same cut that read
  TP -10.3 on luna with the unscoped clause reads -1.3 with the scope, and
  `pilot/noanchor_fn.py` confirms it at the pair level — teammates S1 is absent from the
  lost list in all three runs, where it was seven pairs lost in three runs of three.
  Teammates now loses 2 gold pairs and gains 4.
- **So the anchor block comes out for FREE, and that is the claim — not that removing it
  helps.** Both models' point estimates are slightly negative, so what is defensible is
  that **21.5% of the name judging can be deleted without a measurable quality cost**,
  and luna's calls fall 75.3 -> 74.0 as well. The scope also gave back the unscoped
  version's terra GAIN (FP 28.0 -> 20.0 in its own set): that gain and luna's regression
  were the same clause firing on the whole-name row, and they leave together. A round
  that buys a 21.5% cut at parity is worth more to the paper than one that buys a terra
  gain at the price of a luna regression, because the latter cannot be reported as a head.
- **The arm is also STEADIER on the model that needs it.** Luna's control swings TP
  183/176/188 at FP 37/44/60 across its three runs; the arm reads 180/182/181 at 48/42/49,
  macro F1 range **2.71 -> 1.11**. Removing an evidence field the judge had to weigh
  removed a source of run-to-run disagreement with it.
- **Composition is at the n=3 floor on both models** (+6.6 terra, +7.4 luna, p = 0.10),
  so the coreference linker is still moving pairs behind the name stage. The standing
  caveat holds: a stage arm cannot see this, and in this round the stage understated the
  unscoped change in both directions at once.
- **`s_linker122` IS THE HEAD (2026-09-14). THE PAPER ARM IS STILL `s120`, and that gap
  is deliberate.** The two questions came apart in this round and the ledger says so
  rather than smoothing it.
- **The paper's own arm engine does NOT clear s122**, and it is the read to trust here.
  `pilot/score_runs.py` is link-level (doc-model) and called the arm QUALITY-NEUTRAL on
  both models. `studies/compare_arms.py s122 --base s121ctl` adds the **doc-code
  (file-level, composed)** metrics and per-run sign agreement, scored in-set off the same
  invocations (`../results/s12{1ctl,2}_extracts`, dump slots `{terra,luna}_s12{1ctl,2}`,
  `evaluation/reports/ARM_COMPARE_s122_vs_inset.csv`):

  | metric | terra | luna |
  |---|---|---|
  | dm F1 / F2 | INSIDE NOISE | INSIDE NOISE |
  | **dc F1** | **WORSE -1.02 (3/3)** | **BETTER +0.79 (3/3)** |
  | **dc F2** | **WORSE -0.71 (3/3)** | INSIDE NOISE |
  | dm CMR% | NO CHANGE | NO CHANGE |
  | dc worst / harm F1 | INSIDE NOISE | INSIDE NOISE |

  Terra reads WORSE on both doc-code metrics with every run agreeing on the sign, which
  is the engine's strongest negative verdict, and luna reads BETTER on one. `HOWTO-
  REGENERATE-RQ.md` promotes the paper arm only "if the candidate wins", so **`DEFAULT_ARM`
  is NOT moved and `sync_paper.py` is not run**. Flipping it in the seven modules that
  declare it is one edit away if that call is made deliberately.
- **Why the two engines disagree, and why it matters.** The link-level view sees the cut
  as free; the file-level composed view sees terra lose ground. Removing the anchors does
  not change how many links are found, it changes WHICH components they land on, and the
  doc-code grain is the one the paper's own argument says to read (RQ2's size-aware
  block exists because link-level F1 is the wrong place to read an architecture-
  traceability result). **A cut that is free at the grain you are not reporting is not
  free.** That applies to picking an arm, which is exactly the note `HOWTO-REGENERATE-RQ.md`
  already makes, arriving from the other direction.
- **What s122 is therefore adopted ON:** the 21.5% reduction in name judging at
  link-level parity, as the branch's head and the base every later round forks from. What
  it is NOT adopted on: the paper's reported numbers, which stay `s120` until an arm
  clears the doc-code gate. The RQ engines key an arm to its E2E run directories
  (`rq34.py`'s arm map), so whenever that happens the s122 row is generated from
  `noanchor_e2e_{model}_r{i}_20260914scoped` and NOT from the unscoped `20260914` set,
  which measured a file that no longer exists.
- Round report, every arm and every caveat: `../results/s121_ablations/README.md`.


### The shortlist-annotation round (s124) — the antecedent list says what the system already decided (2026-09-14)

`s_linker122`'s resolver prints a per-case shortlist, `NAMED BEFORE THIS CASE: kurento
(S68), WebRTC-SFU (S68), ...`, computed by `_named_before` from `_states_a_name` — a
purely LEXICAL fact. The union judge has already ruled on every one of those mentions by
then and the module was discarding the verdicts and offering all the entries as equals.
`s_linker124` carries them: `kurento (S68, linked)` / `WebRTC-SFU (S68, named only)`.
Report: `../results/coref_annot_round/README.md`; level 1 `pilot/coref_shortlist_audit.py`;
level 2 `pilot/coref_annot_pilots.py` + `pilot/run_coref_annot.sh`; statistics
`pilot/coref_annot_stats.py`; error analysis `pilot/coref_annot_diff.py`; invariants
`pilot/test_s124.py` (29 checks, no calls).

**It is a COMPOSITION, and the two halves were priced apart.** `s_linker123` (the union
judge's `writes`/`alternatives`/`mention` merged into `written`/`competitors`, its own
round at `../results/s123_written_field/README.md`) and the shortlist mark landed on
`s_linker122` at the same time, from two sessions, and both claimed the number 123. They
touch different stages -- the judge's evidence format and the resolver's case -- and
`pilot/test_s124.py` checks they do not interact rather than asserting it: every
union-judging prompt is `s_linker123`'s byte for byte, every resolver prompt differs only
in the shortlist lines. **Neither half was measured with the other in place**, which is
the first thing an E2E of this arm settles and the reason the entry below is about the
mark alone.

- **`s_linker124` IS THE HEAD (2026-09-14). ADOPTED on the design argument at a measured
  neutral, which is a different claim from an improvement and the ledger states it as
  such.** terra macro F2 **+0.31** (p 0.50) / F1 +0.04 (p 1.00); luna macro F2 **+0.04**
  (p 1.00) / F1 −0.06 (p 1.00); TP +0.67 and FP +0.67 on both, **at the same call count
  and with no authored rule text changed**. QUALITY-NEUTRAL on both models with the F2
  point estimate favourable on both — the standard `s_linker86`, `s_linker89` and
  `s_linker110`-on-luna were adopted under.
- **The design argument, which is what the paper reports.** The shortlist was the one
  place in the module where a stage is shown an UNREFINED version of a fact the pipeline
  has already refined; every other piece of evidence any judge reads is the best the
  system knows at that point. **One fact source, stated once, read everywhere.** The mark
  is rendered in the shortlist line and not written into any constant, so GATE-07's
  accounting does not move and `prompt_defensibility.py` reads what it read for s122.
- **NOT a precision result, and the round says so in the file.** The exchange rate is
  **+0.67 gold per +0.67 spurious a run on both models**; F2's 4:1 recall weighting is
  what makes a one-for-one trade positive, and F1 reads +0.04 / −0.06. **No arm removes a
  net false positive**, and the one net link an annotation would have been designed to
  catch (teammates S131 -> `Storage`, not gold) was *added* by two arms on terra. A paper
  that motivates the mark as a precision device will be wrong.
- **Nothing reaches significance and the entry does not pretend otherwise.** At n = 3 the
  two-sided sign-flip floor is p = 0.25 and the arm's best reading is 0.50. Settling the
  terra F2 estimate needs ~13 paired samples, the luna one ~690. **The claim is
  neutrality plus a design argument; neutrality is what three samples can support.**
- **The mark is close to a constant on three of five projects**, and that is the first
  thing to say rather than the last: 83% (terra) / 79% (luna) of the 1017–1064 entries a
  run read `linked`, mediastore's shortlist is 140/140 `linked`, and **93% of the
  informative `named only` rows are teammates alone.**
- **A FACT CAN BE ENOUGH, AND A WEIGHING ABOUT IT CAN COST — refused arm 1.**
  `annot_clause` adds one sentence ("a `named only` entry is the weaker antecedent") and
  is the worse arm on both models (terra F2 −0.19, luna −0.16). The two arms move the
  resolver **32 pairs apart in opposite directions** (+17.6 proposals a run against
  −14.4) and only the unweighted one lands favourably. **The design law says where a
  weighing goes when you want one; it does not say you want one.**
- **SUPPRESSION REDIRECTS, IT DOES NOT REMOVE — refused arm 2, and the round's sharpest
  transferable result.** `annot_only` drops the refused entries instead of marking them:
  luna TP −1.0, **FP +4.0, macro F2 −0.92**, worse in 3 samples of 3 and worse-or-equal
  in 3 of 3 on terra. It proposes **36 fewer** pairs a run and its net spurious
  **doubles** — **+17 false positives added against 5 removed** — because taking an entry
  off the list does not make the resolver abstain, it makes it attach the same referring
  expression to the next component down. `s_linker109` recorded the rule from the other
  side: **a discovered fact may open a case and may not close one.** A name verdict is
  discovered — another judge's output, resampled every run. **Marking is opening;
  withholding is closing, and s124 marks.**
- **The blindness that is spent, and the blindness that is not.** `s_linker100`
  conditioned the second proposer on the first's OUTPUT LIST and added zero pairs in two
  of three samples. s124 does not: the resolver still reads every sentence, proposes
  independently, and is never told which pairs to produce. What it receives is a property
  of each candidate ANTECEDENT — evidence about a case, not a proposal to copy. **The two
  proposal stages stay blind to each other's link sets.**
- **Level 1 settled the suppressing reading for nothing, and it is why arm 3 existed at
  all.** Off the recorded checkpoints: the resolver leans on refused mentions (12.7 /
  30.7 resolutions a run at a third to a half the precision of accepted-antecedent ones)
  and **the strict coreference judge already deletes 97% / 89% of them**; `link`'s merge
  deletes the rest. **Net new links a run with a refused antecedent: 0.3 on both models,
  0.0 gold** — 35x below the recorded FP floor. Reproduced at 0.0 on the independent
  `union_e2e_*_20260911` set.
- **A STAGE PILOT WITH ZERO COMPOSITION RISK, PROVED RATHER THAN ARGUED.** `link` merges
  by pair with the earlier linker winning, so the composed set is exactly `pinned name
  links | kept coreference`; computed that way off the checkpoints it reproduces a run's
  own final CSV with symmetric difference **0** (TP 183, FP 19, F1 93.55, F2 94.77 both
  ways). Pinning the alias table AND the name-link set means only the resolver is
  resampled. **When the change is to the LAST linker, the merge is a union and the other
  half is pinned, level 2 IS level 4** — which is the measurement policy's level 3 made
  arithmetic instead of an argument.
- **`s_linker124.py` is a SUBCLASS of `s_linker123`**, not a standalone file: the one-file
  policy is for the REPORTED arm and this is not one. **Exactly two methods differ**
  (`pilot/test_s124.py`, 29 checks: every union-judging prompt is `s_linker123`'s byte for
  byte on all five projects, and every resolver prompt differs from it in the
  `NAMED BEFORE THIS CASE` lines and nothing else) and — the check that matters — **its resolver prompts are byte-identical to
  `coref_annot_pilots.Annot`, the arm that was measured, on all five projects**, with
  every non-shortlist byte equal to s122's. A run with nothing linked degrades to every
  entry reading `named only` rather than breaking. *Write the equivalence test against
  the measured arm, not just against the ancestor* — the compaction round's lesson applied
  to a promotion rather than to a compaction.
- **STILL OWED, AND THE ROUND MEASURED THE WRONG GRAIN TO CLOSE IT.** Every number above
  is `pilot/score_runs.py`, which is **LINK-LEVEL**. The read this branch actually
  promotes an arm on is `studies/compare_arms.py`, which adds the **doc-code,
  component-weighted** metrics and per-run sign agreement, and it needs `rq12.py`-scored
  E2E run directories that do not exist for s124. **s122 is the standing warning and it
  is exact**: link-level QUALITY-NEUTRAL on both models, then terra doc-code F1 −1.02 and
  F2 −0.71 with 3/3 runs agreeing on the sign, which is that engine's strongest negative
  verdict — because removing the anchors did not change how many links are found, it
  changed **which components they land on**. the mark's entire effect is +0.67 gold and +0.67
  spurious a run, i.e. *which components a handful of links land on*, so it is exposed to
  that reading in full. **s124 is the head (the base later rounds fork from); it is NOT
  the reported arm, and it is not yet a candidate for one.** The paper arm stays
  `s_linker120`.
- **THE GATE WAS RUN ON 2026-09-14 AND s124 DID NOT CLEAR IT.** Six paired E2E runs (three
  a model, in-set `s_linker123` control, order alternating) ->
  `evaluation/reports/ARM_COMPARE_s124.csv`. Terra reads **dc F1 −1.67 (3/3 WORSE)** and
  **dc worst F1 −2.33 (3/3 WORSE)**, dc harm F1 −0.50 (3/3 WEAK); luna's only BETTER is
  dm F2 +0.71, at the doc-MODEL grain, which does not promote an arm. **Every doc-code
  metric on both models has a negative mean.** This is s122's failure mode reproduced and
  amplified — s122's dc worst F1 was INSIDE NOISE, s124's is 3/3 WORSE at more than twice
  the magnitude. **The prediction written above was exact, and it came true against the
  arm.** `s_linker124` is REFUTED as a reported-arm candidate; the paper arm stays
  `s_linker120`. Round: `../results/coref_annot_round/README.md`.
- **THE E2E HARNESS DOES NOT PIN THE NAME STAGE, AND THAT IS THIS ROUND'S REAL
  METHODOLOGICAL FINDING.** The mark can only reach coreference links, so both arms run an
  *identical, independently sampled* name stage and every `full_name`/`partial_name` delta
  is noise the arm cannot have caused — summed over six runs: terra full_name +8 links at
  **+12 FP**, luna full_name +16 links at +1 FP, against a coreference row of +8/+5 and
  +13/+9. Terra's regression and luna's improvement are largely the same sampling
  phenomenon with opposite signs, both at the recorded floor (TP 4.8 / FP 10.7). **This
  does not rescue the arm** — the confound is symmetric, so it could as easily have
  flattered s124; sign agreement of 3/3 on two component-weighted metrics is what noise is
  not supposed to do; and the mark's *attributable* exchange rate is +7 gold against +14
  spurious (1 : 2), twice as bad as the pinned pilot's 1 : 1. **Any future E2E comparing a
  late-stage arm must pin the earlier stages across arms, or six runs a model minimum.**


### The antecedent-contract round (s125, s126) — the shortlist's promise, kept in code (2026-09-15)

`s_linker124`'s mark on the antecedent shortlist was refused by the doc-code gate. This
round asks what the shortlist is FOR, and ends by deleting it. Report:
`../results/coref_annot_round/README.md`; level 1 `pilot/coref_written_audit.py` and
`pilot/coref_antecedent_dump.py`; level 2 `pilot/coref_shortlist_pilots.py` and
`pilot/coref_exact_pilots.py`; invariants `pilot/test_s125.py` (33) and
`pilot/test_s126.py` (30), no calls.

- **WHY s124 FAILED, measured rather than argued.** The gold component was among the
  resolver's own `candidates` in **0** of the surviving false positives, and **10 of the
  mark's 15 extra FPs had a SINGLE candidate** -- no competition, only the decision
  whether to attach (single-candidate precision 95.5% -> 77.4%, multi-candidate 84.9% ->
  81.5%). It was over-attachment, not mis-discrimination. **The judge did not fail
  either**: it rejects 86.2% against 87.6%, an unchanged rate -- but the mark made the
  resolver propose 8.5% more, and that surplus passes the judge at **31.3% against a
  12.4% base** while being gold only 33.3% of the time. **A fixed-rate sieve cannot
  absorb a volume increase aimed at its blind spot.**
- **THE SHORTLIST NEVER BOUGHT RECALL, and both models agree.** Removing it raises GOLD
  PROPOSALS by 69% (terra 74.0 -> 125.0 a run) and 53% (luna 72.7 -> 111.0), and net gold
  does not move (15.0 -> 14.7, 15.7 -> 14.7), because `link` merges by pair and the name
  stage already holds those pairs. **The coreference stage contributes ~15 gold links a
  run whatever the resolver is shown.** That is the structural reason s124 had no headroom
  and could only move false positives, and it is why the list is a precision device.
- **The fact that DOES separate is the antecedent's own surface.** Over six recorded runs,
  surviving coreference links by what the cited antecedent sentence writes: whole name
  142 gold / 14 FP, short form 39 / 3, **whole name (qualified) 0 / 10, one word or
  off-list 0 / 7**. Seventeen false positives and zero gold in the two rows where the
  sentence does not write the name AS A NAME, holding on each model alone. The manual
  reading names them: a package path offered as a mention (`Package overview contains
  storage.api, ...` for `Storage`) and a sentence that never writes the name
  (`... bbb-html5 uses 2 "frontend" ...` for `HTML5 Server`, **six times**). Every gold
  antecedent writes the name as a free-standing noun phrase.
- **The same fact is negative in the prompt and positive in code — the design law, with a
  number on each side.** `annotexact` (mark the offending entries) reads terra F1 -0.03 /
  F2 -0.01 and luna F1 -0.33 / **F2 -0.47**; `refuse` (drop the resolution before the
  judge) reads **TP identical to the decimal on BOTH models**, FP 13.0 -> 12.3 (terra) and
  40.3 -> 38.3 (luna, -2 in 3 of 3 samples), F1 +0.11 / +0.20. **Third failure of
  annotating this shortlist** after s124's verdict mark and its weighing clause.
- **It may CLOSE a case where s124's mark may not.** `written` is GIVEN -- catalog plus
  document, identical every run -- and a judge's verdict is DISCOVERED. `s_linker109`'s
  rule, with its nesting predicate as the precedent.
- **`s_linker125`** removes the shortlist and adds nothing: terra F1 95.92 -> 95.52, luna
  90.13 -> 90.53. A sign flip, inside noise. Its one measured cost is net spurious
  (terra 1.3 -> 2.7): a resolver with no list cites antecedents further afield.
- **`s_linker126` = s125 + the predicate, and it is the HEAD by an explicit decision on
  SIMPLICITY.** terra is its best case and the composition works exactly as predicted --
  proposals 134.0 -> 171.0 with gold 77.7 -> 115.0, kept 32.3 -> 25.3, **net gold 15.0
  and net spurious 1.3** against the head's 14.7 and 2.0; composed F1 95.76 -> **95.99**,
  F2 95.78 -> **96.18**, TP 184 in 3 of 3 samples. **Luna refuses it: F2 92.94 -> 92.48,
  TP 180.0 -> 179.0.** The mechanism breaks there because luna proposes 242.0 a run
  against terra's 171.0 and keeps 62.0 -- **the predicate can reject an antecedent that
  fails a lexical test, not a plausible-looking wrong one**, and the list was constraining
  the search in a way the predicate does not replicate. **The list narrows what is
  PROPOSED; the rule filters what was CITED. They are not interchangeable**, which is the
  round's transferable result and the reason `refuse` (list AND rule) is the arm that
  holds on both models.
- **What the promotion is and is not.** Adopted as the head on the simplicity argument --
  one method, one prompt paragraph and ~18 kB a run deleted against a four-line predicate
  that reuses an existing computation, with `s_linker78` as the precedent for adopting a
  cut at a measured non-win. **NOT adopted on quality: the F2 sign flip is recorded above
  and is not a win.** The paper arm stays `s_linker120` until the doc-code gate says
  otherwise, and `s_linker126` stays a SUBCLASS until it clears -- the one-file policy is
  for the reported arm, and s124 is the standing reason not to standalone a file that a
  gate may refuse.
- **THE E2E RAN ON 2026-09-15 AND IT CANNOT PRICE THE ARM. The run set is what it
  produced, not a verdict.** Three runs a model, `s_linker126` alone
  (`pilot/run_s126_e2e.sh`, `../results/antecedentrule_e2e_{terra,luna}_r{1,2,3}_20260915`),
  the `s_linker123` control read CROSS-SET off `../results/shortlistmark_e2e_*_20260914`
  by the in-set reuse decision. Link-level (`pilot/score_runs.py`): **terra** TP 180.7 ->
  180.0, FP 13.0 -> 22.0, macro F1 95.11 -> 92.95, F2 94.80 -> 94.01, which the engine
  calls QUALITY-CHANGING against the arm; **luna** TP 178.0 -> 180.3, FP 35.3 -> 48.0,
  F1 89.91 -> 89.65, F2 91.87 -> 91.97, QUALITY-NEUTRAL. Calls 73 against 72.3 and 74.0
  against 75.3.
- **Every one of those deltas is at a stage the change cannot reach**
  (`pilot/source_stats.py`, the same permutation test restricted to each linker's own
  links). s126 touches the RESOLVER only, so `coreference` is the only row it can move.
  **terra's coreference row is IDENTICAL -- TP 14.7 -> 14.7, FP 1.0 -> 1.0, both p =
  1.00** -- while its whole FP +9.0 is `full_name` +8.0 and `partial_name` +1.0. Luna's
  coreference row is TP -0.7 (p = 0.80) at **FP -1.0 (p = 0.60)**, the only favourable
  precision movement in either model, while its FP +12.7 is `full_name` +6.0 and
  `partial_name` +7.7. **The arm reads neutral-to-slightly-favourable on its own row on
  both models and the batch reports a terra regression built entirely out of name-stage
  resampling.**
- **ALL THREE STAGE ROWS ARE QUALITY-NEUTRAL ON BOTH MODELS; ONLY THE POOLED ROW IS
  NOT.** Terra's `ALL` verdict is QUALITY-CHANGING at FP p = 0.10 while `full_name`
  (p = 0.30), `partial_name` (p = 0.70) and `coreference` (p = 1.00) are each neutral --
  the pooled statistic manufactures a verdict none of its parts carries, by summing three
  stages' independent sampling into one count. **Read the pooled row of a late-stage arm
  as a summary, never as the verdict**; `s_linker50` is the precedent this script was
  written for and s126 is its second instance.
- **This is the s124 round's methodological finding reproduced against the very next
  arm.** That entry ends "any future E2E comparing a late-stage arm must pin the earlier
  stages across arms, or six runs a model minimum"; this batch did NEITHER -- cross-set at
  n=3 -- and the runner's own note priced terra's name-stage noise at +12 FP over six runs,
  larger than the effect under test. **The read to trust for this change stays the pinned
  level-2 pilot**, by the round's own `pinned name links | kept coreference` identity
  (symmetric difference 0), which is why a last-stage arm has level 2 IS level 4.
- **What the batch is FOR, and it delivered that.** The RQ engines key an arm to its E2E
  run directories (`rq34.py`'s `ARMS`), so an s126 row needs an s126 run set to exist;
  it now does. `ARMS` still lists `s110`/`s92a`/`s120` only and **is deliberately not
  extended here** -- registering the arm is reporting it, and **the doc-code gate
  (`studies/compare_arms.py`, needing `rq12.py`-scored extracts) has NOT been run for
  s126.** The paper arm stays `s_linker120`.

### Greedy ownership with ambiguous groups discarded (s126 update, 2026-09-16)

- The earlier merged-choice judge was removed. After whole catalog names claim their
  contained words, a remaining surface that proposes several components is discarded;
  it is not sent to a second judge. The deterministic contract has 27 passing checks.
- Six paired flex-tier E2E runs compare s126 with an in-invocation s123 control: three
  each on GPT-5.6-terra and GPT-5.6-luna, all five projects, arm order alternating.
  Link-level means favour s126 on both backends: terra F1/F2 `+2.16/+1.98`, TP/FP
  `+4.0/-4.7`; luna `+0.93/+0.48`, TP/FP `+1.0/-9.3`.
- **The component-weighted reporting gate refuses promotion.** Against the same-run
  control, terra doc-code file F2 is `+2.30` with 3/3 positive signs, while file F1 and
  tail metrics have positive means with mixed signs. Luna is WORSE 3/3 on doc-code file
  F1 (`-0.86`), file F2 (`-1.59`), and worst-component F1 (`-2.32`). The harmonic metric
  is inside noise. These are backend-specific measured results, not evidence of a
  general effect. Full provenance is in `../results/s127_greedy_merge/README.md` and
  `../evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv`.
- **Conclusion:** s126 remains the experimental head because this update removes a
  field and a judge call while enforcing conservative name ownership. It is not the
  reported arm. The paper continues to use `s_linker120`.

### s126 promoted to the PAPER ARM, overriding the gate (2026-09-16, later same day)

- **The gate above was never cleared. It was overridden by explicit author decision
  on simplicity**, on the same precedent as `s_linker78`: adopting a cut at a
  measured non-win rather than at a win. This entry exists so the override is
  greppable and its numbers are the ones actually on disk, not the ones cited at
  the moment the decision was made.
- **The batch above could not stand as reported anyway.** `_validate_coref_links`
  filtered antecedent-form-rejected candidates out of `coref_links` before they ever
  reached the judge, so they never got a `judge_decisions` entry — invisible to
  `rq34.py`'s `NoCitation` reconstruction (`_judged_sets`, which reads
  `judge_decisions` alone). That is not a bookkeeping nicety: `NoCitation`'s
  reconstructed kept/rejected sets are exactly what lands in the paper's RQ3 table,
  so the published NoCitation column would have been a real result computed from an
  incomplete log. Fixed by logging a `path: antecedent_form_rejected` decision for
  each predicate-rejected candidate, the same accounting `_judge_union`'s
  ambiguous-group discard already gives its own rejections. Confirmed on one project
  alone: 23 of 57 raw coreference candidates were previously invisible to the
  ablation.
- **The fix cannot be applied retroactively to already-completed runs** (judge
  decisions are logged at LLM-call time), so the reported batch was rerun in full:
  `results/greedymerge_e2e_{terra,luna}_r{1,2,3}_20260916v2`, same shape as the
  original (six paired flex-tier runs, in-invocation `s_linker123` control, arm order
  alternating). The in-set control (`s123gctl`) was rebuilt from the SAME v2
  invocations rather than reused from the earlier batch, so the comparison stays
  same-invocation-set throughout.
- **The rerun reads a different gate picture than the one the override decision
  above was made against — stated plainly rather than smoothed over.** Doc-model
  link F1/F2 deltas over `s123gctl`: terra `+1.06`/`+1.85`, luna `+0.72`/`-0.07`
  (`evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv`, regenerated from the v2
  batch). Doc-code gate: **terra** file F2 `+0.90` (3/3 BETTER) *alongside*
  worst-component F1 `-2.43` (3/3 WORSE, not flagged in the original batch); **luna**
  file F1 `-0.43` (3/3 WORSE, same direction as before but smaller), file F2 and
  worst-component F1 both softened from 3/3 WORSE to INSIDE NOISE. Net: still no
  clearance, still a real per-backend cost, but not the *same* cost the original
  decision cited. This is the project's own "read the pooled row as a summary, never
  as the verdict" / cross-run-noise warning demonstrated on itself — a second
  independent sample does not settle the question, it relocates it.
- **The RQ4 "No knowledge" row** (`s_linker126_noknow`, registered in
  `run_ablation.py` mirroring `s_linker120_noknow`) was measured fresh
  (`results/greedymerge_noknow_e2e_{terra,luna}_r{1,2,3}_20260916v2`) rather than
  left dropped, since `rq_tables.py` omits the row rather than borrow another arm's
  and the arm actually reported needs its own sweep.
- **`s_linker126` is now the standalone file**, same one-file policy as
  `s_linker120`/`s_linker110`/`s_linker122`: the `s122 -> s123 -> s125 -> s126`
  subclass chain is flattened into one file, no sibling `s_linkerNNN` import, every
  resolved method and prompt constant verified byte-identical to the pre-flatten
  subclassed form on all five projects, `pilot/test_s126.py` (27 checks) unchanged
  and passing.
- **Every paper table (RQ1-RQ4, appendix big-tables) was regenerated and synced**
  from this v2 batch (`evaluation/mini-src/{rq12,rq34,rq34_rq2,rq_tables,csv_to_tex,sync_paper}.py`,
  all seven `DEFAULT_ARM` declarations flipped to `s126`, `check.py` confirms
  agreement). The RQ4 one-call floor stays absent, exactly as it was for `s120` --
  `s_linker126_onecall` was never built either.
- **What did not move:** the mechanism itself, the deterministic contract (27
  checks), and the design rationale in the entries above. Only the reported-arm
  pointer, the standalone-file status, and the honesty of the RQ3 NoCitation column
  changed today.
