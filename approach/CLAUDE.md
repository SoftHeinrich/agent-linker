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

Runtime files retained on this branch:

- `run_ablation.py` — lightweight ablation runner; benchmark inputs are read
  from the sibling `../ardoco` repo. `python run_ablation.py --list-variants`
  prints every runnable variant.
- `src/llm_sad_sam/linkers/experimental/s_linker21.py` — **CANONICAL** Full
  linker (`class SLinker21`, paper variant). Standalone: no inheritance from
  other linkers; all constants inlined. GATE-01: this file must stay
  byte-stable — new work subclasses it, never edits it.
- `src/llm_sad_sam/linkers/experimental/router_direct.py` — direct
  sentence->code linking infra: `CodeUnit`/`load_code_units`/`CodeIndex` (parses
  a `.acm` code model), `DirectCodeLinker` (identifier resolution ->
  file/class/package candidates), `DocCodeSentenceRouter` (per-sentence ARCH/CODE
  triage for the doc->code task), `DirectLinkJudge` (claim-before-verdict
  keep/reject judge). Reusable infra, not a linker itself.
- `src/llm_sad_sam/linkers/experimental/{agentic_router,proposer}.py` —
  promoted GTP (`GroundedTypedProposer`, grounded/context-augmented/typed
  candidate generation) + agentic router (`DocModelAgenticRouter`, per-candidate
  VALIDATE/CODE/REJECT for the doc->model task, `StrictGate`) infra. Also
  reusable infra, imported by the wiring linker below.
- `src/llm_sad_sam/linkers/experimental/s_linker21_agentrouter.py` —
  `SLinker21AgentRouter`, the agentic augmentation variant (experimental=True,
  NOT canonical). Subclasses `SLinker21`; reuses its `link()` pipeline
  unchanged as the floor, then augments with any GTP-proposed, agent-routed,
  gate-approved candidates the base pipeline missed — a gate-floor invariant
  guarantees this can never regress below s21. Measured ~1pp F1 below the
  non-agentic named+routed target (verified gold-incompleteness, not error);
  it is the bounded-autonomy increment shipped here, not a strict improvement.
  CODE-routed candidates are always exposed via `self.code_routed_candidates`;
  when an `acm_path` kwarg is supplied, judged doc->code links land in
  `self.code_links` via `DirectCodeLinker`/`DirectLinkJudge` (not yet plumbed
  by `run_ablation.py`'s current `DATASETS` dict — future work).
- `src/llm_sad_sam/linkers/experimental/s_linker24_role_orchestrator.py` —
  the sole retained S24 orchestrator. A compact multi-turn controller sees
  only general capability contracts and prior outcome counts, then orders
  three self-discovering tools: named components, references to introduced
  components, and contextual participant nouns. Completion is structural
  after all bounded tools run. Participant discovery uses uniquely owned
  runtime catalog-token overlap; a target-blind denotation pass precedes
  grounded identity review to avoid target-label bias. Final fresh
  five-project E2E: 182 TP / 8 FP / 13 FN; participant source 14 TP / 0 FP;
  macro F1/F2 96.07/95.40 and pooled 94.55/93.81. Spike 016 contains the
  hard-coding audit, failed generic-proposer evidence, and repeated
  checkpoint results.
- `src/llm_sad_sam/linkers/experimental/s_linker25.py` — `SLinker25`, the paper
  variant of the S24 design, **standalone** (no linker superclass; the former
  `SLinker24RoleOrchestrator` chain is inlined, matching the `s_linker21`
  convention for paper artifacts). Three linkers in fixed **name-evidence
  order** (full-name -> partial-name -> coreference), no LLM controller: the
  controller selected that identical order on all five projects in both
  promoted E2E runs. Link `source` tags use the paper's vocabulary
  (`full_name` / `full_name_variant` / `partial_name` / `coreference`).
  Every design choice below is measured, five runs per side on all five
  projects, permutation-tested (`pilot/ab_stats.py`); reports in
  `../results/s25_design_pilots/`:
  - **all three linkers subtract** the already-linked set (`_unlinked`), not
    just the partial-name one: -6.8 FP (p=0.01), +0.8 TP (p=0.05), and 57%
    fewer coreference judge cases;
  - **one extraction sample**, not two unioned: TP -1.2 (p=0.30), FP -1.2
    (p=0.42), so the second sample was dropped;
  - **no alias scope**: the discovered "global"/"local" grade only ever
    filtered the extraction prompt; offering every alias is +3.0 TP (p=0.01),
    +1.0 FP (p=0.59);
  - **no ambiguity map**: its only consumer was one evidence-bundle flag;
    removing the call, the field and the prompt is neutral (TP -0.2, p=1.00;
    FP +0.8, p=0.40);
  - **claim-before-verdict kept, unverified on purpose**: removing the quote
    request from the full-name and coreference judges costs 35.2 TP (p=0.01),
    while enforcing a substring check on it voids 0 verdicts in 25
    project-runs;
  - **one coreference judging pass**: a second changes nothing (TP -0.6,
    p=0.40; FP -0.8, p=0.17);
  - **defensive, adopted nothing**: judging one candidate per call instead of
    25 is neutral (TP +0.7, p=0.60; FP +0.3, p=1.00), so batching does not
    decide links; halving `CONTEXT_SENTENCES`/`ANCHOR_LIMIT` to 2 costs 2.0 TP
    (p=0.20) for no precision gain.
  `pilot/design_audit.py` sizes each of these deterministically off promoted-run
  checkpoints before any LLM call; `pilot/design_pilots.py` runs the arms;
  `pilot/test_s25_standalone.py` asserts the prompt parity, the two
  deterministic generators, and the structural invariants above.
  A second round asked of every remaining hand-coded path whether it changes a
  decision (`pilot/complexity_audit.py`, deterministic; `pilot/simplify_pilots.py`,
  arms; reports in `../results/s25_complexity_audit/`):
  - **slim evidence bundle** adopted -- the `Rationale:` line held one distinct
    value across every candidate on every project, and anchors used a second
    name primitive where the rest of the linker uses `_find_exact_form`. Both
    gone: TP +0.0 (p=1.00), FP **-2.2 (p=0.01)**;
  - **no `antecedent_via_alias`** adopted -- request sentence, 488-byte rules
    block and response field, none of it read by any gate: TP +0.6 (p=0.17),
    FP -0.8 (p=0.05);
  - **mention-type classifier kept, now with a number** -- removing the field
    costs **6.6 TP (p=0.01)**, so `MentionType`,
    `_classify_mention_typed` and `_all_occurrences_in_qualified_path` stay;
  - **spelling variants kept** -- 6 candidates over five projects, 2 accepted
    and both gold, both pairs extraction never proposed;
  - **`_inside_qualified_identifier` kept whole** -- 175 of 721 suppressed spans
    are caught by the dotted disjuncts alone, so it does not reduce to an
    adjacency test;
  - **two name primitives kept, each with a measured role** -- they disagree on
    47 of 3697 (name, sentence) pairs and flip the coreference antecedent gate
    on 0 of the promoted run's resolutions.

  Five-project E2E after each round, three runs, same model and settings
  throughout (`../results/s25_postpilot_e2e_r{1,2,3}_20260810` and
  `../results/s25_simplified_e2e_r{1,2,3}_20260810`, summaries in each r1):

  | | pre-change (N=1) | after design pilots (N=3) | after simplification |
  |---|---|---|---|
  | macro F1 | 94.2 | 94.7 +/- 0.8 | **96.4 +/- 0.4** (N=6) |
  | macro F2 | — | — | **95.4 +/- 0.6** (N=6) |
  | TP / FP | 179 / 17 | 179.7 / 9.3 | **180.8 / 4.8** |

  The knowledge module and the extractor were then tested as the same concept
  (`pilot/alias_integration_audit.py`, `pilot/alias_integration_pilots.py`,
  reports in `../results/s25_alias_integration/`). They are: **folding alias
  discovery into the extraction prompt -- one call per batch returning references
  and aliases, table accumulated and fed forward -- is exactly neutral** (TP +0.2,
  p=1.00; F2 +/-0.0, p=0.98). The separation is kept for one measured reason: the
  extraction prompt needs the table before it runs (removing `KNOWN ALIASES` costs
  5.2 TP and 2.0 F2), and a document-wide pass with a judge builds a cleaner table
  than per-batch discovery. Also measured: the alias table earns 23 gold full-name
  links, it cannot be projected from the extractor's `matched_text` (41% recall,
  28 spurious surfaces -- that field is a span, not a name), and dropping the
  alias judge is a stage-level gain (F2 +1.9, p=0.04) that reverses end-to-end
  (F1 94.57 vs 96.42 +/- 0.42) -- reverted.

  **Standing methodological finding:** four independent changes now show a
  stage-level arm pointing opposite to the composed pipeline, always on precision
  (contract filter, mention-classifier restructure, bundle de-duplication, alias
  judge). In a cascade whose stages subtract from one another, single-stage
  ablation screens candidates; it does not decide them. Always confirm with three
  five-project runs against the six-run reference band.

  A final round (`pilot/ablate_all.py`, reports in `../results/s25_ablate_all/`)
  priced the eleven decisions no earlier round had touched. Nine confirmed the
  design, six of them with a number it lacked: the spelling-variant proposer is
  worth 2.4 TP (not the 2 links a checkpoint count suggested), the uniqueness
  pass trades 2.4 recall for 10 precision, the anchor sentences are worth 2.2 TP,
  the target-blind denotation step is worth **12 FP**, the prefix rule earns 1 TP
  over exact word matching, and the unique-owner test buys 2.4 FP for free. Two
  arms -- dropping the matched span and the preceding sentence from the evidence
  bundle, both of which the case header already carries -- were stage-neutral and
  **reverted after the end-to-end check** (FP 8.3 against the 4-6 band, F1 95.2):
  repeating the evidence next to the rubric is not redundant for the model. One
  arm is left open: dropping the whole-name exclusion in the partial-name
  proposer is TP +3.2 / FP +6.2, F1 -0.5 (p=0.13) but **F2 +0.8 (p=0.01)** -- the
  only place F1 and F2 disagree with significance, and a decision about which
  measure leads the paper.

  The N=6 band pools two triples of runs of *verified-identical* code (the
  `s25_simplified_e2e_*` and `s25_micro_reverted_confirm_*` runs; identity
  confirmed at 0 predicate flips over 3697 pairs and 0 mention-label mismatches
  over 170 judged cases against the first triple's own recorded prompts). The
  first triple alone reads 96.8 +/- 0.1 and that tightness was luck — three runs
  do not estimate this pipeline's spread. Quote the six-run band.

  No project regressed in either round; jabref reaches F1 100.0 on all three
  runs. A third round inventoried every code-driven decision
  (`pilot/gate_audit.py`) and tried handing three heuristics to the LLM
  (`pilot/gate_pilots.py`, reports in `../results/s25_gate_audit/`): the
  coreference antecedent gate is worth 12 FP (F2 -1.7 without it); an LLM asked
  directly for partial-name references recovers 4.0 of the 11.0 gold links the
  prefix rule reaches (F2 -4.4); and dropping the stated-name contract filter is
  F2-positive on its own stage (+0.9, p=0.01) but quadruples false positives
  end-to-end (FP 17.3 vs 4.3 at the same recall), so it was adopted, measured,
  and reverted. **Rule from that episode: confirm an adopted arm end-to-end
  before it stays** — this stage feeds `_unlinked`, so a stage arm cannot see
  either the earlier-wins lock-in or the candidates it steals from the later,
  stricter linkers. Post-revert confirming run:
  macro F1 96.5 / F2 95.6 / FP 5 (`../results/s25_reverted_confirm_20260810`). This now exceeds the S24 orchestrator it descends from (182 TP / 8 FP /
  13 FN, macro 96.07, pooled 94.55) without S24's controller, ambiguity map,
  alias scopes or second extraction sample. **Still not re-measured:** the
  no-knowledge ablation the paper reports (`results.tex`, 5.8pp), taken with the
  ambiguity map in place.
- `src/llm_sad_sam/linkers/experimental/s_linker26.py` — `SLinker26`, standalone
  (experimental=True, **NOT promoted**). s25 with the two document-reading
  questions merged into one: a single prompt per sentence batch returns the
  references in that passage *and* any name it establishes for a component, and
  the table is fed forward to the next batch. No knowledge stage, no alias judge
  — two prompts and two LLM calls per project fewer. Every stage after the
  reading is s25's byte for byte, asserted by `pilot/test_s26_unified.py` (31
  shared methods, 9 rubrics, 7 resource bounds, both deterministic generators).
  **Measured worse:** three five-project runs give macro F1 94.27 / F2 93.47,
  TP 175.7, FP 11.0, against s25's 96.42 +/- 0.42 / 95.38 +/- 0.58, 180.8, 4.8
  (`../results/s26_unified_e2e_r{1,2,3}_20260812`, summary in r1). A batch cannot
  see a definition stated elsewhere, and nothing judges what it collects, so the
  table loses real short forms and gains `client` / `core` / `other layers`.
  Kept as the artifact that prices the alternative: before it, the two-stage
  separation could only be defended as "neutral to merge away" on one stage; now
  it is worth **2.2 F1 and 1.9 F2**.
  **Diagnosed** (`pilot/s26_diagnosis.py`, off both variants' checkpoints): s26's
  table is *bigger* (49 terms vs 27, 20 shared). The 7 terms only the global pass
  finds are short abbreviations (`ui`, `e2e`, `gae`, `webui`) -- defined once, far
  from most uses, so a 50-sentence window sees the use and not the definition. The
  29 the reading adds are descriptive phrases and generic words (`logic component`,
  `storage layer`, `client`, `core`) plus `logic.api` / `logic.core`, which the
  alias rubric explicitly forbids -- the rule is followed by a dedicated prompt and
  violated when appended to an extraction prompt. **The damage lands in a stage
  the merge does not touch:** partial-name TP 14.0 -> 7.7, because
  `_name_word_candidates` suppresses any sentence stating a whole name in N(c), so
  a bigger table narrows the strict linker's input. Full-name links go *up*
  (155.3 -> 158.3) while the two stricter linkers contribute TP 38.3 -> 33.7.
  Direct admission explains only 4 of 9 extra FP and 1 of 10 missing TP.
  **Architectural consequence worth stating in the paper:** the alias table both
  admits (full-name) and suppresses (partial-name), so table size trades recall
  between two linkers -- and no single-stage arm can see that.

  **Conclusion of the architecture exploration** (`../results/s25_architecture_exploration/README.md`,
  four variants side by side): no simpler architecture reaches s25. The two
  questions have **opposite optimal granularities** -- references degrade with the
  length of the passage read (s27: F1 98.4 at 37 sentences, 79.7 at 87, 84.1 at
  198; 50 references read where four batches read ~89), while alias definitions are
  stated once and used far away, so names need the whole document (s26 loses `ui`,
  `webui`, `e2e`, `gae`, `test driver`, `akka-apps`). One pass cannot serve both:
  s25's two stages *are* those two granularities. Merging them costs 2.2 F1, or 4.7
  F1 if the merge also drops the batching. The knowledge module is therefore
  defensible as **necessary rather than chosen**, with three implemented
  alternatives pricing the alternative. Two further attempts kept both
  granularities and changed only how the table is curated: `s_linker29` replaces
  the alias judge with a lexical grounding check (macro F1 90.07) and
  `s_linker30` folds the judging into the extraction pass (90.40). Both collapse
  MediaStore's recall to 61.3% because its four aliases carry 10 of its 30 links,
  all gold, and are the hard kind -- the document says `Database` where the model
  says `DB`, interchangeably, never defining the equivalence in any sentence. So a
  check for an establishing sentence has nothing to quote, and an in-context
  confirmation sees ordinary noun phrases. **Five simplifications, five losses:**
  the design is pinned by three independent constraints -- the granularity split,
  the need for a semantic judgment, and the need for that judgment to be lenient
  ("When uncertain, prefer APPROVE" is what keeps a third of MediaStore).
  A sixth attempt, `s_linker31`, keeps the global view, the judge's rubric verbatim,
  the same model and its leniency, and folds the review into the proposing call
  (1 + N calls): it recovers almost all the recall the others lost (TP 178.7 against
  180.8) and still loses 2.3 F1, **entirely on precision** (FP 9.7 against 4.8) --
  a proposer approves its own list. **Unifying principle: a judging step must be
  separate, semantic, lenient AND independent of what it judges.** That is the
  third measurement of independence in this workflow, after the target-blind
  denotation step (worth 12 FP) and claim-before-verdict (worth 35 TP).
  The line that came closest carries the judge's question and rubric **verbatim in
  the extraction calls** -- which already run, already receive the candidate list,
  and did not produce it, so independence is preserved at no extra call:
  `s_linker32` (any batch approves), `s_linker33` (majority), `s_linker34`
  (unanimous). All three are **1 + N calls and 2 prompts against s25's 2 + N and
  3**, and all three land **inside s25's F2 band** (95.01 / 94.97 / 95.20 against
  95.37 +/- 0.57) with **better recall** (TP 181.3 / 181.3 / 181.7 against 180.8).
  The gap is entirely precision: FP 13.0 / 13.0 / 10.7 against 4.8, so macro F1
  reads 94.86 / 94.75 / 95.11 against 96.42 +/- 0.42 -- outside, though one s34 run
  hit 96.04. The approved lists show why: a review carried by a pass whose main job
  is extraction keeps generic phrases (`back end`, `front-end`, `datastore`) that a
  dedicated judge rejects. **Fifth property, and the one that resists: a judging
  step also needs undivided attention.** Threshold tuning was stopped at three
  values on purpose -- fitting a small integer until F1 lands in band is the
  benchmark-fitting this work flags elsewhere as a validity threat.
  `s_linker35` asked the carried review *before* the document, the last available
  lever on the diagnosed cause: precision improves to the best of the carried line
  (FP 8.3) and recall collapses (TP 162.0, MediaStore back to 61.3%), because a
  context-blind review rejects interchangeable aliases. That closes a trade-off
  surface -- carried-after over-approves, carried-before over-rejects -- and shows
  why the dedicated call wins: it is the only arrangement that is simultaneously
  **undivided**, **context-free** (its prompt carries the component list and the
  mappings, not the document, so it judges the phrase not the passage) and
  **lenient**. `s_linker2{6,7,8,9}.py` and `s_linker3{0,1,2,3,4,5}.py` are all
  experimental, none promoted; s25 stands unchanged. **Eleven variants map the
  space: the simplification is real (s34 is one prompt and one call cheaper, matches
  F2, beats recall) but none matched s25 on F1.** Threshold and ordering permutation
  was stopped on purpose -- fitting small integers and word order until F1 lands in
  band is the benchmark-fitting flagged elsewhere here as a validity threat.
  A twelfth variant, `s_linker36`, leaves the knowledge side alone and halves the
  *largest* stage instead: the full-name judge asks its relevance and uniqueness
  criteria in one call rather than two (M calls, not 2M), both rubrics verbatim.
  It is the cheapest point found -- **79 LLM calls per five-project run against
  s25's 89** -- with F2 statistically identical (95.35 against 95.37 +/- 0.57),
  recall higher (TP 181.7 against 180.8) and F1 short by 0.90 (95.52 against
  96.42 +/- 0.42).
  `s_linker37` adds a committed quote per criterion inside that one call and was run
  **six times** for a direct test against s25's six runs (exact two-sided permutation,
  924 splits): **F2 -- s37 +0.08, p = 0.810 (indistinguishable), TP 182.2 against
  180.8, 79 calls against 89; F1 -- s37 -0.77, p = 0.017 (significantly worse),
  FP 8.8 against 4.8.** That is the exploration's final answer: **F2 parity at 11%
  lower cost is available; F1 parity is not.**
  **POSITIVE RESULT -- `s_linker38`.** Independence is what buys precision, and it
  does not need two different prompts. s38 runs **one** link-judging prompt carrying
  both criteria, samples it **twice**, and ANDs the verdicts. Six runs each side,
  exact permutation test over 924 splits: **F1 -0.47 (p = 0.071), F2 +0.20
  (p = 0.457), TP +1.17 (p = 0.429), FP +2.33 (p = 0.175) -- nothing significant,
  F1 included**, at 88 calls against 89. The relevance/uniqueness *pass* distinction
  leaves the architecture: one judge, one rubric, one prompt, a self-agreement gate.
  This is the simplest design measured to hold s25's performance.
  **CORRECTION, from auditing s38's own six runs** (`pilot/s38_audit.py`, report in
  `../results/s38_audit/`): the self-agreement gate is **not** what holds that
  parity. The two samples split on **1.0 of 174.7 judged candidates per run
  (0.6%)**, and ANDing rather than ORing those splits is worth 0.7 false positives
  against 0.3 true positives. s38 is therefore `s_linker36` plus one redundant call
  per judging batch, and the 0.43 F1 between them is noise. Both the module
  docstring and the inline comment claiming self-agreement "is where the precision
  comes from" are corrected in place.
  **AND THE MERGE DIRECTION IS NOW CLOSED, WITH A MECHANISM.** `s_linker36` (the same
  merged prompt, asked once) was taken to **six runs** against s25's six:
  **macro F1 -0.7 (p = 0.01), FP +3.5 (p = 0.01)**, TP +0.8 (p = 0.44), macro F2
  +0.0 (p = 1.00), at 79 calls against 89. So one call does not hold F1, and the
  three variants line up on precision exactly as a single mechanism predicts:
  FP 4.8 (s25, two focused calls) -> 7.2 (s38, one prompt sampled twice) -> 8.3
  (s36, one prompt once). Reading the judges' own verdicts explains it:
  **s25's two focused calls disagree with each other on 4.7 of 172.3 candidates
  (2.7%) -- 1.0 gold, 3.7 not -- while s38's two samples of the merged prompt
  disagree on 1.0 of 174.7 (0.6%), and both arrangements reject the same 11.3
  unanimously.** The unanimous rejections survive the merge; the disagreements do
  not, and the 3.7 false positives that s25's disagreements remove are the 3.5 it
  leads s36 by. **Independence has to come from asking a different question, not
  from resampling the same one** -- two focuses are 4.5x more independent than two
  samples. That is the sixth and sharpest measurement of independence in this
  workflow (after target-blind denotation at 12 FP, claim-before-verdict at 35 TP,
  a proposer reviewing its own aliases at 4.9 FP) and it retires the whole s32-s38
  line: the judging arrangement stays as s25 has it.
  Other results from the same audit, all deterministic off s38's checkpoints and
  traces:
  - **the two criteria are not one question, but they are worth less than the
    two-pass form suggested**: relevance and uniqueness disagree on 3.2 of 400.0
    answers per run (0.8%), always one-directionally, and joined to gold those are
    2.7 false positives against 0.5 gold. Kept (169 bytes of prompt for 2.7 FP);
  - **the five-value mention label collapses to three.** Its case grading changes
    no verdict -- proper-case standalone 96.9% approved (107.0 cases/run), lowercase
    100.0% (25.2), indirect 100.0% (1.7) -- while the two values that separate are
    *how* the name is present: via a discovered alias 82.8% (33.0) and only inside a
    qualified identifier 57.4% (7.8, gold rate 25.5%). Adopted in s42, together with
    dropping the residual value in favour of omitting the field, which also deletes
    the workflow's last case-sensitivity rule;
  - **the full-name judge rejects 12.4 of 174.7 candidates and 78% of those
    rejections are on one project** (zero on MediaStore). This is why every judging
    arrangement from s32 to s38 lands within noise: there is almost nothing to
    reject on four of five projects, so **the arrangement of judging is not what
    makes the approach work; admission is.** Dropping the judge outright reads
    TP +1.5 / FP +10.9;
  - **coreference's whole-document scope cannot be narrowed.** It is 46.3 of 101.5
    calls/run and 64% of what it reports is already linked, but asking only about
    sentences with no link yet would lose 14.5 of 30.0 coreference links per run,
    13.2 of them gold -- a sentence can state one component's name and refer back to
    another, so the subtraction belongs at the pair level, where it already is;
  - **three judging protocols remain, and the asymmetry is principled**: the
    full-name gate approves by default, the coreference gate rejects when uncertain.
    Opposite defaults follow from opposite evidence (the name is stated / the name is
    absent), and the coreference gate's strictness is priced at 12 FP. Making every
    judge sample twice would cost +12.8 calls/run for what A1 shows is nothing;
  - **one name primitive, verified** (`_find_exact_form`, 8 call sites;
    `has_standalone_mention` no longer referenced anywhere in the code). The s25
    docstring still claimed two "each with a measured role" and is corrected;
  - **not proposed, deliberately**: the `source=` field of the evidence line is 99%
    one value (396.0 of 400.0 renderings per run), the same argument that removed the
    constant `Rationale:` line -- but two earlier removals of evidence content
    (`matched_span`, `preceding_text`) were stage-neutral and pipeline-negative
    (FP 8.3 against the 4-6 band). Repeating evidence next to the rubric is not
    redundant for this model, so this one is left alone.
- `src/llm_sad_sam/linkers/experimental/s_linker42.py` — `SLinker42`, standalone
  (experimental=True). `s_linker36`'s single full-name judging call plus the
  three-value mention label; everything else is s25's, asserted by
  `pilot/test_s42_threevalue.py` (36 shared methods, 10 rubrics, 7 resource bounds,
  both deterministic generators, and every one of 3697 (name, sentence) pairs
  relabelling exactly as intended). Three runs a side against s36 in
  `../results/s42_threevalue_e2e_r{1,2,3}_20260812` (which also bring s36 to n=6):
  the label collapse is free **on this base** — TP +/-0.0 (p = 1.00), FP +1.7
  (p = 0.30), F1 -0.1 (p = 0.50), F2 -0.1 (p = 0.70) — but the base itself loses F1
  significantly, and lifting the label change onto s25 (`s_linker43`, then the half of
  it the traces actually support, `s_linker44`) failed both times. Read this row as a
  base-dependence result, not as a licence: n=3 on a base that is already 0.7 F1 down
  cannot see a 0.9 F1 effect.
- `src/llm_sad_sam/linkers/experimental/s_linker43.py` — `SLinker43`, standalone
  (experimental=True). **s25 with exactly one change: the three-value mention label.**
  `pilot/test_s43_threevalue.py` asserts the single change, that
  `_validate_with_evidence` and both judging prompts are s25's byte for byte, and the
  same 3697-pair relabelling. **REJECTED.** Three runs a side, paired with s25 in the
  same invocations (`../results/s43_threevalue_e2e_r{1,2,3}_20260812`): TP -1.7
  (p = 0.40), FP +2.7 (p = 0.30), **macro F1 -1.3 and macro F2 -1.3, both at the n=3
  p-floor of 0.10** — the most extreme of all ten labellings on both scores. So a
  label change that is free on the merged-judging base (s42 vs s36: TP +/-0.0,
  p = 1.00; F1 -0.1, p = 0.50) costs 1.3 F1 on s25. **Sixth instance of an arm
  measured neutral in one composition coming out negative in another, and the first
  where both compositions were end-to-end.** Equal approval rates per label value are
  a screen, not a proof: they aggregate over cases, and rewriting the field changes
  the prompt for 132 cases per run.
- `src/llm_sad_sam/linkers/experimental/s_linker44.py` — `SLinker44`, standalone
  (experimental=True). Splits the pair s43 bundled. s43 merged the two stated-name
  values **and** omitted the field for the residual value; only the first is supported
  by the traces (96.9% vs 100.0% approval), while the second removes evidence content,
  which this workflow has twice measured as pipeline-negative. s44 merges only the case
  grading — five values become four, the field is always present, `_validate_with_evidence`,
  `_build_evidence_bundle`, `_format_evidence` and `_prompt_validation` all byte-identical
  to s25's — which is the change that deletes the workflow's last case-sensitivity rule
  (`matched == comp_name`). `pilot/test_s44_nocasegrade.py` pins the difference to the
  enum and the classifier over 3697 (name, sentence) pairs. **REJECTED at n=6, after
  reading neutral at n=3 — and that reversal is the finding.** Six runs a side, paired
  inside the same invocations (`../results/s44_nocasegrade_e2e_r{1..6}_20260812`):
  TP +0.3 (p = 0.87), FP +1.2 (p = 0.55), **macro F1 -0.9 (p = 0.05)**, macro F2 -0.5
  (p = 0.21). The first three of those runs alone read TP +2.0, FP -0.7, macro F1 -0.0
  (**p = 1.00**), macro F2 +0.3, with s44 holding the tighter within-arm spread; runs
  4-6 put s44 at 94.5-94.9 against s25's 96.4-96.5 and inverted the verdict.
  **Three runs of this pipeline can manufacture a neutral as easily as a regression;
  six paired runs is the bar.** Per project the loss is jabref -3.94 F1 (FP 0.2 -> 1.7
  on a 13-sentence, 18-link document, and jabref carries the highest share of
  merged-value pairs, 20 of 78) and teastore -1.57 (TP 27.0 -> 26.2), against teammates
  **+1.89** (FP 6.7 -> 4.0). Macro averaging weights the smallest document like the
  largest.
  **Standing rule, seventh instance and first from a trace-based screen:** the audit's
  readings were right as readings — the second sample really splits on 0.6% of cases,
  the label's three values really are approved at 96.9 / 100.0 / 100.0% — and neither
  licensed a removal. Equal aggregate behaviour per label value does not make a
  distinction inert, because rewriting the field changes the prompt for every case that
  carries it (132 per run). A trace-derived equivalence is a hypothesis; the test is
  six paired runs.
  **Net result of the whole audit: twenty variants (s26-s46) — twelve on the knowledge
  side, three on judging arrangement, three on the mention label, two on batching and the
  alias table's dual role — and no element of the workflow has been removed without a
  measured cost. The one change that holds (`s_linker45`) removes no element at all: it
  retires a tuned constant and a quarter of the calls.** For the paper that is a stronger
  claim than any single ablation table, and it is the answer to "is anything here
  accreted?"
- `src/llm_sad_sam/linkers/experimental/s_linker45.py` — `SLinker45`, standalone
  (experimental=True). **s25 with one change: `COREFERENCE_BATCH = JUDGE_BATCH`.** The
  coreference batch was the only resource bound never ablated and the only one whose
  value has no counterpart elsewhere, and it is what makes coreference resolution the
  largest call consumer — **46.3 of 101.5 calls per five-project run for 30.0 links**
  (s38 audit A3/A7). Setting it to the judges' batch leaves the workflow with **two**
  batch constants instead of three and about **74 calls instead of 101** (-27%), with no
  stage removed and no prompt reworded. The value is chosen by unification, not search:
  one value, tested once. Counter-evidence it must survive: `s_linker27` showed
  reference reading degrades with passage length (F1 98.4 at 37 sentences, 79.7 at 87),
  though over a far larger range than 10 -> 25.
  **CONFIRMED AT PARITY — this is the one simplification that holds.** Six paired runs
  (`../results/s4546_e2e_r{1..6}_20260812`): **TP +0.8 (p = 0.56), FP +2.2 (p = 0.34),
  macro F1 -0.2 (p = 0.52), macro F2 -0.0 (p = 0.91)**, at **65.3 calls per run against
  88.8 (-26%)**, with coreference resolution falling from 40.0 calls to 17.0. Nothing is
  within reach of significance on any measure and recall is the higher of the two. No
  project collapses: mediastore +0.26 F1, jabref +0.43, teammates -0.26,
  bigbluebutton -0.46, teastore -0.96. So `s_linker27`'s passage-length effect does not
  reach from 10 to 25 sentences for *this* question — resolving a back-reference needs
  the sentences either side, not a short window — and the workflow can state two batch
  constants instead of three.
- `src/llm_sad_sam/linkers/experimental/s_linker46.py` — `SLinker46`, standalone
  (experimental=True). **s25 with one change: the alias table no longer suppresses
  partial-name candidates.** The table's dual role — it *admits* full-name candidates
  (29 alias-only links, 23 gold) and *suppresses* partial-name ones, because
  `_name_word_candidates` treats every discovered alias as a whole name — was surfaced
  by the s26 diagnosis as an architectural liability no single-stage arm can see, and it
  means table size trades recall between two linkers. Here the exclusion consults the
  model name only; every other consumer reads the table unchanged. Sized off six runs'
  real tables before running: **+16 candidates over the five projects (59 -> 75), 3.8 per
  run gold**, concentrated on teammates (28 -> 41). The same removal was tried once, on
  `s_linker26`'s much larger merged table (`s_linker28`, recovered nothing); on a table
  this size with this judge it has never been run. **REJECTED at n=6:** TP -2.0
  (p = 0.39), **FP +6.5 (p = 0.01)**, **macro F1 -1.5 (p = 0.00)**, macro F2 -1.0
  (p = 0.02), and the loss shows at n=3 too (F1 -1.7 at the floor), so it is not a
  variance artefact. Note the direction: **freeing 16 candidates cost 2.0 true
  positives.** Adding candidates cannot remove a link directly, so the loss is batch
  composition in the two-step partial-name judge — the same mechanism the `_unlinked`
  arm measured in the other direction (-6.8 FP purely from changing which cases share a
  batch). **The alias table's dual role is therefore load-bearing in both directions,
  and the paper must state it as a property of the design rather than an accident:**
  the table admits full-name candidates and suppresses partial-name ones, and removing
  the second role costs 1.5 F1.
  `pilot/test_s45_s46_singlechange.py` asserts one change each — every one of 40 method
  bodies, 10 rubrics and 6 bounds byte-identical to s25's apart from the intended one,
  and for s46 that every freed candidate is explained by a discovered alias and none is
  lost. Runs: `../results/s4546_e2e_r{1..6}_20260812`, six paired runs carrying s25,
  s45 and s46 in the same invocations. `pilot/score_runs.py` now accepts more than two
  arms and tests every later arm against the first.
- `src/llm_sad_sam/linkers/experimental/s_linker47.py` — `SLinker47`, standalone
  (experimental=True). **s25 with one MECHANISM removed: the partial-name linker's
  grounded identity review.** Step 1 (target-blind denotation, worth 12 FP) stays; step
  2, which shows the model the target and the sentences naming it, had never been priced
  alone. Six recorded s25 runs price it: **20.3 candidates per run reach it, it keeps
  12.3 (12.2 gold) and rejects 8.0 of which 5.5 are gold** — it trades 5.5 true positives
  for 2.5 false positives, a bad trade for F1 and worse for F2. Removing it makes the
  partial-name linker the same shape as the coreference linker (one proposer, one judging
  call) and deletes a prompt, the anchor bookkeeping, a four-conjunct evidence gate and
  the `alternative` response field. Composition can still move it: partial-name links
  feed `_unlinked`, so 5.5 more of them removes 5.5 pairs from coreference's input.
  **CONFIRMED — the first mechanism removal in the series that costs nothing.** Six
  paired runs (`../results/s4748_e2e_r{1..6}_20260813`): **TP +6.2 (p = 0.00), FP +6.8
  (p = 0.00), macro F1 +0.2 (p = 0.53), macro F2 +1.3 (p = 0.01)**, at 87.2 calls against
  89.2. F1 is a wash and F2 is significantly *better*, so an LLM stage, a prompt, the
  anchor bookkeeping and a four-conjunct gate all leave the design for free. Its F1
  spread is also the tighter of the two (0.74 against 1.54). Per project the review turns
  out to help exactly one document and hurt the rest: teammates F1 89.84 -> 87.15
  (FP 8.0 -> 12.5, where it is the false-positive guard), against **bigbluebutton
  91.34 -> 94.15 F1 and 88.41 -> 94.75 F2** (TP 53.7 -> 59.0), teastore 99.37 -> 100.00,
  jabref 99.10 -> 99.55, mediastore 98.63 -> 98.36. It buys precision on the one project
  with many partial-name proposals and costs recall on the one where partial names carry
  real links.
- `src/llm_sad_sam/linkers/experimental/s_linker48.py` — `SLinker48`, standalone
  (experimental=True). **No mechanism removed; eight copies of three conditions in five
  shapes become three named predicates, and three never-firing conjuncts go.** All four
  merges were sized off six recorded runs first and every one is a *provable* identity,
  not a behavioural claim:
  - **"does this sentence state a name of this component?"** — the identical expression
    `any(_find_exact_form(text, n) for n in (name, *aliases))` at three sites (the
    admission filter, the partial-name whole-name exclusion, the coreference antecedent
    gate) becomes `_states_a_name`. The mention-label classifier keeps its own two calls
    because it must know *which* matched;
  - **"is the model's quote really in the sentence?"** — two copies, one per partial-name
    judging step, become `_claim_supported`. It stays because it fires: 0.2 denotation
    verdicts per run;
  - **"which sentences are near this one?"** — three shapes (two `abs(...) <= C` filters
    and one `range(max(1, n-C), n+C+1)` walk against the sentence map) become `_window`;
    verified over 378 targets on all five documents, 0 divergences, and the coreference
    prompt's marked context strings compare byte-identical;
  - **three conjuncts that never fire** — the identity review approved only on a listed
    anchor AND a quoted sentence AND a named alternative, and `evidence_valid` was False
    **zero** times in 122 recorded cases over six runs. The anchor-listed and
    non-empty-alternative conjuncts are deleted by exactly the argument the paper already
    uses for not adding the claim check to the other judges ("voids 0 verdicts in 25
    project-runs"). **The prompt is untouched** — the model is still asked for all three —
    so what goes is code that re-checked two answers and never caught one. Demanding a
    commitment is worth 35.2 TP here; verifying it is worth nothing, and the two are
    separable.
  `pilot/test_s47_s48_mechanisms.py` (172 checks) asserts both variants' single change,
  every other method body byte-identical to s25's, both merged predicates against the
  expressions they replaced over every (name, sentence) pair on all five benchmarks, and
  **every prompt builder rendering byte-identically on real project data**.
  **CONFIRMED FREE.** Six paired runs (`../results/s4748_e2e_r{1..6}_20260813`, carrying
  s25, s47 and s48 in the same invocations): TP +0.7 (p = 0.65), FP -1.3 (p = 0.50),
  macro F1 +0.3 (p = 0.50), macro F2 +0.2 (p = 0.57) — and the statistic that matters for
  a merge, **composition -0.2 (p = 0.59)**: the two arms' link sets differ *less* between
  arms than within them, which is what behaviour preservation looks like when the only
  remaining source of difference is the model's own nondeterminism. Eight condition copies
  in five shapes become three named predicates at no cost.
- `src/llm_sad_sam/linkers/experimental/s_linker49.py` — `SLinker49`, standalone
  (experimental=True). **s47 and s48 composed**: the grounded identity review removed, so
  all three linkers judge in one call, and the condition merges whose duplication survives
  that removal (`_states_a_name` at three sites, `_window` at two). `_claim_supported` is
  deliberately *not* carried — with the identity review gone its duplication is moot and a
  one-call-site helper is not a simplification. Deterministically verified equivalent to
  s25 on both candidate generators, the antecedent gate and the window on all five
  projects. Runs: `../results/s49_composed_e2e_r{1..6}_20260813`. Composition is checked
  rather than assumed: this workflow has seven instances of an arm that held alone and
  failed in another composition — and here it holds. Six paired runs
  (`../results/s49_composed_e2e_r{1..6}_20260813`): **TP +5.0 (p = 0.01), FP +7.2
  (p = 0.01), macro F1 -0.2 (p = 0.50), macro F2 +0.9 (p = 0.03)**, at 87.2 calls against
  89.2 — replicating s47's numbers almost exactly, so the two simplifications are
  independent and additive. **The design this variant states: the partial-name linker
  judges in one step like the coreference linker (four judging steps in the workflow, not
  five), and two named conditions replace eight copies in five shapes.** The full-name
  judge keeps its two focused calls — merging those is the one change this series measured
  as significantly worse (s36: F1 -0.7, FP +3.5, both p = 0.01). F1 is
  statistically unchanged and F2 is significantly better, so it is a free simplification
  under an F1-led paper and an improvement under an F2-led one.

### The prompt round (s50-s55, s49_null) — ablating the instructions, not the pipeline

Every earlier round ablated structure. This one ablates the **hand-written English**:
ten rule constants, 4022 bytes, carried into 88 calls per five-project run (60.9 kB
of instruction against 948.6 kB of prompt — **6.4% of what the workflow sends**). The
question is the one a reviewer asks directly: are these general guidelines, or a
rulebook grown against five benchmark documents? `pilot/prompt_audit.py` prices every
clause deterministically off s49's six recorded runs first; five arms then answer it,
six paired runs each. Reports: `../results/s5051_prompt_ablation/`,
`../results/s5253_prompt_bisect/`, `../results/s5455_prompt_families/`,
`../results/null_calibration/`.

**THE HARNESS HAS A NULL AND IT IS NOT ZERO — read this before any p value below.**
`s_linker49_null` is `s_linker49` with one difference, the checkpoint namespace; the
file diff is empty modulo the rename and the two document-determined phases send
byte-identical prompts in 30 of 30 comparisons. Six paired runs, s49 first in every
invocation: **TP -4.8 (p = 0.00), macro F1 -0.7 (p = 0.03), macro F2 -1.2 (p = 0.00),
composition +3.1 (p = 0.01) — `score_runs.py` calls it QUALITY-CHANGING**, and the
sign is the same in 6 of 6 runs, concentrated in the full-name stage on teammates.
Each arm re-runs the whole pipeline including the stages it does not modify, so
pairing inside one invocation controls the model, the day and the ordering but not
the upstream sampling. **Consequence: a |F1| of 0.7 and a TP of 5 are inside what
this harness produces from nothing, the pooled permutation test is anti-conservative
here, and every delta in this file should be read against the null's delta rather
than against zero.** Reversing the order (`../results/nullrev_e2e_r{1..6}_20260813`,
s49_null first) rules out arm position: s49 leads in **both** orders (+0.7 F1 first,
+0.8 F1 second, TP +1.8 p = 0.09), and `_VARIANT_NAME` reaches nothing but three
`os.path.join` calls. Sampling is not pinned — `OPENAI_REASONING_EFFORT` set means
`llm_client` sends `reasoning_effort` and omits temperature, `seed=42` is best-effort
— so two runs of one program are two draws and six-run means of draws this wide land
0.7 F1 apart often enough for 924 splits to call it significant. Absolute levels also
drift between invocation sets (s49's FP mean read 10.7, 11.7, 12.5, 14.5 and 16.8
across five sets in one day, one run hitting 33), so **never compare across
invocation sets**.

The clause-level screen (`pilot/prompt_audit.py`, deterministic, no LLM calls):
- the **qualified-path rule is written five times** in five prompts, and one of five
  documents has dotted identifiers at volume (teammates 62/198 sentences; 0-6 on the
  rest, and those are `e.g` / `i.e` / `React.js`). At the full-name judge 4.5
  candidates per run have the name *only* inside a path, **0.0 of them gold**;
- **`COREF_RULES` clause (b)** — resolve a role-referential phrase to the section
  topic *without* a name repetition — licensed **0.0 of 578 recorded resolutions**:
  every one had the name or a discovered alias inside the ±5 context sentences shown,
  and `_antecedent_states_name` discards independently what the clause permits;
- its five listed role phrases cover 17.3 resolutions/run of which **15.3 are `it`**;
  the terminal-word alias sentence covers 1.7 antecedents/run;
- of the full-name gate's 14.7 rejections/run, 2.2 match its condition (1) and 1.8 its
  condition (2) lexically — **73% rest on the two conditions that name no surface form**.

The five arms, all against s49, six paired runs each:

| arm | generalized | rule text | instr. B/run | macro F1 | p |
|---|---|---|---|---|---|
| `s_linker50` | the coreference resolution rule only | -10% | -27% | -0.2 | 0.71 |
| **`s_linker55`** | **the whole coreference family** | **-19%** | **-31%** | **-0.0** | **0.90** |
| `s_linker54` | coreference + knowledge | -28% | -34% | -1.1 | 0.00 |
| `s_linker52` | coreference + full-name | -30% | -41% | -2.1 | 0.00 |
| `s_linker51` | all nine of ten constants | -39% | -44% | -2.4 | 0.00 |
| `s_linker53` | all nine, one clause restored | -37% | -44% | -2.5 | 0.00 |

- **`s_linker55` is the result: the whole coreference family can be stated as
  guidelines for nothing** — TP -1.5 (p = 0.27), FP -1.5 (p = 0.58), F1 -0.0
  (p = 0.90), F2 -0.3 (p = 0.39), composition +0.9 (p = 0.19), and it sat in position
  3 where the null costs 0.7 F1. Three rules, 1773 B, 54 calls/run — the largest
  instruction item in the workflow — lose two lettered clauses, a five-phrase list,
  an alias-shape enumeration, a gloss of "architectural claim" and three named
  fragment shapes. The generalized resolver *proposes* far more (mediastore 6.8 ->
  11.7/run, jabref 1.2 -> 9.7) and the strict judge rejects the surplus: **the
  enumeration was doing work the downstream judge already does.**
- **Both load-bearing families are on the admitting side.** Full-name is worth ~2.1 F1:
  held to the candidates both arms judged, so the proposer is constant, the
  generalized judge approves **3.5 more false positives per run and no more gold**,
  and the extraction rule's dropped aside ("even if the compound identifier is
  semantically related") takes teammates from 68 proposals/50 accepted to 79/63.
  Knowledge is worth ~1.1 F1 because the alias table is the only structure that
  admits rather than rejects. **A rejecting stage that over-rejects is caught by
  recall it never had; an admitting stage that over-admits has no downstream that can
  tell.**
- **`s_linker53` is the eighth instance of a trace-derived reading not surviving its
  arm, and the first that was directionally right and still wrong about the
  mechanism.** Round 1's traces indicted one clause of the alias judge (a grouping of
  several elements is not an alias for one of them) because s51's tables gained
  `core`, `outer shell`, `intermediate layer`, `back end`. Restoring exactly that
  clause recovers **nothing**; reverting the whole knowledge family recovers 3.5 of
  10.3 FP. Two reasons, both general: a **surface** attribution ("this link came in
  through an alias term only this arm has") is not a **causal** one, since every alias
  is fed to the extraction prompt; and the table is not stable enough to attribute
  from — s49 and s50 have byte-identical knowledge prompts and still build tables
  differing by 2.8 terms per run.
**CLAUSE-LEVEL ROUND (s56-s59) — single-stage ablation instead of E2E.** Six paired
five-project runs cost ~90 minutes and fight a 0.7 F1 null; a question about one
sentence of English does not need them. `pilot/prompt_stage_pilots.py` replays **one
stage** with the two wordings against the *same* recorded checkpoint inputs, five
samples a side, and permutation-tests that stage's own output. It asserts first that
its re-declared prompt builders render byte-identically to s49's, so an arm measures
the swapped constant and not the re-declaration. Eleven arms, minutes each; report in
`../results/prompt_stage_pilots/`.

| stage | what varies | TP (p) | FP (p) |
|---|---|---|---|
| alias judge | `DOC_KNOWLEDGE_JUDGE_RULES` | +0.0 (1.00) | -0.4 (1.00) |
| full-name judge | `P1_FOCUS` | +0.0 (1.00) | +0.2 (1.00) |
| alias proposer | extraction + exclusion rules | +0.6 (0.63) | +1.8 (0.33) |
| coreference judge | focus + `LAYERED_COREF_RULES` | +1.8 (0.30) | **-1.2 (0.05)** |
| full-name judge | `LAYERED_ENTITY_RULES` | +0.0 (1.00) | **+2.4 (0.01)** |
| extraction | `ENTITY_EXTRACTION_RULES` | +6.2 (0.03) | **+20.2 (0.01)** |
| coreference resolution | drop the whole preamble (s56) | **-16.2 (0.01)** | **+14.0 (0.01)** |

- **Two hypotheses refuted before an E2E run was paid for.** `s_linker58` (generalize
  the extraction rule) adds 20.2 FP per run at the stage that feeds everything else.
  `s_linker56` (delete the coreference prompt's opening paragraph as a restatement of
  `COREF_RULES`) costs 16.2 TP. Decomposing that paragraph explains it: dropping only
  the strictness sentence reads TP **+23.2** (p = 0.01) / FP +11.4, dropping only the
  protocol sentences TP +9.2 / FP +11.8, dropping both TP -16.2 — strongly
  non-additive, because **the protocol sentences are an input-format contract** (which
  block is the TARGET, that a target with no referring expression yields nothing) that
  `COREF_RULES` never states. Generalizing a rule and deleting a format contract are
  different operations. Both variants are kept as the artifacts that price this.
- **The clause level shows what the family level could not.** The full-name family is
  not uniform: `P1_FOCUS` generalizes for exactly nothing while `LAYERED_ENTITY_RULES`
  costs 2.4 FP on the very same candidates in the same call — 289 bytes of decoration
  in front of 692 bytes of gate.
- **Refined rule: a prompt clause is removable when something downstream rejects by
  default.** The coreference rules sit in front of a gate that rejects when uncertain,
  and they go; `P1_FOCUS` sits in front of `P2_FOCUS` and `LAYERED_ENTITY_RULES`,
  which are the gate, and it goes; `ENTITY_EXTRACTION_RULES` sits in front of a judge
  that approves by default, and it stays.
- `src/llm_sad_sam/linkers/experimental/s_linker59.py` — **`SLinker59`, what survives:**
  the coreference family plus `P1_FOCUS` plus the alias judge rubric, every clause that
  cleared and no others. Rule text 4022 -> 2960 B (**-26%**), instruction bytes per
  five-project run 60 892 -> 40 081 (**-34%**). The three compose: run through the
  knowledge stage and the whole full-name linker, each arm building its own alias table
  and judging its own candidates, TP -1.0 (p = 0.23), FP -0.2 (p = 1.00), composition
  +0.0 (p = 0.45). **Still owed: six paired runs against s49 with an in-set null**, by
  this branch's own standing rule — a stage arm screens, it does not decide.
- `pilot/prompt_audit.py --only P7` now also inventories the authored text *outside*
  the ten constants: **41.5 kB per run of builder text** against 60.9 kB of constants,
  so 40% of the instructions this workflow sends had never been in scope. The three
  largest items are the coreference preamble (253 B x 40, refuted above), the
  claim-before-verdict paragraph (210 B x 25, measured worth 35.2 TP) and the JSON
  skeletons (~150 B x 88, the response contract the parser depends on).
**THE MERGE LINE'S MISSING ARRANGEMENT (s60, s61) — built, measured, refuted with a new
mechanism.** The s26-s34 line never tried *merged alias proposal with the judge kept
separate*: s26/s28 merged the proposal and deleted the judge; s29-s34 kept the separate
proposer and moved the judging. `s_linker60` folds alias proposal into the
entity-extraction reading and keeps `_judge_aliases` as its own call with s49's prompt
and rubric — **three document-reading prompts become two, 83 calls per run against 88**.
`s_linker61` adds `ALIAS_EXCLUSION_RULES` to the judge prompt, because the merged reading
leaks `logic.api`/`logic.core` that the dedicated proposer never did (measured reach on
this benchmark: zero; adopted as design integrity, not as a performance claim).
Report: `../results/merged_alias_design/`; invariants: `pilot/test_s60_s61_merged_alias.py`.

- **The alias side improves and the pipeline still loses.** Stage screen on the alias
  table (`--pilot mergedalias`): TP -0.6 (p = 0.80), **FP -16.6 (p = 0.01)**. End to end
  (`../results/s5960_e2e_r{1..6}_20260813`, six paired runs carrying s49, an **in-set
  null arm**, s59 and s60): **s60 TP -5.0 (p = 0.00), FP +11.2 (p = 0.01), macro F1 -2.7
  (p = 0.00), macro F2 -2.2 (p = 0.00)**, while the null arm reads TP +0.7 (p = 0.54),
  F1 +0.1 (p = 0.71), composition +0.1 — neutral on every measure, so this set has **no
  harness offset to subtract** and the s60 result is the arm's own.
- **The mechanism is the alias table's second job.** `stage_diff.py` puts 13.5 of the
  14 extra false positives on the **partial-name** linker (gained spurious 13.5, gained
  gold 0.0). `_name_word_candidates` excludes a sentence from partial-name proposals when
  it states any name in N(c), so a *tighter* alias table frees candidates. Deterministic,
  no LLM call: on teammates s49's 6-term table yields 31 partial-name candidates (4 gold),
  s60's 10-term table yields 40 (4 gold), and **adding the single term `GAE` to s60's
  table takes it to 30 with no gold lost.** One document-introduced short form, defined
  once far from its uses, explains the whole regression — exactly what a batch-local
  reading cannot see.
- **Per project, the failure lands exactly where the partial-name linker runs.** Six
  runs: jabref 100.00 -> 100.00, mediastore 97.83 -> 97.02, teastore 100.00 -> 99.09 —
  the three projects whose partial-name linker never fires. teammates 91.40 -> **83.56**
  and bigbluebutton 93.10 -> **89.02** — the two where it does. The mechanism confirmed a
  third way, at the project level, with no further measurement.
- **The obvious repair is refuted, in fifteen calls.** Offering every standalone name word
  of a multi-word component to the unchanged judge (`--pilot namewordalias`): the judge
  approves them wholesale — they are not "generic vocabulary", they are parts of the name
  — and the partial-name linker is suppressed out of existence: bigbluebutton 31
  candidates (16 gold) -> 3 (0 gold). Projecting the table from the reading's
  `matched_text` spans is also dead: the reading reports **no span at all** for
  `GAE Datastore`.
- **Final accounting: the merge saves exactly one call per project** (s49 is N+2, s60 is
  N+1) **and costs 2.7 macro F1**, and any repair that restores the table's suppressive
  shape is a document-wide pass — the call the merge was removing. **The knowledge module
  is not removable, and the reason is new: the alias table is not only an alias table.**
  It is broad enough to hold document-introduced short forms like `GAE` and narrow enough
  to exclude ordinary name words like `Server`, and only a document-wide pass told to
  "reject terms whose ordinary English use dominates" produces that shape. This is the
  third and sharpest measurement of the dual role, after the s26 diagnosis and s46 — and
  the first where the alias table got *better* by every alias-side measure while the
  pipeline lost.
- **s_linker59 CONFIRMED, six paired runs in the same invocations**: **TP +1.5
  (p = 0.05)**, FP -2.2 (p = 0.40), macro F1 +0.6 (p = 0.18), **macro F2 +0.5
  (p = 0.03)** against an in-set null reading +0.1 F1. The clause-level prompt
  minimization holds end to end and is mildly *better* than s49 on recall and F2, at
  -26% rule text and -34% instruction bytes per run.
- A checkpoint bug found and fixed: s60/s61 write the knowledge checkpoint before the
  linkers run, and their table does not exist until the reading — inside the *first*
  linker — has built it, so the checkpoint recorded an empty table and the first pass at
  diagnosing these runs read it and drew a wrong conclusion. Both now re-save after the
  linker loop, asserted by the test.
**THE PARTIAL-NAME ROUND (s62, s63, s64, s59_null) — the frontier the merged-alias round
pointed at.** Every alias change fails *through* the partial-name linker's suppression
role, and that linker's two projects carry essentially all of the workflow's false
positives, so this round audits it before proposing anything. Report:
`../results/partial_name_round/`; invariants: `pilot/test_s62_s63_proposer.py`; new
tooling: `pilot/partial_audit.py`, `partial_gap.py`, `partial_hole.py`,
`partial_screen.py`, `statednet_screen.py`, `partial_pilots.py`.

- **The judge is not the bottleneck; the proposer is.** `pilot/partial_audit.py` over the
  six s5960 runs: 60.3 candidates/run, 18.7 of them gold, 21.2 approved -> 17.7 TP /
  3.5 FP; the denotation judge runs at **95% recall over the gold candidates and 83%
  precision**, so a *perfect* judge over the same candidates would be **+1.0 TP,
  -3.5 FP**. All the headroom is upstream: 41.5 gold pairs are open at this stage and
  18.7 are offered.
- **Of the 22.8 it declines, 15.0 are not a loss.** `pilot/partial_gap.py` attributes
  every declined open-gold pair by the deterministic reason and cross-references the
  final links: `no_hook` 15.0/run, **every one recovered by the coreference linker**;
  `states_a_name` 5.8/run, 5.7 lost; `ambiguous` 2.0/run, 2.0 lost. Against a total
  residual recall loss of 8.0 gold pairs per run, **this one stage's two declines are the
  pipeline's entire remaining recall loss** — and the third bucket is the division of
  labour working, which a naive recall reading would have called a hole.
- **The bigger bucket's obvious repair is refuted deterministically.** The whole-name
  test hands the pair to the full-name linker unconditionally; splitting it shows 3.0/run
  the extraction call never proposed and 2.7/run a full-name judge rejected. Deferring
  only where the full-name stage actually *ruled* is **+0.7 gold, +10.0 spurious**
  (`pilot/partial_hole.py`) — because the whole-name test is also the alias table doing
  suppression, so nearly every sentence containing `logic` states an alias of `Logic`.
  Fourth measurement of the dual role.
- **s_linker62: the ownership test was the defect.** `surface.startswith(word)` in both
  directions makes `WebRTC` an owner of `WebRTC-SFU` (exactly) *and* `BBB web` (as a
  continuation of `web`); the proposer needs a unique owner, so two gold links die every
  run. Bounding the prefix to English inflections is **+2.0 gold / +0.0 spurious**
  candidates and dominates the exact-beats-prefix ranking (+2.0 / +1.0), because `cams`
  is not an inflection of `web` either. It also states what the old docstring already
  claimed. With the real judge, five samples a side: **TP +2.0 (p = 0.01), FP +1.0
  (p = 0.01)**; bigbluebutton reaches 61.7 of its 62 gold pairs.
- **A real bug whose repair costs precision (s_linker63).**
  `_inside_qualified_identifier` writes `before in "-_"` with `before == ""` for a
  sentence-initial span, and **`"" in "-_"` is `True`** — so every span at a sentence's
  first character, and every span at the document's last, has been treated as inside a
  qualified identifier by every variant in this branch: **344 spans per run**. Guarding
  it is **TP ±0.0 (p = 1.00), FP +1.2 (p = 0.01)**. On this benchmark the defect is
  load-bearing; s63 exists so it is priced rather than quietly kept.
- **s_linker64: the hand-off is right, so fix the recipient.** For the 3.0/run the
  extraction call never proposed, a deterministic scan for a sentence stating the model
  name **as spelled** is 1.2 new pairs/run at **0.86 gold each** — level with the
  extractor's own 0.87 — where the same scan case-insensitively is 31.3 at 0.06 and
  alias-wide 41.2 at 0.07 (`pilot/statednet_screen.py`). **Case is the whole design:** a
  component named `Common` or `Client` matches ordinary English on every page. It is the
  one site where the workflow's single *lenient* name primitive is the wrong tool.
  Behind the unchanged two-pass judge: **TP +1.2 (p = 0.01), FP +0.4 (p = 0.44)**.
- **End to end (`../results/s6263_e2e_r{1..6}_20260814`, s59 / s59_null / s62 / s63):
  the set is too noisy to read whole-pipeline deltas.** s59's own macro F1 range is 4.44
  and its FP is 17.0 against 12.2 for the same code the day before (teammates FP 9.7 vs
  3.8); the null arm reads TP +1.3 (p = 0.08), **macro F1 +0.4 (p = 0.35)**. s62 reads
  F1 -0.1, s63 -0.6 — all inside the null.
- **Restricting the permutation test to the links whose `source` the change can reach
  makes it readable, and the control proves the instrument.** Partial-name links only:
  null TP +1.2 (0.13); **s62 TP +2.3 (p = 0.00), FP +0.7 (0.67)** — exactly the stage's
  +2.0/+1.0, landing where predicted (bigbluebutton TP 57.8 -> 59.7); **s63 FP +3.8
  (p = 0.03)**. Full-name links only: **every arm neutral** (TP -0.3 / -0.7 / -0.5, all
  p >= 0.51), and so is coreference (all p >= 0.57), which is right — neither change can reach that stage, so `stage_diff.py`'s ±4-link
  swings there are drift.
- **s62 adopted, s63 refuted.** s62's inflection bound is significant at its own source,
  neutral elsewhere, and replaces an unbounded approximation with the predicate the code
  already claimed; its macro F1 claim is *indistinguishable from s59 in a set whose null
  reads +0.4*, not "better". s63's repair of a genuine bug costs 3.8 FP at the source, so
  the `"" in "-_"` defect stays — documented and priced.
- **Standing rule added:** when an audit says a stage declines gold, split the declines
  by their deterministic cause *and* check the final link set before calling any of them
  a loss — 15.0 of 22.8 here are another linker's job, and the two real buckets needed
  opposite repairs at two different stages.
- **Standing rule added:** when an arm changes one linker, read it on that linker's
  `source` first (`pilot/source_stats.py`, or the per-source permutation test in
  `../results/partial_name_round/`) and use the stages it *cannot* reach as the control.
  A whole-pipeline macro F1 mixes the effect with two stages of drift; here the effect
  was p = 0.00 at its source and p = 0.81 in the macro.
- **STANDING METHOD CHANGE — the E2E confirmation rule is now conditional, and the
  condition is checkable.** The rule existed because of one episode (`_keep_stated_names`:
  stage-positive, quadrupled FP composed), and its mechanism is `_unlinked` — a link
  admitted early is locked into the union *and stolen from the later, stricter linkers*.
  That precondition is deterministic: **`pilot/composition_check.py`** reads off the
  checkpoints whether the pairs a change adds or removes are pairs a later stage would
  otherwise have proposed, or were in the final link set. For both changes this round the
  answer is **0.0 pairs per run**, so the stage arm *is* the pipeline answer and an E2E
  measures model drift instead of the change — confirmed after the fact by batch 1, where
  s62's per-source reading (TP +2.3) reproduced its stage pilot (+2.0) while the macro it
  was mixed into read -0.1 against a null of +0.4. **Ablate at the stage that changed,
  run the composition check, and pay for E2E only when it is non-zero.**
- **s_linker64 adopted on stage evidence** (TP +1.2 p = 0.01, FP +0.4 p = 0.44,
  composition risk 0.0). Its E2E batch was stopped after two runs; those are kept at
  `../results/s64_e2e_r{1,2}_20260814` as corroboration only — the null arm moves +4.5 TP
  on the very stage s64 changes, which is why two runs decide nothing.

**THE RULE ROUND (s65) — how many hand-written rules are there, really?** Every earlier
round ablated a rule and priced its removal. This one asks the prior question a reviewer
asks first: the workflow carried **four** lexical rules by s64 — the full-name admission
filter (`_keep_stated_names`), the spelling-variant proposer, the s64 stated-name net,
and the partial-name proposer — each with its own regex, its own ownership test and its
own paragraph of defence. Read as four rules they read as accretion. `pilot/rule_audit.py`
(deterministic, no LLM calls; report in `../results/rule_audit/`) shows they are not four
rules.

- **They are one relation at four settings, and the check is an identity.** A single
  `_name_spans(text, name, form)` reproduces `_find_exact_form`, the net's scan,
  `_spelling_variant_candidates`' owner test and `_is_inflection_of` with **0
  divergences in each of the four comparisons over all 3697 (name, sentence) pairs** of
  the five projects. The relation has two dimensions — *fidelity* (`AS_SPELLED` <
  `ANY_CASE` < `ANY_SPELLING`: how exactly the characters must reproduce the name) and
  *extent* (`ANY_WORD`: the whole name, or one word of it) — and which cell a proposer
  scans is the only thing that distinguishes it from the others.
- **One monotone table replaces four arguments** (`--only A3`, pairs reached over all
  five projects and how many are gold):

  | fidelity / extent | pairs | gold | gold per pair |
  |---|---|---|---|
  | `AS_SPELLED` whole name | 112 | 107 | **0.955** |
  | `ANY_CASE` whole name | 172 | 133 | 0.773 |
  | `ANY_SPELLING` whole name | 176 | 137 | 0.778 |
  | `ANY_WORD` one word | 281 | 161 | 0.573 |

  Precision falls monotonically as the relation loosens and recall rises, and that
  single table *is* the design rationale: **the looser the form a linker scans, the
  stricter the judge behind it.** The full-name linker scans the two tight rows and
  judges in two focused calls that approve by default; the partial-name linker scans the
  loosest row and judges target-blind; the coreference linker reaches what no row reaches
  and rejects when uncertain. **s64's case sensitivity stops being a bespoke rule and
  becomes the top row of this table** — 0.955 gold per pair against 0.773 one row down is
  why the recall floor under the LLM extractor is drawn at `AS_SPELLED`, and it matches
  the 0.86-vs-0.06 reading `pilot/statednet_screen.py` took on *new* pairs.
- **Two cells do not nest, and the tidier claim would have been the false one.**
  teastore's `ImageProvider` is written `Image Provider`, which `ANY_SPELLING` reaches
  and `ANY_WORD` does not (the *name* is split on word boundaries only, so
  `imageprovider` is one word); bigbluebutton's `Redis PubSub` is written `redis
  pubsub`, which `ANY_CASE` reaches and `ANY_SPELLING` does not (the signature splits
  `PubSub`, the document does not). So compound splitting is a *different*
  normalization, not a strictly looser one, and the linker takes the **union** of the
  cells it scans. Six pairs over five projects; stated because a chain would have read
  better and is not true.
- **Nothing in the deterministic layer admits a link** (`--only A1`): 0 of 18 predicates
  put a link in the output without an LLM verdict. Every scan produces a *case for a
  judge*. That is the sentence the paper needs and it is now checkable rather than
  asserted.
- `src/llm_sad_sam/linkers/experimental/s_linker65.py` — `SLinker65`, standalone
  (experimental=True). **s_linker64 with no mechanism, prompt or behaviour change**: the
  three candidate generators become three rows of a `SCANS` table over the one relation,
  and `_antecedent_states_name` (a one-line wrapper with one call site) is deleted.
  `pilot/test_s65_one_relation.py` asserts the identity in **49/49 checks**: 44 shared
  method bodies byte-identical, 11 rule constants / 7 resource bounds / 5 prompt builders
  identical, the relation against all four predicates on 3697 pairs, every candidate set
  of all three generators on all five projects (same pairs, same `matched_text`, same
  `source`, same mention labels), the composed full-name candidate list under **four
  extractor stand-ins per project** (empty, full overlap, disjoint, half overlap — so the
  merge is checked where a scanned pair collides with one the extractor already holds, and
  the existing candidate must win), and GATE-06.
  Measured effect on the audit's own terms: lexical layer 9 methods / 106 code lines ->
  8 / 90, and **methods a reviewer must read to know what a name match is: 5 -> 2**. The
  scan is 3-6x slower in wall-clock (0.02s -> 0.07s on the largest project) against runs
  of ~65 LLM calls, so the cost is not measurable end to end. **No E2E owed:** this is an
  identity over the candidate sets, not a behavioural arm — `pilot/composition_check.py`'s
  precondition is vacuous when the candidate sets are equal.
- **Two defects carried forward deliberately, both priced rather than quietly fixed.**
  `_inside_qualified_identifier` tests `before in "-_"` with `before == ""` for a
  sentence-initial span, and `"" in "-_"` is `True`, so **378 spans over the five
  documents** (exactly one per sentence) are treated as inside a qualified identifier;
  `s_linker63` repaired it at **FP +1.2 (p = 0.01), TP ±0.0**, so on this benchmark the
  defect is load-bearing. `_all_occurrences_in_qualified_path` lowercases the name and
  searches the raw sentence, so it only ever sees lowercase spellings; handling case the
  way the rest of the module does would move **3 mention labels over all five projects**
  (`--only A4`). Both are left alone in s65 because s65 changes no behaviour, and both
  are named in its docstring. **A retained bug that a benchmark rewards is a validity
  threat the paper has to state, not a design choice** — this is the one place where
  "measured" and "defensible" point in different directions.
- New tooling: `pilot/rule_audit.py` (A1 inventory of every deterministic predicate by
  what it decides; A2 the four-predicates-one-relation identity; A3 the monotone yield
  table and the containment check; A4 the residue and the two priced defects; A5 the
  before/after of the restatement) and `pilot/test_s65_one_relation.py`.

**THE BIND ROUND (s65_null, s66, s67) — could the prompts do the deterministic layer's
work?** `rule_audit.py` said how many rules there are; this round asks whether they have
to be code at all. The extraction call already reads every sentence and already receives
the alias table; the judging call is already shown the sentence a label was computed
from. So every arm here is a **relocation**: the rule leaves the code and its content
enters a prompt, and each pilot carries three arms — the linker as it stands, the rule
deleted with no compensation, the rule stated in the prompt — so a working binding can be
told apart from a rule that was never worth anything. Report:
`../results/bind_round/`; tooling: `pilot/bind_audit.py` (deterministic, B0-B7),
`pilot/bind_pilots.py` (seven stage arms); invariants:
`pilot/test_s66_s67_bindcontract.py` (56/56); runner: `pilot/run_s6667_e2e.sh`.

- **B0 licenses the round**: each run's full-name candidate set, rebuilt from the
  extraction call's own logged response plus the scans, matches the checkpoint with
  **0 extra and 0 missing over 30 project-runs**.
- **The binding gap, six runs.** The extractor already proposes **111 of 112**
  `AS_SPELLED` pairs and **6 of 6** spelling-variant pairs, so those scans have almost
  nothing to newly ask for; the partial-name scan's gap is **53.8 pairs and 15.8 gold per
  run** — pairs the extraction call sees and declines, concentrated on the two projects
  where that linker fires.
- **The admission filter is a router, not a gate.** It drops 24.8 proposals per run, 8.0
  gold, and later linkers recover **7.7 of those 8.0**: net recall cost 0.3 gold per run.
  What it buys is precision through `_unlinked`.
- **`s_linker66` — the one relocation that holds.** `_keep_stated_names` deleted, its
  contract stated in `ENTITY_EXTRACTION_RULES`. Stage: deleting the filter with no
  compensation is TP +4.8 / **FP +10.6** (p = 0.01 both); stating it instead is TP -1.4
  (p = 0.21), FP -1.8 (p = 0.47). E2E, six paired runs with an in-set null that is quiet
  (`../results/s6667_e2e_r{1..6}_20260817`): **TP -2.5 (p = 0.15), FP -0.2 (p = 1.00),
  macro F1 -0.2 (p = 0.76), macro F2 -0.5 (p = 0.44)**, at 88 calls against 88. The
  relocation buys rule count, not calls, and the recall delta — twice the null's, not
  significant — is stated rather than buried.
- **`s_linker67` rejected, and it is the round's methodological result.** Relocating the
  two tight scans as well reads **TP -1.2 (p = 0.14) at its own stage** and **TP -4.0
  (p = 0.03), macro F2 -1.1 (p = 0.04) composed**. Ninth instance of a stage arm pointing
  the wrong way, same mechanism every time: what an early stage stops proposing is not
  re-offered to the later, stricter linkers.
- **Telling the model to scan does not make it scan.** Deleting the two tight scans under
  s65's unchanged prompt costs TP 3.6 (p = 0.01), and a clause asking for exactly what
  they scan recovers **none** of it (TP -3.8, FP +3.2). Third measurement of this, after
  s64's 3.0-pairs-per-run hole and `statednet_screen.py`'s 0.86-vs-0.06 gold rate.
- **A mechanism the round did not go looking for: the spelling scan is not a proposer.**
  Of the 2.0 pairs per run that reach the candidate set only through a scan, 1.0 is one
  the extractor never proposed and 1.0 is one it *did* propose and the admission filter
  would have dropped — its surfaces (`X Y` for a name spelled `XY`) do not write the name
  at `ANY_CASE`. **The spelling row is a widening of the admission filter**, which is why
  the two relocations cannot be priced separately, and why `bindboth` looked better than
  `bindscans` at the stage.
- **The mention label's content binds; its performance does not.** Dropping the label is
  **TP -8.4 (p = 0.01)** — the sharpest number yet on a field s43/s44 argued about.
  Asking the judge for it instead is worse on both measures (TP -3.6, FP +2.4). Asking
  for it *and* giving the judging prompt the alias table — the one prompt in the workflow
  that never sees it, and 39.0 of 182.5 labels per run are `via known alias` — recovers
  recall past the control and spends precision: **TP +4.0, FP +5.0**, both p = 0.01. The
  merge law arriving from a new direction.
- **The partial-name scan is refuted a third way.** An extractor asked, in the call that
  already reads every sentence, to report single-word references reaches **TP 5.6 against
  the scan's 18.0** at the same false-positive count (after `gate_pilots.py`'s 4.0-of-11.0
  and the 53.8-pair gap).
- **Four predicates have no prompt form** (`_iter_batches`, `_window`, `_unlinked`,
  `_union`): they are control flow over calls, not statements about text, and the only
  way to bind them is to let one call see everything, which `s_linker27` priced at macro
  F1 91.70. **So the floor on "how many hand-written rules" is above zero and it is
  structural.**
- **`s_linker68` — undecided, and the reason is a reading error worth recording.** s66
  minus the mention label's qualified-path value, the only consumer of
  `_all_occurrences_in_qualified_path`. Stage: **TP +/-0.0 (p = 1.00), FP -0.2
  (p = 1.00), composition p = 1.00**. Three paired runs
  (`../results/s68_e2e_r{1,2,3}_20260817`, batch stopped at three): the macro reads
  **TP -5.0 (p = 0.10 at the n=3 floor), macro F2 -1.6**, with every run agreeing — which
  is what a rejection looks like, and it was first written up here as one. It is not one.
  s68 and s66 send **byte-identical** knowledge, extraction and later-linker prompts, so
  the cut reaches only full-name verdicts on the 28 pairs that carried the deleted value.
  Decomposing where each lost gold link dies (control in brackets, s65 against its own
  null in the same set): **s68's extraction never proposed it 4.0/run [0.3]**, both
  proposed and s68's judge rejected 1.0 [0.7], neither proposed 0.7 [0.7]. Four fifths of
  the gap is at a stage the change cannot touch; it is entirely bigbluebutton (5.3/run,
  zero on three of five projects) and in r2 the two arms built identical alias tables and
  the gap persisted. None of the lost links carried the deleted value (4.0 `lowercase
  mention`, 1.0 `proper case`, 0.7 `indirect`). Restricted to the candidates **both** arms
  proposed — the only pairs the cut can reach — it is **TP -1.0 (p = 0.30), FP -0.3
  (p = 1.00)** against a control of TP -0.7 / FP -1.0. **Not adopted, not refuted.**
  The same set replicates s66 against s65 at **macro F1 -0.0 (p = 1.00)**, so the
  relocation has nine paired runs in two independent sets.
  **Standing rule, sharpened:** the partial-name round's rule ("read an arm on the
  `source` its change can reach, and use the stages it cannot reach as the control") was
  written after s62 read p = 0.00 at its source and p = 0.81 in the macro. Here the same
  rule reverses a *negative* macro rather than rescuing a positive one. **When a change's
  reachable surface is small — 28 pairs in one field — the macro is not the measurement,
  it is one stage's effect mixed with two stages of sampling, and reporting it as a
  verdict is the error.**
- **The deterministic layer is exhausted at `s_linker66`.** Every remaining element is
  measured and none can be removed or relocated at parity: the relation (identity over
  3697 pairs), the two tight scans (s67, TP -4.0), the partial-name scan (refuted three
  ways), `unique_owner` (frees 0.0 gold), `skip_when_named` (s46, F1 -1.5),
  `skip_qualified` with `_inside_qualified_identifier`/`_in_dotted_path` (`cutqualified`,
  FP +5.8), the five-value label (dropping it is -8.4 TP at the stage;
  s43 -1.3 F1, s44 -0.9 F1) with `_all_occurrences_in_qualified_path` still open (s68:
  neutral at its own source, undecided in the macro), `_states_a_name` at its two sites (s46; the antecedent gate
  at 12 FP), and the four structural predicates (no prompt form). **The answer to "how
  few rules can this workflow have" is one relation at four settings, one mention label,
  one name predicate and four structural predicates — with the admission contract written
  in English instead of code.**
- **B7 prices what is left before paying for it.** Freed candidates per run at the
  partial-name scan: dropping the span-boundary test 9.0 pairs / 1.0 gold (worth an arm —
  and after the label's qualified-path value goes, that test is the last consumer of
  `_inside_qualified_identifier` and of the documented `"" in "-_"` defect); dropping the
  unique-owner test 12.0 pairs / **0.0 gold** (cannot help; already priced at 2.4 FP, so
  not paid for); dropping the whole-name exclusion 151.0 / 127.0, which is not a cut but
  the alias table's suppression role (`s_linker46`, F1 -1.5). The same test on the
  spelling scan frees **0.0** pairs, so that row's `skip_qualified` is provably inert.
- New tooling, reusable for any pair of arms: `pilot/stage_diff.py` (stage
  populations, every unshared link attributed to its linker and to gold,
  `--alias-trace` splitting the alias stage into proposer and judge on the terms both
  arms propose, `--judge-trace` comparing verdicts on shared candidates,
  `--prompt-identity` checking the arms sent the same bytes) and
  `pilot/source_stats.py` (the same permutation test restricted to one linker's
  links, so an arm can be judged on the stages its change can actually reach).
  `pilot/source_stats.py` is what showed s50's headline "TP -3.0, p = 0.01" to be
  assembled from the full-name and partial-name stages, which its change cannot reach
  — on coreference, the only stage it touches, it reads TP +0.2 (p = 1.00).
  `pilot/test_s50_s51_prompts.py` and `pilot/test_s54_s55_prompts.py` assert that
  every arm differs from s49 only in the intended constants: all 52 method bodies and
  7 class attributes identical, every prompt builder rendering byte-identically after
  substituting s49's wording back, and no benchmark vocabulary introduced (GATE-06).

- `pilot/score_runs.py` — scores whole five-project run directories from their
  predicted-link CSVs and runs `ab_stats.permutation_report` on the pooled link sets,
  so per-run TP/FP/macro F1/macro F2, the composition statistic and the paired p
  values all come from one place instead of being assembled by hand per comparison.
  **Two alias judges (validity + usage) tested and not supported** (`s_linker39`,
  `s_linker40`): the usage judge is dominated -- on four projects it approves nothing
  the validity judge had not, and on JabRef its only unique admissions are `core` and
  `outer shell`, both false positives (macro F1 93.5). Asked four ways -- use
  confirmation, review-before-document, and two unions -- it never admits a real
  alias the validity judge missed; intersecting over-rejects and costs a third of
  MediaStore. **Alias judging is one judgment: the lenient, context-free one.**
  **General law from twelve variants: every consolidation of two LLM decisions into
  one call raises recall and lowers precision** -- self-review, carried judging at
  three thresholds, merged judging criteria; five instances, no exception. Splitting
  buys precision, merging buys recall, and s25 sits at the precision corner where F1
  rewards it. F2 does not distinguish s25 from s36 or s34.
- `src/llm_sad_sam/linkers/experimental/s_linker27.py` — `SLinker27`, standalone
  (experimental, **NOT promoted**). s26's merge without the batching: one call
  sends the whole document and returns both the references and the names. The
  smallest workflow of all — one prompt, one call. **Rejected, and the most
  informative of the three:** macro F1 91.70, and accuracy tracks document length
  (jabref 13 sents 100.0, mediastore 37 98.4, teastore 43 96.3, bigbluebutton 87
  **79.7**, teammates 198 **84.1**). On teammates the single call reported 50
  references where four 50-sentence batches report ~89. Batching buys
  thoroughness, not prompt-size relief (`../results/s27_singlecall_e2e_20260812`).
- `src/llm_sad_sam/linkers/experimental/s_linker28.py` — `SLinker28`, standalone
  (experimental, **NOT promoted**). s26 with the alias table no longer suppressing
  partial-name candidates, aimed squarely at the diagnosed dual-role effect — one
  condition fewer. **Recovers nothing:** macro F1 93.89 / F2 93.02 against s26's
  94.27 / 93.47 (`../results/s28_nosuppress_e2e_r{1,2,3}_20260812`). The dual role
  is real but is not what the merge costs.
- `src/llm_sad_sam/linkers/experimental/{helper_v3,ilinker3,__init__}.py`
- `src/llm_sad_sam/core/` — `data_types`, `data_types_v2`, `document_loader`,
  `document_loader_v2`, `model_analyzer`
- `src/llm_sad_sam/{llm_client,pcm_parser,pcm_parser_v2}.py`
- `run_s20union_*.sh` — legacy N=3 sweep runners (gpt / sonnet / re_medium /
  noknow), retained from the prior s20U trim.

`experimental/__init__.py` exports `SLinker21` and `SLinker21AgentRouter`; the
run path also imports submodules by full path via `importlib`, so eager
imports of the whole historical linker family are unnecessary.

The pilot investigation that produced `agentic_router.py`, `proposer.py`, and
`s_linker21_agentrouter.py` — feasibility probes, design-space sweeps, judge
experiments, and the measured numbers cited above — is archived at
`.planning/archive/router-pilot-260701/` (history preserved for previously
tracked files via `git mv`; `__pycache__` stripped). Look there for the full
narrative instead of duplicating it here.

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

### The compaction round (s88) — the prompt is mostly not rules

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
- **Open at the time of writing**: `resolve3` on luna (terra reads `notargetrows` at
  stage gold +8.7 / spurious -2.3, composed F1 +0.4, for **-23.8% of the resolver
  prompt**, and `nocasectx` at F1 +0.1 for -8.6%), and the `s_linker88` end-to-end batch
  (two of three paired runs a side; terra +2.8 macro F1 at FP -11.0, luna -1.3 at
  FP +17.0 with TP +3.5 — **not** a verdict at n=2).
