# Proposal — a grounded, typed, context-augmented PROPOSER (+ routing)

> Forward design that supersedes the pilot's regex `DirectCodeLinker`. Grounded in
> the pilot's own numbers (`FINDINGS.md`, `fn_judge/FINDINGS.md`). Direction fixed
> by the goal: **LLM call / structure / context-augmented — never keyword,
> hard-coded rules, or benchmark words.** Design + evaluation plan, **now piloted**:
> the core proposer is built and measured in `gtp/` (`gtp/FINDINGS.md`, see §6). The
> live full-corpus verification remains a spend gate (§8.3).

---

## 0. TL;DR

The pilot proved two things that point the same way:

1. **The recall gap is a *proposer* problem, not a judge problem.** Of the 22
   hard-core model-doc FN (missed in all 3 gpt-5.4 runs), **16 (73%) were
   NEVER-PROPOSED** — no judge ever saw them (`fn_judge/FINDINGS.md §1`). The
   perfect-proposer ceiling is **F1 0.9701 (+3.4pp)**; the realistic
   reject-pool-only gain is **+0.3pp**. *The entire gap between them is proposer
   precision* (`fn_judge §9`).
2. **The current second proposer is the wrong kind of mechanism.** The direct
   `sentence→code` route is a pile of regexes (`_CAMEL`, `_DOTTED`, `_FILE`) and
   hard-coded stop-lists (`_LAYOUT_SEGMENTS`, `_DOTTED_STOP`, the
   `max_files_per_package` cap, the `root_placeholder` hack). It works
   (recall +2.24pt in teammates) but it is exactly the keyword/rule surface the
   goal says to remove.

**Proposal: one LLM proposer that replaces all four candidate sources**
(Framing-C extraction, coref, the regex direct-linker, and the standalone
sentence-router) **with a single *grounded, typed, context-augmented* read that
also emits its own route.** Recall comes from proposing the hard modes; precision
comes from (a) *grounding* every proposal against real targets and (b) *routing*
each proposal by its emitted mode to a mode-calibrated judge — never from
loosening a global rubric (which the pilot showed is dead on arrival, −7pp F1).

Three legs, each tied to a pilot finding:

| leg | what it is | pilot evidence it rests on |
|---|---|---|
| **Grounded** | a candidate is emitted only if its target resolves in the component list *or* the `.acm` code index (given to the LLM as a structured index, not grepped) | §9 "binding constraint is proposer *precision*"; grounding is what bounds proposer FP without rules |
| **Context-augmented (constraining)** | each sentence carries its anchor window (prev/next + section header), nothing more | `fn_judge §3`: anchors cut distractor leak **18%→7% at zero recall cost**; role profiles *backfired* to 43% ("context must constrain, not enrich") |
| **Typed / routed** | the proposer labels each candidate with a reference **mode** (AFFIRMATIVE / CONTRAST / IMPLICIT / ANAPHORA / CODEPATH); the mode is the routing key | `fn_judge §6, §8`: the judge-router recovers **10 of 22 FN at 10% leak, 0% sibling over-link** — "precision comes from ROUTING, not clever judges" |

---

## 1. Where we are — two proposers, both limited

The shipped doc→code linker is **composed**:
`sentence→component` (the s21 model-doc linker) ∘ `component→code` (ArCoTL,
deterministic). Two proposer surfaces feed the model-doc side, and the pilot
bolted on a third:

| proposer | stage | mechanism today | its recall gap |
|---|---|---|---|
| **Framing-C extraction** | s21 Phase 2, 2-pass union | one flat LLM call per 50-sentence batch: *"Extract ALL references to components."* No per-sentence context, no mode label | the **16 never-proposed FN** — implicit / generic-lowercase / anaphora / affirmatively-named-but-missed |
| **Coref** | s21 Phase 5 | LLM pronoun/noun-phrase resolution with a context window | recovers some anaphora; 1 hard-core FN still lost here |
| **Direct-code (pilot)** | `router_direct.py` | **regex identifier extraction** + code-index resolution + hard-coded stop-lists | catches teammates code-structure sentences (+2.24pt R) but *is* the keyword/rule surface to remove; standalone precision only ~0.81–0.83 |

Bridge check: ArCoTL (component→code) loses **0** of the residual FN
(`FINDINGS.md §6`). The bottleneck is entirely on the **sentence→{component,code}**
proposer side. That is what this proposal rebuilds.

---

## 2. Why the proposer, not the judge (the settled question)

`fn_judge/` swept the judge design space exhaustively (global-lenient,
self-consistency vote, evidence-typed, grounded, skeptic-verify, judge-router) and
concluded:

- A **global-lenient** judge buys recall at a precision cliff: it approves the FN
  *and* 76% of real distractors → **−7pp F1**. Dead on arrival.
- **Self-consistency does not rescue it**: a uniformly-lenient rubric agrees with
  itself 3/3, so majority-vote leaves leakage at 76% (`fn_judge §3`).
- The **judge-router** is the frontier structure (recovers 10/22 FN at 10% leak),
  but on the *reject pool alone* even it nets only **+0.3pp F1** at −2.7pp
  precision — because that pool is small and gold-bounded.
- The prize is the **16 never-proposed FN**. A judge cannot recover a candidate it
  never receives. On that set the router approves 43–50% *when handed them* — so
  the lever is **a proposer that surfaces them**, then routes them through the
  *unchanged strict gate* so recall is added at s21's own precision profile and
  **cannot regress by construction** (`fn_judge §9`).

So: recall is proposer-bound, and the honest no-regress path is a *better
proposer* + *routing*, not a looser judge. This proposal builds exactly that.

---

## 3. Design principles (each maps to a measured finding)

1. **Propose into both target spaces from one read.** Components *and* code units
   are candidate targets. A single per-sentence LLM read emits both — replacing
   the model-doc extractor *and* the regex direct-linker. Unifies the two pilot
   threads.
2. **Ground, don't extract.** A candidate survives only if its target resolves to
   a real component or a real compilation unit in the `.acm` index. Grounding is
   the structure leg and the FP bound — it does the job the regex stop-lists and
   the `max_files_per_package` cap did, but by *structural existence*, not
   pattern rules.
3. **Context must pin a referent, not justify a link.** Feed the anchor window
   (prev/next sentence + nearest section header). Do **not** feed component role
   profiles — the pilot measured that as a precision *backfire* (leak 43%).
4. **The proposer emits its own route.** Each candidate carries a reference mode.
   The mode is the routing key into a mode-calibrated judge. No separate
   pre-classifier read (the full-document sentence-router was shown to be nearly
   free / redundant, only ~20 FP dropped over the rule router).
5. **Reasoning-off, evidence-first.** One structured pass; the only "reasoning"
   field is the **anchor quote** the proposer must cite for each candidate —
   identical in spirit to s21's claim-before-verdict. No `OPENAI_REASONING_EFFORT`,
   no thinking (repo rule: linker gains come from prompt/structure alone).

---

## 4. The proposer — Grounded Typed Proposer (GTP)

### 4.1 Interface

```
GTP.propose(sentence, anchors, component_index, code_index) -> list[Candidate]

Candidate = {
  sentence_id : str,
  target      : {space: "COMPONENT"|"CODE", id: <component name | resolved path(s)>},
  mode        : "AFFIRMATIVE"|"CONTRAST"|"IMPLICIT"|"ANAPHORA"|"CODEPATH",
  anchor      : str,     # exact quote the proposer must cite as evidence
}
```

- **`component_index`** — the runtime component list (as s21 already passes it).
- **`code_index`** — a *structured* view of the `.acm`: the set of package paths,
  class stems, and file names present, handed to the LLM as reference context
  (the same `CodeIndex` the pilot built, but used for *grounding a proposal*, not
  for regex resolution). The LLM names a target; the index confirms it exists.
- **Grounding rule:** drop any candidate whose `target.id` does not resolve in the
  named index. This is the entire precision floor of the proposer — no stop-lists,
  no caps, no `root_placeholder` heuristic.

### 4.2 The prompt (taboo-safe sketch)

One call per sentence-batch, structured output. The template carries only generic
English + the *runtime* component/code identifiers (exactly as s21's validator
already injects runtime component names — taboo-safe by construction).

```
You read one software-documentation sentence with its surrounding context and
list every element it references. Elements come from two catalogs:
  COMPONENTS: {runtime component names}
  CODE UNITS: {package / class / file identifiers present in the code model}

For EACH reference, output: the element, which catalog it is in, the reference
MODE, and the exact words that carry the reference.

MODE is how the sentence points at the element:
  AFFIRMATIVE — named or clearly described as present/used/provided/implemented.
  CONTRAST    — stated in a contrast or exception ("other than X", "unlike X")
                but still asserting a fact about X.
  IMPLICIT    — referred to by role/function with no literal name
                (resolve via the CONTEXT window, not a guess).
  ANAPHORA    — referred to by a pronoun/"the <role>" pointing back to context.
  CODEPATH    — a package/class/file identifier naming a code unit.
Only list an element that appears in a catalog above. Quote the words; if you
cannot quote them, do not list it.

CONTEXT: {prev sentence} {SECTION HEADER} >>> {target sentence} <<< {next sentence}
Return JSON: {"refs":[{"element","catalog","mode","quote"}]}
```

Reasoning-off: the `quote` is the sole justification field, mandatory, and used
downstream as the anchor. A ref with no quotable anchor is not emitted.

### 4.3 How GTP subsumes the regex direct-linker, failure-mode by failure-mode

| pilot regex failure | regex "fix" (a rule) | GTP instead |
|---|---|---|
| `x.logic` root-placeholder didn't resolve | `root_placeholder=True` suffix hack (cost −0.27pp F1) | LLM names `logic`; grounding resolves it structurally; context distinguishes a real root placeholder from a `util` collision — no suffix rule |
| `BigBlueButton` product-name == class name | judge post-hoc rejects 22/24 | proposer tags it CODEPATH only if used as a code reference, else it is simply not emitted (no product name in a catalog match → dropped at proposal) |
| class-token precision 0.538 | — | grounded + context-typed proposal; the CODEPATH judge sees the anchor quote, not a bare token |
| package FP = gold incompleteness / naming drift (178 FP, 3.4 TP each) | *rejected* a size cap (net-harmful) | **unchanged** — these are gold artifacts (the `transarc-emp` pillar). GTP keeps them correctly; do not "fix" gold |

Deleted surface (net −): `_CAMEL`, `_DOTTED`, `_FILE`, `_DOTTED_STOP`,
`_LAYOUT_SEGMENTS`, `max_files_per_package`, `root_placeholder`,
`extract_mentions`, `rule_route`. The `.acm` parsing + `CodeIndex` *existence*
lookups stay — that is structure, not keyword rules.

---

## 5. Routing — the proposer is the router

The mode label GTP emits is the routing key. No extra classification read.

| mode | routed judge | rationale (measured) |
|---|---|---|
| AFFIRMATIVE | s21 strict two-pass (unchanged) | never-proposed-but-named FN (e.g. bbb s66 `FreeSWITCH`) enter the *unchanged* gate → recall added at s21's precision, cannot regress (`fn_judge §9`) |
| CONTRAST | contrast judge (approve fact-about-X even in exception) | recovers teastore s7 WebUI, bbb s59 FreeSWITCH |
| IMPLICIT | context judge (anchors pin the referent) | anchors held **7% leak** vs 18% bare (`fn_judge §3`) |
| ANAPHORA | coref gate (unchanged; soften later) | teastore s26 "it"→Persistence |
| CODEPATH | direct-code validity judge (the pilot `DirectLinkJudge`) | class-collision + negation handling |
| *(un-grounded)* | dropped at proposal | grounding is the first gate |

This is the "also routed" the goal asks about — but the elegant form: **the router
is folded into the proposer**, so a candidate arrives *pre-typed* with its
evidence anchor. One proposer read replaces (extraction ∥ coref) + sentence-router
+ regex-linker, and the judges become thin per-mode validators.

Precision safety, restated: distractors route to AFFIRMATIVE(strict) or are
dropped un-grounded; only genuinely mode-flagged, anchor-quoted candidates reach a
lenient judge. This is the structure that gave **10/22 FN at 10% leak, 0% sibling
over-link** — recall decoupled from precision.

---

## 6. Operating point — now MEASURED (`gtp/`, gpt-5.4, reasoning-off)

The one quantity this proposal left open — **proposer precision**, "where GTP
lands between +0.3 and +3.4pp" — was measured by building GTP and running it on
the ceiling set with the gold NOT leaked (`gtp/FINDINGS.md`). Results:

- **Proposer alone** recovers **11/16 (69%)** of the never-proposed FN at **4/42
  (10%)** sibling over-proposal, **0 hallucinated names** (grounding held). Names-
  only beat role-augmented catalog (10% vs 17% over-proposal, same recall) —
  "constrain, not enrich" confirmed on the proposer side.
- **End-to-end** (propose → route by GTP's own mode → judge) keeps **8/16 (50%)**
  of the never-proposed FN at **2/42 (5%)** sibling leak, kept-precision 0.824.
  This **matches the oracle baseline** (the router fed the gold component recovered
  8/16 at 0% leak, `fn_judge §7`) — a real proposer realizes essentially the full
  oracle-router recovery for a 2-sibling precision cost.
- **Bracket, restated with the measurement inside it.** Floor +0.3pp (reject-pool
  router) / +1.07pp (J0_amb); ceiling F1 0.9701 (+3.4pp, perfect proposer). GTP
  reaches the oracle-router recovery on the hardest (never-proposed) lever, so the
  realistic corpus gain is bounded below by the reject-pool floor **plus** the
  never-proposed recall this pilot demonstrated — the remaining gap to the ceiling
  is the gold-debatable residual, not proposer weakness.
- **The 8 unrecovered FN are the gold-debatable cluster** (implicit "the
  BigBlueButton server" → HTML5 Server, WebRTC-SFU sibling-ambiguity) — 3 the
  proposer never surfaced, 5 the strict coref/context gate correctly held. The
  oracle router loses the same ones. Report as gold-incompleteness (`transarc-emp`).

**Corpus macro-F1 — now also MEASURED (full live run, `gtp/live_run.py`,
`gtp/design_space.py`).** GTP over all 378 sentences, grounded to the real PCM
roster, unioned into the frozen s21 finals, rescored 5×3 (baseline reproduces s21
0.9360 exactly; 461 proposals, 0 hallucinations):

- **Best cell: name-grounded proposer + routed judge → F1 0.9506, +1.46pp, at ZERO
  precision loss** (0.9894→0.9897; R +2.6pp). A real, no-regress corpus gain.
- **The proposer × judge grid is a genuine design space** (D-05, below): the judge
  is what makes a proposer usable (aggressive proposer *raw* = 0.563 precision, 219
  FP), and *a firm judge lets the proposer be aggressive* — the strict s21 gate turns
  that 0.563-precision proposer into 0.949 (219 FP → 6). But quality beats
  volume+filter: precise+routed (0.9506) > aggressive+strict (0.9339). The IMPLICIT/
  ANAPHORA modes over-generate FP corpus-wide, so the deployable proposer is
  **name-grounded** (AFFIRMATIVE+CONTRAST), not fully aggressive.

**Decision D-05 — deployable config: name-grounded GTP proposer + routed judge.**
Match proposer precision to judge firmness; reserve the aggressive (all-modes)
proposer for a max-recall setting behind the firm strict gate only.

Still open: Claude/Sonnet replication (D-04); GTP run ×3 for stability; doc→code
file-level F1 via ArCoTL composition.

---

## 7. Scope split — two deployable increments

To keep each step measurable and reversible:

- **Increment A (model-doc side, the bigger prize).** GTP replaces Framing-C
  extraction; components-only target space; AFFIRMATIVE candidates → unchanged
  strict gate; IMPLICIT/CONTRAST/ANAPHORA → routed judges. Target: recover the 16
  never-proposed + 6 rejected FN. Success metric: macro (sentence,component) F1 on
  the 5×3 gpt-5.4 slot, vs s21 baseline 0.9360, precision floor 0.985.
- **Increment B (direct-code side).** GTP's CODE catalog + CODEPATH mode replaces
  the regex `DirectCodeLinker`; feeds `augment_doc_code`. Target: match or beat
  the pilot's doc→code F1 0.9176 **without** the regex/stop-list surface. Success
  metric: doc→code file-level F1, direct-route standalone precision > 0.83 (the
  regex linker's ceiling).

A and B share the one proposer prompt (two catalogs); they differ only in which
target space + judge is wired. Ship A first (no-regress by construction on the
AFFIRMATIVE tier), then B.

---

## 8. Evaluation plan (offline-first; live run is a spend gate)

Reuse the pilot harness and caches — no new infra:

1. **Proposer probe — DONE (`gtp/probe.py`, cached).** GTP run on the ceiling set
   (14 never-proposed sentences, gold not leaked): 11/16 FN surfaced at 10% sibling
   over-proposal, 0 hallucinations, names-only > role. See §6 / `gtp/FINDINGS.md`.
2. **End-to-end propose→route→judge — DONE (`gtp/e2e.py`, cached).** 8/16 FN kept
   at 5% sibling leak, matching the oracle-router baseline (`fn_judge §7`). This is
   the offline analogue of `precision_impact.py`, but with a *real* proposer instead
   of gold-fed candidates.
3. **Live full-corpus run (SPEND GATE — do not launch without sign-off).** The
   remaining unmeasured quantity is the corpus **macro-F1 delta**. Wire GTP into
   `build_unified.py`'s `build_aalinker`, re-run the full pipeline **3× on gpt-5.4**
   (then Claude/Sonnet replication per D-04), score all four RQs. Only step that
   costs real budget; the §8.1–8.2 pilot de-risks it (proposer lever is real and
   precision-safe on the hardest set). Estimate first, mirror the 260628-dnl sweep.

Reasoning stays **off** throughout (repo rule; s21 is a no-reasoning config).

---

## 9. Risks & failure modes (from the pilot's own backfires)

- **Context enrichment backfire.** If GTP's context grows past anchors into
  role/profile grounding, precision drops (measured: leak 18%→43%). Mitigation:
  hard-cap context to the anchor window + section header. Constrain, don't enrich.
- **Grounding too generous.** A package token enrolling every file under it is
  correct for gold but inflates file-level FP that are *gold artifacts*, not
  errors. Do **not** add a granularity cap (net-harmful, −3.4 TP per FP removed).
  Report as gold-incompleteness.
- **Over-correction.** The skeptic/propose→verify variant *over-corrected recall*
  (5 FN remaining but leak-trading). Keep the strict AFFIRMATIVE tier unchanged;
  do not add a global verify pass.
- **Single backend / single run.** All pilot numbers are one gpt-5.4 slot.
  Increment A/B must replicate on Claude/Sonnet (D-04) and report run-to-run
  stability before any paper claim.
- **Mode-label noise.** If the proposer mislabels mode, routing sends a candidate
  to the wrong judge. Mitigation: AFFIRMATIVE is the safe default (strict gate);
  mislabeling *toward* strict costs recall, not precision — the safe direction.

---

## 10. Open questions / next

- **Batch vs per-sentence context.** s21 extracts in 50-sentence batches; GTP
  needs per-sentence anchors. Batch the *proposal* but attach each sentence's own
  anchor window — measure token cost vs the current 2-pass extraction.
- **Ambiguous-but-named sub-route.** The router still over-rejects
  teammates Logic (ambiguous name → strict). An "ambiguous-but-named" tier that
  relaxes when an architectural verb is present (`fn_judge §6`) — reasoning-free —
  is worth a label-level test before wiring.
- **Drop the 2-pass union?** GTP is one grounded read; the Framing-C 2-pass union
  exists to raise recall by sampling twice. Test whether grounded+typed single-pass
  matches the union's recall at lower cost.
- **First concrete step:** implement `GroundedTypedProposer` alongside
  `router_direct.py` (do not delete the regex path yet), run the §8.1 offline
  label-level gate against `fn_judge/cases.json`, and only then decide on the live
  gate.
```
