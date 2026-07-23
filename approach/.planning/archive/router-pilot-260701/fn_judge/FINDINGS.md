# Can a judge recover s21's remaining recall? — FN → judge probe

**Question.** s21 (gpt-5.4) is precision-saturated (macro P 0.989) with recall the sole
bottleneck (macro R 0.891). Goal: take every remaining false negative and feed it to a judge —
under the current structure and under *different* judge structures from our new knowledge — to
see whether a judge can approve them, and at what precision cost. **All judges are
reasoning-off** (s21 is a no-reasoning config; the claim-before-verdict output field is the only
"reasoning", and it is part of the s21 design).

Scripts: `fn_taxonomy.py` (where each FN is lost), `build_cases.py` (labelled case universe),
`run_judges.py` (the judge structures, cached in `verdicts.json`), `report.py` (scoring).
Data: gpt-5.4 s21 slot, 3 runs, 5 ARDoCo projects. Gold = SAD-SAM (sentence, component).

---

## 1. Where the recall is lost (FN taxonomy)

s21 gpt-5.4: gold = 195 model-doc links; per-run FN ≈ 28; **22 FN missed in all 3 runs**
(the hard core). Splitting the 22 by the stage that lost them:

| bucket | n | meaning |
|---|---|---|
| **NEVER-PROPOSED** | **16** | extraction *and* coref never surfaced the candidate — no judge ever saw it |
| ENTITY-REJECTED | 5 | proposed as an entity candidate, the two-pass entity validator rejected it |
| COREF-REJECTED | 1 | proposed by coref, the coref validator rejected it |

**Only 6 of 22 (27%) were ever seen by a judge.** The other 73% are a *proposal* problem, not a
judge problem. The 6 seen-and-rejected FN fall into clean linguistic modes:

| FN | mode | why s21 rejected |
|---|---|---|
| teastore s7 WebUI ("not provided by the WebUi, but…") | **negation** | P1 fail — mention is negated |
| bbb s59 FreeSWITCH ("systems other than FreeSWITCH") | **contrast** | P1+P2 fail — contrastive mention |
| bbb s79 HTML5 Client ("…messages to the client…") | **generic/lowercase** | P2 fail — "the client" not specific |
| teammates s88 Logic ("Logic is a Facade class…") | **ambiguous-name** | P1 fail — flagged ambiguous |
| mediastore s33 FileStorage ("stored in… a file server…") | **implicit + ambiguous** | P1 fail — no literal name |
| teastore s26 Persistence ("As such, it also acts as…") | **anaphora** | coref gate rejected "it" |

The negation/contrast rejects are the rubric working *as designed* (it explicitly rejects
negated/contrastive mentions to kill FP) — yet the gold standard counts them as links.

---

## 2. Judge structures tested (all reasoning-off, gpt-5.4)

| id | structure | from |
|---|---|---|
| **J0_s21** | s21 replica: layered rubric, entity 2-pass (P1∧P2)/coref 1-pass, claim-before-verdict, bare context | current s21 |
| **J0_amb** | J0 + s21's real evidence bundle (ambiguity flag + anchor sentences on the entity pass) | s21 in-run context |
| **J1_soft** | softened rubric: approve concrete examples, reject only exclusion/negation/product-name | pilot `DirectLinkJudge` |
| **J2_recover** | recall rubric: approve contrastive/negated/example/generic/anaphoric mentions + richer context | new (recall-framed) |
| **J3_vote** | self-consistency: J2 rubric sampled K=3 @ temp 1.0, majority vote | survey (self-consistency) |

Cases (distinct, run-independent): **R-TP** 50 = gold links s21's validator rejected (recall
target); **R-TN** 83 = non-gold links it correctly rejected (precision control, *real*
distractors); **NP-FN** 16 = never-proposed gold (ceiling test); **NP-CTRL** 42 = sibling
distractors for the ceiling.

---

## 3. Result — the recall/precision ladder

Approval rate (want HIGH on R-TP/NP-FN = recall, LOW on R-TN/NP-CTRL = precision cost):

| judge | R-TP (recall) | R-TN (leak) | NP-FN (ceiling) | NP-CTRL (over) |
|---|---:|---:|---:|---:|
| J0_s21   | 90% | 18% | 6%  | 0% |
| J0_amb   | 90% | **7%** | 6%  | 0% |
| J1_soft  | 94% | 24% | 44% | 5% |
| J2_recover | **100%** | 76% | 75% | 10% |
| J3_vote (majority) | **100%** | 76% | 75% | 10% |

**Headline on the 22 real FN** (consistent = missed in all 3 runs; the actual recall gap):

| judge | remaining FN approved / 22 | R-TN leak |
|---|---:|---:|
| J0_s21 (bare rubric)   | 6  (27%) | 18% |
| J0_amb (s21 evidence bundle) | 4  (18%) | **7%** |
| J1_soft                | 10 (45%) | 24% |
| J2_recover             | **18 (82%)** | 76% |
| J3_vote (majority)     | 18 (82%) | 76% |

On the **6 seen-and-rejected FN**, J2 approves **all 6**; J1_soft 4; J0_amb 3 (the two
*ambiguous* ones — FileStorage, Logic — it still rejects, because the ambiguity flag tells it
to). On the **16 never-proposed FN**, J2 approves **12/16**; the 4 it never recovers are all
the implicit "BigBlueButton server"→HTML5 Server sibling-ambiguity cases (gold-debatable).

**Self-consistency (J3) is not a precision fix here.** Majority-vote over K=3 samples of the
recall rubric leaves leakage at 76% — a consistently-lenient rubric stays lenient across
samples, so voting removes no distractors. The precision problem is the rubric, not variance.

---

## 4. Findings

1. **Almost every remaining FN is judge-approvable in principle.** A recall-framed judge
   (J2, reasoning-off) approves **6/6** seen-and-rejected FN and **12/16** never-proposed FN.
   The sentences *do* support the links to a lenient reader — the judge rubric is not a hard
   wall for most of the gap.

2. **…but you cannot buy that recall by loosening the global judge.** The very leniency that
   approves the FN also approves **76% of the real distractors** (R-TN) and 10% of sibling
   controls. Recall and precision are coupled through one knob — judge leniency. J2 would move
   recall up and precision off a cliff. A global soften is *not* deployable — and self-consistency
   voting (J3) does not rescue it: the lenient rubric agrees with itself across samples, so
   majority vote leaves leakage at 76%.

3. **The precision lever is CONTEXT, not rubric strictness.** Adding s21's evidence bundle
   (ambiguity flag + anchor sentences) to the *same* rubric cut distractor leakage **18% → 7%
   at zero recall cost** (J0 → J0_amb). Richer evidence discriminates a real sibling from a
   distractor better than a stricter rule does. This is where headroom lives.

4. **The 16 never-proposed FN are an EXTRACTION problem, not a judge problem.** They were never
   candidates, yet a judge approves most when handed them (J1 43%, J2 75%). Example: bbb s66
   *"FreeSWITCH can also be integrated with VOIP providers…"* names FreeSWITCH affirmatively as
   the subject, yet s21 never proposed it. Fix = a recall-oriented *proposer*, then a
   context-rich judge to hold precision — not a looser judge.

5. **A residual ~4 FN are gold-debatable.** The implicit HTML5 Server cases ("the BigBlueButton
   server", "applications running on the server") are rejected even by the lenient judge; they
   are the sibling-ambiguity / annotation-bias cases the `transarc-emp` pillar documents.

---

## 5. Recommendation (reasoning-free, precision-safe)

Do **not** globally soften the validator. Instead, split the gap by its two real causes:

- **Recover the 6 seen-and-rejected FN with TARGETED mode handling, not global leniency.** They
  are three specific modes: (a) negation/contrast — approve the component when the sentence
  asserts a fact *about* it even in contrast; (b) generic-lowercase — resolve "the client"/"the
  logic" via anchor sentences; (c) the coref "it" case. A mode-scoped rubric tweak recovers
  these while the strict default still rejects the 75%-leak distractors J2 would admit.
- **Recover the ~12 never-proposed FN on the EXTRACTION side** (recall-oriented proposer /
  second extraction pass for implicit + generic + sibling mentions), gated by a
  **context-rich** judge (J0_amb style: ambiguity flag + anchor sentences) — which held 7% leak.
- **Verify the precision cost end-to-end** by re-running the full pipeline with the changed
  gate, not just the offline probe (this probe measures the judge in isolation).

Net: the recall ceiling reachable via the judge is high, but the deployable path is
*context-richer judge + recall-oriented proposer + mode-targeted rubric*, not a looser judge.

---

## 6. The elegant realization — an LLM JUDGE-ROUTER (`router_judge.py`)

The mode-targeting from §5 shouldn't be a pile of hard-coded rules. A single reasoning-off LLM
**router** reads GENERAL, taboo-safe signals (how the component is referenced) and dispatches
each candidate to a specialized judge — the router, not a global rubric, becomes the precision
gate. Modes: `AFFIRMATIVE`→strict s21 two-pass; `CONTRAST`→contrast judge; `IMPLICIT`→context
judge (anchor sentences fix the referent); `ANAPHORA`→coref gate; `CODEPATH`/`ABSENT`→reject.

**Result vs the global judges** (gpt-5.4, reasoning-off, same 191 cases):

| approach | remaining FN / 22 | R-TN leak | NP-CTRL over |
|---|---:|---:|---:|
| J0_amb (strict + context) | 4 (18%) | 7% | 0% |
| J2_recover (global lenient) | 18 (82%) | 76% | 10% |
| **Judge-router** | **12 (55%)** | **10%** | **0%** |

The router **triples the strict baseline's recovery (4 → 12 FN) for +3pp leakage**, versus the
global lenient judge's unusable 76%. It decouples recall from precision: distractors route to
`ABSENT`/`CODEPATH` and auto-reject (R-TN 83 → only 8 leak; NP-CTRL 42 → 38 rejected as
`ABSENT`, **0% over-link**), while real FN route to the `CONTRAST`/`IMPLICIT`/`ANAPHORA` judges
that approve them (`teastore s7 WebUI`→CONTRAST✓, `mediastore s33 FileStorage`→IMPLICIT✓,
`teammates s7/122/138/185`→IMPLICIT✓).

**Where the router still loses the other 10 FN — the headroom:**
- **3 route to `ABSENT`** (bbb s6/s39/s47 HTML5 Server, the implicit "BigBlueButton server"
  cases) — the router sees no reference; gold-debatable, J2 misses them too.
- **2 ambiguous-name over-strict** (teammates s8/s88 Logic route `AFFIRMATIVE`→strict gate
  rejects, inheriting s21's ambiguity-strictness) — fixable with an "ambiguous-but-named"
  sub-route that relaxes when an architectural verb is present.
- **3 sibling-ambiguity** (WebRTC-SFU / HTML5 Client `IMPLICIT`, referent not pinnable) +
  **2 anaphora** (`ANAPHORA` coref gate still strict) — softening those two routes recovers more.

So the elegant architecture already works as a proof of concept; the next lift is tuning the
per-route rubrics (ambiguous-named, anaphora) — reasoning-free — plus a recall-oriented proposer
so the never-proposed FN actually reach the router.

**Verify next:** wire the router+judges into the s21 gate and re-run the full pipeline (3 runs)
to confirm the offline 12/22 @ 10%-leak translates to a real macro-F1 recall gain at held
precision.

---

## 7. Concrete precision impact — the second-chance deployment (`precision_impact.py`)

To turn "leak %" into real precision, simulate the deployable move: run s21, then re-judge every
candidate it REJECTED and add back what the judge approves. Macro P/R/F1, 5 proj × 3 runs
(reject-pool only — the 16 never-proposed FN are NOT candidates here, so unreachable this way):

| approach | macro P | macro R | macro F1 | reject-pool added | ΔF1 |
|---|---:|---:|---:|---:|---:|
| baseline s21 | 0.9894 | 0.8913 | 0.9360 | — | — |
| **J0_amb** (strict+context) | 0.9790 | 0.9196 | **0.9468** | +17 TP / +9 FP | **+0.0107** |
| router (judge-router) | 0.9623 | 0.9207 | 0.9389 | +19 TP / +14 FP | +0.0029 |
| J2_recover (global lenient) | 0.8179 | 0.9328 | 0.8659 | +26 TP / +139 FP | −0.0701 |

**How the leak hurts precision, concretely:** each re-approved distractor is a new false
positive. J0_amb adds 9 FP → P 0.989→0.979 (−1.0pp). The router adds 14 FP → P 0.989→0.962
(−2.7pp). J2's "76% leak" = 139 FP → P 0.989→0.818 (−17pp, dead on arrival).

**Honest nuance — on the reject pool alone, strict+context WINS.** J0_amb recovers 17 TP for
only 9 FP (1.9:1) → +1.07pp F1; the router recovers 19 TP for 14 FP (1.4:1) → +0.29pp. The
reject pool is mostly *flaky borderline* links a context-rich strict judge re-approves cleanly,
so the router's extra leniency costs more than it gains **there**.

**Where the router earns its keep is the NEVER-PROPOSED FN** (16 of the 22, and the bigger
prize). On that set: router approves 8/16 (50%) at **0% sibling over-link**; J0_amb only 1/16
(6%). A proposer that surfaces those implicit/generic/contrast candidates needs a judge that is
lenient *on exactly those modes* while staying strict elsewhere — that is the router, and
J0_amb's uniform strictness cannot do it. So:

- **Reject pool (6 seen-and-rejected FN):** ship **J0_amb** (context-rich strict re-judge) —
  +1.07pp macro-F1, recall 0.891→0.920, precision −1pp. Simple and safe.
- **Never-proposed FN (16):** add a recall-oriented proposer **gated by the router** — the only
  structure that recovers implicit/generic mentions (50% vs 6%) without over-linking siblings.
- **Never** the global lenient judge (−7pp F1).

---

## 8. Can any judge design get NO precision regress? — design-space sweep

Chasing a judge that recovers reject-pool FN with **zero** precision loss, I swept the axes the
problem actually has — judge STRUCTURE, PROMPT, CONTEXT augmentation, ROUTER — all reasoning-off,
all prompt/structure (no regex, no code fallbacks). Label metrics on the 191 cases:

| design | axis | remaining FN /22 | R-TN leak | verdict |
|---|---|---:|---:|---|
| global lenient (J2) | prompt (recall rubric) | 18 | 76% | precision dead |
| **judge-router (v1)** | **router** | **12** | **9%** | **frontier** |
| router + skeptic verify | structure (propose→verify) | 5 | 2% | over-corrects recall |
| roster+profile grounded | context augmentation | 12 | 43% | worse — more hooks to approve |
| router + grounded lenient | router + context | 11 | 18% | worse than v1 |
| router + evidence-typed (NAMED+RESOLVED) | prompt (calibration) | 10 | 18% | RESOLVED not clean |
| router + evidence-typed (NAMED only) | prompt (calibration) | 9 | 6% | precision-safe, less recall |
| router + self-consistency (unanimous 3/3) | structure (agreement) | 16 | 32% | judge is *confidently* wrong |

**Four things this establishes:**

1. **The router's precision comes from ROUTING, not clever judges.** Distractors are sent to
   `AFFIRMATIVE`(strict) / `ABSENT`·`CODEPATH`(reject); only genuinely mode-flagged candidates
   reach a lenient judge. No refinement of the *lenient judges* beat plain router-v1 — grounding,
   evidence-typing, and skeptic all either over-corrected recall or added leak.

2. **Context augmentation can BACKFIRE.** Giving the judge each component's role profile made
   precision *worse* (leak 43%): more grounding = more hooks to rationalize a link. Anchors
   (§3) helped because they *pin a referent*; role profiles hurt because they *justify*. Context
   must constrain, not enrich.

3. **The residual distractors are confidently link-like — not low-confidence guesses.** Sampled
   3×, the recall rubric votes 3/3 or 0/3, almost never split (26/33 sticky distractors are
   unanimous). So self-consistency / calibration cannot filter them. And several are **gold
   incompleteness**: `browser`→UI is in s21's *own* LLM-built alias table, yet the gold omits it
   — the same annotation bias the `transarc-emp` pillar documents.

4. **True zero-regress on the reject pool is mathematically impossible.** To hold P ≥ 0.9894
   while adding links, the added links must be ≥98.9% precise; the best design manages ~65%
   (NAMED-only: 13 TP / 10 FP → −0.7pp). Any reject-pool re-judge dilutes precision.

**Deployment frontier** (`precision_impact.py`, macro 5×3, reject-pool re-judge):

| design | macro P | macro R | macro F1 | ΔF1 | precision hit |
|---|---:|---:|---:|---:|---:|
| baseline s21 | 0.9894 | 0.8913 | 0.9360 | — | — |
| **J0_amb** (strict + context) | 0.9790 | 0.9196 | **0.9468** | **+0.0107** | −1.0pp |
| router + NAMED-only | 0.9743 | 0.9141 | 0.9417 | +0.0056 | **−0.7pp** |
| router (v1) | 0.9623 | 0.9207 | 0.9389 | +0.0029 | −2.7pp |
| global lenient (J2) | 0.8179 | 0.9328 | 0.8659 | −0.0701 | −17pp |

### Verdict — the elegant no-regress design

- **The judge-router is the right STRUCTURE** (routing is the precision gate) and the elegant,
  LLM-driven answer. Deploy it with strict `AFFIRMATIVE` + `NAMED`-evidence-only lenient tiers
  for the smallest controlled regress (**−0.7pp precision, +0.6pp F1**).
- **But no judge is truly zero-regress on the reject pool** — it is gold-bounded and
  mathematically capped. The reject pool is the wrong place to chase no-regress recall.
- **The real no-regress lever is the ROUTER applied to a better PROPOSER:** surface the
  affirmatively-named never-proposed FN (e.g. bbb s66 `FreeSWITCH`, never proposed though named
  as the subject) and route them through the *unchanged strict gate* — that adds recall at s21's
  own precision profile, so it does not regress by construction.
- **Report the residual leak honestly as partial gold-incompleteness** (alias-table-supported
  links the benchmark omits) — consistent with the evaluation pillar.

---

## 9. Router recall ceiling — feed ALL false negatives (`router_ceiling.py`)

If a perfect proposer surfaced every gold FN and routed it through the judge-router (all inputs
gold → approvals add only TP → precision cannot regress):

| config | macro P | macro R | macro F1 | vs baseline |
|---|---:|---:|---:|---|
| baseline s21 | 0.9894 | 0.8913 | 0.9360 | — |
| **router, all FN in (v1)** | **0.9905** | **0.9520** | **0.9701** | R **+6.1pp**, F1 **+3.4pp** |
| router, all FN in (NAMED-tier) | 0.9902 | 0.9378 | 0.9617 | R +4.7pp, F1 +2.6pp |

- The router **approves 52% of all FN** (44/84 over 5×3); precision **holds (~0.99, a hair up)**.
- **Split by reachability:** reject-pool FN (already candidates) **73% approved (19/26)** —
  reachable now; never-proposed FN **43% approved (25/58)** — need a proposer.
- **Caps at 52%:** the router rejects the other 48% (implicit HTML5-Server, sibling-ambiguity,
  anaphora) *even when handed them* — the gold-debatable residual, unreachable by any proposer.

**Two caveats.** (1) This is the *perfect-proposer ceiling* — a real proposer also surfaces
non-gold distractors, which leak at the router's measured rate; the realistic reject-pool number
is +19 TP / +14 FP → +0.3pp F1 at −2.7pp precision. The gap between the +3.4pp ceiling and the
realistic figure is **entirely proposer precision**. (2) So the binding constraint is the
proposer, not the judge — engineer the proposer's precision next.
