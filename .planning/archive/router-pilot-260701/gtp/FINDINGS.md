# GTP pilot — a grounded, typed, context-augmented PROPOSER, measured

The `PROPOSAL.md` (one dir up) argued the recall bottleneck is the **proposer**, not
the judge, and specified an LLM/structure/context proposer to replace the regex
`DirectCodeLinker`. It left ONE quantity unmeasured and flagged it explicitly:
**proposer precision** — where a real proposer lands between the +0.3pp reject-pool
floor and the +3.4pp perfect-proposer ceiling. This pilot measures it.

All numbers gpt-5.4, reasoning-off, single run. Cheap + cached (`proposer_cache.json`,
`e2e_cache.json`); re-runs offline. Files: `proposer.py` (GTP), `probe.py` (proposer
recall/precision), `e2e.py` (propose→route→judge).

---

## What is measured, and why it is the honest number

`fn_judge/router_ceiling.py` fed the **gold component** to the router and scored the
judge *assuming a perfect proposer* — an oracle. This pilot removes the oracle: GTP
is handed only the **sentence + previous sentence + the full component catalog**
(gold NOT leaked) and must choose. Scoring is against the real SAD-SAM gold
(benchmark CSVs).

**Target set** = the exact ceiling set from `fn_judge`: the 14 sentences whose gold
link s21 NEVER proposed (16 NP-FN gold pairs) + their 42 curated sibling distractors
(NP-CTRL). This isolates the never-proposed recall lever — 73% of s21's hard-core FN.

---

## 1. Proposer probe (`probe.py`) — recall vs over-proposal, gold not leaked

| catalog | NP-FN recovered | sibling over-proposal | full-gold P | hallucinated names |
|---|---:|---:|---:|---:|
| **name** (names only) | **11/16 (69%)** | **4/42 (10%)** | **0.792** | **0** |
| role (names + role lines) | 11/16 (69%) | 7/42 (17%) | 0.720 | 0 |

Three empirical findings:

1. **The proposer surfaces 69% of the never-proposed FN.** These are candidates
   s21's extraction never generated in any of 3 runs — GTP reaches 11 of 16 from
   the sentence + one line of context. The never-proposed gap is a proposer
   problem, and an LLM/structure/context proposer closes most of it. *Confirms
   `PROPOSAL.md §2` and `fn_judge §4`.*
2. **Grounding is hallucination-free: 0 dropped refs.** Every element GTP named
   resolved to a real catalog component — the regex stop-lists / `max_files` cap /
   `root_placeholder` hacks are unnecessary; structural existence in the catalog is
   the entire FP floor. *Confirms `PROPOSAL.md §3, §4.3`.*
3. **"Context must constrain, not enrich" holds on the PROPOSER side too.** Adding
   role descriptions to the catalog did **not** raise recall (11/16 both) and
   **raised** sibling over-proposal 10%→17% (precision 0.792→0.720). `fn_judge §8`
   found role profiles backfire for the *judge*; this extends it to the *proposer*.
   **Ship names-only.**

The 5 proposer misses (name mode) are all the implicit "HTML5 Server" /
WebRTC-SFU cases (bbb s6/s39/s47/s73) — the sibling-ambiguity cluster `fn_judge §9`
called gold-debatable and "unreachable by any proposer." Independent confirmation
from the proposer side.

---

## 2. End-to-end (`e2e.py`) — propose → route (GTP's own mode) → judge

Each GTP proposal is routed by the mode GTP emitted to the matching specialized
judge (`fn_judge/router_judge.py`), anchors fed to the context judge. Deployable
recall = what survives.

| pipeline | NP-FN kept | sibling leak | kept-P |
|---|---:|---:|---:|
| **GTP proposer → router → judge (real, no oracle)** | **8/16 (50%)** | **2/42 (5%)** | **0.824** |
| oracle baseline: router fed the GOLD component (`fn_judge §7`) | 8/16 (50%) | 0/42 (0%) | — |

**The real proposer matches the oracle-proposer's recovery (8/16).** Feeding GTP's
own choices instead of the gold component costs only **2 sibling leaks (5%)** — the
proposer is not the bottleneck the ceiling analysis feared; it surfaces precisely
the recoverable candidates, and the router still gates precision.

- **8 kept:** teammates Logic ×3 (s7/s8/s185, IMPLICIT→context judge), GAE Datastore
  ×2 (s122/s138, IMPLICIT), mediastore DB (s33, CONTRAST), bbb FreeSWITCH (s66,
  AFFIRMATIVE→strict gate), bbb HTML5 Client (s19, AFFIRMATIVE).
- **8 lost:** the HTML5-Server / WebRTC-SFU sibling-ambiguity cluster — 3 the
  proposer never surfaced, 5 the coref/context gate correctly held (ANAPHORA on the
  implicit "the server"). These are the gold-debatable residual (`transarc-emp`
  annotation-bias pillar); the oracle router loses the same ones.
- **2 sibling leaks:** bbb s39 BBB web, s73 FreeSWITCH — both IMPLICIT-routed
  context-judge admits. The realized precision knob is the context judge, exactly
  where `fn_judge §3` located it.

---

## 3. Where this lands the PROPOSAL

- The proposal's untested claim is now bracketed by measurement: a real
  grounded/typed/names-only proposer recovers **8/16 (50%)** of the never-proposed
  FN at **5% sibling leak** — i.e. it realizes essentially the full oracle-router
  recovery on this set. The gap to the +3.4pp *perfect* ceiling is the
  gold-debatable HTML5-Server residual, not proposer weakness.
- **Names-only, not role-augmented** is the deployable catalog (measured, both legs).
- **Grounding replaces the regex/stop-list surface with zero hallucinations** —
  the direct-route rewrite (`PROPOSAL.md` Increment B) is structurally justified.

## 4. Honest limits (what this does NOT show)

- **Single backend, single run.** gpt-5.4 only; no run-to-run stability, no Claude
  replication (D-04). Needed before any paper claim.
- **Ceiling set only (14 sentences, bbb/teammates-heavy).** This measures the
  never-proposed *recall lever in isolation* — the proposer analogue of how
  `fn_judge` measured the judge in isolation. It is **not** a full-corpus macro-F1.
- **The corpus F1 delta still needs the live 3× pipeline run** (wire GTP into
  `build_unified.py`'s `build_aalinker`, re-score RQ1–4) — the spend gate in
  `PROPOSAL.md §8.3`. This probe de-risks that run: the proposer lever is real and
  precision-safe on the hardest set, so the live run is now worth its cost.

## 5. FULL LIVE RUN — corpus macro-F1 and the proposer × judge design space

`probe.py`/`e2e.py` measured the never-proposed *ceiling set* (14 sentences). This is
the **full-corpus** run (`live_run.py`, `design_space.py`): GTP proposes over all 378
sentences of all 5 projects, grounded to the **real PCM roster** (14/11/8/12/6 comps —
the same roster s21 uses, not the study subset), routed + judged, unioned into the
frozen s21 finals, rescored model-doc macro-F1 (sentence, component) vs SAD-SAM gold,
5 proj × 3 runs. Scoring reproduces the s21 baseline **exactly** (P 0.9894 / R 0.8913
/ F1 0.9360), so deltas are trustworthy. Corpus-wide grounding held: **461 proposals,
0 hallucinated names.**

### The combinatory design space (proposer aggressiveness × judge firmness)

The proposer and judge are not a fixed pipeline: a firmer judge lets the proposer run
at lower precision. Sweep on the SAME cached proposals, per-run marginal scoring:

| proposer | judge | P | R | F1 | ΔF1 | addTP | addFP |
|---|---|---:|---:|---:|---:|---:|---:|
| affirm (AFFIRM only) | none | 0.9490 | 0.9037 | 0.9232 | −0.0129 | 2 | 8 |
| affirm | strict | 0.9897 | 0.9014 | 0.9420 | +0.0059 | 2 | 0 |
| affirm | routed | 0.9896 | 0.9002 | 0.9413 | +0.0053 | 1 | 0 |
| named (AFFIRM+CONTRAST) | none | 0.9432 | 0.9208 | 0.9295 | −0.0066 | 5 | 10 |
| named | strict | 0.9897 | 0.9120 | 0.9479 | +0.0119 | 4 | 0 |
| **named** | **routed** | **0.9897** | **0.9173** | **0.9506** | **+0.0146** | **4** | **0** |
| all (+IMPLICIT/ANAPHORA) | none | 0.5630 | 0.9828 | 0.7021 | −0.2339 | 17 | 219 |
| all | strict | 0.9489 | 0.9218 | 0.9339 | −0.0021 | 4 | 6 |
| all | routed | 0.8707 | 0.9575 | 0.9083 | −0.0278 | 11 | 32 |

### What the grid establishes

1. **Best deployable cell: `named` proposer + `routed` judge → F1 0.9506, +1.46pp,
   at ZERO precision loss** (0.9894→0.9897). Recall +2.6pp (0.8913→0.9173). This is a
   real, no-regress corpus gain — better than the reject-pool floor (+0.3/+1.07pp)
   the earlier pilot bracketed.
2. **The judge is what makes a proposer usable — the `none` column proves it.** The
   aggressive proposer alone floods **219 FP → precision 0.563, F1 0.702**. Unusable
   raw.
3. **The combinatory hypothesis is CONFIRMED (`all` row).** A firm (strict) judge
   turns that 0.563-precision proposer into **0.949** (219 FP → 6): the firm judge
   absorbs a sloppy proposer. The routed judge, being permissive on IMPLICIT/ANAPHORA,
   only reaches 0.871 (32 FP). So *if the judge is firm, the proposer can be
   aggressive* — exactly the decoupling.
4. **…but volume+filter loses to quality.** Even rescued, aggressive+strict (0.9339)
   < precise+routed (0.9506). Corpus-wide, the IMPLICIT/ANAPHORA modes over-generate
   FP that no reasoning-off judge fully cleans (the ceiling set was a favorable
   subset). The deployable proposer is **name-grounded** (AFFIRM+CONTRAST), not fully
   aggressive.
5. **Two kinds of s21 miss.** (a) A few affirmatively-named links s21's extraction
   dropped — GTP recovers these cleanly (+4 TP / 0 FP). (b) Many implicit/anaphoric
   links — genuinely hard, proposing them floods FP; the gold-debatable frontier.
   The win is (a); (b) is where recall saturates without precision damage.

**Deployment rule (the answer to "which combination"):** match proposer precision to
judge firmness. Ship the **name-grounded proposer + routed judge** (+1.46pp F1, no
precision regress). Reserve the aggressive proposer for a max-recall setting, and only
behind the firm strict gate (−0.2pp — recall up, F1 flat).

### Limits

Single backend (gpt-5.4), single GTP pass unioned against 3 s21 runs (GTP not itself
run ×3 — low-temp, but run-to-run stability unmeasured). Claude/Sonnet replication
(D-04) pending. Doc→code file-level F1 (ArCoTL composition) not scored here — this is
the model-doc (sentence, component) level where the recall gap lives.

## 6. Reproduce

```bash
cd approach/pilot/gtp
python3 probe.py               # proposer recall/precision, ceiling set (cached)
python3 e2e.py name            # propose -> route -> judge, ceiling set (cached)
python3 live_run.py --baseline-only   # free: reproduces s21 baseline 0.9360
python3 live_run.py           # full-corpus GTP augmentation (one cell)
python3 design_space.py       # the full proposer x judge grid
```
Delete `*_cache.json` + set `OPENAI_API_KEY` (+ `OPENAI_MODEL_NAME=gpt-5.4`) to
re-run live. Gold + PCM roster from the ARDoCo benchmark.

## 7. Remaining FN after the best cell — where the wins are (`live_run` caches)

After `named+routed` (recall 0.9173), **18 of 195 gold links remain FN** (union over
runs). Classified by GTP's disposition:

| # | category | meaning |
|---|---|---|
| 7 | implicit/anaphora, routed judge KEEPS, filtered by `named` | recoverable *in principle* — but admitting the mode brings the FP flood |
| 5 | implicit/anaphora, judge REJECTS | proposed, judge declined (FileStorage s33, Logic s8, HTML5 s10/s19, WebRTC s65) |
| 5 | GTP never proposed | the "BigBlueButton server" cluster (HTML5 Server s6/s39/s47/s73, WebRTC s73) |
| 1 | named, judge rejects | teammates s88 Logic (ambiguous; strict-gate coin-flip) |

**The implicit mode is annotation-bounded, not method-bounded.** Of the 48
implicit/anaphora candidates the routed judge keeps (the `all+routed` extra):
13 gold : 35 FP. An **anchor gate fails** — 34/35 FP components are *also* named
elsewhere, so "named elsewhere" cannot separate them (drops 1 FP). fn_judge already
showed evidence-typing / self-consistency / skeptic also fail. And the 35 "FP" read
as **gold-incompleteness**: mediastore s2/s10 UserManagement (registration/auth),
teammates s24/s27 UI ("It is … the front-end", "written in Angular"), teammates
s43–46 Logic (servlet processing). Many are defensible links the gold omits.

### Possible wins, ranked

1. **Gold-incompleteness audit (highest value, precision-safe, paper-aligned).** The
   recall ceiling here is gold-bound, not method-bound. Hand-audit the 35 entangled
   "FP"; reclassify the defensible ones FP→gold. Directly feeds the `transarc-emp`
   benchmark-bias pillar and retroactively lifts the aggressive config's precision.
2. **Doc→code direct route (Increment B), the real METHOD win.** Several
   FN/FP are code-naming sentences (teammates s33/s39/s43/s44 name WebPageServlet,
   WebApiServlet, web.xml) — the sentence→code target space, not model-doc. GTP's
   CODEPATH mode + the `.acm` index handles these; the model-doc route structurally
   cannot. Complementary metric (file-level), specced in PROPOSAL §7 Increment B.
3. **Sibling-pair proposer** for the 5 never-proposed HTML5-Server/WebRTC cases:
   propose both siblings, pairwise-discriminate. Low yield (~2/5) — the oracle judge
   (`router_ceiling`) loses the same ones; gold-debatable.
4. **Judge tuning** on the 5 implicit-rejects — marginal, risky.

**Bottom line:** on the model-doc metric, `named+routed` (+1.46pp, R 0.917) is near
the precision-safe ceiling. The remaining recall is ~13 annotation-bounded implicit
links + 5 gold-debatable siblings + a few judge coin-flips. The two real levers are
(1) fixing the gold and (2) the separate doc→code route — not a better model-doc judge.

## 8. Verified anatomy of the 35 entangled "FP" (gold-checked)

Rule-based classification (gold adjacency, code-identifier regex, is-sentence-gold),
9 residual hand-checked. This is the precision "cost" of the aggressive config, verified:

| category | n | verified meaning | evidence |
|---|---:|---|---|
| **Displacement** | 11 | component is gold on an *adjacent* sentence continuing the same description | S24 "It…front-end"→UI (gold S25); S79 "Managing relationships…cascade logic"→Logic (gold S77/78, role-exact) |
| **Code-structure** | 10 | names a code class/package → **doc→code route**, correctly not model-doc gold | S43/44 WebApiServlet/ActionFactory→Logic; S39/40 WebPageServlet→UI; S178/179 x.search/x.webapi |
| **Dual/secondary** | 5 | sentence *is* gold for a co-referenced sibling; GTP tagged the other real referent | S79 "The conversion process…to the client via Redis pubsub": gold={HTML5 Client,Redis PubSub}, GTP=Presentation Conversion (subject; gold S80/81) |
| **Other** | 9 | ~6 defensible role/secondary refs, ~3 genuinely over-eager | over-eager: teastore S31 "logged in"→Auth; teammates S36 "the Web browser"→UI; S165 "report to instructor"→UI |

**Verdict (verified aggregate of 35):**
- **~3 (9%) genuine method errors** — the judge actually over-firing.
- **~22 (63%) gold-incompleteness** — defensible links the benchmark omits (proven by
  gold on the adjacent sentence, gold on the co-referenced sibling, or exact role match).
- **~10 (29%) code-structure** — correctly outside model-doc gold; the doc→code route's
  target population.

So the aggressive config's P 0.99→0.87 "crash" is ~9% real error, ~63% annotation bias,
~29% wrong-task. The model-doc recall ceiling is **gold-bound**; and 10/35 are literal
code-naming sentences — independent re-motivation of the doc→code route (`PROPOSAL §7`).
A FP→gold reclassification list is emit-able for a corrected-gold rescore.

## 9. Can we push precision on the judge side? — decided: NO (it's gold-bound)

Question: the agentic router sits at P 0.9592 (vs named+routed 0.9897). Push precision,
maybe judge-side? Verified answer: **judge-side precision on the model-doc metric is
exhausted.**

1. **Nothing legitimate to remove.** The agentic router's *new* accept-FP are **0 real
   errors, 4 gold-incompleteness** — it already rejected all 3 verified errors. Every
   measured FP is a valid link the benchmark omits.
2. **A skeptic judge is indiscriminate (measured, `precision_push.py`).** An adversarial
   default-refute pass on the 8 new accepts **removed 2 real TP (DB, FreeSWITCH — the
   core recoveries) and only 1 of 4 gold-gap FP; kept 3 gold-gaps.** Valid-implicit and
   gold-incomplete-implicit are the same linguistic kind, so the skeptic deletes them at
   random — hurting recall more than precision. Same recall/precision coupling `fn_judge
   §8` found.

**Decision — precision is a SCOPE knob, not a judge knob:**
- **Need measured P ≥ 0.99? Deploy `named+routed` (P 0.9897).** It holds precision
  *structurally* by admitting only name-present candidates. The agentic router's lower P
  is the price of admitting implicit references — set by WHAT you admit, not judge
  strictness.
- **The one real judge-side precision lever is cross-run CONSENSUS** (accept only if
  proposed+validated in ≥2 of N GTP runs). It removes flaky/lucky passes (real FP)
  without deleting *systematic* valid links — the one mechanism not coupled to recall.
  Needs multi-run GTP (untested; the recommended next experiment).
- **The measured-precision gap is mostly gold-incompleteness** — the honest "fix" is the
  gold audit (raises measured P correctly), not a stricter judge.
