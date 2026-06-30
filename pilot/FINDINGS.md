# Router + direct sentence→code linking — investigation & findings

Full write-up of the `router`-branch pilot: motivation, feasibility study,
implementation, results, a rejected approach, and the residual recall analysis.
`README.md` is the quick reference; this is the narrative + reproduction guide.
All numbers are the **GPT-5.4 S21** slot, macro-averaged over the 5 ARDoCo
projects (× 3 runs where applicable). Matching is the paper's, via
`mono/evaluation/mini-src/metrics.py` (enrolled file-level gold).

---

## TL;DR

The shipped doc-to-code linker is **composed**: `sentence→component` (our LLM
model-doc linker) ∘ `component→code` (ArCoTL, deterministic). That transitive
route is structurally blind to documentation describing *code organisation*
(package/class/file names) — ~23% of doc-code gold links (all in teammates) are
unreachable, at recall 0.

We added a **per-sentence router** + a **direct `sentence→code` route** + a
**keep/reject judge**. Final links = `transitive ∪ direct(judged)`.

| config | P | R | F1 |
|---|---|---|---|
| transitive baseline | 0.9630 | 0.8590 | 0.9063 |
| + direct + judge (**default**) | 0.9592 | 0.8814 | **0.9176** |
| + `x.`-root placeholder (opt-in) | 0.9458 | 0.8893 | 0.9149 |

Recall **+2.24 pts** at F1 **+0.011**, precision essentially held. Two findings
shaped the design: (a) router over-fire is nearly free — the precision gate
belongs on the *linker/judge*, not the router; (b) the residual "FP" are largely
**gold-standard artifacts** (incompleteness + naming drift), not method errors.

---

## 1. Motivation — the transitive blind spot

A failure analysis of the baseline (`analysis/fn_analysis.py`, `fn_overlap.py`,
`probe.py`) found doc-code misses fall into two groups: links cascaded from
model-doc misses, and a distinct **below-the-architecture** group — sentences
like *"Package overview contains logic.api, logic.core"* or *"…throws
InvalidParametersException"* that name code, not architecture. These have no
component to route through, so the composition emits nothing.

`analysis/opportunity.py` sized it: **121 doc-code gold links (23% of the corpus,
all teammates) have recall exactly 0** under transitive composition; the other
four projects have zero such sentences.

## 2. Feasibility pilot — can an LLM decide the route?

- **Decidability** (`analysis/router_pilot.py`): gpt-5.4, reading only the
  sentence (zero-shot, taboo-safe), flags the direct-only sentences with
  **recall 0.96**, barely firing on the clean projects.
- **Over-fire is cheap** (`analysis/overfire_value.py`): re-scored by outcome, 57
  of 75 CODE firings land on sentences with still-missed gold (beneficial); the
  rest are no-ops. Router *precision* against a binary label is the wrong metric.
- **Actionability** (`analysis/direct_recoverability.py`): 78% of the locked
  links name their target package/class verbatim → a structure-match linker can
  recover them.

## 3. Implementation — `src/.../experimental/router_direct.py`

- **`DirectCodeLinker`** — `sentence→code` by **package/code-model structure**,
  not grep (there is no source tree in the benchmark, only the `.acm` code
  model). Extract identifiers (CamelCase classes, dotted packages, files) →
  resolve against a class/package/file index → emit paths. Only identifiers that
  resolve to a real compilation unit are emitted.
- **`SentenceRouter`** — LLM zero-shot router (or the free `rule_route` = CODE iff
  the linker resolves something).
- **`DirectLinkJudge`** — mirrors `s_linker21`'s validation pass: claim-before-
  verdict keep/reject per `(sentence, identifier)`. The softened prompt keeps
  concrete examples (`"such as X"`) and rejects only exclusions / product-name
  collisions.
- **`augment_doc_code`** — `transitive ∪ direct(CODE-routed, judged)`.

Why the precision gate is on the judge, not the router: both routers add the same
TP; the LLM router/judge only changes which FP survive. The direct linker's
standalone precision is ~0.82; the judge lifts the deployed config to ~0.96.

## 4. End-to-end results

`pilot/router_eval.py` (gold-sentence router), `router_eval_full.py` (full-
document router — closes the FP-exposure gap), `router_eval_judge.py` (judge).

- Recall **+2.24 pts** regardless of router; the gain is the direct linker's,
  concentrated entirely in teammates (R 0.735→0.847, F1 0.818→0.883).
- Full-document LLM router (honest config): P 0.9589, +191 FP, 151 of them from
  no-gold sentences. Only marginally better than the free rule router — earlier
  "65% FP suppression" was a gold-only artifact.
- Judge softening (default): recovers the over-rejected `OriginCheckFilter`@s32,
  still rejects 22/24 `BigBlueButton` product-name hits; P 0.9582→0.9592.

## 5. Rejected — package-granularity gate

`analysis` traced every package FP: all 178 come from **4 (sentence, token)
pairs**, none over-enrolment — **110 gold incompleteness** (descriptive *"e2e.cases
contains test cases"* sentences the gold leaves unannotated; the judge keeps them
correctly) + **68 naming drift** (`client.scripts` vs `client/script/`). A size
cap that removes all 178 FP also drops direct TP **870→268** (~3.4 TP per FP),
because the same big packages are genuine elsewhere. **Net-harmful; not added.**
The residual imprecision is gold-bounded — the same annotation bias the
`transarc-emp` evaluation pillar documents.

## 6. Remaining recall gaps (`pilot/remaining_recall.py`)

After the best config, 38 gold `(sentence, component)` links remain fully missed.
By stage (ids align across all three):

| stage | misses |
|---|---|
| ArCoTL bridge (component→code) | **0** |
| model-doc (sentence→component) | **38 (100%)** |

**The bridge is not a recall bottleneck.** By linguistic mode: ~17 code-structure
(direct-route headroom: `x.`-placeholder, judge tuning), and ~22
implicit/generic/negation/anaphora that contain no code identifier and need
model-doc-side work (coref, generic-term grounding) — the direct route cannot
help these.

## 7. Reproducing

Self-contained: LLM decisions are cached in `pilot/cache/` (tracked), so the
evals re-run offline with no API calls. External inputs are the ARDoCo benchmark
and the `sota/recovered-links` S21 dump (absolute paths in the scripts —
prototype-grade). To re-route/re-judge from scratch, delete the relevant cache
and set `OPENAI_API_KEY` (+ `OPENAI_MODEL_NAME=gpt-5.4`).

| script | produces |
|---|---|
| `analysis/fn_analysis.py` | baseline FN, model-doc + doc-code |
| `analysis/fn_overlap.py` | model-doc→doc-code cascade overlap |
| `analysis/probe.py` | teammates code-structure probe |
| `analysis/opportunity.py` | direct-route opportunity sizing (121 links @ R=0) |
| `analysis/router_pilot.py` | LLM router feasibility (recall 0.96) |
| `analysis/direct_recoverability.py` | 78% of locked links are name-recoverable |
| `analysis/overfire_value.py` | router over-fire re-scored by link outcome |
| `router_eval.py` / `router_eval_full.py` | router end-to-end (gold / full-doc) |
| `router_eval_judge.py` | + judge precision recovery |
| `remaining_recall.py` | residual-gap categorization by stage + mode |
| `cache/router_cache*.json`, `judge_cache*.json` | cached gpt-5.4 decisions |

## 8. Limitations & next steps

- One project (teammates) carries the entire direct-route opportunity — the value
  is project-dependent, not universal.
- Single backend (gpt-5.4), single router/judge run; no run-to-run stability.
- Labels are gold-derived proxies; "dual" code+arch sentences blur the binary.
- Next: root-anchor the `x.` fallback (recall without the precision hit); Claude
  replication; wire `augment_doc_code` + `DirectLinkJudge` into
  `sota/recovered-links/build_unified.py`'s `build_aalinker`.
