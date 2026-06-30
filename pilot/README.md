# Router + direct sentence→code linking (pilot, `router` branch)

## Problem

The canonical doc-to-code result is **composed**:
`sentence → component` (our LLM model-doc linker) ∘ `component → code` (ArCoTL,
deterministic). This transitive route is powerful but **structurally blind** to
documentation that describes *code organisation* rather than architecture —
package inventories, class names, config files. Those sentences have no
architecture component to route through, so the composition emits nothing.

A failure analysis of the GPT-5.4 S21 slot found **~23% of doc-code gold links
(121 links, all in teammates) unreachable by transitive composition — recall
exactly 0** on them.

## Fix: a per-sentence router + a direct route

`src/llm_sad_sam/linkers/experimental/router_direct.py`

- **Router** (`SentenceRouter`, LLM zero-shot; or free `rule_route`): decides per
  sentence whether to also use the direct route. Taboo-safe — sentence text only,
  no component names.
- **Direct linker** (`DirectCodeLinker`): `sentence → code` without the component
  step. Final links = `transitive ∪ direct`.

### How direct linking works — package structure, **not** grep

There is **no raw source tree** in the benchmark, only the `.acm` code model. So
direct linking is package/code-model structure matching:

1. parse `.acm` → one `CodeUnit` per compilation unit (package path + class + ext);
2. extract identifiers from the sentence — CamelCase classes (`WebApiServlet`),
   dotted packages (`logic.api`), files (`web.xml`);
3. resolve each against a class/package/file index and emit the file path(s). A
   package token enrols every unit beneath it (mirrors gold enrolment); a class
   token resolves to that unit (+ its `*Test` twin).

Only identifiers that resolve to a real compilation unit are emitted → the
precision gate lives on the linker output, where the pilot showed it belongs.

## Result (gpt-5.4_s21, macro over 5 projects × 3 runs)

`pilot/router_eval_full.py` (full-document routing — the honest config) and
`pilot/router_eval.py` (gold-sentences-only, kept for the contrast):

| config | P | R | F1 | direct prec | +TP | +FP |
|---|---|---|---|---|---|---|
| baseline transitive | 0.9630 | 0.8590 | 0.9063 | — | — | — |
| + direct, **rule** router (all sentences) | 0.9564 | 0.8814 | 0.9163 | 0.812 | +905 | +211 |
| + direct, **LLM** router (all sentences) | 0.9589 | 0.8814 | 0.9174 | 0.827 | +905 | +191 |
| + direct, LLM router (gold sentences only)* | 0.9628 | 0.8814 | 0.9192 | 0.925 | +905 | +74 |

\* optimistic — forces every non-gold sentence to ARCH, so it never exposes
router FP on no-gold sentences. **Not a deployable config**; shown only to mark
the gap the full-document row closes.

- Recall **+2.24 pts** regardless of router (teammates R 0.735→0.847, F1
  0.818→0.883); the four clean projects stay flat. The recall win is the **direct
  linker**, not the router.
- Honest full-document LLM router: F1 **+0.011**, precision −0.4 pt. Of its +191
  FP, **151 come from no-gold sentences** — the exposure the gold-only eval hid.
- Over full documents the LLM router only **marginally** beats the free rule
  router (P 0.9589 vs 0.9564): it drops ~20 FP at no TP cost. The earlier
  "suppresses 65% of FP" was a gold-only artifact.
- **The precision lever is the direct linker** (standalone precision ~0.81–0.83),
  not the router.

## Adding a judge (`DirectLinkJudge`, `pilot/router_eval_judge.py`)

The model-doc side gets its 0.99 precision from an LLM validation pass. The same
mechanism ports to the direct route: a claim-before-verdict judge over each
`(sentence, identifier)` candidate (quote the words asserting the link, else
reject). Result (gpt-5.4_s21, 90 candidates judged, 19 rejected):

| config | P | R | F1 | direct prec |
|---|---|---|---|---|
| + direct (no judge) | 0.9564 | 0.8814 | 0.9163 | 0.812 |
| + direct + judge | 0.9582 | 0.8814 | 0.9171 | 0.823 |

The judge does exactly what a validity gate should — all 19 rejections are
`class` candidates, ~17 of them the **product-name-as-class collision** (the
literal token `BigBlueButton` matching a class named `BigBlueButton`), plus
negations/asides — at a cost of only 2 lost TP. But the aggregate lift is small,
and the FP breakdown by candidate kind says why:

| kind | emitted | FP | precision |
|---|---|---|---|
| class | 78 | 36 | 0.538 |
| package | 1048 | 178 | 0.830 |

`class` precision is what the judge repairs (0.54 → ~0.7+). But **178 of the 214
FP are `package` candidates** whose reference *is* valid — the judge keeps them
correctly. Their FP is **enrolment granularity** (a package token expands to every
file under it, but gold includes only some), not validity. A judge cannot fix
that; only tighter linker granularity can.

## FP provenance — the remaining "FP" are mostly gold artifacts, not errors

Before tightening package granularity we traced every package FP. All 178 come
from **4 (sentence, token) pairs**, none of them over-enrolment:

| source | FP | example |
|---|---|---|
| gold incompleteness | 110 | `e2e.cases`@s190 *"e2e.cases contains test cases."* — a real reference; gold annotates the overview sentence s187 instead and leaves s190 empty. The judge keeps it (correctly). |
| doc↔code naming drift | 68 | `client.scripts` (doc, plural) vs `client/script/` (code, singular). |

A size cap cannot separate these from correct links — they are 2-segment,
correctly-scoped packages whose files are *right*. Measured directly:

| package-size cap | direct TP | direct FP |
|---|---|---|
| ≤30 (kills all FP) | 268 | 0 |
| none | 870 | 178 |

Killing the 178 FP costs **602 TP** (~3.4 TP lost per FP removed), because the
big genuine packages (`common.datatransfer`=107 TP, `e2e.cases`@s187=74 TP, …)
die with the artifacts. **A granularity gate is net-harmful and was not added.**

Conclusion: after the judge handles class-collision validity, the direct route's
residual imprecision is **bounded by the gold standard, not the method** — the
package "FP" are genuine trace links the benchmark under-annotates. This is the
same enrolment/annotation bias the evaluation pillar (`transarc-emp`) documents.

## Remaining recall gaps, categorized (`pilot/remaining_recall.py`)

After the best config (transitive ∪ direct-judged), 39 gold (sentence, component)
links remain fully missed. Attributed by stage (ids align across all three):

| stage | (sent,comp) misses |
|---|---|
| ArCoTL bridge (component→code) | **0** |
| model-doc (sentence→component) | **39 (100%)** |

The bridge is **not** a recall bottleneck — whatever model-doc resolves, ArCoTL
delivers. The entire residual gap is the sentence→component step. By linguistic
mode:

| mode | n | example |
|---|---|---|
| code-structure (names code, not architecture) | 17 | s172 *"Sub-packages contains x.testdriver, x.datatransfer, …"* |
| implicit / functional (no lexical anchor) | 10 | mediastore s33 *"…stored in a specific location (e.g. a file server or database)"* |
| generic common-noun (component name lower-cased) | 7 | teammates s7/s8 *"…application back end logic…"* → Logic |
| negation / contrastive | 3 | teastore s7 *"…not provided by the WebUi, but…"* → WebUI |
| anaphora / coreference | 2 | teastore s26 *"As such, it also acts as a caching layer."* → Persistence |

Addressability:
- The **17 code-structure** misses are the direct route's target but slip through
  on fixable issues: the doc's `x.` root-package placeholder (`x.logic`, `x.search`,
  `x.webapi` — 8 of the 17 don't resolve), the judge over-rejecting a real link
  (s32 `OriginCheckFilter`), and class names absent from the code index. Headroom
  for the direct route: `x.`-root normalisation + a softer judge.
- The **22 others** (implicit/generic/negation/anaphora) are model-doc linguistic
  failures with **no code identifier** — the direct route cannot help; they need
  model-doc-side work (coreference, generic-term grounding, negation handling).
- Concentration: teammates carries the code-structure misses; bigbluebutton the
  implicit/generic/sibling-ambiguity ones (HTML5 Client↔Server, FreeSWITCH↔FSESL,
  often missed as a multi-component pair).

## Direct-route fixes — implemented & measured

Two fixes targeting the 17 code-structure misses (gpt-5.4_s21, macro 5×3):

| config | P | R | F1 | (s,comp) misses |
|---|---|---|---|---|
| + direct + judge (strict, prior) | 0.9582 | 0.8814 | 0.9171 | 39 |
| + direct + judge (**softened — default**) | **0.9592** | 0.8814 | **0.9176** | 38 |
| + `x.`-root placeholder (**opt-in**) | 0.9458 | 0.8893 | 0.9149 | 33 |

1. **Judge softening (default).** The prompt now keeps an element named as a
   concrete example (`"such as X"`, `"e.g. X"`) and rejects only exclusions
   (negation/contrast) or product/brand-name collisions. Strict win: recovers the
   over-rejected `OriginCheckFilter`@s32, still rejects 22/24 `BigBlueButton`
   product-name hits, FP 196→188, F1 +0.0005.

2. **`x.`-root placeholder normalisation (`DirectCodeLinker(root_placeholder=True)`,
   off by default).** Treats a leading single-char package segment as the project
   root (`x.logic`→`logic`), so the doc's placeholder packages resolve. Recovers 5
   more code-structure misses (38→33; R +0.0079) but the 1-segment suffix match
   (`x.util`→every `util` package) over-fires, costing file-level precision
   (F1 −0.0027). A root-anchored match (suffix only at the project root) is the
   proper fix; left as future work. teammates-specific.

## Open / next

- Root-anchor the `x.` fallback so it resolves without the precision hit.
- Model-doc-side: the 22 implicit/generic/negation/anaphora misses need coref and
  generic-term grounding, not the direct route.
- Claude replication; wire `augment_doc_code` + `DirectLinkJudge` into
  `build_unified.py`'s `build_aalinker`.
- Replicate on the Claude backend.
- Wire `augment_doc_code` into the real composition step (`build_unified.py`'s
  `build_aalinker`) rather than the offline eval harness.
