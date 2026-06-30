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

## Open / next

- **Tighten the direct linker** — the dominant FP source. Candidates: drop bare
  package-token enrolment (link only named classes/files), or validate each
  emitted link. This, not the router, is where precision is recoverable.
- Replicate on the Claude backend.
- Wire `augment_doc_code` into the real composition step (`build_unified.py`'s
  `build_aalinker`) rather than the offline eval harness.
