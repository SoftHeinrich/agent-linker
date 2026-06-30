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

## Result (`pilot/router_eval.py`, gpt-5.4_s21, macro over 5 projects × 3 runs)

| config | P | R | F1 | direct precision | new TP | new FP |
|---|---|---|---|---|---|---|
| baseline transitive | 0.9630 | 0.8590 | 0.9063 | — | — | — |
| + direct, **rule** router (all sentences) | 0.9564 | 0.8814 | 0.9163 | 0.812 | +905 | +211 |
| + direct, **LLM** router (cached) | 0.9628 | 0.8814 | 0.9192 | 0.925 | +905 | +74 |

- Recall **+2.24 pts** (teammates R 0.735→0.847, F1 0.818→0.883); other projects flat.
- Both routers add the same TP; the **LLM router suppresses ~65% of the rule
  router's FP with no TP loss** — that is the router's entire value.

## Open / next

- FP exposure on sentences with **no** doc-code gold is only partially covered
  (rule router runs over all sentences; LLM router was evaluated on gold
  sentences). Run the LLM router over the full doc to close this.
- Replicate on the Claude backend.
- Wire `augment_doc_code` into the real composition step (`build_unified.py`'s
  `build_aalinker`) rather than the offline eval harness.
