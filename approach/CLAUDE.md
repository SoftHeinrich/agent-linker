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

## Standing Gates

- **GATE-01**: canonical/paper artifacts stay byte-stable —
  `src/llm_sad_sam/linkers/experimental/s_linker21.py` above all. New variants
  subclass it; edits to shared files (`__init__.py`, `run_ablation.py`) are
  purely additive (new export line, new registry entry).
- **GATE-06**: no benchmark-derived vocabulary introduced in any new code —
  prompts/rubrics stay generic English; the runtime catalog (component names,
  code identifiers) is the only project-specific input.

## Notes

- The variant registry in `run_ablation.py` still lists many older
  non-retained variants from earlier branches (their modules were removed);
  only the `s_linker21*`, `s_linker20_union*`, and other still-present-module
  entries actually resolve to runnable code here.
- Default benchmarking backend is set in `.env` (`LLM_BACKEND=openai`,
  `gpt-5.4`). `.env` is untracked.
