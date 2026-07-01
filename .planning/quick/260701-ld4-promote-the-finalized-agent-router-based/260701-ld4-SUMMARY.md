---
quick_id: 260701-ld4
slug: promote-the-finalized-agent-router-based
date: 2026-07-01
status: complete
---

# Quick Task 260701-ld4 — SUMMARY

> **COMPLETE (2026-07-01).** Promoted the bounded-autonomy agent_router chain
> (`agentic_router.py` + GTP `proposer.py`) from `pilot/gtp/` into production `src/`,
> wired a new `s_linker21_agentrouter` linker (subclasses `SLinker21`, gate-floored
> augmentation, CODE-routed candidates judged via `DirectCodeLinker`/`DirectLinkJudge`
> behind an optional `acm_path` kwarg), registered it as a runnable `run_ablation.py`
> variant, archived the rest of `pilot/` to `.planning/archive/router-pilot-260701/`,
> and rewrote the stale `CLAUDE.md` for the `router` branch. GATE-01 held throughout
> (`s_linker21.py` byte-identical, confirmed by `git diff --stat`).

Reconstructed from the executor's final report + verified post-merge state — the
executor's original uncommitted SUMMARY.md was lost when its isolated worktree was
removed before rescue (orchestrator error; see Notes/landmines).

## User decisions locked in `260701-ld4-CONTEXT.md`

- **Which config:** the literal *agentic* chain (`agentic_router.py`'s
  `BoundedAutonomyAgenticRouter` + its `proposer.py` dependency) — not the simpler
  non-agentic "named+routed" GTP config, even though pilot numbers show the
  non-agentic config scores ~1pp higher F1.
- **Promotion depth:** formalize as clean `src/` modules AND register a
  `run_ablation.py` variant (not just a module move).
- **CODE-routing wiring:** wire it now via the already-promoted
  `router_direct.DirectCodeLinker`/`DirectLinkJudge` (decided after plan-checker
  flagged the original "expose only" design as ambiguous against CONTEXT.md).
- **Archive scope:** move the whole `pilot/` directory into `.planning/archive/`.

## Done (committed)

| Commit | What |
|--------|------|
| `9f17eaa` | Promote `agentic_router.py` + `proposer.py` into `src/llm_sad_sam/linkers/experimental/`, stripped of pilot-only `sys.path`/`.env` bootstrap hacks. |
| `15b6b88` | New `src/llm_sad_sam/linkers/experimental/s_linker21_agentrouter.py` (`SLinker21AgentRouter`, subclasses `SLinker21`); one export line in `__init__.py`; one `VARIANT_SPECS`/`CANONICAL_VARIANTS` entry in `run_ablation.py`. |
| `c90526a` | `git mv pilot/ → .planning/archive/router-pilot-260701/` (untracked `gtp/`, `fn_judge/`, `PROPOSAL.md` moved alongside; `__pycache__` stripped); rewrote `CLAUDE.md` for the `router` branch. |
| `3a06248` | Orchestrator merge of the above three (worktree isolation) back onto `router`. |

**`SLinker21AgentRouter.link()` design** (see module docstring for full detail):
1. `base_final = super().link(...)` — canonical s21 pipeline, unchanged, is the floor.
2. Re-derives the sentence/component working set, runs `GroundedTypedProposer` per
   sentence (live, reasoning-off), builds `Candidate`s from grounded proposals.
3. `BoundedAutonomyAgenticRouter().route(candidates)` — LLM decides
   VALIDATE/CODE/REJECT per candidate; only VALIDATE-and-gate-approved candidates
   are added on top of `base_final` (invariant: `accepted ⇔ VALIDATE ∧ gate approves`,
   so the result can never regress below s21).
4. CODE-routed candidates are always exposed via `self.code_routed_candidates`. When
   an `acm_path` kwarg is supplied to `link()`, those candidates are additionally run
   through `CodeIndex`/`DirectCodeLinker`/`DirectLinkJudge` into `self.code_links`
   (`(sentence_number, code_path)` pairs). `run_ablation.py`'s harness has **no**
   `.acm` path anywhere in its `DATASETS` dict today, so this branch is real, wired,
   and callable but not yet exercised by a live `run_ablation.py` invocation — noted
   explicitly in the module docstring and `CLAUDE.md`, not glossed over.
5. Whole augmentation pass wrapped in try/except — falls back to `base_final` on any
   proposer/router/judge failure.

**Measured numbers cited in the docstring** (pilot, gpt-5.4, from the now-archived
`gtp/FINDINGS.md` §7 / `gtp/AGENT.md` §7-8 — pulled inline since those files moved):

| config | P | R | F1 |
|---|---:|---:|---:|
| baseline s21 | 0.9894 | 0.8913 | 0.9360 |
| named+routed (non-agentic, NOT shipped) | 0.9897 | 0.9173 | 0.9506 |
| **bounded-autonomy agentic router (shipped)** | 0.9592 | 0.9247 | 0.9402 |

The −1pp F1 vs the non-agentic target is stated plainly in both the docstring and
`CLAUDE.md` as 100% verified gold-incompleteness (not error) — this is the agentic
increment the user chose to ship, not a claim that it's strictly better.

## Validation performed post-merge (by the orchestrator, all passed)

- `python run_ablation.py --list-variants` lists `s_linker21_agentrouter`.
- `from llm_sad_sam.linkers.experimental.s_linker21_agentrouter import SLinker21AgentRouter`
  imports; `issubclass(SLinker21AgentRouter, SLinker21)` holds; source contains
  `DirectCodeLinker`/`DirectLinkJudge`/`acm_path`.
- `git diff --stat <pre-plan-HEAD> HEAD -- src/llm_sad_sam/linkers/experimental/s_linker21.py`
  is empty (GATE-01 holds).
- No `sys.path.insert`/`_APPROACH` bootstrap-hack residue in the two promoted modules
  (the plan's own grep check for `\.env` false-positives on `os.environ` — confirmed
  manually the only matches are legitimate `os.environ.pop`/`os.environ[...]` calls,
  not `.env`-file loading).
- `pilot/` no longer exists at the repo root; full content (including the
  previously-untracked `gtp/`, `fn_judge/`, `PROPOSAL.md`) verified byte-identical
  under `.planning/archive/router-pilot-260701/`, no `__pycache__`.
- `CLAUDE.md` describes the `router` branch (mentions `s_linker21_agentrouter` +
  the archive path; no more "s20U branch" text).

## Plan-checker findings (both resolved before/during execution)

- **Cost/latency note (informational, not fixed):** the proposer loop issues one
  live LLM call per sentence with no batching — real operational cost once the
  variant is actually run, but does not affect this task's own verify/done criteria.
  Left as-is; worth a batching pass before any live sweep of this variant.
- **CODE-routing scope ambiguity (resolved):** CONTEXT.md's wording on whether
  `DirectCodeLinker`/`DirectLinkJudge` should be invoked in this task vs. deferred
  was genuinely ambiguous. Re-confirmed with the user mid-execution → wire it now
  (reflected in the final design above).

## Notes / landmines

- **Orchestrator process note:** the executor ran in an isolated worktree per
  `isolation="worktree"` and left `260701-ld4-SUMMARY.md` uncommitted for the
  orchestrator to pick up (correct per convention). The orchestrator force-removed
  the worktree (`git worktree remove --force`) before rescuing that file, so the
  original was lost — this document reconstructs it from the executor's final
  report text and the merged commit contents (all code/verification content is
  complete and re-verified independently; only the executor's own prose is
  reconstructed rather than verbatim).
- A pre-existing unrelated staged file (`run_s21_gpt55_compare.sh`, staged before
  this session started) briefly conflicted with the worktree merge; stashed,
  merged, then restored — untouched otherwise, still staged as the user left it.
- `run_ablation.py`'s `available_variants()` reads `CANONICAL_VARIANTS` (a separate
  list from `VARIANT_SPECS`, despite the name — it also lists non-canonical
  entries). The new variant was added to both; this is why `--list-variants` shows
  it.
- Plumbing an `acm_path` through `run_ablation.py`'s `DATASETS`/`run_variant()` (so
  the CODE-routing branch is actually exercised end-to-end), and the further
  ArCoTL/`build_unified.py` doc↔code composition step in the sibling
  `../sota/recovered-links` repo, are both explicitly out of scope here and remain
  open follow-up work.
