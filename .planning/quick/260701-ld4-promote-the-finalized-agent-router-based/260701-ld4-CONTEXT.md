# Quick Task 260701-ld4: Promote agent_router linker + archive pilot/ - Context

**Gathered:** 2026-07-01
**Status:** Ready for planning

<domain>
## Task Boundary

Clean up the `router` branch: promote the finalized "agent_router" chain out of
`pilot/` into production `src/`, register it as a runnable `run_ablation.py`
variant, archive the rest of `pilot/` (now superseded scratch work) into
`.planning/archive/`, and update the project doc (`CLAUDE.md`, which is stale —
it still describes the old `s20U` branch, not the current `router` branch state).

</domain>

<decisions>
## Implementation Decisions

### Which config is "the finalized agent_router linker"
Promote the **whole agent_router chain**: `pilot/gtp/agentic_router.py`
(`BoundedAutonomyAgenticRouter` — the LLM decides VALIDATE/CODE/REJECT per
candidate, gate-floored by s21's unchanged two-pass validator) **plus its
dependency** `pilot/gtp/proposer.py` (`GroundedTypedProposer`, the GTP proposer
that generates the candidates the router acts on). Do not promote the simpler
non-agentic "named+routed" GTP config as the primary artifact — the user
explicitly wants the *agentic* router, even though pilot findings show it scores
~1pp F1 lower (P 0.9592/F1 0.9402 vs named+routed's P 0.9897/F1 0.9506). Record
that trade-off honestly in the new module's docstring (mirror how
`s_linker21.py`'s docstring cites its own provenance/numbers) — do not overstate
it as strictly better, it is the agentic increment the user asked for.

### Promotion depth
- Move/adapt `agentic_router.py` + `proposer.py` into
  `src/llm_sad_sam/linkers/experimental/` as clean, documented, importable
  modules — the same treatment `router_direct.py` already received on this
  branch (it is NOT a full linker itself; it's reusable infra imported by a
  linker). Fix up their sys.path/`.env`-bootstrap hacks (`_APPROACH = Path(...).parents[2]`
  etc.) since they won't live under `pilot/gtp/` anymore — imports should work
  the same way `router_direct.py` and `s_linker21.py` already do in this src tree
  (no bespoke sys.path insertion needed; they're already on the path once
  installed via `pip install -e .`).
- **Also build one new linker module** that actually wires the chain together
  end-to-end so it is runnable via `run_ablation.py`, e.g.
  `src/llm_sad_sam/linkers/experimental/s_linker21_agentrouter.py` defining
  `SLinker21AgentRouter`. Suggested shape: subclass/wrap `SLinker21` (reuse its
  `link()` pipeline unchanged — do NOT modify `s_linker21.py`, GATE-01 requires
  canonical files byte-stable), then run the GTP proposer + agentic router as an
  **augmentation pass** over the sentence/component set already loaded by the
  base `link()` call, adding any VALIDATE+gate-approved candidates the base
  pipeline missed. CODE-routed candidates are exposed (e.g. via an attribute or
  return-adjacent structure) for a future doc→code wiring — that wiring itself
  (`router_direct.DirectCodeLinker` + `DirectLinkJudge`, already promoted in this
  src tree) is in scope to *call*, but the ArCoTL/`build_unified.py` composition
  step lives in a **sibling repo** (`../sota/recovered-links`) and is explicitly
  OUT of scope here.
- Register this new module in `run_ablation.py`'s `VARIANT_SPECS` /
  `CANONICAL_VARIANTS` following the exact dict shape used for `s_linker21`
  (see `run_ablation.py:810-826`): `module`, `class_name`, `description`
  (cite the pilot's measured numbers), `canonical=False`, `experimental=True`
  (it is NOT the paper's canonical variant — s_linker21 remains canonical).
  Verify registration with `python run_ablation.py --list-variants`.
- GATE-01 (STATE.md standing gate): `s_linker13_min.py`, `s_linker19.py`,
  `s_linker20_union.py`/`s_linker21.py` must stay byte-stable. The new module
  must be purely additive — no edits to `s_linker21.py` or `__init__.py`'s
  existing exports (adding one new export line is fine).

### Archiving pilot/
Move the **entire `pilot/` directory** (all of `router_eval*.py`, `fn_judge/`,
`gtp/`, `analysis/`, `cache/`, `README.md`, `FINDINGS.md`, `PROPOSAL.md`, etc.)
into `.planning/archive/router-pilot-260701/` using `git mv` (preserve history),
after the two files above have been extracted/adapted into `src/`. Strip
`__pycache__` directories (not tracked value, regenerable junk) rather than
archiving them. This mirrors the project's existing archive convention (GSD's
`gsd-cleanup` phase-archiving pattern — `.planning/archive/` is the right home,
not deleting or leaving it under `pilot/`).

### Project doc update
`CLAUDE.md` is currently stale: it describes the **s20U branch** (a different,
now-superseded branch), not the current `router` branch's actual state. Rewrite
it to describe the `router` branch's current active surface: `s_linker21.py`
(canonical), `router_direct.py` (direct code-linking infra), the new
`s_linker21_agentrouter.py` (agentic augmentation, experimental), and note that
the pilot investigation that produced them is archived at
`.planning/archive/router-pilot-260701/` (point there for the full narrative/
numbers instead of duplicating them). Keep the Build & Run section accurate for
this branch's actual runnable commands.

</decisions>

<specifics>
## Specific Ideas

Key files already inspected (read these, don't re-derive from scratch):
- `pilot/gtp/agentic_router.py` — `BoundedAutonomyAgenticRouter`, `Candidate`,
  `Decision`, `StrictGate` (defaults to importing `LAYERED_ENTITY_RULES`/`P1_FOCUS`/
  `P2_FOCUS` from `s_linker21`, with a standalone fallback rubric if that import
  fails — keep this fallback, it's what makes the module importable independent
  of s21's internals).
- `pilot/gtp/proposer.py` — `GroundedTypedProposer`, `ground()`, `build_prompt()`.
  Reasoning-off client bootstrap (`make_client()`), cache-file support.
- `src/llm_sad_sam/linkers/experimental/router_direct.py` — already-promoted
  sibling infra (`CodeIndex`, `DirectCodeLinker`, `SentenceRouter`,
  `DirectLinkJudge`) for the CODE-routed side; the new module's CODE action
  should hand off to this, not reinvent it.
- `src/llm_sad_sam/linkers/experimental/s_linker21.py` — canonical linker,
  `.link(text_path, model_path, **kwargs)` interface, `_VARIANT_NAME` class attr
  pattern, `LAYERED_ENTITY_RULES`/`P1_FOCUS`/`P2_FOCUS` module-level constants.
- `run_ablation.py:810-826` — the `s_linker21` `VARIANT_SPECS` entry to mirror;
  `build_linker()` (~line 1044) shows the `cls(backend=..., **extra)`
  instantiation contract every variant must satisfy.
- `pilot/gtp/FINDINGS.md` §7 and `pilot/gtp/AGENT.md` §7-8 — the measured numbers
  and design rationale for the agentic router (cite in the new module's
  docstring; these files are being archived, so pull the numbers into the
  docstring rather than leaving it as a dangling reference).

Numbers to cite in the new module's docstring (from `gtp/AGENT.md` §7):
baseline s21 P 0.9894/R 0.8913/F1 0.9360; named+routed (non-agentic) target
P 0.9897/R 0.9173/F1 0.9506; bounded-autonomy agentic router P 0.9592/R 0.9247/
F1 0.9402 — gate-floor holds (every accept is gate-approved), all 4 core
recoveries kept, −1pp vs target is 100% verified gold-incompleteness not error,
46/251 marginal candidates routed to CODE, 61 rejected, 144 validated.

</specifics>

<canonical_refs>
## Canonical References

- `.planning/STATE.md` — GATE-01 (canonical files byte-stable), GATE-06 (no
  benchmark vocabulary in new code).
- `pilot/FINDINGS.md`, `pilot/gtp/FINDINGS.md`, `pilot/gtp/AGENT.md`,
  `pilot/PROPOSAL.md` — full narrative (being archived, but read before
  archiving to extract docstring content).

</canonical_refs>
