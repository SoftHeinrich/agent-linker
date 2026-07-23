---
task: quick-260701-ld4
verified: 2026-07-01T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Quick Task 260701-ld4: Promote agent_router linker + archive pilot/ — Verification Report

**Task Goal:** Promote the finalized agent_router-based linker (agentic_router.py + GTP
proposer + full dependency chain) to src/llm_sad_sam/linkers/experimental/, register a
run_ablation.py variant, archive the rest of pilot/ into .planning/archive/, and update
project docs.

**Verified:** 2026-07-01
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Agentic router chain (proposer + router) lives in src/ and imports without sys.path hacks | ✓ VERIFIED | `agentic_router.py`/`proposer.py` present under `src/llm_sad_sam/linkers/experimental/`; `grep -n "sys.path.insert\|_APPROACH\|\.env"` on both files only matches legitimate `os.environ.pop`/`os.environ[...]` calls (not `.env`-file loading, not `sys.path.insert`, not `_APPROACH`). Live import ran clean: `from llm_sad_sam.linkers.experimental.agentic_router import BoundedAutonomyAgenticRouter, Candidate; from llm_sad_sam.linkers.experimental.proposer import GroundedTypedProposer` → succeeded. |
| 2 | `s_linker21_agentrouter` variant appears in `python run_ablation.py --list-variants` | ✓ VERIFIED | Ran `python run_ablation.py --list-variants 2>&1 \| grep -i agentrouter` → returned `s_linker21_agentrouter`, exit 0. Also present in `CANONICAL_VARIANTS` list (run_ablation.py:122) and `VARIANT_SPECS` dict (run_ablation.py:844-…). |
| 3 | `from llm_sad_sam.linkers.experimental import SLinker21AgentRouter` succeeds | ✓ VERIFIED | Ran directly: `import llm_sad_sam.linkers.experimental as e; e.SLinker21AgentRouter` → resolved to `<class '...s_linker21_agentrouter.SLinker21AgentRouter'>`. `__init__.py` has the one added import line + `__all__` entry, `SLinker21` export untouched. |
| 4 | s_linker21.py byte-identical to pre-plan state (GATE-01) | ✓ VERIFIED | `git show 0047520:src/.../s_linker21.py` diffed against working tree copy → empty diff (exit 0). `git diff --stat 0047520 HEAD -- src/.../s_linker21.py` → no output (no changes). |
| 5 | pilot/ archived | ✓ VERIFIED | `pilot/` does not exist at repo root. `.planning/archive/router-pilot-260701/` contains `analysis/`, `cache/`, `FINDINGS.md`, `fn_judge/`, `gtp/` (incl. `agentic_router.py`, `proposer.py`, `AGENT.md`, caches), `PROPOSAL.md`, `README.md`, `remaining_recall.py`, `router_eval*.py`. `find .../router-pilot-260701 -name __pycache__` → no matches (stripped). Git history: `git mv` commit `c90526a` shows the archive move. |
| 6 | CLAUDE.md describes the router branch, not s20U | ✓ VERIFIED | Read full file: opens "This is the **router branch**…", documents `router_direct.py`, `agentic_router.py`/`proposer.py`, `s_linker21_agentrouter.py` (incl. the `acm_path`/`code_links` behavior), points to the archive path, Build & Run section includes `--variants s_linker21_agentrouter`. No "s20U branch" string present (`grep -q "s20U branch" CLAUDE.md` → not found; only "prior s20U trim" phrasing remains, which is accurate historical framing, not a claim this IS the s20U branch). |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llm_sad_sam/linkers/experimental/agentic_router.py` | `BoundedAutonomyAgenticRouter`, `Candidate`, `Decision`, `StrictGate`, bootstrap-hack-free | ✓ VERIFIED | All 4 symbols present (`class Candidate`, `class Decision`, `route`/`accepted`/`routed_to_code`/`rejected` methods confirmed via grep). No sys.path/.env/_APPROACH residue. Imports cleanly. |
| `src/llm_sad_sam/linkers/experimental/proposer.py` | `GroundedTypedProposer`, `build_prompt`, `ground`, bootstrap-hack-free | ✓ VERIFIED | Imports cleanly (`GroundedTypedProposer` confirmed importable). No bootstrap residue. |
| `src/llm_sad_sam/linkers/experimental/s_linker21_agentrouter.py` | `SLinker21AgentRouter` wiring linker subclassing `SLinker21` | ✓ VERIFIED | `class SLinker21AgentRouter(SLinker21):` confirmed; `issubclass(SLinker21AgentRouter, SLinker21)` → `True` at runtime. |
| `run_ablation.py` | `s_linker21_agentrouter` VARIANT_SPECS entry (canonical=False, experimental=True) | ✓ VERIFIED | Entry at run_ablation.py:844 with `canonical=False, experimental=True`, correct `module=`/`class_name=` pointing at the new file/class; also added to `CANONICAL_VARIANTS` list (line 122) which is what gates `--list-variants` output per this codebase's naming quirk. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `s_linker21_agentrouter.py` | `agentic_router.BoundedAutonomyAgenticRouter` + `proposer.GroundedTypedProposer` | augmentation pass over base `link()`'s sentence/component set | ✓ WIRED | Read full file (161 lines). `super().link()` called first as floor; `GroundedTypedProposer(catalog_mode="name")` instantiated and `.propose()` called per sentence; `Candidate` objects built from grounded proposals; `BoundedAutonomyAgenticRouter().route(candidates)` called; `router.accepted(decisions)` filtered into dedup'd `SadSamLink(..., source="agentrouter")` appended to `base_final`. Whole pass wrapped in try/except falling back to `base_final` on failure. |
| `run_ablation.py VARIANT_SPECS` | `llm_sad_sam.linkers.experimental.s_linker21_agentrouter.SLinker21AgentRouter` | module + class_name dict entry | ✓ WIRED | `module="llm_sad_sam.linkers.experimental.s_linker21_agentrouter"`, `class_name="SLinker21AgentRouter"` confirmed in the dict; live `--list-variants` run confirms the entry resolves (no import error at listing time). |

### CODE-routing wiring (late plan addition — specifically re-checked)

Verified `DirectCodeLinker`/`DirectLinkJudge` behind an optional `acm_path` kwarg is
present and **structurally sound**, not just string-matched:

- Imports at top of `s_linker21_agentrouter.py`: `CodeIndex, DirectCodeLinker, DirectLinkJudge, load_code_units` from `router_direct` — all 4 symbols confirmed to actually exist in `router_direct.py` at the expected signatures (`load_code_units(acm_path) -> list[CodeUnit]`, `CodeIndex.__init__(self, units)`, `DirectCodeLinker` is a `@dataclass` with fields `index: CodeIndex`, `include_test: bool = True`, method `candidates(self, text) -> list[tuple[str,str,frozenset]]`, `DirectLinkJudge.__init__(client=None, model=..., batch=..., timeout=...)`, `judge(self, cases) -> dict[int,bool]`).
- `self.code_routed_candidates`/`self.code_links` are unconditionally initialized to `[]` before the try block (so they always exist as instance attributes even on early failure) — matches the plan's "always set this" requirement.
- `acm_path = kwargs.get("acm_path")` branch: when truthy, builds `CodeIndex(load_code_units(acm_path))`, `DirectCodeLinker(idx, include_test=True)` (positional `index` + keyword `include_test`, matches the dataclass field order), iterates CODE-routed candidates through `dl.candidates(cand.sentence)`, batches through `DirectLinkJudge().judge(cases)`, and only pushes `(snum, path)` into `self.code_links` for judge-approved cases, deduped via a `seen_code_links` set.
- When `acm_path` is absent, prints a one-line skip log and leaves `self.code_links = []` — does not raise, does not block the doc augmentation return. Matches plan spec exactly.
- `router.routed_to_code(decisions)` populates `self.code_routed_candidates` unconditionally, before the `acm_path` branch — correct per plan (raw candidates always exposed regardless of acm_path availability).

Conclusion: this wiring is real code (not a stub) — call chain, signatures, and control
flow all check out against the actual `router_direct.py` implementation, not just against
the docstring's claims.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PROMOTE-01 | 260701-ld4-PLAN.md | Promote agent_router chain to src/, register variant, archive pilot/, update docs | ✓ SATISFIED | All 6 truths + 4 artifacts + 2 key links + CODE-routing sub-wiring verified above. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none found | — | `grep -n -iE "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER\|not yet implemented\|coming soon"` across the 3 new/promoted files returned no matches (exit 1). |

No debt markers, no empty-return stubs, no hardcoded-empty data flowing into rendered
output in any of the promoted/new files.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Chain imports without bootstrap hacks | `python -c "from llm_sad_sam.linkers.experimental.agentic_router import BoundedAutonomyAgenticRouter, Candidate; from llm_sad_sam.linkers.experimental.proposer import GroundedTypedProposer"` | `imports OK` | ✓ PASS |
| Variant listed | `python run_ablation.py --list-variants \| grep -i agentrouter` | `s_linker21_agentrouter` | ✓ PASS |
| Class importable + subclass check | `import llm_sad_sam.linkers.experimental as e; e.SLinker21AgentRouter`; `issubclass(SLinker21AgentRouter, SLinker21)` | `True` | ✓ PASS |
| GATE-01 byte-stability | `git show 0047520:.../s_linker21.py` diffed vs working tree; `git diff --stat 0047520 HEAD -- .../s_linker21.py` | empty diff both ways | ✓ PASS |
| Diff scope check | `git diff --stat 0047520 HEAD -- .` (excluding .planning/pilot) | 6 files, all expected (CLAUDE.md, run_ablation.py, __init__.py, agentic_router.py, proposer.py, s_linker21_agentrouter.py) | ✓ PASS |

Live end-to-end execution of the augmentation pass (actual LLM calls to
`GroundedTypedProposer`/`BoundedAutonomyAgenticRouter`/`DirectLinkJudge` against a real
dataset) was NOT run — this requires network/API access and a benchmark dataset, and is
appropriately deferred to human/live-sweep verification per the plan's own scope (the
plan's own verify steps are static-import/registration checks, not live-run checks).

### Human Verification Required

None required for the stated must-haves — all are structurally/statically verifiable and
were verified above. Live-sweep numeric reproduction of the cited pilot metrics
(P 0.9592/R 0.9247/F1 0.9402) was explicitly out of scope for this promotion task (the
task promotes and wires code; it does not re-run the benchmark).

### Gaps Summary

None. All 6 must-have truths, all 4 required artifacts, both key links, and the
late-added CODE-routing wiring (DirectCodeLinker/DirectLinkJudge behind optional
`acm_path`) are verified present, substantive, and correctly wired against the actual
`router_direct.py` API — not just present as docstring claims. GATE-01 held (byte-identical
diff confirmed independently via `git show` against the pre-plan commit `0047520`).
`pilot/` fully archived with `__pycache__` stripped. `CLAUDE.md` rewritten and accurate
for the `router` branch.

---

_Verified: 2026-07-01_
_Verifier: Claude (gsd-verifier)_
