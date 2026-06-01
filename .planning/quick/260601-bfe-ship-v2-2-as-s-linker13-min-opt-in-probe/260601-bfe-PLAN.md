---
quick_id: 260601-bfe
mode: quick
type: execute
phase: quick
plan: 01
wave: 1
depends_on: []
autonomous: true
files_modified:
  - src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py
  - run_ablation.py
  - .planning/milestones/v2.2-ROADMAP.md
  - .planning/milestones/v2.2-REQUIREMENTS.md
  - .planning/milestones/v2.2-MILESTONE-AUDIT.md
  - .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md
  - .planning/ROADMAP.md
  - .planning/MILESTONES.md
  - .planning/PROJECT.md
  - .planning/STATE.md
  - .planning/v2.3-prep/v2.3-KICKOFF-SEED.md
requirements:
  - SHIP-V22-MIN-UNCHANGED
  - SHIP-V22-PROBE-D-OPT-IN
  - DEFER-V4-TO-V23
must_haves:
  truths:
    - "v2.2 ships s_linker13_min unchanged as the v2.2 canonical (Claude 0.9506, gpt-5.4 0.9069 — identical to v2.1 numbers)"
    - "Probe D variant s_linker14_probe_d_upstream_clean is registered as an opt-in carve-out flagged gpt-5.4-only (NOT promoted to canonical=True)"
    - "All declined v2.2 tracks (Probe B preamble+rubric, Probe C as primary, Probe A/A' Voyager v4 architecture) are documented in v2.2 milestone artifacts with links to their negative-finding probe SUMMARYs"
    - "Top-level ROADMAP.md, MILESTONES.md, PROJECT.md, STATE.md reflect v2.2 SHIPPED 2026-06-01 and v2.3 anchored on Voyager v4 with proven prereqs (per-backend cache infra + Probe A' vocab fix)"
    - "v2.3-prep/ has a kickoff seed pointing executors to per-backend cache infrastructure (probe-D-cachekey-fix-SUMMARY.md) and Probe A' vocab fix (v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md) so v2.3 does not re-explore"
  artifacts:
    - path: ".planning/milestones/v2.2-ROADMAP.md"
      provides: "v2.2 milestone roadmap (shipped scope, declined tracks, references)"
    - path: ".planning/milestones/v2.2-REQUIREMENTS.md"
      provides: "v2.2 requirements closure (SHIP-V22-MIN-UNCHANGED, SHIP-V22-PROBE-D-OPT-IN, DEFER-V4-TO-V23) + deferred-to-v2.3 carry-forward table"
    - path: ".planning/milestones/v2.2-MILESTONE-AUDIT.md"
      provides: "v2.2 audit verdict (passed/shipped 2026-06-01) modeled on v2.1-MILESTONE-AUDIT.md"
    - path: ".planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md"
      provides: "Trimmed-scope close summary: surviving artifact, opt-in carve-out, declined tracks, v2.3 handoff"
    - path: ".planning/v2.3-prep/v2.3-KICKOFF-SEED.md"
      provides: "v2.3 anchor pointer to per-backend cache infra + vocab fix prereqs"
    - path: "src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py"
      provides: "Probe D variant with v2.2 OPT-IN CARVE-OUT docstring marker (gpt-5.4 only)"
    - path: "run_ablation.py"
      provides: "VARIANT_SPECS entry for s_linker14_probe_d_upstream_clean updated with 'v2.2 OPT-IN CARVE-OUT (gpt-5.4 only)' description tag"
  key_links:
    - from: ".planning/ROADMAP.md (Milestones section)"
      to: ".planning/milestones/v2.2-ROADMAP.md"
      via: "markdown link in v2.2 milestone row"
      pattern: "v2\\.2.*shipped 2026-06-01"
    - from: ".planning/STATE.md (frontmatter)"
      to: "between milestones — v2.2 archived"
      via: "milestone_name field reset"
      pattern: "v2\\.2 archived"
    - from: ".planning/ROADMAP.md (Next Milestone)"
      to: ".planning/v2.3-prep/v2.3-KICKOFF-SEED.md"
      via: "explicit v2.3 anchor sentence with prereq links"
      pattern: "v2\\.3.*Voyager v4"

---

<objective>
Close v2.2 with the user-decided trimmed scope: ship `s_linker13_min` unchanged as the v2.2 canonical, register Probe D as an opt-in gpt-5.4-only carve-out (NOT promoted to canonical), and defer Voyager v4 multi-role to v2.3 with two already-proven prerequisites carried forward (per-backend cache infrastructure + Probe A' vocab fix).

Purpose: Lock in the v2.2 milestone close so the project state cleanly reflects shipped scope, opt-in carve-out, declined tracks (Probe B, Probe C as primary, Voyager v4 architecture), and v2.3 starting point — without re-running any probes or modifying frozen artifacts.

Output: Three milestone files under `.planning/milestones/`, top-level planning updates (ROADMAP / MILESTONES / PROJECT / STATE), a v2.2-prep close SUMMARY, a v2.3-prep kickoff seed, and a Probe D variant docstring + registry tag marking the carve-out.
</objective>

<execution_context>
This is a quick-mode close-only task. No LLM probe runs. No changes to frozen v2.0 / v2.1 artifacts (`s_linker13.py`, `s_linker13_min.py`, `prompts_v2.py`, `prompts_v3.py`, `helper_v3.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `ilinker*.py`, `s_linker13_clean*.py`). No changes to existing Probe D logic — only docstring + registry description. Probe B and Probe C registry entries remain stable (they were registered with `canonical=False` for ablation runnability and decline status is documented in v2.2-prep SUMMARYs).
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/MILESTONES.md
@.planning/STATE.md
@.planning/v2.2-prep/v2.2-MILESTONE-PROPOSAL.md
@.planning/v2.2-prep/v2.2-SCOPE-DECISION.md
@.planning/v2.2-prep/v2.2-PROBE-WAVE-SUMMARY.md
@.planning/v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md
@.planning/v2.2-prep/probe-D-upstream-SUMMARY.md
@.planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md
@.planning/v2.2-prep/range-D-bbb-SUMMARY.md
@.planning/v2.2-prep/probe-A-voyager-v4-SUMMARY.md
@.planning/v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md
@.planning/v2.2-prep/probe-A-prime-range-bbb-SUMMARY.md
@.planning/v2.2-prep/probe-B-preamble-rubric-SUMMARY.md
@.planning/v2.2-prep/probe-C-selfrefine-SUMMARY.md
@.planning/milestones/v2.1-ROADMAP.md
@.planning/milestones/v2.1-REQUIREMENTS.md
@.planning/milestones/v2.1-MILESTONE-AUDIT.md
@src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py
@run_ablation.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Probe D opt-in carve-out — docstring marker + registry description</name>
  <files>src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py, run_ablation.py</files>
  <action>
Mark Probe D explicitly as the v2.2 opt-in gpt-5.4-only carve-out at both the variant docstring and the VARIANT_SPECS registry entry, WITHOUT modifying its runtime logic (cache-key fix already landed per `probe-D-cachekey-fix-SUMMARY.md`).

**File 1: `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py`**

Prepend a v2.2 carve-out banner to the existing module docstring (the docstring currently starts `"""S-Linker14 Probe D — Upstream-tier rule removal (COREF_RULES).`). Use the Edit tool to insert a new top section ABOVE the existing first line of the docstring (i.e. between `"""` and `S-Linker14 Probe D`). Insert exactly:

```
v2.2 OPT-IN CARVE-OUT (gpt-5.4 only) — shipped 2026-06-01.
====================================================================

Status: opt-in variant, NOT canonical. v2.2 milestone canonical remains
`s_linker13_min` unchanged. This variant is retained because:

  - Probe D mediastore gpt-5.4: STRONG_PASS (+1.59pp vs anchor 0.9677;
    closes the cross-model gap on mediastore — matches Claude baseline
    0.9836). See `.planning/v2.2-prep/probe-D-upstream-SUMMARY.md`.
  - Range D bigbluebutton gpt-5.4: STRONG_PASS (+3.29pp original,
    +1.12pp cache-fix re-run; both above the +0.5pp threshold). Mean
    across the two observations: +2.2pp. See
    `.planning/v2.2-prep/range-D-bbb-SUMMARY.md` +
    `.planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md`.
  - Range D bigbluebutton Claude: FAIL (-4.23pp) — CONFOUNDED by
    cross-backend cache reuse on the old 2-key cache; methodologically
    invalidated. The per-backend cache-key fix below unblocks a fresh
    Claude rubric build in a future turn.

**Gating policy**: enable ONLY when `LLM_BACKEND == openai` (gpt-5.4 path).
Do NOT enable on Claude until a fresh Claude-authored rubric run replaces
the FAIL verdict.

**Per-backend cache infrastructure** (already proven in this file —
SANITY_PASS 2026-06-01): the cache key is per-(text_stem, comp_hash,
backend, model) and cache writes go to
`results/v2_2_probes_range_d_cachefix/cache/` by default
(override via `PROBE_D_CACHE_ROOT` env var). Old 2-key gpt-5.4 rubrics
at `results/v2_2_probes/D_upstream/cache/` are preserved as historical
record but no longer reused at runtime. See
`.planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md` for the full
verification record.

**v2.3 handoff**: the per-backend cache infrastructure here is a v2.3
prerequisite (see `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`). DO NOT
re-explore it — start from this implementation.

```

Then keep the existing docstring body unchanged (the existing `S-Linker14 Probe D — Upstream-tier rule removal (COREF_RULES). ...` content follows).

DO NOT touch any code below the module docstring. DO NOT modify `_cache_key`, `_cache_path`, `CACHE_ROOT`, the rubric builder, or `_coref_cases_in_context`. The cache-fix is already correct.

**File 2: `run_ablation.py`**

Use Edit to update the VARIANT_SPECS description string for `s_linker14_probe_d_upstream_clean` (currently at approximately line 484) from:

```
        description="S-Linker14 Probe D — v2.2 PROBE WAVE (Phase 17, EXT-upstream): runtime coref rubric REPLACES static COREF_RULES. 1 builder LLM call per dataset; cache in results/v2_2_probes/D_upstream/cache/. Forks from s_linker13_clean_v3.",
```

to:

```
        description="S-Linker14 Probe D — v2.2 OPT-IN CARVE-OUT (gpt-5.4 only, shipped 2026-06-01): runtime coref rubric REPLACES static COREF_RULES. NOT canonical — v2.2 canonical remains s_linker13_min unchanged. mediastore gpt-5.4 +1.59pp; BBB gpt-5.4 mean +2.2pp over 2 obs; BBB Claude FAIL was confounded by cross-backend cache reuse (per-backend cache key fix landed). Enable only when LLM_BACKEND==openai. 1 builder LLM call per dataset; cache key per-(text_stem, comp_hash, backend, model); default cache root results/v2_2_probes_range_d_cachefix/cache/ (PROBE_D_CACHE_ROOT env override). Forks from s_linker13_clean_v3. See .planning/v2.2-prep/probe-D-upstream-SUMMARY.md + .planning/v2.2-prep/range-D-bbb-SUMMARY.md + .planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md.",
```

Also update the inline comment in CANONICAL_VARIANTS (approximately line 101) from:

```
    "s_linker14_probe_d_upstream_clean",   # v2.2 PROBE WAVE — Probe D (Phase 17): runtime coref rubric replaces COREF_RULES (EXT-upstream)
```

to:

```
    "s_linker14_probe_d_upstream_clean",   # v2.2 OPT-IN CARVE-OUT (gpt-5.4 only, shipped 2026-06-01): runtime coref rubric replaces COREF_RULES; NOT canonical
```

DO NOT change `canonical=False` (already correct). DO NOT remove or modify Probe B or Probe C registry entries (they remain as ablation-runnable negative-finding variants per the v2.2 scope decision). DO NOT touch `s_linker13_min`'s canonical=True.

GATE-07 compliance: the variant remains registered in CANONICAL_VARIANTS + VARIANT_SPECS as a standalone file with a structured docstring. The carve-out status is now explicit in both surfaces.

BENCHMARK_TABOO compliance: no benchmark words added (banner uses only project + planning paths + abstract terms).

The frozen-compat regression fixture (`tests/test_v20_baseline_regression.py` / `tests/fixtures/v2_0_baseline.json`) is NOT updated this turn — the Probe D variant ships opt-in / non-canonical and is not in the v2.0/v2.1 baseline set (its mediastore gpt-5.4 +1.59pp claim is recorded in `probe-D-upstream-SUMMARY.md` rather than codified as a regression assertion, consistent with how Probe B/C are handled).
  </action>
  <verify>
<automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; python -c "from llm_sad_sam.linkers.experimental import s_linker14_probe_d_upstream_clean as m; assert 'v2.2 OPT-IN CARVE-OUT' in (m.__doc__ or ''); assert 'gpt-5.4 only' in (m.__doc__ or ''); print('docstring carve-out marker OK')" &amp;&amp; python -c "import run_ablation; spec = run_ablation.VARIANT_SPECS['s_linker14_probe_d_upstream_clean']; assert spec.get('canonical', True) is False, 'must remain canonical=False'; desc = spec['description']; assert 'v2.2 OPT-IN CARVE-OUT' in desc; assert 'gpt-5.4 only' in desc; assert 'NOT canonical' in desc; print('registry description OK')" &amp;&amp; python -c "import run_ablation; assert 's_linker13_min' in run_ablation.VARIANT_SPECS; assert run_ablation.VARIANT_SPECS['s_linker13_min'].get('canonical') is True, 's_linker13_min must remain canonical=True'; print('s_linker13_min unchanged OK')"</automated>
  </verify>
  <done>
- Probe D module docstring begins with `v2.2 OPT-IN CARVE-OUT (gpt-5.4 only) — shipped 2026-06-01.` banner referencing mediastore/BBB STRONG_PASS, BBB Claude confound, per-backend cache infra, and v2.3 handoff link.
- `run_ablation.VARIANT_SPECS["s_linker14_probe_d_upstream_clean"]["description"]` starts with `S-Linker14 Probe D — v2.2 OPT-IN CARVE-OUT (gpt-5.4 only, shipped 2026-06-01)` and includes "NOT canonical" + "Enable only when LLM_BACKEND==openai".
- `canonical=False` preserved on Probe D; `canonical=True` preserved on `s_linker13_min`.
- No code below the module docstring modified; no other VARIANT_SPECS entries modified.
- `python -c "import run_ablation"` and `python -c "from llm_sad_sam.linkers.experimental import s_linker14_probe_d_upstream_clean"` both import without error.
  </done>
</task>

<task type="auto">
  <name>Task 2: v2.2 milestone artifacts (ROADMAP / REQUIREMENTS / AUDIT) + v2.2-prep close SUMMARY</name>
  <files>.planning/milestones/v2.2-ROADMAP.md, .planning/milestones/v2.2-REQUIREMENTS.md, .planning/milestones/v2.2-MILESTONE-AUDIT.md, .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md</files>
  <action>
Create the four v2.2 close documents. Model structure on v2.1 counterparts (already read in context). v2.2 is a TRIMMED milestone — it ships an unchanged canonical + one opt-in carve-out + documented declines.

**File 1: `.planning/milestones/v2.2-ROADMAP.md`** — Use Write tool. Structure mirrors `.planning/milestones/v2.1-ROADMAP.md`. Frontmatter / top notice: `v2.2 SHIPPED 2026-06-01`. Content sections:

1. Top banner: shipped notice with link to v2.1-ROADMAP for history and to v2.2-MILESTONE-AUDIT.
2. `## Milestones` list (v1.0, v2.0, v2.1, v2.2 rows; v2.2 description: "Probe-Wave Trimmed Close — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) + Voyager v4 deferred to v2.3").
3. `## v2.2 Goal` — explain the trimmed scope: validate via probe wave whether any of 4 mechanism pillars (Voyager v4, preamble+rubric, Self-Refine, upstream-tier rule removal) could lift macro F1 on either backend without regressing the other; ship surviving mechanisms.
4. `## Outcome` — short summary: 4 probes ran, 1 strong survivor (Probe D / upstream coref rubric), gpt-5.4-only behavior on Range BBB, Claude verdict confounded by cross-backend cache reuse (now methodologically unblocked but not re-run). Decision: ship `s_linker13_min` unchanged as canonical + Probe D as opt-in carve-out + defer Voyager v4 + per-backend cache infra + Probe A' vocab-aligned R3 to v2.3.
5. `## Phases` table — render the v2.2 SCOPE-DECISION mapping verbatim:
   - Phase 14 (Voyager v4 multi-role): PROBE_FAIL (Probe A R5 100% reject) → vocab fix attempted as Probe A' → mediastore STRONG_PASS but BBB WEAK_PASS (R5 0/8, F1 -0.24pp) → DEFERRED to v2.3.
   - Phase 15 (Preamble + cached rubric): Probe B FAIL mediastore -5.24pp → DECLINED.
   - Phase 16 (Self-Refine alias judge): Probe C WEAK_PASS +0.00004pp → DECLINED as primary; contingent only.
   - Phase 17 (Upstream-tier rule removal): Probe D STRONG_PASS mediastore +1.59pp; Range D split (gpt-5.4 STRONG_PASS, Claude FAIL confounded); cache-fix sanity SANITY_PASS → SHIPPED as opt-in gpt-5.4-only carve-out.
   - Phase 18 (Composition + Promotion): degenerated to "promote s_linker13_min as v2.2 canonical unchanged + register Probe D opt-in" — no composition needed.
6. `## Declined Tracks (with negative-finding pointers)` table — Probe B FAIL, Probe C WEAK_PASS, Probe A PROBE_FAIL, Probe A' BBB WEAK_PASS — each with link to its per-probe SUMMARY in v2.2-prep/.
7. `## Standing Gates` — note GATE-01, GATE-02, GATE-06, GATE-07 carried forward; GATE-08 (cost-per-improvement) introduced in v2.2 scope decision and noted for v2.3.
8. `## Key Numbers` — Claude macro 0.9506 / gpt-5.4 macro 0.9069 unchanged from v2.1 (v2.2 ships no new canonical numbers); Probe D opt-in evidence: mediastore gpt-5.4 +1.59pp (single obs), BBB gpt-5.4 mean +2.2pp over 2 obs, BBB Claude FAIL (confounded).
9. `## Carry-Forward to v2.3` — pointer to `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md` and the two proven prereqs.

**File 2: `.planning/milestones/v2.2-REQUIREMENTS.md`** — Use Write tool. Structure mirrors `v2.1-REQUIREMENTS.md` (already in context). Three v2.2 requirements all CLOSED:

| ID | Description | Closure |
|---|---|---|
| SHIP-V22-MIN-UNCHANGED | `s_linker13_min` ships as v2.2 canonical with identical numbers to v2.1 (Claude 0.9506, gpt-5.4 0.9069); no code change | Task 2 of `.planning/quick/260601-bfe-.../260601-bfe-PLAN.md` (this milestone close — Probe wave found no Pareto-positive mechanism to promote) |
| SHIP-V22-PROBE-D-OPT-IN | Probe D registered as `canonical=False` opt-in carve-out with gpt-5.4-only docstring + registry tag | Task 1 of this PLAN |
| DEFER-V4-TO-V23 | Voyager v4 multi-role architecture + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as named prereqs (NOT re-explored) | Task 3 of this PLAN — v2.3-prep/v2.3-KICKOFF-SEED.md |

Include a Deferred-to-v2.3 carry-forward table (mirroring v2.1's deferred-items table style) with rows: Voyager v4 multi-role architecture (Probe A' fallback fork = Compact-B per SCOPE-DECISION); per-backend cache infrastructure (proven this milestone); Probe A' vocab-aligned R3 (proven mediastore-viable); Claude Probe D re-test with cache fix (methodologically ready, not run this turn); ADAPTER-01 (carried from v2.1); EXT-04 (carried from v2.1); link provenance data structure (carried from v2.1); Extended Thinking on judge stages (carried from v2.1); Self-Refine contingent (only if v2.3 mainline both fails); problem-statement preamble alone (un-isolated from Probe B failure — re-test optional, de-prioritized).

**File 3: `.planning/milestones/v2.2-MILESTONE-AUDIT.md`** — Use Write tool. Structure mirrors `v2.1-MILESTONE-AUDIT.md`. Frontmatter: `milestone: v2.2`, `milestone_name: Probe-Wave Trimmed Close`, `status: passed`, `verdict: shipped`, `audited: 2026-06-01`. Sections:

1. Verdict: PASSED — Ship It (trimmed scope; 3/3 requirements closed).
2. Phase Verification Roll-Up table (Phase 14/15/16/17/18 with their probe-wave outcomes — no per-phase VERIFICATION files since v2.2 ran as probe-wave + scope-decision rather than the conventional phase/plan workflow; reference the v2.2-prep SUMMARYs instead).
3. Requirements Closure: 3/3 closed.
4. Shipped Artifacts:
   - Canonical: `s_linker13_min.py` (unchanged from v2.1, canonical=True preserved).
   - Opt-in carve-out: `s_linker14_probe_d_upstream_clean.py` (canonical=False, gpt-5.4 only).
   - Negative-finding variants retained for record: `s_linker14_probe_b_preamble_clean.py`, `s_linker14_probe_c_selfrefine_clean.py` (both canonical=False, registry-stable).
   - Probe harnesses: `scripts/voyager_train_tlr_v4.py`, `scripts/voyager_train_tlr_v4_a_prime.py`, `scripts/run_v2_2_probe.py`, `scripts/run_v2_2_range_d.py`.
   - Per-probe SUMMARYs under `.planning/v2.2-prep/`.
5. Cross-Phase Integration: NOT applicable — no new canonical composition shipped.
6. Key Numbers table: Claude/gpt-5.4 numbers unchanged from v2.1; Probe D opt-in evidence summarized.
7. Methodological Contributions:
   - Probe-wave methodology validated as a milestone-scoping mechanism (4 cheap parallel probes cut decisively across 4 mechanism pillars).
   - Per-backend cache-key methodology for runtime LLM rubrics (per-(text_stem, comp_hash, backend, model)) — unblocks fair cross-model comparison.
   - Probe A' vocab-aligned R3 (discourse/syntactic terms only) — narrows v4 R3/R5 vocabulary deadlock; mediastore-viable, BBB-inactive — surfaces a "v4 mechanism is dataset-conditional" finding for v2.3.
   - GATE-08 cost-per-improvement audit introduced (applied to Probe C WEAK_PASS; would carry to v2.3 confirmation tier).
8. Findings (publishable):
   - Probe D: runtime per-dataset coref rubric replaces static COREF_RULES with gpt-5.4 lift (+1.59pp mediastore, mean +2.2pp BBB); generalizes the trim9 mechanism class (Phase 12 Plan 12-12) from seed tier to coref tier. Cross-backend transfer not confirmed (Claude was cache-confounded).
   - Probe B: LLM-built per-dataset alias-judge rubric introduced an over-aggressive token-reject rule producing 4 FNs vs anchor's 1 → FAIL -5.24pp. Failure mode: cold-generated per-dataset rubrics over-restrict at scale.
   - Probe C: 2-iter Self-Refine matched anchor exactly on mediastore (+0.00004pp); 80% iter-0 contested rate but iter-1 fired without changing the approved set → judge at ceiling. GATE-08 cost flag.
   - Probe A: R3/R5 vocabulary deadlock (R3 textbook SE vocabulary REJECTED by R5 5-style transferability test) → 100% R5 reject → falsification of v4 with that vocabulary.
   - Probe A' (vocab fix): R3 narrowed to discourse/syntactic terms → mediastore STRONG_PASS (+1.69pp) but BBB WEAK_PASS (R5 0/8, F1 -0.24pp) → v4 architecture is mediastore-viable, BBB-inactive on gpt-5.4.
   - Cross-backend cache reuse confounds runtime-rubric cross-model tests; per-backend cache-key methodology resolves.
9. Open Items / Deferred (carry to v2.3): same table as `v2.2-REQUIREMENTS.md` deferred section, condensed.
10. Tech Debt: Low — same as v2.1 carried items + (NEW) frozen-compat fixture does not assert the Probe D opt-in claim (recorded only in per-probe SUMMARYs).
11. Audit Decision: PASSED — milestone ready for completion.

**File 4: `.planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md`** — Use Write tool. ~150-200 line standalone summary suitable for "what shipped in v2.2" question. Frontmatter:

```
---
phase: v2.2-CLOSE
date: 2026-06-01
mode: quick-close
milestone_canonical: s_linker13_min  # unchanged from v2.1
opt_in_carve_out: s_linker14_probe_d_upstream_clean  # gpt-5.4 only
declined: [Probe B preamble+rubric, Probe C as primary, Voyager v4 architecture]
deferred_to_v23: [Voyager v4 multi-role (with proven cache infra + Probe A' vocab fix), Claude Probe D re-test, ADAPTER-01, EXT-04, link provenance, Extended Thinking judge]
tags: [v2.2, milestone-close, trimmed-scope, probe-wave, opt-in-carve-out, defer-v4-to-v2.3]
key-files:
  shipped:
    - src/llm_sad_sam/linkers/experimental/s_linker13_min.py  # canonical (unchanged)
    - src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py  # opt-in carve-out
  milestone_docs:
    - .planning/milestones/v2.2-ROADMAP.md
    - .planning/milestones/v2.2-REQUIREMENTS.md
    - .planning/milestones/v2.2-MILESTONE-AUDIT.md
  v23_handoff:
    - .planning/v2.3-prep/v2.3-KICKOFF-SEED.md
---
```

Body sections:

1. Headline: 3-bullet (shipped unchanged canonical / opt-in gpt-5.4-only carve-out / Voyager v4 deferred with proven prereqs).
2. Trimmed Scope (what was originally planned vs what shipped) — reference `v2.2-MILESTONE-PROPOSAL.md` original 5 phases and explain each phase's collapse via the SCOPE-DECISION.
3. Surviving Artifact: `s_linker13_min` unchanged (Claude 0.9506, gpt-5.4 0.9069) — explain why v2.2 ships no new canonical (no probe was cross-backend Pareto-positive).
4. Opt-in Carve-Out: Probe D — full evidence table (mediastore STRONG_PASS, BBB Range gpt-5.4 STRONG_PASS, BBB Range Claude FAIL CONFOUNDED, cache-fix sanity SANITY_PASS); per-backend cache-key resolution; gating policy (`LLM_BACKEND == openai` only).
5. Declined Tracks: per-track (Probe A / Probe A' / Probe B / Probe C) bullet of mechanism + failure mode + decision + retained-artifact note.
6. v2.3 Handoff: explicit list of proven prereqs (per-backend cache infrastructure; Probe A' vocab-aligned R3; methodologically-ready Claude Probe D re-test path) and explicit instruction "do NOT re-explore — start from these implementations".
7. Files (created / modified / unchanged-but-relevant).
8. Cost: cumulative v2.2 spend ~$3 of $200 envelope (~$2-3 probe wave + ~$0.55 Range A' BBB + cache-fix wave).
9. Gates Compliance: GATE-01/-02/-06/-07/-08 status table.
10. Next Action: `gsd-complete-milestone v2.2` OR continue directly to v2.3 kickoff per user direction.

GATE-06: all four files use ONLY abstract/methodological vocabulary plus the controlled benchmark component names that already appear in the referenced SUMMARYs (Recording Service, FreeSWITCH, etc. are PCM components; their appearance in audit text is descriptive of the per-probe data and does not constitute hardcoding in prompts/logic). No new benchmark words introduced in code.
  </action>
  <verify>
<automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; test -f .planning/milestones/v2.2-ROADMAP.md &amp;&amp; test -f .planning/milestones/v2.2-REQUIREMENTS.md &amp;&amp; test -f .planning/milestones/v2.2-MILESTONE-AUDIT.md &amp;&amp; test -f .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md &amp;&amp; grep -q "v2.2 SHIPPED 2026-06-01\|SHIPPED 2026-06-01" .planning/milestones/v2.2-ROADMAP.md &amp;&amp; grep -q "SHIP-V22-MIN-UNCHANGED" .planning/milestones/v2.2-REQUIREMENTS.md &amp;&amp; grep -q "SHIP-V22-PROBE-D-OPT-IN" .planning/milestones/v2.2-REQUIREMENTS.md &amp;&amp; grep -q "DEFER-V4-TO-V23" .planning/milestones/v2.2-REQUIREMENTS.md &amp;&amp; grep -q "passed" .planning/milestones/v2.2-MILESTONE-AUDIT.md &amp;&amp; grep -q "s_linker14_probe_d_upstream_clean" .planning/milestones/v2.2-MILESTONE-AUDIT.md &amp;&amp; grep -q "opt-in" .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md &amp;&amp; grep -q "v2.3-KICKOFF-SEED.md" .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md &amp;&amp; echo "milestone artifacts OK"</automated>
  </verify>
  <done>
- `.planning/milestones/v2.2-ROADMAP.md` exists with "SHIPPED 2026-06-01", phases 14/15/16/17/18 outcome table, Probe A/B/C/D/A' negative-finding pointers, link to v2.2-MILESTONE-AUDIT and v2.3 seed.
- `.planning/milestones/v2.2-REQUIREMENTS.md` exists with 3 closed requirements (SHIP-V22-MIN-UNCHANGED, SHIP-V22-PROBE-D-OPT-IN, DEFER-V4-TO-V23) and a Deferred-to-v2.3 carry-forward table.
- `.planning/milestones/v2.2-MILESTONE-AUDIT.md` exists with `status: passed`, `verdict: shipped`, frontmatter date 2026-06-01, audit decision "PASSED — milestone ready for completion".
- `.planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md` exists with sections covering shipped/opt-in/declined/v2.3-handoff and references all 4 probe SUMMARYs.
- No benchmark words added to prompts or runtime code (audit-doc references to PCM component names are descriptive only).
  </done>
</task>

<task type="auto">
  <name>Task 3: Top-level updates (ROADMAP / MILESTONES / PROJECT / STATE) + v2.3 kickoff seed</name>
  <files>.planning/ROADMAP.md, .planning/MILESTONES.md, .planning/PROJECT.md, .planning/STATE.md, .planning/v2.3-prep/v2.3-KICKOFF-SEED.md</files>
  <action>
Update top-level planning surfaces to reflect v2.2 SHIPPED + v2.3 anchored, and create the v2.3 kickoff seed.

**File 1: `.planning/ROADMAP.md`** — Use Edit tool. Two changes:

(a) Append to the `## Milestones` list (after v2.1 row):

```
- ✅ **v2.2 — Probe-Wave Trimmed Close** — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) — shipped 2026-06-01. 4 probes ran; 1 strong survivor (Probe D upstream coref rubric) shipped as opt-in carve-out. Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md) and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
```

(b) Add a collapsed details block after the v2.1 details block:

```
<details>
<summary>✅ v2.2 — Probe-Wave Trimmed Close — SHIPPED 2026-06-01</summary>

Probe wave (4 mechanisms) + trimmed close. No new canonical promoted; `s_linker13_min` carried forward unchanged. Probe D ships as opt-in gpt-5.4-only carve-out (`s_linker14_probe_d_upstream_clean`). See `milestones/v2.2-ROADMAP.md`.

</details>
```

(c) Replace the `## Next Milestone` section content (currently "TBD. v2.1 archived. Awaiting v2.2 kickoff. ...") with:

```
## Next Milestone

**v2.3 — Voyager v4 Multi-Role (anchored).** v2.2 archived 2026-06-01. v2.3 anchor: Voyager v4 multi-role architecture (R1–R5) deferred from v2.2 with two PROVEN prerequisites carried forward — DO NOT re-explore:

1. **Per-backend cache infrastructure** — per-(text_stem, comp_hash, backend, model) cache-key for runtime LLM rubrics. Verified SANITY_PASS in `s_linker14_probe_d_upstream_clean.py` (Probe D cache-fix wave 2026-06-01). See [`v2.2-prep/probe-D-cachekey-fix-SUMMARY.md`](v2.2-prep/probe-D-cachekey-fix-SUMMARY.md).
2. **Probe A' vocab-aligned R3** — discourse/syntactic vocabulary (subject-position, anaphora, qualifier clause, ...) replaces textbook SE vocabulary, narrowing the R3/R5 deadlock that falsified original Probe A. Mediastore STRONG_PASS (+1.69pp); BBB WEAK_PASS (R5 0/8, F1 -0.24pp) — v4 is mediastore-viable, BBB-inactive on gpt-5.4. See [`v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md`](v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md) and [`v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md`](v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md).

Full v2.3 kickoff seed: [`v2.3-prep/v2.3-KICKOFF-SEED.md`](v2.3-prep/v2.3-KICKOFF-SEED.md).

Additional v2.3 candidates (carried from v2.1 + v2.2): ADAPTER-01, EXT-04, link provenance data structure, Extended Thinking on judge stages, Claude Probe D re-test with the new cache fix, Self-Refine contingent.
```

**File 2: `.planning/MILESTONES.md`** — Use Edit tool. Prepend a new v2.2 row above the existing v2.0 entry (right after the introductory paragraph; respect the file's existing reverse-chronological ordering — v2.0 currently comes before v1.0 so v2.2 should be inserted at the top of the milestone records, after the file header and intro). Insert content:

```
## v2.2 — Probe-Wave Trimmed Close

**Shipped:** 2026-06-01
**Audit verdict:** `passed` (trimmed scope — see Outcome)
**Production artifact unchanged:** `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` (v2.1 canonical carried forward; no new canonical promoted in v2.2)
**Opt-in carve-out shipped:** `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py` (gpt-5.4 only)

### Delivered

A probe-wave methodology applied to 4 mechanism pillars (Voyager v4 multi-role, problem-statement preamble + cached rubric, Self-Refine on alias judge, upstream-tier rule removal). One strong survivor (Probe D upstream coref rubric, +1.59pp mediastore gpt-5.4, matches Claude baseline) ships as an opt-in gpt-5.4-only carve-out — NOT promoted to canonical because Range BBB Claude FAILED (CONFOUNDED by cross-backend cache reuse, methodologically unblocked but not re-run this milestone). v2.2's canonical is `s_linker13_min` unchanged.

### Stats

| Item | Count |
|------|-------|
| Probes run | 4 (A Voyager v4, B preamble+rubric, C Self-Refine, D upstream rule removal) |
| Probe strong-pass | 1 (D) |
| Probe weak-pass | 1 (C — declined as primary) |
| Probe fail | 1 (B — declined) |
| Probe falsification | 1 (A — R5 100% reject; fixed as A' but BBB WEAK_PASS, deferred to v2.3) |
| New canonical promoted | 0 (s_linker13_min unchanged) |
| Opt-in carve-out shipped | 1 (Probe D, gpt-5.4 only) |
| Variant files retained (negative-finding) | 2 (Probe B, Probe C) |
| Cumulative cost | ~$3 of $200 envelope |

### Per-dataset (carve-out only — Probe D)

| Dataset | Backend | F1 | Anchor | Δ | Verdict |
|---|---|---|---|---|---|
| mediastore | gpt-5.4 | 0.9836 | 0.9677 | +0.0159 | STRONG_PASS (matches Claude baseline) |
| bigbluebutton (original) | gpt-5.4 | 0.7965 | 0.7636 | +0.0329 | STRONG_PASS |
| bigbluebutton (cache-fix re-run) | gpt-5.4 | 0.7748 | 0.7636 | +0.0112 | STRONG_PASS (mean +2.2pp over 2 obs) |
| bigbluebutton | Claude | 0.8073 | 0.8496 | -0.0423 | FAIL — CONFOUNDED by cross-backend cache reuse; per-backend cache fix unblocks re-test |

### Key v2.2 lessons

1. **Probe-wave methodology pays at the milestone-scoping tier.** Four cheap parallel probes (~$3) cut decisively across 4 mechanism pillars in one day, replacing what the original v2.2-MILESTONE-PROPOSAL planned as a 5-phase exploration.
2. **Per-backend cache-key methodology is a precondition for fair cross-model evaluation of runtime LLM rubrics.** The Range D Claude FAIL was indistinguishable from a true cross-model failure until the cache-key fix isolated cross-backend rubric reuse as a confound. Carried to v2.3 as a proven prerequisite.
3. **Vocabulary deadlock is a recurring failure mode in multi-role LLM training architectures.** Probe A's R3/R5 mutual inconsistency on textbook SE vocabulary surfaced via 100% R5 reject. Probe A' resolved it on mediastore by tightening R3 to discourse/syntactic terms — but BBB remained R5 0/8. v4 architecture is dataset-conditional, not universally promotable. Carried to v2.3.
4. **Runtime per-dataset rubrics generalize across pipeline tiers (Phase 12 trim9 seed → Probe D coref) on the same backend, but do NOT trivially transfer across backends.** The trim9 + Probe D pair forms an internal Pareto frontier (Claude likes static, gpt-5.4 likes runtime); shipping both as backend-conditional is a viable v2.3 architecture pattern.

### Files

- Archive: [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md), [`milestones/v2.2-REQUIREMENTS.md`](milestones/v2.2-REQUIREMENTS.md), [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md)
- Probe-wave SUMMARYs: [`v2.2-prep/v2.2-PROBE-WAVE-SUMMARY.md`](v2.2-prep/v2.2-PROBE-WAVE-SUMMARY.md), [`v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md`](v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md), [`v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md`](v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md)
- Per-probe: [`v2.2-prep/probe-D-upstream-SUMMARY.md`](v2.2-prep/probe-D-upstream-SUMMARY.md), [`v2.2-prep/range-D-bbb-SUMMARY.md`](v2.2-prep/range-D-bbb-SUMMARY.md), [`v2.2-prep/probe-D-cachekey-fix-SUMMARY.md`](v2.2-prep/probe-D-cachekey-fix-SUMMARY.md), [`v2.2-prep/probe-A-voyager-v4-SUMMARY.md`](v2.2-prep/probe-A-voyager-v4-SUMMARY.md), [`v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md`](v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md), [`v2.2-prep/probe-A-prime-range-bbb-SUMMARY.md`](v2.2-prep/probe-A-prime-range-bbb-SUMMARY.md), [`v2.2-prep/probe-B-preamble-rubric-SUMMARY.md`](v2.2-prep/probe-B-preamble-rubric-SUMMARY.md), [`v2.2-prep/probe-C-selfrefine-SUMMARY.md`](v2.2-prep/probe-C-selfrefine-SUMMARY.md)
- Scope decision: [`v2.2-prep/v2.2-SCOPE-DECISION.md`](v2.2-prep/v2.2-SCOPE-DECISION.md)

---
```

**File 3: `.planning/PROJECT.md`** — Use Edit tool. Three targeted changes:

(a) `## Current State` section: replace "Status: Between milestones — v2.1 SHIPPED. Awaiting v2.2 kickoff." with "Status: Between milestones — v2.2 SHIPPED 2026-06-01. Awaiting v2.3 kickoff (Voyager v4 multi-role with proven prereqs — see `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`)."

(b) `## Current State` "Canonical artifact" line: keep `s_linker13_min.py` as canonical (unchanged). Append a new line directly below:

```
**Opt-in carve-out (v2.2 shipped 2026-06-01):** `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py` — runtime coref rubric replacing static `COREF_RULES`, enable ONLY when `LLM_BACKEND == openai`. Mediastore gpt-5.4 +1.59pp; BBB gpt-5.4 mean +2.2pp over 2 obs. Claude not promoted (FAIL was confounded — per-backend cache fix unblocks re-test). NOT canonical. See `.planning/v2.2-prep/probe-D-upstream-SUMMARY.md`.
```

(c) `## Past Milestones` section: append a v2.2 bullet after the v2.1 entry:

```
- **v2.2** (2026-06-01) — Probe-Wave Trimmed Close. 4 probes; 1 STRONG survivor (Probe D upstream coref rubric) ships as opt-in gpt-5.4-only carve-out; canonical unchanged (`s_linker13_min`); Voyager v4 multi-role + per-backend cache infra + Probe A' vocab fix deferred to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md), [`milestones/v2.2-REQUIREMENTS.md`](milestones/v2.2-REQUIREMENTS.md), [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
```

(d) `## Next Milestone Candidates (v2.2+)` heading: rename to `## Next Milestone Candidates (v2.3+)` and update the intro line "Active milestone: none. Topics retained from v2.1 deferred items for the next milestone:" to "Active milestone: none. v2.3 anchor: Voyager v4 multi-role with proven per-backend cache infrastructure + Probe A' vocab-aligned R3 (do NOT re-explore — see `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`). Additional topics retained from v2.1 + v2.2 deferred items:". Append two new bullets to the candidates list:

```
- **Claude Probe D re-test with per-backend cache fix** — methodologically ready; will produce a Claude-authored coref rubric and a clean cross-backend Probe D verdict. Cost ~$1.5. Carried from v2.2 (cache-fix wave 2026-06-01).
- **Per-backend cache infrastructure** — proven in `s_linker14_probe_d_upstream_clean.py` per v2.2 SANITY_PASS; v2.3 should adopt as the default pattern for any runtime LLM rubric. Carried from v2.2.
```

(e) Append to the `## Key Decisions` table a new row:

```
| v2.2 trimmed-scope close (ship s_linker13_min unchanged + Probe D opt-in gpt-5.4-only + defer v4 to v2.3) | Probe wave found no Pareto-positive cross-backend mechanism; Probe D is gpt-5.4-only; Voyager v4 architecture is dataset-conditional per Probe A' BBB WEAK_PASS. Hybrid path (Option 1 + Option 3 from `v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md`) preserves both gpt-5.4 lift and Claude floor while deferring v4 to a milestone with adequate prereqs. | Codified 2026-06-01 (v2.2 close) |
```

(f) Update the "Last updated" footer at the bottom: change to "*Last updated: 2026-06-01 — v2.2 milestone close (Probe-Wave Trimmed Close SHIPPED)*".

**File 4: `.planning/STATE.md`** — Use Edit tool. Update the YAML frontmatter:

- `milestone: null` (unchanged — between milestones)
- `milestone_name: "between milestones — v2.2 archived 2026-06-01; v2.3 anchor = Voyager v4 multi-role with proven per-backend cache infra + Probe A' vocab fix"`
- `status: idle`
- `stopped_at: "v2.2 SHIPPED 2026-06-01 (Probe-Wave Trimmed Close). Canonical s_linker13_min unchanged (Claude 0.9506, gpt-5.4 0.9069). Probe D opt-in carve-out registered (gpt-5.4 only). Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab-aligned R3 carried to v2.3 as proven prereqs. See .planning/milestones/v2.2-MILESTONE-AUDIT.md."`
- `last_updated: "2026-06-01T09:30:00.000Z"`
- `last_activity: "2026-06-01 — v2.2 milestone close (quick-mode). Created milestones/v2.2-ROADMAP|REQUIREMENTS|MILESTONE-AUDIT, v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY, v2.3-prep/v2.3-KICKOFF-SEED. Probe D variant docstring + registry description tagged 'v2.2 OPT-IN CARVE-OUT (gpt-5.4 only)'. ROADMAP/MILESTONES/PROJECT updated."`

Body changes:

- `## Current Position` Phase/Plan: leave `none active` / `none active`; update Status to `idle — v2.2 archived 2026-06-01; v2.3 anchor seeded`.
- `## Canonical Artifact (current)` section: unchanged (s_linker13_min still canonical).
- Append a new subsection after Canonical Artifact:

```
## Opt-in Carve-Out (v2.2 shipped 2026-06-01)

- **`src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py`** (`canonical=False`, gpt-5.4 only)
- Mediastore gpt-5.4: F1 0.9836 (+1.59pp vs anchor 0.9677, matches Claude 0.9836 baseline)
- BBB gpt-5.4 (mean of 2 observations): F1 0.7857 (+2.2pp vs anchor 0.7636)
- BBB Claude: FAIL (-4.23pp) — CONFOUNDED by cross-backend cache reuse; per-backend cache fix landed, re-test methodologically ready but not run this milestone
- Mechanism: runtime per-dataset LLM-built coref rubric replaces static `COREF_RULES`
- Gating policy: enable ONLY when `LLM_BACKEND == openai`
```

- Update `## Deferred Items` heading to `## Deferred Items (v2.3 candidates)` and the contained table — keep the existing rows but update the v2.2 rows that resolved this milestone:
  - "Voyager v4 multi-role architecture" row Status: change to `**DEFERRED to v2.3** — Probe A' fix narrowed R3/R5 deadlock; mediastore STRONG_PASS but BBB WEAK_PASS (R5 0/8). v4 is mediastore-viable, BBB-inactive on gpt-5.4. Carried to v2.3 with proven vocab-aligned R3 + per-backend cache infra as named prereqs. See \`.planning/v2.3-prep/v2.3-KICKOFF-SEED.md\`.`
  - "Self-Refine layered on accepted variants" row Status: change to `**DECLINED as primary** — Probe C WEAK_PASS +0.00004pp on mediastore; iter-1 doubled judge cost without changing approved set (GATE-08 flag). Contingent only if v2.3 mainline fails.`
  - "Upstream-tier rule removal (extraction/coref tier — v2.0 EXT-01 evidence)" row Status: change to `**SHIPPED as opt-in carve-out** — Probe D variant \`s_linker14_probe_d_upstream_clean\` registered \`canonical=False\` gpt-5.4-only. See \`.planning/milestones/v2.2-ROADMAP.md\`.`
  - Add a new row: `| v2.3+ | Per-backend cache infrastructure for runtime LLM rubrics | **PROVEN in v2.2** — per-(text_stem, comp_hash, backend, model) cache key landed in s_linker14_probe_d_upstream_clean.py; SANITY_PASS verified. Carry to v2.3 as proven prereq. | v2.2 close (2026-06-01) |`
  - Add a new row: `| v2.3+ | Probe A' vocab-aligned R3 (discourse/syntactic terms) | **PROVEN mediastore, NOT BBB** — discourse-vocab R3 narrows R3/R5 deadlock; mediastore STRONG_PASS (+1.69pp); BBB WEAK_PASS (R5 0/8, F1 -0.24pp). Carry to v2.3 as starting point. | v2.2 close (2026-06-01) |`
  - Add a new row: `| v2.3+ | Claude Probe D re-test with per-backend cache fix | **METHODOLOGICALLY READY** — cache-key isolation landed; fresh Claude rubric will be built on re-run. Cost ~$1.5. Decides whether Probe D extends to cross-backend or stays gpt-5.4-only. | v2.2 close (2026-06-01) |`

- Update `## Session Continuity` block:
  - `Last session: 2026-06-01T09:30:00.000Z`
  - `Stopped at: v2.2 milestone close (quick-mode 260601-bfe). Created milestone artifacts, registered Probe D opt-in carve-out, seeded v2.3 anchor. STATE clean for v2.3 kickoff.`
  - `Resume file: .planning/v2.3-prep/v2.3-KICKOFF-SEED.md`

**File 5: `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`** — Use Write tool. Create the directory implicitly via Write. Content:

```
---
phase: v2.3-PREP
date: 2026-06-01
mode: kickoff-seed
status: anchor-defined-pre-kickoff
v23_anchor: Voyager v4 multi-role (R1-R5) architecture
proven_prereqs:
  - per-backend cache infrastructure (verified 2026-06-01)
  - Probe A' vocab-aligned R3 — discourse/syntactic vocab (verified mediastore 2026-06-01; BBB WEAK_PASS noted)
tags: [v2.3, kickoff-seed, voyager-v4, cache-infra, vocab-fix, do-not-re-explore]
---

# v2.3 Kickoff Seed — Voyager v4 Multi-Role (with Proven Prereqs)

## Purpose

This file exists so that the v2.3 milestone kickoff (whenever it runs) does NOT re-explore work that v2.2 already proved out. It names the anchor and the two proven prereqs as the starting point, with explicit pointers to the verified implementations + per-probe SUMMARYs.

## v2.3 Anchor

**Voyager v4 multi-role architecture (R1-R5)** — deferred from v2.2.

Original proposal: 5-role training harness (R1 actor, R2 generalizer, R3 distillator, R4 categorizer, R5 abstraction validator) producing a skill bank that augments the inference pipeline. See `.planning/v2.2-prep/voyager-v4-architecture-proposal.md` and `scripts/voyager_train_tlr_v4.py` + `scripts/voyager_train_tlr_v4_a_prime.py`.

v2.2 status: **mediastore-viable, BBB-inactive on gpt-5.4**.
- Original Probe A (textbook SE vocab): R5 100% reject → PROBE_FAIL (falsification).
- Probe A' (discourse-vocab R3): mediastore STRONG_PASS (+1.69pp); BBB WEAK_PASS (R5 0/8 reject, F1 -0.24pp).
- The v4 architecture is not falsified per se — it works on mediastore but doesn't lift BBB even with the vocab fix.

v2.3 task: extend v4 to ≥2-dataset viability OR drop to Compact-B (R345 single role with structured CoT, per `.planning/v2.2-prep/v2.2-SCOPE-DECISION.md` fallback note).

## Proven Prereq 1 — Per-Backend Cache Infrastructure

**Status: PROVEN 2026-06-01.** Verified via SANITY_PASS in `s_linker14_probe_d_upstream_clean.py`.

**Implementation reference**: `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py` — `_cache_key`, `_cache_path`, `CACHE_ROOT`. Cache key is per-(text_stem, comp_hash, backend, model). Cache root override via `PROBE_D_CACHE_ROOT` env var. Default cache root: `results/v2_2_probes_range_d_cachefix/cache/`. Legacy 2-key gpt-5.4 rubrics preserved at `results/v2_2_probes/D_upstream/cache/` for historical record.

**Why this is a prereq for v2.3**: any v4 cached artifact (R1 actor proposals, R3 distilled patterns, R5 abstraction-validated skills) MUST be backend-isolated, otherwise the cross-backend confound that broke Range D Claude will silently repeat on every multi-backend evaluation.

**v2.3 instruction**: COPY the cache-key + cache-path pattern from `s_linker14_probe_d_upstream_clean.py` (or refactor it into a small shared utility module if convenient — `helper_v3.py` is the natural location). DO NOT re-derive. DO NOT re-test the methodology.

**Evidence**: `.planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md` (full SANITY_PASS record, before/after delta analysis, gpt-5.4 BBB variance band discussion).

## Proven Prereq 2 — Probe A' Vocab-Aligned R3

**Status: PROVEN mediastore (STRONG_PASS +1.69pp); NOT PROVEN BBB (WEAK_PASS R5 0/8, F1 -0.24pp).**

**Implementation reference**: `scripts/voyager_train_tlr_v4_a_prime.py`. R3 distillator vocabulary tightened to discourse / syntactic terms ONLY: subject-position, predicate, anaphora, parenthetical, namespace prefix, section heading, sentence-position, qualifier clause, cross-reference. FORBIDDEN: role nouns + any architectural-style names.

**Why this is a prereq**: original Probe A R3 used textbook SE vocabulary (controller, scheduler, broker, queue) that R5's 5-style transferability test marked as style-dependent → 100% R5 reject deadlock. The vocab fix unblocks R5 acceptance on mediastore. R3/R5 prompts are now MUTUALLY CONSISTENT on the v4 proposal's design.

**v2.3 instruction**: START from `scripts/voyager_train_tlr_v4_a_prime.py`. DO NOT revert to textbook SE vocabulary. The BBB R5 0/8 result is a downstream-utility finding (validated patterns don't translate into BBB-recoverable inference behavior), not a vocab-alignment failure — treat as a separate v2.3 sub-question rather than a vocab-fix-iteration.

**Evidence**:
- `.planning/v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md` — mediastore STRONG_PASS detail.
- `.planning/v2.2-prep/probe-A-prime-range-bbb-SUMMARY.md` — BBB WEAK_PASS detail.
- `.planning/v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md` — combined rollup with v2.3 implications.

## Open Sub-Questions for v2.3 (Pre-Kickoff Notes)

These do NOT need to be answered before v2.3 kickoff, but should be on the milestone planner's first read:

1. **Compact-B fallback**: per `v2.2-SCOPE-DECISION.md`, if v4 with vocab-aligned R3 fails on a 3rd dataset (teastore or teammates), the fallback is Compact-B — single LLM call internally reconciling proposal + abstraction validation. Is this v2.3 mainline or v2.3 fallback?
2. **Cross-dataset v4 viability**: BBB is one hard dataset; what is the falsification criterion for v4 mainline (≥3 datasets STRONG_PASS? ≥1 dataset STRONG + no FAIL?)
3. **Claude Probe D re-test**: should this run as a v2.3 prep task to unblock the Probe D promotion question, or stay a separate carve-out follow-up?
4. **Backend-conditional architecture pattern**: trim9 (Claude-leaning static) + Probe D (gpt-5.4-leaning runtime) form an internal Pareto frontier. Is `s_linker14_min` = `s_linker13_min` + `if backend == openai: use runtime coref rubric` a v2.3 deliverable, independent of v4?

## Costs Already Spent (Do Not Repeat)

- $2-3 — original probe wave (4 probes × mediastore gpt-5.4).
- ~$2 — Range D BBB (gpt-5.4 + Claude).
- ~$0.55 — Range A' BBB + Probe D cache-fix sanity (this turn).
- **Total v2.2 ≈ $5-6.** No re-runs of these items are needed for v2.3.

## What v2.3 Should NOT Do

1. Re-derive the per-backend cache-key formula. Copy from `s_linker14_probe_d_upstream_clean.py`.
2. Revert R3 to textbook SE vocabulary. The deadlock has been characterized and the discourse-vocab fix is the new baseline.
3. Re-run Probes B and Probe C as primary mechanism candidates. Probe B's failure mode (cold per-dataset rubric over-restriction) and Probe C's failure mode (judge at ceiling → iter-1 no-ops) are characterized. Re-test only if v2.3 mainline (v4 + Probe D extension) fails.
4. Re-run Probe A original (textbook vocab). Falsified.

## What v2.3 SHOULD Do First

1. Read this seed + the 3 referenced SUMMARYs (probe-D-cachekey-fix, probe-A-prime-vocab-aligned, v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX).
2. Decide cross-dataset v4 viability scope (sub-question #2 above).
3. Decide whether Claude Probe D re-test runs first (sub-question #3 above) — if yes, it's a 1-task prep step.
4. Standard `/gsd:new-milestone v2.3` workflow can then proceed with the anchor + prereqs pre-defined.
```

GATE-06 / BENCHMARK_TABOO: all five files use only abstract / methodological vocabulary. PCM component names appearing in the audit/decision text (Recording Service, FreeSWITCH, HTML5 Server, Redis PubSub, etc.) are descriptive references to the v2.2-prep SUMMARY data and do NOT enter prompts or runtime logic. No new prompt examples introduced.

GATE-07: no canonical promotions or demotions changed. `s_linker13_min` stays canonical=True. Probe D stays canonical=False. Registry stable.
  </action>
  <verify>
<automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; test -f .planning/v2.3-prep/v2.3-KICKOFF-SEED.md &amp;&amp; grep -q "v2.2 — Probe-Wave Trimmed Close" .planning/ROADMAP.md &amp;&amp; grep -q "SHIPPED 2026-06-01" .planning/ROADMAP.md &amp;&amp; grep -q "v2.3-KICKOFF-SEED.md\|v2\.3.*Voyager v4" .planning/ROADMAP.md &amp;&amp; grep -q "v2.2 — Probe-Wave Trimmed Close" .planning/MILESTONES.md &amp;&amp; grep -q "s_linker14_probe_d_upstream_clean" .planning/MILESTONES.md &amp;&amp; grep -q "v2.2 SHIPPED 2026-06-01\|v2.2.*SHIPPED" .planning/PROJECT.md &amp;&amp; grep -q "Opt-in carve-out" .planning/PROJECT.md &amp;&amp; grep -q "v2.2 archived 2026-06-01\|v2.2 SHIPPED 2026-06-01" .planning/STATE.md &amp;&amp; grep -q "Opt-in Carve-Out" .planning/STATE.md &amp;&amp; grep -q "Per-Backend Cache Infrastructure" .planning/v2.3-prep/v2.3-KICKOFF-SEED.md &amp;&amp; grep -q "Probe A' Vocab-Aligned R3\|Probe A. Vocab-Aligned" .planning/v2.3-prep/v2.3-KICKOFF-SEED.md &amp;&amp; grep -q "DO NOT re-explore\|do NOT re-explore\|DO NOT.*re-test\|DO NOT.*re-derive" .planning/v2.3-prep/v2.3-KICKOFF-SEED.md &amp;&amp; python -c "import run_ablation; assert run_ablation.VARIANT_SPECS['s_linker13_min'].get('canonical') is True; assert run_ablation.VARIANT_SPECS['s_linker14_probe_d_upstream_clean'].get('canonical') is False; print('registry invariants OK after edits')" &amp;&amp; echo "top-level + v2.3 seed OK"</automated>
  </verify>
  <done>
- `.planning/ROADMAP.md`: v2.2 row added to Milestones list with shipped date 2026-06-01; v2.2 details block added; Next Milestone section replaced with v2.3 anchor + per-backend cache infra + Probe A' vocab fix references + link to v2.3-KICKOFF-SEED.md.
- `.planning/MILESTONES.md`: v2.2 entry inserted at the top of the milestone records (above v2.0) with shipped date, audit verdict, stats, per-dataset carve-out table, key v2.2 lessons, files block.
- `.planning/PROJECT.md`: Current State updated to "v2.2 SHIPPED 2026-06-01 ... Awaiting v2.3 kickoff"; opt-in carve-out line added below canonical artifact; v2.2 bullet appended to Past Milestones; Next Milestone Candidates heading renamed to v2.3+ with v2.3 anchor intro + Claude Probe D re-test + per-backend cache infra candidates added; v2.2 close decision row appended to Key Decisions; footer date updated to 2026-06-01.
- `.planning/STATE.md`: frontmatter `milestone_name` / `stopped_at` / `last_activity` / `last_updated` updated; Current Position Status reflects v2.2 archived; new "Opt-in Carve-Out (v2.2 shipped 2026-06-01)" subsection added; Deferred Items table updated (Voyager v4 → DEFERRED to v2.3, Self-Refine → DECLINED as primary, Upstream-tier → SHIPPED as opt-in carve-out, plus 3 new rows for per-backend cache infra / Probe A' vocab fix / Claude Probe D re-test); Session Continuity updated to point at the v2.3 kickoff seed.
- `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md` exists with v2.3 anchor (Voyager v4 multi-role), 2 proven-prereq subsections (per-backend cache infrastructure + Probe A' vocab-aligned R3), open sub-questions, "What v2.3 should NOT do" and "What v2.3 SHOULD do first" sections, costs-already-spent note.
- Registry invariants still hold: `s_linker13_min` canonical=True, `s_linker14_probe_d_upstream_clean` canonical=False.
- No frozen artifact (`s_linker13*.py`, `prompts_v*.py`, `helper_v3.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `ilinker*.py`) touched.
  </done>
</task>

</tasks>

<verification>
After all 3 tasks complete, run:

```bash
cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && \
  python -c "import run_ablation; assert run_ablation.VARIANT_SPECS['s_linker13_min'].get('canonical') is True; assert run_ablation.VARIANT_SPECS['s_linker14_probe_d_upstream_clean'].get('canonical') is False; assert 'v2.2 OPT-IN CARVE-OUT' in run_ablation.VARIANT_SPECS['s_linker14_probe_d_upstream_clean']['description']; assert 'gpt-5.4 only' in run_ablation.VARIANT_SPECS['s_linker14_probe_d_upstream_clean']['description']; print('GATE-07 + carve-out tag OK')" && \
  for f in .planning/milestones/v2.2-ROADMAP.md .planning/milestones/v2.2-REQUIREMENTS.md .planning/milestones/v2.2-MILESTONE-AUDIT.md .planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md .planning/v2.3-prep/v2.3-KICKOFF-SEED.md; do test -f "$f" || { echo "MISSING: $f"; exit 1; }; done && \
  echo "v2.2 close artifacts all present"
```

Frozen-artifact invariant check (optional — confirms no v2.0/v2.1 artifact was touched):

```bash
cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && \
  git diff --name-only HEAD | grep -E "src/llm_sad_sam/(linkers/experimental/s_linker13(_min|_clean|_clean_v3|)?\.py|linkers/experimental/prompts_v[23]\.py|linkers/experimental/helper_v3\.py|linkers/experimental/ilinker[1-3]\.py|core/data_types_v2\.py|core/document_loader_v2\.py|pcm_parser_v2\.py)" && { echo "FROZEN ARTIFACT TOUCHED — VIOLATION"; exit 1; } || echo "Frozen artifacts untouched OK"
```

The only `src/` file expected to appear in `git diff --name-only` is `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py`.
</verification>

<success_criteria>
- All 5 verification commands in Task 3 pass and the verification block above runs clean.
- `s_linker13_min` remains v2.2 canonical (`canonical=True`, no code change).
- Probe D variant carries the `v2.2 OPT-IN CARVE-OUT (gpt-5.4 only)` marker in BOTH its module docstring AND its `run_ablation.VARIANT_SPECS` description.
- v2.2 milestone artifacts (ROADMAP / REQUIREMENTS / AUDIT) exist under `.planning/milestones/` with `shipped 2026-06-01` and 3/3 requirements closed.
- v2.2-prep close SUMMARY exists.
- ROADMAP.md, MILESTONES.md, PROJECT.md, STATE.md all reflect v2.2 SHIPPED + v2.3 anchored.
- v2.3 kickoff seed exists at `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md` with the 2 proven prereqs and a "do NOT re-explore" instruction.
- No frozen v2.0/v2.1 artifacts touched.
- No new benchmark words enter prompts or runtime logic (BENCHMARK_TABOO clean).
</success_criteria>

<output>
After completion, this quick task has no SUMMARY (quick mode); the milestone close SUMMARY is `.planning/v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md` (created by Task 2).

Recommended follow-up commit message (executor decides; not part of plan):
`docs(v2.2): close milestone — ship s_linker13_min unchanged + Probe D opt-in (gpt-5.4 only) + defer v4 to v2.3`
</output>
