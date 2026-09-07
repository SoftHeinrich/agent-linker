---
phase: quick-260610-lio
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/scratch/prompts_v5.py
  - .planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md
autonomous: true
requirements: [GATE-06, GATE-01]
must_haves:
  truths:
    - "Candidate A (CUT) is applied to the scratch sandbox, the coref golden test passes in scratch mode, and GATE-06 grep finds zero taboo hits."
    - "Candidate B (NON-SE REWRITE) is applied to the scratch sandbox, the coref golden test passes in scratch mode, and GATE-06 grep finds zero taboo hits."
    - "No frozen file (s_linker19.py, s_linker13_min.py, s_linker20.py, src/.../prompts_v5.py) is modified — all candidate edits land only in tests/scratch/prompts_v5.py and are reverted at the end."
    - "SUMMARY records both candidate texts verbatim, the free-checks pass/fail verdict, an honest behavioral caveat, a recommendation, and the paid-confirmation design."
  artifacts:
    - path: ".planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md"
      provides: "Two candidate few-shot texts + free-checks verdict + behavioral caveat + recommendation"
      min_lines: 40
  key_links:
    - from: "tests/scratch/prompts_v5.py (ANTECEDENT_ALIAS_RULES)"
      to: "tests/scratch/s_linker19.py (_prompt_coref via import)"
      via: "from tests.scratch.prompts_v5 import ANTECEDENT_ALIAS_RULES"
      pattern: "ANTECEDENT_ALIAS_RULES"
---

<objective>
Explore whether the few-shot ("Examples:" block) inside the ANTECEDENT_ALIAS_RULES
prompt constant can be either (A) CUT entirely or (B) REWRITTEN to a domain-neutral /
hardware example, for benchmark generality (GATE-06), while staying snapshot-safe
(coref golden test) and GATE-06-clean (no benchmark vocabulary).

The current few-shot uses a software-engineering flavored example ("TaskScheduler" /
"scheduler"). It is not a benchmark component (GATE-06 is nominally clean today), but it
is SE-domain. We want two candidate replacements and a free verdict on each.

Purpose: De-risk a future generality trim of ANTECEDENT_ALIAS_RULES by establishing, with
zero paid LLM calls, which candidate is snapshot-safe and GATE-06-clean — and to be brutally
honest that snapshot-safety is NOT behavior-safety (per 48-REGRESSION-ANALYSIS: golden tests
are cached-replay / behavior-blind).

Output: Two candidate few-shot texts, a free-checks pass/fail table per candidate, a behavioral
caveat, a recommendation, and the design for the paid confirmation that a real ship would need.
This is EXPLORATORY — NOT a shipped change to any frozen file.

CRITICAL CONSTRAINTS (read before any edit):
- GATE-01 (byte-equal frozen files): NEVER touch src/.../prompts_v5.py, s_linker19.py,
  s_linker13_min.py, s_linker20.py. All candidate edits go ONLY into
  tests/scratch/prompts_v5.py (the sandbox copy already wired to tests/scratch/s_linker19.py
  via `from tests.scratch.prompts_v5 import ...`, exercised when SAD_SAM_LINKER_SOURCE=scratch).
- NO paid LLM sweeps / NO openai-backend runs. Free checks only:
  (1) coref golden test in scratch mode, (2) GATE-06 taboo grep, (3) plain inspection.
- The scratch sandbox tests/scratch/prompts_v5.py is currently byte-identical to the
  production constant. It MUST be restored to that byte-identical state at the end of this
  plan (the scratch copy is a shared working file; leave it clean).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/48-sweep/48-REGRESSION-ANALYSIS.md
@CLAUDE.md
@BENCHMARK_TABOO.md
@tests/scratch/prompts_v5.py
@tests/harness/inputs.py

# Coupling notes (verified during planning, no action needed):
# - The few-shot is MID-prompt. The harness reverse-extractors key off the prompt
#   OPENER ("Resolve anaphoric references...") and TERMINAL markers (COREF_RULES /
#   ANTECEDENT_ALIAS_RULES / "Return JSON:" / "JSON only:"). Editing the Examples block
#   inside ANTECEDENT_ALIAS_RULES does NOT touch any ACCEPTED_PREFIXES opener or any
#   terminal_marker, so tests/harness/inputs.py needs NO change.
# - In scratch mode the coref test SKIPS the byte-equal prompt-rebuild assertion
#   (test_s_linker20_prompt_coref.py:81) and keeps ONLY the parsed-output snapshot gate.
#   That gate replays a CACHED response_text — it is blind to live LLM behavior. This is
#   exactly why the verdict below carries a hard behavioral caveat.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Apply + free-check Candidate A (CUT)</name>
  <files>tests/scratch/prompts_v5.py</files>
  <action>
First snapshot the current scratch constant so it can be restored byte-exactly later:
`cp tests/scratch/prompts_v5.py /tmp/lio_prompts_v5.orig`.

Then edit ONLY the ANTECEDENT_ALIAS_RULES triple-quoted string in
tests/scratch/prompts_v5.py (the copy at ~line 104, NOT src/.../prompts_v5.py). Replace
the whole constant body with Candidate A — the CUT variant — which removes the entire
"Examples:" block (both example lines and the blank line that precedes them) while keeping
the true/false definitions and the trailing "Default to true..." line VERBATIM. The exact
new value of ANTECEDENT_ALIAS_RULES is:

"""For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component."""

Do NOT change any other constant, opener, or terminal marker. Do NOT touch
tests/harness/inputs.py. Record the verbatim Candidate A text for the SUMMARY.
  </action>
  <verify>
    <automated>SAD_SAM_LINKER_SOURCE=scratch python -m pytest tests/test_s_linker20_prompt_coref.py -q</automated>
    GATE-06 taboo grep on the candidate text (expect ZERO matches). Build the taboo
    alternation from BENCHMARK_TABOO.md component/alias/keyword terms and grep the NEW
    ANTECEDENT_ALIAS_RULES body case-insensitively, e.g.:
    `python -c "import tests.scratch.prompts_v5 as p; print(p.ANTECEDENT_ALIAS_RULES)" | grep -v '^#' | grep -Ein 'watermark|reencod|recommender|persistence|registry|kurento|freeswitch|redis|pubsub|datastore|bibdatabase|bibentry|globals|\bUI\b|\blogic\b|\bstorage\b|\bcommon\b|\bclient\b|\bfacade\b|\bcache\b|\bauth\b|\bserver\b|\bmodel\b|\badapter\b|\border\b|\bprocessor\b|\bsocket\b|\blayer\b|\bevent\b|\bconfig\b|TaskScheduler|scheduler' || echo "GATE-06 CLEAN: zero taboo hits"`
    (Candidate A removes the SE example, so even the non-benchmark "scheduler"/"TaskScheduler"
    SE token should be gone — the grep includes it to confirm the cut actually happened.)
  </verify>
  <done>
Coref golden test passes in scratch mode (40 snapshots pass, prompt-rebuild assertion
skipped per scratch toggle). GATE-06 grep prints "GATE-06 CLEAN: zero taboo hits" AND shows
no remaining "scheduler"/"TaskScheduler" token (confirming the Examples block was actually
removed). Candidate A text captured verbatim for the SUMMARY.
  </done>
</task>

<task type="auto">
  <name>Task 2: Apply + free-check Candidate B (NON-SE / hardware rewrite)</name>
  <files>tests/scratch/prompts_v5.py</files>
  <action>
Edit ONLY the ANTECEDENT_ALIAS_RULES triple-quoted string in tests/scratch/prompts_v5.py
again, replacing the Candidate A body with Candidate B — the NON-SE REWRITE variant. This
keeps the "Examples:" block but swaps the SE "TaskScheduler"/"scheduler" example for a
domain-neutral hardware example that still demonstrates terminal-word aliasing. Use
"PowerSupplyUnit" (canonical, multi-word hardware component) vs its terminal-word alias
"unit". GATE-06 check: "PowerSupplyUnit", "power", "supply", "unit", "voltage", "regulate"
are NOT among the 5 benchmark projects' component names, aliases, or keywords in
BENCHMARK_TABOO.md (confirmed during planning). The exact new value of
ANTECEDENT_ALIAS_RULES is:

"""For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Examples:
- COMPONENTS contains "PowerSupplyUnit"; antecedent: "the unit regulates voltage" -> true (uses terminal "unit", not canonical "PowerSupplyUnit").
- COMPONENTS contains "PowerSupplyUnit"; antecedent: "PowerSupplyUnit regulates voltage" -> false (canonical name verbatim).

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component."""

Do NOT change any other constant, opener, or terminal marker. Do NOT touch
tests/harness/inputs.py. Record the verbatim Candidate B text for the SUMMARY.
  </action>
  <verify>
    <automated>SAD_SAM_LINKER_SOURCE=scratch python -m pytest tests/test_s_linker20_prompt_coref.py -q</automated>
    GATE-06 taboo grep on the Candidate B text (expect ZERO benchmark matches). Same
    alternation as Task 1 but WITHOUT the SE-token clause (the SE tokens are intentionally
    gone; the hardware example must not introduce any benchmark term):
    `python -c "import importlib,tests.scratch.prompts_v5 as p; importlib.reload(p); print(p.ANTECEDENT_ALIAS_RULES)" | grep -v '^#' | grep -Ein 'watermark|reencod|recommender|persistence|registry|kurento|freeswitch|redis|pubsub|datastore|bibdatabase|bibentry|globals|\bUI\b|\blogic\b|\bstorage\b|\bcommon\b|\bclient\b|\bfacade\b|\bcache\b|\bauth\b|\bserver\b|\bmodel\b|\badapter\b|\border\b|\bprocessor\b|\bsocket\b|\blayer\b|\bevent\b|\bconfig\b|\bmedia\b|\baudio\b|\bpackaging\b|\brecording\b|\bconversion\b|\bcascade\b|\bdedicated\b|\binternal\b|\bpreferences\b' || echo "GATE-06 CLEAN: zero taboo hits"`
    (If the python -c reload caching is awkward, instead pipe the file body directly:
    `grep -v '^#' tests/scratch/prompts_v5.py | grep -A8 'For each resolution' | grep -Ein '<same alternation>' || echo "GATE-06 CLEAN"`.)
  </verify>
  <done>
Coref golden test passes in scratch mode (40 snapshots pass). GATE-06 grep prints
"GATE-06 CLEAN: zero taboo hits" for the hardware example (PowerSupplyUnit/unit/voltage
introduce no benchmark term). Candidate B text captured verbatim for the SUMMARY.
  </done>
</task>

<task type="auto">
  <name>Task 3: Restore scratch sandbox + write exploratory SUMMARY</name>
  <files>tests/scratch/prompts_v5.py, .planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md</files>
  <action>
First RESTORE the scratch sandbox to its original byte-identical state so the shared
working file is left clean: `cp /tmp/lio_prompts_v5.orig tests/scratch/prompts_v5.py`,
then confirm with `git diff --stat tests/scratch/prompts_v5.py` (expect NO diff) and a
final `SAD_SAM_LINKER_SOURCE=scratch python -m pytest tests/test_s_linker20_prompt_coref.py -q`
(expect 40 passed) to prove the restore is sound.

Then write the SUMMARY documenting the exploration. Include:
1. Both candidate texts VERBATIM (Candidate A CUT, Candidate B NON-SE/hardware rewrite).
2. A free-checks results table: per candidate, coref-golden-test (pass/fail) and GATE-06
   grep (clean/hits). Both expected: snapshot-safe + GATE-06-clean.
3. An HONEST behavioral caveat (load-bearing): per 48-REGRESSION-ANALYSIS, the scratch coref
   golden test replays a CACHED parsed response and SKIPS the prompt-rebuild assertion — it is
   behavior-BLIND. Snapshot-safe != behavior-safe. ANTECEDENT_ALIAS_RULES sets the
   antecedent_via_alias flag, which is a coref-sensitive behavior; a real cut/rewrite decision
   needs an N>=3 LIVE coref-sensitive sweep (TeaMmates is the coref-FP-sensitive dataset per the
   regression analysis) — which is OUT OF SCOPE for this free quick task. Cite the N=3/N=6
   variance finding (per-variant macro stdev ~1.4pp; effects <2pp not resolvable at feasible N).
4. A recommendation: which candidate is most promising and WHY (e.g. CUT maximizes generality
   and removes the only SE-domain token, but drops the worked demonstration of terminal-word
   aliasing; the NON-SE rewrite preserves the few-shot's teaching signal while removing SE
   flavor — safer bet if the few-shot is load-bearing for the flag). State the recommendation as
   a hypothesis, not a verdict, given the behavioral caveat.
5. The paid confirmation design: targeted single-dataset (TeaMmates) live coref sweep, N>=3,
   comparing s20 control vs s20+CandidateA vs s20+CandidateB, tallying antecedent_via_alias /
   coref-FP by source-vs-gold; only call a candidate safe if its coref-FP distribution overlaps
   the control within the variance band. Note the approximate cost class (single-dataset, ~$3-4
   per variant per the regression-analysis cost notes) and that it is explicitly deferred.

Use the summary template structure.
  </action>
  <verify>
    <automated>git diff --stat tests/scratch/prompts_v5.py; SAD_SAM_LINKER_SOURCE=scratch python -m pytest tests/test_s_linker20_prompt_coref.py -q</automated>
    Restore confirmed: `git diff --stat tests/scratch/prompts_v5.py` shows NO change, and
    coref golden test prints "40 passed". SUMMARY file exists and contains both verbatim
    candidate texts, the free-checks table, the behavioral caveat, the recommendation, and the
    paid-confirmation design:
    `grep -c "PowerSupplyUnit" .planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md`
    (expect >=1) and
    `grep -Ei "behavior-blind|behavior-safe|cached|N>=3|N=3" .planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md`
    (expect the caveat present).
  </verify>
  <done>
Scratch sandbox restored byte-identical (git shows no diff, 40 snapshots still pass).
SUMMARY exists with both verbatim candidates, free-checks table, explicit behavior-blind
caveat, a recommended candidate (as hypothesis), and the deferred paid-confirmation design.
  </done>
</task>

</tasks>

<verification>
- No frozen file modified: `git status --porcelain src/llm_sad_sam/linkers/experimental/prompts_v5.py src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py src/llm_sad_sam/linkers/experimental/s_linker20.py` is EMPTY.
- Scratch sandbox restored: `git diff --stat tests/scratch/prompts_v5.py` is EMPTY.
- Coref golden test passes in scratch mode at start and end (40 snapshots).
- Both candidates documented verbatim in the SUMMARY with their free-checks verdicts.
- GATE-06 grep clean for both candidates.
- Behavioral caveat explicitly present (snapshot-safe != behavior-safe; live N>=3 needed).
- NO openai-backend runs / NO paid sweeps executed (verify by absence of any run_ablation.py
  invocation in the work log).
</verification>

<success_criteria>
- Candidate A (CUT) and Candidate B (NON-SE/hardware rewrite) each: applied to scratch,
  coref golden test passes, GATE-06 grep clean.
- The only modified tracked file at the end is the SUMMARY (scratch sandbox restored, frozen
  files untouched).
- SUMMARY delivers: 2 verbatim candidate texts + per-candidate free-checks verdict + honest
  behavioral caveat + recommendation (hypothesis) + deferred paid-confirmation design.
</success_criteria>

<output>
Create `.planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md` when done
</output>
