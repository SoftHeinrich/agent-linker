# Intro rewrite — fix-plan & decisions (for later inspection)

Date: 2026-06-27. Scope: `sections/intro.tex` only. Consolidates the five inline
`%TODO` clusters in the intro into one spec. Sibling of
`notes/motivation-fix-plan.md`.

**STATUS: all four clusters applied 2026-06-27** (Wang style + linguistic playbook).
The inline `%TODO` comments were removed as each fix landed (a stale TODO next to
fixed text misleads). Structural checks pass: itemize balanced, exactly 3 `\item`,
braces 35/35, no glued macros. No local LaTeX toolchain — not PDF-compiled here.

## Decisions locked during /gsd-explore

1. **Scope = intro only.** Do not touch other sections here. The paper-wide
   "surface"/simple-English pass stays deferred (see "Out of scope").
2. **Alias-table claim is a factual fix, not just a simplify.** Verified against the
   code (see "Verified facts"): pronoun resolution does **not** always need the alias
   table, so the current wording overstates it.
3. **Drop the replication package as a listed contribution** → four contributions
   become **three**. Replication stays *cited* where it already is (lines 15, 54); it
   is just no longer its own bullet.
4. **"Simple English throughout"** = short sentences (target 15–20 words avg) and
   high-school vocab, per the repo writing rule. Applies to the whole intro.
5. **Inline TODOs stay** in `intro.tex`; clear them only after each fix lands.

## Per-cluster plan

### Cluster 1 — jargon "surface" + simple English (TODO `intro.tex:3`)
- **"surface" reappears, live at line 44:** "…ordinary English words whose *surface
  match* would catch too much." Replace "surface match" with a plain term —
  **"name match"** or **"word match"**. ("mention", used in the motivation, does not
  fit here because this is the lexical string-match sense.)
- Commented-out bullet at line 74 also uses "surface" ("how trace links *surface*…") —
  low priority, it is commented out; align it if/when that bullet is restored.
- **Simple-English pass over the whole intro.** Known rough spots to fix while here:
  - line 10 "the code files that implementing roles" → grammar ("…the code files that
    implement them" / "…that play those roles").
  - line 12–13 "much more easier" → "much easier"; tighten the two clauses.
  - line 47 "builds this knowledge **by inspect over** the document" → "by inspecting".
  - line 33 "so **conincidence** mentions can bring false links" → typo "coincidental";
    consider "so a coincidental word match can add false links".

### Cluster 2 — simplify Para 4 + alias factual fix (TODO `intro.tex:41`)
- **Factual fix (VERIFIED).** Line 45 reads "(iii) resolving a pronoun **needs the
  alias table** to confirm that the pronoun stands for a component." This is too
  strong. Rephrase so the alias table is a **fallback**, e.g.:
  > "(iii) resolving a pronoun means confirming its antecedent names a component —
  > by the component's own name, or, when that is absent, by an alias."
  This also reconciles line 45 with line 49 ("checking each one against the noun it
  points back to"): the check is on the antecedent sentence; the alias table is only
  the fallback path.
- **Simplify the paragraph (lines 40–54).** It is the longest in the intro. Tighten:
  - line 40 "the first multi-stage \ac{LLM} workflow **for architectural documentation
    to model trace-link recovery**" → smoother (e.g. "…for recovering doc-model
    trace links").
  - shorten the (i)/(ii)/(iii) challenge list and the two-linker description; keep one
    idea per sentence.

### Cluster 3 — soften metric claim + simplify Para 5 (TODO `intro.tex:59–60`)
- **Soften / scope the bold claim.** Line 61: "**The standard metric does not measure
  what it claims to measure.**" Scope it to our setting. Candidate phrasings (pick one):
  - "**On these tasks, the standard metric hides the failures that matter.**"
  - "**The standard metric does not capture what matters on these tasks.**"
- **Simplify Para 5 (lines 62–65).** Line 62 is one very long sentence — split it into
  two or three short ones (the 0.998 vs 0.943 contrast, then the dropped whole
  component, then the missed tenth of sentences).

### Cluster 4 — contributions < 10 words each + drop replication (TODO `intro.tex:71`)
- **Drop bullet 4** (replication package, line 78).
- **Update line 69** "This work makes **four** contributions" → "**three**".
- **Tighten each remaining bullet** so its lead statement is **under 10 words** (detail
  + `\autoref` can follow). Draft leads:
  - "\approach, a training-free multi-stage \ac{LLM} linking workflow."
  - "An architecture-driven evaluation suite for the imbalanced benchmark."
  - "An empirical study over five benchmark projects, with ablations."

## Verified facts

- **Pronoun resolution does NOT always use the alias table.** Canonical linker
  `src/llm_sad_sam/linkers/experimental/s_linker20_union.py`,
  `_antecedent_supports_resolution()` (lines 878–895): it first checks the **canonical
  component name** in the antecedent sentence (`has_standalone_mention`, line 884) and
  returns true **without** the alias table; the alias loop (lines 888–894) runs only as
  a fallback, and only when `doc_knowledge` exists. (Note: this branch ships
  `s_linker20_union`, not `s_linker13_min`/`s_linker19`, but the coref logic is the
  same.) → drives the Cluster 2 factual fix.

## Open / verify items

- Softened metric claim (Cluster 3) — RESOLVED, chose: "On these tasks, the standard
  metric hides the failures that matter."
- Contribution bullets are now **bare <10-word leads** (no methodology/consequence
  clause), per the explicit "below 10 words" directive. This is terser than Wang's
  CLAIM->METHODOLOGY->CONSEQUENCE bullet form (SECTION_PLAYBOOK §2.A T4); re-expand if
  reviewers want more substance per bullet.
- Replication is no longer a contribution bullet but stays cited in prose
  (`intro.tex` para 4 + para 1 `\cite{replication}`), so the ICSE early-pointer holds.
- Not PDF-compiled locally (no toolchain) — confirm on the next Overleaf build.

## Out of scope (deferred, flagged)

- **Paper-wide** "surface" → plain term and the broader simple-English pass across all
  sections — already deferred once in `motivation-fix-plan.md` ("Out of scope"); fold
  the intro's "surface" into that same later consistency pass if a paper-wide sweep is
  done instead of an intro-local edit.
- Restoring the commented-out "knowledge account" contribution bullet (line 74) — not
  part of this consolidation.
