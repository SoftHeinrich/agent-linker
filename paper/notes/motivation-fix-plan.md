# Motivation rewrite — fix-plan & decisions (for later inspection)

Date: 2026-06-27. Scope: `sections/motivation.tex` only (terminology/`ruler` cleanup
in `intro.tex`/`metric.tex` deferred — see "Out of scope" below).

## Decisions locked during /gsd-explore

1. **Numbers policy.** No results from *our* approach (`\approach` = AAlinker) in the
   motivation. *Prior-approach* results are allowed — `\Artemis` and `\TransArc` are both
   baselines (eval.tex:71), so their F1s (0.998 / 0.943 / 0.800) stay. Data-distribution
   stats (99.3% / 0.4% / 20%) stay.
2. **Example redesign (need 1 = implicit reference, not alias).** Drop the alias case.
   Add a short antecedent **S6** ("…the gui renders the main user interface.") that
   introduces `gui`; rewrite **S7** to use the pronoun **"it"** ("…it knows the user and
   his preferences."). S7 now carries two needs at once: resolving "it"→`gui` (knowledge)
   and rejecting the `preferences` word match (validation).
3. **Wording.** Never use "surface" — use **"mention"**. Simple, uniform language; no
   poetic terms ("ruler"). Keep S6 as short as possible.
4. **Grain.** Code grain = **package/directory**; reserve "component" for architecture-model
   entities (phrase the skew as components *expanding into* package files).
5. **doc-code is the downstream composition of doc-model** — state explicitly.

## Per-cluster plan (all applied)

- **Opening (4 beats):** task via gold links → two hops, doc-code downstream of doc-model →
  S11→`preferences`→package gold example → transitivity consequence → the challenge
  (S6 introduces gui; S7 "it" + word "preferences").
- **Reusable Project Knowledge:** cut alias paragraph; keep only the implicit-reference
  need, motivating an implicit linker that reuses resolved knowledge. Numberless. Removed
  the `% VERIFY` ablation comment (belongs in Results).
- **Validation:** describe the validator's reasoning on S7 (name/path match vs. sentence
  evidence → reject), numberless. Removed the `% VERIFY` path-filter comment (→ Results).
- **Architecture-Driven Evaluation:** define **link-level F1** first; replace every
  "file-level"→"link-level" (prose + fig:motivation caption); frame skew as package/file
  inequality; keep the Artemis-vs-TransArc baseline contrast; "Therefore" closer pointing
  to `\autoref{sec:metric}`.
- **Style:** fixed typos (`describesthe`, `discused`, `previous approach`→`approaches`);
  applied Wang motivation moves (SCOPE openers, active verbs, `-ing` tails, Therefore closer).

## Figure (`figures/jabref_trace_example`)

- Generator is **`figures/jabref_trace_example.py`** (matplotlib/Agg) — the drawio CLI does
  not run headless in this env. Edited the `.py` (regenerates `.pdf`/`.png`) and kept the
  `.drawio` mirror in sync.
- New layout: `gui` (top) ← S6, S7; `preferences` (bottom) ← S11 → code. Dashed red
  S7→`preferences` = false positive. Sentences in numeric order S6, S7, S11.

## Open / verify items (left as %VERIFY in the .tex)

- Confirm 99.3% / 0.4% / 20% against `fig:motivation` data; "fifty times" = 20 / 0.4.

## Out of scope (deferred, flagged)

- `intro.tex`: 7× "file-level" → "link-level", 5× "ruler" → "metric" (reviewer JK flagged
  "ruler" at intro.tex:74). `metric.tex`: 1× "ruler". Do a paper-wide consistency pass later.
- Benchmark taboo (`agent-linker/BENCHMARK_TABOO.md`) applies to **LLM prompts**, not paper
  prose — JabRef/`gui`/`preferences` in the motivation are fine.
