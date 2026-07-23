# Integrating the dependency-importance result into the paper motivation

## What the two motivation figures currently are

- **Figure 1** `fig:example` (`figures/jabref_trace_example.py`) — the doc→model→code
  *task* diagram: sentences S6/S7/S11 → components `gui`/`preferences` → code packages,
  with the false-positive edge a judge must reject. **Its job is linking *difficulty***
  (aliases, pronouns, false positives), not evaluation.
- **Figure 2** `fig:motivation` (`figures/jabref_motivation.py`, data
  `jabref_motivation_data.csv`) — the *size-bias* bar chart: per-component **link share
  vs sentence share**, caption *"small components own almost none of the links, but the
  documentation describes each of them equally."* This is already the "small ≠
  unimportant" figure, and it lists all six components.

## Verdict: the result belongs in Figure 2, not Figure 1

The dependency-importance finding is an **evaluation-motivation** ("small-footprint
components are important, so missing them should cost more than link-F1 charges"). That
is exactly Figure 2's argument, expressed today via *documentation attention*. Our result
adds a second, independent axis of importance — **code dependence** — for the same
components. Putting coupling numbers into Figure 1 would overload a clean task diagram,
introduce metrics before they are defined, and clash with the plain-English convention.

### Recommended: add a third per-component series to Figure 2

Figure 2 today plots two quantities per component. Add a third: **share of code that
depends on the component** (from `reports/IMPORTANCE.csv`). Two framings:

| component | link share (`linkpair_pct`) | sentence share (`sent_pct`) | **dependency share** (Ca ÷ ΣCa) | Ca_share (Ca ÷ other files) |
|-----------|---------------------------:|----------------------------:|--------------------------------:|----------------------------:|
| model | 18.1% | 60% | 46.6% | 57.1% |
| logic | 47.0% | 40% | 27.9% | 48.9% |
| gui | 34.2% | 40% | **1.6%** | 3.1% |
| **preferences** | **0.44%** | 20% | **17.6%** | 18.8% |
| cli | 0.19% | 20% | 0.1% | 0.1% |
| globals | 0.05% | 40% | 6.3% | 6.7% |

Use **dependency share (Ca ÷ ΣCa)** — it is a share of a whole, directly comparable to
`linkpair_pct`. The story lands hard on the component the section already highlights:

> `preferences` owns **0.44% of the links** but **17.6% of the code's dependencies** —
> and `gui` inverts it (34% of links, 1.6% of dependencies).

So the component Artemis silently drops is not a rounding error by *any* measure of
importance: a fifth of the documented sentences describe it **and** a sixth of the code
depends on it.

### Concrete edit points (proposal, not yet applied)

1. `figures/jabref_motivation_data.csv` — add column `dep_share` with:
   `logic 27.85, gui 1.63, model 46.59, preferences 17.57, cli 0.065, globals 6.31`.
2. `figures/jabref_motivation.py` — add the third grouped bar; legend label in plain
   English, e.g. **"share of code that depends on it"** (never "afferent coupling"/"Ca").
3. Caption addition: *"…and a sixth of the code depends on `preferences`, though it owns
   0.4% of the links."*
4. Data Availability Statement (`main.tex` ~L188): cite this package
   (`evaluation/mini-depimport/`) as the source of the dependency-share numbers.

### If Figure 1 must be touched (light-touch only)

Add a **footprint cue** to the code-package boxes in `jabref_trace_example.py` — e.g.
`org/jabref/preferences/  ·  18 files` and `org/jabref/gui/  ·  641 files`. That plants
the size inversion visually with zero metric jargon. Do **not** add coupling numbers to
Figure 1.

## Constraints to respect

- **Plain English / altitude:** say "how much code depends on the component," not
  "afferent coupling" / "Ca" / "instability."
- **Benchmark taboo:** component *names* already appear in the paper (describing the
  benchmark is allowed); the taboo is about LLM prompts/examples, which this is not.
- **`cli` caveat:** dependency share does not help `cli` (0.1%). It is a consumer/entry
  point; keep the figure's small-but-important emphasis on `preferences` (and `globals`).
