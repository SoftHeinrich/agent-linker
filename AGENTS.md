# Project working agreement

For every user-requested implementation, run a proportionate verification or
benchmark evaluation before reporting completion. Preserve the command,
configuration, and text results in Git; if a run cannot complete, track the
failure evidence and state the blocker explicitly.

## Authored wording gate

The wording standard recorded in `approach/CLAUDE.md` is also binding for
agents. It applies to prompts, rubrics, deterministic gates, method
descriptions, and other authored rules.

- Every clause must stand on one of three grounds: a general rule or logical
  distinction that holds for arbitrary text; general software-engineering
  practice; or prior work already measured in this project or established in
  the literature.
- Do not encode benchmark-derived vocabulary, surface forms, syntax, or
  corpus-specific shapes as general rules. Runtime catalog names are data; the
  authored rule must remain generic English.
- Keep facts about a case in the evidence or data and put judgment, weighting,
  and acceptance criteria in the prompt or rubric. Do not replace a fact with
  an instruction to infer it.
- Treat a wording change as a semantic change unless a fixed-input audit shows
  otherwise. Do not paraphrase a load-bearing criterion merely to make it sound
  more natural without recording the validation.

## Paper-writing restrictions

Paper prose follows the same gate, with an additional evidence requirement:

- Separate measured results, interpretations, design rationale, hypotheses,
  and limitations. Use cautious language when the evidence is exploratory or
  inconclusive.
- Every quantitative or comparative claim must be traceable to a committed
  table, figure, report, or reproducible command. State the relevant arm,
  datasets, model, repetitions, and metric; do not compare results from
  different invocation sets as if they were paired.
- Do not write “improves,” “superior,” “causes,” “robust,” or “general” unless
  the recorded evaluation supports that strength of claim. A neutral result is
  not evidence of equivalence, and an N=1 result is not a replicated finding.
- Keep benchmark-specific names, examples, and surface forms scoped to an
  explicitly identified observation, result, or threat. Do not turn them into
  universal method rules or claims about software in general.
- Preserve inconvenient results and validity threats. If a claim cannot be
  supported from tracked evidence, narrow the wording or mark it as an open
  hypothesis rather than presenting it as established fact.
