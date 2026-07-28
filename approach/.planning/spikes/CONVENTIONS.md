# Spike Conventions

Patterns established by the trace-linking spikes.

## Stack

- Python and the repository's existing virtual environment.
- Small CLI harnesses for benchmark facts; no UI is needed for score experiments.

## Structure

- Spike definitions and investigation trails live under `.planning/spikes/`.
- Raw text evidence and link CSVs live in dated root `results/` directories.

## Patterns

- Compare augmentation on an identical saved floor whenever possible.
- Score marginal additions independently before interpreting a fresh full run.
- Controllers select bounded tools; they do not emit final domain decisions.
- Controllers see compact evidence signals and prior outcome counts, while
  domain tools see only bounded evidence needed for their own decision.
- Let controllers choose evidence-bearing tool order over multiple turns, but
  complete structurally when the finite capability set is exhausted.
- Let each tool discover its own applicability; do not duplicate tool-specific
  lexical tests in the controller.
- When a proposed target biases semantic classification, classify the source
  expression without the target first, then perform grounded identity review.
- Prompts use generic English and runtime catalogs, never benchmark vocabulary.
- Preserve canonical linker files byte-for-byte; promote through subclasses and
  additive runner/export wiring.

## Tools & Libraries

- Use the existing linker phase methods and data types instead of reimplementing
  validators in a separate framework.
