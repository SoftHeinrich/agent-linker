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
- Prompts use generic English and runtime catalogs, never benchmark vocabulary.
- Preserve canonical linker files byte-for-byte; promote through subclasses and
  additive runner/export wiring.

## Tools & Libraries

- Use the existing linker phase methods and data types instead of reimplementing
  validators in a separate framework.
