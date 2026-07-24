# Checkpoint failure

The five-project edge checkpoint failed by one false positive:

- Entity-only: 16 TP, 8 FP, 12 FN; pooled F2 58.82.
- Final: 16 TP, 9 FP, 12 FN; pooled F2 58.39.
- Route diversity persisted.

The false positive was `tests -> Test Driver`, created by synthesized plural
matching. The fresh participation judge also reversed all BBB decisions
relative to the prior fresh run, despite the same structural candidates.

The redesign removes plural synthesis and the unstable semantic judge.
Space-separated compound names expose only their terminal handle; hyphenated
names expose identifier segments; only exact standalone occurrences match.
