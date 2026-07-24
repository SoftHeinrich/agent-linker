# Edge pilot v1 failure

This fresh OpenAI pilot failed its acceptance gate and was not promoted to an
end-to-end run.

The entity-ownership rule removed two true positives and two false positives
on this deliberately difficult slice. More importantly, the role tool added
three true positives but fifteen false positives. The resulting final state
was 15 TP / 17 FP / 12 FN, with macro F1 0.2945, macro F2 0.3117, pooled F1
0.5085, and pooled F2 0.5357.

The role errors had two structural causes:

1. A catalog token was treated as a handle even when it occurred inside a
   dotted code or package identifier.
2. Each shortened handle was judged without the explicit identity usages
   elsewhere in the same project document.

The next design therefore gives candidate generation non-overlapping
ownership (full names and approved aliases belong to the entity phase; dotted
identifiers are not prose handles) and lets a single project-context review
resolve the remaining handles against document-derived identity anchors.
There is no project vocabulary or benchmark-specific threshold in either
mechanism.
