# Checkpoint failure

The first fully fresh edge-slice checkpoint stopped before scoring.

The semantic section planner did not return an exact, ordered partition for
the sparse sentence identifiers in the edge slice. The strict parser raised
instead of allowing a downstream tool to work with incomplete evidence.

The redesign makes section planning advisory: a valid semantic partition is
used, while an invalid partition deterministically falls back to one complete
document section. This preserves the completeness invariant without a retry
count, score threshold, project vocabulary, or benchmark-specific branch.
