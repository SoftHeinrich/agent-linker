# Edge pilot v3 scoring invalidation

The inference protocol used the intact documents, but the scorer compared
full-document predictions with gold restricted to the predeclared edge
sentences. Predictions outside the edge set were consequently counted as false
positives. The reported aggregate metrics are invalid and this run was not
used as promotion evidence.

The corrected scorer restricts predictions to the same predeclared sentence
set before computing metrics. The model calls are rerun fresh because this
pilot did not persist enough intermediate state to reconstruct every filtered
link safely.
