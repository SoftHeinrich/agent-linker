# Edge pilot v2 invalidation

This run was not promoted to end-to-end evaluation.

The context role review reduced the role tool's false positives, but inspection
of the entity trace revealed a protocol flaw: the extractor received a sparse
sequence of original sentence IDs and occasionally answered with local,
contiguous IDs. Those answers were then interpreted as original document IDs.
Consequently, v1 and v2 do not provide valid comparative evidence for the
entity-ownership mechanism.

The corrected pilot runs extraction and validation on the intact document.
Gold still only predeclares the prior error sentences to score after inference;
it is never included in a prompt or runtime decision.
