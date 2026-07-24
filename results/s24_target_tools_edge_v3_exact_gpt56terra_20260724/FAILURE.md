# Combined target-tools v3 failure

Exact catalog-equivalent identifier recovery added 3 TP / 0 FP. Approved-alias
coverage independently added 0 TP / 4 FP after misreading a data field named
`UI name` as the WebUI component.

The combined result therefore regressed both F1 aggregates despite improving
F2. Alias coverage is rejected as unstable across fresh runs. The next
checkpoint isolates only exact catalog-equivalent standalone identifiers.
