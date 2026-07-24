# Checkpoint failure

The catalog-handle candidate mechanism found all nine residual BBB gold links
and one false candidate, but the reused two-pass entity validator rejected
every candidate. Its referential-specificity pass asks whether a generic term
is itself a component name, which contradicts the role tool's premise.

The participation pass accepted seven gold candidates and rejected the false
`web -> BBB web` candidate. The redesign therefore treats catalog uniqueness
as the identity proof and uses one participation validation stage. Capability
discovery also removes the role tool when the project profile has no applicable
handle occurrence.
