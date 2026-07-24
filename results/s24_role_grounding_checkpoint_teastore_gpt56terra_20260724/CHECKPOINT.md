# Grounding checkpoint passed

The fresh TeaStore run completed without a controller-grounding exception and
preserved entity-only performance at 2 TP, 0 FP, and 0 FN.

`pass_gate` in `pilot_results.json` is false because that generic gate requires
both role-resolution and finalize actions across multiple projects. This
single-project checkpoint tests only the grounding contract; it selected
`finalize`, as expected.
