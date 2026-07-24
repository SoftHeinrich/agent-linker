# Identifier identity v2 failure

This fresh pilot was not promoted. Fuzzy shared-token candidate generation
added 1 TP and 2 FP. Both FPs were ordinary hyphenated phrases whose tokens
only partially overlapped a component name, lowering macro and pooled F1.

The accepted TP used an alternate separator spelling whose complete token
sequence matched the catalog component. The next iteration therefore removes
shared-token generation entirely and owns only standalone identifiers with the
same full token sequence as a runtime catalog name.
