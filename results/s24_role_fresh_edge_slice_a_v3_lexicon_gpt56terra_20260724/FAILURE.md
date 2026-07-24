# Checkpoint failure

The full-document learned-lexicon design failed.

- Entity-only: 8 TP, 5 FP, 9 FN; pooled F2 49.38.
- Final: 8 TP, 8 FP, 9 FN; pooled F2 47.62.
- BBB received no additions.
- JabRef received three false-positive mappings from the learned expressions
  `center`, `core`, and `outer layers`.

The design conflated descriptive architectural concepts with component
references. It also missed the shortened forms that mattered in BBB. The next
design derives candidate handles structurally from unique parts of compound
component names and delegates only final acceptance to the inherited validator.
