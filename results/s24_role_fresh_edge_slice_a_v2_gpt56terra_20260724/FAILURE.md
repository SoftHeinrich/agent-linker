# Checkpoint failure

This fully fresh BBB/JabRef edge checkpoint completed but did not improve the
entity-only state.

- Entity-only and final both scored 7 TP, 5 FP, and 10 FN.
- The controller selected relation/role resolution for both projects.
- The local resolver proposed only one BBB mapping, which its next judge
  rejected; it added no links.

The trace showed a causal design problem: a local proposer plus overlapping
mapping judge, literal-referent skeptic, and inherited validator did not first
learn how this particular project uses generic vocabulary. The replacement
design learns a project role lexicon from the full document, applies it
deterministically, and leaves acceptance to the existing evidence validator.
