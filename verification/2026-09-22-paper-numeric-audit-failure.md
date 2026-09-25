# Paper numeric-claim audit failure

Date: 2026-09-22. The repository verification entrypoint was run against the
current checkout before committing the staged work.

## Commands and results

```text
$ ./scripts/verify.sh
```

The benchmark input check and both frozen metric panels passed. The paper
numeric-claim self-test passed, but its inventory audit failed:

```text
FAIL paper numeric-claim audit: 21 error(s); 201 active statements
- numeric-statement inventory changed: expected f035029e85319fbaae8338758a65c175829e0054625394ad3d801c38a2307dd7, found 7cececf58e9aba8f95a8ad5cf1a12ac6fddf4b3df2d3ffaf39436efcb18bbc20
- 20 changed paper lines contain numeric statements not classified by the current policy
```

The full list of affected lines is available by running
`python3 scripts/check-paper-numeric-claims.py --report`. The current checker
does not support accepting the changed inventory without reviewing and
classifying those statements.

The RQ3 engine-to-table audit passed independently:

```text
$ python3 verification/2026-09-21-rq3-real-metrics-check.py
RESULT: PASS
```

`git diff --cached --check` reports trailing whitespace in generated benchmark
logs, CSVs, and study reports. These are preserved output files; no whitespace
normalization was applied to alter the captured data. The paper numeric-claim
audit remains an open verification failure; its digest and classifications were
not changed to suppress it.
