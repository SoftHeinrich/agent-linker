# Paper numeric-claim coverage check — 2026-09-21

## Purpose and configuration

`scripts/check-paper-numeric-claims.py` audits every active numeric-bearing
source line in the main paper.  It scans the abstract and the ten section files
rendered with the current `\showappendixfalse` configuration.  The scanner
recognizes digit forms, signs, decimals, percentages, percentage-point suffixes,
multipliers, thousands separators, and English cardinal and ordinal words,
including magnitude words through billions.  It removes TeX comments and structural arguments such as
citation keys, labels, references, and input paths; captions and accessibility
descriptions remain in scope.

The policy in `verification/paper-numeric-claims-policy.json` provides two
independent gates:

1. A SHA-256 digest covers the ordered set of normalized numeric-bearing lines
   and their extracted tokens.  Any addition, deletion, reordering, wording
   change, or numeric-token change requires review and a new digest.
2. Non-overlapping scopes classify every line as measured, reported, setup,
   method, example, or notation.  Every non-notation scope names evidence that
   must exist and be tracked in the applicable Git repository.  Uncovered and
   multiply covered lines fail.

Generated table cells are not duplicated in this line ledger.  They remain
covered cell by cell by `evaluation/mini-src/check.py` and
`verification/2026-09-21-rq3-real-metrics-check.py`; the prose and inclusion
sites are covered here.

The repository-wide `scripts/verify.sh` invokes this audit after its existing
evaluation checks, so the numeric inventory gate is part of the standard
verification entrypoint.

Current reviewed inventory digest:

```text
f035029e85319fbaae8338758a65c175829e0054625394ad3d801c38a2307dd7
```

To inspect every covered line, its tokens, classification scope, and normalized
text:

```bash
python3 scripts/check-paper-numeric-claims.py --report
```

## Verification

Command:

```bash
python3 scripts/check-paper-numeric-claims.py --self-test
```

Result:

```text
PASS self-test: TeX comments, numeric tokens, citation stripping, and change detection
PASS paper numeric-claim audit: 206 active statements reviewed
PASS ledger classifications: example=16, measured=100, method=47, notation=12, reported=1, setup=30
PASS every non-notation entry names existing tracked evidence
```

Syntax-check command:

```bash
tmp_cache_dir="$(mktemp -d)"
PYTHONPYCACHEPREFIX="$tmp_cache_dir" python3 -m py_compile scripts/check-paper-numeric-claims.py
status=$?
rm -r "$tmp_cache_dir"
exit "$status"
```

Result: exit status 0, with no output.

An initial syntax check without `PYTHONPYCACHEPREFIX` could not create a bytecode
file in the pre-existing `scripts/__pycache__` directory:

```text
[Errno 13] Permission denied: 'scripts/__pycache__/check-paper-numeric-claims.cpython-313.pyc.125798086411968'
```

The isolated temporary bytecode cache above removes that environment-specific
blocker and completes successfully.

Standard verification entrypoint:

```bash
./scripts/verify.sh
```

Result (exit status 0; the existing metric checks printed all five projects for
both tasks):

```text
OK    arm-default   every generator reports arm 's126' (7/7 found)
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
PASS self-test: TeX comments, numeric tokens, citation stripping, and change detection
PASS paper numeric-claim audit: 206 active statements reviewed
PASS ledger classifications: example=16, measured=100, method=47, notation=12, reported=1, setup=30
PASS every non-notation entry names existing tracked evidence
```
