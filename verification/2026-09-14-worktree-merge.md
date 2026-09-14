# 2026-09-14 worktree merge verification

## Configuration

Verification was run from the merged `master` checkout using the repository virtual
environment. No network calls were made.

```sh
cd approach && ../.venv/bin/python pilot/test_s120_standalone.py \
  && ../.venv/bin/python pilot/test_s121_standalone.py \
  && ../.venv/bin/python pilot/test_s122_standalone.py
```

## Result

Exit status: `0` (24.0 seconds).

| Check | Result |
| --- | --- |
| `test_s120_standalone.py` | `135/135 checks pass` |
| `test_s121_standalone.py` | `135/135 checks pass` |
| `test_s122_standalone.py` | `99 checks, 0 failed` |

The checks cover standalone structure, rule-constant and prompt equivalence, candidate
streams across all five projects and both alias settings, plus s122's declared changes.
