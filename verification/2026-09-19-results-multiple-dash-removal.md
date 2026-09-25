# Results multiple dash removal verification

Date: 2026-09-19

Command:

```bash
if rg -n -- '--+' paper/sections/results.tex; then
  exit 1
fi
git -C paper diff --check
```

Result: pass. `paper/sections/results.tex` contains no double or triple ASCII
dashes, and `git diff --check` reported no whitespace errors.
