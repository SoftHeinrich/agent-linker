# Agent guidance wording-policy verification

Date: 2026-09-14

The documentation change adds the GATE-07 wording rule to the agent-facing
guidance and adds evidence-bounded paper-writing restrictions.

## Guidance-content check

Command:

```bash
python3 - <<'PY'
from pathlib import Path

checks = {
    'root authored wording gate': (
        Path('AGENTS.md'),
        ('Authored wording gate', 'general rule or logical',
         'benchmark-derived vocabulary', 'Paper-writing restrictions'),
    ),
    'approach GATE-07 mirror': (
        Path('approach/AGENTS.md'),
        ('GATE-07', 'benchmark-shaped rules', 'fixed-input validation'),
    ),
    'evaluation paper gate': (
        Path('evaluation/AGENTS.md'),
        ('Paper-writing gate', 'evidence-bounded', 'N=1',
         'universal method rules'),
    ),
}

for label, (path, needles) in checks.items():
    text = path.read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f'FAIL: {label}: missing {missing}')
    print(f'PASS: {label}')
print(f'PASS: {len(checks)} guidance checks')
PY
```

Result:

```text
PASS: root authored wording gate
PASS: approach GATE-07 mirror
PASS: evaluation paper gate
PASS: 3 guidance checks
```

## Patch hygiene check

Command:

```bash
git diff --check
```

Result:

```text
(no whitespace errors)
```

Status: PASS.
