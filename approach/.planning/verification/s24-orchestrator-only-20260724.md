# S24 orchestrator-only verification

Date: 2026-07-24

## Scope

- Retain `s_linker24_role_orchestrator` as the sole public/runnable S24 variant.
- Keep its earlier replacement-orchestrator implementation only as a private
  base.
- Archive the superseded anchored, agentic, and dynamic S24 variants and their
  dedicated pilot material.

## Configuration

- Working directory: `approach/`
- Python: `../.venv/bin/python`
- Network/LLM calls: none

## Commands

```bash
../.venv/bin/python -m compileall -q \
  src/llm_sad_sam/linkers/experimental/_s_linker24_orchestrator_base.py \
  src/llm_sad_sam/linkers/experimental/s_linker24_role_orchestrator.py \
  pilot/test_s24_orchestrator.py

../.venv/bin/python pilot/test_s24_orchestrator.py

../.venv/bin/python run_ablation.py --list-variants | rg 's_linker24'

../.venv/bin/python - <<'PY'
from llm_sad_sam.linkers import experimental
assert hasattr(experimental, "SLinker24RoleOrchestrator")
for obsolete in (
    "SLinker24",
    "SLinker24Agentic",
    "SLinker24Dynamic",
    "SLinker24Orchestrator",
):
    assert not hasattr(experimental, obsolete), obsolete
print("PASS: sole public S24 export")
PY

if rg -n '"s_linker24(_agentic|_dynamic|_orchestrator)?"' run_ablation.py; then
  exit 1
else
  echo 'PASS: obsolete S24 registry entries absent'
fi

git diff --check
```

## Results

```text
PASS: SLinker24RoleOrchestrator contracts
s_linker24_role_orchestrator
PASS: sole public S24 export
PASS: obsolete S24 registry entries absent
```

`compileall` and `git diff --check` completed silently with exit status 0.
