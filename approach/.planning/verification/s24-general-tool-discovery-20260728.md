# S24 general tool discovery verification

Date: 2026-07-28

## Scope

- remove controller-side lexical applicability policy;
- replace participant handles and enumerated semantic classes with general
  runtime-catalog overlap plus target-blind semantic review;
- preserve canonical S21 and the S24 promotion floor.

## Contract and structural checks

```bash
../.venv/bin/python pilot/test_s24_orchestrator.py
../.venv/bin/python pilot/test_s24_simple_orchestrator.py
../.venv/bin/python pilot/test_s24_general_discovery.py
../.venv/bin/python pilot/test_s24_discourse_scope.py
../.venv/bin/python pilot/test_s24_lexical_entity.py
../.venv/bin/python -m py_compile \
  src/llm_sad_sam/linkers/experimental/s_linker24_role_orchestrator.py \
  pilot/s24_general_discovery_pilot.py \
  pilot/s24_simple_orchestrator_pilot.py
git diff --exit-code ee926de -- \
  src/llm_sad_sam/linkers/experimental/s_linker21.py
```

Output:

```text
PASS: SLinker24RoleOrchestrator contracts
PASS: S24 simple-orchestrator contracts
PASS: S24 general-discovery contracts
PASS: S24 discourse-scope contracts
PASS: S24 lexical entity contracts
```

Compilation and the canonical-S21 byte-stability check exited 0.

The removed-policy scan also exited cleanly for `_REFERENCE`,
`catalog_role_handles`, `apply_role_handles`, `find_handle`, enumerated
`hardware`/`technology` prompt classes, controller `tool_evidence`, and
`handle_decisions`.

## Benchmark

Configuration:

```text
backend=openai
model=gpt-5.6-terra
reasoning_effort=none
credential mapping: OPENAI_API_KEY="$OAI_KEY" (process-local)
datasets=mediastore,teammates,teastore,bigbluebutton,jabref
variants=s_linker21,s_linker24_role_orchestrator
```

Fresh paired result:

```text
S21: TP=170 FP=15 FN=25
     macro F1/F2=92.6720/91.6292
     pooled F1/F2=89.4737/88.0829
S24: TP=182 FP=8 FN=13
     macro F1/F2=96.0731/95.3953
     pooled F1/F2=94.5455/93.8144
S24 participant source: 14 TP / 0 FP
```

The raw ablation JSON and link CSVs are stored in
`results/s24_general_discovery_e2e_v1_20260728/`.
