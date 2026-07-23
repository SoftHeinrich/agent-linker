# S24 agentic phase tools

## Decision

Promote `SLinker24Agentic` as an experimental successor to S24. It preserves
S21's result set, then lets a bounded controller select zero, one, or two recovery
tools from runtime evidence. The controller never proposes or approves links.

```text
S21 floor
   |
runtime inventory + component catalog
   |
bounded controller
   +-- alias_phase4 ------> Phase-1 aliases -> S21 Phase-4 P1 AND P2
   +-- anchored_reference -> S24 resolver -> S24 anchored validator
   +-- no tool
   |
grounding guards + deduplicating union with the complete S21 floor
```

## Tools

### `alias_phase4`

This tool deterministically scans for exact occurrences of aliases already
approved by S21 Phase 1. Only floor-missed `(sentence, component)` pairs enter
S21's unchanged Phase-4 two-pass validator.

A plain lexical alias is not treated as an exact identifier when Phase 1 marked
the target name ambiguous. This prevents common words from acquiring canonical-
name strength while retaining technical aliases and non-ambiguous phrase aliases.

### `anchored_reference`

This is S24's existing locally anchored sibling/prefix recovery. Its resolver
must cite an exact source phrase and its dedicated validator must find an
architectural claim.

After validation, a longer Phase-1-approved alias for another component defeats
a shorter target contained within that alias. This is runtime catalog grounding,
not a project vocabulary rule.

## Controller contract

- Input: runtime component catalog, accepted floor size, and eligible candidate
  count for each tool.
- Output: zero to two names from the fixed tool registry plus a reason.
- No access to project identity, gold links, benchmark scores, or unbounded link
  generation.
- Unknown tools, duplicate overflow, and tools with zero eligible candidates
  fail closed to the complete S21 floor.

Set preservation is not described as an F1 guarantee. Additions can theoretically
lower precision, so marginal precision remains an empirical promotion gate.

## Fixed-floor evidence

The first pilot iteration found 8 TPs and 4 FPs, improving macro F1 but failing
the declared 95% marginal-precision gate. Error analysis identified the two
grounding defects above.

With the generic grounding corrections, the second pilot passed:

| Measure | Result |
| --- | ---: |
| Marginal TP / FP | 6 / 0 |
| Marginal precision | 100% |
| Fixed-floor macro F1 | 93.34% |
| Agentic macro F1 | 94.88% |
| Delta | +1.54pp |
| Distinct project plans | 4 |

The promoted production class was then replayed independently over all five
fixed floors:

| Project | Selected tools | Marginal TP / FP | Floor F1 | Final F1 |
| --- | --- | ---: | ---: | ---: |
| mediastore | alias | 1 / 0 | 94.92% | 96.67% |
| teastore | alias + anchor | 1 / 0 | 96.15% | 98.11% |
| teammates | alias + anchor | 1 / 0 | 89.91% | 90.91% |
| bigbluebutton | anchor | 2 / 0 | 85.71% | 87.72% |
| jabref | none | 0 / 0 | 100.00% | 100.00% |
| **Macro** | 4 distinct plans | **5 / 0** | **93.34%** | **94.68%** |

The difference between 6 and 5 recovered links across the two passing runs is
resolver variance. Both runs maintained 100% marginal precision and beat the
same fixed S21 floor.

## End-to-end smoke

A normal runner invocation using the Codex backend completed on Mediastore:

- same-run internal S21 floor: 30 TP, 1 FP, 1 FN, F1 96.77%;
- controller plan: `alias_phase4`;
- marginal addition: 1 TP, 0 FP;
- final: 31 TP, 1 FP, 0 FN, F1 98.41%;
- same-run delta: +1.64pp F1.

This smoke validates live state transfer, registry construction, all S21 phases,
controller execution, tool validation, CSV export, and scoring. It is not a
GPT-5.4 replication and should not replace the existing OpenAI paper results.

## Commands

```bash
cd approach
../.venv/bin/python pilot/test_s24_agentic_tools.py
../.venv/bin/python pilot/s24_agentic_tools_pilot.py \
  --results-dir ../results/s24_agentic_promoted_fixed_floor_20260724
LLM_BACKEND=codex ../.venv/bin/python run_ablation.py \
  --variants s_linker24_agentic --datasets mediastore \
  --results-dir ../results/s24_agentic_codex_e2e_mediastore_20260724
```
