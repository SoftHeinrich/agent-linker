# Inference token drivers, 2026-09-24

## Data and method

This audit uses five ARDoCo projects and three runs per system.
The approach uses s126 with GPT-5.6-terra, including replacement MediaStore runs.
Artemis uses separate GPT-5.6-luna runs.
The runs are unpaired.

`analyze_inference_cost_drivers.py` reads `evaluation/reports/INFERENCE_COST_PERRUN.csv`.
It checks each source hash and every per-run token count.
It then checks the totals against `paper/table/inference-cost.csv`.
Each Artemis log supplies five project rows.
The script reads each distinct Artemis log once.

## Command and text result

Run from the repository root:

```text
$ python3 verification/analyze_inference_cost_drivers.py
approach: 72 calls/run; 98.562k input/run; 22.691k output/run; 1369 input/call; 315 output/call
  phase_25_coreference: 40 calls/run; 42.256k input/run; 9.505k output/run
  phase_25_coreference_judge: 8 calls/run; 14.344k input/run; 6.249k output/run
  phase_25_doc_extract: 5 calls/run; 7.713k input/run; 0.716k output/run
  phase_25_doc_judge: 5 calls/run; 1.368k input/run; 0.381k output/run
  phase_25_name_union_judge: 14 calls/run; 32.881k input/run; 5.839k output/run
Artemis: 10 calls/run; 20.034k input/run; 23.368k output/run; 2003 input/call; 2337 output/call
  prompt1: 5 calls/run; 10.488k input/run; 13.639k output/run
  prompt2: 5 calls/run; 9.546k input/run; 9.729k output/run
PASS: all 30 source rows, hashes, and four table totals match

$ git diff --cached --check -- verification/analyze_inference_cost_drivers.py verification/2026-09-24-inference-cost-drivers.md
(no output; exit 0)
```

## Reading the result

The approach makes 72 recorded calls per sweep; Artemis makes 10.
The approach uses 1,369 input tokens per call, versus Artemis's 2,003.
Thus, the input gap comes from call count, despite smaller approach prompts.
Coreference resolution and named-link judging use 75.137k input tokens together.
That is 76.2% of the approach's input total.
The approach repeats component names, rules, and sentence context across batches.
Its code sets ten sentences per coreference call and 25 candidates per judge call.
Artemis makes one extraction call and one JSON conversion call per project.

Artemis produces 2,337 output tokens per call, versus the approach's 315.
Its prompts request component mentions, then repeat them in structured output.
The recorded output totals are therefore close: 23.368k versus 22.691k.
The output difference is 0.677k tokens, or 2.9% of Artemis's total.

These figures describe the logged calls and their prompt designs.
Different models and collection dates prevent a controlled efficiency claim.
Token totals alone do not establish monetary cost.
