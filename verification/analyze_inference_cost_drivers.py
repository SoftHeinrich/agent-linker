#!/usr/bin/env python3
"""Break down the inference-usage runs by recorded call type."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/reports/INFERENCE_COST_PERRUN.csv"
SUMMARY = ROOT / "paper/table/inference-cost.csv"
CALLS = defaultdict(lambda: [0, 0, 0])
PROJECTS = defaultdict(lambda: [0, 0, 0])


def add(bucket, key, input_tokens, output_tokens):
    bucket[key][0] += 1
    bucket[key][1] += input_tokens
    bucket[key][2] += output_tokens


def check_row(row, counts):
    assert counts[1] == round(float(row["input_k"]) * 1000), row
    assert counts[2] == round(float(row["output_k"]) * 1000), row


with SOURCE.open(newline="") as source:
    rows = list(csv.DictReader(source))
assert len(rows) == 30
seen_logs = set()

for row in rows:
    path = ROOT / row["source"]
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == row["sha256"], path
    key = (row["system"], row["run"], row["project"])

    if row["system"] == "approach":
        assert row["model"] == "gpt-5.6-terra"
        for call in json.loads(content):
            assert call["success"] and call["model"] == row["model"]
            usage = call["token_usage"]
            add(CALLS, ("approach", call["phase"]),
                usage["prompt_tokens"], usage["completion_tokens"])
            add(PROJECTS, key, usage["prompt_tokens"], usage["completion_tokens"])
        check_row(row, PROJECTS[key])
        continue

    assert row["system"] == "Artemis" and row["model"] == "gpt-5.6-luna"
    if path in seen_logs:
        continue
    seen_logs.add(path)
    project = None
    for line in content.decode().splitlines():
        start = re.search(r"ArDoCo - Starting (\w+)$", line)
        if start:
            project = start[1]
        usage = re.search(
            r"(prompt[12]|repair) tokens: input=(\d+), output=(\d+), total=(\d+)",
            line,
        )
        if usage:
            assert project and int(usage[2]) + int(usage[3]) == int(usage[4])
            add(CALLS, ("Artemis", usage[1]), int(usage[2]), int(usage[3]))
            add(PROJECTS, ("Artemis", row["run"], project),
                int(usage[2]), int(usage[3]))

for row in rows:
    check_row(row, PROJECTS[row["system"], row["run"], row["project"]])
assert len(seen_logs) == 3

with SUMMARY.open(newline="") as summary:
    totals = {row["system"]: row for row in csv.DictReader(summary)}

for system in ("approach", "Artemis"):
    values = [value for (name, _), value in CALLS.items() if name == system]
    count, input_tokens, output_tokens = (sum(value[i] for value in values)
                                          for i in range(3))
    assert round(input_tokens / 3000, 6) == float(totals[system]["Total_input_k"])
    assert round(output_tokens / 3000, 6) == float(totals[system]["Total_output_k"])
    print(f"{system}: {count / 3:.0f} calls/run; "
          f"{input_tokens / 3000:.3f}k input/run; "
          f"{output_tokens / 3000:.3f}k output/run; "
          f"{input_tokens / count:.0f} input/call; "
          f"{output_tokens / count:.0f} output/call")
    for (name, stage), value in sorted(CALLS.items()):
        if name == system:
            print(f"  {stage}: {value[0] / 3:.0f} calls/run; "
                  f"{value[1] / 3000:.3f}k input/run; "
                  f"{value[2] / 3000:.3f}k output/run")

print("PASS: all 30 source rows, hashes, and four table totals match")
