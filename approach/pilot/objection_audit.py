"""What the strict gate's `objection` field costs, read off the head's own runs.

The judge round's open design question is whether all three judges can reply in one
schema. Two of the three already do; the third differs in three ways, and the uniform
superset carries the strict gate's `objection` at every gate. That field is the only
part of the unification that costs tokens, and the head has already written six
five-project runs of it, so the price is knowable before any arm is built.

    ../.venv/bin/python pilot/objection_audit.py

No calls. Reads `linker_*.pkl` out of the recorded runs `chooser_audit.runs_of` finds.
"""
from __future__ import annotations

import pickle
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from chooser_audit import runs_of                      # noqa: E402
from consolidation_audit import load_projects, model_of  # noqa: E402

#: What the head's judging calls spend today, per five-project run, measured from the
#: recorded request logs (`llm_logs/*.jsonl`, `token_usage.completion_tokens`).
VALIDATE_COMPLETION_TOKENS_A_RUN = 28_600

#: Characters per token, the ratio the recorded replies show for this field's prose.
CHARS_PER_TOKEN = 4.0

STAGES = (("full_name", "linker_full_name.pkl"),
          ("partial_name", "linker_partial_name.pkl"),
          ("coref", "linker_coreference.pkl"))


def rows_by_stage():
    """{model: {stage: [decision rows]}} and the number of runs per model."""
    projects = load_projects()
    out: dict = {}
    runs: dict = {}
    for base in runs_of("s_linker110"):
        model = model_of(base)
        runs[model] = runs.get(model, 0) + 1
        for project in projects:
            for stage, name in STAGES:
                feedback = pickle.load(open(base / project / name, "rb")).get(
                    "feedback", {})
                out.setdefault(model, {}).setdefault(stage, []).extend(
                    feedback.get("judge_decisions") or [])
    return out, runs


def main() -> int:
    by_model, runs = rows_by_stage()
    print("The strict gate's `objection`, and what a uniform schema would add\n")
    for model, stages in sorted(by_model.items()):
        n = runs[model]
        print(f"--- {model} ({n} recorded runs)")
        print(f"  {'stage':<14}{'rows/run':>10}{'approved':>10}{'claim ch':>10}"
              f"{'objection ch':>14}{'appr/rej':>14}")
        for stage, _ in STAGES:
            rows = stages.get(stage) or []
            if not rows:
                continue
            approved = [r for r in rows if r.get("approved")]
            rejected = [r for r in rows if not r.get("approved")]
            claim = st.mean(len(r.get("claim", "")) for r in rows)
            grounds = [r for r in rows if "objection" in r]
            if grounds:
                obj = st.mean(len(r["objection"]) for r in grounds)
                split = (f"{st.mean(len(r.get('objection', '')) for r in approved):.0f}"
                         f" / "
                         f"{st.mean(len(r.get('objection', '')) for r in rejected):.0f}")
            else:
                obj, split = 0.0, "not asked"
            print(f"  {stage:<14}{len(rows) / n:>10.1f}{len(approved) / n:>10.1f}"
                  f"{claim:>10.1f}{obj:>14.1f}{split:>14}")

        lenient = stages.get("full_name") or []
        sortal = stages.get("partial_name") or []
        strict = stages.get("coref") or []
        per_case = st.mean(len(r.get("objection", "")) for r in strict) / CHARS_PER_TOKEN
        added_cases = (len(lenient) + len(sortal)) / n
        added = per_case * added_cases
        print(f"  uniform schema: {added_cases:.1f} more cases a run carry the field "
              f"at ~{per_case:.0f} tokens each\n"
              f"                  = +{added:.0f} completion tokens a run against "
              f"{VALIDATE_COMPLETION_TOKENS_A_RUN} spent judging "
              f"(+{100 * added / VALIDATE_COMPLETION_TOKENS_A_RUN:.0f}%)\n")
    print("So the unification is not a token saving. It has to pay in verdicts:\n"
          "  `s_linker116` prices the field at the lenient gate, `s_linker118` at the\n"
          "  sortal one, `s_linker119` the whole schema, `s_linker117` the order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
