"""Invariants for `s_linker112`: the head's denotation prompt, two substitutions.

`s_linker112` copies `s_linker92._classify_denotations` in order to route its inline
prompt through a seam. A copy can drift, so this checks the copy two ways and spends no
LLM call:

  1. **against a real run** -- the head's prompt is rebuilt here from the recorded
     alias table and the deterministic scan, and its byte length must appear among the
     `prompt_length`s the recorded runs actually logged for the denotation phase. If the
     literal below were not the head's, no recorded call would match it.
  2. **against the variant** -- `s_linker112`'s prompt for the same batch must equal
     that prompt with exactly the two intended strings substituted, and nothing else.

    ../.venv/bin/python pilot/test_s112_order.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from consolidation_audit import load_projects  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import QUALIFIED_CLAUSE  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker112 import SLinker112  # noqa: E402

RESULTS = Path("../results")

#: The head's denotation prompt, verbatim from `s_linker92._classify_denotations`.
HEAD_QUOTE_LINE = "Claim must be a contiguous exact substring of the source sentence."
HEAD_SCHEMA = ('{"judgments":[{"case":1,"denotation":"participant",\n'
               '"claim":"exact source quote"}]}')


def head_prompt(sentence_table, cases) -> str:
    return f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

{QUALIFIED_CLAUSE}

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

{HEAD_QUOTE_LINE}

JSON only:
{HEAD_SCHEMA}
"""


def batches_of(linker, candidates, sentences):
    """The (sentence_table, cases) pairs the gate would build, in order."""
    sent_map = {s.number: s for s in sentences}
    for _, batch in linker._iter_batches(candidates, linker.JUDGE_BATCH):
        ids = {s.number for c in batch
               for s in linker._window(c.sentence_number, sentences)}
        yield ([{"sentence": n, "text": sent_map[n].text} for n in sorted(ids)],
               [{"case": n, "source": c.sentence_number,
                 "expression": c.matched_text} for n, c in enumerate(batch, 1)])


def recorded_lengths() -> set:
    lengths = set()
    for path in glob.glob(str(RESULTS / "*_e2e_*" / "llm_logs" / "*.jsonl")):
        for line in open(path):
            row = json.loads(line)
            if row.get("prompt_preview", "").startswith("Classify what each expression"):
                lengths.add(row.get("prompt_length"))
    return lengths


def main() -> int:
    projects = load_projects()
    lengths = recorded_lengths()
    checks = failures = matched = 0
    for base in sorted(RESULTS.glob("*/phase_states/s_linker110/openai")):
        for project, data in projects.items():
            knowledge = base / project / "knowledge.pkl"
            if not knowledge.exists():
                continue
            head = SLinker110.__new__(SLinker110)
            head.doc_knowledge = pickle.load(open(knowledge, "rb"))["doc_knowledge"]
            arm = SLinker112.__new__(SLinker112)
            arm.doc_knowledge = head.doc_knowledge
            candidates = head._scan(data["sentences"], data["components"])
            if not candidates:
                continue
            for table, cases in batches_of(head, candidates, data["sentences"]):
                expected = head_prompt(table, cases)
                actual = arm._prompt_denotation(table, cases)
                checks += 1
                matched += len(expected) in lengths
                substituted = expected.replace(
                    HEAD_QUOTE_LINE, SLinker112.QUOTE_LINE).replace(
                    HEAD_SCHEMA, SLinker112.SCHEMA)
                if actual != substituted:
                    failures += 1
                    print(f"  DRIFT {base.parts[2]}/{project}")
                if HEAD_QUOTE_LINE in actual or HEAD_SCHEMA in actual:
                    failures += 1
                    print(f"  head text survives in {base.parts[2]}/{project}")
                if actual.index('"claim"') > actual.index('"denotation"'):
                    failures += 1
                    print(f"  verdict still first in {base.parts[2]}/{project}")
    print(f"{checks - failures}/{checks} prompt invariants hold")
    print(f"{matched}/{checks} rebuilt head prompts match a recorded prompt_length "
          f"({len(lengths)} distinct lengths logged)")
    return 1 if failures or not matched else 0


if __name__ == "__main__":
    raise SystemExit(main())
