"""Is `s_linker126` still a standalone file, and did the flatten drift from its ancestors?

`s_linker126` used to subclass `SLinker125`, which subclasses `SLinker123`, which
subclasses `SLinker122` (itself standalone). The reported-variant policy is one
self-contained file with no linker base class (`s_linker120.py`'s precedent). This
does not re-litigate `pilot/test_s126.py`'s candidate-set/decision-logging checks
(T5/T6-equivalent, already passing against the flattened file) -- it covers what that
file does not:

  T1  structure     the MRO is `(SLinker126, object)`, and no `s_linker1NN` sibling
                     module is imported at module scope.
  T2  the untouched  rule constants that neither the evidence-field merge (s123) nor
      constants     the shortlist removal (s125) nor this file's own two deltas ever
                     touch stay byte-identical to `s_linker122`'s -- the knowledge-stage
                     and coreference-judging rules, which this round did not reach.

No LLM calls.

    ../.venv/bin/python pilot/test_s126_standalone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.linkers.experimental import s_linker122 as ANCESTOR  # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker126 as FLAT       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker126 import SLinker126    # noqa: E402

#: Rule text this round's three deltas (evidence-field merge, shortlist removal,
#: greedy ownership + antecedent-form gate) never touch -- the knowledge stage and
#: the coreference judge's own rubric, both untouched since `s_linker122`.
UNTOUCHED_CONSTANTS = (
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "ALIAS_EXCLUSION_RULES",
    "COREF_VALIDATION_FOCUS",
    "COREF_RULES",
)

PASSED = FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL: {label} {detail}")


def main():
    # T1: structure.
    check("T1 MRO is (SLinker126, object)",
          SLinker126.__mro__ == (SLinker126, object))

    src = Path(FLAT.__file__).read_text()
    sibling_imports = [
        line for line in src.splitlines()
        if ("from llm_sad_sam.linkers.experimental.s_linker1" in line
            or "from llm_sad_sam.linkers.experimental import s_linker1" in line)
    ]
    check("T1 no sibling s_linkerNNN import", not sibling_imports,
          detail=str(sibling_imports))

    # T2: untouched constants stay byte-identical to the standalone ancestor's.
    for name in UNTOUCHED_CONSTANTS:
        ancestor_value = getattr(ANCESTOR, name, None)
        flat_value = getattr(FLAT, name, None)
        check(f"T2 {name} byte-identical to s_linker122's",
              ancestor_value is not None and ancestor_value == flat_value,
              detail=f"ancestor={ancestor_value!r} flat={flat_value!r}")

    print(f"{PASSED}/{PASSED + FAILED} checks passed")
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
