"""S-Linker112 — the sortal gate quotes before it commits, like the other two.

Three judges stand in this pipeline and two of them demand a committed quote *before*
the verdict: the lenient gate's reply is `claim` then `approve`, the strict gate's is
`claim` then `objection` then `approve`. The partial-name gate is the exception — its
reply is `denotation` then `claim`, so the verdict is written first and the evidence
after it.

Nothing chose that. `s_linker48` measured the separation that all three rest on and the
head still records it in this gate's own parser: *"demanding a committed quote is worth
35.2 TP, verifying it is worth nothing."* The demand is what pays; here the demand comes
after the answer it was supposed to constrain, which is the same defect `s_linker92f`
repairs at the lenient gate — a JSON-only reply commits in field order.

Two substitutions and no new rule. The line that stated the quote's shape now also
states its place, and the schema lists `claim` ahead of `denotation`. No clause is
added, removed or reworded, the target stays withheld, `QUALIFIED_CLAUSE` is untouched,
and the sentence table, the case list, the batch size and the parser are the head's.
`pilot/test_s112_order.py` holds the head's prompt against this one and asserts the
difference is exactly those two strings.

GATE-07: the ground is `s_linker48`'s own result and the convention the branch's other
two judges already follow. Nothing here names a surface form, a component or a term of
any document.

**REFUSED at level 2: the sign flips between models** (`results/judge_round/README.md`).
Three samples a model over the sortal gate's fixed candidates: terra gold 21.3 -> 23.0 at
spurious 8.0 -> 8.3, luna gold 21.3 -> 18.0 at spurious 9.7 -> 6.3. QUALITY-CHANGING
against on luna at the n=3 floor. The gate's population will not carry the question
either -- only one of the five projects contributes gold to it on either model, so every
number rests on 48 candidates measured three times.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker92 import QUALIFIED_CLAUSE
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110


class SLinker112(SLinker110):
    """The head, with the partial-name gate's evidence demanded before its verdict."""

    _VARIANT_NAME = "s_linker112"

    #: Where the quote is demanded. The head states the quote's shape after the case
    #: list and lets the schema order decide when it is written; this states both.
    QUOTE_LINE = ("For each case, first quote the exact words of the source sentence "
                  "that the expression\nis used in -- a contiguous exact substring -- "
                  "and then classify what it denotes there.")

    #: The reply, evidence first. The head's fields, in the other two judges' order.
    SCHEMA = ('{"judgments":[{"case":1,"claim":"exact source quote",\n'
              '"denotation":"participant"}]}')

    @classmethod
    def _prompt_denotation(cls, sentence_table, cases) -> str:
        """The head's denotation prompt, with the quote demanded before the verdict."""
        return f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

{QUALIFIED_CLAUSE}

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

{cls.QUOTE_LINE}

JSON only:
{cls.SCHEMA}
"""

    def _classify_denotations(self, candidates, sentences):
        """The head's pass, with its inline prompt routed through the seam above.

        Copied from `s_linker92` rather than wrapped: the prompt is built inside the
        loop there, so there is no other place to stand. Everything but the two
        `_prompt_denotation` substitutions is that method verbatim, and the parser
        below is unchanged -- field *order* is what moves, not the field set.
        """
        sent_map = {s.number: s for s in sentences}
        decisions = {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in self._window(candidate.sentence_number, sentences)
            }
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(evidence_ids)
            ]
            cases = [
                {"case": n, "source": c.sentence_number, "expression": c.matched_text}
                for n, c in enumerate(batch, 1)
            ]
            data = self._ask(
                self._prompt_denotation(sentence_table, cases),
                phase="phase_25_partial_denotation",
                require_present="judgments", label="Denotation", timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                valid = denotation in {"participant", "associated"} and bool(claim)
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                    "path": "denotation",
                    "stage": "partial_name",
                }
        participants = [
            c for c in candidates
            if decisions.get((c.sentence_number, c.component_id), {}).get(
                "denotation") == "participant"
            and decisions[(c.sentence_number, c.component_id)]["evidence_valid"]
        ]
        return participants, decisions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker112 (sortal gate: the quote is demanded before the verdict)")
