"""S-Linker92b — the same scan, with the qualified-identifier spans not proposed.

`s_linker92a` hands the lenient full-name gate 53.3 more pairs a run than the LLM
extractor did. **20.8 of them are a component's name written inside a longer dotted
identifier** — the case `QUALIFIED_CLAUSE` already tells that judge to reject. This
variant does not propose them.

The design law says a *weighing* folds into the prompt and a *fact* does not, and
`skip_qualified` was folded, at TP −0.4 (p = 0.44). That price was paid on the
extractor's population: an LLM asked for named references reports few qualified
spans on its own, so the folded clause was speaking about roughly two candidates a
run. Put a scan in front of the same clause and its population grows tenfold. **A
clause is not independently priceable** — s78's result, and this is the same result
from the proposer's side: what changed is not the clause but what it is asked about.

Read off the same 30 recorded runs (`pilot/regex_extract_audit.py`), the skip costs
no gold at all and takes a quarter of the added pairs back out:

    proposer         pairs   gold   prec    +pairs   +gold   -gold
    LLM extraction   175.3  150.1  0.856       -       -       -
    92a (no skip)    221.9  158.3  0.713     53.3   +10.6    -2.4
    92b (skip)       196.9  158.3  0.804     32.9   +10.6    -2.4

Identical gold, 25.0 fewer pairs. In the replayed-gate bracket the difference is
entirely in what the gate would have to reject:

    arm   policy      TP     FP   macro F1   macro F2
    ctrl     -     180.6   36.4      91.04      92.93
    92a   reject  178.5   33.6      91.00      92.50
    92a   approve 187.8   75.7      87.16      92.70
    92b   reject  178.5   32.3      91.19      92.58
    92b   approve 187.8   54.4      89.29      93.84

Same TP either way — the skip removes 0 gold — and **21.3 fewer false positives at
the approve end**, which is the end the bracket is wide at. The clause stays in the
prompt: it is still the right thing to tell a judge about a case that reaches it,
and this only stops manufacturing that case 20.8 times a run.

Everything else is `s_linker92a`'s, which is `s_linker92`'s.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a


class SLinker92b(SLinker92a):
    """The scan does not propose a name it only finds inside a dotted identifier."""

    _VARIANT_NAME = "s_linker92b"

    #: The one line that differs from `s_linker92a`. `_in_dotted_path` is the head's
    #: own predicate, unchanged and not re-implemented here.
    SKIP_QUALIFIED = True
