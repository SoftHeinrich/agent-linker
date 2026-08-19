"""`s_linker76` is `s_linker75` with one number changed — asserted.

The claim this file pins is narrow on purpose: the variant states **two** batch sizes
where s75 states three, and nothing else about it differs. If that holds, `s_linker45`'s
six-run parity result on the s25 base is the only evidence the change needs, and the E2E
paired against s75 is a confirmation rather than a discovery.

    T1  exactly one class attribute differs, and it is `COREFERENCE_BATCH`
    T2  the new value is not a new number: it is `JUDGE_BATCH`
    T3  every authored constant is byte-identical to s75's
    T4  every method body is byte-identical to s75's

    ../.venv/bin/python pilot/test_s76_twobatches.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker75 as L75              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker76 as L76              # noqa: E402

PASS, FAIL = [], []

AUTHORED = [c for c in dir(L75) if c.isupper() and isinstance(getattr(L75, c), str)]


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def t1():
    print("\nT1  one changed bound")
    attrs = {k: v for k, v in vars(L75.SLinker75).items()
             if not callable(getattr(v, "__func__", v)) and not k.startswith("__")}
    diff = sorted(k for k, v in attrs.items()
                  if k != "_VARIANT_NAME" and getattr(L76.SLinker76, k) != v)
    check("exactly one class attribute differs", diff == ["COREFERENCE_BATCH"],
          ", ".join(diff) or "none")
    check("s75 states three batch sizes",
          len({L75.SLinker75.EXTRACTION_BATCH, L75.SLinker75.JUDGE_BATCH,
               L75.SLinker75.COREFERENCE_BATCH}) == 3)
    check("s76 states two",
          len({L76.SLinker76.EXTRACTION_BATCH, L76.SLinker76.JUDGE_BATCH,
               L76.SLinker76.COREFERENCE_BATCH}) == 2)


def t2():
    print("\nT2  the value is unified, not searched")
    check("COREFERENCE_BATCH == JUDGE_BATCH",
          L76.SLinker76.COREFERENCE_BATCH == L76.SLinker76.JUDGE_BATCH,
          f"{L76.SLinker76.COREFERENCE_BATCH} vs {L76.SLinker76.JUDGE_BATCH}")
    check("the context and anchor bounds are untouched",
          L76.SLinker76.CONTEXT_SENTENCES == L75.SLinker75.CONTEXT_SENTENCES
          and L76.SLinker76.ANCHOR_LIMIT == L75.SLinker75.ANCHOR_LIMIT)


def t3():
    print("\nT3  the authored surface is s75's")
    diff = [c for c in AUTHORED if getattr(L75, c) != getattr(L76, c)]
    check(f"all {len(AUTHORED)} authored constants byte-identical", not diff,
          ", ".join(diff) or "none")


def t4():
    print("\nT4  every method body is s75's")
    same = 0
    for attr, obj in vars(L75.SLinker75).items():
        if not callable(getattr(obj, "__func__", obj)):
            continue
        other = vars(L76.SLinker76).get(attr)
        if other is None:
            check(f"method {attr} present", False)
            continue
        a = inspect.getsource(getattr(obj, "__func__", obj)).replace("SLinker75", "X")
        b = inspect.getsource(getattr(other, "__func__", other)).replace(
            "SLinker76", "X").replace("SLinker75", "X")
        if a != b:
            check(f"method {attr} identical", False)
        else:
            same += 1
    check(f"all {same} method bodies identical to s75's", True)


def main():
    print("\ns_linker76 — two batch constants\n" + "=" * 62)
    t1(); t2(); t3(); t4()
    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
