"""`s_linker89` differs from `s_linker88` in one deleted line and nothing else.

    T1  every rule constant is byte-identical to s88's -- no English is deleted; the
        line that goes is builder text, not a rule
    T2  every method body but `_prompt_coref` is byte-identical to s88's
    T3  the resolver prompt is s88's minus exactly one `CONTEXT: ...` line per case,
        rendered on all five projects' real batches, and the input-format contract
        (the preamble s56 priced at TP -16.2) is still in it
    T4  the deletion is 324 B per resolver call, 12 961 B per five-project run
    T5  the bounds and the deterministic layer are untouched
    T6  GATE-06: no benchmark component name in any authored string

    ../.venv/bin/python pilot/test_s89_compact.py
"""
from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker88 as L88              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker89 as L89              # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences         # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
PASS, FAIL = [], []
CHANGED = {"_prompt_coref"}
RENAMED = {"_VARIANT_NAME"}
PROJECTS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository"),
}


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def constants(mod):
    return {n: getattr(mod, n) for n in dir(mod)
            if n.isupper() and isinstance(getattr(mod, n), str)}


def bodies(cls):
    out = {}
    for name, member in vars(cls).items():
        fn = member.__func__ if isinstance(member, staticmethod) else member
        if callable(fn):
            try:
                out[name] = inspect.getsource(fn)
            except (OSError, TypeError):
                pass
    return out


def main():
    print("T1  rule constants")
    c88, c89 = constants(L88), constants(L89)
    diff = [n for n in c88 if n not in RENAMED and c88[n] != c89.get(n)]
    check(f"all {len(c88) - len(RENAMED)} rule constants byte-identical", not diff,
          f"differ: {diff}" if diff else "the cut is builder text, not a rule")

    print("\nT2  method bodies")
    b88, b89 = bodies(L88.SLinker88), bodies(L89.SLinker89)
    check("same method set", set(b88) == set(b89))
    unexpected = [n for n in b88 if n not in CHANGED and b88[n] != b89.get(n)]
    check(f"{len(b88) - len(CHANGED)} of {len(b88)} bodies byte-identical",
          not unexpected, f"unexpected: {unexpected}" if unexpected else
          f"changed exactly: {sorted(CHANGED)}")

    print("\nT3/T4  the resolver prompt on real batches")
    total_saved = total_calls = 0
    for proj, (text, model_path) in PROJECTS.items():
        comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
        sents = load_sentences(os.path.join(BASE, "benchmark", text))
        names = [c.name for c in comps]
        sm = {s.number: s for s in sents}
        size = L88.SLinker88.COREFERENCE_BATCH
        for start in range(0, len(sents), size):
            batch = sents[start:start + size]
            window_ids, targets = set(), []
            for i, sent in enumerate(batch, 1):
                w = [s.number for s in sents
                     if abs(s.number - sent.number) <= L88.SLinker88.CONTEXT_SENTENCES]
                window_ids.update(w)
                targets.append({"case": i, "target": sent.number,
                                "text": sent.text, "context": w})
            table = [{"sentence": n, "text": sm[n].text}
                     for n in sorted(window_ids) if n in sm]
            p88 = L88.SLinker88._prompt_coref(names, table, targets)
            p89 = L89.SLinker89._prompt_coref(names, table, targets)
            total_calls += 1
            total_saved += len(p88) - len(p89)
            stripped = re.sub(r"\nCONTEXT: sentences S\d+-S\d+ above\.", "", p88)
            if stripped != p89:
                check(f"{proj}: s89's prompt is s88's minus the CONTEXT lines", False)
                return
    check("every resolver prompt of all five projects is s88's minus the CONTEXT "
          "lines", True, f"{total_calls} calls compared")
    for contract in ("identify any pronoun or noun phrase in THAT sentence",
                     "return no resolution", "Be conservative"):
        check(f"the input contract keeps: {contract!r}",
              contract in L89.SLinker89._prompt_coref(["X"], [], []))
    per_call = total_saved / max(1, total_calls)
    check("the deletion is ~324 B a call and ~13 kB a five-project run",
          300 < per_call < 340, f"{per_call:.0f} B/call, {total_saved} B over "
          f"{total_calls} calls")

    print("\nT5  bounds and the deterministic layer")
    for name in ("JUDGE_BATCH", "EXTRACTION_BATCH", "COREFERENCE_BATCH",
                 "CONTEXT_SENTENCES", "ANCHOR_LIMIT", "ASK_ATTEMPTS"):
        check(f"{name} unchanged",
              getattr(L88.SLinker88, name) == getattr(L89.SLinker89, name))

    print("\nT6  GATE-06")
    names = set()
    for proj, (_t, model_path) in PROJECTS.items():
        names |= {c.name for c in parse_pcm_repository(
            os.path.join(BASE, "benchmark", model_path))}
    authored = "\n".join(c89.values())
    leaked = sorted(n for n in names if len(n) > 3
                    and re.search(rf"\b{re.escape(n)}\b", authored))
    check(f"no benchmark component name in {len(c89)} authored strings", not leaked,
          f"leaked: {leaked}" if leaked else f"{len(names)} names checked")

    print(f"\n{len(PASS)} checks passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
