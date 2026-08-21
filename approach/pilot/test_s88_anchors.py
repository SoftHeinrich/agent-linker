"""`s_linker88` differs from `s_linker87` in one repetition removed and nothing else.

    T1  every rule constant is byte-identical to s87's -- this variant deletes no
        English at all
    T2  every method body but the two that render evidence is byte-identical to s87's
    T3  `_format_evidence(bundle)` with no earlier case renders s87's exact bytes, for
        every shape of bundle (with and without a mention label, a preceding sentence,
        anchors)
    T4  a judging call whose cases are all different components renders byte-identically
        to s87's -- the change is invisible unless a component repeats in the batch
    T5  on all five projects' real candidates the change is LOSSLESS case by case:
        every anchor sentence s87 shows a case, s88's call still shows it -- in the
        case itself or in the earlier case it points at -- the pointer always names an
        earlier case of the same component, and the extra sentences a case gains are
        counted and reported
    T6  the bounds, the deterministic layer, the rubrics and the response contract are
        untouched
    T7  GATE-06: no benchmark component name appears in any authored string

    ../.venv/bin/python pilot/test_s88_anchors.py
"""
from __future__ import annotations

import glob
import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker87 as L87              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker88 as L88              # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
PASS, FAIL = [], []

CHANGED = {"_format_evidence", "_validate_with_evidence"}
#: The one method s88 adds: the per-batch union that makes writing anchors once lossless.
ADDED = {"_anchor_union"}
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


def capture_prompts(mod, cls, cands, bundles, comps, sent_map):
    """The judging prompts a variant would send for these candidates."""
    lk = cls.__new__(cls)
    lk.doc_knowledge = None
    seen = []
    lk._ask = lambda prompt, **kw: seen.append(prompt) or {}
    lk._run_validation_pass = (
        lambda comp_names, cases, focus, phase_tag=None, strict=False:
        seen.append(cls._prompt_validation(comp_names, cases, focus, strict=strict))
        or {})
    cls._validate_with_evidence(lk, cands, bundles, comps, sent_map,
                                phase_tag="t", stage_label="full_name")
    return seen


def main():
    print("T1  rule constants")
    c87, c88 = constants(L87), constants(L88)
    same = [n for n in c87 if n not in RENAMED and c87[n] == c88.get(n)]
    diff = [n for n in c87 if n not in RENAMED and c87[n] != c88.get(n)]
    check(f"all {len(same)} rule constants byte-identical", not diff,
          f"differ: {diff}" if diff else "no English changes at all")

    print("\nT2  method bodies")
    b87, b88 = bodies(L87.SLinker87), bodies(L88.SLinker88)
    check("the method set gains exactly the union helper",
          set(b88) - set(b87) == ADDED and not set(b87) - set(b88),
          f"added: {sorted(set(b88) - set(b87))}")
    unchanged = [n for n in b87 if n not in CHANGED and b87[n] == b88.get(n)]
    unexpected = [n for n in b87 if n not in CHANGED and b87[n] != b88.get(n)]
    check(f"{len(unchanged)} of {len(b87)} bodies byte-identical", not unexpected,
          f"unexpected: {unexpected}" if unexpected else
          f"changed exactly: {sorted(CHANGED)}")

    print("\nT3  the evidence line, with no earlier case")
    Bundle = L87.EvidenceBundle
    shapes = [
        Bundle(source="full_name", matched_span="X", mention_type="",
               preceding_text="", anchor_sentences=[]),
        Bundle(source="full_name", matched_span="X", mention_type="via known alias",
               preceding_text="Before.", anchor_sentences=["S3: X runs.", "S9: X."]),
        Bundle(source="full_name", matched_span="X Y", mention_type="",
               preceding_text="Before.", anchor_sentences=["S3: X Y runs."]),
    ]
    ok = all(L87.SLinker87._format_evidence(None, b)
             == L88.SLinker88._format_evidence(None, b) for b in shapes)
    check(f"all {len(shapes)} bundle shapes render s87's bytes at shown_in=0", ok)
    ref = L88.SLinker88._format_evidence(None, shapes[1], None, 4)
    check("a later case names the case that carries the anchors",
          "as shown in Case 4." in ref and "S3: X runs." not in ref, ref.splitlines()[-1])

    print("\nT4/T5  whole judging prompts on real project data")
    total_saved, total_87 = 0, 0
    for proj, (text, model_path) in PROJECTS.items():
        comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
        sents = load_sentences(os.path.join(BASE, "benchmark", text))
        sent_map = build_sent_map(sents)
        lk = L87.SLinker87.__new__(L87.SLinker87)
        lk.doc_knowledge = None
        name_to_id = {c.name: c.id for c in comps}
        # the deterministic half of the candidate set: every pair whose sentence
        # writes a whole name. No LLM call, and enough to fill judging batches.
        cands = []
        for s in sents:
            for comp in comps:
                span = lk._find_exact_form(s.text, comp.name)
                if span:
                    cands.append(L87.CandidateLink(
                        s.number, s.text, comp.name, comp.id, span,
                        source="full_name_candidate"))
        if not cands:
            continue
        bundles = {(c.sentence_number, c.component_id):
                   lk._build_evidence_bundle(c, sent_map) for c in cands}
        p87 = capture_prompts(L87, L87.SLinker87, cands, bundles, comps, sent_map)
        p88 = capture_prompts(L88, L88.SLinker88, cands, bundles, comps, sent_map)
        check(f"{proj}: same number of judging calls", len(p87) == len(p88),
              f"{len(p87)} calls")
        for call87, call88 in zip(p87, p88):
            total_87 += len(call87)
            total_saved += len(call87) - len(call88)
            # the pointer always names an earlier case of the same component
            comp_of = {int(n): name for n, name in
                       re.findall(r"^Case (\d+): \"[^\"]*\" -> (.+)$", call88, re.M)}
            starts = {n: call88.index(f"Case {n}: ") for n in comp_of}
            bad = []
            for m in re.finditer(r"^  Anchors \(confirmed refs\): as shown in "
                                 r"Case (\d+)\.$", call88, re.M):
                target = int(m.group(1))
                here = max(n for n in starts if starts[n] < m.start())
                if target >= here or comp_of.get(target) != comp_of.get(here):
                    bad.append((here, target))
            if bad:
                check(f"{proj}: every pointer is an earlier case of the same "
                      f"component", False, f"{bad[:3]}")
                break
        else:
            check(f"{proj}: every pointer is an earlier case of the same component",
                  True, f"{len(p87)} calls checked")

        # T5, the property the union form exists for: no case is shown less than s87
        # shows it. Checked against the bundles themselves, not the rendering.
        lost = gained = cases_checked = 0
        for start in range(0, len(cands), L88.SLinker88.JUDGE_BATCH):
            batch = cands[start:start + L88.SLinker88.JUDGE_BATCH]
            union = L88.SLinker88._anchor_union(batch, bundles)
            for c in batch:
                mine = set(bundles[(c.sentence_number, c.component_id)]
                           .anchor_sentences)
                if not mine:
                    continue
                cases_checked += 1
                shown_set = set(union.get(c.component_name, []))
                lost += len(mine - shown_set)
                gained += len(shown_set - mine)
        check(f"{proj}: lossless -- no case loses an anchor sentence", lost == 0,
              f"{cases_checked} cases, {gained} extra sentences shown, {lost} lost")

        # T4: a call whose components are all distinct must be byte-identical
        distinct = []
        used = set()
        for c in cands:
            if c.component_name not in used:
                used.add(c.component_name)
                distinct.append(c)
        one87 = capture_prompts(L87, L87.SLinker87, distinct[:20], bundles, comps,
                                sent_map)
        one88 = capture_prompts(L88, L88.SLinker88, distinct[:20], bundles, comps,
                                sent_map)
        check(f"{proj}: a batch of distinct components renders s87's bytes",
              one87 == one88)

    check("the change removes bytes and only bytes", total_saved > 0,
          f"{total_saved} B of {total_87} B "
          f"({100 * total_saved / max(1, total_87):.1f}%) over every judging call "
          f"of all five projects")

    print("\nT6  bounds, deterministic layer, response contract")
    for name in ("JUDGE_BATCH", "EXTRACTION_BATCH", "COREFERENCE_BATCH",
                 "CONTEXT_SENTENCES", "ANCHOR_LIMIT", "ASK_ATTEMPTS"):
        a = getattr(L87.SLinker87, name, None)
        b = getattr(L88.SLinker88, name, None)
        check(f"{name} unchanged ({a})", a == b)
    check("`_prompt_validation` byte-identical",
          b87["_prompt_validation"] == b88["_prompt_validation"])
    check("`_build_evidence_bundle` byte-identical",
          b87["_build_evidence_bundle"] == b88["_build_evidence_bundle"])

    print("\nT7  GATE-06")
    names = set()
    for proj, (_t, model_path) in PROJECTS.items():
        names |= {c.name for c in parse_pcm_repository(
            os.path.join(BASE, "benchmark", model_path))}
    authored = "\n".join(c88.values())
    leaked = sorted(n for n in names if len(n) > 3
                    and re.search(rf"\b{re.escape(n)}\b", authored))
    check(f"no benchmark component name in {len(c88)} authored strings", not leaked,
          f"leaked: {leaked}" if leaked else f"{len(names)} names checked")

    print(f"\n{len(PASS)} checks passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
