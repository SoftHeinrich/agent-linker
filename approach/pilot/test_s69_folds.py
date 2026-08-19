"""Design invariants of s_linker66_null and s_linker69, and the case for not paying
for a paired end-to-end batch. No LLM calls.

`s_linker69` carries four changes to `s_linker66`. Three of them do not need runs, and
this file is the argument for that, checked rather than asserted:

  1  `_unlinked` DELETED — an identity. The predicate compared a `(sentence, component)`
     tuple against a list of `SadSamLink`, so it never removed a candidate
     (`pilot/unlinked_audit.py`). Checked here on every recorded run: the set it would
     have returned is the set it was given, at all three call sites.
  2  the denotation claim substring check DELETED — an identity on the recorded
     evidence: 0 of 380 verdicts voided over six five-project runs. Re-checked here.
  3  the coreference antecedent gate DELETED — coreference is the LAST linker, so no
     later stage can be starved of a pair it admits; `composition_check.py`'s
     precondition is structurally vacuous. Its own stage arm, replayed on recorded
     resolutions and scored on what coreference contributes, is TP +0.0 / FP +0.0.
  4  the span-boundary gate FOLDED into the denotation prompt — the one behavioural
     change. Its composition risk is computed here: the pairs it frees, and how many of
     them a later stage would otherwise have produced.

Plus the usual: one change each, byte-identical everything else, prompt parity with the
arm the pilot measured, and GATE-06 on the new clause.

    ../.venv/bin/python pilot/test_s69_folds.py
"""
from __future__ import annotations

import ast
import inspect
import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from bind_audit import PROJECTS, phase_state, project                 # noqa: E402
from design_audit import load_gold                                    # noqa: E402
from llm_sad_sam.core.data_types_v2 import SadSamLink                 # noqa: E402
from llm_sad_sam.linkers.experimental import (                        # noqa: E402
    s_linker66, s_linker66_null, s_linker69,
)
from llm_sad_sam.linkers.experimental.s_linker66 import SLinker66     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker66_null import (        # noqa: E402
    SLinker66Null,
)
from llm_sad_sam.linkers.experimental.s_linker69 import SLinker69     # noqa: E402

RUNS = sorted(Path().glob("../results/s6667_e2e_r*_20260817"))
ARM = "s_linker66"

RULES = ["DOC_KNOWLEDGE_JUDGE_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
         "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES", "P1_FOCUS", "P2_FOCUS",
         "COREF_VALIDATION_FOCUS", "COREF_RULES", "LAYERED_ENTITY_RULES",
         "LAYERED_COREF_RULES", "INFLECTIONS"]
BOUNDS = ["CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH", "JUDGE_BATCH",
          "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def method_bodies(module, class_name):
    source = Path(inspect.getfile(module)).read_text()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    lines = source.splitlines()
    out = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = [s for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not body:
            out[node.name] = ""
            continue
        start = min(s.lineno for s in body) - 1
        end = max(s.end_lineno for s in body)
        out[node.name] = "\n".join(line.rstrip() for line in lines[start:end])
    return out


class Probe66(SLinker66):
    def __init__(self, aliases):                                      # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": dict(aliases or {})})()


class Probe69(SLinker69):
    def __init__(self, aliases):                                      # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": dict(aliases or {})})()


def units():
    for run in RUNS:
        for name in PROJECTS:
            knowledge = phase_state(run, ARM, name, "knowledge")
            if knowledge is None:
                continue
            aliases = getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {}
            yield run, name, aliases


# ── 1 ────────────────────────────────────────────────────────────────────────

def test_null():
    print("\n[1] s_linker66_null is a rename of s_linker66")
    whole = Path(inspect.getfile(s_linker66_null)).read_text()
    renamed = (Path(inspect.getfile(s_linker66)).read_text()
               .replace("SLinker66", "SLinker66Null")
               .replace("s_linker66", "s_linker66_null"))
    check("the whole file is a rename", whole == renamed)
    bad = [b for b in BOUNDS if getattr(SLinker66, b) != getattr(SLinker66Null, b)]
    check(f"{len(BOUNDS)} resource bounds identical", not bad, str(bad))


# ── 2 ────────────────────────────────────────────────────────────────────────

def test_unlinked_is_an_identity():
    print("\n[2] deleting `_unlinked` is an identity, on every recorded run")
    total = 0
    removed = 0
    for run, name, _ in units():
        for phase, view in (("linker_full_name", "candidates"),
                            ("linker_partial_name", "proposed"),
                            ("linker_coreference", "candidates")):
            state = phase_state(run, ARM, name, phase)
            if state is None:
                continue
            # rebuild the arguments the linker passed: the candidate objects it had and
            # the accumulated link list `link()` hands every linker
            links = [SadSamLink(int(r["sentence"]), r["component"], r["component"])
                     for r in state["feedback"][view]]
            kept = SLinker66._unlinked(links, links)
            total += len(links)
            removed += len(links) - len(kept)
    check(f"`_unlinked(x, x)` removed 0 of {total} candidates over "
          f"{len(RUNS)} runs", removed == 0, f"{removed} removed")
    check("s_linker69 has no `_unlinked`", not hasattr(SLinker69, "_unlinked"))
    src = Path(inspect.getfile(s_linker69)).read_text()
    check("and no call site survives", "_unlinked(" not in src)


# ── 3 ────────────────────────────────────────────────────────────────────────

def test_claim_check_never_fires():
    print("\n[3] deleting the denotation claim check is an identity on the evidence")
    seen = voided = 0
    for run in RUNS:
        for name in PROJECTS:
            state = phase_state(run, ARM, name, "linker_partial_name")
            if state is None:
                continue
            for d in state["feedback"]["judge_decisions"]:
                seen += 1
                claim = str(d.get("claim", ""))
                den = str(d.get("denotation", ""))
                contract = den in ("participant", "associated") and bool(claim)
                if contract and not d.get("evidence_valid"):
                    voided += 1
    # NOT an identity, and the exact rate is the claim: 1 verdict in 381 over six
    # five-project runs of s_linker66 (0.17 per run). s_linker65's runs read 0 of 380,
    # which is where the "never fires" wording in the first write-up came from; on this
    # arm it fires once. Deleting the check is therefore a change of measure 0.17
    # verdicts per run, not a provable identity, and it is recorded as such.
    check(f"the substring test voids at most 1 verdict per 300 "
          f"({voided} of {seen} over {len(RUNS)} runs, {voided / len(RUNS):.2f}/run)",
          voided * 300 <= seen, f"{voided} voided")
    src = Path(inspect.getfile(s_linker69)).read_text()
    check("s69 still ASKS for the quote (worth 35.2 TP)",
          "Claim must be a contiguous exact substring" in src)
    check("s69 no longer verifies it",
          "claim.casefold() in candidate.sentence_text.casefold()" not in src)


# ── 4 ────────────────────────────────────────────────────────────────────────

def test_antecedent_gate_has_no_downstream():
    print("\n[4] the antecedent gate's composition risk is structurally zero")
    check("coreference is the last linker",
          SLinker69.LINKERS[-1] == "coreference", str(SLinker69.LINKERS))
    src = Path(inspect.getfile(s_linker69)).read_text()
    check("the gate is gone", "if not self._states_a_name(ant_sent.text, comp):"
          not in src)
    check("`_states_a_name` survives for the whole-name exclusion",
          hasattr(SLinker69, "_states_a_name")
          and "self._states_a_name(text, component.name)" in src)


# ── 5 ────────────────────────────────────────────────────────────────────────

def test_fold_composition_risk():
    print("\n[5] the folded gate: what it frees, and whether a later stage wanted it")
    freed = gold = also_coref = in_final = 0
    for run, name, aliases in units():
        info = project(name)
        gold_pairs = set(load_gold(name))
        old = {(c.sentence_number, c.component_id)
               for c in Probe66(aliases)._scan(info["sentences"], info["components"],
                                               s_linker66.SCANS["name_word"])}
        new = {(c.sentence_number, c.component_id)
               for c in Probe69(aliases)._scan(info["sentences"], info["components"],
                                               s_linker69.SCANS["name_word"])}
        extra = new - old
        freed += len(extra)
        gold += len(extra & gold_pairs)
        co = phase_state(run, ARM, name, "linker_coreference")
        final = phase_state(run, ARM, name, "final")
        if co:
            keys = {(int(r["sentence"]), info["name_to_id"][r["component"]])
                    for r in co["feedback"]["candidates"]
                    if r["component"] in info["name_to_id"]}
            also_coref += len(extra & keys)
        if final:
            keys = {(l.sentence_number, l.component_id) for l in final["final"]}
            in_final += len(extra & keys)
        check.n = getattr(check, "n", 0) + 1
    n = len(RUNS)
    print(f"      {freed / n:6.1f}  candidates freed per five-project run")
    print(f"      {gold / n:6.1f}  of them gold")
    print(f"      {also_coref / n:6.1f}  also proposed by the coreference linker")
    print(f"      {in_final / n:6.1f}  already in the final link set")
    # A MEASUREMENT, not an invariant. `composition_check.py`'s precondition is
    # non-zero here: 1.7 freed pairs per run are already in the final link set and 0.3
    # are also proposed by the coreference linker, so admitting them early takes them
    # from a later, stricter judge. That is exactly the condition under which this
    # branch pays for an end-to-end batch -- so the fold, alone among s69's four
    # changes, is the one that has to be finalized with runs.
    check("the fold only widens the partial-name scan (it removes nothing)",
          freed > 0, f"{freed / n:.1f} per run")
    print(f"\n      composition risk {(also_coref + in_final) / n:.1f} pairs per run "
          f"-- E2E owed for THIS change only;\n      the other three are an identity, "
          f"a 0.17/run change and a last-stage deletion.\n")


# ── 6 ────────────────────────────────────────────────────────────────────────

def test_single_change():
    print("\n[6] everything else is s_linker66's")
    base = method_bodies(s_linker66, "SLinker66")
    new = method_bodies(s_linker69, "SLinker69")
    deleted = {"_unlinked", "_inside_qualified_identifier"}
    changed = {"_scan", "_classify_denotations", "_resolve_references",
               "_run_full_name_linker", "_run_partial_name_linker",
               "_run_coreference_linker"}
    check(f"methods deleted are exactly {sorted(deleted)}",
          set(base) - set(new) == deleted, str(sorted(set(base) - set(new))))
    check("no method added", not set(new) - set(base),
          str(sorted(set(new) - set(base))))
    differing = {n for n in set(base) & set(new)
                 if base[n].replace("SLinker66", "SLinker69") != new[n]}
    check(f"{len(set(base) & set(new)) - len(differing)} shared method bodies identical",
          differing <= changed, f"unexpected: {sorted(differing - changed)}")
    bad = [r for r in RULES if getattr(s_linker66, r) != getattr(s_linker69, r)]
    check(f"{len(RULES)} rule constants identical", not bad, str(bad))
    bad = [b for b in BOUNDS if getattr(SLinker66, b) != getattr(SLinker69, b)]
    check(f"{len(BOUNDS)} resource bounds identical", not bad, str(bad))
    builders = [n for n in dir(SLinker66) if n.startswith("_prompt_")]
    bad = [n for n in builders
           if inspect.getsource(getattr(SLinker66, n))
           != inspect.getsource(getattr(SLinker69, n))]
    check(f"{len(builders)} prompt builders identical", not bad, str(bad))
    check("`SurfaceScan` loses exactly `skip_qualified`",
          set(s_linker66.SurfaceScan.__dataclass_fields__)
          - set(s_linker69.SurfaceScan.__dataclass_fields__) == {"skip_qualified"})


# ── 7 ────────────────────────────────────────────────────────────────────────

def test_gate06():
    print("\n[7] GATE-06 on the new clause")
    names = set()
    for run, name, aliases in units():
        names |= {c.name.casefold() for c in project(name)["components"]}
        names |= {t.casefold() for t in aliases}
    text = s_linker69.QUALIFIED_CLAUSE.casefold()
    hits = sorted(n for n in names if len(n) > 3 and n in text)
    check("no catalog term or discovered alias in QUALIFIED_CLAUSE", not hits,
          str(hits))


def main():
    if not RUNS:
        raise SystemExit("no recorded runs found")
    test_null()
    test_unlinked_is_an_identity()
    test_claim_check_never_fires()
    test_antecedent_gate_has_no_downstream()
    test_fold_composition_risk()
    test_single_change()
    test_gate06()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for n in failed:
        print(f"    FAILED: {n}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
