"""Can the remaining deterministic rules be *stated in a prompt* instead of coded?

`pilot/rule_audit.py` answered "how many rules are there" — one relation at four
settings, plus a mention label, plus structure. This file asks the next question, and
it is the one a reviewer asks about any hand-written layer: **could the extractor
prompt, the judge prompts, or both, do this work, and what would they have to produce
to reach the same links?**

Deterministic, no LLM calls. Every number here is a *precondition* for an arm, in the
branch's standing order: price it off recorded runs first, then pay for a stage pilot,
then run `composition_check.py`, then pay for E2E only if the composition risk is
non-zero.

The framing this file measures against:

    a rule that only *proposes* can be bound to the extractor prompt — the extraction
    call already reads every sentence and already receives the alias table, so a scan
    is a question the extractor could have been asked;

    a rule that only *labels* can be bound to a judge prompt — the judge is already
    shown the sentence the label was computed from, so the label is a precomputation,
    not new information;

    a rule that is *structure* (batching, set subtraction, windowing) is not a
    statement about text at all and has no prompt form; the only way to bind it is to
    let one call see the whole state, which `s_linker27` priced at macro F1 91.70.

Sections:

    B0  self-check: rebuild each run's full-name candidate set from the recorded
        extraction responses and the scans, and compare with the checkpoint
    B1  the binding inventory: every remaining predicate by where it could be bound
    B2  the binding gap: per scan, what the extractor already proposes and what a
        prompt-bound extractor would have to newly produce, in pairs and in gold
    B3  `_keep_stated_names` as a prompt clause: what it drops, what is gold, and
        what a later linker recovers anyway
    B4  the mention label as a judge-side question: the distribution over judged
        cases, and the two priced defects' footprint
    B5  what has no prompt form, with the measurements that already price it

    cd approach
    ../.venv/bin/python pilot/bind_audit.py
    ../.venv/bin/python pilot/bind_audit.py --only B2
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold, load_project            # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker65 import (             # noqa: E402
    SCANS, MentionType, NameForm, SLinker65,
)

DEFAULT_RUNS = "../results/s64_e2e_r*_20260814"
DEFAULT_ARM = "s_linker64"          # behaviourally identical to s_linker65 (49/49)


class Probe(SLinker65):
    """`SLinker65`'s deterministic layer with no LLM client and a pinned alias table."""

    def __init__(self, aliases):                                      # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": dict(aliases or {})})()


_PROJECT: dict = {}


def project(name):
    if name not in _PROJECT:
        info = load_project(name)
        info["gold"] = load_gold(name)
        info["by_id"] = {c.id: c.name for c in info["components"]}
        _PROJECT[name] = info
    return _PROJECT[name]


# ── the recorded extraction call, parsed the way the linker parses it ────────

def extractor_pairs(run: Path, arm: str, name: str):
    """Every (sentence, component) the extraction call proposed and the code accepted.

    The checkpoint's `candidates` view is post-filter and post-scan, so the extractor's
    own proposal set only exists in the call log. The acceptance conditions are copied
    from `_run_extraction_pass`: a known component, an existing sentence, and a
    `matched_text` that really occurs in it.
    """
    info = project(name)
    pairs = set()
    for path in (run / "llm_logs").glob(f"{arm}_openai_{name}_*_calls.json"):
        with path.open() as handle:
            calls = json.load(handle)
        for call in calls:
            if call.get("phase") != "phase_25_full_name_extract":
                continue
            body = (call.get("response_text") or "").strip()
            fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
            if fence:
                body = fence.group(1).strip()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            for ref in data.get("references", []):
                cname = ref.get("component")
                raw = ref.get("sentence")
                snum = int(raw) if str(raw).lstrip("Ss").isdigit() else None
                if snum is None or cname not in info["name_to_id"]:
                    continue
                sent = info["sent_map"].get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                pairs.add((snum, info["name_to_id"][cname]))
    return pairs


def phase_state(run: Path, arm: str, name: str, phase: str):
    path = run / "phase_states" / arm / "openai" / name / f"{phase}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def units(runs, arm):
    """Yield one (run, project, probe, extractor set, checkpoints) tuple per unit."""
    for run in runs:
        for name in PROJECTS:
            knowledge = phase_state(run, arm, name, "knowledge")
            full_name = phase_state(run, arm, name, "linker_full_name")
            if knowledge is None or full_name is None:
                continue
            aliases = getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {}
            yield (run, name, Probe(aliases), extractor_pairs(run, arm, name),
                   full_name,
                   phase_state(run, arm, name, "linker_partial_name"),
                   phase_state(run, arm, name, "final"))


def scan_pairs(probe, info, scan_name):
    return {(c.sentence_number, c.component_id)
            for c in probe._scan(info["sentences"], info["components"],
                                 SCANS[scan_name])}


def full_name_candidates(probe, info, extractor):
    """Rebuild the full-name linker's candidate set from the extractor's proposals.

    Mirrors `_run_full_name_linker`: the extractor's pairs held to the admission
    filter, then the spelling scan, then the stated-name scan. `_unlinked` is the
    identity here — the full-name linker runs first, so nothing is linked yet.
    """
    kept = {pair for pair in extractor
            if probe._states_a_name(info["sent_map"][pair[0]].text,
                                    info["by_id"][pair[1]])}
    return (kept | scan_pairs(probe, info, "spelling")
            | scan_pairs(probe, info, "stated_name")), kept


def checkpoint_pairs(state, info, key="candidates"):
    out = set()
    for item in state["feedback"][key]:
        cid = info["name_to_id"].get(item["component"])
        if cid is not None:
            out.add((int(item["sentence"]), cid))
    return out


def mean(counter, key, runs):
    return counter[key] / runs if runs else 0.0


# ── B0 ───────────────────────────────────────────────────────────────────────

def b0(runs, arm):
    print("=== B0  self-check: the rebuilt candidate set against the checkpoint ===")
    total = Counter()
    for _, name, probe, extractor, full_name, _, _ in units(runs, arm):
        info = project(name)
        rebuilt, _ = full_name_candidates(probe, info, extractor)
        recorded = checkpoint_pairs(full_name, info)
        total["units"] += 1
        total["recorded"] += len(recorded)
        total["missing"] += len(recorded - rebuilt)
        total["extra"] += len(rebuilt - recorded)
    print(f"  units: {total['units']}   recorded candidates: {total['recorded']}")
    print(f"  rebuilt but not recorded: {total['extra']}   "
          f"recorded but not rebuilt: {total['missing']}")
    print("  (a non-zero residue means the extraction log and the checkpoint "
          "disagree; every number below reads the same two sources)\n")


# ── B1 ───────────────────────────────────────────────────────────────────────

BINDING = [
    # predicate, role, where it could be bound, what the prompt would have to say
    ("_scan / SCANS[stated_name]", "proposes", "extractor",
     "report a reference whenever the sentence writes the component's name exactly "
     "as the catalog spells it, however incidental the mention"),
    ("_scan / SCANS[spelling]", "proposes", "extractor",
     "count a name written with different spacing, hyphenation or compound joining "
     "as that name"),
    ("_scan / SCANS[name_word]", "proposes", "extractor",
     "report a sentence that uses one word of a multi-word name alone when only one "
     "component owns that word"),
    ("_keep_stated_names", "proposes", "extractor + judge",
     "report a reference only when the sentence itself writes a name of the "
     "component / reject a case whose sentence writes no name"),
    ("_classify_mention_typed", "labels", "judge",
     "decide for yourself how the name appears in the sentence shown"),
    ("_inside_qualified_identifier", "primitive (of the above)", "judge",
     "(subordinate: the span-boundary test the label and two scans read)"),
    ("_all_occurrences_in_qualified_path", "primitive (of the above)", "judge",
     "(subordinate: the CODE_TOKEN test)"),
    ("_in_dotted_path", "primitive (of the above)", "judge",
     "(subordinate: the single definition of 'inside a qualified name')"),
    ("_find_exact_form / _name_spans / _realizes / _owners", "primitive", "extractor",
     "(subordinate: the relation the three scans read)"),
    ("_states_a_name", "primitive", "extractor + judge",
     "(subordinate: the admission filter, the whole-name exclusion, the "
     "antecedent gate)"),
    ("_name_signature", "primitive", "extractor",
     "(subordinate: what makes two spellings the same name)"),
    ("_iter_batches", "structural", "none",
     "how many sentences or cases one call sees"),
    ("_window", "structural", "none",
     "which sentences are shown as context"),
    ("_unlinked", "structural", "none",
     "the subtraction between linkers; not a statement about text"),
    ("_union", "structural", "none",
     "the merge of three link sets; not a statement about text"),
]


def b1():
    print("=== B1  the binding inventory ===")
    print(f"{'predicate':<52}{'role':<26}{'bindable to':<18}")
    for predicate, role, where, _ in BINDING:
        print(f"{predicate:<52}{role:<26}{where:<18}")
    by_where = Counter(where for _, _, where, _ in BINDING)
    print(f"\n  extractor-bindable: {by_where['extractor']}   "
          f"judge-bindable: {by_where['judge']}   "
          f"both: {by_where['extractor + judge']}   "
          f"no prompt form: {by_where['none']}")
    print("\n  what each prompt would have to say")
    for predicate, _, where, says in BINDING:
        if where != "none" and not says.startswith("("):
            print(f"    {predicate}\n        {says}")
    print()


# ── B2 ───────────────────────────────────────────────────────────────────────

def b2(runs, arm):
    print("=== B2  the binding gap: what a prompt-bound extractor must newly find ===")
    agg = defaultdict(Counter)
    per_project = defaultdict(Counter)
    examples = Counter()
    for _, name, probe, extractor, full_name, partial, final in units(runs, arm):
        info = project(name)
        gold = info["gold"]
        agg["extractor"]["pairs"] += len(extractor)
        agg["extractor"]["gold"] += len(extractor & gold)
        agg["extractor"]["runs"] += 1
        per_project[(name, "extractor")]["pairs"] += len(extractor)
        per_project[(name, "extractor")]["runs"] += 1
        for scan_name in SCANS:
            pairs = scan_pairs(probe, info, scan_name)
            gap = pairs - extractor
            agg[scan_name]["pairs"] += len(pairs)
            agg[scan_name]["gold"] += len(pairs & gold)
            agg[scan_name]["gap"] += len(gap)
            agg[scan_name]["gap_gold"] += len(gap & gold)
            agg[scan_name]["runs"] += 1
            per_project[(name, scan_name)]["gap"] += len(gap)
            per_project[(name, scan_name)]["gap_gold"] += len(gap & gold)
            per_project[(name, scan_name)]["runs"] += 1
        # what the scans are worth in the *output*: links whose pair the extractor
        # never proposed, by the linker that produced them.
        links = final["final"] if final else []
        for link in links:
            pair = (link.sentence_number, link.component_id)
            if pair in extractor:
                continue
            agg["output"][link.source] += 1
            if pair in gold:
                agg["output"][f"{link.source}_gold"] += 1
                examples[(name, link.source, pair[0],
                          info["by_id"][pair[1]])] += 1
        agg["output"]["runs"] += 1

    runs_n = agg["extractor"]["runs"]
    print(f"\n  per run, over {runs_n} project-runs "
          f"({runs_n // len(PROJECTS)} five-project runs)\n")
    print(f"{'source':<16}{'pairs':>9}{'gold':>8}{'gold/pair':>11}"
          f"{'not proposed':>14}{'gold':>7}{'gold/pair':>11}")
    n = runs_n / len(PROJECTS) if runs_n else 1
    print(f"{'extractor':<16}{agg['extractor']['pairs'] / n:>9.1f}"
          f"{agg['extractor']['gold'] / n:>8.1f}"
          f"{agg['extractor']['gold'] / max(agg['extractor']['pairs'], 1):>11.2f}"
          f"{'—':>14}{'—':>7}{'—':>11}")
    for scan_name in SCANS:
        c = agg[scan_name]
        print(f"{scan_name:<16}{c['pairs'] / n:>9.1f}{c['gold'] / n:>8.1f}"
              f"{c['gold'] / max(c['pairs'], 1):>11.2f}"
              f"{c['gap'] / n:>14.1f}{c['gap_gold'] / n:>7.1f}"
              f"{c['gap_gold'] / max(c['gap'], 1):>11.2f}")

    print("\n  the gap per project (pairs the extractor did not propose)\n")
    print(f"{'project':<16}" + "".join(f"{s:>26}" for s in SCANS))
    for name in PROJECTS:
        row = f"{name:<16}"
        for scan_name in SCANS:
            c = per_project[(name, scan_name)]
            r = c["runs"] or 1
            row += f"{c['gap'] / r:>17.1f} ({c['gap_gold'] / r:>4.1f})"
        print(row)

    print("\n  final links whose pair the extraction call never proposed, "
          "by the linker that produced them")
    out = agg["output"]
    for source in sorted(k for k in out if k not in ("runs",)
                         and not k.endswith("_gold")):
        print(f"    {source:<20}{out[source] / n:>7.1f} per run "
              f"({out[f'{source}_gold'] / n:>5.1f} gold)")
    print("\n    these are the links a prompt-bound extractor has to reproduce "
          "before anything else is measured\n")
    if examples:
        print("  the gold ones, by pair")
        for (name, source, snum, comp), count in examples.most_common(25):
            print(f"    {count}x  {name:<14} s{snum:<5} {comp:<24} {source}")
    print()


# ── B3 ───────────────────────────────────────────────────────────────────────

def b3(runs, arm):
    print("=== B3  the admission filter as a prompt clause ===")
    agg = Counter()
    lost = Counter()
    for _, name, probe, extractor, full_name, partial, final in units(runs, arm):
        info = project(name)
        gold = info["gold"]
        _, kept = full_name_candidates(probe, info, extractor)
        dropped = extractor - kept
        linked = {(l.sentence_number, l.component_id) for l in (final["final"] if final else [])}
        agg["runs"] += 1
        agg["extractor"] += len(extractor)
        agg["kept"] += len(kept)
        agg["dropped"] += len(dropped)
        agg["dropped_gold"] += len(dropped & gold)
        agg["dropped_recovered"] += len(dropped & linked)
        agg["dropped_gold_recovered"] += len(dropped & gold & linked)
        for pair in (dropped & gold) - linked:
            lost[(name, pair[0], info["by_id"][pair[1]])] += 1
    n = agg["runs"] / len(PROJECTS) if agg["runs"] else 1
    print(f"\n  per five-project run, over {agg['runs'] // len(PROJECTS)} runs\n")
    print(f"    extractor proposals               {agg['extractor'] / n:>7.1f}")
    print(f"    the filter keeps                  {agg['kept'] / n:>7.1f}"
          f"   ({agg['kept'] / max(agg['extractor'], 1):.0%})")
    print(f"    the filter drops                  {agg['dropped'] / n:>7.1f}"
          f"   of which gold {agg['dropped_gold'] / n:.1f}")
    print(f"    dropped and linked anyway         "
          f"{agg['dropped_recovered'] / n:>7.1f}"
          f"   of which gold {agg['dropped_gold_recovered'] / n:.1f}")
    print(f"    dropped, gold, never recovered    "
          f"{(agg['dropped_gold'] - agg['dropped_gold_recovered']) / n:>7.1f}")
    print("\n    a prompt clause that reproduced the filter exactly would change "
          "nothing;\n    the question an arm answers is whether an extractor told the "
          "contract\n    stops proposing these at all, and what it stops proposing "
          "with them\n")
    if lost:
        print("  gold the filter drops and no later linker recovers")
        for (name, snum, comp), count in lost.most_common(20):
            print(f"    {count}x  {name:<14} s{snum:<5} {comp}")
    print()


# ── B4 ───────────────────────────────────────────────────────────────────────

def b4(runs, arm):
    print("=== B4  the mention label as a judge-side question ===")
    labels = Counter()
    approved = Counter()
    gold_c = Counter()
    visible = Counter()
    for _, name, probe, extractor, full_name, _, _ in units(runs, arm):
        info = project(name)
        gold = info["gold"]
        candidates, _ = full_name_candidates(probe, info, extractor)
        accepted = checkpoint_pairs(full_name, info, "accepted")
        for pair in candidates:
            text = info["sent_map"][pair[0]].text
            comp = info["by_id"][pair[1]]
            label = probe._classify_mention_typed(comp, text)
            labels[label] += 1
            if pair in accepted:
                approved[label] += 1
            if pair in gold:
                gold_c[label] += 1
            # is the evidence for the label present in what the judge is shown?
            # the judge sees the sentence; a label computed from that sentence alone
            # is a precomputation, not an input the judge lacks.
            if label is MentionType.VIA_ALIAS:
                visible["needs the alias table"] += 1
            else:
                visible["computable from the shown sentence"] += 1
        labels["_runs"] += 0
    total = sum(v for k, v in labels.items() if isinstance(k, MentionType))
    n = len(runs) or 1
    print(f"\n  judged full-name cases per five-project run, over {n} runs\n")
    print(f"{'label':<34}{'cases':>8}{'approved':>10}{'gold':>7}{'appr. rate':>12}")
    for label in MentionType:
        c = labels[label]
        if not c:
            continue
        print(f"{label.value:<34}{c / n:>8.1f}{approved[label] / n:>10.1f}"
              f"{gold_c[label] / n:>7.1f}{approved[label] / max(c, 1):>12.1%}")
    print(f"{'TOTAL':<34}{total / n:>8.1f}")
    print("\n  where the label's evidence lives")
    for key, count in visible.most_common():
        print(f"    {key:<40}{count / n:>8.1f} cases per run")
    print("\n  the two defects the label carries (rule_audit A4), and what "
          "binding removes")
    suppressed = 0
    as_written = 0
    cased = 0
    for name in PROJECTS:
        info = project(name)
        probe = Probe({})
        for sentence in info["sentences"]:
            text = sentence.text
            for comp in info["components"]:
                for form in (NameForm.ANY_SPELLING, NameForm.ANY_WORD):
                    for start, end in probe._name_spans(text, comp.name, form):
                        before = text[start - 1] if start else ""
                        after = text[end] if end < len(text) else ""
                        guarded = (before in ("-", "_")
                                   or (before and before.isalnum())
                                   or after in ("-", "_")
                                   or (after and after.isalnum())
                                   or probe._in_dotted_path(text, start, end))
                        if (probe._inside_qualified_identifier(text, start, end)
                                and not guarded):
                            suppressed += 1
                if not probe._find_exact_form(text, comp.name):
                    continue
                as_written += int(probe._all_occurrences_in_qualified_path(
                    comp.name.lower(), text))
                # the case-consistent reading: the spans the module's own relation
                # finds, all of them inside a dotted path (rule_audit A4)
                spans = probe._name_spans(text, comp.name, NameForm.ANY_CASE)
                cased += int(bool(spans) and all(
                    probe._in_dotted_path(text, s, e) for s, e in spans))
    print(f"    candidate spans the `\"\" in \"-_\"` defect suppresses in the two "
          f"qualified-skipping scans: {suppressed}")
    print(f"    CODE_TOKEN labels the classifier can produce as written: "
          f"{as_written}; with case handled as the rest of the module handles "
          f"it: {cased}")
    print("    both live in the label and the two scans that read the same "
          "boundary test;\n    a judge asked the question directly carries "
          "neither\n")


# ── B5 ───────────────────────────────────────────────────────────────────────

def b5():
    print("=== B5  what has no prompt form ===")
    print("""
  _iter_batches   how many sentences or cases one call sees. Binding it means one
                  call for the whole document: `s_linker27` measured macro F1 91.70
                  and accuracy that tracks document length (jabref 100.0 at 13
                  sentences, teammates 84.1 at 198, 50 references reported where four
                  batches report ~89).

  _window         which sentences are shown as context. It is what a prompt *carries*,
                  not what it says; halving CONTEXT_SENTENCES cost 2.0 TP (p=0.20) for
                  no precision gain, and narrowing coreference's scope to unlinked
                  sentences would lose 14.5 of 30.0 coreference links per run.

  _unlinked       the subtraction between linkers, and the mechanism behind seven of
                  this branch's stage-vs-pipeline reversals. Stating it in a prompt
                  means telling a judge what another judge already accepted, which is
                  the shared-context arrangement the merge line (s32-s38) measured as
                  precision-negative every time.

  _union          the merge of three link sets. Set arithmetic over three calls'
                  outputs; there is no sentence of English that performs it.

  These four are the residue of the deterministic layer once the proposing and
  labelling rules are bound. They are control flow over calls, not statements about
  text, so "how many hand-written rules does the workflow have" has a floor above zero
  and the floor is structural.
""")


# ── B6 ───────────────────────────────────────────────────────────────────────

#: Which later stages a change at this stage can be stolen from, in run order.
LATER = {"full_name": (("linker_partial_name", "proposed"),
                       ("linker_coreference", "candidates")),
         "partial_name": (("linker_coreference", "candidates"),)}


def b6(runs, arm):
    """`composition_check.py`'s question, asked of each relocation.

    A stage arm is the pipeline answer only when the pairs it adds or removes are
    not pairs a later stage would otherwise propose, and are not already in the final
    link set. Deterministic; it decides which of these arms has to be paid for
    end to end.
    """
    print("=== B6  composition risk per relocation ===")
    risk = Counter()
    for run, name, probe, extractor, full_name, partial, final in units(runs, arm):
        info = project(name)
        _, kept = full_name_candidates(probe, info, extractor)
        linked = {(l.sentence_number, l.component_id)
                  for l in (final["final"] if final else [])}
        later = {}
        for phase, view in LATER["full_name"]:
            state = phase_state(run, arm, name, phase)
            later[phase] = checkpoint_pairs(state, info, view) if state else set()
        moves = {
            # bindscans, uncompensated: the pairs only the two tight scans propose
            "bindscans removes": ((scan_pairs(probe, info, "stated_name")
                                   | scan_pairs(probe, info, "spelling")) - kept,
                                  "full_name"),
            # bindcontract, uncompensated: the proposals the filter drops
            "bindcontract adds": (extractor - kept, "full_name"),
            # bindpartial, uncompensated: everything the partial-name proposer offers
            "bindpartial removes": (scan_pairs(probe, info, "name_word"),
                                    "partial_name"),
        }
        for label, (pairs, stage) in moves.items():
            risk[(label, "pairs")] += len(pairs)
            risk[(label, "in the final link set")] += len(pairs & linked)
            for phase, _ in LATER[stage]:
                risk[(label, f"also proposed by {phase}")] += len(
                    pairs & later.get(phase, set()))
        risk["units"] += 1
    n = risk["units"] / len(PROJECTS) if risk["units"] else 1
    print(f"\n  per five-project run, over {int(risk['units'] // len(PROJECTS))} "
          f"runs\n")
    for label in ("bindscans removes", "bindcontract adds", "bindpartial removes"):
        print(f"  {label}")
        total = 0
        for key, count in risk.items():
            if not isinstance(key, tuple) or key[0] != label:
                continue
            if key[1] == "pairs":
                print(f"      {count / n:6.1f}   candidates moved")
                continue
            total += count
            print(f"      {count / n:6.1f}   {key[1]}")
        print(f"      composition risk {total / n:.1f} pairs per run — "
              + ("the stage arm is the pipeline answer"
                 if not total else "end-to-end evidence needed"))
    print("\n  bindlabel moves no candidate: it changes verdicts on the same set, "
          "and\n  every full-name verdict feeds `_unlinked`, so its composition risk "
          "is the\n  size of its own verdict delta — read it off the stage pilot, "
          "not from here.\n")


# ── B7 ───────────────────────────────────────────────────────────────────────

def b7(runs, arm):
    """What is left to cut once the extractor side is bound, priced before paying.

    After `s_linker67` the only scan left is the partial-name row, and it carries
    three options and one span-boundary predicate. Each is a decision the target-blind
    denotation judge behind it might make as well; each is also a rule a reviewer has
    to be told about. This sizes the freed candidates and their gold, which is what
    says whether an arm is worth five samples.
    """
    from dataclasses import replace
    print("=== B7  the cuts left after the extractor side is bound ===")
    options = {
        "no span-boundary test (skip_qualified)": dict(skip_qualified=False),
        "no unique-owner test (unique_owner)": dict(unique_owner=False),
        "no whole-name exclusion (skip_when_named)": dict(skip_when_named=False),
    }
    agg = Counter()
    for _, name, probe, _, _, _, _ in units(runs, arm):
        info = project(name)
        gold = info["gold"]
        base = scan_pairs(probe, info, "name_word")
        agg[("base", "pairs")] += len(base)
        agg[("base", "gold")] += len(base & gold)
        for label, override in options.items():
            wide = {(c.sentence_number, c.component_id)
                    for c in probe._scan(info["sentences"], info["components"],
                                         replace(SCANS["name_word"], **override))}
            agg[(label, "pairs")] += len(wide - base)
            agg[(label, "gold")] += len(gold & (wide - base))
        agg["runs"] += 1
    n = agg["runs"] / len(PROJECTS) if agg["runs"] else 1
    print(f"\n  per five-project run, over {int(agg['runs'] // len(PROJECTS))} runs\n")
    print(f"{'the partial-name scan':<44}{'pairs':>8}{'gold':>7}{'gold/pair':>11}")
    base_pairs, base_gold = agg[("base", "pairs")], agg[("base", "gold")]
    print(f"{'as it stands':<44}{base_pairs / n:>8.1f}{base_gold / n:>7.1f}"
          f"{base_gold / max(base_pairs, 1):>11.2f}")
    for label in options:
        pairs, gold_n = agg[(label, "pairs")], agg[(label, "gold")]
        print(f"{'  + ' + label:<44}{pairs / n:>8.1f}{gold_n / n:>7.1f}"
              f"{gold_n / max(pairs, 1):>11.2f}")
    print("\n    a cut is worth an arm when the freed pairs carry gold the judge can "
          "keep;\n    a cut that frees only spurious pairs can at best be neutral and "
          "is not\n    worth five samples — it is already priced here\n")


SECTIONS = {"B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5, "B6": b6,
            "B7": b7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--only", default=None, help="B0..B5")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))
    if not runs:
        raise SystemExit(f"no runs matched {args.runs}")
    print(f"\nbind audit — {args.arm} over {len(runs)} runs: "
          f"{', '.join(r.name for r in runs)}\n")
    for key, fn in SECTIONS.items():
        if args.only and key != args.only:
            continue
        if key in ("B1", "B5"):
            fn()
        else:
            fn(runs, args.arm)


if __name__ == "__main__":
    main()
