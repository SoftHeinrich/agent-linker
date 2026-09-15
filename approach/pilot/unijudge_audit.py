"""One judge for every stream: what the rule may say, and what the evidence must carry.

No LLM calls. Everything is computed off the five benchmark documents, their catalogs,
and the recorded checkpoints of six promoted head runs.

**The question.** `s_linker110` judges three streams with three prompts. Two of them are
already one builder (`_prompt_validation`, `strict=` picking the rubric); the third
(`_classify_denotations`) is a different question asked of a different case, and it is
**target-blind**. The proposal under audit: *one static link rule, asked of every stream,
with the difference carried by an evidence bundle computed from the match* — the design
law ("facts in code, weighings in the prompt") applied to the judge's input rather than
to its rubric, and the fold law of `gate_inventory.py` run forwards: **a gate folds into a
judge's prompt exactly when that judge is shown the information the gate reads.** If the
evidence bundle is what differs, then what each judge may be told is a code fact, and the
rule above it can be one paragraph.

Two measured results stand in the way and both are respected here rather than argued
with:

  * `s25` — showing the denotation judge its target cost **5.5 gold a run**: "shown the
    target, the model confirms identity rather than testing it".
  * `s119` — making the sortal gate reply in the lenient gate's boolean cost net
    **−9.0 terra / −16.0 luna**: a schema carries a default, and the two streams' defaults
    are opposite because their base rates are (0.73–0.75 against 0.24–0.29).

So a union that only rewrites prompts is already refuted. The one open question is
whether **evidence computed from the match** can carry what those two differences carry.
This audit prices that, and only that.

  U1  the census      every axis on which the three judging calls differ, each one
                      classified: FACT of the match (may become an evidence field),
                      WEIGHING (must stay in the one rule), or CONTRACT (the reply).
                      Read off the module's own source, not off a description of it.
  U2  the evidence    every field a merged case could carry, computed for all 296
                      deterministic candidates, with the gold rate of each cell — is
                      the evidence a sufficient statistic for the routing two rubrics
                      do today?
  U3  blindness       the alternative set, in code: for word-only cases, where the
                      gold and where the recorded false positives sit once the case's
                      competing components are enumerated. This is what a shown target
                      would have to be paid for with.
  U4  the prompt      the merged prompt built for real batches, checked case by case
                      against what each case's current stage shows it (no case may be
                      shown less), with byte counts.
  U5  the plan        cases, calls and the stage pilot the arm owes.

Usage, from the approach/ directory:
    ../.venv/bin/python pilot/unijudge_audit.py
    ../.venv/bin/python pilot/unijudge_audit.py --only U2 U3
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from simmerge_audit import (                                         # noqa: E402
    E2E_RUNS, arm_full, arm_partial, head_instance, load_project,
)
from llm_sad_sam.linkers.experimental import s_linker110             # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import (           # noqa: E402
    SLinker110, NameForm, WORD_PATTERN, lemmas,
)

REPORT = Path("../results/unijudge_audit")


# ─────────────────────────────────────────────────────────────────────────────
# U1 — the census of differences, read off the source
# ─────────────────────────────────────────────────────────────────────────────

#: Every axis the three judging calls differ on. `kind` is the audit's claim about
#: where that axis belongs under the design law; `probe` is a predicate over the
#: module's source that fails loudly if the axis is described wrongly here.
AXES = [
    ("rubric constant", "WEIGHING",
     "lenient LAYERED_ENTITY_RULES / strict LAYERED_COREF_RULES / the denotation line",
     lambda src: "LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES" in src),
    ("premise of the rubric", "FACT",
     "'the component is named here' vs 'is NOT named in the sentence itself' — "
     "`_states_a_name` computes exactly this",
     lambda src: "which is NOT named in the sentence itself" in src),
    ("target shown", "FACT",
     "full-name and coref cases print `-> ComponentName`; denotation cases print the "
     "expression only",
     lambda src: '"{c.matched_text}" -> {c.component_name}' in src
     and '"expression": c.matched_text' in src),
    ("catalog shown", "FACT",
     "`COMPONENTS: ...` heads the validation prompt; the denotation prompt has no "
     "catalog",
     lambda src: "COMPONENTS: {', '.join(comp_names)}" in src),
    ("evidence bundle", "FACT",
     "source / span / mention label / preceding sentence / anchor sentences — "
     "validation only",
     lambda src: "Evidence: source=" in src),
    ("context window", "FACT",
     "denotation prints a SENTENCES table over ±CONTEXT_SENTENCES; validation prints "
     "the previous sentence and the anchors",
     lambda src: "self._window(candidate.sentence_number, sentences)" in src),
    ("qualified clause", "WEIGHING",
     "QUALIFIED_CLAUSE joins the lenient rubric and the denotation prompt, never the "
     "strict one",
     lambda src: src.count("{QUALIFIED_CLAUSE}") >= 2),
    ("stricter clause", "WEIGHING",
     "STRICTER_CLAUSE joins the lenient rubric only",
     lambda src: "{STRICTER_CLAUSE}" in src),
    ("objection field", "CONTRACT",
     "asked of the strict gate only; s118 priced adding it elsewhere at ±0.0 net",
     lambda src: '"objection": "<strongest ground to reject, or none>"' in src),
    ("verdict vocabulary", "CONTRACT",
     "`approve` boolean against `denotation` enum — s119 measured the swap at "
     "net −9.0 / −16.0",
     lambda src: '"approve": true' in src and '"denotation":"participant"' in src),
    ("claim quote", "WEIGHING",
     "demanded before the verdict in all three; worth 35.2 TP on its own",
     lambda src: src.count("exact") >= 3),
    ("batch bound", "CONTRACT",
     "JUDGE_BATCH = 25 for all three judging passes",
     lambda src: "JUDGE_BATCH = 25" in src),
]


def u1(sink):
    src = inspect.getsource(s_linker110)
    rows = []
    sink("\n  every axis the three judging calls differ on, and where it belongs")
    sink(f"    {'axis':22s} {'kind':9s} {'probe':6s}  what differs")
    for axis, kind, what, probe in AXES:
        ok = bool(probe(src))
        sink(f"    {axis:22s} {kind:9s} {'ok' if ok else 'FAIL':6s}  {what}")
        rows.append({"axis": axis, "kind": kind, "source_probe": "ok" if ok else "FAIL",
                     "what_differs": what})
    kinds = Counter(kind for _, kind, _, _ in AXES)
    sink(f"\n    {kinds['FACT']} axes are facts of the match, {kinds['WEIGHING']} are "
         f"weighings, {kinds['CONTRACT']} are the reply contract.")
    sink("    A union may move every FACT into the evidence bundle. It may not move a "
         "WEIGHING\n    there — that is the rule — and every CONTRACT axis is a "
         "measured refusal (s118, s119).")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# U2 — the evidence a merged case can carry
# ─────────────────────────────────────────────────────────────────────────────

def word_spans(linker, text, name):
    return linker._name_spans(text, name, NameForm.ANY_WORD)


def alternatives_for(linker, text, component, components):
    """Every OTHER component this case's matched words could equally point to.

    A code fact, computed with the module's own relation: a component whose name
    shares a word with the matched word, at any WordNet reading. This is the
    `s_linker107` shape of the enumeration — the alternative set is a fact when the
    case contains it — applied to the name streams instead of to the resolver.
    """
    mine = {tuple(sorted(lemmas(word)))
            for word in re.findall(WORD_PATTERN, component.name)}
    out = []
    for other in components:
        if other.name == component.name:
            continue
        theirs = {tuple(sorted(lemmas(word)))
                  for word in re.findall(WORD_PATTERN, other.name)}
        if mine & theirs and word_spans(linker, text, other.name):
            out.append(other.name)
    return out


def case_evidence(linker, sentence, component, components, sent_map, aliases):
    """Every field a merged case could carry, all computed from the match."""
    text = sentence.text
    whole = linker._writes_name(text, component.name)
    alias_hit = ""
    if not whole:
        for term, owner in aliases.items():
            if owner == component.name and linker._find_exact_form(text, term):
                alias_hit = term
                break
    span = whole or alias_hit
    words = word_spans(linker, text, component.name)
    if not span and words:
        start, end = words[0]
        span = text[start:end]
    anchors = [s for s in sent_map.values()
               if s.number != sentence.number
               and linker._find_exact_form(s.text, component.name)]
    before = [s.number for s in anchors if s.number < sentence.number]
    others = [c.name for c in components
              if c.name != component.name and linker._states_a_name(text, c.name)]
    naming = ("whole name" if whole else
              "alias" if alias_hit else
              "word only" if words else "no name")
    return {
        "naming": naming,
        "span": span,
        "capitalized": bool(span) and span[0].isupper(),
        "mention_label": linker._retained_mention_label(component.name, text),
        "anchors": len(anchors),
        "last_named_before": (sentence.number - max(before)) if before else -1,
        "alternatives": alternatives_for(linker, text, component, components),
        "covered_by_other_name": linker._only_inside_another_name(
            text, component.name, components),
        "other_components_named_here": others,
    }


def stream_of(pair, full, partial):
    if pair in full:
        return "full_name"
    if pair in partial:
        return "partial_name"
    return "other"


def u2(projects, sink):
    rows = []
    cells = defaultdict(lambda: Counter())
    for data in projects:
        linker, gold = data["linker"], data["gold"]
        components, aliases = data["components"], data["aliases"]
        sent_map = {s.number: s for s in data["sentences"]}
        by_id = {c.id: c for c in components}
        full = arm_full(data)
        partial = arm_partial(data)
        for pair in sorted(full | partial):
            sentence = sent_map[pair[0]]
            evidence = case_evidence(linker, sentence, by_id[pair[1]], components,
                                     sent_map, aliases)
            is_gold = pair in gold
            stream = stream_of(pair, full, partial)
            for field, value in (
                ("naming", evidence["naming"]),
                ("capitalized", evidence["capitalized"]),
                ("mention label", evidence["mention_label"] or "(none)"),
                ("anchors", "0" if evidence["anchors"] == 0
                 else "1-2" if evidence["anchors"] <= 2 else "3+"),
                ("alternatives", str(min(len(evidence["alternatives"]), 2))),
                ("other names in sentence",
                 str(min(len(evidence["other_components_named_here"]), 2))),
                ("stream", stream),
            ):
                cells[field][(str(value), "gold" if is_gold else "not")] += 1
            rows.append({"project": data["name"], "sentence": pair[0],
                         "component": by_id[pair[1]].name, "stream": stream,
                         "gold": int(is_gold),
                         **{k: (len(v) if isinstance(v, list) else v)
                            for k, v in evidence.items()}})
    sink("\n  the merged deterministic stream, by evidence field "
         "(all five projects, 296 cases)")
    for field, counter in cells.items():
        values = sorted({value for value, _ in counter})
        sink(f"\n    {field}")
        for value in values:
            gold = counter[(value, "gold")]
            total = gold + counter[(value, "not")]
            bar = "#" * round(20 * gold / total) if total else ""
            sink(f"      {value:14s} {total:4d} cases  gold {gold:4d}  "
                 f"rate {gold / total:5.3f}  {bar}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# U3 — what blindness is actually buying, priced with the alternative set
# ─────────────────────────────────────────────────────────────────────────────

def _recorded_decisions(project, stage):
    """[(run label, {(sentence, cid): decision})] for one stage over six runs."""
    phase = {"full_name": "linker_full_name",
             "partial_name": "linker_partial_name"}[stage]
    out = []
    for model, runs in E2E_RUNS.items():
        for run in runs:
            path = (Path(run) / "phase_states" / "s_linker110" / "openai" / project
                    / f"{phase}.pkl")
            if not path.exists():
                continue
            with path.open("rb") as handle:
                feedback = pickle.load(handle)["feedback"]
            kept = {(int(row["sentence"]), row["component"])
                    for row in feedback.get("accepted", [])}
            out.append((f"{model}:{Path(run).name}", kept))
    return out


def u3_rows(projects, sink):
    """How today's two judges treat each evidence row — what a union must reproduce."""
    buckets = defaultdict(Counter)
    rows = []
    for data in projects:
        linker, gold = data["linker"], data["gold"]
        components, aliases = data["components"], data["aliases"]
        sent_map = {s.number: s for s in data["sentences"]}
        by_id = {c.id: c for c in components}
        full, partial = arm_full(data), arm_partial(data)
        recorded = {stage: _recorded_decisions(data["name"], stage)
                    for stage in ("full_name", "partial_name")}
        for pair in sorted(full | partial):
            component = by_id[pair[1]]
            evidence = case_evidence(linker, sent_map[pair[0]], component, components,
                                     sent_map, aliases)
            stage = "full_name" if pair in full else "partial_name"
            kept = sum(1 for _, keep in recorded[stage]
                       if (pair[0], component.name) in keep)
            runs = len(recorded[stage])
            key = (evidence["naming"],
                   "capitalized" if evidence["capitalized"] else "lowercase")
            bucket = buckets[key]
            bucket["cases"] += 1
            bucket["gold"] += int(pair in gold)
            bucket["kept"] += kept
            bucket["kept_gold"] += kept if pair in gold else 0
            bucket["kept_fp"] += kept if pair not in gold else 0
            bucket["runs"] = max(bucket["runs"], runs)
    sink("\n  every evidence row, and what today's judges do with it "
         "(six recorded runs)")
    sink(f"    {'naming':12s} {'case':12s} {'gate':13s} {'cases':>6s} {'gold':>5s} "
         f"{'base':>6s} {'kept/run':>9s} {'TP/run':>7s} {'FP/run':>7s}")
    for key in sorted(buckets):
        naming, shape = key
        bucket = buckets[key]
        runs = max(1, bucket["runs"])
        gate = "lenient" if naming in ("whole name", "alias") else "target-blind"
        sink(f"    {naming:12s} {shape:12s} {gate:13s} {bucket['cases']:6d} "
             f"{bucket['gold']:5d} {bucket['gold'] / bucket['cases']:6.3f} "
             f"{bucket['kept'] / runs:9.1f} {bucket['kept_gold'] / runs:7.1f} "
             f"{bucket['kept_fp'] / runs:7.1f}")
        rows.append({"naming": naming, "case_shape": shape, "gate": gate,
                     "cases": bucket["cases"], "gold": bucket["gold"],
                     "base_rate": round(bucket["gold"] / bucket["cases"], 3),
                     "kept_per_run": round(bucket["kept"] / runs, 1),
                     "tp_per_run": round(bucket["kept_gold"] / runs, 1),
                     "fp_per_run": round(bucket["kept_fp"] / runs, 1)})

    sink("\n  what the judge earns on each row: its verdicts against the row's own "
         "default")
    sink(f"    {'naming':12s} {'case':12s} {'approve-all':>12s} {'judge':>14s} "
         f"{'judge earns':>22s}")
    for key in sorted(buckets):
        naming, shape = key
        bucket = buckets[key]
        runs = max(1, bucket["runs"])
        all_tp, all_fp = bucket["gold"], bucket["cases"] - bucket["gold"]
        tp, fp = bucket["kept_gold"] / runs, bucket["kept_fp"] / runs
        sink(f"    {naming:12s} {shape:12s} {f'{all_tp}TP/{all_fp}FP':>12s} "
             f"{f'{tp:.1f}TP/{fp:.1f}FP':>14s} "
             f"{f'{tp - all_tp:+.1f} gold, {all_fp - fp:+.1f} FP killed':>22s}")
    sink("    'approve-all' is what the row's default alone would score — the judge's "
         "call\n    is worth the difference, and a union must keep every row where "
         "that difference is real.")
    return rows


def u3(projects, sink):
    rows = u3_rows(projects, sink)
    detail = []
    buckets = defaultdict(Counter)
    sink("\n  word-only cases, split by the alternative set the code can enumerate")
    for data in projects:
        linker, gold = data["linker"], data["gold"]
        components, aliases = data["components"], data["aliases"]
        sent_map = {s.number: s for s in data["sentences"]}
        by_id = {c.id: c for c in components}
        partial = arm_partial(data)
        if not partial:
            continue
        recorded = _recorded_decisions(data["name"], "partial_name")
        for pair in sorted(partial):
            sentence = sent_map[pair[0]]
            component = by_id[pair[1]]
            evidence = case_evidence(linker, sentence, component, components,
                                     sent_map, aliases)
            ambiguous = "ambiguous" if evidence["alternatives"] else "unique"
            is_gold = pair in gold
            kept = sum(1 for _, keep in recorded
                       if (pair[0], component.name) in keep)
            bucket = buckets[ambiguous]
            bucket["cases"] += 1
            bucket["gold"] += int(is_gold)
            bucket["kept"] += kept
            bucket["kept_gold"] += kept if is_gold else 0
            bucket["kept_fp"] += kept if not is_gold else 0
            bucket["runs"] = max(bucket["runs"], len(recorded))
            detail.append({"project": data["name"], "sentence": pair[0],
                           "component": component.name, "gold": int(is_gold),
                           "alternatives": "|".join(evidence["alternatives"]),
                           "class": ambiguous, "kept_in_runs": kept,
                           "runs": len(recorded)})
    sink(f"    {'class':12s} {'cases':>6s} {'gold':>6s} {'base rate':>10s} "
         f"{'kept/run':>9s} {'TP/run':>7s} {'FP/run':>7s}")
    for name, bucket in sorted(buckets.items()):
        runs = max(1, bucket["runs"])
        sink(f"    {name:12s} {bucket['cases']:6d} {bucket['gold']:6d} "
             f"{bucket['gold'] / max(1, bucket['cases']):10.3f} "
             f"{bucket['kept'] / runs:9.1f} {bucket['kept_gold'] / runs:7.1f} "
             f"{bucket['kept_fp'] / runs:7.1f}")
    sink("\n    'ambiguous' = the sentence carries a word of some OTHER component's "
         "name too,\n    so a judge shown the target could be shown the competing "
         "components as a fact.")
    write_csv(REPORT / "u3_wordonly.csv", detail)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# U4 — the merged prompt, built and checked against what each stage shows today
# ─────────────────────────────────────────────────────────────────────────────

#: The one rule. Every clause of it is one of the head's own, re-scoped to speak about
#: the evidence line instead of about the stream: the lenient default is conditioned on
#: `naming`, which is the fact `_states_a_name` computes, and the reject grounds are
#: `STRICTER_CLAUSE` and `QUALIFIED_CLAUSE` verbatim. No new weighing is introduced.
UNIFIED_RULES = """A link says this sentence makes an architectural claim about this component. Each case states how the sentence reaches the component, and how far you should extend it before asking for more:

- naming=whole name: the component is named here and the document treats it as part of the system. Approve by default; a mention that says nothing further about the component still counts. Reject only on a positive ground.
- naming=alias: as above, using the short form the document established for it.
- naming=word only: the sentence writes one word of the name and never the whole name. Approve only when that word is doing the naming work here, and when no component listed under alternatives is the one the sentence means.
- naming=no name: the component is not named at all; a referring expression must point to THIS component unambiguously. When uncertain, reject."""


def unified_case(index, component_name, evidence, sentence_text, previous,
                 anchors=None, shown_in=0, mode="preserving"):
    """One case of the merged prompt. Every line is a fact computed from the match.

    ``mode="preserving"`` is the union arm proper: the case carries everything its
    current stage shows it — the head's evidence line, the preceding sentence and the
    anchor sentences (written once per component per batch, `s_linker88`'s rule) —
    plus the fields that replace the two rubrics' premises. ``mode="swapped"`` is the
    second arm, registered separately: the anchor sentences give way to the recency
    fact they were there to let the model derive. Substituting evidence is a change of
    what the judge sees and is not free; it is measured apart from the union.
    """
    lines = [f'Case {index}: "{evidence["span"]}" -> {component_name}']
    if previous:
        lines.append(f'  [prev: "{previous}"]')
    lines.append(f'  "{sentence_text}"')
    facts = [f'source={evidence["source"]}', f'naming={evidence["naming"]}',
             f'span="{evidence["span"]}"']
    if evidence["mention_label"]:
        facts.append(f'mention={evidence["mention_label"]}')
    if evidence["alternatives"]:
        facts.append(f'alternatives={", ".join(evidence["alternatives"])}')
    if evidence["last_named_before"] >= 0:
        facts.append(f'named {evidence["last_named_before"]} sentences earlier')
    lines.append("  Evidence: " + ", ".join(facts))
    if mode == "preserving" and anchors:
        if shown_in:
            lines.append(f"  Anchors (confirmed refs): as shown in Case {shown_in}.")
        else:
            lines.append("  Anchors (confirmed refs):")
            lines.extend(f"    {a}" for a in anchors)
    return "\n".join(lines)


def merged_prompt(data, mode):
    """The whole merged judging prompt for one project, cases in document order."""
    linker = data["linker"]
    components, aliases = data["components"], data["aliases"]
    sent_map = {s.number: s for s in data["sentences"]}
    by_id = {c.id: c for c in components}
    full, partial = arm_full(data), arm_partial(data)
    merged = sorted(full | partial)

    cases, shown, window_numbers, gaps = [], {}, set(), 0
    for index, pair in enumerate(merged, 1):
        sentence = sent_map[pair[0]]
        component = by_id[pair[1]]
        evidence = case_evidence(linker, sentence, component, components, sent_map,
                                 aliases)
        evidence["source"] = "full_name" if pair in full else "partial_name"
        previous = sent_map[pair[0] - 1].text if pair[0] - 1 in sent_map else ""
        anchor_lines = [f"S{s.number}: {s.text}" for s in
                        sorted(sent_map.values(), key=lambda x: x.number)
                        if s.number != sentence.number
                        and linker._find_exact_form(s.text, component.name)
                        ][:SLinker110.ANCHOR_LIMIT]
        first = shown.get(component.name, 0)
        if mode == "preserving" and anchor_lines and not first:
            shown[component.name] = index
        cases.append(unified_case(index, component.name, evidence, sentence.text,
                                  previous, anchor_lines, first, mode))
        if mode == "preserving" and pair in partial:
            # The denotation prompt's own window, kept: its sentences are what that
            # judge is shown today, and no case may be shown less.
            window_numbers.update(s.number for s in
                                  linker._window(sentence.number, data["sentences"]))
        if not evidence["span"]:
            gaps += 1
    table = ""
    if window_numbers:
        table = "\nSENTENCES\n" + json.dumps(
            [{"sentence": n, "text": sent_map[n].text}
             for n in sorted(window_numbers)]) + "\n"
    prompt = (UNIFIED_RULES + table + "\n\nCASES\n" + "\n".join(cases)
              + '\n\nJSON only:\n{"validations":[{"case":1,'
                '"claim":"exact source quote","approve":true}]}\n')
    return prompt, len(cases), gaps, cases


def u4(projects, sink):
    rows = []
    sink("\n  the merged prompt, built for real batches, in two modes")
    sink(f"    {'project':15s} {'cases':>6s} {'preserving B':>13s} "
         f"{'swapped B':>10s} {'gaps':>5s}")
    totals = Counter()
    samples = []
    for data in projects:
        preserving, cases, gaps, case_list = merged_prompt(data, "preserving")
        swapped, _, _, _ = merged_prompt(data, "swapped")
        sink(f"    {data['name']:15s} {cases:6d} {len(preserving):13d} "
             f"{len(swapped):10d} {gaps:5d}")
        totals.update({"cases": cases, "preserving": len(preserving),
                       "swapped": len(swapped), "gaps": gaps})
        rows.append({"project": data["name"], "cases": cases,
                     "preserving_bytes": len(preserving),
                     "swapped_bytes": len(swapped), "evidence_gaps": gaps})
        if data["name"] == "bigbluebutton":
            samples = ([c for c in case_list if "naming=word only" in c][:2]
                       + [c for c in case_list if "naming=whole name" in c][:1])
    sink(f"    {'TOTAL':15s} {totals['cases']:6d} {totals['preserving']:13d} "
         f"{totals['swapped']:10d} {totals['gaps']:5d}")
    rows.append({"project": "TOTAL", "cases": totals["cases"],
                 "preserving_bytes": totals["preserving"],
                 "swapped_bytes": totals["swapped"],
                 "evidence_gaps": totals["gaps"]})
    if samples:
        sink("\n    sample cases (bigbluebutton, preserving mode):")
        for case in samples:
            for line in case.splitlines():
                sink(f"      {line[:108]}")
            sink("")

    sent = _bytes_sent_today()
    if sent:
        sink("  against what the two stages actually send today "
             "(mean over six recorded runs)")
        sink(f"    {'project':15s} {'full-name':>10s} {'denotation':>11s} "
             f"{'today':>9s} {'preserving':>11s} {'delta':>8s} {'swapped':>9s} "
             f"{'delta':>8s}")
        run_totals = Counter()
        for data in projects:
            full_b, part_b = sent.get(data["name"], (0.0, 0.0))
            row = next(r for r in rows if r["project"] == data["name"])
            today = full_b + part_b
            run_totals.update({"today": today, "preserving": row["preserving_bytes"],
                               "swapped": row["swapped_bytes"]})
            sink(f"    {data['name']:15s} {full_b:10.0f} {part_b:11.0f} "
                 f"{today:9.0f} {row['preserving_bytes']:11d} "
                 f"{row['preserving_bytes'] - today:+8.0f} "
                 f"{row['swapped_bytes']:9d} {row['swapped_bytes'] - today:+8.0f}")
        sink(f"    {'TOTAL':15s} {'':10s} {'':11s} {run_totals['today']:9.0f} "
             f"{run_totals['preserving']:11.0f} "
             f"{run_totals['preserving'] - run_totals['today']:+8.0f} "
             f"{run_totals['swapped']:9.0f} "
             f"{run_totals['swapped'] - run_totals['today']:+8.0f}")
    return rows


JUDGE_PHASES = {"phase_25_full_name_judge": "full",
                "phase_25_partial_denotation": "partial"}


def _bytes_sent_today():
    """{project: (full-name judge bytes, denotation bytes)} averaged over the runs."""
    sums = defaultdict(Counter)
    runs = Counter()
    for _model, paths in E2E_RUNS.items():
        for run in paths:
            logs = sorted(Path(run).glob("llm_logs/s_linker110_openai_*_calls.json"))
            if not logs:
                continue
            for path in logs:
                project = path.name.split("_openai_")[1].rsplit("_", 3)[0]
                with path.open() as handle:
                    calls = json.load(handle)
                seen = False
                for call in calls:
                    kind = JUDGE_PHASES.get(call.get("phase"))
                    if kind:
                        sums[project][kind] += len(call.get("prompt", ""))
                        seen = True
                if seen:
                    runs[project] += 1
    return {project: (sums[project]["full"] / max(1, runs[project]),
                      sums[project]["partial"] / max(1, runs[project]))
            for project in sums}


# ─────────────────────────────────────────────────────────────────────────────
# U5 — the plan
# ─────────────────────────────────────────────────────────────────────────────

def u5(projects, sink):
    batch = SLinker110.JUDGE_BATCH
    total_full = total_partial = 0
    calls_now = calls_merged = 0
    for data in projects:
        full, partial = len(arm_full(data)), len(arm_partial(data))
        total_full += full
        total_partial += partial
        calls_now += -(-full // batch) + -(-partial // batch)
        calls_merged += -(-(full + partial) // batch)
    sink(f"\n    today   {total_full} + {total_partial} cases in {calls_now} calls "
         f"(two prompts, two rubrics, two reply schemas)")
    sink(f"    merged  {total_full + total_partial} cases in {calls_merged} calls "
         f"(one prompt, one rubric, one reply schema)")
    sink("\n    the pilot this arm owes, before any E2E:")
    sink("      stage arm on fixed recorded candidates, 3 samples a side, both models,")
    sink("      both arms in one invocation; read gold kept and spurious kept per")
    sink("      stream separately — a union that trades the streams against each other")
    sink("      reads neutral in the total and is not neutral.")
    return [{"arrangement": "today", "cases": total_full + total_partial,
             "calls": calls_now, "prompts": 2},
            {"arrangement": "merged", "cases": total_full + total_partial,
             "calls": calls_merged, "prompts": 1}]


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*",
                        default=["U1", "U2", "U3", "U4", "U5"])
    args = parser.parse_args()
    lines = []

    def sink(line=""):
        print(line)
        lines.append(line)

    sink("UNIJUDGE AUDIT — one rule, one reply, evidence computed from the match")
    projects = [load_project(project, True) for project in PROJECTS]
    REPORT.mkdir(parents=True, exist_ok=True)

    if "U1" in args.only:
        sink("\nU1 — the census of differences")
        write_csv(REPORT / "u1_axes.csv", u1(sink))
    if "U2" in args.only:
        sink("\nU2 — the evidence a merged case can carry")
        write_csv(REPORT / "u2_evidence.csv", u2(projects, sink))
    if "U3" in args.only:
        sink("\nU3 — what blindness buys, against the alternative set")
        write_csv(REPORT / "u3_blindness.csv", u3(projects, sink))
    if "U4" in args.only:
        sink("\nU4 — the merged prompt")
        write_csv(REPORT / "u4_prompt.csv", u4(projects, sink))
    if "U5" in args.only:
        sink("\nU5 — cost and the pilot owed")
        write_csv(REPORT / "u5_plan.csv", u5(projects, sink))

    (REPORT / "audit.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwritten: {REPORT}/audit.txt and the CSVs beside it")


if __name__ == "__main__":
    main()
