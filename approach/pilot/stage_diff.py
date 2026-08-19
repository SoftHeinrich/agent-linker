"""Where two arms of the same pipeline diverge, stage by stage and link by link.

Scoring says an arm lost 1.5 F1; it never says where. This reads any set of paired
run directories and reports, for every arm, the size of each stage's population and
the fate of every link the arms do not share — joined to the gold standard and
attributed to the linker that produced it.

The cascade is what makes this necessary: the three linkers subtract from one
another, so a stage that gains candidates can *cost* true positives downstream (the
s46 result: +16 candidates, -2.0 TP), and a stage arm cannot see it. The unit that
explains a score difference is therefore the link, tagged with the stage that made
it and the stage that could have made it instead.

    ../.venv/bin/python pilot/stage_diff.py \
        --arms s_linker49 s_linker50 s_linker51 \
        --runs ../results/s5051_e2e_r*_20260813

Use `--links` to print every differing link rather than the per-source summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold, load_project          # noqa: E402

_INFO: dict = {}


def _project_info(project):
    """Sentences and component names, loaded once."""
    if project not in _INFO:
        info = load_project(project)
        info["id_to_name"] = {c.id: c.name for c in info["components"]}
        _INFO[project] = info
    return _INFO[project]

STAGES = ("knowledge", "linker_full_name", "linker_partial_name",
          "linker_coreference")


def phase(run: Path, arm: str, project: str, name: str):
    path = run / "phase_states" / arm / "openai" / project / f"{name}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def links_of(run: Path, arm: str, project: str):
    """Predicted links of one arm on one project, with the source tag."""
    path = run / f"{arm}_{project}_links.csv"
    if not path.exists():
        return None
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out[(int(row["sentence"]), row["component_id"])] = row.get("source", "?")
    return out


def alias_table(run: Path, arm: str, project: str):
    state = phase(run, arm, project, "knowledge")
    if not state:
        return {}
    knowledge = state.get("doc_knowledge")
    table = getattr(knowledge, "aliases", None)
    if isinstance(knowledge, dict):
        table = knowledge.get("aliases")
    return dict(table or {})


def stage_sizes(run: Path, arm: str, project: str):
    """One row of population counts per stage, for one arm on one project."""
    row = Counter()
    row["aliases"] = len(alias_table(run, arm, project))
    for stage, prefix in (("linker_full_name", "full"),
                          ("linker_partial_name", "part"),
                          ("linker_coreference", "coref")):
        state = phase(run, arm, project, stage)
        if not state:
            continue
        feedback = state.get("feedback", {})
        proposed = feedback.get("candidates") or feedback.get("proposed") or []
        row[f"{prefix}_proposed"] = len(proposed)
        row[f"{prefix}_judged"] = len(feedback.get("judge_decisions", []))
        row[f"{prefix}_accepted"] = len(feedback.get("accepted", []))
        row[f"{prefix}_links"] = len(state.get("links", []))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--alias-trace", action="store_true",
                        help="split the alias stage into proposer and judge, "
                             "off the call traces")
    parser.add_argument("--judge-trace", action="store_true",
                        help="compare judging verdicts on the candidates both arms "
                             "judged, holding the proposer constant")
    parser.add_argument("--prompt-identity", action="store_true",
                        help="check the arms sent the same bytes where the input is "
                             "determined by the document alone")
    parser.add_argument("--links", action="store_true",
                        help="print every differing link, not just the summary")
    args = parser.parse_args()
    runs = [Path(r) for r in args.runs]
    base, others = args.arms[0], args.arms[1:]

    # ── stage populations ────────────────────────────────────────────────────
    print(f"\n=== STAGE POPULATIONS (mean per run over {len(runs)} runs)\n")
    keys = ["aliases", "full_proposed", "full_judged", "full_accepted",
            "part_proposed", "part_judged", "part_accepted",
            "coref_proposed", "coref_judged", "coref_accepted"]
    for project in PROJECTS:
        print(f"  {project}")
        print(f"    {'arm':14s}" + "".join(f"{k.replace('_', ' '):>16s}" for k in keys))
        for arm in args.arms:
            total, seen = Counter(), 0
            for run in runs:
                row = stage_sizes(run, arm, project)
                if row:
                    total.update(row)
                    seen += 1
            if not seen:
                continue
            print(f"    {arm:14s}" +
                  "".join(f"{total[k] / seen:16.1f}" for k in keys))

    # ── link-level difference, attributed ────────────────────────────────────
    print(f"\n=== LINKS NOT SHARED WITH {base} (mean per run, by source and fate)\n")
    for arm in others:
        gained, lost = Counter(), Counter()
        gained_links, lost_links = Counter(), Counter()
        seen = 0
        for run in runs:
            complete = True
            per_run = {}
            for project in PROJECTS:
                a, b = links_of(run, base, project), links_of(run, arm, project)
                if a is None or b is None:
                    complete = False
                    break
                per_run[project] = (a, b)
            if not complete:
                continue
            seen += 1
            for project, (a, b) in per_run.items():
                gold = load_gold(project)
                for key in set(b) - set(a):
                    fate = "gold" if key in gold else "spurious"
                    gained[(b[key], fate)] += 1
                    gained_links[(project, key[0], b[key], fate)] += 1
                for key in set(a) - set(b):
                    fate = "gold" if key in gold else "spurious"
                    lost[(a[key], fate)] += 1
                    lost_links[(project, key[0], a[key], fate)] += 1
        if not seen:
            print(f"  {arm}: no complete paired run")
            continue
        print(f"  {arm} vs {base}   ({seen} paired runs)")
        print(f"    {'':6s} {'source':22s} {'gold':>8s} {'spurious':>10s}")
        sources = {s for s, _ in list(gained) + list(lost)}
        for source in sorted(sources):
            print(f"    {'GAINED':6s} {source:22s} "
                  f"{gained[(source, 'gold')] / seen:8.1f} "
                  f"{gained[(source, 'spurious')] / seen:10.1f}")
            print(f"    {'LOST':6s} {source:22s} "
                  f"{lost[(source, 'gold')] / seen:8.1f} "
                  f"{lost[(source, 'spurious')] / seen:10.1f}")
        net_tp = (sum(v for (s, f), v in gained.items() if f == "gold")
                  - sum(v for (s, f), v in lost.items() if f == "gold")) / seen
        net_fp = (sum(v for (s, f), v in gained.items() if f == "spurious")
                  - sum(v for (s, f), v in lost.items() if f == "spurious")) / seen
        print(f"    NET   TP {net_tp:+.1f}   FP {net_fp:+.1f}  per run\n")

        if args.links:
            print(f"    every differing link ({arm}), most persistent first:")
            for label, table in (("GAINED", gained_links), ("LOST", lost_links)):
                for (project, snum, source, fate), times in table.most_common(40):
                    print(f"      {label:6s} {project:14s} S{snum:<4d} {source:18s} "
                          f"{fate:9s} in {times}/{seen} runs")
            print()

    # ── are the differing links explained by the alias table? ────────────────
    print("=== ATTRIBUTION: differing links whose sentence states no component name,\n"
          "    only a term from that arm's own alias table\n")
    for arm in others:
        counts = Counter()
        witness = Counter()
        seen = 0
        for run in runs:
            ok = True
            for project in PROJECTS:
                if links_of(run, base, project) is None or \
                        links_of(run, arm, project) is None:
                    ok = False
            if not ok:
                continue
            seen += 1
            for project in PROJECTS:
                info = _project_info(project)
                gold = load_gold(project)
                a, b = links_of(run, base, project), links_of(run, arm, project)
                tables = {base: alias_table(run, base, project),
                          arm: alias_table(run, arm, project)}
                for label, own, other_side in (("GAINED", arm, base),
                                               ("LOST", base, arm)):
                    keys = set(b) - set(a) if label == "GAINED" else set(a) - set(b)
                    for snum, cid in keys:
                        name = info["id_to_name"].get(cid, "")
                        text = info["sent_map"].get(snum)
                        text = text.text if text is not None else ""
                        states_name = name and name.casefold() in text.casefold()
                        terms = [t for t, target in tables[own].items()
                                 if str(target) == name and t.casefold() in text.casefold()]
                        exclusive = [t for t in terms if t not in tables[other_side]]
                        fate = "gold" if (snum, cid) in gold else "spurious"
                        if states_name:
                            counts[(label, "name stated", fate)] += 1
                        elif exclusive:
                            counts[(label, "arm-only alias", fate)] += 1
                            witness[(project, exclusive[0], name, fate)] += 1
                        elif terms:
                            counts[(label, "shared alias", fate)] += 1
                        else:
                            counts[(label, "neither", fate)] += 1
        if not seen:
            continue
        print(f"  {arm} vs {base}   ({seen} paired runs)")
        for label in ("GAINED", "LOST"):
            for why in ("name stated", "arm-only alias", "shared alias", "neither"):
                gold = counts[(label, why, "gold")] / seen
                spurious = counts[(label, why, "spurious")] / seen
                if gold or spurious:
                    print(f"    {label:6s} {why:16s} gold {gold:5.1f}   "
                          f"spurious {spurious:5.1f}")
        if witness:
            print(f"    alias terms only this arm's table has, and what they link:")
            for (project, term, name, fate), times in witness.most_common(15):
                print(f"      {project:14s} {term!r} -> {name} ({fate}) "
                      f"x{times / seen:.1f}/run")
        print()

    # ── alias tables, which admit and suppress ───────────────────────────────
    print("=== ALIAS TABLES (terms present in one arm and not the other)\n")
    for project in PROJECTS:
        per_arm = defaultdict(Counter)
        for run in runs:
            for arm in args.arms:
                for term in alias_table(run, arm, project):
                    per_arm[arm][term] += 1
        if not per_arm:
            continue
        base_terms = set(per_arm[base])
        for arm in others:
            only_arm = sorted(set(per_arm[arm]) - base_terms)
            only_base = sorted(base_terms - set(per_arm[arm]))
            if only_arm or only_base:
                print(f"  {project:14s} {arm}: only-here {only_arm} | "
                      f"only-in-{base} {only_base}")

    if args.alias_trace:
        audit_alias_judge(args.arms, runs)
    if args.judge_trace:
        audit_judge_flips(args.arms, runs)
    if args.prompt_identity:
        audit_prompt_identity(args.arms, runs)


def audit_alias_judge(arms, runs):
    """Proposer or judge? The same question, asked of the alias stage's two calls.

    A bigger alias table can come from either end, and the two have opposite
    implications: a looser *proposer* means the extraction prompt lost a
    constraint, a looser *judge* means the rubric did. Separating them needs the
    traces, because the checkpoint only records the table that survived both.

    The decisive population is the terms **both arms propose**: for those, the
    proposer is held constant by construction and any difference in the table is
    the judge's.
    """
    print("\n=== ALIAS STAGE — proposer against judge, per term\n")
    proposed = defaultdict(Counter)
    approved = defaultdict(Counter)
    for arm in arms:
        for run in runs:
            for path in (run / "llm_logs").glob(f"{arm}_openai_*_calls.json"):
                project = path.name.split("_openai_")[1].rsplit("_", 3)[0]
                with path.open() as handle:
                    calls = json.load(handle)
                for call in calls:
                    data = _response_json(call.get("response_text"))
                    if call.get("phase") == "phase_25_doc_extract":
                        for key in ("abbreviations", "synonyms"):
                            for item in (data or {}).get(key, []) or []:
                                term = str(item.get("term", "")).strip().lower()
                                if term:
                                    proposed[arm][(project, term)] += 1
                    elif call.get("phase") == "phase_25_doc_judge":
                        for term in (data or {}).get("approved", []) or []:
                            term = str(term).strip().lower()
                            if term:
                                approved[arm][(project, term)] += 1
    n = len(runs)
    for arm in arms:
        total_p = sum(proposed[arm].values()) / n
        total_a = sum(approved[arm].values()) / n
        print(f"  {arm:16s} proposed {total_p:5.1f}/run   approved {total_a:5.1f}/run"
              f"   approval rate {100 * total_a / max(total_p, 1e-9):5.1f}%")

    base = arms[0]
    for arm in arms[1:]:
        shared = set(proposed[base]) & set(proposed[arm])
        flipped = []
        for key in sorted(shared):
            rate_base = approved[base][key] / max(proposed[base][key], 1)
            rate_arm = approved[arm][key] / max(proposed[arm][key], 1)
            if abs(rate_arm - rate_base) >= 0.5:
                flipped.append((key, proposed[base][key], approved[base][key],
                                proposed[arm][key], approved[arm][key]))
        print(f"\n  terms BOTH arms propose whose approval flips ({arm} vs {base}) — "
              f"the proposer is held constant here, so this is the judge:")
        if not flipped:
            print("    none")
        for (project, term), pb, ab, pa, aa in flipped:
            print(f"    {project:14s} {term:26s} {base}: {ab}/{pb} approved   "
                  f"{arm}: {aa}/{pa} approved")


def audit_judge_flips(arms, runs, stage="linker_full_name"):
    """Same candidate, different verdict: the judging rubric held to account.

    Candidate sets differ between arms, so a raw approval-rate comparison mixes the
    proposer's behaviour into the judge's. Restricting to the (sentence, component)
    pairs *both* arms judged in the same run holds the proposer constant, and the
    remaining difference is the rubric's.
    """
    print(f"\n=== JUDGING RUBRIC — verdict flips on shared candidates ({stage})\n")
    base = arms[0]
    for arm in arms[1:]:
        counts = Counter()
        flips = Counter()
        for run in runs:
            for project in PROJECTS:
                a = phase(run, base, project, stage)
                b = phase(run, arm, project, stage)
                if not a or not b:
                    continue
                gold = load_gold(project)
                verdict_a = {(d["sentence"], d["component_id"]): bool(d["approved"])
                             for d in a["feedback"]["judge_decisions"]}
                verdict_b = {(d["sentence"], d["component_id"]): bool(d["approved"])
                             for d in b["feedback"]["judge_decisions"]}
                for key in set(verdict_a) & set(verdict_b):
                    is_gold = key in gold
                    counts["shared"] += 1
                    if verdict_a[key] == verdict_b[key]:
                        counts["agree"] += 1
                        continue
                    label = ("approved only by " +
                             (arm if verdict_b[key] else base))
                    counts[label] += 1
                    counts[label + (" [gold]" if is_gold else " [spurious]")] += 1
                    info = _project_info(project)
                    flips[(project, key[0],
                           info["id_to_name"].get(key[1], key[1]),
                           label, "gold" if is_gold else "spurious")] += 1
        n = len(runs)
        print(f"  {arm} vs {base}: {counts['shared'] / n:.1f} shared candidates/run, "
              f"{counts['agree'] / n:.1f} same verdict")
        for who in (arm, base):
            label = f"approved only by {who}"
            print(f"    {label:34s} {counts[label] / n:5.1f}/run  "
                  f"gold {counts[label + ' [gold]'] / n:4.1f}  "
                  f"spurious {counts[label + ' [spurious]'] / n:4.1f}")
        if flips:
            print("    most persistent flips:")
            for (project, snum, name, label, fate), times in flips.most_common(12):
                print(f"      {project:14s} S{snum:<4d} {name:24s} {label:28s} "
                      f"{fate:9s} {times}/{len(runs)} runs")


def audit_prompt_identity(arms, runs):
    """Did the arms actually send the same bytes where they should have?

    For a null arm the answer must be yes everywhere the input is determined by the
    document alone — the first call of each phase on each project. If it is yes and
    the outputs still differ systematically, the difference is the provider's, not
    the linker's, and no amount of re-reading the code will find it.
    """
    print("\n=== PROMPT IDENTITY — first call of each phase, per project\n")
    print("  Only `doc_extract` is determined by the document and the component list "
          "alone.\n  Every later phase reads an earlier phase's output, so once the "
          "first call of the\n  run diverges the rest is expected to.\n")
    base = arms[0]
    for arm in arms[1:]:
        counts = Counter()
        for run in runs:
            first = defaultdict(dict)
            for side in (base, arm):
                for path in (run / "llm_logs").glob(f"{side}_openai_*_calls.json"):
                    project = path.name.split("_openai_")[1].rsplit("_", 3)[0]
                    with path.open() as handle:
                        for call in json.load(handle):
                            key = (project, call["phase"])
                            first[side].setdefault(key, call["prompt"])
            for key in set(first[base]) & set(first[arm]):
                same = first[base][key] == first[arm][key]
                counts[(key[1], "compared")] += 1
                counts[(key[1], "identical")] += same
        print(f"  {arm} vs {base}:")
        for phase_name in sorted({k[0] for k in counts}):
            print(f"    {phase_name.replace('phase_25_', ''):26s} "
                  f"{counts[(phase_name, 'identical')]:3d} of "
                  f"{counts[(phase_name, 'compared')]:3d} byte-identical")


def _response_json(text):
    if not text:
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()
