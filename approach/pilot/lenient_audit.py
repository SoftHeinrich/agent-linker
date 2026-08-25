"""Level-1 census of the lenient gate's own errors at the head. No LLM calls.

`pilot/judge_census.py` puts the largest F2-weighted headroom in the pipeline at the
full-name gate -- 17.3 false positives a run on terra and 41.3 on luna, from the same
prompt over the same deterministic scan. This asks what those cases are, using only
properties a *fact* could carry: what the sentence actually writes, how it writes it,
and whether the words are ordinary English by a general lexicon.

Every bucket here is code-computable at runtime and none is a list. `dictionary` reads
WordNet, the general English resource the module already depends on for morphology; no
term of any benchmark appears in this file (GATE-06).

    ../.venv/bin/python pilot/lenient_audit.py [--variant s_linker110]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from nltk.corpus import wordnet  # noqa: E402

from chooser_audit import runs_of  # noqa: E402
from consolidation_audit import (  # noqa: E402
    SCAN, aliases_of, load_projects, model_of, stage_of,
)
from regex_extract_audit import signature, spans  # noqa: E402


def surfaces_of(text, name, aliases):
    """Every way this sentence writes the component, at the scan's own fidelity."""
    forms = [name] + [term for term, comp in aliases.items() if comp == name]
    found = []
    for form in forms:
        for start, end in spans(text, form, SCAN["form"]):
            found.append(text[start:end])
    return found


def ordinary(name) -> bool:
    """Every word of the name is ordinary English, by WordNet and nothing else."""
    words = signature(name)
    return bool(words) and all(wordnet.synsets(word) for word in words)


def buckets(text, name, aliases, other_names):
    found = surfaces_of(text, name, aliases)
    if not found:
        return None
    return {
        "lowercased": all(s == s.lower() for s in found) and name != name.lower(),
        "dictionary": ordinary(name),
        "one word": len(signature(name)) == 1,
        "via alias": not spans(text, name, SCAN["form"]),
        "co-named": any(spans(text, other, SCAN["form"]) for other in other_names),
    }


def audit(variant: str):
    projects = load_projects()
    rows = []
    for base in runs_of(variant):
        row = {"run": base.parts[2], "model": model_of(base)}
        counts = defaultdict(int)
        for project, data in projects.items():
            gold = data["gold"]
            aliases = aliases_of(base, project)
            names = list(data["name_to_id"])
            sentences = {s.number: s.text for s in data["sentences"]}
            stage = stage_of(base, project, "linker_full_name")
            for decision in stage["feedback"].get("judge_decisions", []):
                if not decision.get("approved"):
                    continue
                pair = (decision["sentence"], decision["component_id"])
                name = data["id_to_name"].get(decision["component_id"], "")
                text = sentences.get(decision["sentence"], "")
                marks = buckets(text, name, aliases,
                                [n for n in names if n != name])
                tag = "TP" if pair in gold else "FP"
                counts[tag] += 1
                if marks is None:
                    counts[f"{tag} no surface"] += 1
                    continue
                for key, hit in marks.items():
                    counts[f"{tag} {key}"] += hit
        row.update(counts)
        rows.append(row)
    return rows


BUCKETS = ("lowercased", "dictionary", "one word", "via alias", "co-named",
           "no surface")


def report(rows):
    if not rows:
        print("no recorded runs")
        return
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        mean = lambda k: statistics.mean(r.get(k, 0) for r in group)  # noqa: E731
        tp, fp = mean("TP"), mean("FP")
        print(f"\n{model}, mean per five-project run ({len(group)} runs)"
              f" -- kept {tp:.1f} TP, {fp:.1f} FP")
        print(f"  {'bucket':<14}{'TP in':>8}{'FP in':>8}{'FP share':>10}"
              f"{'precision in bucket':>22}")
        for key in BUCKETS:
            t, f = mean(f"TP {key}"), mean(f"FP {key}")
            share = f / fp if fp else 0.0
            precision = t / (t + f) if (t + f) else float("nan")
            print(f"  {key:<14}{t:>8.1f}{f:>8.1f}{share:>10.0%}{precision:>22.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="s_linker110")
    parser.add_argument("--json")
    args = parser.parse_args()
    rows = audit(args.variant)
    report(rows)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
