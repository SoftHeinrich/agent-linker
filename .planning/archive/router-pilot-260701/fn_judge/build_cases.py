#!/usr/bin/env python3
"""Build the labelled judge-case universe for the FN-recovery experiment.

The judge input (sentence + component + context) is RUN-INDEPENDENT, so we dedup
distinct (project, sentence, component_id) cases across the 3 gpt-5.4 s21 runs.

Labels (mutually exclusive, gold-referenced):
  R-TP     rejected by the s21 validator in >=1 run, AND in gold  -> recall-recovery target
  R-TN     rejected by the s21 validator in >=1 run, NOT in gold  -> precision control (real distractors)
  NP-FN    in gold, never proposed by extraction OR coref in ANY run -> ceiling test
  NP-CTRL  constructed sibling distractor (not gold) for an NP-FN sentence -> ceiling precision control

A case that is BOTH rejected in some run AND never-proposed in others is labelled by gold:
rejected-in-any-run dominates (the validator did see it), so R-TP/R-TN take precedence over NP.

Context attached to each case (all run-independent, from the benchmark + extract rosters):
  sentence, preceding sentence, up to 4 ANCHOR sentences where the component is affirmatively
  named elsewhere in the doc, matched_text/mention_type (if it was ever a candidate),
  coref antecedent (if it was ever a coref candidate), gold flag, source stage.

Writes cases.json next to this script.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ARD = Path("/mnt/hostshare/ardoco-home")
BENCH = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    ARD / "ardoco/core/tests-base/src/main/resources/benchmark"))
EXTRACTS = ARD / "agent-linker/results/v2.6.6_extracts_s21/gpt"
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
GS_SAD_SAM = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}
HERE = Path(__file__).resolve().parent


def load_gold(project):
    gold = set()
    with (BENCH / GS_SAD_SAM[project]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gold.add((int(r["sentence"]), r["modelElementID"]))
    return gold


def sentences(project):
    hits = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt"))
    d = {}
    if hits:
        with open(hits[0], encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f, 1):
                d[i] = ln.strip()
    return d


def load_extracts(project):
    return {run: json.load(open(EXTRACTS / run / f"{project}.json")) for run in RUNS}


def standalone(name, text):
    """affirmative standalone mention of `name` in `text` (word-boundary, case-insensitive
    for multiword / proper; used only for anchor discovery)."""
    if not name:
        return False
    return re.search(rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])', text,
                     re.IGNORECASE) is not None


def main():
    cases = []
    for proj in PROJECTS:
        gold = load_gold(proj)
        sn = sentences(proj)
        ex = load_extracts(proj)

        # component id -> display name, and all component names in project
        id2name = {}
        ambiguous = set()   # component NAMES s21 flagged ambiguous (union over runs)
        for run in RUNS:
            for grp in (ex[run]["entity"]["candidates"], ex[run]["final"]["links"],
                        ex[run]["coref"]["raw"]):
                for it in grp:
                    if it.get("c") and it.get("component_name"):
                        id2name.setdefault(it["c"], it["component_name"])
            for nm in ex[run]["knowledge"]["model_knowledge"]["ambiguous_names"]:
                ambiguous.add(nm)
        # gold component ids might not appear in any roster -> recover names from PCM model
        for gid, gs in [(c, s) for (s, c) in gold]:
            id2name.setdefault(gid, None)
        _fill_names_from_model(proj, id2name)
        all_names = sorted({n for n in id2name.values() if n})

        # per-(s,cid) aggregates across runs
        ever_cand = defaultdict(dict)      # (s,cid) -> {matched_text, mention_type}
        ever_coref = defaultdict(dict)     # (s,cid) -> {antecedent_sentence, antecedent_text, reference}
        rejected_any = set()               # (s,cid) rejected by validator in >=1 run
        proposed_any = set()               # (s,cid) proposed (entity cand OR coref raw) in >=1 run
        for run in RUNS:
            d = ex[run]
            ev = {(e["s"], e["c"]) for e in d["entity"]["validated"]}
            cv = {(e["s"], e["c"]) for e in d["coref"]["validated"]}
            for e in d["entity"]["candidates"]:
                k = (e["s"], e["c"]); proposed_any.add(k)
                ever_cand[k] = {"matched_text": e.get("matched_text"),
                                "mention_type": e.get("mention_type")}
                if k not in ev:
                    rejected_any.add(k)
            for r in d["coref"]["raw"]:
                k = (r["s"], r["c"]); proposed_any.add(k)
                if k not in cv:
                    rejected_any.add(k)
            for m in d["coref"]["metadata"]:
                k = (m["s"], m["c"])
                ever_coref[k] = {"antecedent_sentence": m.get("antecedent_sentence"),
                                 "antecedent_text": m.get("antecedent_text"),
                                 "reference": m.get("reference")}

        def anchors(cid, s):
            name = id2name.get(cid)
            if not name:
                return []
            out = []
            for i in sorted(sn):
                if i == s:
                    continue
                if standalone(name, sn[i]):
                    out.append(f"S{i}: {sn[i]}")
                if len(out) >= 4:
                    break
            return out

        def mkcase(s, cid, label):
            name = id2name.get(cid)
            return {
                "id": f"{proj}|{s}|{cid}",
                "project": proj, "sentence_num": s, "component_id": cid,
                "component": name, "label": label,
                "is_ambiguous": (name in ambiguous) if name else False,
                "sentence": sn.get(s, ""),
                "preceding": sn.get(s - 1, ""),
                "anchors": anchors(cid, s),
                "matched_text": ever_cand.get((s, cid), {}).get("matched_text"),
                "mention_type": ever_cand.get((s, cid), {}).get("mention_type"),
                "coref": ever_coref.get((s, cid)),
                "gold": (s, cid) in gold,
                "proposed": (s, cid) in proposed_any,
                "rejected": (s, cid) in rejected_any,
            }

        # 1) rejected candidates (R-TP / R-TN)
        for (s, cid) in sorted(rejected_any):
            label = "R-TP" if (s, cid) in gold else "R-TN"
            cases.append(mkcase(s, cid, label))

        # 2) never-proposed FN (gold, never proposed in any run)
        np_fn = [(s, cid) for (s, cid) in gold if (s, cid) not in proposed_any]
        np_sentences = sorted({s for (s, cid) in np_fn})
        for (s, cid) in sorted(np_fn):
            cases.append(mkcase(s, cid, "NP-FN"))

        # 3) NP-CTRL: for each NP-FN sentence, up to 3 sibling components (not gold, not
        #    already a case for that sentence) that ARE affirmatively named nearby or are
        #    confusable siblings -> hard-ish distractors for the ceiling precision control.
        existing = {(c["sentence_num"], c["component"]) for c in cases if c["project"] == proj}
        for s in np_sentences:
            picked = 0
            # prefer components whose name appears in the sentence or an adjacent one but are NOT gold
            window = " ".join(sn.get(i, "") for i in (s - 1, s, s + 1))
            ranked = sorted(all_names,
                            key=lambda n: (0 if standalone(n, window) else 1, len(n)))
            for name in ranked:
                cid = next((k for k, v in id2name.items() if v == name), None)
                if cid is None:
                    continue
                if (s, cid) in gold or (s, name) in existing:
                    continue
                c = mkcase(s, cid, "NP-CTRL")
                cases.append(c); existing.add((s, name)); picked += 1
                if picked >= 3:
                    break

    (HERE / "cases.json").write_text(json.dumps(cases, indent=1))
    from collections import Counter
    cnt = Counter(c["label"] for c in cases)
    print(f"wrote {len(cases)} distinct judge cases -> cases.json")
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        print(f"  {lab:<8} {cnt.get(lab,0)}")


def _fill_names_from_model(project, id2name):
    """id -> entityName from the PCM/UML model XML (for gold ids absent from rosters)."""
    need = {k for k, v in id2name.items() if not v}
    if not need:
        return
    for mp in glob.glob(str(BENCH / project / "model_*" / "**" / "*"), recursive=True):
        if not os.path.isfile(mp):
            continue
        try:
            txt = open(mp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for m in re.finditer(r'id="(_[^"]+)"[^>]*entityName="([^"]*)"', txt):
            if m.group(1) in need and m.group(2):
                id2name[m.group(1)] = m.group(2)
        for m in re.finditer(r'entityName="([^"]*)"[^>]*id="(_[^"]+)"', txt):
            if m.group(2) in need and m.group(1):
                id2name[m.group(2)] = m.group(1)


if __name__ == "__main__":
    main()
