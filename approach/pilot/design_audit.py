"""Deterministic audit of the four paper/code mismatches in the s25 workflow.

No LLM call is made. Every number here is read off an existing promoted run
(`--run`, default the 2026-08-10 cleanup-verify five-project run): its per-linker
phase states and its per-call trace. The point is to size each proposed design
change *before* paying for an A/B pilot, because a change whose deterministic
delta is zero on all five projects needs no pilot at all -- it cannot alter a
decision.

The four mismatches, as reported by the workflow audit:

  1. SEQUENCE   -- the paper and the module docstring both say each linker sees
     only what the earlier ones left unlinked. Only the partial-name linker
     subtracts. Question: how many coreference proposals restate a pair an
     earlier linker already produced?
  2. ALIAS SCOPE -- the paper says only global aliases are offered to the
     full-name linker. True of the extraction prompt; false of the lexical
     admission gate, of the coreference antecedent gate, and of every other
     alias consumer, which all read the alias table without its scope field.
     Question: how many candidates and antecedents depend on a local alias?
  3. CLAIM CHECK -- every judge prompt asks the model to quote the words it
     ruled on. The partial-name judge verifies the quote; the full-name and
     coreference judges parse only `approve` and drop the quote. Question: how
     often would a substring check have voided a verdict?
  4. AMBIGUITY MAP -- the model-understanding call produces a set of ambiguous
     names whose only consumer is one boolean field of the evidence bundle.
     Question: how many candidates ever carry that field set?

Usage:
    ../.venv/bin/python pilot/design_audit.py            # all four
    ../.venv/bin/python pilot/design_audit.py --only 2 3
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker25 as _s25
from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25

if not hasattr(_s25, "AliasEntry"):
    @dataclass(frozen=True)
    class AliasEntry:                                    # noqa: D401
        """Unpickling shim for checkpoints written before the scope removal.

        Those runs pickled `s_linker25.AliasEntry(component, scope)` inside the
        alias table. The class is gone from the linker (the table is now
        term -> component name), so pickle cannot resolve it and every older
        checkpoint becomes unreadable. Restoring the name here, in the audit
        rather than in the linker, keeps the old runs loadable without putting
        a dead type back into the paper artifact. Read `.component`; ignore
        `.scope`.
        """

        component: str
        scope: str = "global"

    _s25.AliasEntry = AliasEntry

BENCH = Path("../benchmark")
DEFAULT_RUN = Path("../results/s25_cleanup_verify_20260810")
PROJECTS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository",
                   "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository",
                 "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository",
                  "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository",
                      "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository",
               "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


# ── loading ──────────────────────────────────────────────────────────────────

def load_gold(name):
    gold = set()
    with open(BENCH / PROJECTS[name][2]) as handle:
        for row in csv.DictReader(handle):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                gold.add((int(snum), cid))
    return gold


def load_project(name):
    text, model, _ = PROJECTS[name]
    sentences = load_sentences(str(BENCH / text))
    components = parse_pcm_repository(str(BENCH / model))
    return {
        "sentences": sentences,
        "components": components,
        "sent_map": build_sent_map(sentences),
        "name_to_id": {c.name: c.id for c in components},
        "gold": load_gold(name),
    }


def phase_dir(run, name):
    return run / "phase_states" / "s_linker25" / "openai" / name


def load_phase(run, name, phase):
    path = phase_dir(run, name) / f"{phase}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_calls(run, name):
    """Every per-call trace file this run wrote for one project."""
    calls = []
    for path in sorted((run / "llm_logs").glob(
            f"s_linker25_openai_{name}_*_calls.json")):
        with path.open() as handle:
            calls.extend(json.load(handle))
    return calls


def alias_view(doc_knowledge):
    """{component: {"global": [...], "local": [...]}} from the alias table."""
    view = {}
    for term, entry in getattr(doc_knowledge, "aliases", {}).items():
        bucket = view.setdefault(entry.component, {"global": [], "local": []})
        bucket.setdefault(entry.scope, []).append(term)
    return view


def _states(text, names):
    return any(SLinker25._find_exact_form(text, n) for n in names)


# ── 1. sequence ──────────────────────────────────────────────────────────────

def audit_sequence(run):
    """Coreference proposals and approvals that restate an earlier link.

    Subtracting them can only remove pairs the union already holds, so the final
    link set is unchanged by construction; what the numbers size is how much
    judging work the missing subtraction pays for, and how many judge cases
    would leave each batch.
    """
    print("\n### 1 SEQUENCE — coreference proposals already linked upstream")
    print("     (subtracting them cannot change the final set: the union")
    print("      already holds every one of these pairs)")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        name_to_id = project["name_to_id"]
        full = load_phase(run, name, "linker_full_name")
        partial = load_phase(run, name, "linker_partial_name")
        coref = load_phase(run, name, "linker_coreference")
        prior = {(l.sentence_number, l.component_id) for l in full["links"]}
        prior |= {(l.sentence_number, l.component_id) for l in partial["links"]}

        proposals = [
            (item["sentence"], name_to_id.get(item["component"]))
            for item in coref["feedback"]["candidates"]
        ]
        proposals = [p for p in proposals if p[1] is not None]
        approved = [(l.sentence_number, l.component_id) for l in coref["links"]]
        dup_prop = [p for p in proposals if p in prior]
        dup_appr = [p for p in approved if p in prior]
        gold = project["gold"]
        print(f"  {name:14s} prior {len(prior):3d} | coref proposals "
              f"{len(proposals):3d} of which already linked {len(dup_prop):3d} "
              f"| approved {len(approved):2d} of which already linked "
              f"{len(dup_appr):2d} (TP {sum(1 for p in dup_appr if p in gold)})")
        totals.update(proposals=len(proposals), dup_prop=len(dup_prop),
                      approved=len(approved), dup_appr=len(dup_appr))
    saved = totals["dup_prop"] / max(totals["proposals"], 1) * 100
    print(f"  TOTAL          proposals {totals['proposals']} | already linked "
          f"{totals['dup_prop']} ({saved:.0f}% of judge cases) | approved "
          f"{totals['approved']} | already linked {totals['dup_appr']}")
    return dict(totals)


# ── 2. alias scope ───────────────────────────────────────────────────────────

def audit_alias_scope(run):
    """Every alias consumer, split by whether a local alias is load-bearing.

    Consumers, from the module: the extraction prompt (already global-only),
    the full-name admission gate `_keep_stated_names`, the partial-name
    suppressor, the identity-review anchors, and the coreference antecedent
    gate. All but the first read the table without its scope field.
    """
    print("\n### 2 ALIAS SCOPE — what a local alias is currently load-bearing for")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        aliases = alias_view(knowledge["doc_knowledge"])
        n_local = sum(len(v["local"]) for v in aliases.values())

        # (a) full-name admission: candidates that state no name and no global
        # alias, and are admitted only because a local alias appears.
        full = load_phase(run, name, "linker_full_name")
        only_local_admit = []
        for item in full["feedback"]["candidates"]:
            comp = item["component"]
            text = item["text"]
            bucket = aliases.get(comp, {"global": [], "local": []})
            if _states(text, [comp, *bucket["global"]]):
                continue
            if _states(text, bucket["local"]):
                only_local_admit.append((item["sentence"], comp))
        accepted = {(i["sentence"], i["component"]) for i in full["feedback"]["accepted"]}
        only_local_accepted = [p for p in only_local_admit if p in accepted]

        # (b) coreference antecedent gate: resolutions whose antecedent sentence
        # states only a local alias.
        coref = load_phase(run, name, "linker_coreference")
        id_to_name = {c.id: c.name for c in project["components"]}
        sent_map = project["sent_map"]
        only_local_ante = []
        for meta in coref["feedback"]["metadata"]:
            comp = id_to_name.get(meta["component_id"])
            ante = sent_map.get(meta.get("antecedent_sentence"))
            if not comp or not ante:
                continue
            bucket = aliases.get(comp, {"global": [], "local": []})
            if has_standalone_mention(comp, ante.text):
                continue
            if _states(ante.text, bucket["global"]):
                continue
            if _states(ante.text, bucket["local"]):
                only_local_ante.append((meta["sentence"], comp))
        approved_coref = {(l.sentence_number, id_to_name.get(l.component_id))
                          for l in coref["links"]}
        only_local_ante_kept = [p for p in only_local_ante if p in approved_coref]

        print(f"  {name:14s} local aliases {n_local:2d} | full-name candidates "
              f"admitted only by a local alias {len(only_local_admit):2d} "
              f"(accepted {len(only_local_accepted):2d}) | coref antecedents "
              f"only-local {len(only_local_ante):2d} (kept "
              f"{len(only_local_ante_kept):2d})")
        totals.update(local_aliases=n_local,
                      admit=len(only_local_admit),
                      admit_accepted=len(only_local_accepted),
                      ante=len(only_local_ante),
                      ante_kept=len(only_local_ante_kept))
    print(f"  TOTAL          local aliases {totals['local_aliases']} | "
          f"only-local admissions {totals['admit']} (accepted "
          f"{totals['admit_accepted']}) | only-local antecedents "
          f"{totals['ante']} (kept {totals['ante_kept']})")
    return dict(totals)


# ── 3. claim check ───────────────────────────────────────────────────────────

_CASE_SPLIT = re.compile(r"^Case (\d+): ", re.MULTILINE)
_QUOTED_LINE = re.compile(r'^  (?:\[prev: .*?\]\s*)?"(.*)"\s*$', re.MULTILINE)

JUDGE_PHASES = ("phase_25_full_name_p1", "phase_25_full_name_p2",
                "phase_25_coreference_judge")


def _cases_from_prompt(prompt):
    """{case number: sentence text} for a shared-judge prompt."""
    body = prompt.split("CASES:\n", 1)[-1]
    parts = _CASE_SPLIT.split(body)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        number, block = int(parts[i]), parts[i + 1]
        match = _QUOTED_LINE.search(block)
        if match:
            out[number] = match.group(1)
    return out


def _claim_verdicts(calls):
    """Yield (phase, sentence_text, approve, claim) for every judged case."""
    for call in calls:
        if call.get("phase") not in JUDGE_PHASES or not call.get("response_text"):
            continue
        cases = _cases_from_prompt(call["prompt"])
        text = call["response_text"]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            continue
        for item in data.get("validations", []):
            number = item.get("case")
            if number not in cases:
                continue
            yield (call["phase"], cases[number],
                   item.get("approve") is True
                   or (isinstance(item.get("approve"), str)
                       and item["approve"].lower() == "true"),
                   str(item.get("claim", "")).strip().strip("\"'“”‘’"))


def audit_claim_check(run):
    """How often a substring check on the reported quote would fire.

    The rule audited is the fabrication rule, not the partial-name linker's
    stricter presence rule: a verdict is voided only when the judge reported a
    quote that is not a contiguous substring of the sentence it ruled on.
    Reporting "none" is allowed, because the full-name rubric approves a bare
    mention that makes no architectural claim.
    """
    print("\n### 3 CLAIM CHECK — verdicts a fabricated-quote rule would void")
    totals = Counter()
    for name in PROJECTS:
        calls = load_calls(run, name)
        gold = load_gold(name)
        per_project = Counter()
        for phase, sentence, approve, claim in _claim_verdicts(calls):
            per_project["verdicts"] += 1
            if approve:
                per_project["approvals"] += 1
            if not claim:
                per_project["empty"] += 1
                continue
            if claim.casefold() in ("none", "n/a"):
                per_project["none"] += 1
                continue
            if claim.casefold() in sentence.casefold():
                per_project["grounded"] += 1
            else:
                per_project["fabricated"] += 1
                if approve:
                    per_project["fabricated_approve"] += 1
        print(f"  {name:14s} verdicts {per_project['verdicts']:4d} | quoted "
              f"{per_project['grounded'] + per_project['fabricated']:4d} | "
              f"none/empty {per_project['none'] + per_project['empty']:3d} | "
              f"not a substring {per_project['fabricated']:3d} "
              f"(of which approvals {per_project['fabricated_approve']:3d})")
        totals.update(per_project)
        del gold
    print(f"  TOTAL          verdicts {totals['verdicts']} | approvals "
          f"{totals['approvals']} | not a substring {totals['fabricated']} "
          f"(approvals {totals['fabricated_approve']})")
    return dict(totals)


# ── 4. ambiguity map ─────────────────────────────────────────────────────────

def audit_ambiguity(run):
    """How many candidates ever carry the flag the ambiguity map produces."""
    print("\n### 4 AMBIGUITY MAP — reach of its only consumer")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        ambiguous = set(getattr(knowledge["model_knowledge"], "ambiguous_names", set()))
        full = load_phase(run, name, "linker_full_name")
        candidates = full["feedback"]["candidates"]
        flagged = [c for c in candidates if c["component"] in ambiguous]
        accepted = {(i["sentence"], i["component"]) for i in full["feedback"]["accepted"]}
        flagged_accepted = [c for c in flagged
                            if (c["sentence"], c["component"]) in accepted]
        gold = project["gold"]
        name_to_id = project["name_to_id"]
        flagged_gold = [
            c for c in flagged_accepted
            if (c["sentence"], name_to_id.get(c["component"])) in gold
        ]
        print(f"  {name:14s} ambiguous names {len(ambiguous)} {sorted(ambiguous)} | "
              f"candidates carrying the flag {len(flagged):3d} of "
              f"{len(candidates):3d} | accepted {len(flagged_accepted):3d} "
              f"(gold {len(flagged_gold):3d})")
        totals.update(names=len(ambiguous), candidates=len(candidates),
                      flagged=len(flagged), flagged_accepted=len(flagged_accepted),
                      flagged_gold=len(flagged_gold))
    print(f"  TOTAL          ambiguous names {totals['names']} | flagged "
          f"candidates {totals['flagged']} of {totals['candidates']} | accepted "
          f"{totals['flagged_accepted']} (gold {totals['flagged_gold']})")
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--only", nargs="+", default=["1", "2", "3", "4"],
                        choices=["1", "2", "3", "4"])
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s25_design_audit/audit.json"))
    args = parser.parse_args()
    print(f"run: {args.run}")
    audits = {"1": audit_sequence, "2": audit_alias_scope,
              "3": audit_claim_check, "4": audit_ambiguity}
    report = {name: audits[name](args.run) for name in args.only}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump({"run": str(args.run), "audits": report}, handle, indent=2)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
