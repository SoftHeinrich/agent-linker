"""Level 1: would annotating the shortlist with `written` discriminate, where the
name judge's VERDICT did not? No LLM calls.

`s_linker124` marked each shortlist entry with what the union judge decided about that
mention (`linked` / `named only`) and the doc-code gate refused it. Two things were
wrong with that fact, and only the first was recorded at the time:

  1. It is a DISCOVERED fact -- another judge's output, resampled every run -- so it
     injects the name stage's variance into the resolver. `s_linker109` already ruled
     that a discovered fact may open a case and may not close one.
  2. It was nearly constant: 83% (terra) / 79% (luna) of entries read `linked`.

`written` is the alternative: the same deterministic lexical fact `s_linker123` already
computes for the union judge, given rather than discovered, and stable across runs by
construction. This audit asks the only question that decides whether it is worth a call:

  R1  What is `written` over the shortlist entries the resolver actually sees?
      A near-constant field cannot discriminate, whatever it says.
  R2  Of the coreference links that SURVIVE the judge, does the antecedent's `written`
      value separate gold from false positive? That difference is the whole ceiling --
      if gold and FP have the same profile, no marking of this fact can help.
  R3  The same, restricted to the single-candidate cases, which is where s124 did its
      damage (precision 95.5% -> 77.4%, 10 of its 15 extra FPs).

Run:

    python3 pilot/coref_written_audit.py ../results
"""
from __future__ import annotations

import csv
import glob
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "results")
ARMS = ("s_linker123", "s_linker124")
VALUES = ("whole name", "whole name (qualified)", "short form", "one word")


def written_probe(knowledge):
    """An `s_linker123` with just enough state to answer `_written_as`."""
    linker = SLinker123.__new__(SLinker123)
    linker.doc_knowledge = knowledge
    return linker


def phase(run_dir, variant, project, name):
    path = run_dir / "phase_states" / variant / "openai" / project / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def resolutions(run_dir, variant, project):
    """Recorded resolutions: (sentence, component, antecedent_sentence, n_candidates)."""
    out = []
    for path in sorted(glob.glob(str(
            run_dir / "llm_logs" / f"{variant}_openai_{project}_*_calls.json"))):
        for call in json.load(open(path)):
            if call.get("phase") != "phase_25_coreference":
                continue
            try:
                data = json.loads(call.get("response_text") or "{}")
            except json.JSONDecodeError:
                continue
            for r in data.get("resolutions", []):
                comp, sent = r.get("component"), r.get("sentence")
                ant = r.get("antecedent_sentence")
                if comp and isinstance(sent, int) and isinstance(ant, int):
                    out.append((sent, comp, ant,
                                len(r.get("candidates") or [])))
    return out


def survivors(run_dir, variant, project):
    path = run_dir / f"{variant}_{project}_links.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        return {(int(r["sentence"]), r["component_name"])
                for r in csv.DictReader(fh) if r.get("source") == "coreference"}


def main():
    entries = Counter()        # R1: `written` over shortlist entries
    kept = Counter()           # R2: surviving links by (written, gold?)
    single = Counter()         # R3: the same, single-candidate only
    onlist = Counter()         # R4: was the cited antecedent ON the shortlist?
    sentences_cache = {}

    for model in ("terra", "luna"):
        for run in (1, 2, 3):
            run_dir = RESULTS / f"shortlistmark_e2e_{model}_r{run}_20260914"
            if not run_dir.is_dir():
                continue
            for project in sorted(DATASETS):
                text, repo, gold_path = DATASETS[project]
                if project not in sentences_cache:
                    sentences_cache[project] = (
                        {s.number: s.text for s in load_sentences(str(BENCH / text))},
                        parse_pcm_repository(str(BENCH / repo)),
                        gold_pairs(BENCH / gold_path))
                sent_text, components, gold = sentences_cache[project]
                by_name = {c.name: c.id for c in components}

                for variant in ARMS:
                    know = phase(run_dir, variant, project, "knowledge")
                    surv = survivors(run_dir, variant, project)
                    if know is None or surv is None:
                        continue
                    probe = written_probe(know["doc_knowledge"])

                    # R1: every shortlist entry the resolver was shown, which is every
                    # (component, earlier sentence) pair `_states_a_name` accepts.
                    for name in by_name:
                        for number, body in sent_text.items():
                            if probe._states_a_name(body, name):
                                entries[probe._written_as(body, name)] += 1

                    # R2 / R3: the antecedent each surviving link leaned on.
                    for sent, comp, ant, ncand in resolutions(
                            run_dir, variant, project):
                        if (sent, comp) not in surv or comp not in by_name:
                            continue
                        body = sent_text.get(ant)
                        if body is None:
                            continue
                        value = probe._written_as(body, comp)
                        hit = (sent, by_name[comp]) in gold
                        kept[(value, hit)] += 1
                        kept[(model, value, hit)] += 1
                        if ncand <= 1:
                            single[(value, hit)] += 1
                        # R4: the prompt asserts the antecedent is in the list. Was it?
                        listed = (ant < sent
                                  and probe._states_a_name(body, comp))
                        onlist[(model, listed, hit)] += 1

    total = sum(entries.values()) or 1
    print("R1 — `written` over every shortlist entry the resolver is shown\n")
    for v in VALUES:
        n = entries[v]
        print(f"  {v:<24}{n:>7}  {n / total * 100:5.1f}%")
    print(f"  {'TOTAL':<24}{total:>7}")

    def table(counter, label):
        print(f"\n{label}\n")
        print(f"  {'antecedent written as':<24}{'gold':>7}{'FP':>7}"
              f"{'n':>7}{'precision':>11}")
        tg = tf = 0
        for v in VALUES:
            g, f = counter[(v, True)], counter[(v, False)]
            tg, tf = tg + g, tf + f
            n = g + f
            if n:
                print(f"  {v:<24}{g:>7}{f:>7}{n:>7}{g / n * 100:>10.1f}%")
        n = tg + tf or 1
        print(f"  {'ALL':<24}{tg:>7}{tf:>7}{n:>7}{tg / n * 100:>10.1f}%")

    table(kept, "R2 — surviving coreference links, by their antecedent's `written`")

    def per_model(model):
        c = Counter()
        for key, n in kept.items():
            if len(key) == 3 and key[0] == model:
                c[(key[1], key[2])] += n
        table(c, f"R2/{model} — the same, {model} only")

    per_model("terra")
    per_model("luna")
    table(single, "R3 — the same, single-candidate cases only (where s124 broke)")

    print("\nR4 — was the cited antecedent actually an entry on this case's shortlist?")
    print("     (the resolver prompt asserts it is: 'where the antecedent will be')\n")
    print(f"  {'model':<8}{'antecedent':<16}{'gold':>7}{'FP':>7}{'n':>7}{'precision':>11}")
    for model in ("terra", "luna"):
        for listed in (True, False):
            g, f = onlist[(model, listed, True)], onlist[(model, listed, False)]
            n = g + f
            if n:
                print(f"  {model:<8}{('ON list' if listed else 'OFF list'):<16}"
                      f"{g:>7}{f:>7}{n:>7}{g / n * 100:>10.1f}%")


main()
