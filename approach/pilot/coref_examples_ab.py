"""Coref-only A/A'/B sweep for the ANTECEDENT_ALIAS_RULES "Examples:" cut.

Question: does cutting the Examples block change which coreference resolutions
the model emits, beyond run-to-run noise?

Design
  arm A  = the s21 constant verbatim (Examples block present)  -- control
  arm B  = the s25 constant (Examples block cut)               -- candidate
  N runs per arm; the pairwise A-vs-A distances give the noise band that the
  A-vs-B distance must fall inside for the cut to be behaviour-neutral.

Only `_resolve_references` is called -- the coreference proposal step plus its
structural antecedent gate. The judge is skipped on purpose: it is unchanged by
this edit and only subtracts, so proposal-set drift is the whole effect and
scoring it directly avoids paying for a second stochastic stage.

Document knowledge (aliases) is computed ONCE per project and reused by every
arm and run: its prompts are untouched by the edit, so recomputing it would only
inject noise into the comparison.

Arms run sequentially because the prompt text lives in a module-level constant;
units within an arm run concurrently.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src")

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental import s_linker21 as L21
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25

BENCH = Path("../benchmark")
DATASETS = {
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

RUNS = int(os.environ.get("AB_RUNS", "3"))
WORKERS = int(os.environ.get("AB_WORKERS", "5"))
OUT = Path(os.environ.get("AB_OUT", "../results/coref_examples_ab"))
MODEL = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-terra")

ARM_TEXT = {"A_with_examples": L21.ANTECEDENT_ALIAS_RULES,
            "B_examples_cut": L25.ANTECEDENT_ALIAS_RULES}
assert ARM_TEXT["A_with_examples"] != ARM_TEXT["B_examples_cut"]


def load_gold(path):
    gold = set()
    with open(path) as handle:
        for row in csv.DictReader(handle):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                gold.add((int(snum), cid))
    return gold


def new_linker():
    linker = SLinker25(backend=LLMBackend.OPENAI, model=MODEL)
    return linker


def build_inputs():
    data = {}
    for name, (text, model, gold) in DATASETS.items():
        sentences = load_sentences(str(BENCH / text))
        components = parse_pcm_repository(str(BENCH / model))
        data[name] = {
            "sentences": sentences,
            "components": components,
            "name_to_id": {c.name: c.id for c in components},
            "sent_map": build_sent_map(sentences),
            "gold": load_gold(BENCH / gold),
        }
    return data


def knowledge_cache(inputs):
    """One alias-discovery pass per project, reused by every arm and run."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "doc_knowledge.pkl"
    if path.exists():
        with path.open("rb") as handle:
            print(f"[knowledge] reusing {path}")
            return pickle.load(handle)

    def one(name):
        linker = new_linker()
        knowledge = linker._learn_document_knowledge(
            inputs[name]["sentences"], inputs[name]["components"])
        return name, knowledge, linker._llm_calls

    cache, calls = {}, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, name) for name in inputs]
        for future in as_completed(futures):
            name, knowledge, made = future.result()
            cache[name] = knowledge
            calls += len(made)
            print(f"[knowledge] {name}: {len(knowledge.aliases)} aliases "
                  f"({len(made)} calls)")
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    print(f"[knowledge] {calls} LLM calls total -> {path}")
    return cache


def resolve(name, inputs, knowledge, tag):
    linker = new_linker()
    linker.doc_knowledge = knowledge
    item = inputs[name]
    started = time.time()
    links, _meta = linker._resolve_references(
        item["sentences"], item["components"], item["name_to_id"],
        item["sent_map"])
    tokens = sum((c.get("token_usage") or {}).get("total_tokens", 0)
                 for c in linker._llm_calls)
    print(f"    {tag} {name}: {len(links)} resolutions "
          f"({len(linker._llm_calls)} calls, {time.time() - started:.0f}s)")
    return {
        "links": sorted({(l.sentence_number, l.component_id) for l in links}),
        "calls": len(linker._llm_calls),
        "tokens": tokens,
        "elapsed_s": round(time.time() - started, 1),
    }


def run_arm(arm, inputs, knowledge):
    L25.ANTECEDENT_ALIAS_RULES = ARM_TEXT[arm]
    assert ("Examples:" in L25.ANTECEDENT_ALIAS_RULES) == (arm == "A_with_examples")
    print(f"\n=== arm {arm} ({len(ARM_TEXT[arm])} ch) ===")
    units = [(run, name) for run in range(1, RUNS + 1) for name in inputs]
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(resolve, name, inputs, knowledge[name], f"r{run}"):
                (run, name)
            for run, name in units
        }
        for future in as_completed(futures):
            run, name = futures[future]
            try:
                results[f"{run}|{name}"] = future.result()
            except Exception as exc:                      # noqa: BLE001
                print(f"    !! r{run} {name} FAILED: {exc}")
                results[f"{run}|{name}"] = {"error": str(exc)}
    return results


def pooled(results, run, inputs):
    """All five projects' resolutions for one run, as project-qualified keys."""
    out = set()
    for name in inputs:
        item = results.get(f"{run}|{name}", {})
        for pair in item.get("links", []):
            out.add((name, tuple(pair)))
    return out


def score(pairs, inputs):
    tp = sum(1 for name, pair in pairs if tuple(pair) in inputs[name]["gold"])
    return tp, len(pairs) - tp


def analyse(sets, inputs):
    """Permutation test over the six runs.

    A min-distance-vs-max-noise rule is not a test: it passes whenever the noise
    band is wide, which is exactly when the design has no power. Instead relabel
    the runs every possible way and ask how extreme the true A|B labelling is.

    Two statistics, because they answer different questions:
      * set distance -- does the cut change WHICH resolutions appear;
      * TP / FP counts -- does it change how many are right or wrong.
    Composition can shift while quality does not; that combination is a real
    result and the old rule could not express it.
    """
    labelled = [("A", sets["A_with_examples"][r]) for r in sorted(sets["A_with_examples"])]
    labelled += [("B", sets["B_examples_cut"][r]) for r in sorted(sets["B_examples_cut"])]
    runs = [s for _, s in labelled]
    n = len(runs)
    half = n // 2

    def spread(group_a):
        a = [runs[i] for i in group_a]
        b = [runs[i] for i in range(n) if i not in group_a]
        within = [len(x ^ y) for g in (a, b)
                  for i, x in enumerate(g) for y in g[i + 1:]]
        between = [len(x ^ y) for x in a for y in b]
        return sum(between) / len(between) - sum(within) / len(within)

    def counts(group_a, key):
        vals = [key(s) for s in runs]
        a = [vals[i] for i in group_a]
        b = [vals[i] for i in range(n) if i not in group_a]
        return sum(b) / len(b) - sum(a) / len(a)

    from itertools import combinations
    truth = tuple(range(half))
    splits = [s for s in combinations(range(n), half) if 0 in s]   # halve by symmetry
    observed = spread(truth)
    null = sorted((spread(s) for s in splits), reverse=True)
    p_distance = sum(1 for v in null if v >= observed) / len(splits)

    quality = {}
    for label, key in (("TP", lambda s: score(s, inputs)[0]),
                       ("FP", lambda s: score(s, inputs)[1])):
        delta = counts(truth, key)
        p = sum(1 for s in splits if abs(counts(s, key)) >= abs(delta)) / len(splits)
        quality[label] = {"delta_B_minus_A": round(delta, 1), "p": round(p, 2)}

    floor = 1 / len(splits)
    if quality["TP"]["p"] > 0.2 and quality["FP"]["p"] > 0.2:
        verdict = (f"QUALITY-NEUTRAL: TP p={quality['TP']['p']}, "
                   f"FP p={quality['FP']['p']}. Composition p={p_distance:.2f} "
                   f"(floor {floor:.2f} at this N)")
    else:
        verdict = (f"QUALITY-CHANGING: TP p={quality['TP']['p']}, "
                   f"FP p={quality['FP']['p']}")

    pairs = [(i, j) for i in range(1, RUNS + 1) for j in range(i + 1, RUNS + 1)]
    stats = {
        "noise_AA": [len(sets["A_with_examples"][i] ^ sets["A_with_examples"][j])
                     for i, j in pairs],
        "noise_BB": [len(sets["B_examples_cut"][i] ^ sets["B_examples_cut"][j])
                     for i, j in pairs],
        "cross_AB": [len(sets["A_with_examples"][i] ^ sets["B_examples_cut"][j])
                     for i in range(1, RUNS + 1) for j in range(1, RUNS + 1)],
        "distance_stat": round(observed, 1),
        "distance_null": [round(v, 1) for v in null],
        "p_distance": round(p_distance, 2),
        "p_floor": round(floor, 2),
        "quality": quality,
        "verdict": verdict,
    }

    print("\nsymmetric difference (links)")
    print(f"  A vs A                {stats['noise_AA']}")
    print(f"  B vs B                {stats['noise_BB']}")
    print(f"  A vs B                min {min(stats['cross_AB'])} / mean "
          f"{sum(stats['cross_AB'])/len(stats['cross_AB']):.1f} / "
          f"max {max(stats['cross_AB'])}")
    print("\npermutation test over all six runs")
    print(f"  composition (mean between - mean within): {observed:+.1f}")
    print(f"    null spread {stats['distance_null']}")
    print(f"    p = {p_distance:.2f}  (floor {floor:.2f} at N={RUNS}/arm)")
    for label in ("TP", "FP"):
        print(f"  {label}: B-A {quality[label]['delta_B_minus_A']:+.1f}   "
              f"two-sided p {quality[label]['p']:.2f}")
    print(f"  verdict: {verdict}")
    return stats


def main():
    inputs = build_inputs()
    reanalyse = "--reanalyse" in sys.argv or "--reanalyze" in sys.argv
    if reanalyse:
        # Re-score a completed sweep from its saved resolution sets. Free.
        with (OUT / "report.json").open() as handle:
            arms = json.load(handle)["arms"]
        print(f"[reanalyse] {OUT / 'report.json'}")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
        knowledge = knowledge_cache(inputs)
        arms = {}
        for arm in ("A_with_examples", "B_examples_cut"):
            arms[arm] = run_arm(arm, inputs, knowledge)
        L25.ANTECEDENT_ALIAS_RULES = ARM_TEXT["B_examples_cut"]  # restore shipped

    sets = {arm: {run: pooled(res, run, inputs) for run in range(1, RUNS + 1)}
            for arm, res in arms.items()}

    print("\n" + "=" * 72)
    print(f"{'arm':<18}{'run':>4}{'resolutions':>13}{'would-be TP':>13}"
          f"{'would-be FP':>13}")
    for arm in sets:
        for run in range(1, RUNS + 1):
            tp, fp = score(sets[arm][run], inputs)
            print(f"{arm:<18}{run:>4}{len(sets[arm][run]):>13}{tp:>13}{fp:>13}")

    stats = analyse(sets, inputs)
    noise, b_noise, cross = stats["noise_AA"], stats["noise_BB"], stats["cross_AB"]
    verdict = stats["verdict"]

    # links present in every A run but no B run, and vice versa
    a_all = set.intersection(*sets["A_with_examples"].values())
    b_all = set.intersection(*sets["B_examples_cut"].values())
    a_any = set.union(*sets["A_with_examples"].values())
    b_any = set.union(*sets["B_examples_cut"].values())
    lost, gained = a_all - b_any, b_all - a_any
    print(f"\nstable-loss  (every A run, no B run): {len(lost)}  "
          f"TP {score(lost, inputs)[0]} / FP {score(lost, inputs)[1]}")
    for name, pair in sorted(lost):
        mark = "TP" if tuple(pair) in inputs[name]["gold"] else "FP"
        print(f"    -{mark} {name} S{pair[0]} {pair[1]}")
    print(f"stable-gain  (every B run, no A run): {len(gained)}  "
          f"TP {score(gained, inputs)[0]} / FP {score(gained, inputs)[1]}")
    for name, pair in sorted(gained):
        mark = "TP" if tuple(pair) in inputs[name]["gold"] else "FP"
        print(f"    +{mark} {name} S{pair[0]} {pair[1]}")

    calls = sum(v.get("calls", 0) for res in arms.values() for v in res.values())
    tokens = sum(v.get("tokens", 0) for res in arms.values() for v in res.values())
    print(f"\ncoref LLM calls: {calls}   tokens: {tokens:,}")

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL, "runs": RUNS,
        "arms": {arm: {k: v for k, v in res.items()} for arm, res in arms.items()},
        **stats,
        "stable_loss": sorted([[n, list(p)] for n, p in lost]),
        "stable_gain": sorted([[n, list(p)] for n, p in gained]),
        "calls": calls, "tokens": tokens,
    }
    path = OUT / "report.json"
    with path.open("w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
