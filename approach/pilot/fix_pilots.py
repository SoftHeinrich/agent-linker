"""A/B pilots for the three measurable fixes from the s25 workflow audit.

    --fix 2   batch _review_identity like its sibling stage
    --fix 3   pack the coreference prompt as a numbered sentence table
    --fix 4   exclude qualified-path-only mentions in _keep_stated_names

Fix 1 (the stale-payload path in _ask) is not here: it fires only when a retry
fails to parse, which is too rare to measure and needs no measurement -- it
returns data the method already rejected.

Every pilot follows the same shape. Upstream stages that the fix does not touch
are computed ONCE and cached, so the only stochastic stage in the comparison is
the one under test; otherwise upstream noise swamps the effect. Arms then run N
times each and go through the permutation test in ab_stats.

s_linker25.py is untouched by fixes 2-4: each variant is a subclass overriding
exactly one method, so a rejected pilot leaves nothing to revert.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.s_linker25 import (
    SLinker25, COREF_RULES, ANTECEDENT_ALIAS_RULES,
)

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
MODEL = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-terra")
OUT = Path(os.environ.get("AB_OUT", "../results/s25_fix_pilots"))
# The coref sweep already paid for alias discovery under the same prompts and
# model; reusing it keeps every pilot on one alias set.
SHARED_KNOWLEDGE = Path("../results/coref_examples_ab/doc_knowledge.pkl")
COREF_CONTROL = Path("../results/coref_examples_ab/report.json")


# ── variants ─────────────────────────────────────────────────────────────────

class Fix2BatchedIdentity(SLinker25):
    """Batch the identity judge instead of one prompt for every participant."""

    def _review_identity(self, candidates, full_sentences):
        approved, decisions = [], {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            got, made = super()._review_identity(batch, full_sentences)
            approved.extend(got)
            decisions.update(made)
        return approved, decisions


class Fix3TableCoref(SLinker25):
    """Numbered sentence table instead of inlining each case's context.

    Same components, same instruction paragraph, same two rubrics, same
    response schema. Only the packaging of the evidence changes -- the shape
    the denotation and identity prompts already use.
    """

    def _prompt_coref(self, comp_names, cases):
        table, seen = [], set()
        for case in cases:
            for line in case["context"]:
                number = int(line.split("S", 1)[1].split(":", 1)[0])
                if number not in seen:
                    seen.add(number)
                    table.append({"sentence": number,
                                  "text": line.split(": ", 1)[1]})
        table.sort(key=lambda item: item["sentence"])
        case_list = [
            {"case": i + 1,
             "target": case["sent"].number,
             "context": sorted(int(line.split("S", 1)[1].split(":", 1)[0])
                               for line in case["context"])}
            for i, case in enumerate(cases)
        ]
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.

SENTENCES
{json.dumps(table)}

CASES
{json.dumps(case_list)}

{COREF_RULES}

{ANTECEDENT_ALIAS_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name", "antecedent_via_alias": false}}]}}

JSON only:"""


class Fix4QualifiedPath(SLinker25):
    """Drop candidates whose every name occurrence sits inside a dotted path.

    The two other candidate generators already apply this boundary rule; the
    full-name contract filter did not, leaving the judge rubric as the only
    defence.
    """

    def _keep_stated_names(self, candidates):
        kept = super()._keep_stated_names(candidates)
        names_by_component = self._names_by_component()
        out = []
        for candidate in kept:
            names = [candidate.component_name,
                     *names_by_component.get(candidate.component_name, [])]
            free = False
            for name in names:
                for match in _finditer_name(candidate.sentence_text, name):
                    if not self._inside_qualified_identifier(
                        candidate.sentence_text, match.start(), match.end()
                    ):
                        free = True
                        break
                if free:
                    break
            if free:
                out.append(candidate)
        return out


def _finditer_name(text, name):
    return re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)


def qualified_only(linker, candidate):
    """True when no occurrence of any of the component's names is path-free."""
    names_by_component = linker._names_by_component()
    names = [candidate.component_name,
             *names_by_component.get(candidate.component_name, [])]
    for name in names:
        for match in _finditer_name(candidate.sentence_text, name):
            if not linker._inside_qualified_identifier(
                candidate.sentence_text, match.start(), match.end()
            ):
                return False
    return True


# ── shared plumbing ──────────────────────────────────────────────────────────

def load_gold(path):
    gold = set()
    with open(path) as handle:
        for row in csv.DictReader(handle):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                gold.add((int(snum), cid))
    return gold


def build_inputs():
    data = {}
    for name, (text, model, gold) in DATASETS.items():
        sentences = load_sentences(str(BENCH / text))
        components = parse_pcm_repository(str(BENCH / model))
        data[name] = {
            "text_path": str(BENCH / text),
            "sentences": sentences,
            "components": components,
            "name_to_id": {c.name: c.id for c in components},
            "sent_map": build_sent_map(sentences),
            "gold": load_gold(BENCH / gold),
        }
    return data


def new_linker(cls=SLinker25, doc_knowledge=None, model_knowledge=None,
               text_path=None):
    linker = cls(backend=LLMBackend.OPENAI, model=MODEL)
    linker.doc_knowledge = doc_knowledge
    linker.model_knowledge = model_knowledge
    linker._current_text_path = text_path
    return linker


def doc_knowledge_cache(inputs):
    if SHARED_KNOWLEDGE.exists():
        with SHARED_KNOWLEDGE.open("rb") as handle:
            print(f"[knowledge] reusing {SHARED_KNOWLEDGE}")
            return pickle.load(handle)
    OUT.mkdir(parents=True, exist_ok=True)

    def one(name):
        linker = new_linker()
        return name, linker._learn_document_knowledge(
            inputs[name]["sentences"], inputs[name]["components"])

    cache = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for future in as_completed([pool.submit(one, n) for n in inputs]):
            name, knowledge = future.result()
            cache[name] = knowledge
    with SHARED_KNOWLEDGE.open("wb") as handle:
        pickle.dump(cache, handle)
    return cache


def cached(path, build):
    """Pickle-memoise an upstream stage so every arm sees the same input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as handle:
            print(f"  [cache] {path.name}")
            return pickle.load(handle)
    value = build()
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return value


def parallel(units, fn):
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fn, *u): u for u in units}
        for future in as_completed(futures):
            key = futures[future]
            try:
                out[key] = future.result()
            except Exception as exc:                        # noqa: BLE001
                print(f"    !! {key} FAILED: {exc}")
                out[key] = None
    return out


def scorers(inputs):
    def tp(pairs):
        return sum(1 for p, pair in pairs if tuple(pair) in inputs[p]["gold"])

    def fp(pairs):
        return len(pairs) - tp(pairs)

    return {"TP": tp, "FP": fp}


def report(name, stats, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    with path.open("w") as handle:
        json.dump({"model": MODEL, "runs": RUNS, "stats": stats,
                   **(extra or {})}, handle, indent=2, default=str)
    print(f"report -> {path}")


# ── fix 2: identity batching ─────────────────────────────────────────────────

def pilot_fix2(inputs, knowledge):
    print("\n### fix 2 — batch _review_identity")
    print("Upstream cached: partial-name candidates are deterministic; the")
    print("denotation judge runs once per project and both arms reuse it.")
    print("current_links is empty for both arms, so the candidate set is a")
    print("superset of production's -- same input to both, and it stresses")
    print("batching harder than production would.")

    participants = {}
    for name, item in inputs.items():
        linker = new_linker(doc_knowledge=knowledge[name],
                            text_path=item["text_path"])
        candidates = linker._name_word_candidates(
            item["sentences"], item["components"], [])
        if not candidates:
            participants[name] = []
            print(f"  {name}: 0 candidates")
            continue
        got = cached(
            OUT / f"fix2_denotation_{name}.pkl",
            lambda: linker._classify_denotations(candidates, item["sentences"])[0],
        )
        participants[name] = got
        print(f"  {name}: {len(candidates)} candidates -> {len(got)} participants")

    def one(arm, run, name):
        if not participants[name]:
            return set()
        cls = SLinker25 if arm == "A_single_prompt" else Fix2BatchedIdentity
        linker = new_linker(cls, knowledge[name], None, inputs[name]["text_path"])
        approved, _ = linker._review_identity(
            participants[name], inputs[name]["sentences"])
        return {(name, (c.sentence_number, c.component_id)) for c in approved}

    arms = {}
    for arm in ("A_single_prompt", "B_batched"):
        units = [(arm, run, name) for run in range(1, RUNS + 1) for name in inputs]
        got = parallel(units, one)
        arms[arm] = [set().union(*[got[(arm, run, n)] or set() for n in inputs])
                     for run in range(1, RUNS + 1)]
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    stats = permutation_report(arms, scorers(inputs),
                               title="fix 2 — identity batching")
    report("fix2_identity_batching", stats,
           {"participants": {k: len(v) for k, v in participants.items()}})


# ── fix 3: coref prompt packaging ────────────────────────────────────────────

def pilot_fix3(inputs, knowledge):
    print("\n### fix 3 — numbered-table coreference prompt")
    if not COREF_CONTROL.exists():
        sys.exit(f"control arm missing: {COREF_CONTROL}")
    with COREF_CONTROL.open() as handle:
        control_raw = json.load(handle)["arms"]["B_examples_cut"]
    print(f"Control arm reused from {COREF_CONTROL} (same model, same alias")
    print("cache, same Examples-cut rubric). Only the new arm is paid for.")

    control = [
        {(name, tuple(pair))
         for name in inputs
         for pair in control_raw[f"{run}|{name}"]["links"]}
        for run in range(1, RUNS + 1)
    ]

    sizes = {"A_inline": 0, "B_table": 0}
    for name, item in inputs.items():
        for cls, key in ((SLinker25, "A_inline"), (Fix3TableCoref, "B_table")):
            linker = new_linker(cls, knowledge[name], None, item["text_path"])
            smap = item["sent_map"]
            for start in range(0, len(item["sentences"]), linker.COREFERENCE_BATCH):
                batch = item["sentences"][start:start + linker.COREFERENCE_BATCH]
                cases = []
                for sent in batch:
                    lo = max(1, sent.number - linker.CONTEXT_SENTENCES)
                    context = [
                        f"{'>>>' if i == sent.number else '   '} S{i}: {smap[i].text}"
                        for i in range(lo, sent.number + linker.CONTEXT_SENTENCES + 1)
                        if i in smap
                    ]
                    cases.append({"sent": sent, "context": context})
                names = [c.name for c in item["components"]]
                sizes[key] += len(linker._prompt_coref(names, cases))
    print(f"  prompt bytes over all batches: inline {sizes['A_inline']:,}  "
          f"table {sizes['B_table']:,}  "
          f"({sizes['B_table'] / sizes['A_inline'] * 100:.0f}%)")

    def one(run, name):
        item = inputs[name]
        linker = new_linker(Fix3TableCoref, knowledge[name], None,
                            item["text_path"])
        links, _ = linker._resolve_references(
            item["sentences"], item["components"], item["name_to_id"],
            item["sent_map"])
        return {(name, (l.sentence_number, l.component_id)) for l in links}

    units = [(run, name) for run in range(1, RUNS + 1) for name in inputs]
    got = parallel(units, one)
    table_arm = [set().union(*[got[(run, n)] or set() for n in inputs])
                 for run in range(1, RUNS + 1)]
    print(f"  B_table: {[len(s) for s in table_arm]}")

    stats = permutation_report({"A_inline": control, "B_table": table_arm},
                               scorers(inputs),
                               title="fix 3 — coref prompt packaging")
    report("fix3_coref_table", stats, {"prompt_bytes": sizes})


# ── fix 4: qualified-path filter ─────────────────────────────────────────────

def pilot_fix4(inputs, knowledge):
    print("\n### fix 4 — qualified-path filter in _keep_stated_names")
    print("Extraction is cached per project: it is upstream of the filter and")
    print("untouched by it, so both arms judge the same extraction output.")

    prepared = {}
    ambiguity = {}
    for name, item in inputs.items():
        linker = new_linker(doc_knowledge=knowledge[name],
                            text_path=item["text_path"])
        # Only feeds a display flag in the evidence bundle, but keeping it makes
        # the pilot's prompts match production's.
        ambiguity[name] = cached(
            OUT / f"fix4_ambiguity_{name}.pkl",
            lambda: linker._analyze_model(item["components"]),
        )
        raw = cached(
            OUT / f"fix4_extraction_{name}.pkl",
            lambda: linker._extract_named_mentions(
                item["sentences"], item["components"], item["name_to_id"],
                item["sent_map"]),
        )
        base = list(raw.values())
        kept_a = SLinker25._keep_stated_names(linker, base)
        drop = [c for c in kept_a if qualified_only(linker, c)]
        prepared[name] = {"raw": base, "dropped": len(drop),
                          "dropped_keys": {(c.sentence_number, c.component_id)
                                           for c in drop}}
        print(f"  {name}: {len(base)} extracted -> {len(kept_a)} stated-name "
              f"-> arm B drops {len(drop)}")

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_current" else Fix4QualifiedPath
        linker = new_linker(cls, knowledge[name], ambiguity[name],
                            item["text_path"])
        candidates = linker._keep_stated_names(prepared[name]["raw"])
        candidates = linker._add_spelling_variants(
            candidates, item["sentences"], item["components"])
        bundles = {
            (c.sentence_number, c.component_id):
                linker._build_evidence_bundle(c, item["sent_map"])
            for c in candidates
        }
        approved, _ = linker._validate_with_evidence(
            candidates, bundles, item["components"], item["sent_map"],
            p1_tag="pilot_fix4_p1", p2_tag="pilot_fix4_p2",
            stage_label="full_name")
        return {(name, (c.sentence_number, c.component_id)) for c in approved}

    arms = {}
    for arm in ("A_current", "B_path_filtered"):
        units = [(arm, run, name) for run in range(1, RUNS + 1) for name in inputs]
        got = parallel(units, one)
        arms[arm] = [set().union(*[got[(arm, run, n)] or set() for n in inputs])
                     for run in range(1, RUNS + 1)]
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    # What arm A actually approves among the pairs arm B never offers.
    print("\n  qualified-path-only pairs that arm A approves:")
    for run, links in enumerate(arms["A_current"], 1):
        hit = {(n, p) for n, p in links if p in prepared[n]["dropped_keys"]}
        tp = sum(1 for n, p in hit if p in inputs[n]["gold"])
        print(f"    run {run}: {len(hit)} approved  TP {tp}  FP {len(hit) - tp}")

    stats = permutation_report(arms, scorers(inputs),
                               title="fix 4 — qualified-path filter")
    report("fix4_qualified_path", stats,
           {"dropped": {k: v["dropped"] for k, v in prepared.items()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", nargs="+", required=True,
                        choices=["2", "3", "4"])
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
    started = time.time()
    inputs = build_inputs()
    knowledge = doc_knowledge_cache(inputs)
    for fix in args.fix:
        {"2": pilot_fix2, "3": pilot_fix3, "4": pilot_fix4}[fix](inputs, knowledge)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
