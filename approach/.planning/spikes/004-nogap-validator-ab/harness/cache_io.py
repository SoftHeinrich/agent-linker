#!/usr/bin/env python3
"""Spike 004 — shared loaders for replaying ONLY the validator layer.

Loads pre-computed candidates / decisions / final links from the frozen
s_linker20_union phase_cache (no upstream rerun), plus the per-dataset benchmark
sentence text + components + gold. Reused by Stage 0b (rule trap rejecter) and
Stage 1/2 (LLM validator replay).

Importing this registers the linker dataclasses (AliasEntry / EvidenceBundle) so
the pickles unpickle natively.
"""
import os
import pickle
import sys
from dataclasses import dataclass

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import run_ablation as R  # reuse identical loaders + DATASETS + eval_metrics
import llm_sad_sam.linkers.experimental.s_linker20_union  # noqa: F401  registers classes

NOTHINK_ROOT = "results/v2.6.5_s20union_sonnet_nothink_20260627"
THINKING_ROOT = "results/v2.6.5_s20union_sonnet"
GPT_ROOT = "results/v2.6.5_s20union/gpt"        # gpt-5.4 NO-REASONING baseline (macro 89.4)
DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]


def cache_cell_dir(sweep_root, run, dataset, cache_backend="claude"):
    return os.path.join(REPO, sweep_root, run, "phase_cache",
                        "s_linker20_union", cache_backend, dataset)


def load_cell(sweep_root, run, dataset, cache_backend="claude"):
    """Load layer1/3/4/final pickles for one (sweep, run, dataset) cell."""
    base = cache_cell_dir(sweep_root, run, dataset, cache_backend)
    out = {}
    for layer in ("layer1", "layer3", "layer4", "final"):
        with open(os.path.join(base, f"{layer}.pkl"), "rb") as fh:
            out[layer] = pickle.load(fh)
    return out


# Per-dataset benchmark cache (sent_map / components / gold) — loaded once.
_BENCH = {}


def load_benchmark(dataset):
    if dataset in _BENCH:
        return _BENCH[dataset]
    p = R.DATASETS[dataset]
    comps = R.parse_pcm_repository(str(p["model"]))
    id_to_name = {c.id: c.name for c in comps}
    sents = R.DocumentLoader.load_sentences(str(p["text"]))
    sent_map = {s.number: s for s in sents}
    gold = R.load_gold_sam(str(p["gold_sam"]))
    _BENCH[dataset] = dict(components=comps, id_to_name=id_to_name,
                           sent_map=sent_map, gold=gold)
    return _BENCH[dataset]


@dataclass
class LinkCtx:
    """Everything a trap rule or a re-validator needs about one final link."""
    sentence_number: int
    component_id: str
    component_name: str
    source: str                 # "entity" | "coreference"
    sentence_text: str
    # entity evidence (from layer3 evidence_bundles); empty for coref-only links
    matched_span: str = ""
    mention_type: str = ""
    is_ambiguous: bool = False
    anchor_sentences: tuple = ()
    # coref evidence (from layer4 coref_metadata); empty for entity links
    reference: str = ""
    antecedent_text: str = ""
    antecedent_sentence: int = 0
    in_gold: bool = False
    name_in_text: bool = False


def build_contexts(cell, dataset):
    """Return list[LinkCtx], one per FINAL link in the cell, enriched with
    benchmark text + per-source evidence + gold membership."""
    bench = load_benchmark(dataset)
    sent_map, gold, id_to_name = bench["sent_map"], bench["gold"], bench["id_to_name"]
    ev = cell["layer3"]["evidence_bundles"]
    cmeta = cell["layer4"]["coref_metadata"]
    ctxs = []
    for lk in cell["final"]["final"]:
        key = (lk.sentence_number, lk.component_id)
        s = sent_map.get(lk.sentence_number)
        text = s.text if s else ""
        name = id_to_name.get(lk.component_id, lk.component_name)
        c = LinkCtx(
            sentence_number=lk.sentence_number,
            component_id=lk.component_id,
            component_name=name,
            source=lk.source,
            sentence_text=text,
            in_gold=key in gold,
            name_in_text=(name.lower() in text.lower()) if text else False,
        )
        if key in ev:
            b = ev[key]
            c.matched_span = b.get("matched_span", "")
            c.mention_type = b.get("mention_type", "")
            c.is_ambiguous = bool(b.get("is_ambiguous", False))
            c.anchor_sentences = tuple(b.get("anchor_sentences", []))
        if key in cmeta:
            m = cmeta[key]
            c.reference = m.get("reference", "")
            c.antecedent_text = m.get("antecedent_text", "")
            c.antecedent_sentence = m.get("antecedent_sentence", 0)
        ctxs.append(c)
    return ctxs


def score_pairs(pairs, gold):
    return R.eval_metrics(set(pairs), set(gold))
