"""Empirical: does a KNOWLEDGE-INFORMED blocks proposer close the recall gap to
s21's Framing-C entity pass? The blind blocks proposer is a recall ceiling but NOT
a superset of Framing-C — on bbb it loses ~2 gold that Framing-C uniquely catches,
almost certainly ALIAS-mediated mentions (Framing-C injects Phase-1 doc aliases
into its extraction prompt; the blocks proposer is knowledge-blind).

This harness injects s21's OWN Phase-1 knowledge into the blocks prompt and sweeps
which part helps:

  blind  — the shipped blocks:20 prompt (no knowledge)                [baseline]
  alias  — + "known alternative terms in THIS document: term -> Component"
  ambig  — + "these catalog names are also ordinary words; link only when the
             sentence really means the component"
  both   — alias + ambig

Metric = gold-candidate RECALL ceiling (grounded (sentence,component) pairs that
are gold / all gold), plus candidate VOLUME (precision proxy for the downstream
gate) and — the crux — whether the config recovers the gold Framing-C catches but
blind blocks misses (`s21_only`).

s21's Phase-1 knowledge AND its Framing-C keys are captured in ONE s21 run (stop
after Phase 2). All results cached. GATE-01: s21 untouched (capture subclass only);
GATE-06: aliases/ambiguous names are runtime doc-derived input s21 already consumes,
not benchmark vocabulary baked into code.

    OPENAI_SERVICE_TIER=default python pilot/knowledge_proposer_compare.py --dataset bigbluebutton
    OPENAI_SERVICE_TIER=default python pilot/knowledge_proposer_compare.py --dataset teammates
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21, ENTITY_EXTRACTION_RULES
from llm_sad_sam.linkers.experimental.proposer import (
    _catalog_block, _parse_batch, _COMMON_REF_RULE, make_client)

# Reference rules the proposer can use. "default" = the proposer's own thin rule;
# "framingc" = s21's canonical ENTITY_EXTRACTION_RULES (participant-in-interaction +
# "Favor inclusion" recall bias + code-path / ordinary-English exclusions), ported
# verbatim so the proposer gets Framing-C's full extraction strength. GATE-06 safe:
# ENTITY_EXTRACTION_RULES is generic English, the canonical s21 rule.
RULES = {"default": _COMMON_REF_RULE, "framingc": ENTITY_EXTRACTION_RULES}

# config -> (use_alias, use_ambig, rule_key)
CONFIG_SPEC = {
    "blind":    (False, False, "default"),
    "alias":    (True,  False, "default"),
    "ambig":    (False, True,  "default"),
    "both":     (True,  True,  "default"),
    "fc":       (False, False, "framingc"),
    "fc_alias": (True,  False, "framingc"),
    "fc_both":  (True,  True,  "framingc"),
}

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
DS = {
    "mediastore": ("mediastore/model_2016/pcm/ms.repository", "mediastore/text_2016/mediastore.txt", "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/model_2020/pcm/teastore.repository", "teastore/text_2020/teastore.txt", "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/model_2021/pcm/teammates.repository", "teammates/text_2021/teammates.txt", "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository", "bigbluebutton/text_2021/bigbluebutton.txt", "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/model_2021/pcm/jabref.repository", "jabref/text_2021/jabref.txt", "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}
KCACHE = Path("pilot/cache/knowledge_cache.json")            # per-dataset: aliases, ambiguous, s21 framing-c keys
PCACHE = Path("pilot/cache/knowledge_proposer_cache.json")   # proposer raw responses
CONFIGS = ("blind", "alias", "ambig", "both")   # default set; --configs to add fc*
BATCH = 20


# ── capture s21 Phase-1 knowledge + Phase-2 Framing-C keys in one run ─────────

class _Stop(Exception):
    def __init__(self, cands, mk, dk):
        self.cands, self.mk, self.dk = cands, mk, dk


class _Capture(SLinker21):
    def _run_framing_c(self, sentences, components, name_to_id, sent_map):
        cands = super()._run_framing_c(sentences, components, name_to_id, sent_map)
        # Phase 1 has run by now → self.model_knowledge / self.doc_knowledge are set
        raise _Stop(cands, self.model_knowledge, self.doc_knowledge)


def _alias_pairs(dk):
    """Global-scope (term -> component) pairs, exactly as s21 Framing-C uses."""
    out = []
    for term, entry in (dk.aliases or {}).items():
        comp = getattr(entry, "component", entry)          # AliasEntry or bare str
        scope = getattr(entry, "scope", "global")
        if scope == "global":
            out.append((term, comp))
    return out


def get_knowledge(dataset, repo, text):
    cache = json.loads(KCACHE.read_text()) if KCACHE.exists() else {}
    if dataset in cache:
        c = cache[dataset]
        return c["aliases"], c["ambiguous"], {tuple(k) for k in c["s21_keys"]}
    linker = _Capture(backend=LLMBackend.OPENAI, model="gpt-5.4")
    try:
        linker.link(str(text), str(repo))
    except _Stop as s:
        aliases = _alias_pairs(s.dk)
        ambiguous = sorted(s.mk.ambiguous_names or [])
        s21_keys = sorted(list(k) for k in s.cands.keys())
        cache[dataset] = {"aliases": aliases, "ambiguous": ambiguous, "s21_keys": s21_keys}
        KCACHE.parent.mkdir(parents=True, exist_ok=True)
        KCACHE.write_text(json.dumps(cache, indent=1))
        return aliases, ambiguous, {tuple(k) for k in s21_keys}
    raise RuntimeError("Phase 2 did not fire")


# ── knowledge-injected blocks prompt (superset of proposer.build_batch_prompt) ─

def build_prompt(chunk, names, prev_of, aliases, ambiguous, use_alias, use_ambig,
                 rule_key="default"):
    catalog = _catalog_block(names, None)
    ref_rule = RULES[rule_key]
    blocks = "\n\n".join(
        f'ITEM {s.number}\n  PREVIOUS: "{prev_of.get(s.number, "")}"\n'
        f'  SENTENCE: "{s.text}"' for s in chunk)
    know = ""
    if use_alias and aliases:
        lines = "\n".join(f'  - "{t}" -> {c}' for t, c in aliases)
        know += ("\nKnown alternative terms used in THIS document — if a SENTENCE "
                 "uses the term (or its wording), it refers to the mapped catalog "
                 "component (quote the term as the words):\n" + lines + "\n")
    if use_ambig and ambiguous:
        know += ("\nThese catalog names are also ordinary English words: "
                 + ", ".join(ambiguous) +
                 ". Only link one when the SENTENCE really refers to that software "
                 "component, not the everyday word.\n")
    return (
        "Below are independent ITEMS. Treat each ITEM as a self-contained task: "
        "decide which catalog components its SENTENCE refers to, using its PREVIOUS "
        "line only as context. Give every item the same independent attention.\n\n"
        f"Choose components ONLY from this catalog (copy the exact name):\n{catalog}\n"
        f"{know}\n{ref_rule}\n\n{blocks}\n\n"
        'Return JSON: {"items":[{"item":<int>,"refs":[{"component":"<name>",'
        '"quote":"<words>"}]}]}\nJSON only:')


def run_config(dataset, config, sentences, names, prev_of, aliases, ambiguous, name_to_id):
    use_alias, use_ambig, rule_key = CONFIG_SPEC[config]
    cache = json.loads(PCACHE.read_text()) if PCACHE.exists() else {}
    client = None
    lut = {n.lower(): n for n in names}
    keys, ncand = set(), 0
    for i in range(0, len(sentences), BATCH):
        chunk = sentences[i:i + BATCH]
        ck = f"{dataset}|{config}|{chunk[0].number}-{chunk[-1].number}"
        if ck in cache:
            raw = cache[ck]
        else:
            if client is None:
                client = make_client()
            prompt = build_prompt(chunk, names, prev_of, aliases, ambiguous,
                                  use_alias, use_ambig, rule_key)
            resp = client.query(prompt, timeout=240)
            raw = _parse_batch(resp.text if resp.success else "", "blocks")
            cache[ck] = raw
            PCACHE.parent.mkdir(parents=True, exist_ok=True)
            PCACHE.write_text(json.dumps(cache, indent=1))
        for r in raw:
            canon = lut.get(r["component"].lower())
            cid = name_to_id.get(canon) if canon else None
            if cid:
                ncand += 1
                keys.add((int(r["sentence"]), cid))
    return keys, ncand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigbluebutton")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()
    repo, text, gcsv = (BASE / p for p in DS[args.dataset])

    comps = parse_pcm_repository(str(repo))
    name_to_id = {c.name: c.id for c in comps}
    names = [c.name for c in comps]
    sentences = load_sentences(str(text))
    sent_map = build_sent_map(sentences)
    prev_of = {s.number: (sent_map.get(s.number - 1).text
                          if sent_map.get(s.number - 1) else "") for s in sentences}
    gold = set()
    for row in csv.DictReader(open(gcsv)):
        gold.add((int(row["sentence"]), row["modelElementID"]))
    gold = {(s, c) for (s, c) in gold if c in name_to_id.values()}

    aliases, ambiguous, s21_keys = get_knowledge(args.dataset, repo, text)
    s21_gold = s21_keys & gold

    print(f"\n=== knowledge-informed proposer: {args.dataset} "
          f"(gold={len(gold)}, aliases={len(aliases)}, ambiguous={len(ambiguous)}) ===")
    print(f"s21 Framing-C recall (reference): {len(s21_gold)}/{len(gold)} = "
          f"{len(s21_gold)/len(gold):.3f}   cands={len(s21_keys)}")
    print(f"\n{'config':<8}{'cands':>7}{'gold_hit':>9}{'recall':>8}"
          f"{'recovers_s21only':>18}{'adds_vs_s21':>13}")
    s21_only_base = None
    rows = {}
    for cfg in args.configs:
        keys, ncand = run_config(args.dataset, cfg, sentences, names, prev_of,
                                 aliases, ambiguous, name_to_id)
        g = keys & gold
        s21_only = s21_gold - keys                 # gold s21 caught that this config missed
        adds = len(g - s21_gold)                    # gold this config caught that s21 missed
        rows[cfg] = (keys, g, s21_only)
        if cfg == "blind":
            s21_only_base = s21_only
        recovered = (len(s21_only_base - s21_only) if s21_only_base is not None
                     and cfg != "blind" else 0)
        print(f"{cfg:<8}{ncand:>7}{len(g):>9}{len(g)/len(gold):>8.3f}"
              f"{('+' + str(recovered)) if cfg != 'blind' else '-':>18}"
              f"{'+' + str(adds):>13}")

    # crux detail: what blind misses that s21 catches, and who recovers it
    if "blind" in rows:
        _, _, blind_s21only = rows["blind"]
        print(f"\ngold Framing-C catches but BLIND blocks misses ({len(blind_s21only)}):")
        for k in sorted(blind_s21only):
            who = [c for c in args.configs if c != "blind" and k not in rows[c][2]]
            print(f"  S{k[0]} {k[1]}   recovered by: {', '.join(who) or 'NONE'}")
        # union of best config with s21
        best = max((c for c in args.configs), key=lambda c: len(rows[c][1]))
        u = (rows[best][0] | s21_keys) & gold
        print(f"\nbest config = {best} (recall {len(rows[best][1])/len(gold):.3f}); "
              f"union(best, s21) recall = {len(u)/len(gold):.3f}")


if __name__ == "__main__":
    main()
