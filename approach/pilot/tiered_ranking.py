"""Tiered-ranking reframe of doc->model TLR: instead of a binary keep/reject (bin
prediction), assign each candidate an EVIDENCE TIER (firm / probable / weak) and read
the whole precision-recall curve, so the operating point (precision-first vs
recall-first, i.e. F1 vs F2) is a CHOICE on the ranking rather than one fixed cut.

Design: per-candidate evidence signals already latent in the pipeline —
  match  : EXACT (component name stands alone in the sentence) | TERMINAL (only the
           name's terminal word appears, e.g. "the client" for HTML5 Client) |
           ALIAS (quote is a known doc alias) | ROLE (generic/role phrase)
  votes  : s21 two-pass gate P1+P2 approvals in {0,1,2}
  source : FC (Framing-C) | BLK (blocks proposer, alias+sibling aware)
The EXPENSIVE parts (extraction + gate) run ONCE per dataset and are CACHED to
pilot/cache/tier_signals_<ds>.json; tier schemes are applied offline, so the
design->error-analysis->redesign loop costs no API.

    OPENAI_SERVICE_TIER=default python pilot/tiered_ranking.py --dataset bigbluebutton [--recapture]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names, has_standalone_mention
from llm_sad_sam.linkers.experimental.s_linker23_verify import SLinker23Verify

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("pilot/cache")
DS = {
    "mediastore": ("mediastore/model_2016/pcm/ms.repository", "mediastore/text_2016/mediastore.txt", "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/model_2020/pcm/teastore.repository", "teastore/text_2020/teastore.txt", "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/model_2021/pcm/teammates.repository", "teammates/text_2021/teammates.txt", "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository", "bigbluebutton/text_2021/bigbluebutton.txt", "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/model_2021/pcm/jabref.repository", "jabref/text_2021/jabref.txt", "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


class _Stop(Exception):
    pass


class _Cap(SLinker23Verify):
    def _run_framing_c(self, s, c, n, m):
        self._fc = super()._run_framing_c(s, c, n, m)
        raise _Stop()


def _match_type(name, quote, sentence, alias_terms):
    s = sentence.lower()
    if has_standalone_mention(name, sentence):
        return "EXACT"
    words = name.split()
    if len(words) >= 2 and has_standalone_mention(words[-1], sentence):
        return "TERMINAL"
    if quote and quote.lower() in alias_terms:
        return "ALIAS"
    return "ROLE"


def capture(dataset):
    repo, text, gcsv = (BASE / p for p in DS[dataset])
    comps = parse_pcm_repository(str(repo))
    name_to_id = {c.name: c.id for c in comps}
    names = [c.name for c in comps]
    comp_names = get_comp_names(comps)
    sentences = load_sentences(str(text))
    sent_map = build_sent_map(sentences)
    prev_of = {s.number: (sent_map.get(s.number - 1).text if sent_map.get(s.number - 1) else "")
               for s in sentences}
    gold = {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(open(gcsv))}
    gold = {k for k in gold if k[1] in set(name_to_id.values())}

    v = _Cap(backend=LLMBackend.OPENAI, model="gpt-5.4")
    try:
        v.link(str(text), str(repo))
    except _Stop:
        pass
    aliases = v._global_aliases(sentences) or []
    alias_terms = {t.lower() for t, _ in aliases}

    fc = {(c.sentence_number, c.component_id): c for c in v._fc.values()}
    props = v._propose(sentences, names, prev_of, base_final=[])
    blk = {}
    for r in props:
        cid = name_to_id.get(r["component"]); s = sent_map.get(r["sentence"])
        if cid and s:
            blk.setdefault((r["sentence"], cid),
                           CandidateLink(r["sentence"], s.text, r["component"], cid,
                                         r.get("quote", ""), source="entity"))
    allkeys = set(fc) | set(blk)
    cands = [fc.get(k) or blk[k] for k in allkeys]

    # one gate run over ALL candidates -> per-candidate p1/p2
    bundles = {(c.sentence_number, c.component_id): v._build_evidence_bundle(c, sent_map) for c in cands}
    _, decisions = v._validate_with_evidence(cands, bundles, comps, sent_map,
                                             p1_tag="tier_p1", p2_tag="tier_p2", stage_label="tier")
    rows = []
    for c in cands:
        k = (c.sentence_number, c.component_id)
        d = decisions.get(k, {})
        rows.append({
            "sn": c.sentence_number, "cid": c.component_id, "name": c.component_name,
            "quote": c.matched_text or "", "sentence": c.sentence_text,
            "source": "FC" if k in fc else "BLK",
            "match": _match_type(c.component_name, c.matched_text or "", c.sentence_text, alias_terms),
            "p1": int(bool(d.get("p1"))), "p2": int(bool(d.get("p2"))),
            "gold": 1 if k in gold else 0,
        })
    out = {"n_gold": len(gold), "rows": rows}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"tier_signals_{dataset}.json").write_text(json.dumps(out, indent=1))
    return out


# ── TIER SCHEME v2 — empirically derived from match x votes x source gold-rates ──
# (EXACT,2)=1.00  (ROLE,2,FC)=1.00  (ALIAS,2)=0.89  (ROLE,1,BLK)=0.74  (ROLE,2,BLK)=0.50
# (ROLE,1,FC)=0.17  (ROLE,0,*)=0.04  (EXACT,1)=0.00 . Source flips ROLE reliability:
# blocks-proposer (sibling-aware) role refs are trustworthy; Framing-C role refs are not.
def assign_tier(r):
    votes = r["p1"] + r["p2"]
    m, src = r["match"], r["source"]
    if votes == 2 and m in ("EXACT", "TERMINAL"):
        return "FIRM"
    if votes == 2 and m == "ROLE" and src == "FC":
        return "FIRM"
    if votes == 2 and m == "ALIAS":
        return "PROBABLE"
    if votes == 1 and m == "ROLE" and src == "BLK":
        return "PROBABLE"          # the sibling-recovered role links (0.74)
    if votes == 2 and m == "ROLE" and src == "BLK":
        return "WEAK"              # coin-flip (0.50)
    return "REJECT"


TIER_ORDER = ["FIRM", "PROBABLE", "WEAK"]   # cumulative cuts, firm first


def prf(kept, gold_n, rows):
    tp = sum(1 for r in kept if r["gold"])
    fp = len(kept) - tp
    fn = gold_n - tp
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    F2 = 5 * P * R / (4 * P + R) if (4 * P + R) else 0.0
    return tp, fp, fn, P, R, F1, F2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigbluebutton")
    ap.add_argument("--recapture", action="store_true")
    args = ap.parse_args()
    f = CACHE / f"tier_signals_{args.dataset}.json"
    data = capture(args.dataset) if (args.recapture or not f.exists()) else json.loads(f.read_text())
    rows, gold_n = data["rows"], data["n_gold"]
    for r in rows:
        r["tier"] = assign_tier(r)

    print(f"\n=== tiered ranking: {args.dataset} (gold={gold_n}, candidates={len(rows)}) ===")
    # gold vs non-gold distribution across tiers
    print(f"{'tier':<9}{'#cand':>6}{'#gold':>6}{'#nongold':>9}{'purity':>8}")
    for t in TIER_ORDER + ["REJECT"]:
        sub = [r for r in rows if r["tier"] == t]
        g = sum(r["gold"] for r in sub)
        print(f"{t:<9}{len(sub):>6}{g:>6}{len(sub)-g:>9}{(g/len(sub) if sub else 0):>8.2f}")

    print(f"\ncumulative cuts (firm -> add lesser tiers):")
    print(f"{'cut':<22}{'kept':>5}{'TP':>4}{'FP':>4}{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}")
    for i in range(len(TIER_ORDER)):
        incl = set(TIER_ORDER[:i + 1])
        kept = [r for r in rows if r["tier"] in incl]
        tp, fp, fn, P, R, F1, F2 = prf(kept, gold_n, rows)
        print(f"{'+'.join(TIER_ORDER[:i+1]):<22}{len(kept):>5}{tp:>4}{fp:>4}"
              f"{P:>7.3f}{R:>7.3f}{F1:>7.3f}{F2:>7.3f}")

    # error analysis: gold that landed in WEAK/REJECT (under-tiered) & FP in FIRM
    print("\n-- ERROR: gold under-tiered (in WEAK or REJECT) --")
    for r in sorted([r for r in rows if r["gold"] and r["tier"] in ("WEAK", "REJECT")],
                    key=lambda r: r["sn"]):
        print(f"   [{r['tier']:<7} {r['match']:<8} v{r['p1']+r['p2']} {r['source']}] "
              f"S{r['sn']} {r['name']}: \"{r['quote']}\"")
    print("-- ERROR: non-gold in FIRM (firm-tier FP) --")
    for r in sorted([r for r in rows if not r["gold"] and r["tier"] == "FIRM"],
                    key=lambda r: r["sn"]):
        print(f"   [{r['match']:<8} v{r['p1']+r['p2']} {r['source']}] "
              f"S{r['sn']} {r['name']}: \"{r['quote']}\"  | {r['sentence'][:70]}")


if __name__ == "__main__":
    main()
