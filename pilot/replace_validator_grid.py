"""Design-space grid for the REPLACE question: can a stronger VALIDATOR make the
blocks proposer viable as a replacement for s21's Framing-C extractor?

The blind e2e result was: blocks (alone or unioned) has a higher recall ceiling but
leaks FP when dumped straight into s21's gate (macro FP 4->21). This harness isolates
the VALIDATOR from extraction — it runs the SAME candidate sets through different
validators and measures kept TP/FP directly, so we can see whether the failure is
the extractor or the gate, and whether a stronger gate rescues replace.

EXTRACTORS (candidate sets, pre-validation):
  FC     — s21 Framing-C (alias-injected, 2-pass union) — the incumbent
  BLK    — blocks:20 proposer, alias-informed — the replacement
  UNION  — FC ∪ BLK

VALIDATORS:
  g_s21     — s21's real 2-pass evidence-bundle gate (what replace/union used) [BASELINE]
  g_router  — DocModelAgenticRouter (VALIDATE/CODE/REJECT) THEN the s21 evidence gate
              (the s23_verify stack, applied to ALL candidates, not just augmented)

Everything reuses SLinker23Verify's OWN methods (captured after Phase 1), so the gate
and router are the real ones. GATE-01: s21 untouched (capture subclass only).

    OPENAI_SERVICE_TIER=default python pilot/replace_validator_grid.py --dataset teastore
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
from llm_sad_sam.linkers.experimental.s_linker23_verify import SLinker23Verify
from llm_sad_sam.linkers.experimental.agentic_router import DocModelAgenticRouter, Candidate

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
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
    """Run s21 through Phase 2, capture Framing-C + Phase-1 knowledge, then stop —
    keeping the instance usable for its real gate / router / proposer methods."""
    def _run_framing_c(self, sentences, components, name_to_id, sent_map):
        self._fc = super()._run_framing_c(sentences, components, name_to_id, sent_map)
        raise _Stop()


def _prf(kept, gold):
    tp = len(kept & gold); fp = len(kept - gold); fn = len(gold - kept)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, P=p, R=r, F1=f1, F2=f2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="teastore")
    args = ap.parse_args()
    repo, text, gcsv = (BASE / p for p in DS[args.dataset])

    components = parse_pcm_repository(str(repo))
    name_to_id = {c.name: c.id for c in components}
    names = [c.name for c in components]
    comp_names = get_comp_names(components)
    sentences = load_sentences(str(text))
    sent_map = build_sent_map(sentences)
    prev_of = {s.number: (sent_map.get(s.number - 1).text
                          if sent_map.get(s.number - 1) else "") for s in sentences}
    gold = set()
    for row in csv.DictReader(open(gcsv)):
        gold.add((int(row["sentence"]), row["modelElementID"]))
    gold = {(s, c) for (s, c) in gold if c in name_to_id.values()}

    # ── capture s21 Phase-1 knowledge + Framing-C candidates ─────────────────
    v = _Cap(backend=LLMBackend.OPENAI, model="gpt-5.4")
    try:
        v.link(str(text), str(repo))
    except _Stop:
        pass

    fc = list(v._fc.values())                                   # list[CandidateLink]
    # blocks-alias candidates (base_final empty => pure blocks, alias-informed)
    proposals = v._propose(sentences, names, prev_of, base_final=[])
    blk, seen = [], set()
    for r in proposals:
        cid = name_to_id.get(r["component"]); sent = sent_map.get(r["sentence"])
        if cid is None or sent is None or (r["sentence"], cid) in seen:
            continue
        seen.add((r["sentence"], cid))
        blk.append(CandidateLink(r["sentence"], sent.text, r["component"], cid,
                                 r.get("quote", ""), source="entity"))
    # union (FC wins key collisions)
    umap = {(c.sentence_number, c.component_id): c for c in blk}
    umap.update({(c.sentence_number, c.component_id): c for c in fc})
    union = list(umap.values())

    extractors = {"FC": fc, "BLK": blk, "UNION": union}

    # ── validators ───────────────────────────────────────────────────────────
    def g_s21(cands):
        if not cands:
            return set()
        bundles = {(c.sentence_number, c.component_id): v._build_evidence_bundle(c, sent_map)
                   for c in cands}
        validated, _ = v._validate_with_evidence(
            cands, bundles, components, sent_map,
            p1_tag="grid_s21_p1", p2_tag="grid_s21_p2", stage_label="grid")
        return {(c.sentence_number, c.component_id) for c in validated}

    def g_router(cands):
        if not cands:
            return set()
        router = DocModelAgenticRouter(gate=v._router_gate(components, comp_names, sent_map))
        objs = [Candidate(id=f"{c.sentence_number}|{c.component_id}", sentence=c.sentence_text,
                          component=c.component_name, prev=prev_of.get(c.sentence_number, ""),
                          quote=c.matched_text or "") for c in cands]
        decisions = router.route(objs)
        acc = {d.candidate.id for d in decisions if d.accepted}
        return {(c.sentence_number, c.component_id) for c in cands
                if f"{c.sentence_number}|{c.component_id}" in acc}

    validators = {"g_s21": g_s21, "g_router": g_router}

    # ── run grid ─────────────────────────────────────────────────────────────
    fc_keys = {(c.sentence_number, c.component_id) for c in fc}
    print(f"\n=== replace-validator grid: {args.dataset} (gold={len(gold)}) ===")
    print(f"pre-gate recall ceiling:  "
          f"FC={len(fc_keys & gold)}/{len(gold)}  "
          f"BLK={len({(c.sentence_number,c.component_id) for c in blk} & gold)}/{len(gold)}  "
          f"UNION={len({(c.sentence_number,c.component_id) for c in union} & gold)}/{len(gold)}")
    print(f"candidate volume:         FC={len(fc)}  BLK={len(blk)}  UNION={len(union)}\n")
    print(f"{'extractor':<8}{'validator':<10}{'kept':>5}{'TP':>4}{'FP':>4}"
          f"{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}{'  FP: FC-origin / BLK-only'}")
    for ename, cands in extractors.items():
        for vname, gate in validators.items():
            kept = gate(cands)
            m = _prf(kept, gold)
            fps = kept - gold
            fp_fc = len(fps & fc_keys); fp_blk = len(fps - fc_keys)
            print(f"{ename:<8}{vname:<10}{len(kept):>5}{m['tp']:>4}{m['fp']:>4}"
                  f"{m['P']:>7.3f}{m['R']:>7.3f}{m['F1']:>7.3f}{m['F2']:>7.3f}"
                  f"       {fp_fc} / {fp_blk}")


if __name__ == "__main__":
    main()
