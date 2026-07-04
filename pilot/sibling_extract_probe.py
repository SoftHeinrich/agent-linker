"""Root-cause-targeted probe (error-mode driven, not metric-chasing): does the
sibling-disambiguation hint recover the extraction misses identified in
pilot/ERROR_MODES.md? On bbb, 13 gold links (HTML5 Client/Server, WebRTC-SFU) were
NEVER extracted. This runs the REAL proposer.propose_batch with sibling_disambig
off vs on (alias-informed in both), grounds, and reports recall of the whole gold
AND specifically of the sibling-family gold. No gate — pure extraction ceiling.

    OPENAI_SERVICE_TIER=default python pilot/sibling_extract_probe.py --dataset bigbluebutton
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.proposer import (
    GroundedTypedProposer, filter_generic_aliases, _sibling_families)
from llm_sad_sam.linkers.experimental.s_linker23_verify import SLinker23Verify

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
    def _run_framing_c(self, s, c, n, m):
        raise _Stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigbluebutton")
    args = ap.parse_args()
    repo, text, gcsv = (BASE / p for p in DS[args.dataset])

    comps = parse_pcm_repository(str(repo))
    name_to_id = {c.name: c.id for c in comps}
    names = [c.name for c in comps]
    sentences = load_sentences(str(text))
    sent_map = build_sent_map(sentences)
    prev_of = {s.number: (sent_map.get(s.number - 1).text
                          if sent_map.get(s.number - 1) else "") for s in sentences}
    gold = {(int(r["sentence"]), r["modelElementID"])
            for r in csv.DictReader(open(gcsv))}
    gold = {k for k in gold if k[1] in set(name_to_id.values())}

    # sibling-family gold = the modes this change targets
    fam_names = {n for fam in _sibling_families(names) for n in fam}
    fam_ids = {name_to_id[n] for n in fam_names}
    fam_gold = {k for k in gold if k[1] in fam_ids}

    # capture Phase-1 aliases (alias-informed in both arms, filtered)
    v = _Cap(backend=LLMBackend.OPENAI, model="gpt-5.4")
    try:
        v.link(str(text), str(repo))
    except _Stop:
        pass
    aliases = v._global_aliases(sentences)

    def rec(sib):
        p = GroundedTypedProposer(catalog_mode="name")
        props = p.propose_batch(sentences, names, batch_size=20, strategy="blocks",
                                prev_of=prev_of, aliases=aliases, sibling_disambig=sib)
        keys = set()
        for r in props:
            cid = name_to_id.get(r["component"])
            if cid:
                keys.add((int(r["sentence"]), cid))
        return keys

    base = rec(False)
    sib = rec(True)

    print(f"\n=== sibling-disambig extraction probe: {args.dataset} ===")
    print(f"gold={len(gold)}  sibling-family gold={len(fam_gold)} "
          f"(families: {[' / '.join(f) for f in _sibling_families(names)]})")
    for label, keys in [("baseline (alias)", base), ("+sibling", sib)]:
        print(f"  {label:20} recall {len(keys & gold)}/{len(gold)}={len(keys & gold)/len(gold):.3f}"
              f"   family {len(keys & fam_gold)}/{len(fam_gold)}"
              f"   cands={len(keys)}")
    print("\nfamily gold recovered by +sibling that baseline missed:")
    id2n = {v_: k for k, v_ in name_to_id.items()}
    for k in sorted((sib & fam_gold) - (base & fam_gold)):
        print(f"   +S{k[0]} {id2n[k[1]]}")
    for k in sorted((base & fam_gold) - (sib & fam_gold)):
        print(f"   -S{k[0]} {id2n[k[1]]} (LOST)")


if __name__ == "__main__":
    main()
