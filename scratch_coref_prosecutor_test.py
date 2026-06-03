"""Checkpoint-level verification of the coref adversarial prosecutor (option 3).

Loads the FROZEN layer2.pkl coref output (no coref re-run -> zero LLM variance),
labels each coref link TP/FP vs gold, runs the prosecutor on the exact frozen
links, then recomputes the FULL deduped pipeline metrics before vs after.
"""
import os, sys, pickle, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.linkers.experimental.s_linker15 import SLinker15
from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker15")
DATASETS = {
    "teammates": (BASE/"teammates/text_2021/teammates.txt", BASE/"teammates/model_2021/pcm/teammates.repository",
                  BASE/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "teastore":  (BASE/"teastore/text_2020/teastore.txt", BASE/"teastore/model_2020/pcm/teastore.repository",
                  BASE/"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "bigbluebutton": (BASE/"bigbluebutton/text_2021/bigbluebutton.txt", BASE/"bigbluebutton/model_2021/pcm/bbb.repository",
                  BASE/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}

def load_gold(p):
    g = set()
    for row in csv.DictReader(open(p)):
        cid, sn = row.get("modelElementID","").strip(), row.get("sentence","").strip()
        if cid and sn: g.add((int(sn), cid))
    return g

def dedup(links):
    seen, out = set(), []
    for lk in links:
        k = (lk.sentence_number, lk.component_id)
        if k not in seen:
            seen.add(k); out.append(lk)
    return out

def metrics(pairs, gold):
    tp = len(pairs & gold); fp = len(pairs - gold); fn = len(gold - pairs)
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return tp, fp, fn, p, r, f1

def pairs_of(links): return {(l.sentence_number, l.component_id) for l in links}

backend = LLMBackend.CLAUDE if os.environ.get("LLM_BACKEND","").lower()!="openai" else LLMBackend.OPENAI
linker = SLinker15(backend=backend)
print(f"Backend: {backend}")

macro_before, macro_after = [], []
for ds,(txt,mdl,goldp) in DATASETS.items():
    l2 = pickle.load(open(CACHE/ds/"layer2.pkl","rb"))
    seed_links, validated, coref_links = l2["seed_links"], l2["validated"], l2["coref_links"]
    entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, source=c.source) for c in validated]
    sentences = load_sentences(str(txt)); components = parse_pcm_repository(str(mdl))
    sent_map = build_sent_map(sentences); name_to_id = {c.name:c.id for c in components}
    gold = load_gold(goldp)

    # coref-source TP/FP before
    coref_pairs = pairs_of(coref_links)
    c_tp = len(coref_pairs & gold); c_fp = len(coref_pairs - gold)

    print(f"\n===== {ds} =====")
    print(f"  coref BEFORE: {len(coref_links)} links | TP={c_tp} FP={c_fp}")

    kept = linker._coref_prosecutor(coref_links, components, name_to_id, sent_map)
    kept_pairs = pairs_of(kept)
    k_tp = len(kept_pairs & gold); k_fp = len(kept_pairs - gold)
    # what the prosecutor killed, labeled by gold
    killed_pairs = coref_pairs - kept_pairs
    killed_fp = len(killed_pairs - gold); killed_tp = len(killed_pairs & gold)
    print(f"  coref AFTER : {len(kept)} links | TP={k_tp} FP={k_fp}  (killed {killed_fp} FP, {killed_tp} TP)")

    # full pipeline before/after
    before = dedup(seed_links + entity_links + coref_links)
    after  = dedup(seed_links + entity_links + kept)
    b = metrics(pairs_of(before), gold); a = metrics(pairs_of(after), gold)
    print(f"  FINAL before: P={b[3]*100:.1f} R={b[4]*100:.1f} F1={b[5]*100:.1f} (TP={b[0]} FP={b[1]} FN={b[2]})")
    print(f"  FINAL after : P={a[3]*100:.1f} R={a[4]*100:.1f} F1={a[5]*100:.1f} (TP={a[0]} FP={a[1]} FN={a[2]})")
    macro_before.append(b[5]); macro_after.append(a[5])

print(f"\n===== MACRO (these {len(DATASETS)} ds) =====")
print(f"  before F1: {sum(macro_before)/len(macro_before)*100:.1f}")
print(f"  after  F1: {sum(macro_after)/len(macro_after)*100:.1f}")
