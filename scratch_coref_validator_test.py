"""Checkpoint-level test: route coref links through the EXISTING entity validator.

No new prompt. Coref currently bypasses validation entirely. Here we feed the
frozen coref links into s_linker15c._validate_with_evidence — the same 2-pass
intersection (architectural-participation AND referential-specificity, evidence
-bundle backed) that entity candidates already pass. Keep coref links the
validator approves. Runs on FROZEN layer2 coref (no re-run -> zero variance).
"""
import os, sys, pickle, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.data_types_v2 import SadSamLink, CandidateLink
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker15c import SLinker15c
from llm_sad_sam.llm_client import LLMBackend

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker15")   # s15/s15c share coref logic; reuse frozen coref
DATASETS = {
    "teammates": (BASE/"teammates/text_2021/teammates.txt", BASE/"teammates/model_2021/pcm/teammates.repository",
                  BASE/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "teastore":  (BASE/"teastore/text_2020/teastore.txt", BASE/"teastore/model_2020/pcm/teastore.repository",
                  BASE/"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "bigbluebutton": (BASE/"bigbluebutton/text_2021/bigbluebutton.txt", BASE/"bigbluebutton/model_2021/pcm/bbb.repository",
                  BASE/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}

def load_gold(p):
    g=set()
    for row in csv.DictReader(open(p)):
        cid,sn=row.get("modelElementID","").strip(),row.get("sentence","").strip()
        if cid and sn: g.add((int(sn),cid))
    return g

def dedup(links):
    seen,out=set(),[]
    for lk in links:
        k=(lk.sentence_number,lk.component_id)
        if k not in seen: seen.add(k); out.append(lk)
    return out

def pairs_of(links): return {(l.sentence_number,l.component_id) for l in links}

def metrics(pairs,gold):
    tp=len(pairs&gold); fp=len(pairs-gold); fn=len(gold-pairs)
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0
    return tp,fp,fn,p,r,f1

backend = LLMBackend.OPENAI if os.environ.get("LLM_BACKEND","").lower()=="openai" else LLMBackend.CLAUDE
linker = SLinker15c(backend=backend)
print(f"Backend: {backend}")

macro_before, macro_after = [], []
for ds,(txt,mdl,goldp) in DATASETS.items():
    l1=pickle.load(open(CACHE/ds/"layer1.pkl","rb"))
    l2=pickle.load(open(CACHE/ds/"layer2.pkl","rb"))
    seed_links,validated,coref_links=l2["seed_links"],l2["validated"],l2["coref_links"]
    entity_links=[SadSamLink(c.sentence_number,c.component_id,c.component_name,source=c.source) for c in validated]
    sentences=load_sentences(str(txt)); components=parse_pcm_repository(str(mdl))
    sent_map=build_sent_map(sentences); gold=load_gold(goldp)

    # restore knowledge so the validator's generic-filter / evidence bundles work
    linker.model_knowledge=l1["model_knowledge"]; linker.doc_knowledge=l1["doc_knowledge"]
    linker._current_text_path=None

    coref_pairs=pairs_of(coref_links)
    print(f"\n===== {ds} =====  coref BEFORE: {len(coref_links)} | TP={len(coref_pairs&gold)} FP={len(coref_pairs-gold)}")

    # coref SadSamLink -> CandidateLink (matched_text unknown for anaphora -> use comp name)
    cand=[]
    for lk in coref_links:
        s=sent_map.get(lk.sentence_number)
        cand.append(CandidateLink(lk.sentence_number, s.text if s else "", lk.component_name,
                                  lk.component_id, lk.component_name, source="coreference"))
    bundles={(c.sentence_number,c.component_id): linker._build_evidence_bundle(c, sent_map) for c in cand}
    approved,_=linker._validate_with_evidence(cand, bundles, components, sent_map)
    keep_keys={(c.sentence_number,c.component_id) for c in approved}
    kept=[lk for lk in coref_links if (lk.sentence_number,lk.component_id) in keep_keys]

    for lk in coref_links:
        if (lk.sentence_number,lk.component_id) not in keep_keys:
            is_tp=(lk.sentence_number,lk.component_id) in gold
            print(f"    REJECT S{lk.sentence_number}->{lk.component_name} [{'TP' if is_tp else 'FP'}]")
    kp=pairs_of(kept)
    print(f"  coref AFTER: {len(kept)} | TP={len(kp&gold)} FP={len(kp-gold)}")

    before=dedup(seed_links+entity_links+coref_links); after=dedup(seed_links+entity_links+kept)
    b=metrics(pairs_of(before),gold); a=metrics(pairs_of(after),gold)
    print(f"  FINAL before: P={b[3]*100:.1f} R={b[4]*100:.1f} F1={b[5]*100:.1f} (TP={b[0]} FP={b[1]} FN={b[2]})")
    print(f"  FINAL after : P={a[3]*100:.1f} R={a[4]*100:.1f} F1={a[5]*100:.1f} (TP={a[0]} FP={a[1]} FN={a[2]})")
    macro_before.append(b[5]); macro_after.append(a[5])

print(f"\n===== MACRO ({len(DATASETS)} ds) =====")
print(f"  before F1: {sum(macro_before)/len(macro_before)*100:.1f}")
print(f"  after  F1: {sum(macro_after)/len(macro_after)*100:.1f}")
