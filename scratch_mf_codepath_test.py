"""Checkpoint-level test of FIX A: Cause-B code-path axiom in s17e validation.

The multi_framing dotted-path FPs (logic.api, storage.entity, x.e2e, client.util)
pass validation because the Cause-B code-path exclusion never reached s17. Here we
append that axiom to the validation rules and re-run Phase 4 validation on the
FROZEN union candidates (layer3.pkl). Coref (Phase 5) is held fixed from final.pkl.
No extraction re-run -> isolates the validation-rule change.
"""
import os, sys, pickle, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker17e import SLinker17e
from llm_sad_sam.llm_client import LLMBackend

CAUSE_B_RULE = (
    "Code-path / package-structure exclusion: a component name that appears ONLY as a "
    "segment of a dotted code identifier (such as 'x.name', 'name.y', or 'a.b.name') or "
    "ONLY inside a sentence describing package, module, or class-file structure is NOT an "
    "architectural reference — it names a code location, not the component acting. Reject "
    "such a case UNLESS the same sentence independently describes the component performing "
    "an operation, providing a service, or taking part in runtime behavior."
)

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker17e")
DATASETS = {
    "teammates": (BASE/"teammates/text_2021/teammates.txt", BASE/"teammates/model_2021/pcm/teammates.repository",
                  BASE/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": (BASE/"jabref/text_2021/jabref.txt", BASE/"jabref/model_2021/pcm/jabref.repository",
                  BASE/"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": (BASE/"bigbluebutton/text_2021/bigbluebutton.txt", BASE/"bigbluebutton/model_2021/pcm/bbb.repository",
                  BASE/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}

def load_gold(p):
    g=set()
    for row in csv.DictReader(open(p)):
        cid,sn=row.get("modelElementID","").strip(),row.get("sentence","").strip()
        if cid and sn: g.add((int(sn),cid))
    return g

def dedup(pairs_links):
    seen,out=set(),[]
    for sn,cid,src in pairs_links:
        if (sn,cid) not in seen: seen.add((sn,cid)); out.append((sn,cid,src))
    return out

def metrics(pairs,gold):
    tp=len(pairs&gold); fp=len(pairs-gold); fn=len(gold-pairs)
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0
    return tp,fp,fn,p,r,f1

backend = LLMBackend.OPENAI if os.environ.get("LLM_BACKEND","").lower()=="openai" else LLMBackend.CLAUDE
linker = SLinker17e(backend=backend)
linker._VALIDATION_RULES = linker._VALIDATION_RULES + "\n\n" + CAUSE_B_RULE   # FIX A
print(f"Backend: {backend}  (Cause-B axiom appended to validation rules)")

mb, ma = [], []
for ds,(txt,mdl,goldp) in DATASETS.items():
    l1=pickle.load(open(CACHE/ds/"layer1.pkl","rb"))
    l3=pickle.load(open(CACHE/ds/"layer3.pkl","rb"))
    fin=pickle.load(open(CACHE/ds/"final.pkl","rb"))["final"]
    candidates=l3["candidates"]; base_validated=l3["validated"]
    sentences=load_sentences(str(txt)); components=parse_pcm_repository(str(mdl))
    sent_map=build_sent_map(sentences); gold=load_gold(goldp)
    linker.model_knowledge=l1["model_knowledge"]; linker.doc_knowledge=l1["doc_knowledge"]
    linker._current_text_path=None

    coref_final=[l for l in fin if l.source=="coreference"]

    # BEFORE final = baseline final.pkl
    before=set((l.sentence_number,l.component_id) for l in fin)

    # AFTER: re-validate framing candidates with Cause-B axiom, keep coref fixed
    bundles={(c.sentence_number,c.component_id): linker._build_evidence_bundle(c, sent_map) for c in candidates}
    new_validated,_=linker._validate_with_evidence(candidates, bundles, components, sent_map)
    after_links=dedup([(l.sentence_number,l.component_id,"mf") for l in new_validated]
                      +[(l.sentence_number,l.component_id,"coref") for l in coref_final])
    after=set((sn,cid) for sn,cid,_ in after_links)

    # which mf links dropped vs baseline validated
    base_mf=set((l.sentence_number,l.component_id) for l in base_validated)
    new_mf=set((l.sentence_number,l.component_id) for l in new_validated)
    dropped=base_mf-new_mf
    for sn,cid in sorted(dropped):
        print(f"    {ds}: DROP mf S{sn} {cid} [{'TP' if (sn,cid) in gold else 'FP'}]")

    b=metrics(before,gold); a=metrics(after,gold)
    print(f"  {ds}: BEFORE F1={b[5]*100:.1f} (TP={b[0]} FP={b[1]} FN={b[2]})  ->  AFTER F1={a[5]*100:.1f} (TP={a[0]} FP={a[1]} FN={a[2]})")
    mb.append(b[5]); ma.append(a[5])

print(f"\nMACRO ({len(DATASETS)} ds): before {sum(mb)/len(mb)*100:.1f}  ->  after {sum(ma)/len(ma)*100:.1f}")
