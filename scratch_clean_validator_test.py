"""Test a CLEAN, full-LLM, single-question validator for the multi_framing union.

No regex mention_type, no dotted-path detector, no per-class special rules. ONE
general principle the LLM applies: a name inside a code identifier / package path
is a code location (NOT a reference); a name in ordinary prose IS a reference,
whether it describes the component acting or merely relating to other parts.

Replaces s17e's 2-pass + generic-filter validation. Run on FROZEN layer3 union
candidates; coref held fixed from final.pkl. If this matches/beats s17e with TM
dotted-path FPs gone and no BBB/JAB regression -> it becomes s17f's validator.
"""
import os, sys, pickle, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker17e import SLinker17e
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
from llm_sad_sam.llm_client import LLMBackend

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker17e")
DS = {
 "mediastore":("mediastore/text_2016/mediastore.txt","mediastore/model_2016/pcm/ms.repository","mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
 "teastore":("teastore/text_2020/teastore.txt","teastore/model_2020/pcm/teastore.repository","teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
 "teammates":("teammates/text_2021/teammates.txt","teammates/model_2021/pcm/teammates.repository","teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
 "bigbluebutton":("bigbluebutton/text_2021/bigbluebutton.txt","bigbluebutton/model_2021/pcm/bbb.repository","bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
 "jabref":("jabref/text_2021/jabref.txt","jabref/model_2021/pcm/jabref.repository","jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}
def gold(p):
    g=set()
    for r in csv.DictReader(open(p)):
        c,s=r.get('modelElementID','').strip(),r.get('sentence','').strip()
        if c and s: g.add((int(s),c))
    return g
def met(pairs,g):
    tp=len(pairs&g);fp=len(pairs-g);fn=len(g-pairs)
    p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f=2*p*r/(p+r) if p+r else 0
    return tp,fp,fn,p,r,f

backend = LLMBackend.OPENAI if os.environ.get("LLM_BACKEND","").lower()=="openai" else LLMBackend.CLAUDE
lk = SLinker17e(backend=backend)
print(f"Backend: {backend}")

# ── the clean, single, general validation principle ────────────────────────────
def clean_validate(candidates, components, sent_map):
    comp_names = get_comp_names(components)
    approved = []
    for start in range(0, len(candidates), 25):
        batch = candidates[start:start+25]
        cases = []
        for i,c in enumerate(batch):
            s = sent_map.get(c.sentence_number)
            prev = sent_map.get(c.sentence_number-1)
            p = f"[prev: {prev.text[:70]}] " if prev else ""
            cases.append(f'Case {i+1}: candidate component "{c.component_name}"\n  {p}"{s.text if s else ""}"')
        prompt = f"""Decide, for each case, whether the sentence REFERS TO the named component as an element of the software system.

COMPONENTS: {', '.join(comp_names)}

Answer YES when the component name is used in ordinary prose to refer to that component — whether the sentence says what it does, what it contains, or how it relates to other components.

Answer NO when the name is NOT a prose reference to the component, specifically:
- it appears only as part of a code identifier, file path, or package name (for example a token like a.b.name or name.x);
- it is used only in its ordinary English meaning, not as the name of this component;
- it names a different component that merely shares a word.

CASES:
{chr(10).join(cases)}

Return JSON: {{"results": [{{"case": 1, "refers": true, "reason": "brief"}}]}}
JSON only:"""
        data=None
        for _ in range(2):
            data=lk.llm.extract_json(lk.llm.query(prompt, timeout=180))
            if data and data.get("results"): break
        rm={}
        if data:
            for r in data.get("results",[]):
                ci=r.get("case")
                if isinstance(ci,int): rm[ci-1]=r
        for i,c in enumerate(batch):
            r=rm.get(i,{})
            if r.get("refers", True):   # default-keep on missing
                approved.append(c)
    return approved

mb,ma=[],[]
for ds,(t,m,gp) in DS.items():
    l1=pickle.load(open(CACHE/ds/"layer1.pkl","rb")); l3=pickle.load(open(CACHE/ds/"layer3.pkl","rb"))
    fin=pickle.load(open(CACHE/ds/"final.pkl","rb"))["final"]
    cands=l3["candidates"]
    sents=load_sentences(str(BASE/t)); sm=build_sent_map(sents); g=gold(BASE/gp)
    comps=parse_pcm_repository(str(BASE/m))
    lk.model_knowledge=l1["model_knowledge"]; lk.doc_knowledge=l1["doc_knowledge"]
    coref_final=[l for l in fin if l.source=="coreference"]
    before=set((l.sentence_number,l.component_id) for l in fin)

    new_val=clean_validate(cands, comps, sm)
    after=set((l.sentence_number,l.component_id) for l in new_val) | \
          set((l.sentence_number,l.component_id) for l in coref_final)

    base_mf=set((l.sentence_number,l.component_id) for l in fin if l.source=="multi_framing")
    new_mf=set((l.sentence_number,l.component_id) for l in new_val)
    for sn,cid in sorted(base_mf-new_mf):
        print(f"    {ds}: DROP mf S{sn} [{'TP' if (sn,cid) in g else 'FP'}]")
    b=met(before,g); a=met(after,g)
    print(f"  {ds}: {b[5]*100:.1f} (P{b[3]*100:.0f} R{b[4]*100:.0f} FP{b[1]}) -> {a[5]*100:.1f} (P{a[3]*100:.0f} R{a[4]*100:.0f} FP{a[1]})")
    mb.append(b[5]); ma.append(a[5])
print(f"\nMACRO: {sum(mb)/5*100:.1f} -> {sum(ma)/5*100:.1f}")
