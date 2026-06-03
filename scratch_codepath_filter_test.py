"""Test a CLEAN full-LLM code-path FILTER (not a re-validation).

Keep s17e's proven validation untouched. Add ONE narrow LLM question over the
final multi_framing links: is the name used ONLY as a code token (dotted id /
package / path), with no prose reference? Drop only those. No regex, no
mention_type, one rule. This is the full-LLM equivalent of the dotted-path
detector — scoped by the QUESTION's narrowness, not by code.
"""
import os, sys, pickle, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker17e import SLinker17e
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

def codepath_filter(links, sent_map):
    """Return links to KEEP. Drop only confident code-token-only mentions."""
    keep=[]
    for start in range(0,len(links),25):
        batch=links[start:start+25]
        cases=[]
        for i,l in enumerate(batch):
            s=sent_map.get(l.sentence_number)
            cases.append(f'Case {i+1}: component "{l.component_name}"\n  "{s.text if s else ""}"')
        prompt=f"""Each case links a component to a sentence that contains the component's name.

Flag a case as CODE_TOKEN only when EVERY occurrence of the name is a segment of a longer DOT-SEPARATED identifier — a package or namespace path where the name is joined to other segments by dots (for example name.x, x.name, or a.name.b) — AND the sentence does NOT also mention the component by name in ordinary prose. Such a dotted path points to a sub-package or code location, not the component itself.

NOT a code token (answer false):
- the name used as a normal word in prose, even while describing structure, contents, or relationships;
- a hyphenated or compound product/deployment name (like a-b or a-b-c) — that IS the component's name, not a code path;
- the name standing alone, even in lowercase.

CASES:
{chr(10).join(cases)}

Return JSON: {{"results": [{{"case": 1, "code_token": false, "reason": "brief"}}]}}
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
        for i,l in enumerate(batch):
            r=rm.get(i,{})
            if r.get("code_token", False) is True:
                s=sent_map.get(l.sentence_number)
                print(f"    DROP S{l.sentence_number} {l.component_name}: {r.get('reason','')[:70]}")
            else:
                keep.append(l)
    return keep

mb,ma=[],[]
for ds,(t,m,gp) in DS.items():
    fin=pickle.load(open(CACHE/ds/"final.pkl","rb"))["final"]
    sents=load_sentences(str(BASE/t)); sm=build_sent_map(sents); g=gold(BASE/gp)
    before=set((l.sentence_number,l.component_id) for l in fin)
    mf=[l for l in fin if l.source=="multi_framing"]; other=[l for l in fin if l.source!="multi_framing"]
    print(f"=== {ds} ===")
    kept_mf=codepath_filter(mf, sm)
    after=set((l.sentence_number,l.component_id) for l in kept_mf+other)
    # label drops
    for sn,cid in sorted(before-after):
        print(f"      -> [{'TP' if (sn,cid) in g else 'FP'}] dropped")
    b=met(before,g); a=met(after,g)
    print(f"  {ds}: {b[5]*100:.1f} (FP{b[1]}) -> {a[5]*100:.1f} (FP{a[1]})")
    mb.append(b[5]); ma.append(a[5])
print(f"\nMACRO: {sum(mb)/5*100:.1f} -> {sum(ma)/5*100:.1f}")
