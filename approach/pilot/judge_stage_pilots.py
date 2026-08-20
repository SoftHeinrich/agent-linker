"""Stage pilots for the judge-calibration round, on recorded inputs.

Replays ONE judging stage of the variant under test against the checkpoints of a
recorded run, so an arm costs that stage's calls and nothing else. Pair it with
`pilot/pipeline_exact.py`, which substitutes the kept pairs back into the same run's
other stages and produces an exact pipeline score rather than a projection.

  partial : s83's denotation judge over the same scan, scored against gold
  coref   : s83's coreference judge over s82's own recorded resolutions

Usage: pilot83.py {partial|coref} <model-tag> <runs>
"""
import csv, glob, json, os, pickle, sys, collections
BASE="/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE+"/approach/src")
os.environ.setdefault("PHASE_CACHE_DIR","/tmp/pilot83_cache")
from llm_sad_sam.linkers.experimental.s_linker85 import SLinker85 as Variant
from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map

MODE, TAG, RUNS = sys.argv[1], sys.argv[2], int(sys.argv[3])
RECORDED={"terra":"audit_e2e_s82hy_r*_20260820","luna":"audit_e2e_s82luna_r*_20260820"}[TAG]
P={"mediastore":("mediastore/text_2016/mediastore.txt","mediastore/model_2016/pcm/ms.repository","mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
   "teastore":("teastore/text_2020/teastore.txt","teastore/model_2020/pcm/teastore.repository","teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
   "teammates":("teammates/text_2021/teammates.txt","teammates/model_2021/pcm/teammates.repository","teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
   "bigbluebutton":("bigbluebutton/text_2021/bigbluebutton.txt","bigbluebutton/model_2021/pcm/bbb.repository","bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
   "jabref":("jabref/text_2021/jabref.txt","jabref/model_2021/pcm/jabref.repository","jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv")}
def gold(gp):
    with open(os.path.join(BASE,"benchmark",gp)) as f:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(f)}
recorded=sorted(glob.glob(os.path.join(BASE,"results",RECORDED,"phase_states")))

def state(run_idx, proj, phase):
    fn=os.path.join(recorded[run_idx % len(recorded)],"s_linker82","openai",proj,f"{phase}.pkl")
    return pickle.load(open(fn,"rb")) if os.path.exists(fn) else None

tot=collections.Counter(); base=collections.Counter(); DUMP={}
for r in range(RUNS):
    for proj,(t,mo,gp) in P.items():
        comps=parse_pcm_repository(os.path.join(BASE,"benchmark",mo))
        sents=load_sentences(os.path.join(BASE,"benchmark",t))
        sm=build_sent_map(sents); g=gold(gp)
        lk=Variant(backend=LLMBackend.OPENAI)
        k=state(r, proj, "knowledge")
        lk.doc_knowledge=k["doc_knowledge"] if k else None
        if MODE=="partial":
            links,_=lk._run_partial_name_linker(sents,comps,sm)
            kept={(l.sentence_number,l.component_id) for l in links}
            rec=state(r, proj, "linker_partial_name")
            reck={(l.sentence_number,l.component_id) for l in rec["links"]} if rec else set()
        else:
            rec=state(r, proj, "linker_coreference")
            if not rec: continue
            meta={(m["sentence"],m["component_id"]): m for m in rec["feedback"]["metadata"]}
            n2i={c.name:c.id for c in comps}; i2n={v:k2 for k2,v in n2i.items()}
            raw=[SadSamLink(s_,c_,i2n[c_],source="coreference") for (s_,c_) in meta if c_ in i2n]
            appr,_=lk._validate_coref_links(raw, sm, comps, meta)
            kept={(l.sentence_number,l.component_id) for l in appr}
            reck={(l.sentence_number,l.component_id) for l in rec["links"]}
        DUMP.setdefault(f"run{r+1}", {})[proj]=sorted([list(x) for x in kept])
        tg=sum(1 for x in kept if x in g); bg=sum(1 for x in reck if x in g)
        tot["g"]+=tg; tot["n"]+=len(kept)-tg
        base["g"]+=bg; base["n"]+=len(reck)-bg
        print(f"run{r+1} {proj:<14} arm {tg:3d}g/{len(kept)-tg:3d}n   s82(recorded) {bg:3d}g/{len(reck)-bg:3d}n", flush=True)
print(f"\n{MODE} on {TAG}, {RUNS} runs, per five-project run:")
print(f"  arm            gold {tot['g']/RUNS:6.1f}  spurious {tot['n']/RUNS:6.1f}")
print(f"  s82 (recorded) gold {base['g']/RUNS:6.1f}  spurious {base['n']/RUNS:6.1f}")
import json as _j
_out=os.environ.get("PILOT_DUMP")
if _out:
    _j.dump(DUMP, open(_out,"w"))
    print("kept pairs written to", _out)
