"""Exact pipeline score for a judging arm: its kept pairs unioned with the recorded
full-name and partial-name links of the same run."""
import pickle, csv, glob, json, os, sys, statistics as st
BASE="/mnt/hostshare/ardoco-home/alinker-replication-package"; sys.path.insert(0,BASE+"/approach/src")
G={"mediastore":"mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv","teastore":"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv","teammates":"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv","bigbluebutton":"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv","jabref":"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"}
M={"terra":"audit_e2e_s82hy_r*_20260820","luna":"audit_e2e_s82luna_r*_20260820"}
def gold(p):
    with open(os.path.join(BASE,"benchmark",G[p])) as f:
        return {(int(r["sentence"]),r["modelElementID"]) for r in csv.DictReader(f)}
def scores(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    f1=2*P*R/(P+R) if P+R else 0.0; f2=5*P*R/(4*P+R) if 4*P+R else 0.0
    return f1*100,f2*100
def run(model, dumpfile=None):
    runs=sorted(glob.glob(os.path.join(BASE,"results",M[model],"phase_states")))
    dump=json.load(open(dumpfile)) if dumpfile else None
    per=[]
    for i,r in enumerate(runs,1):
        f1s=[];f2s=[];TP=FP=0
        for proj in G:
            g=gold(proj); links=set()
            for stage in ("full_name","partial_name","coreference"):
                fn=os.path.join(r,"s_linker82","openai",proj,f"linker_{stage}.pkl")
                if not os.path.exists(fn): continue
                if stage=="coreference" and dump is not None:
                    links |= {tuple(x) for x in dump.get(f"run{i}",{}).get(proj,[])}
                else:
                    links |= {(l.sentence_number,l.component_id) for l in pickle.load(open(fn,"rb"))["links"]}
            tp=len(links&g); fp=len(links)-tp; fn_=len(g)-tp
            TP+=tp; FP+=fp
            a,b=scores(tp,fp,fn_); f1s.append(a); f2s.append(b)
        per.append((st.mean(f1s),st.mean(f2s),TP,FP))
    return per
if __name__=="__main__":
    label=sys.argv[1]
    for model in M:
        dump=sys.argv[2].replace("MODEL",model) if len(sys.argv)>2 else None
        if dump and not os.path.exists(dump): print(f"{model}: no dump"); continue
        per=run(model,dump)
        f1=st.mean(x[0] for x in per); f2=st.mean(x[1] for x in per)
        print(f"{label:<28} {model:<6} macroF1 {f1:6.2f}  macroF2 {f2:6.2f}  "
              f"TP {st.mean(x[2] for x in per):6.1f}  FP {st.mean(x[3] for x in per):6.1f}")
