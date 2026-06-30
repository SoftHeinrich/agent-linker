#!/usr/bin/env python3
"""Pilot Step 2 — can an LLM DECIDE the route?

Binary router on all 196 doc-code gold sentences (blind to gold/project):
  CODE         = sentence names concrete code structure (package/class/file/
                 method/exception/config) -> route DIRECT to code.
  ARCHITECTURE = sentence describes a component/role/interaction -> route TRANSITIVE.
Ground truth: direct-only (no SAD-SAM gold) => should be CODE; arch-anchored => ARCHITECTURE.
Zero-shot, taboo-safe (generic instructions, no benchmark component names).
Batched JSON to keep call count low; results cached to disk.
"""
import csv, glob, importlib.util, json, os, sys, hashlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
MINI=Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec=importlib.util.spec_from_file_location("metrics",MINI); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
from llm_sad_sam.llm_client import LLMClient, LLMBackend

BENCH=M.BENCHMARK; PROJECTS=M.PROJECTS
CACHE=Path(__file__).resolve().parent.parent / "cache" / "router_cache.json"

def sentences(p):
    h=glob.glob(str(BENCH/p/"text_*"/f"{p}.txt")); d={}
    if h:
        for i,l in enumerate(open(h[0],errors="replace"),1): d[str(i)]=l.strip()
    return d

# ---- build labeled set ----
items=[]   # (gid, project, sent_id, text, true_label)
for p in PROJECTS:
    sc_raw=M.load_gs_sad_code_raw(p); ss=M.load_gs_sad_sam(p)
    sc_sents={s for s,_ in sc_raw}; ss_sents={s for _,s in ss}
    S=sentences(p)
    for s in sorted(sc_sents, key=lambda v:int(v) if v.isdigit() else 0):
        true="CODE" if s not in ss_sents else "ARCH"
        items.append((f"{p}:{s}", p, s, S.get(s,""), true))

PROMPT_HEAD = (
    "You triage software-documentation sentences for trace-link recovery. "
    "For each sentence choose ONE route:\n"
    '- "ARCH": the sentence describes a high-level component, its responsibility, '
    "or how components interact at the architecture level.\n"
    '- "CODE": the sentence refers to concrete code-level structure such as a '
    "package name, class name, file name, method, exception, or configuration "
    "file (often dotted identifiers, CamelCase names, or file extensions).\n"
    "Choose CODE only when a specific code-level identifier/artifact is named; "
    "otherwise choose ARCH.\n"
    'Reply with ONLY a JSON array of {"id": <int>, "route": "ARCH"|"CODE"}.\n\n'
    "Sentences:\n")

def run_batch(client, batch):
    prompt = PROMPT_HEAD + "\n".join(f'{i}. {t}' for i,(_,t) in enumerate(batch))
    r = client.query(prompt, timeout=120)
    if not r.success:
        return {}
    txt=r.text.strip()
    a=txt.find("["); b=txt.rfind("]")
    if a<0 or b<0: return {}
    try:
        arr=json.loads(txt[a:b+1])
    except Exception:
        return {}
    out={}
    for o in arr:
        try: out[int(o["id"])]=str(o["route"]).upper().strip()
        except Exception: pass
    return out

def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo=[it for it in items if it[0] not in cache]
    if todo:
        os.environ.setdefault("OPENAI_MODEL_NAME","gpt-5.4")
        client=LLMClient(backend=LLMBackend.OPENAI, model="gpt-5.4", enable_logging=False)
        B=12
        for k in range(0,len(todo),B):
            chunk=todo[k:k+B]
            res=run_batch(client,[(it[0],it[3]) for it in chunk])
            for j,it in enumerate(chunk):
                route=res.get(j,"?")
                cache[it[0]]={"route":route,"true":it[4],"proj":it[1],"sid":it[2],"text":it[3]}
            CACHE.write_text(json.dumps(cache))
            print(f"  batch {k//B+1}/{(len(todo)+B-1)//B} done ({len(res)}/{len(chunk)} parsed)", file=sys.stderr)
    # ---- score ----
    tp=fp=tn=fn=unk=0   # positive class = CODE (direct-only)
    rows=[cache[it[0]] for it in items]
    for r in rows:
        pred=r["route"]; pos = r["true"]=="CODE"
        if pred not in ("ARCH","CODE"): unk+=1; continue
        pc = pred=="CODE"
        if pos and pc: tp+=1
        elif pos and not pc: fn+=1
        elif (not pos) and pc: fp+=1
        else: tn+=1
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0; acc=(tp+tn)/max(1,(tp+tn+fp+fn))
    print("\n=== ROUTER (positive = CODE / direct-only) ===")
    print(f"  n={len(rows)} unparsed={unk}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} accuracy={acc:.3f}")
    # per-project FP rate (over-firing on clean projects)
    print("\n  per-project: predicted CODE / total  (gold direct-only)")
    by=defaultdict(lambda:[0,0,0])
    for r in rows:
        b=by[r["proj"]]; b[1]+=1
        if r["route"]=="CODE": b[0]+=1
        if r["true"]=="CODE": b[2]+=1
    for p in PROJECTS:
        b=by[p]; print(f"    {p:<14} CODE={b[0]:>3}/{b[1]:<3}  gold_direct={b[2]}")
    # disagreements
    print("\n  --- FALSE POSITIVES (router said CODE but gold arch-anchored) ---")
    for r in rows:
        if r["true"]=="ARCH" and r["route"]=="CODE":
            print(f"    {r['proj']:<12} s{r['sid']:>3}: {r['text'][:92]}")
    print("\n  --- FALSE NEGATIVES (router said ARCH but gold direct-only) ---")
    for r in rows:
        if r["true"]=="CODE" and r["route"]=="ARCH":
            print(f"    {r['proj']:<12} s{r['sid']:>3}: {r['text'][:92]}")

if __name__=="__main__":
    main()
