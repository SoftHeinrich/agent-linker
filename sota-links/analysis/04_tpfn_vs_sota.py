"""TP/FN decomposition of our approach against SWATTR and ArTEMiS (model-doc / SAD-SAM).

Answers, from committed data only (no LLM calls):

  1. How do TP / FP / FN split per system and per project?
  2. Which gold links does each system get that the other misses (paired, McNemar)?
  3. WHAT KIND of gold link is it?  Every gold link is classified by the reference
     form the sentence uses -- the same taxonomy the approach is built around --
     and recall is reported per form.  This is where the gap lives.
  4. Which of our mechanisms recovers each form (linker `source` column), and
     which ablation arm loses it (no-knowledge / one-call floor)?
  5. What do the false positives look like on each side?

Reported arm: s_linker110, backend terra (GPT-5.6), majority-of-3 runs.
ArTEMiS: gpt-5.6 terra, majority-of-3 runs.  SWATTR: deterministic, single run.
Majority-of-3 is used so all three systems are compared as single link sets;
per-run and union/intersection sensitivity is printed in section 1b.

Usage:  python3 analysis/04_tpfn_vs_sota.py [--md]
Env:    ARDOCO_BENCHMARK (SAD text + PCM models), ALINKER_RESULTS (run dirs)
"""
import os, sys, csv, re, glob, math, argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common as C  # noqa: E402

RL = os.path.dirname(HERE)
RUNS = os.environ.get("ALINKER_RESULTS",
                      os.path.normpath(os.path.join(RL, "..", "results")))
OURS_CFG = "terra_s110"
ARTEMIS_CFG = "terra_5.6"
P = C.PROJECTS

FORMS = ["F1 exact-name", "F2 partial-name", "F3 coreferent", "F4 renamed"]
FORM_DESC = {
    "F1 exact-name":   "sentence writes the component name (verbatim after case/separator folding)",
    "F2 partial-name": "sentence writes some but not all words of a multi-word name",
    "F3 coreferent":   "sentence writes no word of the name; the name occurs in the 3 preceding sentences",
    "F4 renamed":      "no word of the name in the sentence or its 3-sentence context (document uses another term)",
}


# ── link sets ────────────────────────────────────────────────────────────────
def majority(sets, k=2):
    c = Counter()
    for s in sets:
        c.update(s)
    return {x for x, n in c.items() if n >= k}


def ours_run(proj, r):
    return C.links(f"{RL}/model-doc/aalinker/{OURS_CFG}/run{r}/{proj}.csv")


def artemis_run(proj, r):
    return C.links(f"{RL}/model-doc/artemis/{ARTEMIS_CFG}/run{r}/{proj}.csv")


def swattr_run(proj):
    return C.links(f"{RL}/model-doc/swattr-{proj}.csv")


SYS = {"ours": {}, "artemis": {}, "swattr": {}}
GOLD = {}
for proj in P:
    GOLD[proj] = C.gold(proj)
    SYS["ours"][proj] = majority([ours_run(proj, r) for r in (1, 2, 3)])
    SYS["artemis"][proj] = majority([artemis_run(proj, r) for r in (1, 2, 3)])
    SYS["swattr"][proj] = swattr_run(proj)
SYSTEMS = ["ours", "artemis", "swattr"]


# ── reference-form classification of every gold link ─────────────────────────
def _ctx(sents, sid, k=3):
    return " ".join(sents.get(sid - i, "") for i in range(1, k + 1))


def classify(name, sent, ctx):
    lab, ev = C.lexical_class(name, sent)
    if lab == "LEXICAL-full":
        return "F1 exact-name", ev
    if lab == "LEXICAL-partial":
        return "F2 partial-name", ev
    clab, cev = C.lexical_class(name, ctx)
    if clab != "SEMANTIC-none":
        return "F3 coreferent", "ctx:" + cev
    return "F4 renamed", ev


ROWS = []
for proj in P:
    n2, sents = C.id_to_name(proj), C.sentences(proj)
    for (sid, tid) in sorted(GOLD[proj]):
        nm, st = n2.get(tid, tid), sents.get(sid, "")
        form, ev = classify(nm, st, _ctx(sents, sid))
        ROWS.append(dict(proj=proj, sid=sid, tid=tid, name=nm, sent=st, form=form, ev=ev,
                         **{s: (sid, tid) in SYS[s][proj] for s in SYSTEMS}))


# ── our internal mechanism: which linker proposed each recovered link ────────
SRC = defaultdict(Counter)
for r in (1, 2, 3):
    for proj in P:
        f = f"{RUNS}/consolidation_e2e_terra_r{r}_20260825/s_linker110_{proj}_links.csv"
        if os.path.exists(f):
            for row in csv.DictReader(open(f)):
                SRC[(proj, int(row["sentence"]), row["component_id"])][row["source"]] += 1


def srcof(r):
    c = SRC.get((r["proj"], r["sid"], r["tid"]))
    return "+".join(sorted(c)) if c else "—"


def arm(pattern, tag):
    out = {}
    for proj in P:
        c, n = Counter(), 0
        for d in sorted(glob.glob(pattern)):
            f = f"{d}/{tag}_{proj}_links.csv"
            if not os.path.exists(f):
                continue
            n += 1
            c.update({(int(x["sentence"]), x["component_id"]) for x in csv.DictReader(open(f))})
        out[proj] = {x for x, k in c.items() if k >= 2} if n >= 2 else set()
    return out


ARMS = {
    "Full (s110)":    lambda: arm(f"{RUNS}/consolidation_e2e_terra_r?_20260825", "s_linker110"),
    "-knowledge":     lambda: arm(f"{RUNS}/consolidation_noknow_e2e_terra_r?_20260902", "s_linker110_noknow"),
    "-decomposition": lambda: arm(f"{RUNS}/onecall_e2e_terra_r?_20260902", "s_linker110_onecall"),
    "-evidence":      lambda: arm(f"{RUNS}/noevidence_e2e_terra_r?_20260902", "s_linker110_noevidence"),
}


# ── stats ────────────────────────────────────────────────────────────────────
def prf(pred, g):
    tp = len(pred & g)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(g) if g else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


O = []
def out(s=""):
    O.append(s)
    print(s)


# ── 1. headline ──────────────────────────────────────────────────────────────
out("=" * 100)
out("1. LINK-LEVEL TP / FP / FN  (SAD-SAM, 5 ArDoCo projects, majority-of-3 runs)")
out("=" * 100)
out(f"{'project':15}{'gold':>5}   " + "".join(f"{s + ' TP/FP/FN':>22}" for s in SYSTEMS))
tot = {s: [0, 0, 0] for s in SYSTEMS}
macro = {s: [] for s in SYSTEMS}
for proj in P:
    g = GOLD[proj]
    line = f"{proj:15}{len(g):5d}   "
    for s in SYSTEMS:
        st = SYS[s][proj]
        tp, fp, fn = len(st & g), len(st - g), len(g - st)
        tot[s][0] += tp; tot[s][1] += fp; tot[s][2] += fn
        macro[s].append(prf(st, g))
        line += f"{tp:8d}/{fp:3d}/{fn:3d}   "
    out(line)
out()
out(f"{'system':10}{'TP':>5}{'FP':>5}{'FN':>5}   {'micro-P':>8}{'micro-R':>8}{'micro-F1':>9}   "
    f"{'macro-P':>8}{'macro-R':>8}{'macro-F1':>9}")
for s in SYSTEMS:
    tp, fp, fn = tot[s]
    p, r = tp / (tp + fp), tp / (tp + fn)
    mp = sum(x[0] for x in macro[s]) / len(P)
    mr = sum(x[1] for x in macro[s]) / len(P)
    mf = sum(x[2] for x in macro[s]) / len(P)
    out(f"{s:10}{tp:5d}{fp:5d}{fn:5d}   {p:8.3f}{r:8.3f}{2 * p * r / (p + r):9.3f}   {mp:8.3f}{mr:8.3f}{mf:9.3f}")

out()
out("1b. run-to-run sensitivity (recall over 195 gold links)")
for nm, fn_ in (("ours", ours_run), ("artemis", artemis_run)):
    per = [sum(len(GOLD[p_] & fn_(p_, r)) for p_ in P) for r in (1, 2, 3)]
    u = sum(len(GOLD[p_] & (fn_(p_, 1) | fn_(p_, 2) | fn_(p_, 3))) for p_ in P)
    i = sum(len(GOLD[p_] & (fn_(p_, 1) & fn_(p_, 2) & fn_(p_, 3))) for p_ in P)
    m = sum(len(GOLD[p_] & SYS[nm][p_]) for p_ in P)
    out(f"   {nm:8} per-run TP {per}  spread {max(per) - min(per):2d} | majority {m} | union {u} | intersection {i} (spread {u - i})")
out(f"   {'swattr':8} deterministic, TP {tot['swattr'][0]}")

# ── 2. paired comparison ─────────────────────────────────────────────────────
out()
out("=" * 100)
out("2. PAIRED GOLD-LINK PARTITION  (who recovers what; McNemar exact two-sided)")
out("=" * 100)
for base in ("swattr", "artemis"):
    both = ours_only = base_only = neither = 0
    per = {}
    for proj in P:
        g, o, b = GOLD[proj], SYS["ours"][proj], SYS[base][proj]
        per[proj] = (len(g & o & b), len((g & o) - b), len((g & b) - o), len(g - o - b))
        both += per[proj][0]; ours_only += per[proj][1]; base_only += per[proj][2]; neither += per[proj][3]
    out(f"\n   ours vs {base.upper()}:  both={both}  ours-only={ours_only}  {base}-only={base_only}  "
        f"neither={neither}   McNemar p={mcnemar_exact(ours_only, base_only):.3g}  "
        f"(odds {ours_only}:{base_only})")
    for proj in P:
        out(f"     {proj:15} both={per[proj][0]:3d}  ours-only={per[proj][1]:3d}  "
            f"{base}-only={per[proj][2]:3d}  neither={per[proj][3]:3d}")

# ── 3. recall by reference form ──────────────────────────────────────────────
out()
out("=" * 100)
out("3. RECALL BY REFERENCE FORM  (every gold link classified by how the sentence refers)")
out("=" * 100)
for f in FORMS:
    out(f"   {f:18} = {FORM_DESC[f]}")
out()
out(f"{'reference form':18}{'N':>5}   " + "".join(f"{s:>17}" for s in SYSTEMS))
bycat = {}
for f in FORMS:
    sub = [r for r in ROWS if r["form"] == f]
    bycat[f] = sub
    if not sub:
        continue
    line = f"{f:18}{len(sub):5d}   "
    for s in SYSTEMS:
        n = sum(1 for r in sub if r[s])
        line += f"{n:6d} ({n / len(sub) * 100:5.1f}%)  "
    out(line)
line = f"{'ALL':18}{len(ROWS):5d}   "
for s in SYSTEMS:
    n = sum(1 for r in ROWS if r[s])
    line += f"{n:6d} ({n / len(ROWS) * 100:5.1f}%)  "
out(line)
tail = [r for r in ROWS if r["form"] != "F1 exact-name"]
line = f"{'  tail (F2-F4)':18}{len(tail):5d}   "
for s in SYSTEMS:
    n = sum(1 for r in tail if r[s])
    line += f"{n:6d} ({n / len(tail) * 100:5.1f}%)  "
out(line)

out()
out("3b. where the recall gap comes from (gold links recovered, ours minus baseline)")
out(f"{'reference form':18}{'N':>5}{'ours-SWATTR':>14}{'ours-ArTEMiS':>14}")
for f in FORMS + ["ALL"]:
    sub = ROWS if f == "ALL" else bycat[f]
    if not sub:
        continue
    o = sum(1 for r in sub if r["ours"])
    d1 = o - sum(1 for r in sub if r["swattr"])
    d2 = o - sum(1 for r in sub if r["artemis"])
    out(f"{f:18}{len(sub):5d}{d1:+14d}{d2:+14d}")

# ── 4. our residual FNs ──────────────────────────────────────────────────────
out()
out("=" * 100)
out("4. OUR FALSE NEGATIVES  (gold links we miss, majority-of-3)")
out("=" * 100)
for r in ROWS:
    if not r["ours"]:
        out(f"   {r['proj']:14} s{r['sid']:<4} {r['name']:<18} {r['form']:<16} "
            f"artemis={'Y' if r['artemis'] else 'n'} swattr={'Y' if r['swattr'] else 'n'} "
            f"proposed-by={srcof(r)}")
        out(f"        \"{r['sent'][:130]}\"")

for base in ("swattr", "artemis"):
    sub = [r for r in ROWS if r[base] and not r["ours"]]
    out(f"\n   of these, {base} recovers {len(sub)}: " +
        ", ".join(f"{r['proj'][:4]}/s{r['sid']}/{r['name']}" for r in sub))

# ── 5. false-positive profile ────────────────────────────────────────────────
out()
out("=" * 100)
out("5. FALSE-POSITIVE PROFILE  (what kind of error each system makes on the FP side)")
out("=" * 100)
out("   anchored      = the predicted component's name (or part) occurs in the sentence -> over-firing on a real mention")
out("   unanchored    = no word of the name in the sentence            -> invented or propagated from context")
out("   sibling-swap  = the sentence's own gold target was missed and shares a name word with the prediction")
out("   code-ident    = a name word occurs only inside a dotted/hyphenated identifier (logic.api, bbb-html5)")
out("   drift         = unanchored AND the same component was a TP within the 3 preceding sentences")
out("   lex-overfire  = code-ident OR the whole single-word name occurs only in lower case (ordinary-vocabulary use)")
out()
out(f"{'system':10}{'FP':>5}{'anchored':>11}{'unanchored':>12}{'sibling-swap':>14}{'code-ident':>12}{'drift':>8}{'lex-overfire':>15}")
FPDETAIL = defaultdict(list)
for s in SYSTEMS:
    n = anc = sib = cid = dri = lex = 0
    for proj in P:
        g, n2, sents, pred = GOLD[proj], C.id_to_name(proj), C.sentences(proj), SYS[s][proj]
        gold_by_sent = defaultdict(set)
        for sid, tid in g:
            gold_by_sent[sid].add(tid)
        tp_sents = defaultdict(list)
        for sid, tid in pred & g:
            tp_sents[tid].append(sid)
        for (sid, tid) in sorted(pred - g):
            n += 1
            name, st = n2.get(tid, tid), sents.get(sid, "")
            lab, _ = C.lexical_class(name, st)
            missed = [x for x in gold_by_sent[sid] if (sid, x) not in pred]
            if lab != "SEMANTIC-none":
                anc += 1
            idents = re.findall(r"\b[a-zA-Z][\w]*(?:[.\-][\w]+)+\b", st)
            if any(set(C.camel_tokens(n2.get(x, x))) & set(C.camel_tokens(name)) for x in missed):
                sib += 1
                FPDETAIL[(s, "sibling-swap")].append((proj, sid, name, [n2.get(x, x) for x in missed]))
            if any(any(t in i.lower() for t in C.camel_tokens(name)) for i in idents):
                cid += 1
                FPDETAIL[(s, "code-ident")].append((proj, sid, name, st[:70]))
            gw = False
            tk = C.camel_tokens(name)
            if len(tk) == 1:
                occ = re.findall(r"\b" + re.escape(tk[0]) + r"\w*\b", st, re.I)
                gw = bool(occ) and all(o[0].islower() for o in occ)
            if gw or any(any(t in i.lower() for t in tk) for i in idents):
                lex += 1
            if lab == "SEMANTIC-none" and any(0 < sid - ps <= 3 for ps in tp_sents.get(tid, [])):
                dri += 1
                FPDETAIL[(s, "drift")].append((proj, sid, name, st[:70]))
    out(f"{s:10}{n:5d}{anc:7d} ({anc / n * 100:3.0f}%){n - anc:8d} ({(n - anc) / n * 100:3.0f}%)"
        f"{sib:9d} ({sib / n * 100:3.0f}%){cid:7d} ({cid / n * 100:3.0f}%){dri:4d} ({dri / n * 100:3.0f}%){lex:9d} ({lex / n * 100:3.0f}%)")
out("   (sibling-swap / code-ident / drift overlap; they are diagnostic markers, not a partition)")

out()
out("   SWATTR code-identifier FPs (top 12):")
for d in FPDETAIL[("swattr", "code-ident")][:12]:
    out(f"     {d[0]:14} s{d[1]:<4} pred={d[2]:<14} \"{d[3]}\"")
out("   ArTEMiS drift FPs (top 12):")
for d in FPDETAIL[("artemis", "drift")][:12]:
    out(f"     {d[0]:14} s{d[1]:<4} pred={d[2]:<24} \"{d[3]}\"")
out("   our sibling-swap FPs (all):")
for d in FPDETAIL[("ours", "sibling-swap")]:
    out(f"     {d[0]:14} s{d[1]:<4} pred={d[2]:<24} missed gold here = {d[3]}")

# ── 6. component-level and sentence-level silent failure ─────────────────────
out()
out("=" * 100)
out("6. SILENT FAILURE  (a component or a multi-target sentence that gets nothing)")
out("=" * 100)
for s in SYSTEMS:
    miss, totc = [], 0
    for proj in P:
        comps = {t for _, t in GOLD[proj]}
        totc += len(comps)
        got = {t for _, t in SYS[s][proj]}
        n2 = C.id_to_name(proj)
        miss += [f"{proj}:{n2.get(c, c)}" for c in sorted(comps - got)]
    out(f"   {s:10} components with zero recovered links: {len(miss):2d}/{totc}  {miss}")
out()
buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for proj in P:
    cnt = Counter(s for s, _ in GOLD[proj])
    for (sid, tid) in GOLD[proj]:
        b = "1 target" if cnt[sid] == 1 else ("2 targets" if cnt[sid] == 2 else ">=3 targets")
        for s in SYSTEMS:
            buckets[b][s][1] += 1
            if (sid, tid) in SYS[s][proj]:
                buckets[b][s][0] += 1
out(f"   {'sentence has':14}" + "".join(f"{s:>19}" for s in SYSTEMS))
for b in ("1 target", "2 targets", ">=3 targets"):
    line = f"   {b:14}"
    for s in SYSTEMS:
        tp, nn = buckets[b][s]
        line += f"{tp:8d}/{nn:3d} ({tp / nn * 100:3.0f}%)"
    out(line)

# ── 7. mechanism attribution ─────────────────────────────────────────────────
out()
out("=" * 100)
out("7. WHERE OUR RECALL COMES FROM  (mechanism x reference form)")
out("=" * 100)
out("7a. which linker proposed the link we recovered (union of 3 terra runs)")
tab = defaultdict(Counter)
for r in ROWS:
    if r["ours"]:
        tab[r["form"]][srcof(r)] += 1
srcs = sorted({s for f in tab for s in tab[f]})
out(f"   {'reference form':18}" + "".join(f"{s:>26}" for s in srcs))
for f in FORMS:
    out(f"   {f:18}" + "".join(f"{tab[f][s]:>26}" for s in srcs))

out()
out("7b. ablation: recall by reference form when a design decision is removed")
out(f"   {'arm':18}" + "".join(f"{f[3:]:>20}" for f in FORMS) + f"{'ALL':>12}{'FP':>6}")
for nm, mk in ARMS.items():
    a = mk()
    if not any(a[p_] for p_ in P):
        out(f"   {nm:18}  (no run data found -- skipped)")
        continue
    line = f"   {nm:18}"
    for f in FORMS:
        sub = bycat[f]
        n = sum(1 for r in sub if (r["sid"], r["tid"]) in a[r["proj"]])
        line += f"{n:9d}/{len(sub):3d} ({n / len(sub) * 100:3.0f}%)"
    n = sum(1 for r in ROWS if (r["sid"], r["tid"]) in a[r["proj"]])
    fp = sum(len(a[p_] - GOLD[p_]) for p_ in P)
    out(line + f"{n:7d}/{len(ROWS)}{fp:6d}")


# ── 8. mirror backend: are our FNs structural or an operating point? ─────────
out()
out("=" * 100)
out("8. MIRROR BACKEND  (same pipeline, GPT-5.6-luna) -- is a residual FN structural?")
out("=" * 100)
LUNA = {p_: majority([C.links(f"{RL}/model-doc/aalinker/luna_s110/run{r}/{p_}.csv") for r in (1, 2, 3)])
        for p_ in P}
line = f"   {'luna, recall by form':24}"
for f in FORMS:
    sub = bycat[f]
    n = sum(1 for r in sub if (r["sid"], r["tid"]) in LUNA[r["proj"]])
    line += f"{f[3:]}={n}/{len(sub)} ({n / len(sub) * 100:.0f}%)  "
nall = sum(1 for r in ROWS if (r["sid"], r["tid"]) in LUNA[r["proj"]])
fpall = sum(len(LUNA[p_] - GOLD[p_]) for p_ in P)
out(line + f"| ALL={nall}/{len(ROWS)}  FP={fpall}")
rec = [r for r in ROWS if not r["ours"] and (r["sid"], r["tid"]) in LUNA[r["proj"]]]
out(f"   of our {sum(1 for r in ROWS if not r['ours'])} terra FNs, luna recovers {len(rec)}: " +
    ", ".join(f"{r['proj'][:4]}/s{r['sid']}/{r['name']}" for r in rec))
out("   -> the residual FNs are an operating point (recall bought with FP 24 -> %d), not a blind spot." % fpall)
out()
out("   per-run presence of each terra FN (terra r1r2r3 | luna r1r2r3):")
for r in ROWS:
    if r["ours"]:
        continue
    t = "".join("Y" if (r["sid"], r["tid"]) in ours_run(r["proj"], i) else "n" for i in (1, 2, 3))
    l = "".join("Y" if (r["sid"], r["tid"]) in C.links(f"{RL}/model-doc/aalinker/luna_s110/run{i}/{r['proj']}.csv") else "n" for i in (1, 2, 3))
    out(f"     {r['proj']:14} s{r['sid']:<4} {r['name']:<14} {r['form']:<16} terra={t} luna={l}")

# ── write artifacts ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "tpfn_vs_sota"))
args, _ = ap.parse_known_args()
with open(args.out + ".txt", "w") as fh:
    fh.write("\n".join(O) + "\n")
with open(args.out + "_goldlinks.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["project", "sentence_id", "target_id", "component", "reference_form",
                "evidence", "ours", "artemis", "swattr", "our_linker", "sentence"])
    for r in ROWS:
        w.writerow([r["proj"], r["sid"], r["tid"], r["name"], r["form"], r["ev"],
                    int(r["ours"]), int(r["artemis"]), int(r["swattr"]), srcof(r), r["sent"]])
print(f"\n[written] {args.out}.txt   {args.out}_goldlinks.csv")

# ── 9. unified failure-mode taxonomy: every error, FN side and FP side ───────
# One rule set, applied to all systems and to our own ablation arms. Modes are
# assigned in the listed priority order, so each error lands in exactly one.
MODES = ["M1 sibling confusion", "M2 code identifier", "M3 ordinary vocabulary",
         "M4 topic drift", "M5 wrong antecedent", "M6 unanchored invention",
         "M7 non-architectural mention", "M8 partial-name reach",
         "M9 coreference reach", "M10 renaming reach", "M11 plain name miss"]
MODE_DOC = {
    "M1 sibling confusion":       "two components share a name word; the system took the other one (always an FN and an FP together)",
    "M2 code identifier":         "the name occurs only inside a dotted/hyphenated identifier (logic.api, bbb-html5, akka-apps)",
    "M3 ordinary vocabulary":     "a one-word name used in lower case as ordinary English ('the core logic of the system')",
    "M4 topic drift":             "FP: no name word, but the component was a TP within the 3 preceding sentences -> topic assumed to continue",
    "M5 wrong antecedent":        "FP: no name word; the name is in the 3-sentence context, so a reference was resolved to the wrong component",
    "M6 unanchored invention":    "FP: no name word in the sentence or its context and no recent TP",
    "M7 non-architectural mention": "FP: the name really is written here, but the gold does not count this mention as a link",
    "M8 partial-name reach":      "FN: the sentence writes part of a multi-word name and the system never reached it",
    "M9 coreference reach":       "FN: the sentence writes no name word; the name is in the 3-sentence context",
    "M10 renaming reach":         "FN: no name word in the sentence or its context (the document uses its own term)",
    "M11 plain name miss":        "FN: the sentence writes the name in full and the system still did not link it",
}


def _idspans(st):
    return [(m.start(), m.end()) for m in re.finditer(r"\b[a-zA-Z][\w]*(?:[.\-][\w]+)+\b", st)]


def only_in_identifier(name, st):
    spans, low, hit = _idspans(st), st.lower(), False
    for t in C.camel_tokens(name):
        occ = [m.start() for m in re.finditer(r"\b" + re.escape(t), low)]
        if not occ:
            continue
        hit = True
        for pos in occ:
            if not any(a <= pos < b for a, b in spans):
                return False
    return hit


def lowercase_only(name, st):
    tk = C.camel_tokens(name)
    if len(tk) != 1:
        return False
    occ = re.findall(r"\b" + re.escape(tk[0]) + r"\w*\b", st, re.I)
    return bool(occ) and all(o[0].islower() for o in occ)


def classify_errors(predby):
    """predby: project -> link set. Returns (fn_by_mode, fp_by_mode) of example lists."""
    fn, fp = defaultdict(list), defaultdict(list)
    for proj in P:
        g, n2, sents = GOLD[proj], C.id_to_name(proj), C.sentences(proj)
        pred = predby[proj]
        gbys, pbys = defaultdict(set), defaultdict(set)
        for s, t in g:
            gbys[s].add(t)
        for s, t in pred:
            pbys[s].add(t)
        recent = defaultdict(list)
        for s, t in pred & g:
            recent[t].append(s)
        tks = lambda t: set(C.camel_tokens(n2.get(t, t)))
        for (sid, tid) in sorted(g - pred):                       # ---- misses
            nm, st = n2.get(tid, tid), sents.get(sid, "")
            surf = C.lexical_class(nm, st)[0]
            if [x for x in pbys[sid] if x not in gbys[sid] and tks(x) & tks(tid)]:
                m = "M1 sibling confusion"
            elif only_in_identifier(nm, st):
                m = "M2 code identifier"
            elif lowercase_only(nm, st):
                m = "M3 ordinary vocabulary"
            elif surf == "LEXICAL-partial":
                m = "M8 partial-name reach"
            elif surf == "SEMANTIC-none":
                m = ("M9 coreference reach"
                     if C.lexical_class(nm, _ctx(sents, sid))[0] != "SEMANTIC-none"
                     else "M10 renaming reach")
            else:
                m = "M11 plain name miss"
            fn[m].append((proj, sid, nm, st[:88]))
        for (sid, tid) in sorted(pred - g):                       # ---- spurious
            nm, st = n2.get(tid, tid), sents.get(sid, "")
            surf = C.lexical_class(nm, st)[0]
            if [x for x in gbys[sid] if x not in pbys[sid] and tks(x) & tks(tid)]:
                m = "M1 sibling confusion"
            elif only_in_identifier(nm, st):
                m = "M2 code identifier"
            elif lowercase_only(nm, st):
                m = "M3 ordinary vocabulary"
            elif surf != "SEMANTIC-none":
                m = "M7 non-architectural mention"
            elif any(0 < sid - ps <= 3 for ps in recent.get(tid, [])):
                m = "M4 topic drift"
            elif C.lexical_class(nm, _ctx(sents, sid))[0] != "SEMANTIC-none":
                m = "M5 wrong antecedent"
            else:
                m = "M6 unanchored invention"
            fp[m].append((proj, sid, nm, st[:88]))
    return fn, fp


out()
out("=" * 100)
out("9. UNIFIED FAILURE-MODE TAXONOMY  (every FN and every FP, one rule set, one mode each)")
out("=" * 100)
for m in MODES:
    out(f"   {m:30} {MODE_DOC[m]}")
out()
ERR = {s: classify_errors(SYS[s]) for s in SYSTEMS}
out(f"   {'failure mode':30}" + "".join(f"{s + ' FN/FP (sum)':>24}" for s in SYSTEMS))
for m in MODES:
    line = f"   {m:30}"
    for s in SYSTEMS:
        fn, fp = ERR[s]
        line += f"{len(fn[m]):11d} /{len(fp[m]):<4d}({len(fn[m]) + len(fp[m]):3d}) "
    out(line)
line = f"   {'TOTAL ERRORS':30}"
for s in SYSTEMS:
    fn, fp = ERR[s]
    a, b = sum(map(len, fn.values())), sum(map(len, fp.values()))
    line += f"{a:11d} /{b:<4d}({a + b:3d}) "
out(line)
out("   NOTE M1 is reach-conditional: a system can only confuse two siblings on a sentence where it")
out("        proposed one of them. A system that reaches neither scores 0 on M1 and pays under M2/M8.")

out()
out("9b. the same taxonomy on our own ablation arms -- which design decision suppresses which mode")
arms_cached = {nm: mk() for nm, mk in ARMS.items()}
usable = {nm: a for nm, a in arms_cached.items() if any(a[p_] for p_ in P)}
out(f"   {'failure mode':30}" + "".join(f"{nm + ' FN/FP':>22}" for nm in usable))
AERR = {nm: classify_errors(a) for nm, a in usable.items()}
for m in MODES:
    line = f"   {m:30}"
    for nm in usable:
        fn, fp = AERR[nm]
        line += f"{len(fn[m]):13d} /{len(fp[m]):<7d}"
    out(line)
line = f"   {'TOTAL ERRORS':30}"
for nm in usable:
    fn, fp = AERR[nm]
    line += f"{sum(map(len, fn.values())):13d} /{sum(map(len, fp.values())):<7d}"
out(line)

out()
out("9c. examples per mode and system (up to 4 each)")
for m in MODES:
    out(f"\n   -- {m}")
    for s in SYSTEMS:
        fn, fp = ERR[s]
        for side, lst in (("FN", fn[m]), ("FP", fp[m])):
            for (proj, sid, nm, st) in lst[:4]:
                out(f"      {s:8} {side} {proj:14} s{sid:<4} {nm:<22} \"{st}\"")

with open(args.out + ".txt", "w") as fh:
    fh.write("\n".join(O) + "\n")
with open(args.out + "_errors.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["system", "side", "failure_mode", "project", "sentence_id", "component", "sentence"])
    for s in SYSTEMS:
        fn, fp = ERR[s]
        for side, d in (("FN", fn), ("FP", fp)):
            for m in MODES:
                for (proj, sid, nm, st) in d[m]:
                    w.writerow([s, side, m, proj, sid, nm, st])
print(f"[written] {args.out}_errors.csv")
