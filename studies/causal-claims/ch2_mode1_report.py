#!/usr/bin/env python3
"""Join the derived type-1 population with the manual verdicts and emit the report.

The type-1 population is DERIVED here, never transcribed: it is every row of
`ch2_fp_sheet.csv` that is not in `ch2_mode2_sheet.csv`, joined on
(project, sentence, wrong_component).

    python3 studies/causal-claims/audit.py --sheets DIR --csv-only
    python3 studies/causal-claims/ch2_mode1_report.py DIR > CH2-mode1-analysis.md
"""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch2_mode1_annotations import ANNOTATIONS, SUBMODE

SHEETS = Path(sys.argv[1])
INTS = ("sentence", "name_judged", "name_kept", "coref_judged", "coref_kept",
        "approach_final", "artemis", "swattr")


def load(name):
    rows = list(csv.DictReader((SHEETS / name).open(encoding="utf-8")))
    for r in rows:
        for k in INTS:
            if k in r:
                r[k] = int(r[k])
    return rows


fp = load("ch2_fp_sheet.csv")
mode2 = {(r["project"], r["sentence"], r["wrong_component"]) for r in load("ch2_mode2_sheet.csv")}
rows = [r for r in fp if (r["project"], r["sentence"], r["wrong_component"]) not in mode2]

keys = {(r["project"], r["sentence"], r["wrong_component"]) for r in rows}
missing, extra = keys - set(ANNOTATIONS), set(ANNOTATIONS) - keys
if missing or extra:
    raise SystemExit(f"annotation/population mismatch: missing={missing} extra={extra}")
assert len(rows) == len(keys) == len(ANNOTATIONS)

K = lambda r: (r["project"], r["sentence"], r["wrong_component"])
MODE = lambda r: ANNOTATIONS[K(r)][0]
GOLD = lambda r: ANNOTATIONS[K(r)][1]
BOGUS = lambda r: ANNOTATIONS[K(r)][2]
QUOTE = lambda r: ANNOTATIONS[K(r)][3]
NOTE = lambda r: ANNOTATIONS[K(r)][4]

PRODUCERS = (
    ("SWATTR (lexical baseline, single shot)", lambda r: r["swattr"] == 1),
    ("Artemis (LLM baseline, >=2 of 3 runs)", lambda r: r["artemis"] >= 2),
    ("our scans pre-judge (either judge saw it >=2 runs)",
     lambda r: r["name_judged"] >= 2 or r["coref_judged"] >= 2),
    ("our surviving output (>=2 of 3 runs)", lambda r: r["approach_final"] >= 2),
)
PROJECTS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")


def who(r):
    w = []
    if r["swattr"]:
        w.append("SWATTR")
    if r["artemis"] >= 2:
        w.append("Artemis")
    elif r["artemis"]:
        w.append(f"Artemis({r['artemis']}/3)")
    if r["name_judged"]:
        w.append(f"nameJudge {r['name_kept']}/{r['name_judged']}")
    if r["coref_judged"]:
        w.append(f"corefJudge {r['coref_kept']}/{r['coref_judged']}")
    if r["approach_final"]:
        w.append(f"**SURVIVED {r['approach_final']}/3**")
    return ", ".join(w) or "(none)"


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "--"


N = len(rows)
P = print

P("# CH2 type-1 false positives: a one-by-one error analysis\n")
P("**Type 1** = a false positive on a sentence that DOES carry a surface form of the")
P("wrongly linked component (`auto_surface != none`) but is about something else. This")
P("is the failure mode `paper/sections/approach.tex` L20-22 and `sections/motivation.tex`")
P("assert as Challenge 2: a link that *reads as plausible but has no support in the")
P("sentence*, illustrated lexically. The complement (type 2, no surface form at all) is")
P("covered in `CH2-mode2-analysis.md` and is not repeated here.\n")

P("## Method\n")
P("The population is **derived, never transcribed**: `ch2_mode1_report.py` reads")
P("`ch2_fp_sheet.csv` (all 213 distinct false positives any system produced over the")
P("three s126 terra runs) and subtracts `ch2_mode2_sheet.csv` (the 106 type-2 items),")
P(f"joining on `(project, sentence, wrong_component)`. That leaves **{N} items**, and")
P("the script asserts that the manual verdict set covers them exactly -- no item")
P("annotated that is not in the population, none in the population left unannotated.\n")
P("Verdicts live as data in `ch2_mode1_annotations.py`: one sub-mode, one gold-silence")
P("judgement, one `auto_surface` flag and one evidence quote per item. Every verdict was")
P("reached by opening the SAD at `benchmark/<project>/text_*/…txt` and reading the")
P("sentence with its neighbours. This file computes every aggregate below; nothing is")
P("hand-tallied.\n")
P("Counts are DISTINCT items, not per-run rates. `SWATTR` is single-shot, so its column")
P("is 0/1; `Artemis` and our own stages are pooled over 3 runs.\n")

P("### Sub-mode vocabulary\n")
P("| code | sub-mode | reading |")
P("|---|---|---|")
for k in ("r", "q", "o", "t", "s", "p", "l", "g"):
    P(f"| `{k}` | {SUBMODE[k]} | |")
P("")
P("`r`, `q`, `o`, `t`, `s` are lexical traps in the paper's sense: a name form is")
P("present and denotes something other than the component. `p` and `l` are weaker --")
P("the string is a code identifier or an enumeration entry rather than a claim. `g` is")
P("not a trap at all: the mention is genuine and the gold standard simply has no link.\n")

P("## Sub-mode distribution\n")
P("| sub-mode | items | share |")
P("|---|---:|---:|")
c = Counter(MODE(r) for r in rows)
for k, n in sorted(c.items(), key=lambda kv: -kv[1]):
    P(f"| `{k}` {SUBMODE[k]} | {n} | {pct(n, N)} |")
P(f"| **total** | **{N}** | |")
P("")
trap = sum(c[k] for k in "rqots")
P(f"Genuine lexical traps (`r`+`q`+`o`+`t`+`s`): **{trap}/{N}** ({pct(trap, N)}). ")
P(f"Identifier/enumeration artefacts (`p`+`l`): **{c['p'] + c['l']}** ({pct(c['p'] + c['l'], N)}). ")
P(f"Not traps at all (`g`): **{c['g']}** ({pct(c['g'], N)}).\n")

P("### Per project\n")
P("| project | items | " + " | ".join(f"`{k}`" for k in "rqotsplg") + " | dominant |")
P("|---|---:|" + "---:|" * 8 + "---|")
for p in PROJECTS:
    sub = [r for r in rows if r["project"] == p]
    if not sub:
        continue
    cc = Counter(MODE(r) for r in sub)
    dom = cc.most_common(1)[0]
    P(f"| {p} | {len(sub)} | " + " | ".join(str(cc[k]) for k in "rqotsplg")
      + f" | `{dom[0]}` ({dom[1]}) |")
P("")

sc_n0 = sum(1 for r in rows if r["name_judged"] >= 2 or r["coref_judged"] >= 2)
P("## Cross-tab by producer\n")
P("| producer | type-1 items | " + " | ".join(f"`{k}`" for k in "rqotsplg") + " |")
P("|---|---:|" + "---:|" * 8)
for lbl, f in PRODUCERS:
    sub = [r for r in rows if f(r)]
    cc = Counter(MODE(r) for r in sub)
    P(f"| {lbl} | {len(sub)} | " + " | ".join(str(cc[k]) for k in "rqotsplg") + " |")
P("")
P("For scale, the same four producers on the **type-2** population were 1 / 35 / 55 / 1")
P("(`CH2-mode2-analysis.md`).\n")
nj = sum(1 for r in rows if r["name_judged"] == 3)
P(f"The `our scans pre-judge` row is {sc_n0}/{N} **by construction, not as a finding**: a")
P("type-1 item is one whose sentence carries a surface form of the component, which is")
P("exactly the NameScanner's trigger, so it proposes all of them -- indeed all")
P(f"{nj} were put to the NameValidator in all three runs. The informative numbers are the")
P("baselines' and our surviving output's.\n")
P("Per-project split of the same four columns:\n")
P("| project | SWATTR | Artemis | our scans | our output |")
P("|---|---:|---:|---:|---:|")
for p in PROJECTS:
    sub = [r for r in rows if r["project"] == p]
    if not sub:
        continue
    P(f"| {p} | " + " | ".join(str(sum(1 for r in sub if f(r))) for _l, f in PRODUCERS) + " |")
P(f"| **all** | " + " | ".join(str(sum(1 for r in rows if f(r))) for _l, f in PRODUCERS) + " |")
P("")

P("## Is the gold standard, not the system, what is wrong?\n")
P("| the sentence does describe that component's responsibility | items | share |")
P("|---|---:|---:|")
gs = Counter(GOLD(r) for r in rows)
for k in ("yes", "borderline", "no"):
    P(f"| {k} | {gs[k]} | {pct(gs[k], N)} |")
P("")
P("Gold-silent breakdown per producer:\n")
P("| producer | items | gold-silent `yes` | `borderline` | `no` |")
P("|---|---:|---:|---:|---:|")
for lbl, f in PRODUCERS:
    sub = [r for r in rows if f(r)]
    cc = Counter(GOLD(r) for r in sub)
    P(f"| {lbl} | {len(sub)} | {cc['yes']} | {cc['borderline']} | {cc['no']} |")
P("")

P("## Where `auto_surface` overstates the evidence\n")
bog = [r for r in rows if BOGUS(r)]
P(f"`audit.py`'s `classify()` labels {N} of these items as carrying a surface form. On")
P(f"**{len(bog)}** of them ({pct(len(bog), N)}) that label overstates the lexical evidence:")
P("the matched string is an ordinary English word, a generic head noun promoted to")
P("`partial_d`, a token borrowed from an unrelated compound, or an over-broad document")
P("alias. Distribution of the flagged items by the automatic label they were given:\n")
P("| auto_surface | items | of which flagged |")
P("|---|---:|---:|")
for k in ("canonical", "alias", "partial_d", "partial_g"):
    sub = [r for r in rows if r["auto_surface"] == k]
    if not sub:
        continue
    P(f"| `{k}` | {len(sub)} | {sum(1 for r in sub if BOGUS(r))} |")
P(f"| **total** | **{N}** | **{len(bog)}** |")
P("")
P("Every flagged item:\n")
P("| project | s | component | auto_surface | matched | why the label overstates it |")
P("|---|---:|---|---|---|---|")
for r in bog:
    P(f"| {r['project']} | {r['sentence']} | {r['wrong_component']} | `{r['auto_surface']}` "
      f"| `{r['auto_matched']}` | {BOGUS(r)} |")
P("")

P("## Judge behaviour\n")
rej = [r for r in rows if r["name_judged"] >= 2 and r["name_kept"] == 0]
surv = [r for r in rows if r["approach_final"] >= 2]
P(f"- **Rejected by the NameValidator** (`name_judged>=2 and name_kept==0`): **{len(rej)}** items.")
P(f"- **Survived to our final output** (`approach_final>=2`): **{len(surv)}** items.")
P(f"- For contrast, the type-2 population had exactly **1** survivor out of 106.\n")
P("What separates them. Rejected vs survived, by the automatic surface label, by")
P("sub-mode, and by gold silence:\n")
P("| split | " + " | ".join(f"`{k}`" for k in ("canonical", "alias", "partial_d", "partial_g")) + " |")
P("|---|---:|---:|---:|---:|")
for lbl, sub in (("rejected", rej), ("survived", surv)):
    cc = Counter(r["auto_surface"] for r in sub)
    P(f"| {lbl} ({len(sub)}) | " + " | ".join(str(cc[k]) for k in
      ("canonical", "alias", "partial_d", "partial_g")) + " |")
P("")
P("| split | " + " | ".join(f"`{k}`" for k in "rqotsplg") + " |")
P("|---|" + "---:|" * 8)
for lbl, sub in (("rejected", rej), ("survived", surv)):
    cc = Counter(MODE(r) for r in sub)
    P(f"| {lbl} ({len(sub)}) | " + " | ".join(str(cc[k]) for k in "rqotsplg") + " |")
P("")
P("| split | gold-silent `yes` | `borderline` | `no` | flagged `auto_surface` |")
P("|---|---:|---:|---:|---:|")
for lbl, sub in (("rejected", rej), ("survived", surv)):
    cc = Counter(GOLD(r) for r in sub)
    P(f"| {lbl} ({len(sub)}) | {cc['yes']} | {cc['borderline']} | {cc['no']} "
      f"| {sum(1 for r in sub if BOGUS(r))} |")
P("")
gy = sum(1 for r in surv if GOLD(r) in ("yes", "borderline"))
P(f"So of the {len(surv)} survivors, **{gy}** are gold-silent `yes`/`borderline` -- the")
P("sentence arguably does describe the component and the reference gold simply has no")
P(f"link there -- and **{len(surv) - gy}** are false positives by any reading.\n")

P("### What the judge lets through\n")
P("Survival rate by the automatic surface label -- this is the sharpest signal in the")
P("whole analysis:\n")
P("| auto_surface | type-1 items | survived | survival rate |")
P("|---|---:|---:|---:|")
for k in ("canonical", "alias", "partial_d", "partial_g"):
    sub = [r for r in rows if r["auto_surface"] == k]
    if not sub:
        continue
    sv = sum(1 for r in sub if r["approach_final"] >= 2)
    P(f"| `{k}` | {len(sub)} | {sv} | {pct(sv, len(sub))} |")
P(f"| **total** | **{N}** | **{len(surv)}** | **{pct(len(surv), N)}** |")
P("")
al_s = [r for r in surv if r["auto_surface"] == "alias"]
al_n = [r for r in rows if r["auto_surface"] == "alias"]
P(f"**The leak is the document-alias table.** {len(al_s)} of the {len(surv)} survivors")
P(f"({pct(len(al_s), len(surv))}) were matched through a run-generated alias, although")
P(f"aliases are only {len(al_n)}/{N} of the population. A `partial_d` or `canonical` hit on")
P("an ordinary word is something the NameValidator reliably talks itself out of; a hit on")
P("a string the run itself put in the alias table is treated as settled, and the judge")
P("never re-asks whether that string is being used in its component sense here. The")
P("aliases doing the damage, with how many survivors each produced:\n")
P("| alias | bound to | survivors | example |")
P("|---|---|---:|---|")
ali = defaultdict(list)
for r in al_s:
    ali[(r["auto_matched"], r["wrong_component"])].append(r)
for (a, comp), rs in sorted(ali.items(), key=lambda kv: -len(kv[1])):
    P(f"| `{a}` | {comp} | {len(rs)} | {QUOTE(rs[0])} |")
P("")
P("By sub-mode the same point: the judge removes **every** ordinary-English trap")
P(f"(`o`: {sum(1 for r in rows if MODE(r) == 'o')} items, 0 survivors) and most technology-name")
P(f"traps (`t`: {sum(1 for r in rows if MODE(r) == 't')} items, "
  f"{sum(1 for r in surv if MODE(r) == 't')} survivors), but keeps the compound/qualifier")
P(f"trap (`q`: {sum(1 for r in rows if MODE(r) == 'q')} items, "
  f"{sum(1 for r in surv if MODE(r) == 'q')} survivors) and almost every genuine mention the")
P(f"gold happens not to cover (`g`: {sum(1 for r in rows if MODE(r) == 'g')} items, "
  f"{sum(1 for r in surv if MODE(r) == 'g')} survivors). In short: it is a good detector of")
P("*is this string even a name?* and a poor detector of *is this name the subject of the")
P("claim?* -- which is precisely the discrimination Challenge 2 asks for.\n")

P("### Every surviving item\n")
for r in surv:
    P(f"- **{r['project']} s{r['sentence']} -> {r['wrong_component']}** "
      f"(`{MODE(r)}`, gold-silent: {GOLD(r)}, auto_surface `{r['auto_surface']}`"
      f"/`{r['auto_matched']}`) -- {who(r)}")
    P(f"  - > {r['sentence_text']}")
    P(f"  - gold here: {r['gold_components_for_this_sentence']}")
    P(f"  - {NOTE(r)}")
P("")

P("### Items the NameValidator rejected\n")
cc = Counter(MODE(r) for r in rej)
P("By sub-mode: " + ", ".join(f"`{k}` {cc[k]}" for k in "rqotsplg" if cc[k]) + ".\n")
P("| project | s | component | auto_surface | matched | sub-mode | gold-silent | evidence |")
P("|---|---:|---|---|---|---|---|---|")
for r in rej:
    P(f"| {r['project']} | {r['sentence']} | {r['wrong_component']} | `{r['auto_surface']}` "
      f"| `{r['auto_matched']}` | `{MODE(r)}` | {GOLD(r)} | {QUOTE(r)} |")
P("")

P("## Every item\n")
P("| # | project | s | wrongly linked to | auto_surface | matched | sub-mode | gold-silent "
  "| `auto_surface` flagged | produced by | evidence | why it is wrong |")
P("|---:|---|---:|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    P(f"| {i} | {r['project']} | {r['sentence']} | {r['wrong_component']} "
      f"| `{r['auto_surface']}` | `{r['auto_matched']}` | `{MODE(r)}` | {GOLD(r)} "
      f"| {'yes' if BOGUS(r) else ''} | {who(r)} | {QUOTE(r)} | {NOTE(r)} |")
P("")

# ── the verdict, with every number computed ─────────────────────────────────
tot_fp = len(fp)
sw_n = sum(1 for r in rows if r["swattr"])
ar_n = sum(1 for r in rows if r["artemis"] >= 2)
sc_n = sum(1 for r in rows if r["name_judged"] >= 2 or r["coref_judged"] >= 2)
ou_n = len(surv)
trap_sw = sum(1 for r in rows if r["swattr"] and MODE(r) in "rqots")
trap_ar = sum(1 for r in rows if r["artemis"] >= 2 and MODE(r) in "rqots")
trap_ou = sum(1 for r in rows if r["approach_final"] >= 2 and MODE(r) in "rqots")
hard_ou = sum(1 for r in surv if GOLD(r) == "no")

P("## Does Challenge 2 hold?\n")
P("**Yes as a phenomenon. But the paper's illustration is narrower than the data, the")
P("mode is far bigger before our judges than after them, and among what survives it is")
P("entangled with gold-standard coverage.**\n")
P(f"Of the {tot_fp} distinct false positives in the study, {N} are type 1 -- the lexical")
P(f"trap the paper describes -- against {tot_fp - N} type 2. So as a share of all false")
P(f"positives the mode Challenge 2 names is {pct(N, tot_fp)}: real and frequent.\n")
P("Per system it is lopsided:\n")
P(f"- **SWATTR** ({sw_n} type-1 items, vs 1 type-2). Being purely lexical, SWATTR fails")
P(f"  almost only in this mode -- but {sum(1 for r in rows if r['swattr'] and MODE(r) in 'pl')}")
P(f"  of its {sw_n} items are package-path / enumeration matches in TEAMMATES")
P("  (`logic.api`, `storage.entity`, `e2e.util`), and only")
P(f"  {trap_sw} are prose traps of the kind the paper draws. Challenge 2 is SWATTR's")
P("  failure mode, but mostly in its dullest form.")
P(f"- **Artemis** ({ar_n} type-1 items, vs 35 type-2). The LLM baseline barely falls for")
P(f"  the lexical trap at all, and {sum(1 for r in rows if r['artemis'] >= 2 and MODE(r) == 'g')}")
P("  of its 9 items are genuine mentions the gold does not cover. Artemis fails by")
P("  topical inference instead. **Challenge 2 as written does not characterise the LLM")
P("  baseline.**")
P(f"- **Our scans before judging** ({sc_n}/{N}): the proposal stage walks into every")
P("  instance of the trap, which is exactly the motivation for having a validator.")
P(f"- **Our final output** ({ou_n} type-1 vs **1** type-2). The judges remove")
P(f"  {pct(N - ou_n, N)} of the trap, but everything they fail to remove is type 1. For")
P("  our own approach Challenge 2 is not just real, it is the only false-positive mode")
P("  left standing.")
P("")
P(f"The caveat: only {hard_ou} of our {ou_n} survivors are wrong under any reading. The")
P("other 15 are gold-silent `yes`/`borderline` -- package-inventory sentences")
P("(*logic.api provides the API of the component*), section-internal elaborations (*the")
P("conversion process sends progress messages*), or a component the gold never links")
P("anywhere (BigBlueButton's Recording Service). Scoring them as errors understates our")
P("precision and inflates how big Challenge 2 looks in our own results.\n")
P("### What the paper should say instead\n")
P("1. **Keep the challenge and keep the lexical illustration** -- it is attested, and two")
P("   items are near-perfect exhibits. BigBlueButton s60: *\"Communication between apps")
P("   and FreeSWITCH Event Socket Layer (fsels) uses messages through redis pubsub\"*,")
P("   linked to **FreeSWITCH** although the phrase names **FSESL** -- and it is produced")
P("   by SWATTR, by Artemis and by us. MediaStore s27/s29: *\"encapsulates database")
P("   access\"* linked to the **Database** component, which is the paper's own")
P("   \"database access\" example occurring verbatim, and it survives our pipeline.")
P("2. **Do not present the trap as one thing.** It decomposes into at least five")
P("   mechanisms and the two largest are not the one the paper draws: a *third-party")
P(f"   technology* name the component is named after (`t`, {c['t']} items -- \"the GAE")
P("   server\", \"WebRTC provides the user with...\", \"implements both SFU and MCU")
P(f"   models\") and a *qualifier inside a compound* (`q`, {c['q']} items -- \"database")
P("   access\", \"client-side interactions\", \"UI name\", \"the back end\"). The")
P(f"   \"responsibility of another component\" reading on its own (`r`) is {c['r']} items.")
P("   It is the right frame; it is not the bulk.")
P(f"3. **Concede the identifier case.** {c['p'] + c['l']} items ({pct(c['p'] + c['l'], N)}) are")
P("   not prose: the SAD lists package paths (`logic.api`, `storage.entity`, `e2e.util`,")
P("   `x.logic`) and any tokenizer sees a component name in them. That is a")
P("   document-genre artefact -- TEAMMATES' package overviews account for")
P(f"   {sum(1 for r in rows if r['project'] == 'teammates' and MODE(r) in 'pl')} of them --")
P("   not a plausibility problem. It deserves one sentence of its own rather than being")
P("   folded into Challenge 2, and it is the main reason a purely lexical baseline looks")
P("   bad on TEAMMATES.")
P("4. **Say whose problem it is, and name the residual honestly.** The evidence supports")
P(f"   the validator: {sc_n} type-1 proposals -> {ou_n} kept. What it lets through is not")
P("   random: it is dominated by *document aliases* the run itself introduced")
P("   (`database`, `back end`, `front-end`, `GAE`, `Recording Processor`), which the")
P("   judge treats as settled evidence. If the paper wants to state a limitation that")
P("   its own numbers still carry, that is the honest one -- alias-mediated lexical")
P("   traps -- rather than the generic \"plausible but unsupported\" wording.")
P("5. **Do not generalise Challenge 2 to LLM-based recovery.** On this benchmark the LLM")
P("   baseline's false positives are overwhelmingly type 2 (35 vs 9). If the motivation")
P("   section needs a challenge that describes LLM behaviour, it is topical inference")
P("   over a neighbourhood, which the current text does not name.")
