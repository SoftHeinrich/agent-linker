#!/usr/bin/env python3
"""Join the derived type-2 population with the manual verdicts and emit the report.

    python3 studies/causal-claims/audit.py --sheets DIR --csv-only
    python3 studies/causal-claims/ch2_mode2_report.py DIR > CH2-mode2-analysis.md
"""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch2_mode2_annotations import ANNOTATIONS, SUBMODE

sheet = Path(sys.argv[1]) / "ch2_mode2_sheet.csv"
rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
for r in rows:
    for k in ("sentence", "name_judged", "name_kept", "coref_judged", "coref_kept",
              "approach_final", "artemis", "swattr"):
        r[k] = int(r[k])

keys = {(r["project"], r["sentence"], r["wrong_component"]) for r in rows}
missing, extra = keys - set(ANNOTATIONS), set(ANNOTATIONS) - keys
if missing or extra:
    raise SystemExit(f"annotation/population mismatch: missing={missing} extra={extra}")

def who(r):
    w = []
    if r["artemis"] >= 2: w.append("Artemis")
    elif r["artemis"]: w.append(f"Artemis({r['artemis']}/3)")
    if r["swattr"]: w.append("SWATTR")
    if r["name_judged"]: w.append(f"nameJudge {r['name_kept']}/{r['name_judged']}")
    if r["coref_judged"]: w.append(f"corefJudge {r['coref_kept']}/{r['coref_judged']}")
    if r["approach_final"]: w.append(f"**SURVIVED {r['approach_final']}/3**")
    return ", ".join(w)

P = print
P("# CH2 type-2 false positives: a one-by-one error analysis\n")
P("**Type 2** = a false positive on a sentence that carries **no surface form of the")
P("wrongly linked component** — no catalog name, no document alias, not even a word of")
P("the name. It is the failure mode `approach.tex` does *not* name: Challenge 2 as")
P("written describes a lexical trap (a known form is present but denotes something")
P("else), which is type 1.\n")
P("Population derived by `audit.py --sheets` (`ch2_mode2_sheet.csv`, 106 items over 5")
P("projects, pooled across 3 runs). Verdicts recorded item-by-item in")
P("`ch2_mode2_annotations.py`; this file only joins and counts them. Each verdict was")
P("reached by reading the sentence and its neighbours in the SAD.\n")
P("Counts below are DISTINCT items, not per-run rates.\n")

P("## Who produces type-2 errors\n")
P("| producer | type-2 items |")
P("|---|---:|")
for lbl, f in (("Artemis (LLM baseline, ≥2 of 3 runs)", lambda r: r["artemis"] >= 2),
               ("SWATTR (lexical baseline)", lambda r: r["swattr"] == 1),
               ("our scans proposed it (either judge saw it ≥2 runs)",
                lambda r: r["name_judged"] >= 2 or r["coref_judged"] >= 2),
               ("our pipeline EMITTED it (≥2 of 3 runs)", lambda r: r["approach_final"] >= 2)):
    P(f"| {lbl} | {sum(1 for r in rows if f(r))} |")
P("")

P("## Sub-modes\n")
P("| sub-mode | items | share |")
P("|---|---:|---:|")
c = Counter(ANNOTATIONS[(r['project'], r['sentence'], r['wrong_component'])][0] for r in rows)
for k in sorted(c, key=lambda x: -c[x]):
    P(f"| `{k}` {SUBMODE[k]} | {c[k]} | {100*c[k]/len(rows):.1f}% |")
P(f"| **total** | **{len(rows)}** | |")
P("")

P("## Is the gold standard, not the system, what is wrong?\n")
P("| the sentence does describe that component's responsibility | items | share |")
P("|---|---:|---:|")
gs = Counter(ANNOTATIONS[(r['project'], r['sentence'], r['wrong_component'])][1] for r in rows)
for k in ("yes", "borderline", "no"):
    P(f"| {k} | {gs[k]} | {100*gs[k]/len(rows):.1f}% |")
P("")
P("Cross-tabulated by producer:\n")
P("| producer | gold-silent `yes` | `borderline` | `no` |")
P("|---|---:|---:|---:|")
for lbl, f in (("Artemis", lambda r: r["artemis"] >= 2),
               ("our scans (pre-judge)", lambda r: r["name_judged"] >= 2 or r["coref_judged"] >= 2),
               ("our final output", lambda r: r["approach_final"] >= 2)):
    sub = [ANNOTATIONS[(r['project'], r['sentence'], r['wrong_component'])][1] for r in rows if f(r)]
    cc = Counter(sub)
    P(f"| {lbl} | {cc['yes']} | {cc['borderline']} | {cc['no']} |")
P("")

P("## Per-project\n")
P("| project | items | dominant sub-mode | gold-silent `yes` |")
P("|---|---:|---|---:|")
for p in ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref"):
    sub = [r for r in rows if r["project"] == p]
    if not sub: continue
    ms = Counter(ANNOTATIONS[(r['project'], r['sentence'], r['wrong_component'])][0] for r in sub)
    ys = sum(1 for r in sub
             if ANNOTATIONS[(r['project'], r['sentence'], r['wrong_component'])][1] == "yes")
    P(f"| {p} | {len(sub)} | `{ms.most_common(1)[0][0]}` ({ms.most_common(1)[0][1]}) | {ys} |")
P("")

P("## Every item\n")
P("`dist` = sentences between this one and the nearest sentence the gold DOES link to")
P("that component (blank = the component has no gold link anywhere).\n")
P("| # | project | s | wrongly linked to | dist | sub-mode | gold-silent | produced by | why it is wrong |")
P("|---:|---|---:|---|---:|---|---|---|---|")
for i, r in enumerate(rows, 1):
    m, g, note = ANNOTATIONS[(r["project"], r["sentence"], r["wrong_component"])]
    d = r["distance_to_nearest_gold_sentence"] or "—"
    P(f"| {i} | {r['project']} | {r['sentence']} | {r['wrong_component']} | {d} | `{m}` | {g} "
      f"| {who(r)} | {note} |")
P("")

P("## The items that survived our judges\n")
surv = [r for r in rows if r["approach_final"] >= 2]
if not surv:
    P("None.\n")
else:
    for r in surv:
        m, g, note = ANNOTATIONS[(r["project"], r["sentence"], r["wrong_component"])]
        P(f"- **{r['project']} s{r['sentence']} → {r['wrong_component']}** (`{m}`, gold-silent: {g}) — "
          f"{who(r)}\n  - > {r['sentence_text']}\n  - {note}")
    P("")

P("## Exhibit: the cleanest type-2 case\n")
P("Selection is a human judgement (unambiguous, zero surface trace, reproducible,")
P("baseline-only); every fact below is computed from the data.\n")
import re as _re, sys as _sys
_sys.path.insert(0, "/mnt/hostshare/ardoco-home/agent-linker/studies/causal-claims")
import audit as _A
for _proj, _span, _wrong, _rank in (("bigbluebutton", range(40, 46), "WebRTC-SFU", "PRIMARY"),
                                    ("bigbluebutton", range(29, 32), "kurento", "RUNNER-UP")):
    _cat, _sents, _g = _A.catalog(_proj), _A.sentences(_proj), _A.gold(_proj)
    _inv = {v: k for k, v in _cat.items()}
    P(f"### {_rank} — {_proj} s{_span.start}-{_span.stop-1}, wrongly linked to `{_wrong}`\n")
    for _k in _span:
        _gl = ", ".join(sorted(_cat.get(c, c) for (x, c) in _g if x == _k)) or "—"
        P(f"- `s{_k}` **[gold: {_gl}]** {_sents[_k]}")
    P("")
    _txt = " ".join(_sents[k] for k in _span)
    _toks = _re.sub(r"[^a-z0-9]", " ", _txt.lower()).split()
    # a shared 4-character prefix, but only between tokens that are themselves >=4
    # characters -- otherwise short function words ("we") spuriously stem-match a
    # component name ("webrtc").
    _trace = sorted({f"{w}~{tk}"
                     for w in _re.sub(r"[^a-z0-9]", " ", _wrong.lower()).split()
                     if len(w) >= 4
                     for tk in _toks
                     if len(tk) >= 4 and tk[:4] == w[:4]})
    P(f"- surface trace of `{_wrong}` in the passage, including stems: "
      f"**{_trace or 'none at all'}**")
    P(f"- `{_wrong}` is gold-linked at sentences {sorted(x for (x, c) in _g if c == _inv[_wrong])}"
      f" — {min(abs(x - _span.start) for (x, c) in _g if c == _inv[_wrong])}+ sentences away")
    P(f"- gold links anywhere in this passage: "
      f"**{len([1 for (x, _c) in _g if x in _span])}**")
    for _lbl, _get in (("Artemis", lambda r: _A.artemis_dm(_proj, r[1])),
                       ("ours", lambda r: set(_A.final_links(_A.RUNDIR[r], _proj)))):
        _per = [sorted(f"s{x}->{_cat.get(c, c)}" for (x, c) in _get(r) if x in _span)
                for r in _A.RUNS]
        P(f"- {_lbl}, the three runs: {_per}")
    P(f"- SWATTR: {sorted(f's{x}->' + _cat.get(c, c) for (x, c) in _A.swattr_dm(_proj) if x in _span)}")
    P("")

P("## Classifier artifacts\n")
art = [r for r in rows
       if ANNOTATIONS[(r['project'], r['sentence'], r['wrong_component'])][0] == "g"]
P(f"{len(art)} of the {len(rows)} items are not really type 2: a surface form of the")
P("component IS present and the automatic tokenizer missed it. These should be counted")
P("as type-1 (lexical) false positives, which makes the lexical mode slightly larger and")
P("the inference mode slightly smaller than the automatic pass reported.\n")
for r in art:
    _m, _g, note = ANNOTATIONS[(r["project"], r["sentence"], r["wrong_component"])]
    P(f"- {r['project']} s{r['sentence']} → {r['wrong_component']}: {note}")
