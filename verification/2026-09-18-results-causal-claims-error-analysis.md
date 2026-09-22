# Error analysis of the causal claims in `results.tex` — 2026-09-18

## Scope

`verification/2026-09-17-s126-paper-results-provenance.md` checked that every
**number** in `paper/sections/results.tex` regenerates from the committed `s126`
tables. It did not check the `This is because ...` clauses. This document checks
those: for each causal or mechanistic claim in `results.tex`, whether the recorded
evidence supports the stated mechanism, and at what strength.

Arm: `s_linker126`, GPT-5.6-terra, three runs
(`results/greedymerge_e2e_terra_r{1,2,3}_20260916v2`), knowledge ablation
`results/greedymerge_noknow_e2e_terra_r{1,2,3}_20260916v2`. Baselines from
`sota-links/` at the paths `evaluation/mini-src/rq12.py` reads.

Everything is re-derived by `studies/causal-claims/audit.py` (stdlib only,
read-only), which scores through `evaluation/mini-src/metrics.py` and reads phase
state through `evaluation/mini-src/rq34.py`, so no definition is re-implemented.
The unit of analysis is the individual `(sentence, component)` link; doc-model gold
is 195 links, so every count below is enumerable by hand from
`studies/causal-claims/items.csv` (294 rows).

```bash
python3 studies/causal-claims/audit.py
```

## Reproduction gate

The auditor's own numbers against the committed `evaluation/reports/rq34/s126/`
tables — 16/16 assertions passed, so the analysis is on the paper's own basis:

```text
OK  macro F1 Full 0.935 / F2 0.949 / NoValidator 0.784 / NoNameValid 0.835 / NoCitation 0.854
OK  coref rejected-FP 53.3, rejected-TP 84.3, unique rejected-TP 2.3
OK  both judges: 145.7 distinct FP rejected, 10.7 TP lost outright
OK  unique TPs FullName 136.7 / PartialName 13.0 / Coref 15.0
OK  doc-code worst-component F1 0.773 (Full) -> 0.442 (no-knowledge); file F1 -10.9pp
GATE PASSED
```

A side finding while establishing the gate: `evaluation/reports/RQ12_PERPROJECT.csv`
(canonical, regenerated) and `evaluation/reports/RQ12_PERPROJECT_s126.csv` disagree —
e.g. teammates doc-model F1 `0.8679` vs `0.9020`. Direct computation confirms the
**unsuffixed** file. The `_s126` copy predates commit `aa45a1b4` ("greedy
ambiguous-name discard") and is stale; nothing the paper reads uses it, but it is a
trap for anyone auditing per-project claims.

## Verdicts

| # | claim (line in `results.tex`) | verdict |
|---|---|---|
| C1 | L38 alias table reused across sentences → the gain | **partial** — the reuse clause holds, the *alias* clause carries only 13% of the gain |
| C2 | L39 Artemis "neither reaches the alias-heavy mentions nor removes the plausible-but-wrong links" | **first half not supported**, second half supported |
| C3 | L43 the scan-plus-judge split is what makes precision and recall possible at once | **supported** |
| C4 | L44 a single extraction pass cannot recover a mention it did not emit | design rationale, not measured |
| C5 | L62 the doc-code advantage inherits the doc-model gain | **supported with one counterexample** (MediaStore) |
| C6 | L92 the link-level gap is driven by the few large components every system recovers | **supported on 4/5 projects** |
| C7 | L124 coref rejects 84.3 TP but loses 2.3 because linkerN already proposed the rest | **supported** (true by the column's construction) |
| C8 | L126 each judge reads a different *kind of case* | **not supported as written** — same cases, different evidence |
| C9 | L131 a judge buys precision and can never add recall | **supported** (15/15 cells) |
| C10 | L132 the coref judge "judges the least precise input" | **contradicted** |
| C11 | L134 the judge stack, not the linker, turns a high-recall scan into a precise result | **supported** |
| C12 | L166 the smaller forms reach mentions linkerB cannot quote in full | **supported** (94% of their unique TPs) |
| C13 | L172 removing linkerB collapses every metric because it recovers most direct links | **supported** |
| C14a | L178 an alias is the only route to a component the document never names canonically | **supported** |
| C14b | L178 "and those components are small … losing them costs little link volume" | **contradicted** |
| C15 | L172 (and `approach.tex` L30) the word scan "skips any sentence that writes a whole name" | **not supported as written** — the skip is per (sentence, component) pair (added 2026-09-19) |
| C16 | CH1's weakest cell is a scan-coverage gap (stated in this document, not the paper) | **contradicted** — all 7 partial-form misses were proposed and judge-rejected (added 2026-09-19) |

## Findings

### C1/C2 — the RQ1 mechanism (L38–39)

The gain set is the 30 gold links `\approach` recovers in ≥2 of 3 runs and Artemis
does not. Classified by how the sentence refers to the component:

| how the sentence refers to the component | links | share of the gain |
|---|---:|---:|
| writes the canonical catalog name | 12 | 40.0% |
| writes only a generic head noun of the name | 9 | 30.0% |
| writes a document alias | 4 | 13.3% |
| no surface form at all (coreference) | 4 | 13.3% |
| writes a distinctive token of the name | 1 | 3.3% |

**The alias table is not the dominant mechanism.** 22 of the 30 links (73%) are
sentences that write the name or a word of it; Artemis simply did not emit them
(`teammates` s8 "The main logic of the application is in POJOs" → `Logic`; s20 "The
Common component contains utility code" → `Common`). The alias route accounts for
4 links. The clause in L38 that *is* supported is the other one — "it proposes a
link wherever a name … occurs instead of only where a single extraction pass
fires". The recall-by-class table shows the same: `canonical` +11, `partial_g` +9,
`alias` +4, `none` +4.

The second half of L39 holds. Artemis's 44 majority-run false positives are 35
`none` (a sentence with no surface form of the component at all) against
`\approach`'s 25 FPs, of which 12 are alias and 6 canonical. Artemis's errors are
exactly the "plausible-but-wrong" shape the sentence describes. The loss set — gold
links Artemis recovers and `\approach` does not — is 1 link.

`% TODO, This reason is too coarse, use the minimal suitable one.` at L37 is
correct, and this quantifies it.

### C3/C9/C11 — the scan-plus-judge split (L43, L131, L134)

| system | P | R | F1 | F2 |
|---|---:|---:|---:|---:|
| Full (scan + judges) | 0.913 | 0.959 | 0.935 | 0.949 |
| scan only (both judges off) | 0.668 | 0.998 | 0.784 | 0.891 |
| Artemis, same backend | 0.805 | 0.850 | 0.809 | 0.828 |
| SWATTR | 0.869 | 0.772 | 0.799 | 0.778 |

The scan alone has the **lowest precision of any system in the table** (0.668,
below Artemis's 0.805 and SWATTR's 0.869) and the highest recall (0.998). Neither
component alone produces the reported lead; both are load-bearing. C3 and C11 are
supported. C9 is supported structurally: in 15/15 (run, project) cells the
judges-on true-positive set is a subset of the judges-off one, so no judge ever
adds recall.

One accounting asymmetry worth a word: of the 145.7 distinct false positives the
judges reject, 3.3 are readmitted by the other linker and 142.3 are actually kept
out. L124 pairs a **gross** rejection count with a **net** loss count (10.7).

### C8/C10 — what the two judges read (L126, L132)

`\corefValidator{}` candidates: 174.0 per run, of which **113.7 (65.3%) are the
same `(sentence, component)` pair the `\nameValidator{}` also rules on**, and on
those shared cases the two judges **disagree 75.1% of the time**. They are not
reading different *cases*; they are reading different *evidence* about largely the
same cases — which is what L126's own continuation says ("a match the code
computed" vs "a resolution the model committed to"). The noun is wrong, the
distinction is right.

C10 is contradicted:

| judge | candidates | survive | input precision vs gold | on its exclusive candidates |
|---|---:|---:|---:|---:|
| `\nameValidator{}` | 300.7 | 191.0 (63.5%) | **59.4%** | 40.6% |
| `\corefValidator{}` | 174.0 | 36.3 (20.9%) | **68.2%** | 26.5% |

The coref judge's input is the **more** precise of the two, not "the least
precise". Its 20.9% survival rate reflects redundancy — it rejects 84.3 true links
a run that `\linkerN{}` already holds — not a dirty input. The claim is only
defensible if narrowed to the 60.3 candidates no other linker proposes (26.5%).
The per-case weight arithmetic itself is fine: 0.047 pp F1 per candidate against
0.033.

### C12/C13 — the proposal forms (L166, L172)

Unique true positives and the surface class of the sentence they land on (3 runs
pooled):

| form | unique TPs/run | surface classes of those TPs |
|---|---:|---|
| `\linkerB{}` (FullName) | 136.7 | canonical 348, alias 62 |
| `\linkerD{}` (PartialName) | 13.0 | partial_g 27, partial_d 6, canonical 5, none 1 |
| `\linkerC{}` (Coref) | 15.0 | none 45 |

79 of the 84 pooled unique TPs of the two smaller forms (94%) are on sentences that
do not write the whole name, and every one of the coref form's is on a sentence
with no surface form at all. C12 is supported. C13 follows from the 136.7 of ~184
TPs the FullName form uniquely holds.

Two residuals worth knowing about, neither large enough to change the claim: the
PartialName form's unique TPs include 5 (pooled) on sentences that *do* write the
whole name modulo hyphenation (`bbb-web` → `BBB web`), and 1 on a sentence with no
name token at all (`teammates` s168, "This component automates the testing of
TEAMMATES." → `Test Driver`) — a partial-name link with no partial name in it.

**C15 — the scope of the scan's whole-name skip is mis-stated.** L172 excuses the
\linkerD{}'s standalone $0.071$ \avgfone\ with "the \linkerD{} skips any sentence
that writes a whole name"; the figure description in `approach.tex` L30 says the
same. In `_scan` the skip is per *(sentence, component) pair* — `if
self._states_a_name(text, component.name): continue` sits inside the component
loop — so a sentence that writes one component's whole name is still scanned for
every other component. bigbluebutton s6 is the worked case: it writes "HTML5
client" in full and the scan still proposes `HTML5 Server` for it. The correct
phrasing is "for each component, the word scan skips the sentences that write that
component's whole name", and the standalone-score excuse is correspondingly
weaker, because the form does see most sentences.

### C14 — the knowledge module (L178)

**C14a is supported, sharply.** The two components that fall to doc-code F1 0 when
the alias table is removed are MediaStore `DB` and TeaStore `ImageProvider`, and
the auditor confirms neither canonical name is ever written in its SAD: the
document says "the Database component" and "the Image Provider". That is exactly
the predicted mechanism. Of the 83 gold doc-model links lost across the three runs,
59 (71%) are alias-surface.

**C14b is contradicted.** Those components are the *largest*, not the smallest:

| component | gold doc-code links | share of its project | F1 Full → no-knowledge |
|---|---:|---:|---|
| MediaStore `DB` | 28 | **47.5%** | 0.768 → 0.000 |
| TeaStore `ImageProvider` | 320 | **45.3%** | 1.000 → 0.000 |
| TeaStore `Persistence` | 180 | 25.5% | 1.000 → 0.909 |
| TeaStore `WebUI` | 114 | 16.1% | 1.000 → 0.939 |
| MediaStore `Reencoding` | 1 | 1.7% | 1.000 → 0.000 |

Across all 38 gold components, the 11 that lose >5pp have a median within-project
link share of 16.1%; the 27 unaffected have 5.6%. The ablation hurts the *larger*
components. Nor does losing them "cost little link volume": MediaStore doc-code
file F1 falls 24.0pp and TeaStore 34.5pp. The macro −10.9pp is small only because
it nets those two against two projects the alias table **hurts**:

| project | doc-model F1 Δ | doc-code file F1 Δ | worst-component F1 Δ |
|---|---:|---:|---:|
| mediastore | −18.6pp | −24.0pp | −76.8pp |
| teastore | −14.1pp | −34.5pp | −100.0pp |
| teammates | **+1.5pp** | **+3.3pp** | 0.0pp |
| bigbluebutton | −1.4pp | **+4.1pp** | **+15.9pp** |
| jabref | −0.9pp | −3.2pp | −4.8pp |

So "the knowledge module is worth 6.7pp" is a five-project average over a bimodal
distribution in which one project is positive on doc-model F1 and two are positive
on doc-code F1. And 12 of the 83 lost links are on sentences that write the
canonical name outright (`teammates` s7/s8 `Logic`, s9 `Storage`, s4 `UI`), so
switching the table off changes more than the alias route — as the SPEC comment at
L150–156 already notes ("the table's absence changes what the rule reads, not only
what the scans match"). The prose does not carry that caveat.

### C5/C6 — inheritance and the size-aware gap (L62, L92)

| project | doc-model gain vs Artemis | doc-code gain |
|---|---:|---:|
| mediastore | +1.3pp | **−4.4pp** |
| teastore | +27.9pp | +27.4pp |
| teammates | +11.3pp | +3.9pp |
| bigbluebutton | +13.9pp | +2.0pp |
| jabref | +8.4pp | +5.6pp |

Pearson r = +0.941 (n = 5; no significance is claimed at that n). The direction
holds and the magnitude attenuates, which the paper already reports (+12.5 vs
+6.9). MediaStore is a genuine counterexample: a positive doc-model gain becomes a
negative doc-code one, so "carries through" is true of 4 of 5 projects, not of the
mechanism unconditionally.

For C6, composing both systems through the same sam-code map, the mean
per-component gap on the top-2 components against the gap on the rest:
mediastore −0.026 / +0.032, teastore +0.167 / +0.337, bigbluebutton +0.002 /
+0.099, jabref +0.056 / +0.153 — and teammates +0.101 / +0.063, the exception. The
top two components carry 56.6–81.2% of gold link mass everywhere. Supported on 4 of
5 projects.

## Suggested narrowing

Under the `AGENTS.md` paper gate ("if a claim cannot be supported from tracked
evidence, narrow the wording"), the changes the evidence forces:

1. **L38–39 (C1/C2).** Replace the alias-centred explanation with the measured one:
   the scan proposes at every name occurrence, so it reaches the 22 of 30
   name-bearing sentences a single extraction pass did not emit; the alias table
   accounts for 4. Keep the second half of L39 — Artemis's FPs are 35/44 on
   sentences with no surface form of the component.
2. **L126 (C8).** "a different kind of case" → "a different kind of evidence". Two
   thirds of the coref judge's input is a link the name judge also rules on.
3. **L132 (C10).** Drop "because it judges the least precise input", or narrow it
   to the candidates no other linker proposes. Measured against gold, its input is
   the more precise of the two (68.2% vs 59.4%).
4. **L178 (C14b).** Drop "and those components are small … losing them costs little
   link volume". The collapsed components are the largest in their projects (47.5%
   and 45.3% of gold link mass) and cost 24.0pp and 34.5pp of their projects'
   doc-code file F1. The honest statement is that the alias table is the only route
   to a component the document never names canonically, and that the −6.7pp macro
   nets two large losses against two projects it does not help.
5. **L124 (accounting).** State that 142.3 of the 145.7 rejected false positives
   are actually kept out; 3.3 are readmitted by the other linker.
6. **L62 (C5).** Note MediaStore, where a positive doc-model gain becomes a
   negative doc-code one.
7. **`approach.tex` L30 (figure description) and L172 (C13).** Both say the word
   scan "skips any sentence that (already) writes a whole name". The skip in
   `_scan` is per *(sentence, component) pair*: a sentence that writes one
   component's whole name is still scanned for every other component. L172 uses
   the sentence-level reading to excuse the \linkerD{}'s standalone $0.071$
   \avgfone, so the excuse is weaker than stated. Correct phrasing: "for each
   component, the word scan skips the sentences that write that component's whole
   name."
8. **Partial-form recall (CH1).** Do not attribute it to scan coverage anywhere.
   All 7 partial-form gold links `\approach` misses were proposed by the word scan
   and rejected by a judge, in all three runs. See *CH1's honest datum*.

## Threats to this analysis

- The `alias` surface category is decided by the run's **own** knowledge table, so
  it is downstream of the approach; a mention the table missed is scored
  `partial_*` or `none`. The `canonical` / `partial` / `none` split does not depend
  on it, and the C1/C2 finding rests on the canonical share, which is unaffected.
- `canonical` is contiguous token containment, so hyphenation variants
  (`bbb-web` → `BBB web`) count as canonical. This is stated where it matters
  (C12) and if anything understates the C1/C2 finding.
- `partial_g` uses a fixed list of generic architectural head nouns
  (`client`, `server`, `layer`, `service`, …) in `audit.py`. It is authored, not
  benchmark-derived, but it is a judgement call; the raw `matched` string is in
  `items.csv` for every row so the call can be re-made.
- Majority-of-3 is used for the gain and FP sets to keep them stable across runs;
  the per-run sets are in `items.csv` (`approach_runs`, `artemis_runs`) for anyone
  who wants a different rule.
- n = 5 projects throughout. No significance test is run and none should be read.

---

## Addendum — evidence indexed by challenge, not by RQ

`approach.tex` L9–27 states three challenges and answers each with one design
decision. The RQs cut across them (CH1 ↔ RQ4 forms, CH2 ↔ RQ3 judges, CH3 ↔ RQ4
knowledge), so `results.tex` never states a per-challenge result. Sections 7–9 of
`studies/causal-claims/report.txt` regenerate the evidence in that shape.

### CH1 — links are carried by various reference forms

| the sentence writes | gold | SWATTR | Artemis | `\approach` | proposed by |
|---|---:|---:|---:|---:|---|
| the whole name | 135 | 134 | 122 | 133 | full\_name 131, partial\_name 2 |
| a document alias | 22 | 5 | 17 | 21 | full\_name 21 |
| a distinctive word of the name | 5 | 3 | 1 | **2** | partial\_name 2 |
| a generic word of the name | 13 | 6 | 1 | 10 | partial\_name 10 |
| no name at all | 20 | 0 | 16 | 20 | coreference 15, full\_name 4, partial\_name 1 |

60 of 195 gold links (30.8%) do not write the canonical name. Recall there:
SWATTR 23.3%, Artemis 58.3%, `\approach` 88.3%. The form each module was designed
for is the form it reaches, and the coreference form is the only one that reaches
the 20 links with no surface form at all — SWATTR recovers 0 of them.

The `alias` row is classified with run 1's knowledge table, which is rediscovered
per run; four mediastore links read `none` here and `alias` under their own run's
table (`AudioAccess`, `DataStorage`).

### CH2 — a link can read as plausible with no support in the sentence

Of the 153.3 false positives the judges reject per run, **61.7% are on sentences
that do contain a surface form of the component** (whole name 25.9%, distinctive
word 22.2%, alias 12.6%): a form being present is not sufficient, which is the
condition CH2 names. The `\nameValidator{}` removes mostly form-present cases
(canonical 36.0, partial\_d 31.3, alias 16.0 of 100.0); the `\corefValidator{}`
removes mostly form-absent ones (none 43.7 of 53.3).

**The running example does not hold up.** `motivation.tex` uses MediaStore S27
("The MediaAccess component encapsulates database access …", gold = `MediaAccess`
only) as the case where a mechanical match proposes a wrong `DB` link and
"without a check against the sentence's own evidence, the false link stands". The
judge rejects that `DB` link in **1 of 3 runs**; runs 1 and 3 emit it. The pronoun
half of the same figure does hold: S24 is recovered by the coreference form in
3 of 3 runs.

### CH3 — the document coins aliases as it goes

Supported: `\approach` reaches 21 of 22 alias-surface gold links against SWATTR's
5; removing the alias table loses 59 alias-surface links across three runs; and
the two components that fall to doc-code F1 0 without it (MediaStore `DB`,
TeaStore `ImageProvider`) are never named canonically in their documents. See
C14a/C14b above for what must not be claimed alongside it.

### CH1's honest datum

The weakest cell is a *distinctive* word of a multi-word name: 2 of 5, the only
cell where SWATTR (3 of 5) beats `\approach`. All three misses are BigBlueButton
(`HTML5 Server` s6, `WebRTC-SFU` s65 and s73).

**All three are judge rejections, not scan misses.** (Corrected 2026-09-19. The
earlier reading of `fig:approach-overview` was wrong: the whole-name skip in
`_scan` is per *component*, not per sentence, so a sentence that writes one
component's name in full is still scanned for every other component.) The
recorded decisions in
`phase_states/s_linker126/openai/bigbluebutton/linker_name.pkl` show the
partial-name scan proposed all three pairs and the `\nameValidator{}` rejected
them:

| link | full arm (3 runs) | no-knowledge arm (3 runs) |
|---|---|---|
| s6 → `HTML5 Server` | rejected 3/3 | rejected 1/3, **approved 2/3** |
| s65 → `WebRTC-SFU` | rejected 3/3 | rejected 3/3 |
| s73 → `WebRTC-SFU` | rejected 3/3 | rejected 3/3 |

Every decision carries `naming: "word only"` and `path: partial_name_rejected`.
In the two approvals the judge quoted "connects directly with the BigBlueButton
server over port 443 (SSL)"; in 7 of the 9 full-arm rejections it recorded
`claim: none`, i.e. it found no phrase in the sentence naming the component.

This is not confined to the weakest cell. Checking all 7 partial-form gold links
`\approach` misses — teammates s122 `GAE Datastore`, bigbluebutton s6/s39/s47
`HTML5 Server`, s65/s73 `WebRTC-SFU`, s73 `HTML5 Server` — every one was proposed
by the partial-name scan in all three runs and rejected by a judge in all three
runs. **The partial-form recall shortfall is entirely a judging result, not a
linker-coverage result.**

Three consequences:

1. Neither the paper nor `CH1-link-analysis.md` may attribute this cell — or the
   partial-form row generally — to scan coverage. It is judge calibration on
   generic head nouns ("the server", "the datastore", "WebRTC"); the linker
   proposed every one of these links. Raising recall on this form means changing
   the name judge, not adding scan coverage.
2. s65/s73 are arguably *correct* rejections — "WebRTC" there denotes the
   protocol, and the manual pass already flags that gold as debatable. s6 is a
   real miss: the sentence says "the BigBlueButton server".
3. The alias table costs this link. With the table on, s6 is rejected 3/3; with
   it off, kept 2/3. On three runs that is suggestive, not established. The
   plausible mechanism: the table binds `bbb-html5` to `HTML5 Server`, giving the
   judge a concrete alternative name to test "the BigBlueButton server" against,
   which it then fails.

---

## Correction to C14a (2026-09-18, after the link-by-link CH1 pass)

The C14a finding above states that the two components which fall to doc-code F1 0
without the alias table "are never named canonically in their documents — the
documents say 'the Database component' and 'the Image Provider'". The second half
is wrong, and the manual CH1 pass (`CH1-link-analysis.md`) caught it.

`ImageProvider` **is** named canonically in the TeaStore SAD. The document writes
"Image Provider" — the catalog name with a space. Exact-token containment finds it
in 0 sentences; orthographic normalisation (lowercase, strip non-alphanumerics)
finds it in 4 (s2, s7, s10, s12). It is a tokenisation variant, not a coined alias.
MediaStore `DB` remains a true vocabulary mismatch: neither matcher reaches it at
any of its gold sentences.

Re-classifying the 83 gold links the no-knowledge arm loses:

| what the sentence actually writes | links | share |
|---|---:|---:|
| a true alias, or no surface form at all | 44 | 53.0% |
| **an orthographic variant of the catalog name** (whitespace/case) | **15** | **18.1%** |
| a word of a multi-word catalog name | 12 | 14.5% |
| the exact catalog name | 12 | 14.5% |

All 15 orthographic cases are TeaStore (`ImageProvider` ×12, `Persistence` ×3).
So roughly **47% of the knowledge module's measured contribution is not vocabulary
knowledge**: 18% would be reached by a normalisation rule, 14.5% by partial-name
matching, and 14.5% is on sentences writing the exact catalog name, where the table
changes what the judge reads rather than what the scan matches.

This matters most for the tail claim. Of the two components driving the −33.2pp
worst-component F1, one (MediaStore `DB`) is a genuine vocabulary mismatch and one
(TeaStore `ImageProvider`, 45.3% of that project's gold link mass) is a spacing
variant. The alias table should therefore not be credited with the whole tail
result, and a reviewer comparing against a whitespace-insensitive lexical matcher
would find part of it unavailable.

Verified independently of the annotation:

```bash
python3 - <<'PY'
import sys, re; sys.path.insert(0, "studies/causal-claims"); import audit as A
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
sents = A.sentences("teastore")
print([s for s, t in sents.items() if norm("ImageProvider") in norm(t)])   # [2, 7, 10, 12]
print([s for s, t in sents.items()                                          # []
       if A.contains_seq(A.toks(t), A.toks("ImageProvider"))])
PY
```

---

## Are the three challenges themselves well supported?

Link-by-link passes: `CH1-link-analysis.md` (195 gold links), `CH2-mode1-analysis.md`
(107 lexical false positives), `CH2-mode2-analysis.md` (106 form-absent false
positives). Each has verdicts recorded as data and a generator that asserts exact
coverage and recomputes every count; all three regenerate identically.

### CH1 — supported as an existence claim, not as a general property

All four reference forms occur, and the non-canonical quarter (55 of 195 after
manual correction, 28.2%) is where the lexical baseline loses recall: SWATTR scores
**0/17 on alias and 0/16 on pronoun**. But the variety is not general.

| form | links | projects | concentration |
|---|---:|---:|---|
| alias | 17 | 2/5 | 11 of 17 are MediaStore, from four naming decisions in one document |
| partial | 22 | 3/5 | 18 of 22 are BigBlueButton; 16 are `HTML5 Client`/`HTML5 Server` as "the client"/"the server" |
| pronoun | 16 | 4/5 | the only spread form, and the smallest |

JabRef contributes **0** non-canonical links. The `partial_name` scan, one of the two
scans inside `\linkerN{}`, is motivated almost entirely by two components of one
project. Scope it to an identified observation; do not state it as a property of
architecture documentation.

### CH2 — supported, but it describes SWATTR, not the LLM baseline

Splitting all 213 false positives by whether a surface form of the wrongly linked
component is present (type 1) or absent (type 2):

| producer | type 1 | type 2 |
|---|---:|---:|
| SWATTR | 39 | 1 |
| Artemis | 9 | 35 |
| our scans, pre-judge | 107 | 55 |
| **our final output** | **24** | **1** |

Challenge 2 as written is the type-1 trap, and it is SWATTR's failure mode almost
exclusively. Artemis fails 35-to-9 the *other* way. Two qualifications the manual
pass forced:

- Of Artemis's 35 type-2 items, **20 are arguably correct links the gold omits**
  (19 of them TeaStore, where the gold links only the sentence that names the
  component). Artemis's second mode is majority gold-noise, so it must not be
  presented as an unnamed second challenge. Our own scans' type-2 items are wrong
  32-to-4, and the judges remove 54 of 55.
- Within type 1, the paper's exact illustration ("a responsibility of another
  component") is only **2 of 107** items. The two large mechanisms are
  technology-name reuse (15) and compound/qualifier (20), plus **41 package-path
  and enumeration artefacts** from TEAMMATES' package overviews that should be
  conceded separately.

### CH3 — the vocabulary problem is real; two stated premises are not

- Of the 10 components reachable only via an alias, **7 are also named canonically
  somewhere**; 3 never are. "Aliases the document coins as it goes" describes the
  first group; the second is a fixed catalog↔document naming mismatch.
- Carry distance is genuine: teammates uses "datastore" 113–132 sentences after the
  canonical mention; teastore s4 uses "PersistenceProvider" 18 sentences *before*
  the canonical name first appears.
- **"Handing the whole document to an \ac{LLM} does not settle this by itself" is
  contradicted.** Artemis sees the whole document and recovers 17 of 22 alias links.
  Our no-knowledge arm gets 4 of 22 because *our* scans are lexical over catalog
  names — a fact about this architecture, not about \acp{LLM}.
- See the C14a correction above: 18% of the ablation's effect is orthographic, not
  vocabulary.

### What the alias table costs

The alias table is also the single largest source of the residual false positives it
leaves behind. Survival of false-positive candidates through the judge stack, by the
surface form the candidate sits on (mean of 3 runs):

| candidate sits on | judged | survived | survival |
|---|---:|---:|---:|
| a document alias | 26.7 | 10.7 | **40.0%** |
| the canonical name | 41.0 | 6.3 | 15.4% |
| a word of the name | 36.3 | 5.0 | 13.8% |
| no surface form | 59.3 | 0.7 | 1.1% |

**44.4% of our surviving false positives sit on a document alias**, and the judges
are ~2.6× less effective on alias evidence than on a canonical-name match. This is
the same trade the 2×2 shows from the other side: turning the alias table off raises
precision to 0.953, the highest of any configuration measured.

It also costs recall, which the `partial` row of the CH1 recall table hides. Both
arms recover 15 of 22 partial-form links — but not the same 15. The table wins
teastore s8 (`WebUI` referred to as "The UI") and loses bigbluebutton s6 (`HTML5
Server` referred to as "the BigBlueButton server"). The equal totals are a swap,
not a tie, so the alias table is not purely additive as `results.tex` L178 and the
CH3 paragraph present it.
