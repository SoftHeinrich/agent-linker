# Where Artemis beats us, and why — a validator study (SAD-SAM / model-doc)

Empirical answer to two questions about the model-doc (SAD-SAM) task:

1. **What kind of links does Artemis recover that our approach (`aalinker`) misses?**
2. **Why do we miss them — and what is the more general fix to our validator?**

All numbers below are reproduced by the scripts in this directory from committed
data (`model-doc/`, `model-doc/gold/`, `phase-cache-s21/`) plus the ArDoCo
benchmark (SAD text + PCM model). No new LLM calls — the validator findings come
from *recombining the per-pass verdicts our gate already emitted*.

- `01_artemis_unique_tp.py` — Artemis-unique TP set + lexical vs. semantic profile
- `02_failure_trace.py` — where each missed link dies in our pipeline
- `03_validator_frontier.py` — pass decomposition + validator policy/ensemble frontier
- `_common.py` — shared loaders (gold, recovered links, PCM names, lexical classifier)

```bash
python3 analysis/01_artemis_unique_tp.py            # add --intersection for the adversarial view
python3 analysis/02_failure_trace.py
python3 analysis/03_validator_frontier.py
```

`"Ours"` = `aalinker gpt-5.4_s21`, union of 3 runs (a link is ours if recovered in
*any* run — the view most favourable to us). Sonnet results are shown for robustness.

---

## 1. Artemis's unique TPs are 100% lexical, 0% semantic

Recall over all 5 projects (195 gold links): **Artemis 141 (72%)**, **ours 173 (89%)**.
We dominate on recall — the set Artemis uniquely recovers is tiny.

| | TP recovered | Artemis-unique over us | Ours-unique over Artemis |
|---|---|---|---|
| union of our 3 runs | 173 | **5** | **37** |
| intersection (adversarial to us) | 161 | 9 | 30 |

Classifying every Artemis-unique TP by lexical overlap between the SAD sentence and
the PCM component name:

| | Artemis-unique (N=5) | Ours-unique (N=37) |
|---|---|---|
| **Lexical** (name appears, full or partial) | **100%** | 61% |
| **Semantic** (no name overlap) | **0%** | **37%** |

**Every** link Artemis uniquely recovers is a sentence that spells out the component
name almost verbatim — exactly what its Architecture Entity Recognition (NER) is
built to catch:

```
mediastore   MediaAccess  "The MediaAccess component encapsulates database access..."   [verbatim]
teastore     WebUI        "...not provides by the WebUi, but ... the Image Provider"     [verbatim]
teammates    Logic        "Logic is a Facade class which connects to the Logic classes"  [verbatim]
teammates    Storage      "...EntityAlreadyExistsException (escalated from Storage)"      [verbatim]
teammates    E2E          "Its primary function is for E2E tests and L&P tests."          [verbatim]
bigbluebutton FreeSWITCH  "...voice conference systems other than FreeSWITCH..."          [verbatim]
mediastore   FileStorage  "all audio files are stored in ... a dedicated file server"     [file + stored]
```

Artemis has **no embedding/semantic edge** we lack (embedding-based recovery is
LiSSA's mechanism, not Artemis's). Conversely, **37% of our unique TPs are pure
semantic** — links with no name mention at all, which NER structurally cannot reach.

---

## 2. Why we miss them — the entity validation gate, by design

Tracing the 9 missed links through our pipeline (`02_failure_trace.py`):
**8 of 9 are correctly extracted** — we even resolve the right alias (`WebUi`,
`DataStorage`, `FreeSWITCH`) — and are then **rejected by the entity validation
gate**. Only FreeSWITCH s66 is never extracted.

The gate is a **two-pass AND** (`approved = p1 and p2`):

- **P1 — architectural participation:** is the component named as a participant
  (performing/providing/behaving), not just mentioned?
- **P2 — referential specificity:** is the name used for *this specific* element,
  or as a generic technical term?

Each failure maps onto a rubric reject condition:

| missed link | reject reason | pass |
|---|---|---|
| WebUI s7 — "**not** provided by the WebUi" | negated predication | P1 |
| Logic s88 — "Logic is a **Facade class**" | names a code class / generic word | P1 |
| FileStorage s33 — "files are **stored** in a file server" | no participant named | P1 |
| E2E s186 — "for **E2E** tests" | generic term | P2 |
| FreeSWITCH s59 — "systems **other than** FreeSWITCH" | external / generic | P2 |

ArDoCo's gold counts every one of these as a link regardless of polarity or framing.
Our gate, tuned for precision, throws them out. Over 3 gpt runs the gate kills 130
candidates: **105 FP (correct) + 25 gold TP (recall cost)** — 81% of kills are real
FPs, 4.2 FP removed per gold link lost. It lifts entity precision **81.3% → 99.5%**.
The handful Artemis "uniquely" recovers are just the gold links that fall on the
wrong side of that trade.

A slice of the 9 is merely **stochastic** (MediaAccess s27, Logic s87, Storage s101
flip approve/reject across runs) — recoverable, not a blind spot.

---

## 3. A more general fix — vote, don't veto

Decomposing each pass by the FPs it *uniquely* removes vs. the gold TPs it *uniquely*
costs (`03_validator_frontier.py`) exposes the real problem:

| | GPT-5.4 | Sonnet |
|---|---|---|
| **P1** unique veto | removes **0** FP, costs **9** TP | removes 2 FP, costs 0 TP |
| **P2** unique veto | removes 5 FP, costs 9 TP | removes 12 FP, costs 1 TP |
| **consensus** veto (both agree) | removes 100 FP, costs 7 TP (14:1 ✓) | removes 67 FP, costs 2 TP |

On **GPT, P1 is a near-pure recall tax** — zero unique FPs removed, 9 gold TPs lost.
On **Sonnet the same rubric is healthy.** A hand-fixed AND/OR rule cannot be right for
both. Only *consensus* vetoes are a good trade; single-pass vetoes are not, and they
vary by backend.

**Measured policy frontier** (entity stage, gold=195). `AND` = current (reject if
*either* pass rejects); `OR` = reject only if *both* reject (consensus):

GPT-5.4
| policy | TP | FP | P% | R% | **F1%** |
|---|---|---|---|---|---|
| AND — current, 1 run | 147 | 1 | 99.5 | 75.4 | 85.8 |
| P2-only (drop redundant P1) | 150 | 1 | 99.6 | 76.9 | 86.8 |
| OR (consensus-reject) | 153 | 2 | 98.5 | 78.5 | 87.3 |
| **OR + ≥2/3 vote** | **153** | **1** | **99.4** | **78.5** | **87.7** |
| OR + ≥1/3 (recall-max) | 157 | 6 | 96.3 | 80.5 | 87.7 |

Sonnet (robustness)
| policy | TP | FP | P% | R% | **F1%** |
|---|---|---|---|---|---|
| AND — current, 1 run | 145 | 6 | 96.2 | 74.5 | 84.0 |
| **AND/P2-only + ≥2/3 vote** | **147** | **6** | **96.1** | **75.4** | **84.5** |

### Takeaways

1. **Self-consistency / consensus voting (≥k of N samples) is the robust win.**
   `≥2/3` voting lifts GPT (+1.9 F1) and never hurts Sonnet, while *restoring*
   precision (it removes the FP leaks). It is a tunable P/R knob (k=1 recall-max,
   k=N precision-max) and needs **no rubric change** — just sample the validator N
   times and vote. Cheaper variant: re-sample only the candidates where passes
   disagree ("uncertainty-triggered escalation").
2. **Prune checks by marginal value.** The FP-removed-vs-TP-cost decomposition is a
   reusable diagnostic: keep a veto only if its ratio clears the precision floor.
   On GPT, P1 fails this (0:9) → demote it to advisory. This is model-specific, so
   it should be learned per backend, not hard-coded.
3. **Align reject semantics with the gold's linking definition** (needs a re-run to
   score): gold links *negated* and *externally-framed* mentions, but the rubric
   vetoes them. Making P1/P2 polarity- and framing-neutral would recover the
   WebUI / FreeSWITCH class.
4. **Soft confidence + a single tuned threshold** instead of two hard ANDed booleans
   would let the operating point sit exactly on the precision floor (not measurable
   from cache; principled next step).

### Recommended default

Set the entity gate to **consensus-reject + ≥2-of-3 self-consistency voting** — a
merge-logic change in `_validate_with_evidence`, no extra model call per candidate.
Measured effect: GPT **85.8 → 87.7 F1** at unchanged precision (99.4%), Sonnet
neutral-to-better. Then, as a follow-up that does need re-running, test the
polarity-neutral rubric to claw back the negated/external-system class.
