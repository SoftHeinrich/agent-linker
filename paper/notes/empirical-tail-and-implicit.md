# Empirical: where ArchLinker's strength actually lives (doc-model, link-level)

Reproduced from raw recovered links (`sota-recovered-links/model-doc/`) + benchmark
PCM names + sentence text. Scripts: `scratchpad/tail_links.py`, `compare_systems.py`.
ArchLinker = GPT pure-full, majority of 3 runs. Baselines single-shot (by design).

## Two ORTHOGONAL strengths, each its own slice

ArchLinker does NOT win the easy case. On named links the lexical SWATTR (89%) beats it (84%).
It wins on two distinct hard slices, driven by two distinct modules:

### A. Ambiguity map → ordinary-English NAMED components others abandon
Silent-failure count (components with 0 recovered links), from raw links:
**Artemis 3, SWATTR 3, LiSSA 1, ArchLinker 0** (of 40).
The 3 Artemis abandons — names DO appear, but Artemis's own NER prompt excludes them
(rule 2 "exclude domain entities", rule 7 "no action nouns: conversion/…"):

| project | component | gold | Artemis | ArchLinker | recovered sentence |
|---|---|--:|--:|--:|---|
| jabref | preferences | 2 | 0 | 1 | s11 "The preferences represents all information customizable by a user…" |
| bigbluebutton | Presentation Conversion | 2 | 0 | 1 | s80 "Presentation conversion flow." |
| teammates | GAE Datastore | 5 | 0 | 2 | s9 "…persistence framework provided by GAE Datastore, a NoSQL database." |

This is NOT implicit reference — every name is on the page. It is the ambiguity-map / knowledge story.

### B. ILinker (coreference) → name-free sentences
Provenance from ablation JSON `sources` field (authoritative): coreference linker =
**13–15% of all true links** (≈22–24 / run). Per project:

| project | implicit (coref) share of TPs |
|---|--:|
| jabref | 0% |
| mediastore | ~11% |
| bigbluebutton | ~14% |
| teammates | ~16% |
| teastore | ~20–29% |

By construction these are sentences with NO component name → DLinker/NER cannot reach them.
Concrete (ArchLinker recovers, Artemis misses):
- mediastore s24 → DB: "It stores user information and meta-data of audio files…"
- teastore s6 → WebUI: "It contains logic to save and retrieve values from cookies."
- teastore s28 → Recommender: "It is trained using all existing orders."
- teammates s78 → Logic: "In particular, it is responsible for the following."
- bigbluebutton s38 → BBB web: "It implements the BigBlueButton API and holds a copy of the meeting state."
- bigbluebutton s69 → kurento: "KMS is responsible for streaming of webcams…" (alias KMS, no "kurento")

Ablation corroboration: Indirect-only variant yields 20 unique doc-code TPs Direct can't reach.

## Cross-system Pareto table (the sharp artifact)

Gold: 131 named + 64 name-free links, 40 components. (name-free recall conflates coref +
alias resolution — directional; the pure-coref number is the 13–15% above.)

| system | profile | named recall | name-free recall | abandoned |
|---|---|--:|--:|--:|
| SWATTR (lexical) | surface matcher | **89%** | 50% | 3 |
| Artemis (NER) | named-only | 82% | 52% | 3 |
| LiSSA (RAG) | semantic, recall-tilted | 77% | 73% | 1 |
| **ArchLinker** | **knowledge + 2 linkers + judges** | 84% | **77%** | **0** |

**The point:** every competitor wins ONE axis and loses the other.
- Lexical/NER (SWATTR, Artemis): strong named, collapse on name-free, abandon 3.
- RAG (LiSSA): strong name-free, weakest named.
- ArchLinker is the only system near the top of BOTH axes and alone at 0 abandoned.

## How to frame (sharp)
1. Lead with the **Pareto table** — "prior work trades off; we don't." It's one figure that
   kills the incrementality charge without a single inflated pp number.
2. Map the two columns to the two modules: ambiguity map → named/ordinary-word + 0 abandonment;
   ILinker → name-free recall (13–15% of links, pure coref).
3. Drill-downs: the 3 abandoned components (A) and the "It…/KMS…" implicit examples (B).
4. Honesty: ArchLinker loses named recall to SWATTR (89 vs 84) and only partially recovers the
   tail (1/2, 1/2, 2/5). The categorical claim is "abandons none + holds both axes," not "dominates."
5. ILinker's edge over the closest competitor (LiSSA, 73%) is small; ILinker's BIG gap is vs
   Artemis (52%). State it as "vs the NER baseline," not absolute.
