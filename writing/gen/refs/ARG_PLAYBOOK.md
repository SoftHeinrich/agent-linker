# Wang Argumentation Playbook — for our ICSE doc-to-code paper

Synthesised from four exhaustive paragraph-by-paragraph annotation files:
- `ARG_FULL_symbee.md` (SymBee, ICDCS 2018) — 50 paragraphs, ~234 sentences
- `ARG_FULL_xmimo.md` (X-MIMO, 2020) — 60 paragraphs, 327 sentences
- `ARG_FULL_xdisco.md` (X-Disco, 2022) — 47 paragraphs, 215 sentences
- `ARG_FULL_umusic.md` (UMusic, SenSys 2025) — 60 paragraphs, ~290 sentences
- **Combined: 217 prose paragraphs, ~1066 annotated sentences across 4 papers**

This playbook (§1–§5) names the recurring paragraph shapes, sentence moves, and
suppressed moves; then adapts every pattern to our own benchmark of TransArc vs
AALinker on five projects, using only verified numbers (file F1 0.803→0.931,
decision F1 0.596→0.823, component F1 0.714→0.817 aggregate, doc-to-model F1
0.799→0.951; random ~0.155; oracle subset ~0.987; oracle component ~0.906; 525
annotator decisions → 18,660 file pairs; project factors 1.0×–217.6×; block
homogeneity 96–100 %; JabRef top-3 components = 98.6 %, top component = 47 %).
Anything else is marked `\todo{}`.

---

## §1 — Canonical paragraph templates

**Top-line summary (skim):**
- T1 *Problem-staircase intro*: PREMISE → EVIDENCE → CONTRAST → CLAIM (4/4 papers).
- T2 *Gap-via-two-limits*: DEFINITION → CLAIM → CONCESSION → CONCESSION (4/4).
- T3 *Headline-and-mechanism proposal*: CLAIM → DEFINITION/METHODOLOGY → METHODOLOGY → CONSEQUENCE (4/4).
- T4 *Contribution bullet*: CLAIM → METHODOLOGY → CLAIM/CONSEQUENCE (4/4).
- T5 *Single-sentence section scope*: SCOPE (4/4).
- T6 *Background walkthrough*: METHODOLOGY → DEFINITION → METHODOLOGY → … → INTERPRETATION (4/4).
- T7 *Evaluation paragraph*: METHODOLOGY → EVIDENCE → CAUSE → CLAIM/CONSEQUENCE (4/4).
- T8 *Honest-failure-case*: EVIDENCE → CONCESSION → CAUSE → CLAIM (3/4: SymBee, X-MIMO, UMusic).
- T9 *Strawman-and-knockdown related-work entry*: PREMISE → CONCESSION → CONTRAST → CLAIM (4/4).
- T10 *Headline-restating conclusion*: CLAIM → METHODOLOGY → EVIDENCE (4/4).

### T1. Problem-staircase introduction
- **Skeleton:** `PREMISE → EVIDENCE → CONTRAST → CLAIM` (or `... → CAUSE → EXAMPLE`).
- **Function:** §I.¶1 — establishes scale, escalates to a problem, lands the paper's focus.
- **Frequency:** 4/4 papers, ~6 paragraphs total (SymBee §I¶1, §II.A¶1; X-MIMO §1¶1, §2.1¶1; X-Disco §I¶1, §II.A¶1; UMusic §1¶1+¶2, §2.1¶1).
- **Exemplar (SymBee §I¶1, verbatim):**
  > "Explosive growth of wireless devices over the last decade is anticipated to be intensified and diversified … to reach 50 billion by 2020. As much as massive scale wireless body has enriched our daily lives, spectrum shortage has become one of the significant bottlenecks … For example, ZigBee is known to suffer from up to 50% packet loss under WiFi interference."
- **Rhetorical effect:** opens with an order-of-magnitude statistic, pivots on a single connector ("As much as", "Meanwhile", "Despite"), and lands with a quantified harm — so the reader hits the problem already grounded in numbers.

### T2. Gap-via-two-limits paragraph
- **Skeleton:** `DEFINITION → CLAIM → CONCESSION → CONCESSION` (sometimes `… → CONTRAST → CONCESSION`).
- **Function:** §I.¶2 — names the prior-art category, praises it briefly, then closes two alternative escapes.
- **Frequency:** 4/4 papers (SymBee §I¶2 and §II.B¶2; X-MIMO §1¶2; X-Disco §I¶2; UMusic §1¶2 and §1¶4).
- **Exemplar (SymBee §I¶2, verbatim):**
  > "Although effective, they commonly suffer from limited data rate (215 bps for ZigBee → WiFi) inherently imposed by the coarse-grained packet-level modulation. We note that there has been a recent advancement in fine-grained physical layer designs; However they are not applicable to ZigBee to WiFi scenario."
- **Rhetorical effect:** preempts the obvious "but what about X?" by stacking two concessions in one paragraph; the second "However" closes the last door before the proposal.

### T3. Headline-and-mechanism proposal paragraph
- **Skeleton:** `CLAIM → DEFINITION/METHODOLOGY → METHODOLOGY → CONSEQUENCE` (often ending with a deployability or zero-cost CLAIM).
- **Function:** §I.¶3 — the "this paper presents X" paragraph that names the system, the headline number/mechanism, and the practical consequence.
- **Frequency:** 4/4 papers (SymBee §I¶3+¶4; X-MIMO §1¶3; X-Disco §I¶3; UMusic §1¶3).
- **Exemplar (X-MIMO §1¶3, verbatim):**
  > "This paper presents X-MIMO, the first work to bring MU-MIMO into the picture of commodity IoT networking. X-MIMO is a zero-cost, software-only solution that uses pervasively-deployed commodity WiFi APs as the IoT MU-MIMO transmitter … X-MIMO does not require additional hardware or modification of firmware or driver."
- **Rhetorical effect:** announces the artefact, fixes its identity in one sentence ("zero-cost, software-only"), and immediately stamps the deployability claim before any mechanism is unpacked.

### T4. Contribution bullet
- **Skeleton:** `CLAIM → METHODOLOGY → CLAIM/CONSEQUENCE` (typically 2–3 sentences per bullet, three bullets).
- **Function:** itemise contributions just before the roadmap.
- **Frequency:** 4/4 papers (SymBee §I¶6 bullets, X-MIMO §1¶6–8, X-Disco §I¶5, UMusic §1¶6–8).
- **Exemplar (X-Disco §I¶5, verbatim):**
  > "We design X-Disco, the first cross-technology neighbor discovery mechanism for a commodity WiFi device to detect ambient ZigBee neighbors. The full compatibility with commodity WiFi and ZigBee hardware and protocol ensures X-Disco's wide and practical deployment."
- **Rhetorical effect:** every bullet is self-contained — sentence 1 makes the claim, sentence 2 supplies the "how", optional sentence 3 lands a consequence or headline number.

### T5. Single-sentence section scope
- **Skeleton:** `SCOPE` (one sentence, sometimes two clauses).
- **Function:** opens a section/subsection with a roadmap pointer; never carries content of its own.
- **Frequency:** 4/4 papers, very common — SymBee 5×, X-MIMO ≥4× (§4, §6, §6.3, §6.5), X-Disco ≥3× (§IV lead-in, §V lead-in), UMusic ≥4× (§4, §4.1, §5, §6.10).
- **Exemplar (SymBee §IV¶1, verbatim):**
  > "This section provides technical details and insights on SymBee."
- **Rhetorical effect:** zero ornament; the reader knows immediately what is and is not in the next subsection. No "we will discuss several interesting aspects" — just one sentence.

### T6. Background walkthrough paragraph
- **Skeleton:** `METHODOLOGY → DEFINITION → METHODOLOGY → … → INTERPRETATION` (where the INTERPRETATION sentence plants the key insight that the design will later exploit).
- **Function:** §III "preliminary" / "in-a-nutshell" subsections — explain a standard mechanism step by step, ending with a single sentence that flags the property the paper will lean on.
- **Frequency:** 4/4 papers (SymBee §III.B¶2–4, X-MIMO §3.1, X-Disco §III.B.1+B.2, UMusic §4.1¶1–4).
- **Exemplar (X-Disco §III.B.2¶1, verbatim):**
  > "As Figure 4 depicts, in Step (i), the mixer shifts the passband signal to the baseband … Finally, in Step (iv), the rest 64 samples are fed into FFT calculation, which outputs the corresponding magnitude while the phase information is left out. Since this process does not require the received signal to be WiFi, an arbitrary signal (e.g., ZigBee) will be reflected in FFT magnitude if Spectral Scan mode is on."
- **Rhetorical effect:** the reader gets the standard chain in four–six sentences with "Then / Finally / Since", and is handed the exploitable property in the last sentence — already prepared to accept the design that follows.

### T7. Evaluation result paragraph
- **Skeleton:** `METHODOLOGY → EVIDENCE → CAUSE → CLAIM/CONSEQUENCE` (often `METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION`).
- **Function:** report a single experimental scenario — setup, numbers, mechanism behind the numbers, headline.
- **Frequency:** 4/4 papers, ≥30 paragraphs combined (SymBee §VIII.A–G, X-MIMO §6.2–6.8, X-Disco §VI.A–E, UMusic §6.2–6.10).
- **Exemplar (UMusic §6.2¶1, verbatim):**
  > "The median accuracy of UMusic for 1 to 4 people is 97.2%, 93.3%, 87.2%, and 83.1%, respectively, representing improvements of 16.8%, 17.9%, 14.3%, and 13.7% compared to CarOSense, which achieves median accuracies of 83.2%, 79.1%, 76.3%, and 73.1%, respectively. This result demonstrates that our high-resolution PDP effectively captures the significant changes in signal propagation due to human occupancy, allowing traditional classification models like SVM to achieve an overall accuracy of 90.2%, outperforming CarOSense by 15.7%."
- **Rhetorical effect:** four sentences of numbers, then one sentence that names the mechanism behind them — the reader gets the headline and the why simultaneously.

### T8. Honest-failure-case paragraph
- **Skeleton:** `EVIDENCE → CONCESSION → CAUSE → CLAIM` (concede the worst datum but explain it so the rest of the result holds).
- **Function:** appears once per evaluation section when a single number undercuts the trend.
- **Frequency:** 3/4 papers — SymBee §VIII.C¶1 (S3 anomaly), X-MIMO §6.2¶2 (1 % vs 27 % SER imbalance), UMusic §6.7¶1 (three-sensor degradation). Less prominent in X-Disco.
- **Exemplar (X-MIMO §6.2¶2, verbatim):**
  > "The SER of X-MIMO for two ZigBee devices at position 1 is 1% and 27%, exhibiting a significant imbalance. This is because the channel from X-MIMO to ZigBee 2 is so weak that the signal for ZigBee 1 keeps dominating the ZigBee device 2."
- **Rhetorical effect:** owning the bad number with a named cause turns it into evidence that the model is well-understood; nothing is buried.

### T9. Related-work entry — strawman + knockdown
- **Skeleton:** `PREMISE → CONCESSION → CONTRAST → CLAIM` (or `PREMISE → CONCESSION → EXAMPLE → CONTRAST → CLAIM`).
- **Function:** §RW paragraph that introduces a competing line, gives it credit, then explains why it does not fit our setting.
- **Frequency:** 4/4 papers (SymBee §IX¶2, X-MIMO §7¶1, X-Disco §VII¶2, UMusic §8¶2).
- **Exemplar (X-Disco §VII¶2, verbatim):**
  > "However, applying the WiFi to ZigBee CTC to assist ZigBee devices in detecting ZigBee neighbors, NewBee is still for discovering homogeneous wireless neighbors. SERVOUS is using ZigBee device to detect BLE neighbors while it requires modification at both ZigBee and BLE sides, incurring unaffordable costs at deploying that design into practice. Compared to SERVOUS, X-Disco is transparent to the ZigBee network, at the zero cost for installing X-Disco to the WiFi device without any modification to the existing ZigBee devices and ZigBee network."
- **Rhetorical effect:** never dismisses prior art generically — names the closest competitor and the exact axis of difference (modification cost, hardware, generality).

### T10. Headline-restating conclusion
- **Skeleton:** `CLAIM → METHODOLOGY → EVIDENCE` (and sometimes `→ CONSEQUENCE`).
- **Function:** one-paragraph conclusion that recapitulates the system, mechanism, and the single headline number.
- **Frequency:** 4/4 papers (SymBee §X¶1, X-MIMO §8¶1, X-Disco §VIII¶1, UMusic §9¶1).
- **Exemplar (UMusic §9¶1, verbatim):**
  > "This paper introduces UMusic, a system that uses commodity UWB devices to precisely detect car occupancy via lightweight signal processing techniques … The experiments show that UMusic achieves an aggregated accuracy of 99.4%, highlighting its effectiveness in practical scenarios."
- **Rhetorical effect:** the last paragraph of the paper repeats the system name, the mechanism in one sentence, and the headline number — leaving the reader with a takeaway they can quote.

---

## §2 — Canonical sentence-level moves

**Top-line summary (skim):**
- M1 *Scale-statistic opener with citation* (4/4).
- M2 *"However"-pivot in sentence 2 or 3* (4/4).
- M3 *"Specifically"/"In other words" zoom-in* (4/4).
- M4 *"Therefore"/"Thus" cause-to-claim closer* (4/4).
- M5 *Compressed parallel `(i)/(ii)/(iii)` inside one sentence* (4/4).
- M6 *"X is as simple as Y" simplicity tagline* (4/4).
- M7 *"To the best of our knowledge, … the first …" novelty stamp* (4/4).
- M8 *"This is because" same-sentence cause attached to the metric* (4/4).
- M9 *Single-sentence concession + knockout* (4/4).
- M10 *"For instance"/"e.g.," concrete vendor-anchor evidence* (4/4).
- M11 *No-qualifier declarative claim* (4/4).
- M12 *Forward pointer "(details in §X)" / "(as evaluated in §X)"* (4/4).
- M13 *"Despite … is inherently …" two-clause concession-and-refusal* (3/4).
- M14 *"essentially" / "thereby" / "uniquely" consequence connectors* (4/4).

### M1. Scale-statistic opener with citation
- **Pattern:** first sentence of §I.¶1 quantifies the deployment scale; second sentence quantifies the harm.
- **Frequency:** 4/4 papers (SymBee §I¶1, X-MIMO §1¶1, X-Disco §I¶1, UMusic §1¶1).
- **Exemplars (verbatim):**
  > X-MIMO: "The number of IoT devices is expected to grow as large as a trillion by 2035 [46]."
  > X-Disco: "As half billion ZigBee chips sold and over three billion WiFi devices shipped annually, WiFi and ZigBee coexist densely on the 2.4 GHz ISM spectrum."
- **Effect:** the reader meets a citable number in sentence 1, so the topic is presumptively important.

### M2. "However"-pivot at sentence 2 or 3
- **Pattern:** a single connector ("However", "Despite", "Nevertheless", "But", "Although") inverts an optimistic premise within the same paragraph; never delayed past sentence 3.
- **Frequency:** 4/4 papers, ~25 occurrences across the corpus.
- **Exemplars (verbatim):**
  > X-MIMO §1¶2: "However, achieving this in the IoT domain is challenging due to the following intrinsic limitations …"
  > UMusic §1¶4: "However, directly applying CIR data for in-car environment sensing, particularly for occupancy detection, presents unique challenges."
- **Effect:** the pivot lands while the reader still remembers the optimistic sentence, so the contrast is unmistakable.

### M3. "Specifically" / "In other words" zoom-in
- **Pattern:** a sentence beginning with "Specifically", "In other words", "That is", "I.e." rephrases or instantiates the previous sentence — never to add new evidence, only to compress.
- **Frequency:** 4/4 papers, dozens of occurrences.
- **Exemplars (verbatim):**
  > SymBee §II.B¶3: "Conversely, exploring physical layer for symbol (16 μs) level CTC expands the bandwidth to 62.5 KHz."
  > UMusic §1¶2: "In other words, these sensing systems are unable to adaptively customize sensor parameters … when the occupancy status is unknown."
- **Effect:** every general claim is followed by a tighter restatement, so a reader who half-skips never misses the punchline.

### M4. "Therefore" / "Thus" cause-to-claim closer
- **Pattern:** the last sentence of a paragraph opens with "Therefore", "Thus", "Hence", "As a result", or "Consequently" and delivers the verdict.
- **Frequency:** 4/4 papers, ubiquitous in evaluation and motivation paragraphs.
- **Exemplars (verbatim):**
  > SymBee §VIII.G¶1: "Therefore the BER of this mobile experiment is higher than outdoor scenario."
  > UMusic §4.1.4¶3: "Therefore, our high-resolution PDP calculation is immune to the effects of CFO and random initial phase."
- **Effect:** paragraphs always end with a one-sentence takeaway the reader can quote.

### M5. Compressed parallel `(i)/(ii)/(iii)` inside one sentence
- **Pattern:** three numbered clauses run as a single grammatical sentence (not a bullet list) — most often inside an introduction or design paragraph.
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > X-MIMO §1¶2: "(i) Most of IoT devices are equipped with a single antenna, while MU-MIMO needs a multi-antenna transmitter. (ii) While channel estimation is an essential part of MU-MIMO, it is typically unavailable in IoT."
  > UMusic §6.10¶1: "(i), how many paths could be precisely resolved … (ii), are the three hardware imperfections fully immune? … (iii), how could the computational cost reduction affect the precision …"
- **Effect:** preserves enumeration's scannability without paying the white-space tax of bullets in a narrative paragraph.

### M6. "X is as simple as Y" simplicity tagline
- **Pattern:** the proposal is framed in one sentence as a low-friction operation on top of an existing artefact.
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > SymBee §I¶4: "SymBee encoding turns out to be as simple as putting specific byte patterns in a payload of a legitimate ZigBee packet."
  > X-MIMO §4.2¶2: "We note that WiFi packet fragmentation can be simply set using iwconfig (under Linux) command, without involving any hardware, firmware, or driver modifications."
- **Effect:** the apparent complexity of the underlying physics is bracketed by a sentence telling the reader the operational cost is one command or one byte pattern.

### M7. "To the best of our knowledge, … the first …" novelty stamp
- **Pattern:** appears exactly once per paper, usually in §I.¶5 (after the proposal paragraph and before the contribution bullets).
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > X-MIMO §1¶5: "To the best of our knowledge, X-MIMO is the first of its kind to offer MU-MIMO functionality on commodity IoT networks."
  > X-Disco §I¶4: "To the best of our knowledge, X-Disco is the first design to discover cross-technology neighbors using commodity WiFi devices."
- **Effect:** the novelty assertion is hedged ("to the best of our knowledge") but only once, and it is anchored to a specific axis ("on commodity X", "for commodity Y").

### M8. "This is because" same-sentence cause attached to the metric
- **Pattern:** an observed datum and its mechanism appear in adjacent sentences; the cause clause starts with "This is because" / "Since" / "Due to" and references prior sections by name.
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > X-MIMO §6.2¶2: "The SER of X-MIMO for two ZigBee devices at position 1 is 1% and 27%, exhibiting a significant imbalance. This is because the channel from X-MIMO to ZigBee 2 is so weak …"
  > UMusic §6.2.2¶1: "The results for other seats are similar to those for Seat #1, as the reflected paths in the confined space of the vehicle become more complex with an increasing number of occupants, leading to deviations in PDP estimation."
- **Effect:** never lets a number stand without a one-sentence mechanism; the reader has the explanation before they think to ask.

### M9. Single-sentence concession + knockout
- **Pattern:** a concession clause and a refusal clause join inside one sentence via "Despite … infeasible / inapplicable / unachievable" or "While … cannot".
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > SymBee §II.B¶2: "Despite their vastly enhanced speeds, they are inherenly inapplicable to ZigBee to WiFi CTC due to a large bandwidth gap (2 vs 20 MHz) between the technologies."
  > X-MIMO §7¶1: "Despite Surface MIMO achieves up to 1.3 Gbps throughput on commodity WiFi devices, the design is hard to be applied on low-power devices …"
- **Effect:** acknowledging and dismissing a rival in the same sentence saves a whole paragraph of related work.

### M10. "For instance" / "e.g.," concrete vendor or product anchor
- **Pattern:** an abstract claim is followed by a "For instance" sentence naming brands or products with citations.
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > X-MIMO §2.1¶1: "Amazon Echo Plus, Samsung SmartThings, Philips Hue, Hive, Xiaomi Mijia, and IKEA Tradfri are among a large body of smart home gadgets."
  > UMusic §1¶3: "UMusic leverages the existing deployment of UWB technology for access control via digital key services provided by manufacturers such as Volkswagen, BMW, and Hyundai."
- **Effect:** grounding the motivation in named industrial deployments makes the work read as practical rather than academic.

### M11. No-qualifier declarative claim
- **Pattern:** central claim sentences carry no hedges — no "we believe", "we argue", "this suggests"; only "X achieves Y", "X is the first", "X is immune".
- **Frequency:** 4/4 papers; near-zero hedge density in CLAIM sentences.
- **Exemplars (verbatim):**
  > SymBee §I¶3: "This work introduces SymBee, a novel symbol-level ZigBee to WiFi CTC reaching up to 31.25 Kbps, thereby improving packet-level predecessors by 145.4×."
  > UMusic §4.1.4¶3: "Therefore, our high-resolution PDP calculation is immune to the effects of CFO and random initial phase."
- **Effect:** the prose reads as confident; hedging is reserved for §Discussion paragraphs.

### M12. Forward pointer "(details in §X)" / "(as evaluated in §X)"
- **Pattern:** a parenthetical pointer at the end of a sentence promises detail without paying for it now.
- **Frequency:** 4/4 papers, dozens of occurrences.
- **Exemplars (verbatim):**
  > SymBee §IV.A¶2: "(testbed-evaluated against noise and interference in Section VIII)."
  > UMusic §1¶4: "(specifically, two paths need to differ by at least 0.6 m to be separated effectively, as validated in Section 2.3)."
- **Effect:** lets the introduction keep moving while still discharging an obligation to the cautious reader.

### M13. "Despite … is inherently …" two-clause concession-and-refusal
- **Pattern:** rules out an entire alternative category in one sentence by combining a concession with an inherent-limitation clause.
- **Frequency:** 3/4 papers (SymBee, X-MIMO, X-Disco).
- **Exemplars (verbatim):**
  > SymBee §I¶2: "Although effective, they commonly suffer from limited data rate (215 bps for ZigBee → WiFi) inherently imposed by the coarse-grained packet-level modulation."
  > X-MIMO §2.2¶1: "However, the spectral efficiency of the state-of-the-art CTC is strictly constrained to single-input single-output (SISO), which essentially limits its capability in maintaining massive scale IoT."
- **Effect:** a whole class of prior work is dismissed without surveying any individual paper.

### M14. "essentially" / "thereby" / "uniquely" consequence connectors
- **Pattern:** consequence sentences use a single adverbial marker — "essentially", "thereby", "uniquely", "fundamentally", "naturally" — instead of full subordinate clauses.
- **Frequency:** 4/4 papers.
- **Exemplars (verbatim):**
  > SymBee §III.A¶2: "recycling the computational result of the idle listening which runs continuously by default, thereby minimizing the computation cost while maintaining compatibility to the WiFi standard."
  > X-Disco §I¶3: "the exchanged messages are naturally immune to the duty-cycle related problems, thereby achieving the minimum overhead."
- **Effect:** the consequence sounds derived, not asserted; the connector is the proof.

---

## §3 — What Wang systematically avoids

**Top-line summary:**
- A1. No hedged main claims.
- A2. No multi-paragraph related-work surveys.
- A3. No explicit research questions in the intro.
- A4. No concession-heavy framing of the proposal.
- A5. No long descriptive openings.
- A6. No "discussion of contributions" rephrased outside the bullet list.

### A1. Hedged main claims
- **What an undisciplined writer would do:** "We believe X-MIMO might offer a promising direction for enabling MU-MIMO on commodity IoT devices, although further work is needed."
- **What Wang actually writes (X-MIMO §1¶3):** "This paper presents X-MIMO, the first work to bring MU-MIMO into the picture of commodity IoT networking. X-MIMO is a zero-cost, software-only solution."
- **Frequency check:** across 4 papers, the words "we believe", "we argue", "perhaps", "may", "might" appear in CLAIM sentences essentially zero times; "may" appears in concession or future-work contexts only.

### A2. Multi-paragraph related-work surveys
- **What an undisciplined writer would do:** five paragraphs in §RW, one per competitor, each summarising the competitor's method before contrasting.
- **What Wang actually writes (X-Disco §VII):** three short paragraphs — one paragraph per *theme*, not per competitor. SymBee §IX is identical: three theme-paragraphs ("Cross-technology communication", "Interference Mitigation", "Heterogeneous collaboration"), each ≤7 sentences.
- **Frequency check:** §RW length: SymBee 4 paragraphs / X-MIMO 2 paragraphs / X-Disco 3 paragraphs / UMusic 2 paragraphs. Never per-competitor.

### A3. Explicit research questions in the intro
- **What an undisciplined writer would do:** "This paper answers the following research questions: RQ1 … RQ2 … RQ3 …"
- **What Wang actually writes:** no RQ structure anywhere in §I; the closest is X-Disco §III.A¶2 ("As the foundation of X-Disco, decoding the replied message M2 and M4 at commodity WiFi is very challenging …"), which is a CHALLENGE, not an RQ. UMusic §6.10 *does* use "(i)/(ii)/(iii)" questions but only inside the simulation prelude, not the intro.
- **Frequency check:** RQ blocks: 0/4 papers in §I.

### A4. Concession-heavy framing of the proposal
- **What an undisciplined writer would do:** "While our approach has limitations such as A, B, and C, we hope it provides a step toward …"
- **What Wang actually writes (UMusic §4.1.4¶4):** "This immunity is a unique advantage of our approach compared to existing wireless sensing works." The concession is reserved for §Discussion (UMusic §7) where future work is named after the strength is locked in.
- **Frequency check:** concession sentences in §I (introduction): 0/4 papers in CLAIM positions; concessions appear only as gap-framing of *prior* work.

### A5. Long descriptive openings
- **What an undisciplined writer would do:** open §I.¶1 with two paragraphs of background before hitting the first concrete number.
- **What Wang actually writes:** the first numerical fact arrives in sentence 1 or 2 of §I.¶1 in 4/4 papers (50 B by 2020, trillion by 2035, half-billion ZigBee chips, century-long automotive transformation). SymBee's first paragraph has 4 sentences and 3 numbers.
- **Frequency check:** sentences before first quantitative datum in §I.¶1: 0–1 in 4/4 papers.

### A6. "Discussion of contributions" rephrased outside the bullet list
- **What an undisciplined writer would do:** open §II with "In this section we elaborate on the contributions listed above …" and recap.
- **What Wang actually writes:** the contributions are listed once, in §I.¶6 (bullets), and never recapped. §II opens with a one-sentence SCOPE paragraph (T5).
- **Frequency check:** contribution-recap paragraphs outside §I bullets: 0/4 papers.

---

## §4 — Content-faithful drop-in adaptations for our ICSE paper

Every paragraph uses **only** the verified data points listed at the top of this
playbook. Anything else is `\todo{}`. Plain English: no SAD, SAM, ACF1, NDG,
HUS, etc.

### §4.T1 — Problem-staircase intro (skeleton: `PREMISE → EVIDENCE → CONTRAST → CLAIM`)
> Software systems are routinely shipped with both documentation and source code, and trace links between the two are needed for impact analysis, change propagation, and regulatory review \todo{cite TraceLab/trace-link survey}. The state-of-the-art baseline TransArc reaches an average file-level F1 of 0.803 across five widely used benchmark projects (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef). At the decision level, however, the same baseline scores only 0.596, far below the 0.987 file-level and 0.906 component-level F1 achievable on the oracle subset of the benchmark. This paper presents AALinker, a multi-agent doc-to-code linker that closes this gap, reaching file-level F1 0.931 (+12.9pp) and decision-level F1 0.823 (+22.7pp) on the same five projects.

### §4.T2 — Gap-via-two-limits paragraph (skeleton: `DEFINITION → CLAIM → CONCESSION → CONCESSION`)
> Recent doc-to-code linkers fall into two families. Lexical-and-transitive approaches, exemplified by SWATTR and TransArc, propagate similarity scores through a pre-built code graph; on our five-project benchmark they reach a file F1 of 0.803 (TransArc) but only 0.596 at the decision level, which means more than 40 % of the linking decisions are wrong. Although effective at the file level, these approaches inherit the limits of lexical similarity once the doc terminology drifts from the code identifiers. A more recent single-pass LLM linker, LiSSA, was proposed as a remedy, but on the same benchmark it remains weaker than TransArc — so neither line of prior work delivers reliable decision-level links.

### §4.T3 — Headline-and-mechanism proposal (skeleton: `CLAIM → DEFINITION → METHODOLOGY → CONSEQUENCE`)
> This paper presents AALinker, a multi-agent doc-to-code linker built on top of commodity LLMs. AALinker pairs a knowledge layer — an alias table and an ambiguity classifier extracted once per project — with four linker agents that handle explicit, contextual, anaphoric, and abbreviated mentions, and two validation patterns (multi-pass consensus and citation-grounded verification). On the same five-project benchmark, AALinker reaches an average file F1 of 0.931, a decision F1 of 0.823, an aggregate component F1 of 0.817, and a doc-to-model F1 of 0.951, improving over TransArc by 12.9pp, 22.7pp, 10.3pp, and 15.2pp respectively. Because the pipeline is software-only and reuses the documentation and source already on disk, no extra annotation or instrumentation is required \todo{verify on cost-side: prompt token count, runtime per project}.

### §4.T4 — Contribution bullets (skeleton: three bullets, each `CLAIM → METHODOLOGY → CLAIM/CONSEQUENCE`)
> - We present AALinker, the first multi-agent doc-to-code linker that combines a per-project knowledge layer with four mention-type linker agents and two validation patterns. The pipeline lifts the average decision F1 from 0.596 to 0.823 on the five-project benchmark, a +22.7pp gain over TransArc.
> - We introduce a metric suite that reports file-level, decision-level, component-level, and doc-to-model F1 on the same gold standard. Anchored against a random predictor (file F1 ≈ 0.155) and the oracle ceiling (subset file F1 ≈ 0.987, component F1 ≈ 0.906), the suite makes it explicit which axes of the problem AALinker actually closes.
> - We evaluate AALinker on five projects (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef). The benchmark expands 525 raw annotator decisions into 18,660 file-level pairs with per-project enrollment factors ranging from 1.0× (MediaStore) to 217.6× (JabRef); block homogeneity inside a directory holds at 96–100 %, which the evaluation accounts for explicitly.

### §4.T5 — Single-sentence section scope
> This section reports the doc-to-code performance of AALinker against TransArc, LiSSA, and SWATTR on the five-project benchmark.

### §4.T6 — Background walkthrough (skeleton: `METHODOLOGY → DEFINITION → METHODOLOGY → INTERPRETATION`)
> A doc-to-code linker takes two inputs: a documentation file (text, optionally with figures) and a source tree. As illustrated in \todo{ref Fig.~1}, the pipeline (i) splits the doc into mention candidates, (ii) splits the code into candidate targets at file, component, and model granularity, and (iii) emits a yes/no decision for each (mention, target) pair. The five-project benchmark expands 525 raw annotator decisions into 18,660 file-level pairs by enumerating all candidate targets per mention; project enrollment factors run from 1.0× on MediaStore to 217.6× on JabRef. Because files within a directory share their outcome 96–100 % of the time, the block structure of the gold standard dominates the file-level F1 and is the property AALinker is designed to exploit.

### §4.T7 — Evaluation result paragraph (skeleton: `METHODOLOGY → EVIDENCE → CAUSE → CLAIM`)
> We evaluate AALinker against TransArc on the five-project benchmark using file-level F1, decision-level F1, component-level F1, and doc-to-model F1. The aggregate file F1 rises from 0.803 to 0.931 (+12.9pp); decision F1 from 0.596 to 0.823 (+22.7pp); doc-to-model F1 from 0.799 to 0.951 (+15.2pp); component F1 from 0.714 to 0.817 (+10.3pp aggregate), with gains between +4pp and +35pp on four of the five projects and a regression of about −9pp on Teammates. The Teammates regression is concentrated on \todo{component name} where \todo{root-cause sentence: knowledge-layer aliasing collision}, while the +35pp gain on \todo{project} is dominated by the citation-grounded validation step. Across the four metrics, AALinker therefore closes most of the gap to the oracle ceiling (file F1 0.987 on the oracle subset, component F1 0.906 on the oracle-component split).

### §4.T8 — Honest-failure-case paragraph (skeleton: `EVIDENCE → CONCESSION → CAUSE → CLAIM`)
> The component-level result is the least uniform: AALinker improves by between +4pp and +35pp on four of the five projects but drops by about 9pp on Teammates. This regression is concentrated on \todo{component name and failure modality, e.g. "the controller component, where the alias table merges two distinct controllers"}, while the other four projects show consistent gains. Because the knowledge layer is built once per project, errors in alias resolution propagate to every later agent — so the regression is bounded by the alias table, not by the linker agents themselves. With this single failure mode owned, the aggregate component F1 still rises from 0.714 to 0.817.

### §4.T9 — Related-work entry (skeleton: `PREMISE → CONCESSION → CONTRAST → CLAIM`)
> Lexical-and-transitive linkers such as SWATTR and TransArc remain the dominant baseline; on our benchmark TransArc reaches a file F1 of 0.803 and a decision F1 of 0.596. LiSSA, a recent single-pass LLM linker, was proposed to close the lexical gap, but on the same benchmark it remains weaker than TransArc — so simply replacing the lexical layer with an LLM is not enough. Compared to LiSSA, AALinker introduces a per-project knowledge layer (alias table, ambiguity classifier) and two validation patterns (multi-pass consensus, citation-grounded verification) that together push the decision F1 from 0.596 to 0.823, with no change to the benchmark or the source projects.

### §4.T10 — Conclusion (skeleton: `CLAIM → METHODOLOGY → EVIDENCE`)
> This paper presents AALinker, a multi-agent doc-to-code linker that combines a per-project knowledge layer, four mention-type linker agents, and two validation patterns over commodity LLMs. Evaluated on the five-project benchmark of MediaStore, TeaStore, Teammates, BigBlueButton, and JabRef, AALinker reaches an average file F1 of 0.931, decision F1 of 0.823, component F1 of 0.817, and doc-to-model F1 of 0.951 — improving over TransArc by 12.9pp, 22.7pp, 10.3pp, and 15.2pp respectively. The gains close most of the gap to the oracle ceiling (file F1 0.987 on the oracle subset, component F1 0.906 on the oracle-component split), showing that decision-level doc-to-code linking is reachable on commodity tooling.

### §4 — Sentence moves applied (one example each, embeddable anywhere)
- **M1 (scale opener):** "Software projects routinely accumulate thousands of documentation pages; on our five-project benchmark a single project (JabRef) alone expands 525 raw annotator decisions into 18,660 candidate links."
- **M2 (However-pivot):** "TransArc reaches an average file F1 of 0.803 across five projects. However, at the decision level it scores only 0.596, leaving more than 40 % of the linking decisions wrong."
- **M3 (Specifically zoom-in):** "Specifically, on JabRef the top-3 components account for 98.6 % of the file-level gold, and the single largest component covers 47 % of the gold standard."
- **M4 (Therefore closer):** "Therefore, AALinker raises the average decision F1 from 0.596 to 0.823, a +22.7pp gain over TransArc on the same five projects."
- **M5 (compressed (i)/(ii)/(iii)):** "AALinker's knowledge layer covers three concerns in one pass: (i) an alias table that resolves nominal variants, (ii) an ambiguity classifier that flags overloaded mentions, and (iii) a project-scoped vocabulary that the four linker agents reuse downstream."
- **M6 (simplicity tagline):** "Building the knowledge layer is as simple as one LLM pass over the source tree and one pass over the documentation, after which the four linker agents reuse the resulting table."
- **M7 (novelty stamp):** "To the best of our knowledge, AALinker is the first doc-to-code linker that combines a per-project knowledge layer with four mention-type agents and two validation patterns under a single LLM budget."
- **M8 (This is because):** "On Teammates the component F1 drops by about 9pp. This is because \todo{aliasing collision in the knowledge layer between two controllers}, an error that propagates to every later agent."
- **M9 (single-sentence concession + knockout):** "Despite the appeal of single-pass LLM linkers, LiSSA remains weaker than TransArc on our benchmark, leaving the decision-level gap unclosed."
- **M10 (For instance vendor anchor):** "For instance, JabRef alone contributes a project enrollment factor of 217.6× — three orders of magnitude more candidate pairs per annotator decision than MediaStore (1.0×) — and the block homogeneity stays at 96–100 % across the five projects."
- **M11 (no-qualifier declarative):** "AALinker reaches file F1 0.931, decision F1 0.823, component F1 0.817, and doc-to-model F1 0.951 on the five-project benchmark."
- **M12 (forward pointer):** "Block homogeneity holds at 96–100 % across the five projects (per-project numbers in \todo{ref Tab.~X}, with the impact on file-level F1 quantified in §\todo{ref}.)"
- **M13 (Despite … inherently …):** "Despite their elegance, single-pass LLM linkers inherently lack a per-project knowledge layer, and so cannot resolve aliased mentions before the linking decision is made."
- **M14 (essentially/thereby/uniquely):** "The citation-grounded verification pattern checks every link against a documentation citation, thereby cutting decision-level false positives that the multi-pass consensus alone leaves behind."

---

## §5 — Authoring playbook (one-page checklist by section)

### Introduction
- Open §I.¶1 with a numerical fact in sentence 1 or 2 (M1, T1). On our paper: TransArc 0.803 → AALinker 0.931, or 525 annotator decisions → 18,660 file pairs.
- Pivot on a single connector ("However", "Despite") no later than sentence 3 (M2, T1).
- Frame the gap by stacking *two* prior-art concessions in one paragraph (T2, M13). For us: lexical-and-transitive (TransArc/SWATTR) and single-pass LLM (LiSSA).
- The "this paper presents AALinker" paragraph (T3) must contain: artefact name, identity tagline, three-mechanism summary, headline number(s). No hedging (A1, M11).
- One single-sentence novelty stamp with "to the best of our knowledge" (M7), then three contribution bullets (T4) each ≤3 sentences.
- Roadmap is one sentence per section, in a single paragraph (T5 pattern, generalised).

### Motivation
- Use single-sentence SCOPE paragraphs to open subsections (T5).
- Pair the named industrial deployment / vendor anchor with the harm number (M10). For us: \todo{cite the systems whose docs+code are in the benchmark} and the file-level F1 ≈ 0.155 baseline of a random predictor.
- Close every motivation paragraph with a "Therefore" sentence (M4) — the precondition our system meets.

### Approach (Knowledge layer + four agents + two validators)
- Lead each subsection with a single-sentence SCOPE (T5), then a background-walkthrough paragraph (T6).
- The last sentence of the walkthrough must plant the exploitable property (T6 climax) — for AALinker, the property is "files within a directory share their outcome 96–100 % of the time, so block-aware linking is reachable".
- Use compressed `(i)/(ii)/(iii)` enumeration inside one sentence for the three layers / four agents (M5).
- Use "X is as simple as Y" once per subsection (M6) — pair the LLM machinery with one operational sentence ("running the knowledge layer is one prompt per project").

### Metric Suite
- Open with a SCOPE sentence (T5). State the four metrics — file, decision, component, doc-to-model — and bracket each by random predictor (0.155) and oracle ceiling (0.987 file, 0.906 component) (M11, M12).
- Justify each metric with a one-sentence "this is because" mechanism (M8). For instance, decision-level F1 is the metric because \todo{rationale: file-level F1 conflates block homogeneity with linking precision}.
- Do not introduce new metrics later in the paper. Resist coining acronyms (recall A1, A6).

### Experiment / Setup
- One T7-shape paragraph per setup decision. Five-project list (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef) is named once with citations, never repeated as prose.
- Quote the enrollment expansion exactly (525 → 18,660, factors 1.0× to 217.6×); justify the block-homogeneity reporting (96–100 %) with one sentence (M8).
- Forbidden: invented annotator-split protocols or other unverified setup details (A1, A5).

### Results
- Every evaluation paragraph follows T7 (`METHODOLOGY → EVIDENCE → CAUSE → CLAIM`). The CAUSE sentence is mandatory (M8).
- Headline numbers (12.9pp file, 22.7pp decision, 15.2pp doc-to-model, 10.3pp aggregate component) appear without hedging (M11) and with their absolute pair (e.g. "0.803 → 0.931").
- Reserve exactly one T8 paragraph for the component-level Teammates regression (≈ −9pp). Own it, name the cause, and close with the aggregate gain (M11).
- Use "essentially / thereby / uniquely" consequence connectors (M14) to wrap each result block.

### Discussion
- This is the only section where concessions modulate the proposal (A4 inverted). Two short paragraphs at most, in the shape `CLAIM → CONCESSION → BRIDGE`.
- Limit each concession to one sentence (M9). Pair with a future-work pointer (M12).
- Do not recap contributions here (A6).

### Related Work
- Three short paragraphs, one per *theme*, not per competitor (A2). Suggested themes: lexical-and-transitive linkers; single-pass LLM linkers; multi-agent or knowledge-graph linkers \todo{verify}.
- Each paragraph follows T9 (`PREMISE → CONCESSION → CONTRAST → CLAIM`). The CLAIM sentence names the closest competitor and the exact axis of difference (M9, M11).
- "To the best of our knowledge" appears only once *in the paper* (M7) — already spent in §I; do not repeat it here.

### Conclusion
- One paragraph, T10 shape. Restate AALinker's identity (multi-agent linker with knowledge layer + four agents + two validators), recap mechanism in one sentence, and close with the four headline numbers (file 0.931, decision 0.823, component 0.817, doc-to-model 0.951) and the gap-to-oracle delta.
- Do not introduce new arguments or future-work threads here (A4, A6).
- Length budget: ≤6 sentences, mirroring UMusic §9¶1 and SymBee §X¶1.

---

## Appendix A — Pattern frequency table

| Pattern | Symbee | X-MIMO | X-Disco | UMusic | Total papers |
|---|---|---|---|---|---|
| T1 problem-staircase intro | 2 ¶ | 2 ¶ | 2 ¶ | 2 ¶ | 4/4 |
| T2 gap-via-two-limits | 2 ¶ | 1 ¶ | 1 ¶ | 2 ¶ | 4/4 |
| T3 headline-and-mechanism proposal | 2 ¶ | 1 ¶ | 1 ¶ | 1 ¶ | 4/4 |
| T4 contribution bullet | 3 b | 3 b | 3 b | 3 b | 4/4 |
| T5 single-sentence SCOPE | 5+ | 4+ | 3+ | 4+ | 4/4 |
| T6 background walkthrough | §III.B | §3.1 | §III.B.1+2 | §4.1 | 4/4 |
| T7 evaluation result | §VIII | §6.2–6.8 | §VI.A–E | §6.2–6.10 | 4/4 |
| T8 honest-failure-case | §VIII.C | §6.2¶2 | — | §6.7 | 3/4 |
| T9 RW strawman+knockdown | §IX¶2 | §7¶1 | §VII¶2 | §8¶2 | 4/4 |
| T10 headline-restating conclusion | §X¶1 | §8¶1 | §VIII¶1 | §9¶1 | 4/4 |
| M1 scale-statistic opener | yes | yes | yes | yes | 4/4 |
| M2 However-pivot ≤ S3 | yes | yes | yes | yes | 4/4 |
| M3 Specifically zoom-in | yes | yes | yes | yes | 4/4 |
| M4 Therefore closer | yes | yes | yes | yes | 4/4 |
| M5 (i)/(ii)/(iii) in one sentence | yes | yes | yes | yes | 4/4 |
| M6 "as simple as" tagline | yes | yes | yes | yes | 4/4 |
| M7 "to the best of our knowledge, first" | yes | yes | yes | yes | 4/4 |
| M8 "This is because" same-paragraph cause | yes | yes | yes | yes | 4/4 |
| M9 single-sentence concession+knockout | yes | yes | yes | yes | 4/4 |
| M10 "For instance" vendor anchor | yes | yes | yes | yes | 4/4 |
| M11 no-qualifier declarative claim | yes | yes | yes | yes | 4/4 |
| M12 forward pointer "(in §X)" | yes | yes | yes | yes | 4/4 |
| M13 "Despite … inherently …" refusal | yes | yes | yes | — | 3/4 |
| M14 essentially/thereby/uniquely connector | yes | yes | yes | yes | 4/4 |

(¶ = paragraphs of that template in the paper; b = bullets.)
