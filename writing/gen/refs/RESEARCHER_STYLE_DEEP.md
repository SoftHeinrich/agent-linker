# Shuai Wang — Deep Style Analysis Across Four First-Author Papers

Sample: SymBee (ICDCS 2018), X-MIMO (SenSys 2020), X-Disco (SECON 2022), UMusic (SenSys 2025). All four are first-author, wireless-networking systems papers. Findings below are scoped strictly to these four artifacts.

---

## §1 Per-paper deep read

### 1.1 SymBee (ICDCS 2018) — 11 pages, including 2-appendix tail

**Section structure (verbatim).** I Introduction (~1.5 pp) → II Motivation [A Opportunities for CTC; B The Need for Symbol-level CTC] (~1 pp) → III Design Overview and Background [A SymBee in a Nutshell; B ZigBee-WiFi Cross-observability] (~1.5 pp) → IV SymBee Design [A Robust Payload Encoding@ZigBee; B Understanding Stable Phase; C Extremely Light-weight Decoding@WiFi] (~2 pp) → V Enhanced Decoding with SymBee Preamble (~1 pp) → VI SymBee Features [A WiFi-ZigBee Cross-technology Broadcast; B Compatibility to 40MHz WiFi] (~0.5 pp) → VII Analytics (~0.5 pp) → VIII Evaluation [A Throughput; B Bit Error Rate; C None Line of Sight Scenario; D Impact of Transmission Power; E Robustness to Interference; F Impact of τ and preamble; G Mobility] (~3 pp) → IX Related Work [three threaded subsections: Cross-technology communication, Interference Mitigation, Heterogeneous collaboration] → X Conclusion (5 sentences) → Appendix A Phase Difference Computation; Appendix B Channel Frequency Offset Compensation.

**Sentence mechanics.** Across 10 sampled sentences (intro 4, design 3, eval 3), mean length ≈ 27 words; range 14–46. Voice mix is roughly 60% active / 40% passive — passive concentrates in design ("SymBee is uniquely built on…"; "the bytes are selected such that…"). "We" appears in intro (contributions list), design ("we design", "we note"), and evaluation ("we evaluate", "we implement"). Hedges ("may", "might", "could") are rare (~3 occurrences total); the prose is assertive. Sentence openings are dominated by the subject of the system ("SymBee is/SymBee design/SymBee message…") and by "We" in evaluation. Em-dashes used heavily for in-line apposition ("breakthrough in the rate, and to expand the practical use of ZigBee to WiFi CTC."). Semicolons rare; colons are used to introduce enumerations and equations.

**Section conventions.** Background sits inside §III; he never gives it a stand-alone "Preliminaries". Approach is bottom-up: he first proves a *physical phenomenon* (cross-observability), then derives the design from it. No running example as such — the example is the *phenomenon itself* (symbol 6, then (6,7) vs (E,F)). Evaluation has **no RQs**; it is a flat list of 7 phenomena (A–G); figures dominate (~10 figures, only 1 table — the chip-sequence table). No threats / limitations section. Related work is right before conclusion, organized as three thematic threads with one-paragraph treatments. Conclusion is 5 sentences: restates artifact, restates encoding trick, restates analytical+experimental method, restates headline number. No future-work gesture.

**Figure / table craft.** Figure 1 (page 2) is a schematic "what SymBee is" diagram. The paper mixes architecture schematics (Figs 1–2, 4), waveform plots (Figs 3, 5–8), and bar charts (Figs 13–14, 16, 21). Captions are short narrative ("SymBee preamble is essentially four consecutive 0's, prepended to SymBee message") not just labels. Only one table — taxonomy of ZigBee symbol→chip mapping.

**Motivation craft.** Gap sentence (verbatim): "Although effective, they commonly suffer from limited data rate (215 bps for ZigBee → WiFi [34]) inherently imposed by the coarse-grained packet-level modulation." Competing approaches named **inline** in §II-B, not in a table. "Limitation of Gateway" / "Limitations of the State-of-the-art" appear as **bold inline subheadings**.

**Quantitative claims.** Plain text, no bold/italic on numbers. Mixes comparative ("145.4× faster") and absolute ("31.25 Kbps", "4.2 μs"). Reports point estimates only — no variance/CI. Negative-result handling: he reports the mall/library numbers honestly (≥21 Kbps, lower than other scenarios) and explains the cause (shoppers, interference) without defensiveness.

**Citations.** Intro/§II combined have ~25 citation slots over ~3 pages → roughly 8 cites/page; related work has ~25 cites in 1 page. Citations are clustered ([7],[19],[36],[9],[38]) more than sprinkled. No visible self-citation in this paper (first first-author publication in the set).

**Signature moves.** "To the best of our knowledge" (1×), "To sum up" (2×), "I.e.," / "i.e.," extremely frequent (~30+), "Specifically" frequent (~10), "We note that" frequent (~8), "the highlight of … lies in" (1×). The "i.e.," parenthetical gloss is a fingerprint.

### 1.2 X-MIMO (SenSys 2020) — 14 pages

**Section structure.** 1 Introduction → 2 Motivation [2.1 The Need for IoT MU-MIMO; 2.2 Opportunity #1: CTC; 2.3 Opportunity #2: Multi-antenna WiFi AP] → 3 X-MIMO Overview [3.1 Preliminary: MU-MIMO] → 4 X-MIMO Design [4.1 Cross-technology Channel Estimation; 4.2 Timing Control via WiFi Fragmentation; 4.3 Cross-technology Precoding] → 5 Multi-stream CTC [5.1 Spectral Efficient Emulation] → 6 Evaluation [6.1 Implementation; 6.2 X-MIMO Performance; 6.3 Scalability of X-MIMO; 6.4 X-MIMO Spectral Efficiency; 6.5 Cross-tech. Channel Estimation in Practice; 6.6 Obtaining WiFi-ZigBee Mixed Signal; 6.7 Impact of Transmission Power; 6.8 Immunity to ZigBee ACK Jitter] → 7 Related Work → 8 Conclusion → Appendix (Compensating HW Imperfections).

**Sentence mechanics.** Mean ≈ 26 words; passive ratio slightly higher than SymBee (~45%) due to derivations. "We" dominates intro and §6; §4 leans on impersonal "X-MIMO does X". Hedges remain rare. Em-dashes still common; "—That is," and "I.e.," chains appear in every subsection.

**Section conventions.** Motivation is split into a "Need" subsection and **two numbered Opportunity subsections** — explicit signposting. Approach is again bottom-up, derivation-led (CSI equation → channel estimation → precoding). No running example. Evaluation has 8 subsections, each is a single experimental question (no formal RQs). Figures (~30) dominate hugely over tables (none in main body). Related work is one page, three threads woven together (MU-MIMO, software-defined MU-MIMO, CTC). No threats/limitations/discussion section. Conclusion is 8 sentences, restates contributions and numbers — no future work.

**Figure/table craft.** Figure 1 is again a schematic of the system "in three steps". Figures are dominated by signal plots (I/Q, CSI amplitude/phase) and small bar charts. Captions are descriptive narrative, often two sentences. No tables in main text — surprising for SE readers.

**Motivation gap sentence.** "However, the spectral efficiency of the state-of-the-art CTC is strictly constrained to single-input single-output (SISO), which essentially limits its capability in maintaining massive scale IoT."

**Quantitative claims.** Plain numbers. Heavy use of comparative ("2× of state-of-the-art WEBee", "near-linear", "3× of legacy ZigBee", "28.8× of WEBee"). Reports point estimates; one figure (Fig 23) shows error bars. Negative results: he reports the SER imbalance at Position 1 (1% vs 27%) honestly and explains via channel weakness.

**Citations.** Intro: ~12 cites in 2 pages → 6/page. Related work: ~40 cites in 1 page. Clustered. **Self-citation: SymBee [52] is cited in related work and motivation** — this is the first appearance of self-referencing the prior paper.

**Signature moves.** "To the best of our knowledge" (1×), "We note that" (≥10×), "Specifically" (≥12×), "i.e./I.e." (40+), "for instance" (4×), "we discuss/discusses" frequent. New move: numbered **"Opportunity #1 / Opportunity #2"** subsections — a fingerprint that recurs (see X-Disco).

### 1.3 X-Disco (SECON 2022) — 9 pages

**Section structure.** I Introduction → II Motivation [A The Need for Cross-technology Neighbor Discovery; B Opportunities → B.1 Cross-technology Communication; B.2 Fine-grained PHY-layer Information at WiFi] → III Overview of X-Disco and Background [A X-Disco in a Nutshell; B How ZigBee signal is interpreted at WiFi → B.1 ZigBee Transmitter; B.2 WiFi Spectral Scan] → IV Design of X-Disco [A ZigBee Symbol Extraction → A.1 ZigBee Cross-detection; A.2 Fine-grained Synchronization; B ZigBee Coordinator Detection; C Neighbor Information Acquisition] → V Advanced Features of X-Disco [A ZigBee Neighbor Validation; B Interruption Mitigation] → VI Evaluation [A X-Disco Performance; B Impact of WiFi traffic; C Multi-channel Discovery; D ZigBee Neighbor Validation; E Mobile] → VII Related Work → VIII Conclusion.

**Sentence mechanics.** Mean ≈ 28 words. Active/passive ratio similar (60/40). The "in a Nutshell" subsection has the most assertive prose ("X-Disco is a two-step approach…"). Hedges still rare. "We" dominates intro and contributions; "X-Disco does X" dominates design.

**Section conventions.** Motivation structure repeats X-MIMO **near-verbatim**: "Need for …" + "Opportunities" with numbered subsections. Background is **embedded inside §III "Overview and Background"** — the *same chunked location* as SymBee. Approach is bottom-up: shows the surprising FFT-magnitude pattern, then builds the decoder. Evaluation has 5 subsections, no RQs; no threats section. Related work appears before conclusion, single paragraph + a "compared to X / Y" framing.

**Figure/table.** Figure 1 is the system schematic. Captions short-to-medium. No data tables. Bar/line plots dominate evaluation.

**Motivation gap sentence.** "However, developing a universal neighbor discovery mechanism across multiple wireless protocols might require significant modification on the billions of existing IoT devices [10], resulting in impractical use cases and expensive costs at scale."

**Quantitative claims.** Plain. Mix of absolute ("within 70ms", "9 ZigBee neighbors") and comparative. Variance not reported.

**Citations.** Intro density similar (~6/page). Related work is **shorter than in earlier papers** (~1 page, ~6 cites). **Self-citation: SymBee [23] and X-MIMO [15] both cited** — Wang has begun consistently citing his own prior work.

**Signature moves.** "X-Disco is a two-step approach" repeats the "in a Nutshell" pattern. "To the best of our knowledge, X-Disco is the first design to discover cross-technology neighbors" (1×). Numbered opportunities re-used.

### 1.4 UMusic (SenSys 2025) — 14 pages

**Section structure.** 1 Introduction → 2 Background and Motivation [2.1 The Need of In-car Occupancy Sensing; 2.2 UWB Primer; 2.3 The Limitation of UWB CIR Amplitude] → 3 Design Overview → 4 Main Design [4.1 High-resolution Power Delay Profile in UWB → 4.1.1 Reverting Low-pass Filter; 4.1.2 Reflected Paths Separation; 4.1.3 CFR Transformation; 4.1.4 Immunity to Hardware Imperfections; 4.2 Car Occupancy Detection using PDP] → 5 Efficiency Enhancement [5.1 Computational Cost Optimization; 5.2 Aliasing Avoidance; 5.3 PDP Synchronization] → 6 Evaluation [6.1 Implementation; 6.2 Overall Performance + 6.2.1 Single-person; 6.2.2 Multiple-people; 6.3 Stationary vs Driving; 6.4 Impact of Out-car Environments; 6.5 Aggregated Performance; 6.6 Impact of Different Car Models and UWB Devices Deployment; 6.7 Impact of the Number of UWB Sensors; 6.8 Performance on Unseen Passenger; 6.9 Impact of Environment Augmentation; 6.10 PDP Calculation Precision → 6.10.1/2/3] → 7 Discussion and Future Work → 8 Related Work → 9 Conclusion.

**Sentence mechanics.** Mean ≈ 25 words. Slightly more "Specifically" / "In other words" / "That is" connectives than earlier papers. Active voice dominant in intro/eval, passive in derivations. "We" frequency in §6 is high (≥40 occurrences). Hedges remain rare; one notable mild hedge is "could potentially create more robust hybrid models … left for future work" in §8.

**Section conventions.** First paper in the set with an explicit **§7 Discussion and Future Work** — appears between Evaluation and Related Work. It is itemized into four bolded mini-topics: "Impact of tall passenger.", "Distinguishability between the passenger and large luggage.", "Extension to general sensing applications.", "Full support for HVAC and vital sign applications." This is a clear maturation move. Background is again merged with motivation (§2). Approach is bottom-up (CIR ambiguity → CFR transform → MUSIC → downsampling). No running example. Evaluation has 10 subsections plus a dedicated **§6.10 simulation section answering three pre-posed questions** — the closest he comes to RQ framing.

**Figure/table.** Figure 1 is again the system schematic; Figures 2–9 are the design-motivating phenomena (paired CIR plots, PDP plots). One table (Table 1: FP/FN comparison). Captions are descriptive.

**Motivation gap sentence.** "However, directly applying CIR data for in-car environment sensing, particularly for occupancy detection, presents unique challenges. The metal structure confines reflected UWB signals within a compact (2m×2m) space, leading to rich reflections with similar propagation path lengths."

**Quantitative claims.** Plain numbers. Comparative ("15.7% outperforming", "16.8% improvement"). Reports **breakdown by seat / by people-count** rather than averages — a more granular reporting style than earlier papers. Negative-result handling: explicitly reports accuracy degradation when sensor count drops to 3 and explains "due to the reduced spatial diversity in PDP data". Future work flagged.

**Citations.** Intro density similar (~7/page). Related work is back to ~1 page with thread structure (WiFi, acoustic/vision, mmWave, UWB, BLE/LoRa/RFID). **Self-citation: SymBee [70], X-MIMO [69], X-Disco [68]** all cited — a complete self-citation chain.

**Signature moves.** "i.e./In specific/Specifically" usage extremely high. "We note that" (≥6×). "To the best of our knowledge" absent here, replaced by "the first" / "innovative" claims in contributions. New move: **bolded itemized discussion paragraphs** (Impact of …, Distinguishability between …) instead of inline prose.

---

## §2 Cross-paper patterns

### 2.1 Section structure — STABLE across 4/4
Every paper follows: Intro → Motivation (with "Need + Opportunities") → Overview/Background → Design → (optional) Features/Efficiency Enhancement → Evaluation → Related Work → Conclusion. Background is **always merged** into either Motivation or Overview — never its own top-level section. Threats/limitations are absent in three papers and only appear (as Discussion) in the most recent (UMusic 2025). This is the clearest macro fingerprint.

### 2.2 Sentence mechanics — STABLE
Mean sentence length 25–28 words across all four. Active voice modestly dominant. Hedges scarce in every paper. "i.e./I.e." inline parenthetical glosses appear 30–60× per paper — the most reliable micro-signature. Em-dashes used for apposition in every paper. Semicolons rare. Wang's prose is **assertive, dense, parenthetical, and gloss-heavy**.

### 2.3 Section openings — STABLE
Every Motivation §2.1-equivalent starts with "The Need for X". Every Design section opens with one to two sentences that announce what the section will cover ("This section discusses…", "Here we introduce…", "As the core of UMusic, decomposing the reflected path … is demonstrated first in this section, followed by …"). Every Conclusion opens with "This paper presents/proposes/introduces NAME, a/an …" — verbatim formula in 4/4.

### 2.4 Figure/table craft — STABLE on figures, weak on tables
Figure 1 is the system schematic in 4/4. Figures dominate; tables are almost absent (SymBee: 1 taxonomy table; X-MIMO: 0; X-Disco: 0; UMusic: 1 FP/FN table). Captions in all four are short descriptive narrative, not bare labels. Plots emphasize phenomena (I/Q, CIR, PDP, BER, accuracy) and are typically small multiples.

### 2.5 Motivation craft — STABLE on "Need + Opportunities", evolves on naming
SymBee uses "Opportunities for CTC" + "The Need for Symbol-level CTC" (with inline bold "Limitation of Gateway" / "Limitations of the State-of-the-art"). X-MIMO crystallizes the pattern into "The Need for IoT MU-MIMO" + "Opportunity #1" + "Opportunity #2". X-Disco repeats this verbatim. UMusic keeps "The Need …" but drops the numbered Opportunities in favor of an integrated Background + Limitation flow. The "name the prior baseline, quote its number, declare your multiplicative win" pattern recurs in 4/4 (215 bps→31.25 Kbps; WEBee→2×; CarOSense→15.7%; existing CTC→70 ms).

### 2.6 Quantitative claim style — STABLE
Plain text, no bolding/italic on numbers in any paper. Numbers always come in pairs (yours vs. baseline's) and ratios are foregrounded (145.4×, 2×, 28.8×, 15.7%). Variance and CIs are essentially never reported in the four papers; only one figure-level error bar (X-MIMO Fig 23). Negative results are reported honestly and explained, never hidden — but always with a one-sentence physical/environmental cause (interference, blockage, reduced diversity).

### 2.7 Citation density and placement — STABLE
Intro: ~6–8 cites/page; Related Work: 25–40 cites in ~1 page. Citations cluster ([a],[b],[c],[d]) rather than sprinkle. Self-citation builds a chain from X-MIMO onward and is universal by UMusic.

### 2.8 Rhetorical signature moves — STABLE
- "i.e./I.e.,/In other words/Specifically/That is" parenthetical glosses (4/4, every page).
- "We note that" as concession/qualifier (4/4).
- "To the best of our knowledge" appears in 3/4 (SymBee, X-MIMO, X-Disco), absent in UMusic.
- "In a Nutshell" overview subsection: SymBee §III-A, X-Disco §III-A; (X-MIMO §3 is functionally identical without the literal phrase; UMusic §3 "Design Overview" is the same idea). 3/4 explicit, 4/4 functional.
- Three-fold contribution list with bullets: 4/4.
- Headline number in the last sentence of the abstract and re-stated in the intro: 4/4 (31.25 Kbps/145.4×; 495 Kbps/2×; 70 ms/nine neighbors; 99.4%/15.7%).
- Bottom-up exposition (phenomenon → derivation → design) in 4/4.

**One-off moves (1–2 papers only)**: numbered "Opportunity #1/#2" subsections (X-MIMO, X-Disco only); bolded itemized Discussion subsection (UMusic only); appendices with mathematical derivations (SymBee, X-MIMO only); explicit "Limitations of the State-of-the-art." inline bold (SymBee only); pre-posed numbered RQs in a sub-section (UMusic §6.10 only).

---

## §3 Transferable vs domain-specific

### 3.1 Transferable to SE writing
- Macro outline: Intro → Motivation ("Need for X" + opportunities/baseline gap) → Overview → Design → Eval → Related Work → Conclusion. Cleanly applies to an ICSE tool/system paper.
- The **"X in a Nutshell" overview subsection** that telegraphs the whole system in one paragraph before any details — gold for SE reviewers who skim.
- **Three-fold contribution bullets** with the first bullet always being a "to-the-best-of-our-knowledge / for the first time" claim and the last bullet always being implementation + empirical wins.
- **Headline-number forward reference** in abstract last sentence, restated in intro contributions, restated again in conclusion — a triple-anchor pattern that gives the reader the "win" three times.
- **Bottom-up exposition from a motivating phenomenon**: show a figure that reveals something surprising, then derive the design from it. Maps perfectly to SE empirical/design papers ("Here is a code-smell distribution that nobody noticed → here is our detector").
- **Plain-text quantitative claims paired with a baseline number**: never just "we achieve X"; always "we achieve X, Y× over [baseline]". 
- **Descriptive figure captions** (one full sentence, not just a noun phrase).
- **Bolded inline mini-subheadings** ("Limitation of Gateway.", "Power Control.", "Detailed timing.") inside long subsections — improves skimmability without inflating the TOC.
- **Honest reporting of weak results** with a one-sentence physical/causal explanation rather than defensive prose.
- **Clustering related citations** ([a],[b],[c]) when grouping a family of techniques.
- The **UMusic-style Discussion section with bolded itemized topics** ("Impact of tall passenger.", "Distinguishability …") — directly applicable as a "Threats / Discussion" section in SE.

### 3.2 Domain-specific — do NOT adopt
- Vocabulary: CTC, CSI, CFO, DSSS, OQPSK, ZigBee/WiFi/UWB chip details, dBm/MHz/SNR/SINR, MUSIC algorithm, Vandermonde matrices, ADC/DAC/FFT. None of this carries to SE; do not import the jargon density.
- **No data tables**: 3/4 of his papers have zero or one table. For an SE empirical paper this would be a problem — reviewers expect ablation tables, dataset tables, RQ-result tables. Adopt his figure style but do NOT adopt his table-aversion.
- **No formal RQs**: wireless-systems venues let him list experimental phenomena (A,B,C,D…) instead. SE/ICSE venues expect explicit RQ1/RQ2/RQ3 framing. Reject this aspect of his style.
- **No threats-to-validity**: only the 2025 paper has Discussion, and it is forward-looking ("Future work") rather than threats-style (construct/internal/external validity). For ICSE you must add this section regardless of what Wang does.
- **Heavy mathematical derivations and appendices**: appropriate for wireless venues, inappropriate for most SE work.
- **Self-citation chain** is acceptable but for a *new* SE author with no prior chain to cite, this lever does not exist.
- Hardware/dBm/SNR-flavored evaluation prose ("at 5 meters", "in the office at midnight") does not translate; SE equivalents are dataset shards / project sizes.

---

## §4 ICSE-paper recommendations (ordered by impact)

1. **Adopt the "X in a Nutshell" overview subsection.** A 150–250 word, figure-anchored summary of the whole system between motivation and design — present in SymBee §III-A, X-Disco §III-A, functionally in X-MIMO §3 and UMusic §3. Reviewers love it.
2. **Triple-anchor the headline number** (abstract last sentence → intro contributions third bullet → conclusion). All four Wang papers do this without exception.
3. **Open every section with a one-sentence scope statement.** "This section …" or "Here we …" appears as the first sentence of nearly every section in all four papers. It is unfashionable but extremely effective for reviewer skim.
4. **Use bolded inline mini-subheadings inside long subsections.** Pattern from SymBee §II-B and X-MIMO §4.2. Lets you keep a flat ToC while still signposting.
5. **Pair every quantitative claim with a baseline number** in the same sentence ("X-MIMO achieves 495 Kbps, almost doubling the throughput of legacy ZigBee"; "UMusic … outperforming CarOSense by 15.7%"). Never present a naked accuracy.
6. **Frame motivation as "Need for X + gap in prior work" with explicit numbered opportunities or limitations.** X-MIMO §2 and X-Disco §II are the canonical examples; for SE, "Opportunity #1: existing tool X is widely deployed; Opportunity #2: artifact Y is now machine-readable" works.
7. **Lead the contributions list with a "to-the-best-of-our-knowledge first" claim.** SymBee, X-MIMO, and X-Disco all do this in bullet 1. For SE, scope it carefully ("first detector for …" or "first dataset that …").
8. **Show the motivating phenomenon as a figure inside the motivation/background section.** SymBee Fig 5 (cross-observation), X-MIMO Fig 4 (CSI distortion), X-Disco Fig 5 (FFT-mag pattern), UMusic Fig 2 (paired CIR-vs-PDP). The reader has *seen* the win before reading the design.
9. **Cluster related citations**: instead of "[3] showed A, [4] showed B, [5] showed C", group families: "Several works [3,4,5] explore A, while another line [6,7] pursues B." Used in all four related-work sections.
10. **End the design section with a one-paragraph "compatibility / overhead" claim**, e.g. SymBee §III-A's closing claim on lightweight decoding. For SE: cost (LOC, runtime, CI overhead) sentence.
11. **Honest reporting of weak/degraded numbers with a one-sentence causal explanation.** SymBee §VIII (mall throughput drop → shoppers); X-MIMO §6.2 (SER imbalance → channel weakness); UMusic §6.7 (3-sensor drop → reduced spatial diversity). Builds reviewer trust.
12. **Use UMusic-style bolded itemized Discussion** ("Impact of tall passenger.", "Distinguishability between …", "Extension to general sensing applications.", "Full support for HVAC and vital sign applications.") as a model for a Discussion/Threats-to-Validity section.
13. **Conclusion formula**: 5–9 sentences, start "This paper presents/introduces NAME, a …", restate the headline number with the multiplicative comparison, do NOT introduce new claims. 4/4 Wang papers obey.
14. **Captions as one full descriptive sentence**, not a noun phrase ("X-MIMO operates in three steps of (a), (b), and (c) to deliver multiple packets in parallel, up to the number of antennas and parallel streams supported by NIC."). Train yourself out of "Fig. 3. Architecture.".
15. **Use "i.e.,/Specifically/That is" parenthetical glosses, but sparingly in SE prose** — Wang uses 30–60 per paper, which is too many for SE; budget 10–15 to retain the precision-signaling effect without the density that SE reviewers find heavy.
