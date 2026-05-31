# Argumentation Annotation — Wang et al., "Symbol-level Cross-technology Communication via Payload Encoding" (SymBee, ICDCS 2018)

## Table of Contents
- I. Introduction
- II. Motivation
  - II.A Opportunities for CTC
  - II.B The Need for Symbol-level CTC
- III. Design Overview and Background
  - III.A SymBee in a Nutshell
  - III.B ZigBee-WiFi Cross-observability
- IV. SymBee Design
  - IV.A Robust Payload Encoding@ZigBee
  - IV.B Understanding Stable Phase
  - IV.C Extremely Light-weight Decoding@WiFi
- V. Enhanced Decoding with SymBee Preamble
- VI. SymBee Features
  - VI.A WiFi-ZigBee Cross-technology Broadcast
  - VI.B Compatibility to 40MHz WiFi
- VII. Analytics
- VIII. Evaluation
  - VIII.A Throughput
  - VIII.B Bit Error Rate
  - VIII.C None Line of Sight Scenario
  - VIII.D Impact of Transmission Power
  - VIII.E Robustness to Interference
  - VIII.F Impact of τ and preamble
  - VIII.G Mobility
- IX. Related Work
- X. Conclusion
- Endnotes

---

## §I. Introduction · p.500 · ¶1

**Verbatim:**
> Explosive growth of wireless devices over the last decade is anticipated to be intensified and diversified as we step into the Internet of Things (IoT) era, to reach 50 billion by 2020 [2]. As much as massive scale wireless body has enriched our daily lives, spectrum shortage has become one of the significant bottlenecks to efficient networking. I.e., overcrowded unlicensed ISM band has led to severe cross-technology interference (CTI) [12], which has become a major hurdle to network reliability and spectrum efficiency. For example, ZigBee is known to suffer from up to 50% packet loss under WiFi interference [21].

1. PREMISE (EVIDENCE) — quantitative growth projection.
2. CONTRAST — "As much as … has become one of the significant bottlenecks".
3. CAUSE (EVIDENCE) — "I.e., overcrowded … has led to severe CTI".
4. EXAMPLE (EVIDENCE) — "For example, ZigBee … 50% packet loss".

**Shape:** PREMISE → CONTRAST → CAUSE → EXAMPLE.
**Persuasive move:** problem-framing-by-escalation.
**Flow notes:** S1 sets stage; S2 "As much as" pivots from benefit to bottleneck; S3 "I.e." narrows to CTI; S4 "For example" anchors with concrete number.

---

## §I. Introduction · p.500 · ¶2

**Verbatim:**
> To address CTI, latest studies have introduced cross-technology communication (CTC) [7], [19], [36], [9], [38] that enable direct communication among heterogeneous wireless devices with different physical layers. CTC not only fundamentally resolves CTI via cross-technology channel coordination, but also serves as a building block to advanced services through collaboration beyond technologies. Specifically, they convey messages by modulating the timing [19], power [9], and duration [7] per packet basis. Although effective, they commonly suffer from limited data rate (215 bps for ZigBee → WiFi [34]) inherently imposed by the coarse-grained packet-level modulation. We note that there has been a recent advancement in fine-grained physical layer designs [16], [20]; However they are not applicable to ZigBee to WiFi scenario.

1. DEFINITION (EVIDENCE) — introduces CTC with citations.
2. CLAIM — CTC's dual value.
3. METHODOLOGY — names the modulation dimensions in prior work.
4. CONCESSION (CONTRAST) — "Although effective, … limited data rate".
5. CONCESSION — acknowledges recent PHY designs but rules them out.

**Shape:** DEFINITION → CLAIM → METHODOLOGY → CONCESSION → CONCESSION.
**Persuasive move:** gap-in-prior-work.
**Flow notes:** S1 introduces category; S2 praises; S3 "Specifically" enumerates; S4 "Although effective" turns; S5 "However" closes off remaining alternative.

---

## §I. Introduction · p.500 · ¶3

**Verbatim:**
> This work introduces SymBee, a novel symbol-level ZigBee to WiFi CTC reaching up to 31.25Kbps, thereby improving packet-level predecessors by 145.4×. SymBee is uniquely built on the new insight on ZigBee-WiFi cross-observability. I.e., due to frequency overlap, ZigBee signal naturally flows into the WiFi RF front-end to be processed by the idle listening mechanism, where the output of the module illustrates how ZigBee signal is observed at WiFi. This reflects radio asymmetries between ZigBee and WiFi, including sampling rate/bandwidth and central frequencies. By carefully inspecting through the physical layer properties of ZigBee signal, SymBee is designed so that the combinations of ZigBee symbols (thus symbol-level) yield a special output pattern at the WiFi idle listening to maximize decoding reliability.

1. CLAIM — introduces SymBee and headline result.
2. CLAIM — names the core insight.
3. DEFINITION (CAUSE) — explains cross-observability mechanism.
4. INTERPRETATION — what cross-observability reflects.
5. METHODOLOGY (CLAIM) — how the design uses it.

**Shape:** CLAIM → CLAIM → DEFINITION → INTERPRETATION → METHODOLOGY.
**Persuasive move:** design-derived-from-phenomenon.
**Flow notes:** S1 announces; S2 names insight; S3 "I.e." unpacks; S4 generalizes; S5 closes with design implication.

---

## §I. Introduction · p.500 · ¶4

**Verbatim:**
> The highlight of SymBee design lies in its simplicity and compatibility – despite the underlying rationale involving in-depth details on physical layer signal, SymBee encoding turns out to be as simple as putting specific byte patterns in a payload of a legitimate ZigBee packet, which we call payload encoding. Specifically, the bytes are selected such that the corresponding ZigBee symbols generate intended output upon cross-observation at WiFi idle listening. This ensures full compatibility to the off-the-shelf ZigBee device and WiFi standard. Furthermore, since WiFi idle listening continuously runs by default (to detect any incoming WiFi packets), recycling its output amortizes much of the computational cost and enables light-weight decoding. This keeps the overhead and intrusion to the legacy communication minimal.

1. CLAIM (DEFINITION) — simplicity/compatibility + names "payload encoding".
2. METHODOLOGY — how bytes are selected.
3. CONSEQUENCE — compatibility.
4. CAUSE (CONSEQUENCE) — recycling idle listening output.
5. CONSEQUENCE — minimal overhead.

**Shape:** CLAIM → METHODOLOGY → CONSEQUENCE → CAUSE → CONSEQUENCE.
**Persuasive move:** simplicity-and-compatibility pitch.
**Flow notes:** S1 frames highlight via em-dash concession ("despite"); S2 "Specifically" details; S3-S5 chain "This ensures", "Furthermore", "This keeps" to stack benefits.

---

## §I. Introduction · p.500 · ¶5

**Verbatim:**
> To summerize, SymBee is designed as a ZigBee to WiFi CTC in the aim to support upstream (or convergecast) which takes majority portion of IoT traffic, e.g., uploading sensing data. The contribution of this work is three-fold:

1. SCOPE (CLAIM) — recap of design aim and applicability.
2. SCOPE — announces 3-fold contribution list.

**Shape:** SCOPE → SCOPE.
**Persuasive move:** deployment-justification + roadmap.
**Flow notes:** "To summerize" signals recap; second sentence pivots to bullet list.

---

## §I. Introduction · p.500-501 · ¶6 (contribution bullets)

Paragraph is a list of bullets — reproduced verbatim, not sentence-tagged.

> • To the best of our knowledge, for the first time, we analyze the physical layer cross-observability of ZigBee signal at WiFi RF front-end (via packet detection mechanism), both experimentally and analytically through rigorous derivations. Although the case of WiFi and ZigBee was investigated, the observation can be generally applied to understanding the interaction between disparate RF front-ends, and can be extended to designing symbol-level CTC between other technologies.
> • A novel ZigBee to WiFi CTC of SymBee is introduced. Payload encoding is as simple as customizing byte patterns in the payload of a legitimate ZigBee packet, which can be performed on any off-the-shelf devices. Decoding facilitates the default WiFi idle listening operation defined in the 802.11 standard, making SymBee a non-intrusive and energy-economic design. Simple preamble-based decoding enables robust communication under heavy noise.
> • We evaluate SymBee both analytically and experimentally, where we implement the prototype on TelosB and USRP platforms. We extensively evaluate the performance under six different indoor and outdoor scenarios, where it is demonstrated that SymBee throughput reaches 31.25kbps, improving the state-of-the-art by more than 145.4×.

**Shape:** bullet list — contribution catalog.
**Persuasive move:** contribution-enumeration.
**Flow notes:** three parallel bullets: (1) analytical insight, (2) design, (3) evaluation.

---

## §I. Introduction · p.501 · ¶7 (roadmap)

**Verbatim:**
> The rest of this paper is organized as follows. Section II introduces the motivation, where Section III provides a design overview. Section IV presents technical details of our design, followed by a robust technique and other notable features in Sections V and VI, respectively. Analytical and experimental performances of SymBee are in Sections VII and VIII. Related work are discussed in Section IX. Finally, Section X concludes the paper.

1. SCOPE — roadmap header.
2. SCOPE.
3. SCOPE.
4. SCOPE.
5. SCOPE.
6. SCOPE.

**Shape:** SCOPE×6.
**Persuasive move:** paper-roadmap.
**Flow notes:** standard "Section II … Finally, Section X" structure.

---

## §II. Motivation · p.501 · ¶1

**Verbatim:**
> This section illustrates the values that SymBee would bring in diverse domains for wireless networking, followed by the need for symbol-level ZigBee to WiFi CTC.

Single-sentence paragraph: SCOPE.
**Shape:** SCOPE.
**Persuasive move:** section-scope.
**Flow notes:** pure orientation sentence.

---

## §II.A Opportunities for CTC · p.501 · ¶1

**Verbatim:**
> With ever-increasing body of devices with incompatible heterogeneous wireless technologies, CTI has now become one of the major causes of network degradation [14], [37], [38]. This is even more severe for the low-power wireless techniques such as ZigBee, where it has been reported that up to 50% of the ZigBee packets are lost due to WiFi interference [21]. SymBee offers opportunity to mitigate the CTI and coordinate heterogeneous devices via, for example, (i) explicit coordination among IoT devices using cross-technology RTS/CTS instead of implicit CSMA/CA, (ii) cross-technology broadcasting of ZigBee spectrum occupancy to assist WiFi devices to switch to idle or less crowded channels. Such cross-technology channel coordination requires real-time and distributed message exchages, which can be exclusively achieved by CTC. Additionally, CTC enables IoT devices to deliver data (e.g., sensing info.) directly to WiFi (i.e., to the Internet and cloud), subtantially enhancing IoT in various aspects including connectivity, accessibility, and responsiveness.

1. CLAIM (EVIDENCE) — CTI is major degradation cause.
2. EVIDENCE — 50% loss figure for ZigBee.
3. CLAIM (EXAMPLE) — SymBee's opportunities with two examples.
4. PREMISE — requirement only CTC satisfies.
5. CONSEQUENCE — IoT enhancements.

**Shape:** CLAIM → EVIDENCE → CLAIM → PREMISE → CONSEQUENCE.
**Persuasive move:** opportunity-from-problem.
**Flow notes:** S1 problem; S2 "even more severe" intensifies; S3 "SymBee offers opportunity … for example" pivots to solutions; S4-S5 justify uniqueness and expand.

---

## §II.B The Need for Symbol-level CTC · p.501 · ¶1 (Limitation of Gateway)

**Verbatim:**
> **Limitation of Gateway.** Traditionally solution for communication between multiple technolologies has been through mult-radio gateways. However, deployment of gateways impose several practical issues (especially compared to CTC) such as: (i) increase in communication hops, leading to not only more delay, and more importantly, more traffic (flowing into and out from gateway). This further aggravates competition in ISM spectrum, (ii) increase in network deployment complexity, and (iii) the lack of mobility support.

1. PREMISE — names traditional gateway solution.
2. CLAIM — gateways have practical issues (introduces list).
3. CONSEQUENCE — aggravation of ISM spectrum competition (embedded in list).

**Shape:** PREMISE → CLAIM → CONSEQUENCE.
**Persuasive move:** alternative-elimination (gateways).
**Flow notes:** S1 sets baseline; S2 "However" pivots and enumerates; S3 expands one of the items.

---

## §II.B · p.501-502 · ¶2 (Limitations of the State-of-the-art)

**Verbatim:**
> **Limitations of the State-of-the-art.** A stream of CTC designs in literature take packet-level approaches [7], [9], [19], [38], where they use the packet as the basic unit in modulation (analogous to 'pulse' in physical layer) – E.g., [9] uses power of each packet to modulate CTC message. While packet-level designs are simple to adopt and are highly compatible with the legacy devices, they inherently suffer from bounded bandwidth, or throughput. For example, the state-of-the-art ZigBee to WiFi CTC reports the throughput of 215bps [34], limiting the usage to delivering short control information. A recent line of CTC studies take physical-layer approaches, where signal from one wireless device closely emulates the waveform of the other [16], [20]. Despite their vastly enhanced speeds, they are inherenly inapplicable to ZigBee to WiFi CTC due to a large bandwidth gap (2 vs 20MHz) between the technologies – essentially limiting the ZigBee signal's degree of freedom to fall short (for emulating WiFi).

1. DEFINITION (EVIDENCE) — packet-level CTC defined with example.
2. CONCESSION (CLAIM) — simple but bandwidth-bounded.
3. EVIDENCE — 215bps state-of-the-art figure.
4. DEFINITION — physical-layer CTC approach.
5. CONCESSION (CAUSE) — bandwidth gap makes them inapplicable.

**Shape:** DEFINITION → CONCESSION → EVIDENCE → DEFINITION → CONCESSION.
**Persuasive move:** two-fold-gap (packet-level and PHY-level both fail here).
**Flow notes:** S1 names category; S2 "While … inherently"; S3 "For example" anchors; S4 introduces second category; S5 "Despite … inherenly inapplicable" closes second door.

---

## §II.B · p.502 · ¶3 (Advantages and Challenges of Symbol-level CTC)

**Verbatim:**
> **Advantages and Challenges of Symbol-level CTC.** This calls for a new method (i.e., Symbol-level approach) for the breakthrough in the rate, and to expand the practical use of ZigBee to WiFi CTC. Considering duration of the minimal ZigBee packet of 576 us (i.e., 18 bytes), the bandwidth of packet-level CTC becomes 1.736KHz (=1/576us). Conversely, exploring physical layer for symbol (16μs) level CTC expands the bandwidth to 62.5KHz. By Shannon's law, the theoretical bound of the data rate is linear to the bandwidth, therefore, symbol-level approach can vastly improve the throughput of the packet-level approach by 36×. The challenge of symbol-level CTC is in maintaining the compatibility to the legacy devices. This is because symbols are strictly defined in the physical layer where their manipulation could easily lead to standard incompatibility, if not carefully designed.

1. CLAIM (BRIDGE) — calls for symbol-level method.
2. EVIDENCE — packet-level bandwidth.
3. EVIDENCE (CONTRAST) — symbol-level bandwidth (Conversely).
4. CLAIM (CAUSE) — Shannon-based throughput gain.
5. CONCESSION — challenge of compatibility.
6. CAUSE — why compatibility is fragile.

**Shape:** CLAIM → EVIDENCE → EVIDENCE → CLAIM → CONCESSION → CAUSE.
**Persuasive move:** quantified-opportunity-then-challenge.
**Flow notes:** S1 sets target; S2/S3 "Conversely" compute the gap; S4 "By Shannon's law … therefore" claims the prize; S5-S6 acknowledge the cost.

---

## §III. Design Overview and Background · p.502 · ¶1

**Verbatim:**
> This section discusses the overview followed by technical background of our design design.

Single-sentence paragraph: SCOPE.
**Shape:** SCOPE.
**Persuasive move:** section-scope.
**Flow notes:** standalone orientation.

---

## §III.A SymBee in a Nutshell · p.502-503 · ¶1

**Verbatim:**
> SymBee is a ZigBee to WiFi CTC technique that vastly improves the data rate of the state-of-the-art designs by exploiting physical layer signatures. This is effectively achieved by the two unique features: (i) ZigBee's physical layer signature is indirectly controlled by manipulating the payload bytes, which we call payload encoding, such that (ii) the payload exposes intended (i.e., message-bearing) patterns when the it is processed by the WiFi's idle listening mechanism. In other words, SymBee is carefully designed not only considering the physical layer properties of ZigBee and WiFi, but also their cross-observability – i.e., how ZigBee signal is processed when fed into WiFi's idle listening.

1. CLAIM (DEFINITION) — SymBee's headline.
2. METHODOLOGY (DEFINITION) — two features.
3. INTERPRETATION — restates design philosophy.

**Shape:** CLAIM → METHODOLOGY → INTERPRETATION.
**Persuasive move:** design-in-a-nutshell.
**Flow notes:** S1 announces; S2 "two unique features"; S3 "In other words" reframes.

---

## §III.A · p.503 · ¶2

**Verbatim:**
> SymBee design is extremely light-weight and fully compatible to standards, making it nondistruptive to ZigBee and WiFi operations. Figure 1 illustrates how SymBee message is embedded into ZigBee packet payload. Encoding at the transmitter (i.e., ZigBee) is as simple as selecting byte patterns of the payload, which does not require any hardware/firmware change to commodity ZigBee platform. Decoding at WiFi receiver recycles the computational result of the idle listening which runs continuously by default, thereby minimizing the computation cost while maintaining compatibility to the WiFi standard.

1. CLAIM — light-weight + compatible.
2. METHODOLOGY — figure reference.
3. METHODOLOGY (CONSEQUENCE) — transmitter encoding simplicity.
4. METHODOLOGY (CONSEQUENCE) — receiver recycling.

**Shape:** CLAIM → METHODOLOGY → METHODOLOGY → METHODOLOGY.
**Persuasive move:** non-intrusiveness pitch.
**Flow notes:** S1 thesis; S2 visual anchor; S3/S4 list two sides (tx, rx) with consequences embedded.

---

## §III.B ZigBee-WiFi Cross-observability · p.503 · ¶1

**Verbatim:**
> We discusses how ZigBee signal is cross-observed at the WiFi idle listening, which serves as the basis of our design.

Single-sentence paragraph: SCOPE.
**Shape:** SCOPE.
**Persuasive move:** subsection-scope.
**Flow notes:** orientation.

---

## §III.B · p.503 · ¶2 (ZigBee transmitter)

**Verbatim:**
> **ZigBee transmitter.** Figure 2 illustrates the ZigBee transmitter running Offset Quadrature Phase-Shift Keying (OQPSK) modulation, which covers the entire conversion procedure from input symbol to output signal wave. Initially, data to be transmitted is cut in units of 4bits called symbol (thus ranging 0-F). Then, in step (i), each symbol is mapped to unique 32 chip sequences as per Table I – a technique called Direct Sequence Spread Spectrum (DSSS).

1. METHODOLOGY — figure references OQPSK chain.
2. DEFINITION — symbol = 4 bits.
3. METHODOLOGY (DEFINITION) — DSSS chip mapping.

**Shape:** METHODOLOGY → DEFINITION → METHODOLOGY.
**Persuasive move:** background-walkthrough.
**Flow notes:** "Initially … Then" stepwise narration.

---

## §III.B · p.503 · ¶3 (after Table I, transmitter continued)

**Verbatim:**
> In step (ii), 32 chips are divided into odd and even chips where chip 0 and 1 are converted to positive and negative half-sine pulses, respectively. Pulses generated from odd chips are referred to as in-phase signal. On the other hand, in (iv) pulses from even chips are delayed by half pulse duration of 0.5us, and are referred to as Quadrature signal. Figure 3 depicts an example of in-phase and quadrature (i.e., I/Q) signals corresponding to symbol 6. Gray indicates both in-phase and quadrature are continuous sinusoidal, which can easily be cross-observed at WiFi (details in the following parts of the section). I/Q signals are merged and in step (iv), converted to analog continuous waveform via DAC. Finally, in (v) the mixer shifts the baseband signal to the carrier frequency (i.e., passband) which is then pushed to the antenna for transmission.

1. METHODOLOGY — chip-to-pulse conversion.
2. DEFINITION — in-phase.
3. DEFINITION — quadrature.
4. METHODOLOGY — figure 3 reference.
5. INTERPRETATION (CLAIM) — gray = continuous sinusoidal = cross-observable.
6. METHODOLOGY — DAC step.
7. METHODOLOGY — mixer to passband.

**Shape:** METHODOLOGY → DEFINITION → DEFINITION → METHODOLOGY → INTERPRETATION → METHODOLOGY → METHODOLOGY.
**Persuasive move:** stepwise-background, planting key insight (S5) mid-way.
**Flow notes:** "In step (ii) … On the other hand … Figure 3 … Gray indicates … Finally" — sequential walk; the gray-region remark seeds the design rationale.

---

## §III.B · p.503 · ¶4 (WiFi receiver)

**Verbatim:**
> **WiFi receiver.** To provide insight on cross-observability, Figure 4 illustrates WiFi receiver operation up to idle listening. Frequency overlap with WiFi naturally lead ZigBee signal to flow into WiFi RF front-end. Then, in (a) mixer shifts this signal to baseband, where in (b) it is sampled to digital domain at 20Msps (i.e., Nyquist rate) or above. The samples are fed into the idle listening. In search for WiFi packet, idle listening continuously processes any incoming signal including ZigBee. This is done in (c) by computing the phase differences between 16 samples denoted as p[n]. WiFi signal is prepended with Short Training Sequence (STS), which is a sequence of repeated signals with interval of 16 samples (i.e., 0.8μs) for the total duration 160 samples. In other words, (d) detects WiFi packet only when p[n]=0 holds for STS duration, which then passes the signal for demodulation.

1. SCOPE — frames figure 4.
2. CAUSE — frequency overlap routes ZigBee into WiFi front-end.
3. METHODOLOGY — mixer + ADC.
4. METHODOLOGY — feed to idle listening.
5. METHODOLOGY — idle listening processes everything.
6. DEFINITION — p[n] phase difference.
7. DEFINITION — STS structure.
8. INTERPRETATION — detection condition.

**Shape:** SCOPE → CAUSE → METHODOLOGY → METHODOLOGY → METHODOLOGY → DEFINITION → DEFINITION → INTERPRETATION.
**Persuasive move:** background-walkthrough with embedded cross-observability rationale.
**Flow notes:** "Then … In search for … In other words" chained walkthrough.

---

## §III.B · p.503-504 · ¶5 (Cross-observation of symbol 6)

**Verbatim:**
> When ZigBee signal is fed into idle listening, (c) yields corresponding p[n] pattern of the signal, which is the cross-observation of the ZigBee from WiFi. Figure 5 depicts an example for ZigBee symbol 6 obtained from our experiment. With the mathematical derivation of phase (i.e., p[n]) in Appendix A, the figure clearly demonstrates that cross-observation outputs notable patterns where the phase is kept stable in the gray region. This corresponds to the gray portion in Figure 3, where in-phase and quadrature signals are both continuous sinusoidal waves, indicating that such stable phases are easily detectable with minimal computational overhead (details in Section IV-C). SymBee exploits this unique feature in cross-observation to achieve reliable and light-weight CTC. To sum up, by (i) exploring the intrinsic properties of ZigBee symbols as defined in the ZigBee standard, and (ii) recycling the phase values computed by default in WiFi, SymBee remains fully compatible to both standards and non-disruptive to their operation.

1. DEFINITION (CLAIM) — p[n] is the cross-observation.
2. EVIDENCE — figure 5 experimental example.
3. INTERPRETATION (EVIDENCE) — stable phase in gray.
4. INTERPRETATION (CONSEQUENCE) — easily detectable.
5. CLAIM — SymBee exploits this.
6. CLAIM (BRIDGE) — compatibility summary.

**Shape:** DEFINITION → EVIDENCE → INTERPRETATION → INTERPRETATION → CLAIM → CLAIM.
**Persuasive move:** insight-then-exploit.
**Flow notes:** "When … Figure 5 … This corresponds … SymBee exploits … To sum up" — climax structure.

---

## §IV. SymBee Design · p.504 · ¶1

**Verbatim:**
> This section provides technical details and insights on SymBee.

Single-sentence paragraph: SCOPE.
**Shape:** SCOPE.
**Persuasive move:** section-scope.
**Flow notes:** orientation.

---

## §IV.A Robust Payload Encoding@ZigBee · p.504 · ¶1

**Verbatim:**
> SymBee's payload encoding is built on top of the observation on the stable phase (i.e., p[n]), where we design a technique to maximize robustness. SymBee essentially selects optimal combinations of ZigBee symbols such that (i) they yield the longest possible stable phase that maximizes detection under noise and interference, and at the same time, (ii) the phase of different combinations are maximally distinct, which minimizes decoding errors. The combinations are assembled from the 16 (i.e., 0-F) symbols defined in the ZigBee standard (Table I), thereby optimizing the performance while maintaining full compatibility to tens of millions of commercial ZigBee devices.

1. CLAIM — encoding built on stable phase observation.
2. METHODOLOGY (CLAIM) — twin optimization criteria.
3. CLAIM (CONSEQUENCE) — uses standard symbols → compatibility.

**Shape:** CLAIM → METHODOLOGY → CLAIM.
**Persuasive move:** design-criteria-statement.
**Flow notes:** S1 grounds in earlier insight; S2 enumerates (i)(ii); S3 closes with compatibility consequence.

---

## §IV.A · p.504 · ¶2

**Verbatim:**
> Specifically, ZigBee symbol combinations of 6 and 7 are used to convey SymBee bit 0 while E and F represent bit 1. For simplicity, the combinations are denoted as (6,7) and (E,F) thereafter. Given the symbol combinations, SymBee's payload encoding is as simple as converting bits to be transmitted, to either (6,7) and (E,F). We note that since a symbol is worth 4 bits, combination of two symbols is expressed as a single byte. In other words, every SymBee bit is represented as a byte put into the payload of a ZigBee packet, thus encoding payload. In the following, we study the uniqueness and the optimality of the symbol combinations (6,7) and (E,F) in terms of communication robustness (testbed-evaluated against noise and interference in Section VIII).

1. METHODOLOGY (DEFINITION) — bit-to-combination mapping.
2. DEFINITION — notation.
3. METHODOLOGY — encoding procedure.
4. PREMISE — 4-bit symbol arithmetic.
5. INTERPRETATION — payload-encoding restatement.
6. SCOPE (BRIDGE) — what follows.

**Shape:** METHODOLOGY → DEFINITION → METHODOLOGY → PREMISE → INTERPRETATION → SCOPE.
**Persuasive move:** mechanism-spec then forward-pointer.
**Flow notes:** "Specifically … For simplicity … We note … In other words … In the following" — stepwise definition.

---

## §IV.A · p.504 · ¶3 (Longest Stable Phase)

**Verbatim:**
> **Longest Stable Phase.** To maximize robustness to noise/interference, symbol combinations with the longest stable phase are found through a careful analysis on chip sequences of symbols – such that serially concatenating them yields the longest continuous sinusoidal wave at the junction. Figure 6 illustrates the two combinations with the longest stable phase, (6,7) and (E,F) representing SymBee bits 1 and 0, respectively. Parts with continuous sinusoidal I/Q (cross-observed as stable phase) are marked in gray, which are 5μs long. This is reflected as 4.2μs-long stable phases at cross-observing WiFi (0.8μs difference corresponds to 16 samples) for both (6,7) and (E,F). The two combinations yield the longest stable phase among any other combinations with arbitrary number of symbols, indicating that they are indeed the optimal selection for maximum robustness to noise/interference.

1. METHODOLOGY (CLAIM) — selection criterion = longest stable phase.
2. EVIDENCE — figure 6 shows the chosen pair.
3. EVIDENCE — measurement (5μs).
4. EVIDENCE (INTERPRETATION) — 4.2μs cross-observed.
5. CLAIM — optimality among all combinations.

**Shape:** METHODOLOGY → EVIDENCE → EVIDENCE → EVIDENCE → CLAIM.
**Persuasive move:** optimality-argument.
**Flow notes:** S1 states aim; S2-S4 visual/numerical proofs; S5 "indicating that … indeed the optimal" closes.

---

## §IV.A · p.504 · ¶4 (Optimized SymBee Bit Distinction)

**Verbatim:**
> **Optimized SymBee Bit Distinction.** The symbol combinations of (6,7) and (E,F) has another powerful characteristic that optimizes them for robust decoding. Figure 7 demonstrates the actual p[n] values when the SymBee bits 0 and 1 are sequentially transmitted, where the stable phase corresponding to the SymBee bits are in gray. As noticeable in the figure, stable phases indicating 0 and 1 are ∓4π/5, respectively, which correspond to the minimum and maximum among all possible phase values in the cross-observation (derivation in the section IV-B). In summary, (6,7) and (E,F) are optimized to decoding, with maximum possible distinction between 0 and 1 (i.e., 8π/5).

1. CLAIM — second optimality property.
2. EVIDENCE — figure 7 demonstration.
3. INTERPRETATION (EVIDENCE) — phase values are extrema.
4. CLAIM — summary of maximal distinction.

**Shape:** CLAIM → EVIDENCE → INTERPRETATION → CLAIM.
**Persuasive move:** second-optimality-property.
**Flow notes:** S1 forecasts; S2 visualizes; S3 "As noticeable … which correspond" interprets; S4 "In summary" closes.

---

## §IV.B Understanding Stable Phase · p.504-505 · ¶1

**Verbatim:**
> This section provides conceptual description of the stable p[n], followed by a mathematical derivation of the value. Figure 8 shows the continuous sinusoidal signal generated by (6,7) (i.e., gray area in Figure 6(a).) as well as the π/2 phase difference between in-phase and quadrature signal caused by half-chip (0.5μs) offset. The stable phase indicates that such a signal, when fed into WiFi idle listening, yields consistent phase difference. Specifically, p[n] = 4π/5 between 16 samples at 20Msps. Derivation of the stable phase difference is as follows.

1. SCOPE — section preview.
2. METHODOLOGY (EVIDENCE) — figure 8.
3. INTERPRETATION — stable phase meaning.
4. EVIDENCE — quantitative value.
5. BRIDGE — derivation forthcoming.

**Shape:** SCOPE → METHODOLOGY → INTERPRETATION → EVIDENCE → BRIDGE.
**Persuasive move:** scope-then-anchor-then-forward.
**Flow notes:** "Figure 8 shows … Specifically … as follows" — orientation to derivation handoff.

---

## §IV.B · p.505 · ¶2 (Derivation — prose around equations)

**Verbatim:**
> **Derivation.** As depicted in Figure 8, 1μs half-sine chip indicates the frequency of the continuous sinusoid is 0.5MHz. The figure also shows π/2 phase difference between In-phase and Quadrature continuous sinusoidal; therefore they can effectively be presented as −cos(2π·10^6/2·t) and sin(2π·10^6/2·t) respectively, or −e^{−j2π·10^6/2·t} in complex representation. Recall that, from Figure 4(c), p[n] is computed by the WiFi idle listening as:

1. EVIDENCE — frequency reading from figure 8.
2. METHODOLOGY (CAUSE) — representation as complex exponential.
3. BRIDGE — recall p[n] formula.

**Shape:** EVIDENCE → METHODOLOGY → BRIDGE.
**Persuasive move:** ground-then-formalize.
**Flow notes:** "As depicted … therefore … Recall that" linkage between intuition and math.

---

## §IV.B · p.505 · ¶3 (post-equation prose)

**Verbatim:**
> where x[n] is n-th sample and x∗[n + 16] is a complex conjugate of the sample n + 16. That is, p[n] essentially indicates the phase difference between the two samples, x[n] and x[n+16], which are 16 samples apart. Under sampling rate of 20MHz, 16 samples interval represents 0.8μs in time, which can be directly plugged in to t of −e^{−j2π·10^6/2·t} (i.e., complex representation of I/Q signals in (6,7)) to obtain p[n] = 4π/5. Similarly, p[n] = −4π/5 for continuous sinusoidal within (E,F), as its I/Q signal is precisely the conjugate of the continuous sinusoidal in (6,7) as can be observed in Figure 6. We note that the stable phase difference of ±4π/5 induced by (6,7) and (E,F) are kept consistent for 4.2μs until in-phase and/or quadrature becomes discontinuous, providing the longest stable phase among all ZigBee symbol combinations and optimal bit distinction with maximum and minimum p[n] among all 17 possibilities derived in Appendix A.

1. DEFINITION — variable explanation.
2. INTERPRETATION — what p[n] represents.
3. METHODOLOGY (EVIDENCE) — plug-in yields 4π/5.
4. CLAIM — symmetric result for (E,F).
5. CLAIM — uniqueness across 17 possibilities (recalling Appendix A).

**Shape:** DEFINITION → INTERPRETATION → METHODOLOGY → CLAIM → CLAIM.
**Persuasive move:** derivation-as-proof-of-optimality.
**Flow notes:** "where … That is … Similarly … We note that" - step-by-step proof culminating in optimality claim.

---

## §IV.C Extremely Light-weight Decoding@WiFi · p.505 · ¶1

**Verbatim:**
> The use of the stable phases enables extremely light-weight decoding. This is essentially done by checking the signs of phase values, where the decision boundary of 0 minimizes the error (i.e., negative ↔ nonnegative) under random noise. More specifically, since the stable phases is 4.2μs long consisting of 84 phase values, decoding is simply monitoring consecutive 84 phase values if they are consistently kept negative or nonnegative (i.e., below or above the decision boundary of 0), which indicate SymBee bit 0 or 1. In practice, phase values suffer from noise and up to threshold (τ) number of errors are tolerated. In our experiment τ is set to be 10 where both false positive and negative are kept under 3% at SNR as low as -5dB, a harsh SNR for most scenarios [31]. To sum up, at any time t, SymBee decoding is performed by monitoring the phase values with the window size of 84 – from p[n] to p[n + 83]. SymBee bit 1 or 0 is detected whenever the window has more than 84-τ negative or nonnegative values, respectively. In all other cases, SymBee bit is not detected.

1. CLAIM — light-weight decoding from stable phases.
2. METHODOLOGY — sign check / decision boundary.
3. METHODOLOGY — 84-value window check.
4. CONCESSION (METHODOLOGY) — noise tolerance via τ.
5. EVIDENCE — empirical τ=10, <3% errors at −5dB.
6. METHODOLOGY (CLAIM) — summary of algorithm.
7. METHODOLOGY — detection rule.
8. METHODOLOGY — default reject rule.

**Shape:** CLAIM → METHODOLOGY → METHODOLOGY → CONCESSION → EVIDENCE → METHODOLOGY → METHODOLOGY → METHODOLOGY.
**Persuasive move:** algorithm-specification with noise-robustness justification.
**Flow notes:** "More specifically … In practice … In our experiment … To sum up" — escalating specification.

---

## §V. Enhanced Decoding with SymBee Preamble · p.505 · ¶1

**Verbatim:**
> Here we introduce an optional technique that further improves the resilience to noise, by prepending a simple SymBee preamble. SymBee bit consists of a pair of ZigBee symbols(i.e., (6,7) or (E,F)), or in other words, SymBee bits repeat periodically with every two ZigBee symbols, corresponding to 640 samples (=32us). This indicates that the stable phases (i.e., the actual regions containing SymBee bits) is only 84 samples (=4.2us) out of 640, where other parts need not be considered in SymBee decoding. SymBee preamble essentially explores this property to substantially reduce the chance of decoding error. SymBee preamble, which is simply four consecutive SymBee bit 0, enables precise detection of the bit start time, such that only the part holding SymBee bit is considered in the decoding, effectively filtering out non-contributing portions.

1. SCOPE (CLAIM) — introduces preamble technique.
2. PREMISE — periodic structure of SymBee bits.
3. INTERPRETATION (CONSEQUENCE) — only 84 of 640 samples matter.
4. CLAIM — preamble exploits this.
5. METHODOLOGY (DEFINITION) — preamble = four 0 bits.

**Shape:** SCOPE → PREMISE → INTERPRETATION → CLAIM → METHODOLOGY.
**Persuasive move:** opportunity-from-structure.
**Flow notes:** S1 announces; S2 premise; S3 "This indicates" deduces; S4-S5 deploy mechanism.

---

## §V · p.505 · ¶2 (Capturing SymBee Preamble)

**Verbatim:**
> **Capturing SymBee Preamble.** Figure 9 illustrates SymBee preamble, or four consecutive 0's leading SymBee message. The preamble can be effectively and reliably captured via folding – a technique introduced in [30] to detect periodic signal under noise, which in our case, is the four repeated stable phases due to four (E,F) (i.e., SymBee bit 0). The process of folding is illustrated in Figure 10: One (E,F) takes up 640 values (32 μs), and hence, the length of the SymBee preamble is 640×4 = 2560 values. This vector of 2560 phase values are sliced into four subvectors of size 640 and stacked up (i.e., folded) to form a 4×640 matrix. Then, each columns are summed to yield a vector of size 640, which we refer to as Fold Sum. In other words, Fold sum = Σ p[n + 640i] where 0 ≤ n < 640. Then, capturing preamble is achieved by applying the decoding (described in Section IV-C) to the fold sum.

1. METHODOLOGY — figure 9 reference.
2. METHODOLOGY (EVIDENCE) — folding cited from [30].
3. METHODOLOGY — folding mechanics.
4. METHODOLOGY — matrix construction.
5. DEFINITION — Fold Sum.
6. DEFINITION — Fold Sum formula.
7. METHODOLOGY (CLAIM) — apply IV-C decoding to fold sum.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → DEFINITION → DEFINITION → METHODOLOGY.
**Persuasive move:** mechanism-spec borrowing prior technique.
**Flow notes:** "Figure 9 illustrates … via folding … The process … Then … In other words … Then" — recipe form.

---

## §V · p.506 · ¶3 (Figure 11 discussion)

**Verbatim:**
> Figure 11 demonstrates an example of preamble capturing in practice. The figure shows a scenario of poor signal quality (SNR = -10dB), where p[n] is very unstable and thus standard decoding is infeasible (top figure). Under the harsh environment, folding stabilizes the stable phase values (middle figure) and enables SymBee preamble to be safely captured (bottom figure) thereby significantly enhancing robustness. We also note that preamble can be further protected by increasing the repetitions, where four offered reliable (>98.7%) capturing in our experiments under low SNR of -10dB.

1. EVIDENCE — figure 11 example.
2. EVIDENCE (CAUSE) — −10dB makes plain decoding infeasible.
3. CLAIM (EVIDENCE) — folding rescues capture.
4. EVIDENCE — 98.7% capture rate.

**Shape:** EVIDENCE → EVIDENCE → CLAIM → EVIDENCE.
**Persuasive move:** robustness-by-construction.
**Flow notes:** S1 sets up case; S2 baseline failure; S3 "Under the harsh environment, folding stabilizes" — pivot to win; S4 quantifies.

---

## §V · p.506 · ¶4 (Decoding under synchronized bit timing)

**Verbatim:**
> **Decoding under synchronized bit timing.** Capturing preamble enables precisely locating the SymBee bits for decoding. This significantly decreases the chance of error, by skipping regions that does not bear the SymBee bits. Locating the ZigBee bit is straight-forward; Upon capturing preamble, the receiver marks the index of the initial phase value within the captured preamble. Suppose the index is n0, then the first symbol (i.e., initial part of SymBee data) starts at n1 = n0 + (640 × 4), which takes the preamble length into account. The following parts of the SymBee data occurs every 640, i.e., n2 = n1+640, n3 = t2+640, and so on, until reaching the end of the SymBee message. Decoding is performed on 84 samples starting at each bit location (e.g., nk to nk + 83 for k-th SymBee bits), where the samples outside the range are ignored as they do not embed any SymBee bit. Since the position of potential SymBee bits are already located, we set τsync = 42 (i.e., half of the stable phase) to decide the SymBee bits: out of 84 values in stable phases, τsync or more above 0 indicates SymBee bit 1, otherwise, 0 – essentially turning decoding to majority voting for higher noise tolerance (cf. Section IV-C).

1. CLAIM — preamble enables precise localization.
2. CONSEQUENCE — error reduction via skipping.
3. METHODOLOGY — locate initial index.
4. METHODOLOGY — index arithmetic.
5. METHODOLOGY — subsequent bits every 640.
6. METHODOLOGY — 84-sample decoding window.
7. METHODOLOGY (CLAIM) — τsync=42 majority vote.

**Shape:** CLAIM → CONSEQUENCE → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY.
**Persuasive move:** synchronized-decoding-spec.
**Flow notes:** S1 thesis; S2 "This significantly decreases …"; S3-S7 walkthrough; S7 closes with "essentially turning decoding to majority voting" reframing.

---

## §VI. SymBee Features · p.506 · ¶1

**Verbatim:**
> This section discusses unique features and simple extensions that enable a boader applicability.

Single-sentence paragraph: SCOPE.
**Shape:** SCOPE.
**Persuasive move:** section-scope.
**Flow notes:** standalone orientation.

---

## §VI.A WiFi-ZigBee Cross-technology Broadcast · p.506 · ¶1

**Verbatim:**
> SymBee message is embedded in a normal ZigBee packet, only with specific payload. Therefore, the same SymBee message can naturally be received by a ZigBee node simultaneously as being delivered to WiFi – i.e., SymBee is capable of transmitting cross-technology broadcast to both WiFi and ZigBee. SymBee message reception at the ZigBee is done in two simple steps: First, SymBee preamble is captured with four consecutive bytes of 0x67 (i.e., symbols (6,7)), corresponding to four SymBee bits of '0'. Then, following bytes of 0x67 or 0xEF are interpreted as bit '0' or '1', respectively. We note that this can be done at the application code on any standard ZigBee device, without any change to the firmware.

1. PREMISE — embedding in normal packet.
2. CLAIM (CONSEQUENCE) — dual reception → cross-technology broadcast.
3. METHODOLOGY — preamble capture at ZigBee.
4. METHODOLOGY — bit interpretation.
5. CLAIM — application-layer only, no firmware change.

**Shape:** PREMISE → CLAIM → METHODOLOGY → METHODOLOGY → CLAIM.
**Persuasive move:** bonus-capability-from-design.
**Flow notes:** "Therefore … i.e. … First … Then … We note that" — logical cascade.

---

## §VI.A · p.506 · ¶2

**Verbatim:**
> Such cross-technology broadcast may serve as a key enabler to various services, including explicit channel access control between WiFi and ZigBee. For example, a SymBee message may include the time/frequency allocation for ZigBee, which is notified to WiFi (to restrain channel usage) and ZigBee (to promote channel usage) at the same time. This would yield precise, efficient, and immediate channel utilization; that is, without the inevitable inefficiency and potential risk of interference that typical implicit, contention-based channel access (e.g., CSMA/CA) mechanisms commonly suffer from.

1. CLAIM — broadcast enables services.
2. EXAMPLE — time/frequency allocation message.
3. CONSEQUENCE (CONTRAST) — precise utilization vs CSMA/CA.

**Shape:** CLAIM → EXAMPLE → CONSEQUENCE.
**Persuasive move:** application-vignette.
**Flow notes:** "For example … This would yield … that is, without …" contrast against legacy.

---

## §VI.B Compatibility to 40MHz WiFi · p.506 · ¶1

**Verbatim:**
> Technical descriptions throughout the paper focuses on the widely deployed WiFi with 20MHz bandwidths (e.g., 802.11g/n), only for the sake of clarity. SymBee is in fact fully compatible to 40MHz 802.11n WiFi, with the sender side (i.e., ZigBee) kept identical. The receiver side (i.e., WiFi) is simply scaled to cope with the doubled sampling rate, which enhanced the decoding reliability. Specifically, the p[n] is computed as p[n] = ∠(x[n]x∗[n + 32]) as per twice the sampling rate (cf. Eq. 1). The stable phase values remain ±4π/5, while the number of stable phase values is doubled to 168 (= 84×2). To locate SymBee bits, 640×4×2 = 5120 phase values should be skipped following the SymBee preamble. At the decoding stage, the interval between two SymBee bits is 1280 at 40MHz WiFi receiver, as opposed to 640 under 20MHz WiFi. Finally, 84 steady phase values above 0 (i.e., the decision boundary) out of 168 indicate SymBee bit '1', and '0' otherwise. Overall, doubled stable phase values improves the robustness with the capacity to tolerate twice the errors.

1. SCOPE (CONCESSION) — paper focused on 20MHz for clarity.
2. CLAIM — also compatible with 40MHz.
3. CLAIM — receiver scaled, more reliable.
4. METHODOLOGY — new p[n] formula.
5. METHODOLOGY — doubled stable values.
6. METHODOLOGY — preamble skip count.
7. METHODOLOGY — interval doubles.
8. METHODOLOGY — decision rule.
9. CLAIM (CONSEQUENCE) — twice the error tolerance.

**Shape:** SCOPE → CLAIM → CLAIM → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CLAIM.
**Persuasive move:** extension-and-robustness-bonus.
**Flow notes:** "Technical descriptions … focuses on 20MHz … SymBee is in fact fully compatible to 40MHz … Specifically … Finally … Overall" — full bookended extension.

---

## §VII. Analytics · p.506 · ¶1

**Verbatim:**
> This section offers analysis on Bit Error Rate (BER) of SymBee, followed by the bitrate. BER is computed with respect to SNR;. Intuitively, low SNR (high noise) leads to phase value to fluctuate out of decision boundary to cause decoding error. We use Pr to denote the the probability of error in phase value (i.e., crossing the boundary). More specifically, this is when the phase value of SymBee bit 0 or 1 is higher or lower than the decision boundary, respectively. This yields Pr = Pr(p[n] > 0|bit = 0) = Pr(p[n] < 0|bit = 1) under random noise. Since SymBee bit is decoded following the majority voting, BER is computed as: [Eq. 2] where the distribution of Pr under different SNR is obtained from widely-used GNURadio. As shown in Figure 12, BER of SymBee is lower than 10% even under SNR of -10dB. On the other hand, ZigBee throughput can be found via straightforward computation: Since SymBee transmits 1 bit per two ZigBee symbols while ZigBee delivers 4 bits per symbol. This yields 1/8 the bitrate compared to ZigBee, i.e., 31.25kbps.

1. SCOPE — section preview.
2. METHODOLOGY — BER vs SNR.
3. CAUSE — low SNR → fluctuation.
4. DEFINITION — Pr notation.
5. INTERPRETATION (DEFINITION) — when error occurs.
6. METHODOLOGY (DEFINITION) — formal Pr expression.
7. METHODOLOGY — BER formula (with eq).
8. EVIDENCE (CLAIM) — Fig 12: BER<10% at −10dB.
9. METHODOLOGY (CONTRAST) — throughput computation transition.
10. PREMISE — 1 bit / 2 symbols, 4 bits/symbol.
11. CLAIM — 31.25kbps result.

**Shape:** SCOPE → METHODOLOGY → CAUSE → DEFINITION → INTERPRETATION → METHODOLOGY → METHODOLOGY → EVIDENCE → METHODOLOGY → PREMISE → CLAIM.
**Persuasive move:** analytical-validation.
**Flow notes:** "Intuitively … We use … More specifically … This yields … As shown … On the other hand" — derivation-then-result twin track.

---

## §VIII. Evaluation · p.507 · ¶1

**Verbatim:**
> We implement SymBee prototype on TelosB and USRP B210 with GNURadio 3.7.9 [3] and evaluate them in six representative areas of outdoor, library, classroom, dormitory, office and mall, as illustrated in Figure 15. We set the maximum payload to 127 including 2 bytes control information, 1 byte data sequence and 2 bytes check sum. WiFi idle listening has been implemented as a built-in block in GNURadio. We extract phase information from this block and implement SymBee receiver on USRP B210. We also note that SymBee can be implemented in 802.11 compliant platforms such as WARP [1] with minimum code modification.

1. METHODOLOGY — platform and venues.
2. METHODOLOGY — payload format.
3. METHODOLOGY — idle listening block.
4. METHODOLOGY — receiver implementation.
5. CLAIM — portability to WARP.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CLAIM.
**Persuasive move:** setup-description.
**Flow notes:** flat enumeration; final "We also note" inserts compatibility claim.

---

## §VIII.A Throughput · p.507 · ¶1

**Verbatim:**
> We present the effectiveness of our design by evaluating throughput under six scenarios in Figure 15, at distance of 5∼25 meters. As shown in Figure 13, 31.25Kbps can be achieved within 15 meters while the throughput of SymBee still remains at 30Kbps at the distance of 25 meters in the outdoor scenario. Since there are no cross-technology interference and obstacles, the throughput of SymBee in outdoor scenario is higher than other scenarios. The throughput of classroom, ≥ 27.5Kbps within 25 meters, is the 2nd highest among 6 scenarios. In the dormitory environment mild WiFi traffic was occurring during the experiment, which caused mild interference to SymBee. In the office and dormitory, most computers are connected through high speed wire cables. However, since the number of private WiFi access points and users in office is less than in dormitory, SymBee in office achieves ≥ 26.9Kbps within 25 meters, which is higher than ≥ 25.8Kbps in the dormitory.

1. SCOPE — describes evaluation.
2. EVIDENCE — outdoor throughput.
3. CAUSE — explains outdoor superiority.
4. EVIDENCE — classroom value.
5. EVIDENCE (CAUSE) — dormitory mild interference.
6. PREMISE — wired connections in office/dormitory.
7. CAUSE (EVIDENCE) — fewer APs → office wins.

**Shape:** SCOPE → EVIDENCE → CAUSE → EVIDENCE → EVIDENCE → PREMISE → CAUSE.
**Persuasive move:** scenario-by-scenario explanation.
**Flow notes:** "As shown … Since … In the dormitory … However" - alternation between datum and explanation.

---

## §VIII.A · p.507 · ¶2

**Verbatim:**
> As shown in Figure 13, SymBee only achieves ≥ 21Kbps within 25 meters in the mall due to the signal blockage from shoppers and a large amount of private WiFi access points in the stores. In the library, almost all students are connected to campus WiFi via laptops or smartphones, causing significant WiFi interference. Therefore the throughput is lower than other scenarios. The throughput of SymBee in the mall and library achieve only ≥ 21 and ≥ 24.4Kbps within 25 meters, respectively.

1. EVIDENCE (CAUSE) — mall blockage and APs.
2. CAUSE — library WiFi interference.
3. CONSEQUENCE — lower throughput.
4. EVIDENCE — values for mall, library.

**Shape:** EVIDENCE → CAUSE → CONSEQUENCE → EVIDENCE.
**Persuasive move:** worst-case-explanation.
**Flow notes:** "As shown … due to … In the library … Therefore" - cause→consequence chains.

---

## §VIII.A · p.507 · ¶3

**Verbatim:**
> We also compare SymBee with FreeBee[19], A-FreeBee[19], EMF[8], DCTC[15], C-Morse[34], 5 cross-technology techniques supporting ZigBee to WiFi communication, in the same setting. In our experiment, ZigBee senders send out 100 packets with 50 repeated SymBee bits '01' per second with maximum transmission power (0 dBm). Throughput of C-morse is 215bps[34] when distance between sender and receiver is 1.5 meters in the office scenario. Figure.16 indicates SymBee outperforms C-Morse, the state-of-the-art ZigBee to WiFi cross-technology approach, by 145.4×.

1. METHODOLOGY — comparison setup.
2. METHODOLOGY — sender params.
3. EVIDENCE — C-Morse throughput baseline.
4. CLAIM (EVIDENCE) — SymBee wins by 145.4×.

**Shape:** METHODOLOGY → METHODOLOGY → EVIDENCE → CLAIM.
**Persuasive move:** comparative-superiority.
**Flow notes:** "We also compare … In our experiment … Throughput of … Figure.16 indicates" — setup→baseline→result.

---

## §VIII.B Bit Error Rate · p.507-508 · ¶1

**Verbatim:**
> We present robustness of our design by evaluating bit error rate (BER). The BER of 6 representative scenarios are presented in Figure 14. We can see the trend of BER in 6 areas clearly. SymBee reaches lowest BER, i.e. ≤ 5%, regardless of distance in the outdoor, showing that SymBee is robust enough to resist severe noise. For the indoor environments, SymBee achieves ≤ 10% bit error rate within 10 meters even in the crowded mall and library.

1. SCOPE — BER evaluation purpose.
2. METHODOLOGY — figure 14 reference.
3. INTERPRETATION — trend visible.
4. EVIDENCE (CLAIM) — ≤5% outdoor.
5. EVIDENCE — ≤10% indoor short range.

**Shape:** SCOPE → METHODOLOGY → INTERPRETATION → EVIDENCE → EVIDENCE.
**Persuasive move:** robustness-by-evidence.
**Flow notes:** purpose→figure→summary observation→quantitative anchors.

---

## §VIII.B · p.508 · ¶2

**Verbatim:**
> Figure 17 shows the constellation diagram of outdoor scenario along with the decoded SymBee bits when the 2 bits of '01' is sent 2500 times. The x-axis of this figure indicates the number of stable phases above the decision boundary for each SymBee symbol. Decoding is successful when SymBee bit 1 (blue square) resides inside the right part and SymBee bit 0 (red square) resides inside the left part of the constellation diagram. Figure 17 depicts the distribution of the dots, in which ≥ 98% are successfully decoded.

1. METHODOLOGY — figure 17 description.
2. DEFINITION — axis meaning.
3. DEFINITION — success criterion.
4. EVIDENCE (CLAIM) — ≥98% decoded.

**Shape:** METHODOLOGY → DEFINITION → DEFINITION → EVIDENCE.
**Persuasive move:** visual-evidence-anchor.
**Flow notes:** describe→define→criterion→result.

---

## §VIII.C None Line of Sight Scenario · p.508 · ¶1

**Verbatim:**
> Performance of NLOS setting is tested in office environment where we deploy ZigBee nodes at corridor and separate rooms as shown in Figure 18(a). In this evaluation, 4 ZigBee senders working with maximum Tx power on ZigBee channel 13 are placed at S1 ∼ S4 and WiFi receiver is placed at R. The throughput of S1 ∼ S4 are 29.5, 28.2, 27.9 and 27.3 Kbps respectively. Since S1 is the closest to R, throughput of S1 is highest among 4 nodes. Even though S3 is closer to R than S2, throughput of S3 is lower than S2 due to more blockages from walls. This indicates the walls decreases the throughput of SymBee severely along with the distance between sender and receiver.

1. METHODOLOGY — NLOS setup.
2. METHODOLOGY — sender placement.
3. EVIDENCE — throughput numbers.
4. CAUSE (INTERPRETATION) — closeness explains S1 highest.
5. CONTRAST (CAUSE) — S3 anomaly explained by walls.
6. CLAIM — walls degrade throughput.

**Shape:** METHODOLOGY → METHODOLOGY → EVIDENCE → CAUSE → CONTRAST → CLAIM.
**Persuasive move:** explained-anomaly.
**Flow notes:** "Since S1 … Even though S3 … This indicates" — exception-driven inference.

---

## §VIII.D Impact of Transmission Power · p.508 · ¶1

**Verbatim:**
> We investigate the impact of transmission power on SymBee. Transmission power of a ZigBee node affects its coverage. Different TX power yields different multi-path reflections and fading. We set a TelosB node to different Tx power (−15 ∼ 0dBm) and deploy it 5 meters away from WiFi receiver in the office at midnight and outdoor as a comparison. As shown in Figure.19(a), SymBee reaches BER ≤ 10% within -10dBm and ≤ 23% within -15dBm. As demonstrated in Figure 19, SNR of same TX power in the indoor environment is lower than outdoor, thus resulting in higher BER. This is because multi-path effect in indoor environment caused by the blockage and bounce of walls is much more severe than outdoor environment.

1. SCOPE — investigate TX power.
2. PREMISE — TX power affects coverage.
3. PREMISE — TX power affects multipath.
4. METHODOLOGY — setup.
5. EVIDENCE — BER values.
6. EVIDENCE (CAUSE) — indoor SNR lower → higher BER.
7. CAUSE — multipath explanation.

**Shape:** SCOPE → PREMISE → PREMISE → METHODOLOGY → EVIDENCE → EVIDENCE → CAUSE.
**Persuasive move:** parameter-sweep-with-mechanism.
**Flow notes:** premise stacking then experiment then explanation chain via "thus" and "This is because".

---

## §VIII.E Robustness to Interference · p.508 · ¶1

**Verbatim:**
> The ubiquitous WiFi interference is a major reason of ZigBee packet corruption. At first, we obverse that SymBee bits could always be decoded correctly even from the severe interfered signal. Figure 20 shows a segment of SymBee packet, where all SymBee bits are '1's, is interfered by a 270 μs WiFi signal. The signal to interference plus noise ratio is 0dB indicating that the WiFi is as strong as SymBee signal. The stable phase values is ideally 84 samples long while under interference it drops to approximate 60; but being still larger than 42, (i.e., the half of ideal length) it is correctly decoded. Thus, this SymBee packet is robust enough to overcome the 0dB interference.

1. PREMISE (CLAIM) — WiFi interference is major corruption cause.
2. CLAIM — SymBee resists severe interference.
3. EVIDENCE — figure 20 description.
4. EVIDENCE — 0dB SINR scenario.
5. EVIDENCE (INTERPRETATION) — 60 > 42 → correct decoding.
6. CLAIM (CONSEQUENCE) — robust at 0dB.

**Shape:** PREMISE → CLAIM → EVIDENCE → EVIDENCE → EVIDENCE → CLAIM.
**Persuasive move:** robustness-by-construction.
**Flow notes:** "At first, we obverse … Figure 20 shows … The stable … but being still larger than 42 … Thus" — datum→threshold→verdict.

---

## §VIII.E · p.508-509 · ¶2

**Verbatim:**
> To further verify the robustness of SymBee under different interference level, we conduct a trace driven experiment based on the pure SymBee signal and WiFi 802.11g signal we collect on USRP B210. Mixed with different power level WiFi signal, Bit error rate of SymBee are represented by blue boxes in the figure 21. BER turns to be 19.5% when SINR drops to -10dB, meaning that the strength of WiFi interference is 2 times of SymBee signal. Even though the BER under strong interference is high, frame reception ratio could be increased via link layer coding. By applying Hamming (7,4) link layer coding on top of SymBee, BER of SymBee with coding decreases to almost half of SymBee without coding. Even though Hamming (7,4) coding can only correct one bit out of 7 bits, this experiment shows the big potential of SymBee in terms of robustness.

1. METHODOLOGY — trace-driven setup.
2. EVIDENCE — BER boxes.
3. EVIDENCE (INTERPRETATION) — 19.5% at −10dB SINR.
4. CONCESSION (CLAIM) — high BER mitigated by coding.
5. EVIDENCE — Hamming halves BER.
6. CONCESSION (CLAIM) — limited coding still shows potential.

**Shape:** METHODOLOGY → EVIDENCE → EVIDENCE → CONCESSION → EVIDENCE → CONCESSION.
**Persuasive move:** robustness-with-coding-supplement.
**Flow notes:** "Even though … By applying … Even though" — double concession framing of optimism.

---

## §VIII.F Impact of τ and preamble · p.509 · ¶1

**Verbatim:**
> We show the how τ affects detection of SymBee bits under SNR of -5dB in Figure 22(a). Higher τ indicates less SymBee bits would be missed while the false positive (F/P) ratio is getting higher. Therefore, we set τ to 10 where both false positive and false negative (F/N) are well balanced at a reasonably low values. Figure 22(b) depicts the bit error rate (BER) with and without preamble. Under the SNR of -5dB, the BER of SymBee without preamble achieves 27.4%, where it drops to 7.6% with preamble. The significant enhancement of SymBee via prepending preamble is clearly shown in this figure.

1. SCOPE — figure 22(a) about τ.
2. INTERPRETATION — tradeoff in τ.
3. METHODOLOGY (CLAIM) — τ=10 chosen.
4. SCOPE — figure 22(b) about preamble.
5. EVIDENCE — 27.4% → 7.6% with preamble.
6. CLAIM — significant enhancement.

**Shape:** SCOPE → INTERPRETATION → METHODOLOGY → SCOPE → EVIDENCE → CLAIM.
**Persuasive move:** parameter-justification.
**Flow notes:** "Higher τ … Therefore … Under SNR -5dB" — tradeoff to choice; second half mirrors with preamble.

---

## §VIII.G Mobility · p.509 · ¶1

**Verbatim:**
> The mobile scenario is also taken into account in our experiments. We evaluate SymBee on a track&field as shown in Figure 23(a). We deploy a WiFi receiver (a laptop with USRP B210) on a track&field where ZigBee senders (TelosB nodes) pass by the receiver at different speed: walking (3.4 mph), running (5.3 mph) and riding a bicycle (9.3 mph). The BER of different speed, tested by 3 ZigBee senders, are 7.15%, 8.48% and 8.9% respectively as shown in Figure 23(b). The blockage and vibration of bag, physical body and bicycle mainly cause the distortion of ZigBee signal and received error bits. Therefore the BER of this mobile experiment is higher than outdoor scenario in Figure 15.

1. SCOPE — mobility considered.
2. METHODOLOGY — track&field setup.
3. METHODOLOGY — speed conditions.
4. EVIDENCE — BER per speed.
5. CAUSE — vibration/blockage explanation.
6. CONSEQUENCE — higher BER than static outdoor.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → EVIDENCE → CAUSE → CONSEQUENCE.
**Persuasive move:** stress-test-with-mechanism.
**Flow notes:** "We evaluate … We deploy … The BER … The blockage … Therefore" — setup→data→cause→effect.

---

## §IX. Related Work · p.509 · ¶1

**Verbatim:**
> This work lies in the intersection of three areas: cross-technology communication, interference mitigation, and heterogeneous collaboration.

Single-sentence paragraph: SCOPE.
**Shape:** SCOPE.
**Persuasive move:** related-work-roadmap.
**Flow notes:** taxonomy preview.

---

## §IX · p.509 · ¶2 (Cross-technology communication)

**Verbatim:**
> **Cross-technology communication.** CTC was introduced to enable direction communication without the need for the gateway [4]. Doing so not only is beneficial in terms of cost savings, but also is advantageous in network planning as it largely simplifies network structure complexity, and enhances spectrum efficiency by removing the traffic running into and out of the gateway [28]. Most of the CTC work take the packet-level approach where a packet serves as the unit of modulation (similar to pulse in digital communication systems): Esense [7] and HoWiEs [38] modulates length of a single or sequence of packets, while FreeBee [19] modulates via beacon timings. B2W2 [9] delivers messages from Bluetooth to WiFi by controlling the power of Bluetooth packets. SymBee takes a unique approach of symbol-level CTC for throughput breakthrough. GapSense [37] introduces a fine-grained physical layer design, where it requires a special hardware. The latest work of WEBee [20] and BlueBee [16], with the physical layer approach and high-throughput, are most similar, but are infeasible for ZigBee to WiFi communication.

1. DEFINITION (EVIDENCE) — CTC origin.
2. CLAIM (EVIDENCE) — benefits of CTC.
3. EVIDENCE (DEFINITION) — packet-level taxonomy with examples.
4. EXAMPLE (EVIDENCE) — B2W2.
5. CLAIM (CONTRAST) — SymBee = symbol-level.
6. CONCESSION (EVIDENCE) — GapSense needs special HW.
7. CONCESSION (CONTRAST) — WEBee/BlueBee similar but infeasible.

**Shape:** DEFINITION → CLAIM → EVIDENCE → EXAMPLE → CLAIM → CONCESSION → CONCESSION.
**Persuasive move:** position-against-prior-art.
**Flow notes:** taxonomy then "SymBee takes a unique approach" pivot then differentiations.

---

## §IX · p.509 · ¶3 (Interference Mitigation)

**Verbatim:**
> **Interference Mitigation.** There had been much effort in the networking and wireless community to resolve interference under a wide range of scenarios and systems [11][23], where they can be roughly divided into PHY and link layer approaches. The former includes MIMO [25] [35] and OFDM [24] techniques. These techniques can be further categorized to interference cancellation [13], [27], [40] and interference alignment [5]. Another stream of work in interference mitigation is link layer designs [6] [29], such as enhancing packet robustness and recovering from errors [33]. Our design is fundamentally different from these work, where SymBee aims at achieving coordination via direction communication.

1. PREMISE (EVIDENCE) — broad prior effort.
2. DEFINITION — PHY techniques.
3. DEFINITION (EVIDENCE) — subcategories.
4. DEFINITION (EVIDENCE) — link layer techniques.
5. CLAIM (CONTRAST) — SymBee is fundamentally different.

**Shape:** PREMISE → DEFINITION → DEFINITION → DEFINITION → CLAIM.
**Persuasive move:** differentiate-by-aim.
**Flow notes:** taxonomy → "Our design is fundamentally different" closer.

---

## §IX · p.509 · ¶4 (Heterogeneous collaboration)

**Verbatim:**
> **Heterogeneous collaboration.** Connection and interoperation between heterogeneous wireless systems is traditionally established via gateways equipped with multiple radio interfaces to perform translation tasks [10] [18] [22]. Based on this structure, many systems that enables heterogeneous collaborations to explore the synergistic effect have been introduced [17], [26], [32], [39]. In WiZi-Cloud [17], for example, offers energy efficient Internet connectivity to mobile devices by utilizing ZigBee device with the access to the Internet via gateway.

1. PREMISE (DEFINITION) — gateway tradition.
2. EVIDENCE — systems built on this.
3. EXAMPLE — WiZi-Cloud.

**Shape:** PREMISE → EVIDENCE → EXAMPLE.
**Persuasive move:** alternative-tradition-survey.
**Flow notes:** "traditionally … Based on this structure … In WiZi-Cloud, for example" — tradition then exemplar (no explicit SymBee positioning; implicit by section 2 contrast).

---

## §X. Conclusion · p.509 · ¶1

**Verbatim:**
> We propose SymBee, a cross-technology communication framework that aims to bridge capacity and compatibility by customizing ZigBee packets. SymBee's encoding is as simple as putting specific byte patterns in the ZigBee packet payload, maximizing its applicability. This generates pattern at the PHY layer that can easily be detected at the WiFi idle listening. Theoretical analysis and extensive testbed experiments on TelosB nodes and USRP B210 reveal that SymBee is a reliable and efficient under various practical settings with the throughput up to 31.25Kbps, 145.4× of the state-of-the-art.

1. CLAIM — restates contribution.
2. METHODOLOGY (CLAIM) — encoding simplicity.
3. CAUSE — generates detectable PHY pattern.
4. EVIDENCE (CLAIM) — empirical + analytical results.

**Shape:** CLAIM → METHODOLOGY → CAUSE → EVIDENCE.
**Persuasive move:** wrap-up-with-headline-numbers.
**Flow notes:** "We propose … SymBee's encoding … This generates … Theoretical analysis and extensive testbed experiments … reveal" — recap to evidence.

---

## Endnotes

- **Total paragraphs annotated:** 50 (excluding pure equation blocks, pseudocode, figure/table captions, references, acknowledgements, and the contribution bullet list — which is reproduced verbatim but not sentence-tagged).
- **Total sentences annotated:** ~234.
- **Three most frequent paragraph shapes:**
  1. METHODOLOGY-led chains (e.g., METHODOLOGY → METHODOLOGY → … → CLAIM) — dominant in §III.B, §IV.C, §V, §VI.B, §VIII setup paragraphs.
  2. EVIDENCE/INTERPRETATION cause-explanation chains (e.g., EVIDENCE → CAUSE → CONSEQUENCE / EVIDENCE → INTERPRETATION → CLAIM) — common in §VIII evaluation paragraphs.
  3. CLAIM → METHODOLOGY → CONSEQUENCE — recurring "design-then-payoff" template in §I, §III.A, §VI.A.
- Single-sentence SCOPE paragraphs appear five times (§II ¶1, §III ¶1, §III.B ¶1, §IV ¶1, §VI ¶1, §IX ¶1) as section bridges.
- CONCESSION-driven pivots ("Although effective", "Despite", "Even though") cluster in §I ¶2, §II.B ¶2/¶3, §VIII.E ¶2 — the persuasive locations where SymBee differentiates from baselines.
