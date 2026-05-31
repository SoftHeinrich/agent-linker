# Exhaustive Argumentation Annotation — Wang et al. (2022), "X-Disco: Cross-technology Neighbor Discovery"

Source: `/mnt/hostshare/ardoco-home/mono/writing/gen/refs/wang-2022-xdisco.pdf`

## Table of Contents

- §I. Introduction
- §II. Motivation
  - §II.A The Need for Cross-technology Neighbor Discovery
  - §II.B Opportunities
    - §II.B.1 Cross-technology Communication
    - §II.B.2 Fine-grained PHY-layer Information at WiFi
- §III. Overview of X-Disco and Background
  - §III.A X-Disco in a Nutshell
  - §III.B How ZigBee signal is interpreted at WiFi
    - §III.B.1 ZigBee Transmitter
    - §III.B.2 WiFi Spectral Scan
- §IV. Design of X-Disco
  - §IV.A ZigBee Symbol Extraction
    - §IV.A.1 ZigBee Cross-detection
    - §IV.A.2 Fine-grained Synchronization
  - §IV.B ZigBee Coordinator Detection
  - §IV.C Neighbor Information Acquisition
- §V. Advanced Features of X-Disco
  - §V.A ZigBee Neighbor Validation
  - §V.B Interruption Mitigation
- §VI. Evaluation
  - §VI.A X-Disco Performance
  - §VI.B Impact of WiFi traffic
  - §VI.C Multi-channel Discovery
  - §VI.D ZigBee Neighbor Validation
  - §VI.E Mobile
- §VII. Related Work
- §VIII. Conclusion
- §Endnotes

---

## §I. Introduction · p.163 · ¶1

**Quote:** "We have witnessed the explosive growth of IoT devices , including WiFi, ZigBee, and Bluetooth, along with various applications supported by heterogeneous wireless technologies in the past decades. As half billion ZigBee chips sold [1] and over three billion WiFi devices shipped annually [2], WiFi and ZigBee coexist densely on the 2.4 GHz ISM spectrum and physical places such as smart homes and factories, raising critical coexistence issues such as cross-technology interference (CTI) [3], [4]. To avoid such interference, cross-technology coordination [5], [6] and cooperation [7] are proposed for better accommodating WiFi and ZigBee devices. Nevertheless, the coordination across multiple wireless technologies inevitably requires wireless devices to maintain the cross-technology neighbor information. Therefore, this paper focuses on enabling a universal neighbor discovery mechanism for a WiFi device to detect the ambient ZigBee neighbors, namely cross-technology neighbor discovery."

1. PREMISE — sets up IoT growth backdrop.
2. EVIDENCE (PREMISE) — quantitative shipment figures grounding coexistence claim.
3. BRIDGE — cites prior coordination/cooperation proposals.
4. PREMISE (CAUSE) — coordination presupposes neighbor information.
5. CLAIM — paper scope statement.

**Shape:** PREMISE → EVIDENCE → BRIDGE → PREMISE → CLAIM
**Persuasive move:** problem-staircase to scope statement.
**Sentence flow notes:** "As" introduces evidence; "To avoid such interference" links problem to prior solutions; "Nevertheless" pivots to a precondition not met; "Therefore" delivers the paper's focus.

---

## §I. Introduction · p.163 · ¶2

**Quote:** "As an essential step of establishing and maintaining a network, neighbor discovery is inherently supported in the homogeneous ZigBee [8] and WiFi networks [9]. However, discovering cross-technology neighbors is non-trivial due to two challenges: (i), WiFi and ZigBee devices cannot directly communicate with each other due to the incompatible PHY layers. (ii), developing a universal neighbor discovery mechanism across multiple wireless protocols might require significant modification on the billions of existing IoT devices [10], resulting in impractical use cases and expensive costs at scale."

1. PREMISE — neighbor discovery solved within homogeneous networks.
2. CONTRAST (CLAIM) — cross-tech case is hard; previews two challenges.
3. PREMISE — challenge (i): PHY incompatibility.
4. PREMISE (CONSEQUENCE) — challenge (ii): modification costs are prohibitive.

**Shape:** PREMISE → CONTRAST → PREMISE → PREMISE
**Persuasive move:** gap-in-prior-work via two-challenge framing.
**Sentence flow notes:** "However" inverts the prior support claim; the enumerators "(i)" and "(ii)" itemize the two challenges; "resulting in" adds consequence to challenge (ii).

---

## §I. Introduction · p.163 · ¶3

**Quote:** "This paper proposes X-Disco, the first software-only cross-technology neighbor discovery mechanism, to enable a WiFi device to discover the ambient ZigBee neighbors without any modification to the ZigBee devices. X-Disco achieves this by leveraging the Device and Service Discovery mechanism [8], where the ZigBee neighbor information, such as addresses, is shared per neighbor information request sent to the ZigBee coordinator 1. At a high level, after the X-Disco device (commodity WiFi) transmits a neighbor information request via the recent proposed cross-technology communication (CTC) [11], the ZigBee coordinator reacts to that request as if that request is from a ZigBee device and replies with a message, containing all associated ZigBee devices' addresses, which are further decoded and obtained by the X-Disco device. Decoding the replied ZigBee message is uniquely inspired by our newly discovered insight: the ZigBee signal is recognizable and decodable by exploiting the special patterns extracted from FFT magnitude (accessible by driver) collected at WiFi Spectral Scan, ensuring compatibility to commodity WiFi. As fetching neighbor information from ZigBee coordinators strictly follows ZigBee Device and Service Discovery mechanism, X-Disco maintains transparency to ZigBee network. In addition, working in active mode, the ZigBee coordinators are designed to be responsive to the neighbor information request, thereby incurring minimum overhead, as no duty cycle is involved."

1. CLAIM — proposes X-Disco as the first software-only CTND.
2. METHODOLOGY — leveraging Device and Service Discovery.
3. METHODOLOGY (EXAMPLE) — high-level four-step interaction sketch.
4. CLAIM (INTERPRETATION) — novel insight on FFT-magnitude decoding.
5. CONSEQUENCE — protocol-fidelity yields transparency.
6. CONSEQUENCE — active-mode coordinators yield minimum overhead.

**Shape:** CLAIM → METHODOLOGY → METHODOLOGY → CLAIM → CONSEQUENCE → CONSEQUENCE
**Persuasive move:** design-derived-from-phenomenon plus list of derived virtues.
**Sentence flow notes:** "At a high level" signals an explanatory zoom-in; "uniquely inspired by our newly discovered insight" stakes novelty; "As" and "In addition" chain two consequences (transparency, low overhead) from the mechanism.

---

## §I. Introduction · p.163–164 · ¶4

**Quote:** "X-Disco is built with three new technical highlights: (i) ZigBee Symbol Extraction, (ii) ZigBee Coordinator Detection, and (iii) Neighbor Information Acquisition, where the compatibility with the hardware and protocols is the key. ZigBee Symbol Extraction ensures that ZigBee messages are reliably decoded at commodity WiFi, under the challenge where the phase information is totally discarded, to overcome the PHY-layer incompatibility issue. ZigBee Coordinator Detection detects ambient ZigBee coordinators, from which Neighbor Information Acquisition obtains the ZigBee neighbor information by exchanging ZigBee compatible messages, yielding 100% transparency to the ZigBee network. To the best of our knowledge, X-Disco is the first design to discover cross-technology neighbors using commodity WiFi devices. X-Disco effectively utilizes the widely deployed WiFi infrastructures to detect ambient ZigBee devices, demonstrating the pervasive application in practice at zero cost. To summarize, the contribution of this paper is three-fold:"

1. CLAIM (METHODOLOGY) — three technical highlights with compatibility as theme.
2. DEFINITION — role of ZigBee Symbol Extraction.
3. DEFINITION — roles of Coordinator Detection and Neighbor Information Acquisition.
4. CLAIM — novelty claim ("first design").
5. CLAIM (CONSEQUENCE) — deployment value at zero cost.
6. BRIDGE — into the bulleted contributions.

**Shape:** CLAIM → DEFINITION → DEFINITION → CLAIM → CLAIM → BRIDGE
**Persuasive move:** component decomposition culminating in novelty + deployment claim.
**Sentence flow notes:** Enumerators (i)–(iii) set up; following sentences define each piece; "To the best of our knowledge" prefaces novelty assertion; "To summarize" transitions to the bullets.

---

## §I. Introduction · p.164 · ¶5 (Contributions bullet list — treated as one block)

**Quote:** "• We design X-Disco, the first cross-technology neighbor discovery mechanism for a commodity WiFi device to detect ambient ZigBee neighbors. The full compatibility with commodity WiFi and ZigBee hardware and protocol ensures X-Disco's wide and practical deployment. • X-Disco introduces three main techniques: ZigBee Symbol Extraction, ZigBee Coordinator Detection, and Neighbor Information Acquisition, which allow a commodity WiFi device to decode the responded ZigBee messages, detect the ambient ZigBee coordinators, and acquire the ZigBee neighbor information respectively. In addition, we propose two enhancements (ZigBee Neighbor Validation and Interruption Mitigation) to improve the reliability of X-Disco. • X-Disco is evaluated on the commodity WiFi (TP-link WDR 4300 router), software-defined radio (USRP B210), and commodity ZigBee (Telosb motes). The results demonstrate that X-Disco successfully detects nine ZigBee neighbors within 70ms in the office."

1. CLAIM — design contribution + novelty.
2. CONSEQUENCE — compatibility implies deployability.
3. CLAIM (METHODOLOGY) — three core techniques.
4. CLAIM — two reliability enhancements.
5. METHODOLOGY — implementation platforms.
6. EVIDENCE — "nine ZigBee neighbors within 70ms" result.

**Shape:** CLAIM → CONSEQUENCE → CLAIM → CLAIM → METHODOLOGY → EVIDENCE
**Persuasive move:** standard tripartite contribution list anchored by headline metric.
**Sentence flow notes:** Bullets discretize claims; "In addition" appends the enhancements; "The results demonstrate" tags the empirical headline.

---

## §II.A The Need for Cross-technology Neighbor Discovery · p.164 · ¶1

**Quote:** "Numerous IoT devices with different wireless technologies densely coexist on the ISM band and the physical world to support various applications. For instance, 53 million Amazon Echo devices [12], equipped with WiFi and ZigBee transceivers, Philips Hue Smart Bulb, and Samsung SmartThings, were shipped in 2020 to support smart homes. ZigBee-based route management systems and WiFi modules are installed for smart factories [13]. In such a dense WiFi and ZigBee coexisting environment, severe ZigBee transmission loss (≥50% ZigBee packets [4]), caused by the cross-technology interference from WiFi, degrades spectral efficiency and results in failures of ZigBee applications. Nevertheless, all these problems could be avoided via heterogeneous coordination [7], [5], [6], [14], if heterogeneous wireless devices are aware of each other – i.e., the neighbor information is dynamically maintained and shared across multiple wireless technologies. Therefore, we present X-Disco for commodity WiFi devices to detect the ZigBee neighbors. To minimize the deployment cost, X-Disco is designed to be a software-only approach and 100% transparency to the ZigBee network, which are inspired by the following two opportunities."

1. PREMISE — coexistence pervades ISM band.
2. EXAMPLE (EVIDENCE) — Amazon Echo, Hue, SmartThings shipment exemplars.
3. EXAMPLE — smart factory deployments.
4. CAUSE (EVIDENCE) — quantified ZigBee loss attributed to CTI.
5. CONTRAST (CLAIM) — heterogeneous coordination conditional on neighbor awareness.
6. CLAIM — therefore we propose X-Disco.
7. CLAIM (METHODOLOGY) — design constraints: software-only and transparent.

**Shape:** PREMISE → EXAMPLE → EXAMPLE → CAUSE → CONTRAST → CLAIM → CLAIM
**Persuasive move:** market-evidence → harm → remedy-precondition → proposal.
**Sentence flow notes:** "For instance" and follow-up provide concrete examples; "Nevertheless ... if" introduces the corrective conditional; "Therefore" delivers the proposal; "To minimize the deployment cost" frames design constraints.

---

## §II.B.1 Cross-technology Communication · p.164 · ¶1

**Quote:** "The recent advanced Cross-technology Communication designs [11], [15] enable a commodity WiFi device to send messages to a commodity ZigBee device directly. Specifically, a WiFi device emulates the target ZigBee message via carefully customizing the payload of a WiFi packet such that the corresponding transmitted WiFi signal is recognized as a legitimate ZigBee packet with the intended message by the commodity ZigBee devices. In this paper, X-Disco incorporates CTC to trigger the response from the ambient ZigBee coordinators. Meanwhile, decoding the responded ZigBee messages at commodity WiFi is inspired by the next opportunity."

1. PREMISE — CTC enables WiFi→ZigBee direct messaging.
2. DEFINITION (METHODOLOGY) — explains payload-emulation mechanism.
3. CLAIM — X-Disco uses CTC to trigger responses.
4. BRIDGE — transitions to next opportunity.

**Shape:** PREMISE → DEFINITION → CLAIM → BRIDGE
**Persuasive move:** lean on existing primitive + foreshadow.
**Sentence flow notes:** "Specifically" zooms in; "In this paper" applies CTC to X-Disco; "Meanwhile" segues to the second opportunity.

---

## §II.B.2 Fine-grained PHY-layer Information at WiFi · p.164 · ¶1

**Quote:** "The commodity WiFi device exposes fine-grained PHY-layer information such as Channel State Information (CSI) [16] and Fast Fourier Transformation (FFT) magnitude of the received signal [17], [18] to the WiFi driver. As a proprietary mode supported by many WiFi drivers and commodity WiFi devices 2, Spectral Scan [18], [19] continuously collects the FFT magnitude of the received signal, regardless of the signal type. That is, upon the arrival of a ZigBee signal, WiFi Spectral Scan converts that ZigBee signal into a series of FFT magnitudes, demonstrating special patterns of ZigBee signal. However, it is still quite challenging to decode the ZigBee message because ZigBee modulates information in the phase, whereas WiFi Spectral Scan only provides magnitude information without phase. In next Section, we demonstrate an overview of X-Disco, followed by insights for overcoming this challenge."

1. PREMISE — WiFi exposes PHY-layer info (CSI, FFT magnitude).
2. DEFINITION (PREMISE) — Spectral Scan continuously yields FFT magnitudes.
3. INTERPRETATION — ZigBee signals appear as recognizable FFT patterns.
4. CONTRAST (CONCESSION) — but ZigBee modulates in phase, magnitude alone is insufficient.
5. BRIDGE — preview of next section.

**Shape:** PREMISE → DEFINITION → INTERPRETATION → CONTRAST → BRIDGE
**Persuasive move:** opportunity-meets-obstacle to motivate next section.
**Sentence flow notes:** "That is" elaborates; "However" pivots to the phase problem; "In next Section" closes with forward pointer.

---

## §III.A X-Disco in a Nutshell · p.164–165 · ¶1

**Quote:** "X-Disco is a two-step approach, containing four messages (M1 to M4) exchanged between an X-Disco device (commodity WiFi) and ambient ZigBee coordinators. As illustrated in Figure 1, in Step (a), the X-Disco device transmits an emulated ZigBee broadcast packet in message M1, triggering the ZigBee coordinator to rebroadcast in the message M2, from which the X-Disco device obtains the essential ZigBee network information for customizing a neighbor information request message in the next step. In Step (b), the X-Disco device requests the ZigBee neighbor information in the carefully customized message M3, which triggers the ZigBee coordinator to attach the associated ZigBee devices' addresses in the responded message M4. By leveraging the ZigBee Device and Service Discovery mechanism, X-Disco detects ZigBee neighbors via fetching the neighbor information from the ZigBee coordinator, with only four messages exchanged. As ZigBee coordinators are always in active mode, the exchanged messages are naturally immune to the duty-cycle related problems, thereby achieving the minimum overhead."

1. DEFINITION — two-step, four-message architecture.
2. METHODOLOGY — Step (a) details (M1, M2).
3. METHODOLOGY — Step (b) details (M3, M4).
4. CLAIM (CONSEQUENCE) — efficiency from protocol reuse (four messages).
5. CONSEQUENCE — active-mode coordinators imply duty-cycle immunity and minimum overhead.

**Shape:** DEFINITION → METHODOLOGY → METHODOLOGY → CLAIM → CONSEQUENCE
**Persuasive move:** mechanism-with-derived-virtues.
**Sentence flow notes:** "As illustrated in Figure 1" anchors to diagram; "In Step (b)" sequels Step (a); "By leveraging" wraps derivation; "As ZigBee coordinators are always in active mode" supports overhead claim.

---

## §III.A · p.165 · ¶2

**Quote:** "As the foundation of X-Disco, decoding the replied message M2 and M4 at commodity WiFi is very challenging because WiFi Spectral Scan does not provide any phase information, whereas ZigBee modulation relies on phase. To address this issue, we propose ZigBee Symbol Extraction, which decodes the ZigBee signal only using the FFT magnitude without phase information. To understand ZigBee Symbol Extraction, we demonstrate how the ZigBee signal is constructed at the ZigBee PHY layer and how the ZigBee signal is interpreted at WiFi Spectral Scan in next Section."

1. QUESTION (CAUSE) — restates phase-vs-magnitude challenge as foundational obstacle.
2. CLAIM — proposes ZigBee Symbol Extraction.
3. BRIDGE — outlines the next subsection's expository plan.

**Shape:** QUESTION → CLAIM → BRIDGE
**Persuasive move:** obstacle-and-counter-proposal with roadmap.
**Sentence flow notes:** "As the foundation" elevates stakes; "To address this issue" introduces remedy; "To understand" forecasts background section.

---

## §III.B.1 ZigBee Transmitter · p.165 · ¶1

**Quote:** "The transmission of a ZigBee packet starts with assembling every four bits into one symbol ('0'-'F'), which is the basic unit to carry information in ZigBee [21]. As illustrated in Figure 2, the ZigBee PHY Layer first converts the input ZigBee symbol into a unique and pseudo-random 32-bit chip sequence [21] in Step (i), i.e., Direct Sequence Spread Spectrum (DSSS). Then, the chips '1' and '0' are shaped into positive and negative 1µs half-sine pulses in Step (ii) and (iii), where the quadrature signal, corresponding to the chips on the even indices, is delayed by half pulse duration 0.5µs, compared to the in-phase signal (the chips on the odd indices). As the in-phase (I) and quadrature signal (Q) are merged, this half-pulse delay leads to complex sinusoidal waves with constant magnitude, while expressing 0/1 chips in the phase, namely Offset Quadrature Phase Shift Keying (OQPSK). Finally, in Step (iv) the digital-to-analog converter (DAC) translates the discrete I/Q signal into a continuous analog baseband signal, which is then shifted by the mixer to the ZigBee's carrier frequency (passband) and transmitted into the air in Step (v)."

1. DEFINITION — ZigBee symbol structure.
2. METHODOLOGY — DSSS chip mapping.
3. METHODOLOGY — pulse shaping with 0.5µs Q delay.
4. INTERPRETATION (DEFINITION) — OQPSK encodes chips in phase, constant magnitude.
5. METHODOLOGY — DAC and mixer to passband.

**Shape:** DEFINITION → METHODOLOGY → METHODOLOGY → INTERPRETATION → METHODOLOGY
**Persuasive move:** background-walk-through preparing the magnitude-vs-phase argument.
**Sentence flow notes:** Step enumerators (i)–(v) carry the flow; "Then," "As ... merged," "Finally" sequence the PHY stages.

---

## §III.B.1 · p.165 · ¶2

**Quote:** "To demonstrate the intuition of the ZigBee signal, we plot the I/Q waveform of ZigBee symbol '0' in Figure 3. The ZigBee PHY Layer converts one ZigBee symbol into 32 chips, where the In-phase and Quadrature take 16 chips each, yielding the ZigBee signal of 16µs with a constant magnitude. Since the duration of each chip is 1µs, the complex ZigBee signal consists of 2MHz positive or negative half-sine waves, resulting in 2MHz bandwidth. Next, we show the architecture of WiFi Spectral Scan, which provides the fundamental insight for decoding a ZigBee message at commodity WiFi."

1. METHODOLOGY — illustrative plot.
2. PREMISE — 32-chip / 16µs / constant magnitude facts.
3. CAUSE (PREMISE) — chip duration implies 2MHz bandwidth.
4. BRIDGE — pointer to Spectral Scan architecture.

**Shape:** METHODOLOGY → PREMISE → CAUSE → BRIDGE
**Persuasive move:** quantify-the-signal-then-pivot.
**Sentence flow notes:** "Since the duration" introduces causal derivation; "Next" hands off to Spectral Scan subsection.

---

## §III.B.2 WiFi Spectral Scan · p.165 · ¶1

**Quote:** "As Figure 4 depicts, in Step (i), the mixer shifts the passband signal to the baseband, which is further sampled at 20MHz by the analog-to-digital converter (ADC) in Step (ii). Then in Step (iii), FFT Windowing is performed every 4µs 3 to cut the continuously received samples into fragments of 80 samples, where the last 16 samples (e.g., grayed parts in Figure 3) are discarded as cyclic prefix (CP), designed for avoiding inter symbol interference. Finally, in Step (iv), the rest 64 samples (3.2µs non-grayed segments in Figure 3) are fed into FFT calculation, which outputs the corresponding magnitude while the phase information is left out. Since this process does not require the received signal to be WiFi, an arbitrary signal (e.g., ZigBee) will be reflected in FFT magnitude if Spectral Scan mode is on."

1. METHODOLOGY — mixer + ADC stage.
2. METHODOLOGY — FFT windowing with CP discard.
3. METHODOLOGY (INTERPRETATION) — 64 samples to FFT, phase dropped.
4. CLAIM — Spectral Scan accepts any signal including ZigBee.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → CLAIM
**Persuasive move:** mechanism-derives-applicability.
**Sentence flow notes:** "Then" / "Finally" sequence; "Since this process does not require" justifies general-signal claim.

---

## §III.B.2 · p.165 · ¶2

**Quote:** "When the signal of ZigBee symbol '0', transmitted on ZigBee channel 13 (2.415GHz), arrives at WiFi Spectral Scan working on WiFi channel 1 (2.412GHz), the mixer shifts the Zigbee signal to the center frequency of the WiFi channel, a 3MHz frequency offset is introduced in the baseband signal, yielding the overlap with WiFi subcarriers 38 to 45. Then, as depicted in Figure 3, FFT Windowing cuts the baseband signal into four non-grayed segments. As illustrated in Figure 5, the results depict an interesting insight — four FFT magnitudes demonstrate different patterns on the overlapped subcarriers. This insight is quite counter-intuitive because the four non-grayed segments have the same and constant magnitude, as explained in Section III-B1, whereas the corresponding FFT magnitudes are different."

1. METHODOLOGY (PREMISE) — channel offset and overlap.
2. METHODOLOGY — windowing yields four segments.
3. INTERPRETATION (CLAIM) — four distinct FFT magnitudes — the key insight.
4. CONTRAST (INTERPRETATION) — counter-intuitive given constant time-domain magnitude.

**Shape:** METHODOLOGY → METHODOLOGY → INTERPRETATION → CONTRAST
**Persuasive move:** surprise-finding-as-foundation.
**Sentence flow notes:** "Then" sequels; "As illustrated" introduces empirical observation; "This insight is quite counter-intuitive because" highlights the paradox.

---

## §III.B.2 · p.166 · ¶3 ("How to understand this insight?")

**Quote:** "How to understand this insight? We use the 2nd and 3rd segments of ZigBee symbol '0' as an example to explain this insight. Due to the randomness in the Symbol-to-Chip Mapping, the 0/1 chips in a ZigBee symbols are random, as well as the corresponding positive and negative sinusoidal signals. As compared in Figure 6, the yellow marked sinusoidal waves are the flipped in the 2nd and 3rd segments. Such flip only changes the phase of the signal while the magnitude of the time domain signal is still the same. When we feed those two segments into the FFT calculation, the results reflect two segments on the frequency domain, not the time domain. According to the property of FFT, the two segments induce the same FFT magnitude, if and only if one segment is the phase-shifted version of the other. Since not all 1µs sinusoidal waves are flipped, the entire 3.2µs 2nd segment is not the phase-shifted version of the entire 3rd segment. Therefore, the FFT magnitudes of those two segments are different. Not only the FFT magnitudes of different segments in one ZigBee symbol are unique, the randomness in the Symbol-to-Chip Mapping makes different ZigBee symbols induce a unique and specific pattern in the corresponding four FFT magnitudes, indicating the feasibility of decoding the ZigBee symbol without the phase information on commodity WiFi."

1. QUESTION — rhetorical setup.
2. METHODOLOGY — choose 2nd/3rd segments as exemplar.
3. PREMISE — random chips give random sign signals.
4. EVIDENCE — Figure 6 shows flips.
5. INTERPRETATION — flip changes phase, not magnitude.
6. PREMISE — FFT acts in frequency domain.
7. PREMISE (DEFINITION) — FFT magnitude equality iff phase-shift relation.
8. CAUSE — partial flips break phase-shift relation.
9. CONSEQUENCE — therefore FFT magnitudes differ.
10. CLAIM — randomness yields unique per-symbol pattern, enabling phase-free decoding.

**Shape:** QUESTION → METHODOLOGY → PREMISE → EVIDENCE → INTERPRETATION → PREMISE → PREMISE → CAUSE → CONSEQUENCE → CLAIM
**Persuasive move:** Socratic derivation of feasibility from FFT properties.
**Sentence flow notes:** Opens with a rhetorical question; "Due to," "Such flip only," "According to the property of FFT," "Since," "Therefore" stage a deductive chain ending in feasibility claim.

---

## §IV. Design of X-Disco · p.166 · ¶0 (lead-in)

**Quote:** "Based on the insight introduced in the last Section, the detailed designs of X-Disco are demonstrated here."

1. BRIDGE — chapter transition.

**Shape:** BRIDGE
**Persuasive move:** section pivot.
**Sentence flow notes:** Single-sentence handoff anchored by "Based on the insight."

---

## §IV.A ZigBee Symbol Extraction · p.166 · ¶1

**Quote:** "Extracting the ZigBee symbol information on commodity WiFi is realized by exploring the uniqueness of the four FFT magnitudes induced by different ZigBee symbols. To simplify the notation, we define the four FFT magnitudes calculated from one ZigBee symbol to be an FFT group. At a high level, decoding the ZigBee symbol is achieved by comparing the received FFT group with the template FFT groups, which are calculated from the transmitted signals of 16 ZigBee symbols. The ZigBee symbol of the received FFT group corresponds to the template FFT group with the highest similarity."

1. CLAIM (METHODOLOGY) — extraction relies on FFT-magnitude uniqueness.
2. DEFINITION — "FFT group".
3. METHODOLOGY — template-matching high-level scheme.
4. DEFINITION (METHODOLOGY) — decision rule (max similarity).

**Shape:** CLAIM → DEFINITION → METHODOLOGY → DEFINITION
**Persuasive move:** algorithm-overview-before-formalism.
**Sentence flow notes:** "To simplify the notation" introduces definition; "At a high level" foreshadows formalization.

---

## §IV.A · p.166 · ¶2 (formalization paragraph)

**Quote:** "Specifically, we denote the template FFT group of the ZigBee symbol i by Zi, where i ∈ {0, ··· , F}. Then the FFT group is specifically defined as Zi ≜ [Zi,1, Zi,2, Zi,3, Zi,4], where Zi,k represents the FFT magnitude of the k-th segment in Symbol i. With only eight WiFi subcarriers overlapped with one ZigBee channel, we define Zi,k ≜ [Zi,k[L], Zi,k[L + 1], ..., Zi,k[L + 7]], where Zi,k[L] is the magnitude of the L-th subcarrier in Zi,k and L is the index of the left most subcarrier overlapped with the ZigBee channel. Based on that, we define the similarity between the FFT groups induced by ZigBee symbol i and j as follows:"

1. DEFINITION — template group notation Zi.
2. DEFINITION — Zi as 4-tuple.
3. DEFINITION — Zi,k indexing eight overlapped subcarriers.
4. BRIDGE — leading into similarity definition.

**Shape:** DEFINITION → DEFINITION → DEFINITION → BRIDGE
**Persuasive move:** notation lockdown for formal claim.
**Sentence flow notes:** "Specifically," "Then," "With only eight WiFi subcarriers," "Based on that" sequence the build-up.

---

## §IV.A · p.166 · ¶3 (after Definition 1)

**Quote:** "where corr calculates the correlation coefficient between two vectors. With this definition, if any two FFT magnitudes in two ZigBee symbols are different, the similarity drops significantly."

1. DEFINITION — corr operator.
2. INTERPRETATION (CONSEQUENCE) — discriminative property of similarity.

**Shape:** DEFINITION → INTERPRETATION
**Persuasive move:** justify discriminability of formal metric.
**Sentence flow notes:** "With this definition" links back to Eq.(1).

---

## §IV.A · p.166 · ¶4

**Quote:** "To demonstrate the insight of decoding ZigBee symbols without phase information, we plot the similarity between the FFT groups of arbitrary two ZigBee symbols in Figure 7. Apparently, the similarity between the template FFT groups of different ZigBee symbols is quite small, indicating that we could decode ZigBee symbols by comparing the similarity, which is purely calculated from FFT magnitude without any phase information on commodity WiFi."

1. METHODOLOGY — pairwise similarity visualization.
2. EVIDENCE (CLAIM) — small off-diagonal similarity confirms decoding feasibility.

**Shape:** METHODOLOGY → EVIDENCE
**Persuasive move:** empirical confirmation of theoretical claim.
**Sentence flow notes:** "Apparently" signals visual interpretation.

---

## §IV.A · p.166 · ¶5

**Quote:** "In X-Disco, extracting ZigBee symbols at commodity WiFi starts with forming the four received FFT magnitudes Yn,Yn+1,Yn+2, and Yn+3, into an FFT group Yn ≜ [Yn,Yn+1,Yn+2,Yn+3]. If the ZigBee symbol i's template FFT group has the highest similarity, the decoding result is the ZigBee symbol 'i'. Therefore, extracting the ZigBee symbol from the received FFT group Yn is achieved by checking which template FFT group has the highest similarity:"

1. METHODOLOGY — group formation Yn.
2. DEFINITION (METHODOLOGY) — decoding rule.
3. BRIDGE — leading into Eq.(2).

**Shape:** METHODOLOGY → DEFINITION → BRIDGE
**Persuasive move:** crystallization into argmax rule.
**Sentence flow notes:** "If ... the decoding result is" then "Therefore" formalizes it.

---

## §IV.A · p.166–167 · ¶6 (after Eq.(2))

**Quote:** "Nevertheless, directly applying Equation 2 to decode a ZigBee packet faces two practical issues. As illustrated in Figure 8, the X-Disco device continuously collects the FFT magnitude Y1 to Y5 from Spectral Scan. Since the X-Disco device is not synchronized to the ZigBee packets, two issues occur: (i), the unknown arrival time of a ZigBee packet. For instance, in Figure 8, the first ZigBee symbol in a ZigBee packet, captured by the FFT magnitude Y2 to Y5, should be detected by X-Disco before extracting the ZigBee symbols. (ii), the unknown delay in the FFT window. The FFT magnitude Y2 is calculated from the 1.15 µs noise concatenated by the first 2.05µs ZigBee signal due to the delay. Directly comparing such received FFT group with our template FFT groups, which are calculated in the synchronized case, would degrade the similarity and the accuracy of ZigBee Symbol Extraction. To resolve these two issues, we present two new designs: ZigBee Cross-detection and Fine-grained Synchronization."

1. CONCESSION (CONTRAST) — Eq.(2) has practical issues.
2. METHODOLOGY — illustrate continuous collection.
3. CAUSE — desync introduces two issues.
4. EXAMPLE — issue (i): unknown arrival time.
5. PREMISE — issue (ii): unknown FFT-window delay.
6. CONSEQUENCE — naïve comparison degrades accuracy.
7. CLAIM (BRIDGE) — two designs proposed.

**Shape:** CONCESSION → METHODOLOGY → CAUSE → EXAMPLE → PREMISE → CONSEQUENCE → CLAIM
**Persuasive move:** problem-then-twin-remedies.
**Sentence flow notes:** "Nevertheless" yields the concession; enumerated "(i)" and "(ii)" name the issues; "To resolve these two issues" launches the remedies.

---

## §IV.A.1 ZigBee Cross-detection · p.167 · ¶1

**Quote:** "as illustrated in Figure 9, the ZigBee packet starts with eight repeated '0's as the preamble field. With this observation, detecting the arrival of a ZigBee packet at commodity WiFi is achieved by checking if the eight consecutive FFT groups are the same. Specifically, we calculate the multiplication of the similarity between the current FFT group Yn and the seven following FFT groups:"

1. PREMISE — preamble of eight '0's.
2. CLAIM (METHODOLOGY) — detect arrival via eight-FFT-group equality.
3. BRIDGE — leads to Eq.(3).

**Shape:** PREMISE → CLAIM → BRIDGE
**Persuasive move:** exploit protocol regularity for detection.
**Sentence flow notes:** "With this observation" links premise to method; "Specifically" leads to formula.

---

## §IV.A.1 · p.167 · ¶2 (after Eq.(3))

**Quote:** "If this value reaches to a threshold (e.g., 0.4 4), a ZigBee packet is detected. Then, we know the current FFT group Yn captures the start of the ZigBee signal, and the FFT magnitude Yn is the first FFT magnitude of the first ZigBee symbol '0' in the ZigBee preamble."

1. DEFINITION (METHODOLOGY) — threshold detection rule.
2. CONSEQUENCE — identifies start of ZigBee signal.

**Shape:** DEFINITION → CONSEQUENCE
**Persuasive move:** thresholding pinpoints synchronization.
**Sentence flow notes:** "If ... a ZigBee packet is detected" then "Then, we know" derives positional knowledge.

---

## §IV.A.2 Fine-grained Synchronization · p.167 · ¶1

**Quote:** "we note that the random delay shifts the whole ZigBee signal and changes the FFT magnitude. By leveraging the knowledge that the first eight ZigBee symbols (preamble) are known, X-Disco detects the delay via matching the delayed version of the template FFT group of ZigBee symbol '0' and the FFT group of the first received ZigBee symbol '0'. Specifically, we create the template FFT group for each of 16 ZigBee symbols with all possible delays – Zd_i ≜ [Zd_i,1, Zd_i,2, Zd_i,3, Zd_i,4], where Zd_i,k represents the k-th FFT magnitude of the ZigBee symbol i with a delay of d samples. Thus, the random delay τ is detected by finding which τ maximizes the similarity between the FFT group of the first received ZigBee symbol '0' and the delayed template FFT group:"

1. PREMISE (CAUSE) — random delay distorts FFT magnitude.
2. CLAIM (METHODOLOGY) — match against delayed preamble template.
3. DEFINITION — delayed templates Zd_i.
4. BRIDGE — leads to Eq.(4).

**Shape:** PREMISE → CLAIM → DEFINITION → BRIDGE
**Persuasive move:** known-preamble as anchor for delay estimation.
**Sentence flow notes:** "By leveraging the knowledge" cites preamble; "Specifically" introduces formal templates; "Thus" yields the argmax.

---

## §IV.A.2 · p.167 · ¶2 (after Eq.(4))

**Quote:** "Figure 10 demonstrates the similarity between the FFT group of the ZigBee signal, depicted in Figure 8, and the delayed template FFT group. The similarity reaches the maximum at the delay of 1.15µs, which is exactly the delay of the signal in Figure 8, validating the effectiveness of this design."

1. EVIDENCE — Figure 10 result.
2. EVIDENCE (CLAIM) — peak at 1.15µs matches ground truth, validating design.

**Shape:** EVIDENCE → EVIDENCE
**Persuasive move:** anecdote-confirms-method.
**Sentence flow notes:** "validating the effectiveness of this design" is the closing assertion.

---

## §IV.A.2 · p.167 · ¶3

**Quote:** "As the random delay is detected, the ZigBee symbols within the PHY-layer payload field, which are also shifted by the same delay, are decoded by checking which template FFT group of the delay τ is the closest to the received FFT group:"

1. METHODOLOGY (BRIDGE) — payload decoding using detected delay.

**Shape:** METHODOLOGY
**Persuasive move:** transfer-of-calibration.
**Sentence flow notes:** "As the random delay is detected" connects prior step to new rule (Eq.5).

---

## §IV.A.2 · p.167 · ¶4 (after Eq.(5))

**Quote:** "As a result, applying the decoding approach described in Equation 5 on all the received FFT groups, the X-Disco device decodes the entire ZigBee packet. Built on top of ZigBee Symbol Extraction, X-Disco is able to decode the ZigBee messages exchanged in ZigBee Coordinator Detection and Neighbor Information Acquisition. To achieve zero cost in deploying X-Disco into practice, we need X-Disco to be transparent to the existing ZigBee network. That is, our design should be compatible with the ZigBee protocol. Next, we introduce detailed designs to meet this goal."

1. CONSEQUENCE — packet fully decodable.
2. CLAIM — symbol extraction enables higher-layer steps.
3. PREMISE — transparency required for zero-cost deployment.
4. DEFINITION — transparency = protocol compatibility.
5. BRIDGE — to next subsections.

**Shape:** CONSEQUENCE → CLAIM → PREMISE → DEFINITION → BRIDGE
**Persuasive move:** capability summary → design-constraint reminder.
**Sentence flow notes:** "As a result," "Built on top of," "To achieve zero cost," "That is," "Next" stage the closing chain.

---

## §IV.B ZigBee Coordinator Detection · p.167 · ¶1

**Quote:** "Detecting the ZigBee coordinators using commodity WiFi is non-trivial because we need to maintain transparency to the existing ZigBee network. One straightforward way is to let the X-Disco device passively listen to the ZigBee channel until the periodic broadcasted ZigBee beacon packets are captured and decoded at commodity WiFi via ZigBee Symbol Extraction. Nevertheless, most ZigBee networks are non-beacon-enabled networks, which do not support beacon packets."

1. CLAIM (CAUSE) — detection hard under transparency constraint.
2. CONCESSION — naive passive listening is conceivable.
3. CONTRAST — but most networks lack beacons.

**Shape:** CLAIM → CONCESSION → CONTRAST
**Persuasive move:** strawman-and-knock-down.
**Sentence flow notes:** "One straightforward way" stages strawman; "Nevertheless" knocks it down.

---

## §IV.B · p.167–168 · ¶2

**Quote:** "In contrast to the passive listening, our proposed ZigBee Coordinator Detection actively triggers the ambient ZigBee coordinators to share their essential ZigBee network information with the commodity WiFi devices. This is achieved by leveraging the ZigBee Passive Acknowledgement mechanism specified in the ZigBee protocol [8], where the ZigBee coordinators would rebroadcast any received broadcast packets as a confirmation of packet reception, as opposed to explicitly transmitting the MAC-layer ACK packets. It is worth mentioning that: (i), rebroadcasting packets at ZigBee coordinators indicates the existence of the ZigBee coordinators. (ii), the rebroadcasted packets encapsulate essential ZigBee network information, such as PAN IDs and addresses, in their header fields, as illustrated in Figure 11."

1. CLAIM (METHODOLOGY) — active triggering approach.
2. METHODOLOGY (DEFINITION) — leverages Passive Acknowledgement.
3. CONSEQUENCE — rebroadcasts imply coordinator presence.
4. EVIDENCE — rebroadcasts carry PAN ID and addresses.

**Shape:** CLAIM → METHODOLOGY → CONSEQUENCE → EVIDENCE
**Persuasive move:** protocol-feature-as-lever.
**Sentence flow notes:** "In contrast to the passive listening" positions the new method; "It is worth mentioning that" enumerates derived virtues.

---

## §IV.B · p.168 · ¶3

**Quote:** "Our design achieves this by exchanging two messages (M1 and M2) between the X-Disco device and the Zigbee coordinator. As illustrated in Figure 12, the X-Disco device (commodity WiFi) sends out an emulated ZigBee broadcast packet via CTC [11] in the WiFi message M1 and switches to the Spectral Scan mode. Upon the reception of the emulated ZigBee broadcast packet, the ZigBee coordinator rebroadcasts with a message M2, from which the ZigBee coordinator is detected and the essential ZigBee network information is obtained by the X-Disco device."

1. METHODOLOGY — two-message exchange.
2. METHODOLOGY — M1 transmission and Spectral Scan switch.
3. CONSEQUENCE — M2 yields detection and info extraction.

**Shape:** METHODOLOGY → METHODOLOGY → CONSEQUENCE
**Persuasive move:** stepwise concretization.
**Sentence flow notes:** "As illustrated in Figure 12" tied to diagram; "Upon the reception" sequences M2.

---

## §IV.B · p.168 · ¶4

**Quote:** "Specifically, the X-Disco device configures each field, as illustrated in Figure 11, where the header is set to be the broadcast mode for the emulated ZigBee broadcast packet. Then, we apply CTC to transmit that customized ZigBee broadcast packet on the X-Disco device. After the ZigBee coordinator receives the emulated ZigBee broadcast packet, it fills its PAN ID and address fields into the MAC Header and the Network Header to construct the rebroadcasted packet. Eventually, running on the Spectral Scan mode, the X-Disco device applies ZigBee Symbol Extraction to obtain this encapsulated essential ZigBee network information, which is further utilized in Neighbor Information Acquisition. If there are multiple ZigBee coordinators nearby, the rebroadcasted ZigBee packets are transmitted with different delays due to CSMA, ensuring that ZigBee network information of all ambient ZigBee coordinators is collected without collision, thus the minimum overhead."

1. METHODOLOGY — field configuration.
2. METHODOLOGY — CTC transmission.
3. METHODOLOGY — coordinator fills headers.
4. CONSEQUENCE — Symbol Extraction yields network info.
5. CLAIM (CONSEQUENCE) — CSMA naturally serializes multiple coordinators.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → CONSEQUENCE → CLAIM
**Persuasive move:** mechanism + scalability dividend.
**Sentence flow notes:** "Specifically," "Then," "After," "Eventually," "If there are multiple ZigBee coordinators nearby" run sequentially.

---

## §IV.C Neighbor Information Acquisition · p.168 · ¶1

**Quote:** "Acquiring the ZigBee neighbor information from the ambient ZigBee coordinators leverages the Device and Service Discovery in ZigBee protocol [8], which allows a ZigBee device to request the Network addresses of all the ZigBee neighbors associated with a specific ZigBee coordinator. This is realized via exchanging IEEE_addr_req and IEEE_addr_rsp messages, where the formats are described in Figure 13. By setting the \"RequestType\" and \"StartIndex\" to be 0x01 and 0x00 respectively, a ZigBee device transmits a IEEE_addr_req packet to trigger the ZigBee coordinator with the network address of \"NWKAddrOfInterest\" to respond with a IEEE_addr_rsp message, containing the number of the associated ZigBee devices in the \"NumAssocDev\" field and the network addresses of all associated ZigBee devices in the \"NWKAddr AssocDevList\" field."

1. METHODOLOGY (PREMISE) — leverages Device and Service Discovery.
2. METHODOLOGY — uses req/rsp message pair.
3. METHODOLOGY (DEFINITION) — concrete field settings to elicit list response.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY
**Persuasive move:** protocol-feature-reuse.
**Sentence flow notes:** "This is realized via" specifies messages; "By setting" details parameters.

---

## §IV.C · p.168 · ¶2

**Quote:** "Our Neighbor Information Acquisition also contains two messages (M3 and M4) exchanged between the X-Disco device and the Zigbee coordinator, as illustrated in Figure 1(b). Specifically, the X-Disco device uses the message M3 to emulate an IEEE_addr_req packet, where the \"NWKAddrOfInterest\" is set to be the ZigBee coordinator's network address obtained in the ZigBee Coordinator Detection. After the X-Disco device transmits this emulated packet, the ZigBee coordinator responds with the corresponding IEEE_addr_rsp message, i.e., M4, which is decoded via ZigBee Symbol Extraction by the X-Disco device. Then, X-Disco skips the first 42 symbols (all fields before \"NumAssocDev\") and obtains the number of the ZigBee neighbors from the 43rd to 44th symbols (\"NumAssocDev\" field). Eventually, the X-Disco device gets the network address of each ZigBee neighbor from 47th to the last symbol (\"NWKAddr AssocDevList\" field) in the packet, thereby completing the discovery of the ZigBee neighbors."

1. METHODOLOGY — M3/M4 architecture.
2. METHODOLOGY — M3 with NWKAddrOfInterest from previous step.
3. METHODOLOGY — M4 reception and decoding.
4. METHODOLOGY — symbol parsing (42-skip, 43-44 count).
5. CONSEQUENCE — address list yields discovery completion.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CONSEQUENCE
**Persuasive move:** byte-level mechanism completion.
**Sentence flow notes:** "Specifically," "After," "Then," "Eventually" sequence the byte-level walk.

---

## §V. Advanced Features of X-Disco · p.168 · ¶0 (lead-in)

**Quote:** "In addition to the main design, X-Disco supports two advanced features for discovering the ZigBee neighbors in more generalized scenarios."

1. BRIDGE — preview of two advanced features.

**Shape:** BRIDGE
**Persuasive move:** scope expansion teaser.
**Sentence flow notes:** "In addition to the main design" hands off to subsections.

---

## §V.A ZigBee Neighbor Validation · p.168–169 · ¶1

**Quote:** "Sometimes, a ZigBee coordinator's associated device does not mean it is also the X-Disco device's neighbor. In other words, some of the discovered ZigBee neighbors might be the hidden terminals, as the X-Disco device might be out of these devices' coverage while they are still associated with the ZigBee coordinator. To address this situation, we leverage the Network Address and IEEE Address Conversion, which is also provided by the ZigBee Device and Service Discovery mechanism, to further validate the fetched ZigBee neighbor information. If \"NWKAddrOfInterest\" is the network address of the ZigBee device and \"RequestType\" is \"0x00\" in the IEEE_addr_req packet, as illustrated in Figure 13, only this specific ZigBee device would respond with the IEEE_addr_rsp packet, which contains its IEEE address in the \"IEEEAddr RemoteDev\" field. Given that observation, to validate if a specific ZigBee device is the X-Disco device's neighbor, the X-Disco device emulates an IEEE_addr_req with that specific ZigBee device's network address, obtained in Neighbor information Acquisition, and waits for a response. That ZigBee device is a valid cross-technology neighbor if a IEEE_addr_rsp packet is captured by ZigBee Symbol Extraction; otherwise, not."

1. CONCESSION (PREMISE) — association ≠ neighbor.
2. DEFINITION — hidden terminal scenario.
3. CLAIM (METHODOLOGY) — use Address Conversion to validate.
4. PREMISE — protocol rule for 0x00 RequestType.
5. METHODOLOGY — emulate per-device IEEE_addr_req.
6. DEFINITION (CONSEQUENCE) — decision rule for validity.

**Shape:** CONCESSION → DEFINITION → CLAIM → PREMISE → METHODOLOGY → DEFINITION
**Persuasive move:** corner-case-handling via protocol feature.
**Sentence flow notes:** "Sometimes," "In other words," "To address this situation," "Given that observation," "otherwise" stage problem → remedy → rule.

---

## §V.A · p.169 · ¶2

**Quote:** "Another rare scenario for cross-technology neighbor discovery is an independent ZigBee device. Unlike WiFi devices, which associate and dissociate with WiFi routers frequently, it is very rare for the ZigBee devices not to associate with any ZigBee coordinators because the ZigBee devices are usually deployed in a network scale and set up manually. Therefore, X-Disco would detect the ZigBee neighbors in most of the scenarios, including smart homes and smart factories, thereby showing vast potential for wide deployment."

1. SCOPE (CONCESSION) — flags edge case of independent ZigBee device.
2. CONTRAST (PREMISE) — manual deployment makes such cases rare.
3. CONSEQUENCE (CLAIM) — coverage holds for most realistic scenarios.

**Shape:** SCOPE → CONTRAST → CONSEQUENCE
**Persuasive move:** edge-case-discounted-by-deployment-reality.
**Sentence flow notes:** "Unlike WiFi devices," "Therefore" pivot from concession to general validity.

---

## §V.B Interruption Mitigation · p.169 · ¶1

**Quote:** "When we deploy X-Disco in the WiFi traffic intensive environment, the FFT magnitude collection is easily interrupted by ambient WiFi traffic. This is because the X-Disco device would switch back to the packet reception mode to start the decoding process if a WiFi packet arrives. To mitigate this interruption, we change the X-Disco device's center frequency to minimize the spectrum overlapped with ambient WiFi traffic. In specific, most 2.4GHz WiFi traffic, modulated by OFDM (802.11g and 802.11n) or CCK (802.11b), is on WiFi channels 1, 6, and 11. Such WiFi packets are so sensitive to the WiFi center frequency that even a 1MHz misalignment results in zero packet reception rate. According to our evaluation in Section VI-B, working on 2.425GHz, the X-Disco device effectively mitigates the interruption caused by the 802.11g/n (OFDM) and 802.11b (CCK) traffic."

1. PREMISE (CAUSE) — WiFi traffic interrupts FFT collection.
2. CAUSE — radio switches to RX mode on incoming WiFi.
3. CLAIM (METHODOLOGY) — shift center frequency to mitigate.
4. PREMISE — WiFi channels concentrated at 1/6/11.
5. EVIDENCE — 1MHz misalignment kills reception.
6. EVIDENCE (CONSEQUENCE) — 2.425GHz operation mitigates interruption.

**Shape:** PREMISE → CAUSE → CLAIM → PREMISE → EVIDENCE → EVIDENCE
**Persuasive move:** root-cause → frequency-offset remedy → empirical confirmation.
**Sentence flow notes:** "This is because," "To mitigate this interruption," "In specific," "According to our evaluation" chain cause/effect to validation.

---

## §VI. Evaluation · p.169 · ¶1

**Quote:** "We build X-Disco on USRP B210 and TP-link WDR 4300 WiFi router, as illustrated in Figure 14. Implementing X-Disco on commodity WiFi devices is supported by the WiFi driver. We implement X-Disco in two parts for evaluation purposes. Specifically, we use a TP-link WiFi router to emulate the ZigBee broadcast packet and IEEE_addr_req packet while the USRP is for collecting FFT magnitudes of the received signal as WiFi Spectral Scan mode. Since our WiFi router does not support Spectral Scan mode, the FFT data collection is implemented at USRP. We also implement the ZigBee passive acknowledgement mechanism and IEEE_addr_rsp packets on TelosB motes. The primary metric to evaluate X-Disco is the time consumed for discovering all the ZigBee neighbors. We evaluate X-Disco in the office (None Line-of-sight) and the hallway (Line-of-sight). We also evaluate the advanced features of X-Disco in the office."

1. METHODOLOGY — hardware platform.
2. CLAIM — driver supports commodity implementation.
3. METHODOLOGY — split-implementation rationale.
4. METHODOLOGY — TP-link and USRP roles.
5. CAUSE (METHODOLOGY) — router lacks Spectral Scan → USRP for FFT.
6. METHODOLOGY — TelosB role.
7. DEFINITION — primary metric: discovery time.
8. SCOPE — NLoS office + LoS hallway.
9. SCOPE — advanced features in office.

**Shape:** METHODOLOGY → CLAIM → METHODOLOGY → METHODOLOGY → CAUSE → METHODOLOGY → DEFINITION → SCOPE → SCOPE
**Persuasive move:** testbed-and-metric setup.
**Sentence flow notes:** "Specifically," "Since our WiFi router does not support" justify USRP use; final sentences enumerate scopes.

---

## §VI.A X-Disco Performance · p.169 · ¶1

**Quote:** "As depicted in Figure 15(a), we deploy the ZigBee network in the office where eight TelosB motes marked in green circles work as the ZigBee end devices and one TelosB mote as the ZigBee coordinator is placed in six different positions for six experiments. The X-Disco device is working on 2.425GHz and all ZigBee devices are working on ZigBee channel 16 (2.43GHz). The detailed results are demonstrated in Figure 15(b). The average time to detect all nine ZigBee devices, including the ZigBee coordinator, is 42ms, 65.4ms, 41.7ms, 68.2ms, 59.2ms, and 53.7ms, when the ZigBee coordinator is placed at the positions 1 to 6. The reason the time varies for different positions is that when the ZigBee coordinator is placed at some positions (e.g., positions 2 and 4), the packet emulation does not work well since the emulated packets are sensitive to the low SNR and thus not successfully received, thereby resulting in more retransmission of emulated ZigBee packets until the response is triggered. Therefore, for those positions, X-Disco takes a longer time to obtain the ZigBee neighbor information."

1. METHODOLOGY — NLoS setup.
2. METHODOLOGY — frequency assignments.
3. BRIDGE — to results figure.
4. EVIDENCE — per-position discovery times.
5. CAUSE (INTERPRETATION) — low-SNR positions trigger retransmissions.
6. CONSEQUENCE — those positions yield longer discovery time.

**Shape:** METHODOLOGY → METHODOLOGY → BRIDGE → EVIDENCE → CAUSE → CONSEQUENCE
**Persuasive move:** quantitative-result-with-causal-explanation.
**Sentence flow notes:** "The reason the time varies" introduces causal account; "Therefore" closes loop.

---

## §VI.A · p.169 · ¶2

**Quote:** "We also deploy the X-DISCO and ZigBee devices in the hallway for evaluating X-Disco in the Line-of-Sight (LoS) scenario, as illustrated in Figure 16(a), where we deploy the ZigBee coordinator at the distance of 6 to 21 meters with respect to X-Disco. Accordingly, the time consumed for fetching the neighbor information from the ZigBee coordinator is 53.7ms, 46.7ms, 53.3ms, 67.5ms, 57.1ms, and 62.1ms. In the LoS and NLoS experiments, we finish the neighbor discovery within 70ms on average, showing the effectiveness and reliability of X-Disco."

1. METHODOLOGY — LoS setup over 6–21m.
2. EVIDENCE — per-distance discovery times.
3. CLAIM (INTERPRETATION) — sub-70ms average demonstrates effectiveness.

**Shape:** METHODOLOGY → EVIDENCE → CLAIM
**Persuasive move:** numbers-back-the-headline.
**Sentence flow notes:** "Accordingly," "In the LoS and NLoS experiments" deliver the headline finding.

---

## §VI.B Impact of WiFi traffic · p.170 · ¶1

**Quote:** "As we explained in Section V-B, we shift the center frequency of the X-Disco device to avoid the interruption caused by the ambient WiFi traffic. In this experiment, we place a WiFi transmitter and WiFi receiver at the distance of 1 meter working on the default transmission power (17dBm). We control the frequency misalignment by shifting the WiFi transmitter's center frequency by 1MHz at a time and checking the packet reception rate of the 802.11b (modulated by DBPSK) and 802.11n (modulated by OFDM) packets. As illustrated in Figure 17(a), the reception rate of 802.11n packets drops to near 0% with more than 1MHz frequency misalignment while the reception rate of 802.11b packets drops to 23% with 10MHz frequency misalignment. Therefore, we let the X-Disco device work on the 2.425GHz (middle of WiFi channels 1 and 6), which is supported by setting register \"freq\" [15] in ath9k driver, to avoid the interruption caused by WiFi traffic."

1. BRIDGE (PREMISE) — recap of V-B remedy.
2. METHODOLOGY — measurement setup.
3. METHODOLOGY — sweeping misalignment.
4. EVIDENCE — 802.11n dies past 1MHz, 802.11b drops to 23%.
5. CONSEQUENCE (CLAIM) — operate at 2.425GHz.

**Shape:** BRIDGE → METHODOLOGY → METHODOLOGY → EVIDENCE → CONSEQUENCE
**Persuasive move:** empirical-justification-of-design-choice.
**Sentence flow notes:** "Therefore" links data to design choice.

---

## §VI.B · p.170 · ¶2

**Quote:** "Based on this setting, we evaluate the performance of X-Disco with and without WiFi interference. In this experiment, we control a USRP to inject a WiFi 802.11b packet of 3.5ms every 10ms (35% channel occupation rate). As illustrated in Figure 17(b), the average time consumed to detect the ZigBee neighbors under WiFi interference is 50.1ms, which is only 4ms longer than the case without interference, indicating the effectiveness of our interruption mitigation design."

1. METHODOLOGY — with/without interference.
2. METHODOLOGY — 35% channel occupancy injection.
3. EVIDENCE (CLAIM) — 50.1ms vs +4ms overhead validates mitigation.

**Shape:** METHODOLOGY → METHODOLOGY → EVIDENCE
**Persuasive move:** small-delta-proves-robustness.
**Sentence flow notes:** "indicating the effectiveness" closes with verdict.

---

## §VI.C Multi-channel Discovery · p.170 · ¶1

**Quote:** "As depicted in Figure 18(a), in this experiment, we control the X-Disco device (2.425GHz) to discover the ZigBee neighbors on two ZigBee channels simultaneously. The X-Disco device emulates ZigBee broadcast packets on ZigBee channels 14 (2.42GHz) and 16 (2.43GHz), where each channel contains 9 ZigBee neighbors, via CTC [11]. Correspondingly, the FFT magnitudes capture the rebroadcasted ZigBee packets on these two channels in the subcarriers 12-19 and subcarriers 45-52, from which the ZigBee network information is extracted via ZigBee symbol extraction. Then, X-Disco emulates IEEE_addr_req messages on two channels and acquires the ZigBee neighbor information from the responded IEEE_addr_rsp packets accordingly. The distance between the X-Disco device and ZigBee coordinators is 6 meters. With multi-channel discovery, the time consumed to discover all 18 ZigBee neighbors on two channels is 79.2ms, as illustrated in Figure 18(b). In contrast, without the help of multi-channel discovery, the X-Disco device has to discover the ZigBee neighbors on channel 14 and then switches center frequency to discover the rest Zigbee neighbors on ZigBee channel 16, which costs 119.3ms on average."

1. METHODOLOGY — dual-channel setup.
2. METHODOLOGY — emulate on channels 14 and 16.
3. METHODOLOGY — subcarrier capture of rebroadcasts.
4. METHODOLOGY — req/rsp on both channels.
5. METHODOLOGY — 6m distance.
6. EVIDENCE — 79.2ms multi-channel time.
7. CONTRAST (EVIDENCE) — 119.3ms baseline without multi-channel.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE → CONTRAST
**Persuasive move:** ablation-style speedup demonstration.
**Sentence flow notes:** "Correspondingly," "Then," "With multi-channel discovery," "In contrast" sequence the comparison.

---

## §VI.D ZigBee Neighbor Validation · p.170 · ¶1

**Quote:** "To show the performance of ZigBee neighbor validation, we deploy a ZigBee network of 9 ZigBee devices in the office, as depicted in Figure 19(a). After the X-Disco device obtains the ZigBee neighbor information, it validates the neighbor one by one. We run the experiments 200 times and record the total time to discover and validate all 9 ZigBee neighbors. The detailed results are shown in Figure 19(b), where all neighbors are validated within 177ms in the 50% of experiments and all validations are finished within 382ms."

1. METHODOLOGY — 9-device office setup.
2. METHODOLOGY — sequential validation.
3. METHODOLOGY — 200 repetitions.
4. EVIDENCE — 177ms median, 382ms max.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE
**Persuasive move:** distributional-evidence-for-feature.
**Sentence flow notes:** Successive declaratives end on the CDF-based numbers.

---

## §VI.E Mobile · p.170–171 · ¶1

**Quote:** "We also evaluate X-disco in the mobile scenario, as depicted in Figure 20(a). As we walk along the blue dotted trace with the X-Disco device at the speed of 1m/s, the X-Disco device keeps discovering the ambient ZigBee neighbors. The whole walk takes 23 seconds and the average time to discover all ZigBee neighbors is 63.7ms, 62.2ms, 77.1ms, 57ms, 82ms, 69.6ms, 64.6ms, 49.7ms, 110ms, 33.5ms, 46.8ms, and 59ms over the time, as shown in Figure 20(b)."

1. METHODOLOGY — mobile scenario setup.
2. METHODOLOGY — 1m/s walking discovery.
3. EVIDENCE — per-window discovery times.

**Shape:** METHODOLOGY → METHODOLOGY → EVIDENCE
**Persuasive move:** mobility-robustness demonstration.
**Sentence flow notes:** "As we walk along," "The whole walk takes" stack into the results.

---

## §VII. Related Work · p.171 · ¶1

**Quote:** "Neighbor discovery has been widely studied in ad-hoc networks [22]. Nevertheless, in a more practical scenario, involving heterogeneous wireless technologies, the requirement of direct communication is not satisfied. We note that the recently proposed cross-technology communication designs [11], [15], [23] help a commodity WiFi device to transmit a message to a ZigBee device directly."

1. PREMISE — prior ad-hoc neighbor-discovery literature.
2. CONTRAST (CONCESSION) — heterogeneous case lacks direct comm.
3. BRIDGE — CTC fills that gap.

**Shape:** PREMISE → CONTRAST → BRIDGE
**Persuasive move:** position-against-classical-literature.
**Sentence flow notes:** "Nevertheless" and "We note that" sequence the pivot.

---

## §VII. Related Work · p.171 · ¶2

**Quote:** "Based on CTC, many works improve the performance of channel coordination [5] and cooperation [7]. Two papers [10], [24] claim they focus on the cross-technology neighbors discovery. However, applying the WiFi to ZigBee CTC to assist ZigBee devices in detecting ZigBee neighbors, NewBee [24] is still for discovering homogeneous wireless neighbors. SERVOUS [10] is using ZigBee device to detect BLE neighbors while it requires modification at both ZigBee and BLE sides, incurring unaffordable costs at deploying that design into practice. Compared to SERVOUS, X-Disco is transparent to the ZigBee network, at the zero cost for installing X-Disco to the WiFi device without any modification to the existing ZigBee devices and ZigBee network."

1. PREMISE — CTC-based coordination/cooperation works.
2. CONCESSION — two papers claim CTND.
3. CONTRAST — NewBee actually does homogeneous discovery.
4. CONTRAST — SERVOUS modifies both sides, deployment-costly.
5. CLAIM — X-Disco is transparent and zero-cost vs. SERVOUS.

**Shape:** PREMISE → CONCESSION → CONTRAST → CONTRAST → CLAIM
**Persuasive move:** rebuttal-of-closest-prior-work.
**Sentence flow notes:** "However," "Compared to SERVOUS" execute the contrast and positive distinction.

---

## §VII. Related Work · p.171 · ¶3

**Quote:** "X-Disco leverages the WiFi to ZigBee high-throughput CTC [11] to emulate ZigBee packets at commodity WiFi. For decoding the responded ZigBee packets at commodity WiFi, X-Disco utilizes the WiFi Spectral Scan mode to extract the FFT magnitude. Even though SymBee [23] and LEGO-Fi [25] are able to decode ZigBee packets at WiFi device, these designs require significant modification to WiFi PHY layer. Given the cross-technology neighbor information fetched by X-Disco, we are able to avoid the cross-technology interference [5] , demonstrating the tremendous applications of X-Disco in the future."

1. METHODOLOGY — CTC for emulation.
2. METHODOLOGY — Spectral Scan for decoding.
3. CONCESSION (CONTRAST) — SymBee/LEGO-Fi exist but require PHY-layer mods.
4. CONSEQUENCE (CLAIM) — X-Disco enables CTI avoidance and future applications.

**Shape:** METHODOLOGY → METHODOLOGY → CONCESSION → CONSEQUENCE
**Persuasive move:** distinguish-on-commodity-compatibility-then-tout-applications.
**Sentence flow notes:** "Even though" introduces concession; "Given the cross-technology neighbor information" closes with utility claim.

---

## §VIII. Conclusion · p.171 · ¶1

**Quote:** "In this paper, we present X-Disco to enable a WiFi device to detect the ambient ZigBee neighbors. We demonstrate the feasibility that a commodity WiFi device is capable of decoding the ZigBee packets just using the FFT magnitude extracted from WiFi Spectral Scan. Based on that, we complete X-Disco for a commodity WiFi device to fetch the ZigBee neighbor information from the ambient ZigBee coordinators. Evaluated in the office (LoS and NLoS), X-Disco discovers nine ZigBee neighbors within 70ms, demonstrating its efficacy in discovering the cross-technology neighbors. More experiments for validating X-Disco's enhanced features and performance in mobile scenarios are performed to show the potential to deploy X-Disco into practice."

1. CLAIM — paper presents X-Disco.
2. CLAIM (CONSEQUENCE) — phase-free decoding feasibility shown.
3. CLAIM — full X-Disco system built atop feasibility.
4. EVIDENCE — 70ms / nine-neighbor headline.
5. CLAIM — enhanced features and mobility tested for deployment potential.

**Shape:** CLAIM → CLAIM → CLAIM → EVIDENCE → CLAIM
**Persuasive move:** recap-with-headline-metric.
**Sentence flow notes:** "We demonstrate," "Based on that," "Evaluated in the office," "More experiments" recap design → core insight → result → extensions.

---

## §Endnotes

- Paragraph count: 47
- Sentence count: 215
- Three most frequent paragraph shapes:
  1. METHODOLOGY → METHODOLOGY → METHODOLOGY ... chains (sequential methodology paragraphs, e.g., §III.B.1¶1, §IV.B¶3, §IV.C¶2, §VI.C¶1, §VI.D¶1) — most frequent shape family.
  2. PREMISE → CONTRAST → CLAIM (and its variants such as CONCESSION → CONTRAST → CLAIM), used in §I¶2, §II.A¶1, §IV.B¶1, §VII¶1 — second most frequent.
  3. METHODOLOGY → EVIDENCE / METHODOLOGY → ... → EVIDENCE closing, used in §IV.A.2¶2, §VI.A¶2, §VI.B¶2, §VI.E¶1 — third most frequent.
