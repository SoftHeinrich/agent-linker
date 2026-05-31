# Exhaustive Argumentation Annotation — Wang et al. (2020), "X-MIMO: Cross-Technology Multi-User MIMO"

Source: `/mnt/hostshare/ardoco-home/mono/writing/gen/refs/wang-2020-xmimo.pdf`

## Table of Contents

- §1 Introduction
- §2 Motivation
  - §2.1 The Need for IoT MU-MIMO
  - §2.2 Opportunity #1: CTC
  - §2.3 Opportunity #2: Multi-antenna WiFi AP
- §3 X-MIMO Overview
  - §3.1 Preliminary: MU-MIMO
- §4 X-MIMO Design
  - §4.1 Cross-technology Channel Estimation
  - §4.2 Timing Control via WiFi Fragmentation
  - §4.3 Cross-technology Precoding
- §5 Multi-stream CTC
  - §5.1 Spectral Efficient Emulation
- §6 Evaluation
  - §6.1 Implementation
  - §6.2 X-MIMO Performance
  - §6.3 Scalability of X-MIMO
  - §6.4 X-MIMO Spectral Efficiency
  - §6.5 Cross-tech. Channel Estimation in Practice
  - §6.6 Obtaining WiFi-ZigBee Mixed Signal
  - §6.7 Impact of Transmission Power
  - §6.8 Immunity to ZigBee ACK Jitter
- §7 Related Work
- §8 Conclusion
- §Appendix — Compensating HW Imperfections
- §Endnotes

---

## §1 Introduction · p.1 · ¶1

**Quote:** "The body of wireless devices is experiencing rapid growth with the emergence of the Internet of Things (IoT) era. The number of IoT devices is expected to grow as large as a trillion by 2035 [46], with the vision of providing pervasive services spanning every corner of our daily lives. To achieve this, the key factor in IoT is the capability to extend to an extreme scale in a spectrum efficient manner, thereby enabling prevalent deployment. This is indeed critical considering that the IoT standards inevitably suffer from a slow transmission rate (and thus low spectrum efficiency), in order to simplify the modulation and keep the receiver radio architecture simple, low-cost, and power-efficient. For instance, ZigBee and Bluetooth have 0.125 and 1 bits/s/Hz, which are 240 and 30 times lower spectrum efficiencies compared to WiFi 802.11n (30 bits/s/Hz)."

1. PREMISE — frames IoT growth backdrop.
2. EVIDENCE (PREMISE) — trillion-device projection citation.
3. CLAIM — spectral efficiency is the key scaling factor.
4. PREMISE (CAUSE) — IoT standards trade rate for simplicity.
5. EVIDENCE (EXAMPLE) — quantitative ratio against WiFi.

**Shape:** PREMISE → EVIDENCE → CLAIM → PREMISE → EVIDENCE
**Persuasive move:** scale-versus-efficiency framing motivates the need.
**Sentence flow notes:** "The number of IoT devices is expected..." quantifies S1; "To achieve this" pivots from growth to required capability; "This is indeed critical considering that" justifies the criticality; "For instance" instantiates the ratio.

---

## §1 Introduction · p.1 · ¶2

**Quote:** "MU-MIMO, by enabling a transmitter to simultaneously deliver different packets to multiple receivers, has been adopted in a wide range of practical wireless systems including WiFi and LTE. MU-MIMO serves as the foundational component to extend the scalability under limited channel resource, recently to a massive level (e.g., massive MIMO in 5G [33]) – a feature which IoT would critically benefit from. However, achieving this in the IoT domain is challenging due to the following intrinsic limitations: (i) Most of IoT devices are equipped with a single antenna, while MU-MIMO needs a multi-antenna transmitter. (ii) While channel estimation is an essential part of MU-MIMO, it is typically unavailable in IoT. This is because, for low-power operation and economical hardware, IoT devices are commonly designed as non-coherent receivers where channel estimation is not performed. To overcome these two practical limitations, the existing approaches rely on the high-end software-defined radio [41, 54] and complex signal processing [13, 19], unachievable with commodity IoT."

1. DEFINITION (PREMISE) — defines MU-MIMO and notes adoption.
2. CLAIM — MU-MIMO is foundational scalability lever IoT needs.
3. CONTRAST (CLAIM) — transposing it to IoT is challenging; enumerates two limits.
4. PREMISE — challenge (ii) elaborated: channel estimation absent.
5. CAUSE — non-coherent receiver design explains (ii).
6. CONCESSION (CONTRAST) — prior fixes use SDR/complex processing, unachievable on commodity.

**Shape:** DEFINITION → CLAIM → CONTRAST → PREMISE → CAUSE → CONCESSION
**Persuasive move:** gap-in-prior-work via two-limit framing.
**Sentence flow notes:** "However" inverts the optimistic foundational claim; "(i)…(ii)" structure the limits; "This is because" supplies the cause; "To overcome these two practical limitations" introduces prior-art critique.

---

## §1 Introduction · p.1 · ¶3

**Quote:** "This paper presents X-MIMO, the first work to bring MU-MIMO into the picture of commodity IoT networking. X-MIMO is a zero-cost, software-only solution that uses pervasively-deployed commodity WiFi APs as the IoT MU-MIMO transmitter, to simultaneously deliver different packets to multiple ZigBee devices. X-MIMO does not require additional hardware or modification of firmware or driver. X-MIMO is inspired by the recent advancement in cross-technology communication (CTC), enabling commodity WiFi to transmit ZigBee packets via physical-layer signal emulation [36]. To uniquely enable MU-MIMO CTC, X-MIMO effectively leverages the MIMO capability of 802.11n (the most widely deployed WiFi variant) WiFi APs – That is, multiple antennas and multi-stream signal processing (typically 3 or above) for parallel transmission. Furthermore, among various MIMO technologies, unique features of MU-MIMO make it especially well-suited for the IoT scenario: First, MU-MIMO is designed to support receivers with a single antenna (or fewer than that of the sender) [37, 41, 47], which applies to most of the commodity IoT. Also, by using the technique of precoding, what reaches the receiver (i.e., commodity IoT) is a legitimate ZigBee packet, thereby incurring no extra signal processing overhead on the (typically low-end) IoT."

1. CLAIM — proposes X-MIMO as the first MU-MIMO for commodity IoT.
2. DEFINITION (METHODOLOGY) — characterises X-MIMO as zero-cost software-only.
3. SCOPE — disclaims hardware/firmware/driver modifications.
4. BRIDGE — situates X-MIMO as an evolution of CTC.
5. METHODOLOGY — exploits 802.11n MIMO (multi-antenna, multi-stream).
6. CLAIM (PREMISE) — MU-MIMO is especially apt for IoT; single-antenna receiver friendly.
7. CONSEQUENCE — precoding yields a legitimate ZigBee packet with no extra IoT overhead.

**Shape:** CLAIM → DEFINITION → SCOPE → BRIDGE → METHODOLOGY → CLAIM → CONSEQUENCE
**Persuasive move:** design-derived-from-phenomenon (turn deployed WiFi into IoT MU-MIMO).
**Sentence flow notes:** "X-MIMO is a zero-cost…" elaborates the proposal; "X-MIMO does not require…" amplifies feasibility; "X-MIMO is inspired by" credits CTC heritage; "To uniquely enable" introduces the differentiator; "Furthermore" stacks an alignment argument; "First" and "Also" enumerate the alignment.

---

## §1 Introduction · p.2 · ¶4

**Quote:** "X-MIMO is built with three new mechanisms of (i) cross-technology channel estimation, (ii) cross-technology precoding, and (iii) multi-stream CTC, where making them fully compatible with commodity devices incurs significant challenge. Cross-technology channel between ZigBee and WiFi is measured using WiFi CSI, from which the physical-layer signal (and the channel accordingly) of the received ZigBee is computed. Cross-technology precoding ensures immunity to signal distortion caused by hardware uncertainties in commodity devices. X-MIMO is evaluated on commodity devices of Atheros AR9334 and TelosB as well as on USRP B210 for in-depth analysis. Result demonstrates 495Kbps under <1% symbol error rate (SER) and 704.24 Kbps with 6.1% SER for two and three streams, where near-linear throughput improvement shows the effectiveness of X-MIMO."

1. METHODOLOGY (CLAIM) — enumerates three new mechanisms and notes the challenge.
2. METHODOLOGY — describes the CSI-based channel estimation step.
3. METHODOLOGY (CLAIM) — precoding immunises against hardware uncertainties.
4. METHODOLOGY — names evaluation platforms.
5. EVIDENCE — headline throughput and SER numbers; near-linear scaling.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE
**Persuasive move:** results-preview of mechanism trio.
**Sentence flow notes:** Each sentence narrates one of the three mechanisms before pivoting via "X-MIMO is evaluated on commodity devices" to platforms; the closing "Result demonstrates" delivers the numerical pay-off.

---

## §1 Introduction · p.2 · ¶5

**Quote:** "To the best of our knowledge, X-MIMO is the first of its kind to offer MU-MIMO functionality on commodity IoT networks. In particular, X-MIMO effectively utilizes pervasively available WiFi infrastructure to bring IoT MU-MIMO into practical use, so as to immediately adoptable to billions of households and offices under zero cost. To summarize, our contribution is three-fold:"

1. CLAIM — novelty assertion ("first of its kind").
2. CLAIM (CONSEQUENCE) — adoption pathway via existing WiFi infrastructure.
3. BRIDGE — handoff to bullet list.

**Shape:** CLAIM → CLAIM → BRIDGE
**Persuasive move:** deployment-justification with novelty stamp.
**Sentence flow notes:** "In particular" specialises the novelty claim toward deployability; "To summarize" pivots to the enumerated contributions list.

---

## §1 Introduction · p.2 · ¶6 (contribution bullet 1)

**Quote:** "We design X-MIMO, the first design to support MU-MIMO on commodity IoT devices without hardware or firmware modification. X-MIMO applies WiFi-ZigBee channel estimation and multiple precoded ZigBee signals emulation at commodity WiFi, which yields the different ZigBee packets at commodity ZigBee devices. X-MIMO is totally compatible with and easy to be deployed on commodity devices."

1. CLAIM — first-of-kind design contribution.
2. METHODOLOGY — describes channel-estimation-plus-precoded-emulation pipeline.
3. CLAIM — compatibility and deployability assertion.

**Shape:** CLAIM → METHODOLOGY → CLAIM
**Persuasive move:** contribution-as-deployable-first.
**Sentence flow notes:** S2 unpacks the "how" behind S1's "first" claim; S3 wraps with a compatibility tagline.

---

## §1 Introduction · p.2 · ¶7 (contribution bullet 2)

**Quote:** "To apply X-MIMO in practice, we address three practical challenges: precise timing control, hardware imperfection compensation, and multi-stream ZigBee signals emulation on 802.11n WiFi device. Moreover, our theoretical analysis shows that X-MIMO is immune to the phase uncertainty caused by the carrier frequency offset and hardware jitter."

1. CLAIM (METHODOLOGY) — three practical challenges addressed.
2. CLAIM (EVIDENCE) — theoretical immunity claim about phase uncertainty.

**Shape:** CLAIM → CLAIM
**Persuasive move:** practicality-plus-theory contribution framing.
**Sentence flow notes:** "Moreover" stacks the theoretical immunity claim onto the practical-challenge claim.

---

## §1 Introduction · p.2 · ¶8 (contribution bullet 3)

**Quote:** "We implement and evaluate X-MIMO on commodity devices (TP-link WDR4300 wireless router with Atheros AR9334 and AR9580 WNIC and TelosB ZigBee mote) and USRP. Our experimental results demonstrate X-MIMO achieves reliable and high-throughput performances under line-of-sight and non-line-of-sight scenarios. In all the settings, the ZigBee symbol error rate is less than 1% and the throughput reaches above 495 Kbps, which is 2× of state-of-the-art WEBee [36]. In addition, our evaluation for three-stream X-MIMO reveals the near-linear throughput improvement showing the effectiveness of X-MIMO."

1. METHODOLOGY — describes platforms used.
2. CLAIM (EVIDENCE) — reliability + throughput in LOS/NLOS.
3. EVIDENCE — SER <1%, throughput >495 Kbps, 2× WEBee.
4. EVIDENCE — three-stream near-linear improvement.

**Shape:** METHODOLOGY → CLAIM → EVIDENCE → EVIDENCE
**Persuasive move:** quantitative-evidence-laddering against baseline.
**Sentence flow notes:** "Our experimental results demonstrate" introduces the headline; "In all the settings" tightens the quantitative claim; "In addition" stacks a scalability data point.

---

## §2.1 The Need for IoT MU-MIMO · p.2 · ¶1

**Quote:** "This paper presents spectral efficient IoT down-link by enabling cross-technology multi-user MIMO. As the IoT reaches a massive scale and given the naturally limited resource of the wireless medium, it is critical to manage IoTs in a spectral efficient manner. We note that a large number of ZigBee/802.15.4 IoT devices are widely deployed to support variant applications across different sectors including smart homes and factories. Amazon Echo Plus, Samsung SmartThings, Philips Hue, Hive, Xiaomi Mijia, and IKEA Tradfri are among a large body of smart home gadgets. Smart factories often operate under 802.15.4-based protocols, such as WirelessHART [45], ISA100.11a [7], and TSCH [8]. For instance, Emerson's smart factory IoT network using WirelessHART is deployed at 54K smart factories worldwide, serving over 19 billion operating hours [5]. Managing massive scale IoT involves extensive traffic in controlling operation, updating the firmware for bug fixes, and reprogramming for failure recovery, The IoT traffic is anticipated to increase further with emerging applications such as AR/VR, where the intensive interactions incur heavy real-time traffic. Furthermore, the advancement in on-device AI is expected to increase the down-link traffic (as a tradeoff for reduced up-link) for downloading and updating the trained model. Achieving MU-MIMO for IoT at zero-cost only using the existing WiFi infrastructure is uniquely achieved by the two opportunities discussed in the following."

1. CLAIM — paper enables spectral-efficient IoT down-link via CT-MU-MIMO.
2. PREMISE (CAUSE) — IoT massive scale plus limited spectrum mandates efficiency.
3. PREMISE — large ZigBee deployment landscape.
4. EVIDENCE (EXAMPLE) — smart-home brand list.
5. EVIDENCE (EXAMPLE) — smart-factory protocol list.
6. EVIDENCE — Emerson WirelessHART scale figure.
7. PREMISE (CONSEQUENCE) — management workloads generate extensive traffic; AR/VR adds more.
8. PREMISE (CONSEQUENCE) — on-device AI further inflates down-link.
9. BRIDGE — handoff to "two opportunities" subsections.

**Shape:** CLAIM → PREMISE → PREMISE → EVIDENCE → EVIDENCE → EVIDENCE → PREMISE → PREMISE → BRIDGE
**Persuasive move:** traffic-pressure stack-up to motivate down-link MU-MIMO.
**Sentence flow notes:** "As the IoT reaches a massive scale" supplies the conditional; "We note that" introduces deployment evidence; "For instance" instantiates with Emerson; "Furthermore" stacks AR/VR; the closing sentence bridges to opportunities.

---

## §2.2 Opportunity #1: CTC · p.2 · ¶1

**Quote:** "The emerging technique of CTC is a software-only solution enabling direct communication between commodity wireless running heterogeneous standards, without any hardware or firmware modification [34, 36, 52]. Communication between WiFi and ZigBee [36] is achieved by elaborately customizing the WiFi payload such that the transmitted WiFi signal is also interpreted and decoded as a ZigBee packet. This offers an opportunity to utilize the existing WiFi devices to manage IoT networks without introducing additional hardware cost. Furthermore, WiFi's high transmission power compared to low-power IoT offers an extended communication range advantageous in IoT management. However, the spectral efficiency of the state-of-the-art CTC is strictly constrained to single-input single-output (SISO), which essentially limits its capability in maintaining massive scale IoT. The current CTC designs simply waste the non-overlapped bandwidths between the transmitter and the receiver – 18 MHz between WiFi (20 MHz) and ZigBee (2 MHz) – further exacerbating the spectral efficiency. Taking CTC as a building block, X-MIMO builds MU-MIMO that fundamentally resolves such inefficiencies, paving a practical pathway to supporting massive IoT. This necessitates generating parallel CTC streams, which is enabled by the next opportunity."

1. DEFINITION — what CTC is, citation-grounded.
2. METHODOLOGY (EXAMPLE) — WiFi-to-ZigBee CTC works via payload customisation.
3. CLAIM (CONSEQUENCE) — opportunity to manage IoT with existing WiFi, no extra hardware cost.
4. PREMISE — WiFi's higher TX power extends range.
5. CONCESSION (CONTRAST) — CTC limited to SISO, capping massive-scale capacity.
6. PREMISE (CAUSE) — 18 MHz of unused bandwidth exacerbates inefficiency.
7. CLAIM — X-MIMO uses CTC as building block to fix this.
8. BRIDGE — bridges to next opportunity (multi-antenna WiFi).

**Shape:** DEFINITION → METHODOLOGY → CLAIM → PREMISE → CONCESSION → PREMISE → CLAIM → BRIDGE
**Persuasive move:** feature-inversion-of-concession (CTC's SISO limit becomes X-MIMO's opening).
**Sentence flow notes:** "Furthermore" stacks range advantage; "However" pivots to limitation; "Taking CTC as a building block" reframes the limitation into a design opening; "This necessitates" links to the next opportunity.

---

## §2.3 Opportunity #2: Multi-antenna WiFi AP · p.2 · ¶1

**Quote:** "In response to the higher throughput demand under the limited ISM spectrum (e.g., 100MHz on 2.4 GHz), WiFi has evolved to support various MIMO technologies. Widely deployed 802.11n WiFi APs are often equipped with multiple antennas (≥3) for MIMO functionality. From the signal processing standpoint, the antenna diversity is achieved by multi-stream data processing that enables simultaneous emission of separate waveforms from each antenna, in parallel. Multi-antenna and multi-stream processing are foundations to turning the WiFi AP into a MU-MIMO IoT transmitter in X-MIMO, where it essentially offers the opportunity for multi-stream CTC. We note that the number of antennas and streams are ever-increasing with the WiFi's evolution towards higher throughput. For instance, 802.11ax supports up to eight antennas and streams. This potentially offers extended opportunity and improved performance for X-MIMO."

1. PREMISE — WiFi evolved to MIMO to handle throughput demand.
2. PREMISE — 802.11n APs commonly have ≥3 antennas.
3. DEFINITION (PREMISE) — multi-stream data processing characterises antenna diversity.
4. CLAIM — these foundations let X-MIMO repurpose AP as MU-MIMO IoT transmitter.
5. PREMISE — antenna/stream counts continue rising.
6. EVIDENCE (EXAMPLE) — 802.11ax up to eight antennas/streams.
7. CONSEQUENCE — implies future X-MIMO performance gains.

**Shape:** PREMISE → PREMISE → DEFINITION → CLAIM → PREMISE → EVIDENCE → CONSEQUENCE
**Persuasive move:** infrastructure-trend-as-tailwind.
**Sentence flow notes:** "From the signal processing standpoint" reframes from hardware to processing; "Multi-antenna and multi-stream processing are foundations" delivers the connection to X-MIMO; "For instance" instantiates with 802.11ax; "This potentially offers" projects forward.

---

## §3 X-MIMO Overview · p.3 · ¶1

**Quote:** "Figure 1 illustrates the three steps of X-MIMO operation, to achieve MU-MIMO on commodity IoT using a WiFi AP: (a) In cross-tech. channel estimation, X-MIMO utilizes the WiFi fragmentation function to precisely control the timings of ZigBee and WiFi packets (from an arbitrary WiFi device associated with the X-MIMO WiFi AP) such that they overlap in time. This yields CSI measurement that reflects the overlapped ZigBee signal, from which X-MIMO recovers the received physical-layer ZigBee signal and further, the corresponding ZigBee channel. (b) In cross-tech. precoding, different ZigBee packets are precoded into multiple streams. such that upon the reception of the precoded streams, all the ZigBee devices are able to decode the different packets simultaneously. Then, (c) multi-stream CTC converts the precoded streams into a WiFi packet with the customized payload. Eventually, X-MIMO transmits this WiFi packet through multiple antennas on the commodity WiFi device and the ZigBee devices decode the different ZigBee packets simultaneously. Next, we introduce the MU-MIMO preliminaries for further understanding of the whole X-MIMO design. Lastly, we note that the number of ZigBee's that X-MIMO can support is throttled by the number of streams limited by the WiFi NIC hardware, which is typically 3 or higher (3 in our experimental device of TP-link WDR4300)."

1. METHODOLOGY — step (a): fragmentation-based timing to overlap WiFi/ZigBee.
2. CONSEQUENCE (METHODOLOGY) — overlap yields CSI from which ZigBee signal/channel are recovered.
3. METHODOLOGY — step (b): precoding into multiple streams enabling simultaneous decoding.
4. METHODOLOGY — step (c): convert precoded streams to a customised WiFi packet.
5. CONSEQUENCE — final transmission delivers different ZigBee packets simultaneously.
6. BRIDGE — points to MU-MIMO preliminaries next.
7. SCOPE — caps user count by WiFi NIC stream count (3 in their setup).

**Shape:** METHODOLOGY → CONSEQUENCE → METHODOLOGY → METHODOLOGY → CONSEQUENCE → BRIDGE → SCOPE
**Persuasive move:** three-step pipeline walkthrough.
**Sentence flow notes:** "(a)…(b)…(c)" enumerators chain the pipeline; "Then" and "Eventually" sequence steps; "Next" and "Lastly" bridge to subsequent material and scope.

---

## §3.1 Preliminary: MU-MIMO · p.3 · ¶1

**Quote:** "MU-MIMO supports multiple users to receive different signals simultaneously via precoding, which weighs each stream with an appropriate phase and amplitude according to the channel between AP and users. As a type of MU-MIMO, implicit MU-MIMO [39, 44, 48], uses up-link channel (Users to AP), which is estimated by AP, to perform precoding. In a typical implicit MU-MIMO scenario, as depicted in Figure 2, after the AP transmits the precoded signals 𝑿 = [𝑋1,𝑋2]⊺, it expects the two users to receive two independent streams, 𝑺 = [𝑆1,𝑆2]⊺:"

1. DEFINITION — MU-MIMO via precoding using AP-user channel.
2. DEFINITION — implicit MU-MIMO uses up-link channel for precoding.
3. METHODOLOGY (EXAMPLE) — typical scenario sketch leading into equation (1).

**Shape:** DEFINITION → DEFINITION → METHODOLOGY
**Persuasive move:** background-grounding for the implicit-MU-MIMO choice.
**Sentence flow notes:** "As a type of MU-MIMO" narrows from general to implicit; "In a typical implicit MU-MIMO scenario, as depicted in Figure 2" sets up the formal model.

---

## §3.1 Preliminary: MU-MIMO · p.3 · ¶2

**Quote:** "where 𝑯 represents the estimated up-link channel. As a consequence, the AP obtains the precoded signals 𝑿 via 𝑿 = 𝑯−1 𝑺. Since it is impractical to extract channel information at commodity low-power IoT devices, X-MIMO adopts an implicit MU-MIMO approach, where precoding is performed at the X-MIMO side, incurring zero modification to the IoT devices, e.g., ZigBee."

1. DEFINITION — variable 𝑯 explained.
2. CONSEQUENCE — derivation of 𝑿 via inverse channel.
3. CAUSE (CLAIM) — IoT can't estimate channels, so X-MIMO adopts implicit MU-MIMO with AP-side precoding.

**Shape:** DEFINITION → CONSEQUENCE → CAUSE
**Persuasive move:** design-choice-justified-by-IoT-constraint.
**Sentence flow notes:** "As a consequence" derives the formula; "Since" supplies the rationale; the trailing clause adds the zero-modification benefit.

---

## §4 X-MIMO Design · p.3 · ¶1

**Quote:** "To support implicit MU-MIMO, we introduce how the cross-tech. channel estimation is performed to implicitly collect the up-link channel information at X-MIMO, followed by the cross-technology precoding in this section."

1. SCOPE — outlines what §4 will cover (channel estimation, then precoding).

**Shape:** SCOPE
**Persuasive move:** roadmap-sentence.
**Sentence flow notes:** Single-sentence section preview chaining estimation to precoding.

---

## §4.1 Cross-technology Channel Estimation · p.3 · ¶1

**Quote:** "Cross-technology channel estimation leverages the channel state information (CSI) provided by commodity WiFi WNICs, to obtain the ZigBee channel. CSI indicates the channel coefficient computed from the HT-LTF field of the WiFi preamble – i.e., by comparing the HT-LTF signal to what is received over the wireless channel:"

1. METHODOLOGY (CLAIM) — leverages WiFi CSI to obtain ZigBee channel.
2. DEFINITION — CSI computed from HT-LTF via signal comparison, lead-in to equation.

**Shape:** METHODOLOGY → DEFINITION
**Persuasive move:** mechanism-grounded-in-standard-feature.
**Sentence flow notes:** "CSI indicates" pivots from claim to definition; "i.e., by comparing" makes the definition operational.

---

## §4.1 Cross-technology Channel Estimation · p.3 · ¶2

**Quote:** "where 𝑋 and 𝑌 are the WiFi HT-LTF and received signals in the frequency domain, respectively. We note that 𝑌 incorporates not only the HT-LTF but also other interfering wireless signals. Interestingly, physical-layer raw samples of such a signal can be recovered from CSI, by 𝑌 = 𝐶𝑆𝐼 × 𝑋 𝑤 . This serves as the fundamental idea behind cross-technology channel estimation – enforce ZigBee to interfere CSI, from which the ZigBee signal and channel are extracted. Next, we discuss this procedure in detail, followed by the mechanism carefully designed to be fully compatible with the WiFi and ZigBee standards and commodity devices."

1. DEFINITION — variable conventions for 𝑋, 𝑌.
2. PREMISE — 𝑌 includes interference, not only HT-LTF.
3. INTERPRETATION (CLAIM) — raw samples recoverable from CSI.
4. CLAIM — the fundamental idea: force ZigBee to interfere CSI, extract signal/channel.
5. BRIDGE — flags upcoming detail and compatibility design.

**Shape:** DEFINITION → PREMISE → INTERPRETATION → CLAIM → BRIDGE
**Persuasive move:** key-insight-by-inverting-interference.
**Sentence flow notes:** "Interestingly" flags the key insight; "This serves as the fundamental idea" elevates it to a thesis; "Next, we discuss" bridges to detail.

---

## §4.1 Cross-technology Channel Estimation · p.3 · ¶3

**Quote:** "Figure 3(a) illustrates the scenario of obtaining the interfering ZigBee channel from the CSI measurement. 𝐻 and 𝐻 represent the WiFi and ZigBee channels, respectively, while 𝑋 indicates the interfering ZigBee signal. Under this scenario the signal at X-MIMO becomes the mixture of the WiFi and ZigBee signals received through the corresponding channels, yielding 𝑌 = 𝐻 𝑤 𝑋 𝑤 + 𝐻 𝑧 𝑋 𝑧 . Plugging this into Eq. 2 we get"

1. METHODOLOGY (EXAMPLE) — references the figure scenario.
2. DEFINITION — variable conventions for channels and signals.
3. METHODOLOGY — derives 𝑌 as mixture of WiFi+ZigBee channels.
4. METHODOLOGY — equation-substitution lead-in.

**Shape:** METHODOLOGY → DEFINITION → METHODOLOGY → METHODOLOGY
**Persuasive move:** formal-derivation-walkthrough.
**Sentence flow notes:** Variables introduced in S2 feed S3's mixture equation; "Plugging this into Eq. 2" sets up Eq. 3.

---

## §4.1 Cross-technology Channel Estimation · p.4 · ¶1

**Quote:** "Doing so indicates that 𝐻 can be computed as the RHS is entirely known: (i) 𝑋 is the standard WiFi HT-LTF, a known signal. (ii) 𝐻 can be found from the previously received WiFi packet, within the coherence time. Lastly, (iii) 𝑋 is also a known signal under the accurate timing in Figure 3(b). That is, by aligning the beginning of the ZigBee packet with WiFi HT-LTF, 𝑋 becomes the first 3.2 𝜇s (i.e., WiFi HT-LTF duration) of the ZigBee preamble, which is known. This effectively demonstrates that cross-technology (i.e., ZigBee) channel can be estimated using WiFi CSI."

1. CLAIM — RHS is fully known so 𝐻𝑧 is computable; enumerates three knowns.
2. PREMISE — known (i): WiFi HT-LTF.
3. PREMISE — known (ii): prior WiFi channel within coherence.
4. PREMISE — known (iii): ZigBee preamble under accurate timing.
5. METHODOLOGY (EXAMPLE) — alignment yields the known 3.2 μs ZigBee preamble.
6. CONSEQUENCE (CLAIM) — demonstrates ZigBee channel can be estimated via WiFi CSI.

**Shape:** CLAIM → PREMISE → PREMISE → PREMISE → METHODOLOGY → CONSEQUENCE
**Persuasive move:** computability-via-three-knowns.
**Sentence flow notes:** "(i)…(ii)…Lastly, (iii)" enumerate knowns; "That is" elaborates known (iii); "This effectively demonstrates" closes with the consequence.

---

## §4.1 Cross-technology Channel Estimation · p.4 · ¶2

**Quote:** "To further understand this intuition, Figure 4 depicts an experimental example of 𝐶𝑆𝐼 phase and amplitude, compared to 𝐻 . The figure shows that subcarriers overlapping with the ZigBee vary significantly, as the 𝐶𝑆𝐼 incorporates ZigBee (𝑋 ) signal and channel (𝐻 ). Meanwhile, the non-overlapped subcarriers (in gray) remain consistent. We note that the Figure 4 is obtained after compensating the offsets that inevitably occurs in practical systems. For brevity, we rigorously discuss the compensation algorithm in Appendix. Figure 5(a) illustrates the computed ZigBee channel, denoted by 𝐻 b𝑧 and Figure 5(b) demonstrates the computed ZigBee signal (i.e., 𝐻 b𝑧 𝑋 𝑧) . They closely approximate the ground truth, providing empirical validation of our technique."

1. EVIDENCE (METHODOLOGY) — Figure 4 supplies an experimental example.
2. EVIDENCE — overlapped subcarriers vary, supporting the mixture model.
3. EVIDENCE — non-overlapped subcarriers stay consistent.
4. SCOPE (CONCESSION) — offsets compensated in practice; details in Appendix.
5. BRIDGE — points to Appendix for compensation algorithm.
6. EVIDENCE — Figure 5(a),(b) show computed channel and signal.
7. CLAIM (EVIDENCE) — close match to ground truth empirically validates the technique.

**Shape:** EVIDENCE → EVIDENCE → EVIDENCE → SCOPE → BRIDGE → EVIDENCE → CLAIM
**Persuasive move:** empirical-validation-of-derivation.
**Sentence flow notes:** "The figure shows" and "Meanwhile" contrast overlapped vs. non-overlapped; "We note that" qualifies; "For brevity" defers detail; "They closely approximate" delivers the validation punch.

---

## §4.1 Cross-technology Channel Estimation · p.4 · ¶3

**Quote:** "Until now we have demonstrated cross-technology channel estimation under the condition that the first 3.2 𝜇s ZigBee preamble (i.e., 𝑋 ) precisely overlaps with the WiFi HT-LTF. In practice, the requirement of such strict timing control is inherently difficult to satisfy. This is because the commodity devices running contention-based MAC protocols (i.e., CSMA), including WiFi and ZigBee, have uncontrollable channel access delays. In the following, we discuss X-MIMO's unique and highly precise timing control mechanism under practical settings."

1. SCOPE — recaps the precondition (3.2 μs overlap).
2. CONCESSION (CLAIM) — strict timing is hard in practice.
3. CAUSE — CSMA introduces uncontrollable delays.
4. BRIDGE — flags the upcoming timing-control mechanism.

**Shape:** SCOPE → CONCESSION → CAUSE → BRIDGE
**Persuasive move:** precondition-then-problem-then-mechanism teaser.
**Sentence flow notes:** "Until now" recaps; "In practice" pivots to the gap; "This is because" supplies the cause; "In the following" bridges to the fix.

---

## §4.2 Timing Control via WiFi Fragmentation · p.4 · ¶1

**Quote:** "X-MIMO's timing control only uses standard-defined functionalities for full compatibility to commodity WiFi and ZigBee – therefore it is, (i) non-disruptive to coexisting networks, (ii) does not require any modification to the firmware or driver, and (iii) is very light-weight, as it does not involve any extra coordination or time synchronization protocols. Further, the timing control operates under a typical WiFi network setting where a WiFi device is associated to a WiFi AP (running X-MIMO). This indicates a wide applicability."

1. CLAIM (METHODOLOGY) — uses only standard functions; enumerates three properties.
2. SCOPE — operates within typical WiFi-AP association.
3. CONSEQUENCE — implies wide applicability.

**Shape:** CLAIM → SCOPE → CONSEQUENCE
**Persuasive move:** deployability-by-standard-only-use.
**Sentence flow notes:** "therefore it is" enumerates properties; "Further" stacks the network-setting claim; "This indicates" rounds off with consequence.

---

## §4.2 Timing Control via WiFi Fragmentation · p.4 · ¶2

**Quote:** "Timing WiFi and ZigBee signals to precisely overlap in time leverages the WiFi packet fragmentation function. Commonly provided in WiFi NICs, this function cuts down a large fragment to smaller pieces where the fragment interval is precisely kept at 60 𝜇s (=2×SIFS (16 𝜇s) + WiFi ACK duration). Meanwhile, ZigBee ACK is triggered exactly 192 𝜇s (macSifsPeriod) after a packet reception. They are both strictly enforced by the standards [3, 6] on commodity devices and serve as our basis to precise time control. We note that WiFi packet fragmentation can be simply set using iwconfig (under Linux) command, without involving any hardware, firmware, or driver modifications."

1. METHODOLOGY (CLAIM) — fragmentation function used to overlap signals.
2. DEFINITION — fragmentation cuts packets with 60 μs interval.
3. PREMISE — ZigBee ACK fires 192 μs after reception.
4. PREMISE (CLAIM) — both intervals are standard-enforced and form the timing basis.
5. METHODOLOGY — fragmentation set via iwconfig with no modifications.

**Shape:** METHODOLOGY → DEFINITION → PREMISE → PREMISE → METHODOLOGY
**Persuasive move:** lever-found-in-standard-MAC-timings.
**Sentence flow notes:** "Meanwhile" pairs the two standard-mandated intervals; "They are both strictly enforced" elevates them to design pillars; "We note that" adds operational simplicity.

---

## §4.2 Timing Control via WiFi Fragmentation · p.4 · ¶3

**Quote:** "Figure 6 illustrates how the timing works using two fragmented WiFi packets. As in Figure 6(a), the WiFi device transmits the first fragmented WiFi packet, which emulates a ZigBee packet1 (i.e., CTC). Note that this is a legitimate WiFi packet encapsulating a ZigBee packet in its payload – therefore, it is received by both WiFi (i.e., X-MIMO) and the ZigBee. Upon receiving this packet, X-MIMO obtains the WiFi channel (WiFi device→X-MIMO) estimation, 𝐻 𝑤 . Meanwhile, packet reception at ZigBee triggers an ACK, as defined in the standard. We note that the entire process leverages the standard MAC protocol and thus does not require coordination between WiFi and ZigBee."

1. METHODOLOGY (EXAMPLE) — figure walkthrough preview.
2. METHODOLOGY — fragment 1 is a CTC-emulating WiFi packet.
3. CONSEQUENCE (CLAIM) — being legitimate, the packet reaches both WiFi and ZigBee.
4. METHODOLOGY (CONSEQUENCE) — X-MIMO gets 𝐻𝑤 estimate from fragment 1.
5. CONSEQUENCE — ZigBee triggers an ACK on reception.
6. CLAIM — entire process is standard-only and needs no coordination.

**Shape:** METHODOLOGY → METHODOLOGY → CONSEQUENCE → METHODOLOGY → CONSEQUENCE → CLAIM
**Persuasive move:** two-fragment choreography walkthrough.
**Sentence flow notes:** "As in Figure 6(a)" anchors the first half; "Note that…therefore" justifies dual reception; "Upon receiving" sequences the channel acquisition; "Meanwhile" parallels the ACK trigger; "We note that" closes with a no-coordination guarantee.

---

## §4.2 Timing Control via WiFi Fragmentation · p.4 · ¶4

**Quote:** "As shown in Figure 6(b), the second fragment from the WiFi device and the ZigBee ACK simultaneously arrive at X-MIMO. This is because both the WiFi fragment and the ZigBee ACK are transmitted with a fixed delay. Furthermore, this overlapped packet is highly likely to be correctly received at X-MIMO given WiFi's significantly higher power. The ZigBee channel, 𝐻 b𝑧 , is computed using the CSI from this packet. We note that the CSI and the estimation results demonstrated in Figures 4 and 5 are obtained from commodity WiFi and ZigBee, using this timing mechanism."

1. METHODOLOGY (EVIDENCE) — fragment 2 and ZigBee ACK overlap at X-MIMO.
2. CAUSE — fixed delays explain the overlap.
3. CLAIM (PREMISE) — overlap likely received correctly thanks to WiFi power.
4. METHODOLOGY — 𝐻𝑏𝑧 computed from this CSI.
5. EVIDENCE — Figures 4–5 use this timing mechanism on commodity devices.

**Shape:** METHODOLOGY → CAUSE → CLAIM → METHODOLOGY → EVIDENCE
**Persuasive move:** timing-deterministic-yields-clean-CSI.
**Sentence flow notes:** "This is because" supplies the cause; "Furthermore" adds the reception likelihood; "We note that" closes by linking back to earlier figures.

---

## §4.2 Timing Control via WiFi Fragmentation · p.5 · ¶1 (Detailed timing)

**Quote:** "Detailed timing. Figure 7 presents the timing details in the channel estimation process. The beginning of the ZigBee ACK is aligned to the HT-LTF of the second fragment, so as to exploit the known ZigBee preamble for ZigBee channel estimation. To do so, symbols are padded after the ZigBee CTC packet such that padded symbol duration, fragment interval (60 𝜇s), and preamble duration (32 𝜇s) add up to ZigBee ACK delay (192 𝜇s). This results in the 100 𝜇s-long padded symbols. Since a WiFi symbol is 4 𝜇s long (including CP), a total of 25 symbols are padded. In practice, this is done by simply setting the fragmentation threshold to the size of the ZigBee CTC packet plus the padded symbols. The overhead for the channel estimation remains light, as (i) the padded symbols can be used for data delivery from the WiFi device to X-MIMO and need not be wasted, and (ii) the narrow-band 2.4 GHz ZigBee channel remains consistent (i.e., the coherence time) over seconds [40, 49], far longer than wider-band systems such as WiFi (∼ 40𝑚𝑠) coherence time [56]. Given that ZigBee packets are typically ∼ 1 ms long, thousands of ZigBee packets can be delivered following an estimation. Channels of multiple ZigBees may be obtained separately by simply repeating the process. Alternatively, the channels may be rapidly estimated back-to-back by increasing fragments, for instance to 4, 6, or more, where two fragments are consumed per channel estimation."

1. BRIDGE — section header sentence introducing figure 7.
2. METHODOLOGY (CLAIM) — ZigBee ACK aligned to HT-LTF of fragment 2.
3. METHODOLOGY — padded symbols achieve the alignment (192 μs decomposition).
4. METHODOLOGY — derives 100 μs of padding.
5. METHODOLOGY — 25 symbols (4 μs each).
6. METHODOLOGY — operationalises via fragmentation threshold.
7. CLAIM — overhead is light; supports with two reasons.
8. PREMISE (CONSEQUENCE) — thousands of ZigBee packets per estimation.
9. METHODOLOGY (SCOPE) — multiple ZigBees: estimate separately or back-to-back.
10. METHODOLOGY (EXAMPLE) — increasing fragments enables rapid batched estimation.

**Shape:** BRIDGE → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CLAIM → PREMISE → METHODOLOGY → METHODOLOGY
**Persuasive move:** arithmetic-derivation-plus-overhead-defence.
**Sentence flow notes:** "To do so" sequences justification; "This results in" derives 100 μs; "Since" derives symbol count; "In practice" operationalises; "as (i)…(ii)" enumerates overhead reasons; "Given that" delivers the throughput argument; "Alternatively" presents the batched option.

---

## §4.3 Cross-technology Precoding · p.5 · ¶1

**Quote:** "Here we discuss how uplink channel estimation is applied to precoding to achieve MU-MIMO. As in Figure 8, for simplicity we present our design for two ZigBee receivers, where it can be straight-forwardly extended to support more users as evaluated in Section 6.3. Let us denote the two ZigBee packets to be delivered as 𝒁 = [𝑍1,𝑍2]⊺. By directly applying the estimated uplink channel"

1. SCOPE — preview of subsection on precoding from estimated uplink channel.
2. SCOPE — two-receiver case for simplicity; extends to more.
3. DEFINITION — notation for the two ZigBee packets, leading into matrix equation.

**Shape:** SCOPE → SCOPE → DEFINITION
**Persuasive move:** simplified-derivation-then-generalisation.
**Sentence flow notes:** "As in Figure 8, for simplicity" introduces the simplification; "Let us denote" sets up notation.

---

## §4.3 Cross-technology Precoding · p.5 · ¶2

**Quote:** "into Eq. 1, the precoded signal is computed as: 𝑿 = 𝑯b𝒛 −1 𝒁 (4) Obviously, the quality (i.e., SNR) of the signals received at the ZigBee devices reflects the precision of channel estimation. In other words, X-MIMO's performance is largely affected by the accuracy of 𝑯b𝒛 . As X-MIMO operates on commodity devices only using standard functions, 𝑯b𝒛 incorporates inevitable phase errors that are unique in X-MIMO. The error stems from three sources: (i) carrier frequency offset (CFO) between the ZigBee and X-MIMO. This is because X-MIMO locks its carrier frequency to the WiFi device (via phase-locked loop), not ZigBee. Therefore, CFO between ZigBee and WiFi persists. (ii) The initial phase offset between X-MIMO and ZigBee. Lastly, (iii) the jitter in the ZigBee ACK arrival time (< 0.1 𝜇𝑠 in our experiment) introduces an additional phase error. While these uncertainties may add up to a large phase error, interestingly, it has zero impact on the ZigBee signal quality. That is, X-MIMO is inherently immune to such uncertainties, which we prove via a rigorous derivation."

1. METHODOLOGY — precoded signal formula (Eq. 4) lead-in.
2. CLAIM — SNR reflects channel estimation precision.
3. INTERPRETATION — performance depends on accuracy of 𝑯̂𝑧.
4. PREMISE (CAUSE) — using only standard functions makes phase errors inevitable.
5. METHODOLOGY (CLAIM) — three sources of error; (i) CFO.
6. CAUSE — X-MIMO locks to WiFi, not ZigBee.
7. CONSEQUENCE — CFO persists.
8. PREMISE — (ii) initial phase offset.
9. PREMISE — (iii) ACK jitter (<0.1 μs).
10. CONCESSION (CLAIM) — uncertainties sum to large phase error.
11. CLAIM — yet zero impact on signal quality; promises rigorous proof.

**Shape:** METHODOLOGY → CLAIM → INTERPRETATION → PREMISE → METHODOLOGY → CAUSE → CONSEQUENCE → PREMISE → PREMISE → CONCESSION → CLAIM
**Persuasive move:** concede-then-overturn (immunity claim teased).
**Sentence flow notes:** "Obviously" frames the dependence; "In other words" restates; "As X-MIMO operates on commodity devices" supplies the cause for errors; "(i)…(ii)…Lastly, (iii)" enumerate sources; "While these uncertainties…interestingly" sets up the surprise; "That is" stamps the immunity claim.

---

## §4.3 Cross-technology Precoding · p.5 · ¶3

**Quote:** "We refer back to the example scenario in Figure 8 involving two ZigBees, where we first consider the case for ZigBee 1. Let us denote the jittered ACK (from ZigBee 1) reception time and the timing jitter from that time as 𝑡1 and 𝜏1 , respectively. We further indicate the CFO and the phase offset between X-MIMO and ZigBee 1 as Δ𝑓1 and 𝜃1 , respectively. Letting ∡𝑝𝑘 1 represent the total phase change incurred on ZigBee 1 channel estimation on the subcarrier 𝑘, this becomes:"

1. SCOPE — narrows derivation to two-ZigBee case starting with ZigBee 1.
2. DEFINITION — notation for ACK time and jitter.
3. DEFINITION — notation for CFO and phase offset.
4. DEFINITION — notation for total phase change ∠p₁ᵏ leading to Eq. 5.

**Shape:** SCOPE → DEFINITION → DEFINITION → DEFINITION
**Persuasive move:** notation-setup for the immunity proof.
**Sentence flow notes:** "Let us denote", "We further indicate", and "Letting" sequence notation introductions.

---

## §4.3 Cross-technology Precoding · p.5 · ¶4

**Quote:** "where 1 ≤ 𝑘 ≤ 64 (subcarrier index). We note that 𝑝𝑘, a complex value with the amplitude of 1, is embedded in the corresponding subcarriers of ℎ b11 and ℎ b12 (in Figure 8), as they are obtained from the same ACK sent by ZigBee 1. Similarly, the phase uncertainty for ZigBee 2 for subcarrier 𝑘 is ∡𝑝𝑘 2 , where this is included in ℎ b21 and ℎ b22 . By denoting phase shifts for ZigBee 1 and 2 in all subcarriers as ∡𝑝1 and ∡𝑝2 , the relationship between the estimated and the true channel can be represented as ℎ b11 = 𝑝1ℎ11 and ℎ b12 = 𝑝1ℎ12 (similar for ZigBee 2). Therefore,"

1. SCOPE (DEFINITION) — subcarrier index range.
2. METHODOLOGY (PREMISE) — 𝑝ᵏ has unit amplitude and is shared across ℎ̂₁₁,ℎ̂₁₂.
3. METHODOLOGY — analogous statement for ZigBee 2.
4. METHODOLOGY (DEFINITION) — relates estimated to true channels via 𝑝₁ multipliers.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → METHODOLOGY
**Persuasive move:** algebraic-factorisation-of-uncertainty.
**Sentence flow notes:** "We note that" introduces the embedding observation; "Similarly" mirrors for ZigBee 2; "By denoting" assembles the matrix relation; "Therefore" hands off to the next equation.

---

## §4.3 Cross-technology Precoding · p.5 · ¶5

**Quote:** "where 𝑯𝒛 is the ground truth ZigBee channel and 𝑷 = (cid:16) 𝑝 0 1 𝑝 0 2 (cid:17) . X-MIMO's precoded signal,𝑿, isdesigned to yield 𝒁 after passing through the estimated channel, 𝑯b𝒛 – i.e., 𝑯b𝒛𝑿 = 𝒁. Applying Eq. 6 and solving for 𝑿 we get"

1. DEFINITION — interprets 𝑯𝒛 and matrix 𝑷.
2. METHODOLOGY — design intent: 𝑯̂𝑧𝑿 = 𝒁.
3. METHODOLOGY — lead-in to Eq. 7.

**Shape:** DEFINITION → METHODOLOGY → METHODOLOGY
**Persuasive move:** derivation-step bridge.
**Sentence flow notes:** "X-MIMO's precoded signal, 𝑿, is designed to yield" makes intent explicit; "Applying Eq. 6 and solving" cues the next equation.

---

## §4.3 Cross-technology Precoding · p.6 · ¶1

**Quote:** "In reality, 𝑿 passes through the channel 𝑯𝒛 to reach ZigBee 1 and 2. Therefore, what is received by the ZigBee devices when X is the transmitted (precoded) signal, is:"

1. METHODOLOGY — bridges from design intent to actual channel pass-through.
2. METHODOLOGY — lead-in to equation describing what is received.

**Shape:** METHODOLOGY → METHODOLOGY
**Persuasive move:** ideal-vs-actual reconciliation.
**Sentence flow notes:** "In reality" pivots from design to reality; "Therefore" introduces the receive-side equation.

---

## §4.3 Cross-technology Precoding · p.6 · ¶2

**Quote:** "where we used Eq. 7. This indicates that received signals at the two ZigBee 1 and 2 are simply phase-shifted (by ∡𝑝−1 and ∡𝑝−1) versions of the target signal 𝒁. The phase-shifted signal does not impact on ZigBee reception – this is because the commodity ZigBee receivers only rely on the phase differences (i.e., the relative phase) between symbols for decoding [21]. Thus, the rotation of the entire signal is harmless. This effectively demonstrates that X-MIMO is indeed immune to the unique and inevitable uncertainties in the cross-technology channel estimation. The evaluation in sections 6.2 and 6.5 empirically validate the immunity to phase shifts and demonstrates high robustness of the cross-technology channel estimation under various channel conditions including LOS and non-LOS scenarios."

1. METHODOLOGY — cites Eq. 7 used in derivation.
2. INTERPRETATION — received signals are merely phase-shifted versions of target.
3. CLAIM (CAUSE) — phase shift harmless because ZigBee decodes from relative phase.
4. CONSEQUENCE — entire-signal rotation is harmless.
5. CLAIM — proven immunity to cross-tech channel uncertainties.
6. EVIDENCE — forward reference to §6.2/§6.5 evaluations.

**Shape:** METHODOLOGY → INTERPRETATION → CLAIM → CONSEQUENCE → CLAIM → EVIDENCE
**Persuasive move:** algebraic-immunity-then-empirical-promise.
**Sentence flow notes:** "This indicates that" interprets; "this is because" supplies the cause; "Thus" delivers the consequence; "This effectively demonstrates" stamps the proof; "The evaluation in sections 6.2 and 6.5" promises empirical validation.

---

## §4.3 Cross-technology Precoding · p.6 · ¶3 (Power Control)

**Quote:** "Power Control. X-MIMO is based on the principle of the implicit MU-MIMO, which requires the transmission power of the AP to be identical to that of the users. Correspondingly, X-MIMO sets the transmission power of the precoded ZigBee (8 subcarriers in X-MIMO) and ZigBee devices to the same level, using the iwconfig command (in Linux), without driver or firmware modification. Specifically, X-MIMO sets the transmission power of each antenna to be 8.45 dB higher than the transmission power of the ZigBee device. This is because only eight WiFi subcarriers (among a total of 56) corresponds to the ZigBee, occupying 8/56 of the WiFi signal. In other words, the power of the entire WiFi signal (emitted from each antenna) should be higher than the target ZigBee signal power by 8.45 dB (=10𝑙𝑜𝑔10(8/56)). Therefore WiFi AP equipped with two antennas should set up to overall power of 11.45 dB (twice of 8.45 dB). For instance, the default transmission power of CC2530 radio chip (ZigBee) is 4.5 dBm [1]. Then, the transmission power of each antenna at X-MIMO should be 12.95 dBm for the total transmission power of 15.95 dBm for two antennas. As 15.95 dBm is close to the default transmission power of typical WiFi at ∼ 17 dBm (e.g., AR9334), the impact of X-MIMO's power control to WiFi communication is insignificant."

1. PREMISE — implicit MU-MIMO requires identical AP/user TX power.
2. METHODOLOGY — X-MIMO matches power via iwconfig, no driver mod.
3. METHODOLOGY — per-antenna power set 8.45 dB higher than ZigBee.
4. CAUSE — only 8/56 subcarriers carry ZigBee.
5. INTERPRETATION — restates 8.45 dB derivation.
6. CONSEQUENCE — two-antenna AP totals 11.45 dB.
7. EXAMPLE — CC2530 default 4.5 dBm.
8. METHODOLOGY (EXAMPLE) — derives 12.95 dBm/antenna, 15.95 dBm total.
9. CLAIM (CONSEQUENCE) — close to typical 17 dBm WiFi default, so impact insignificant.

**Shape:** PREMISE → METHODOLOGY → METHODOLOGY → CAUSE → INTERPRETATION → CONSEQUENCE → EXAMPLE → METHODOLOGY → CLAIM
**Persuasive move:** arithmetic-justification-of-power-budget.
**Sentence flow notes:** "Correspondingly" links principle to operation; "Specifically" sharpens; "This is because" supplies the ratio reason; "In other words" restates; "Therefore" derives total power; "For instance" instantiates with CC2530; "As 15.95 dBm is close to" concludes negligibility.

---

## §5 Multi-stream CTC · p.6 · ¶1

**Quote:** "Multi-stream CTC is uniquely designed to transmit the precoded signal on a commodity WiFi AP (X-MIMO), by leveraging its 802.11n MIMO features and functionality – i.e., multiple antennas and multi-stream signal processing. Compared with the latest CTC designs, X-MIMO significantly improves the flexibility of signal manipulation from a single stream to multiple streams. Furthermore, X-MIMO incorporates spectral efficient emulation to avoid the spectral wastage in the state-of-the-art CTC, caused by the unused subcarriers."

1. METHODOLOGY (CLAIM) — multi-stream CTC built on 802.11n MIMO features.
2. CLAIM (CONTRAST) — improves over latest CTC: single → multi-stream.
3. CLAIM — adds spectral-efficient emulation to fix bandwidth waste.

**Shape:** METHODOLOGY → CLAIM → CLAIM
**Persuasive move:** dual-improvement-over-state-of-the-art-CTC.
**Sentence flow notes:** "Compared with the latest CTC designs" frames the contrast; "Furthermore" stacks the second improvement.

---

## §5 Multi-stream CTC · p.6 · ¶2

**Quote:** "Multi-stream CTC is achieved by reversing each step of 802.11n transmission to find an appropriate 802.11n payload such that the corresponding transmitted signal is the precoded signal, or equivalently using an 802.11n signal to emulate the precoded signal. As illustrated in Figure 9, our first step is to reverse the Cyclic Shift Diversity (CSD), which is inserted to prevent unintended beamforming via multiplying the QAM mapped signal on the subcarrier k, stream i by a complex value 𝑐 𝑖 𝑘 (= 𝑒 𝑗𝜋 𝑘 m 4 o ( d 𝑖− 8 1 (𝑖 ) −1) , for i ≥ 2, 1 for all other i) [2]. Specifically, reversing CSD is simply performed by multiplying the precoded signal on the subcarrier k stream i by the conjugate of 𝑐𝑘 to obtain the corresponding QAM mapped signal, denoted by 𝑞𝑘. To approximate 𝑞𝑘 with minimum error, the closest QAM sample in the constellation diagram is selected, yielding a bit sequence for each signal stream. As the 'Reverse CSD+QAM mapper' in the Figure 9 illustrates, the red arrow is computed from the gray arrow (precoded signal) by reversing the CSD. The red arrow is then approximated by the red QAM sample, from which the bit sequence '100111' is generated."

1. METHODOLOGY (CLAIM) — multi-stream CTC obtained by reversing 802.11n TX chain.
2. METHODOLOGY (DEFINITION) — first step: reverse CSD; defines CSD formula.
3. METHODOLOGY — operational rule: multiply by conjugate of cₖ.
4. METHODOLOGY — approximate qₖ by closest QAM sample, yielding bits.
5. EXAMPLE — figure example: gray→red arrow via CSD reversal.
6. EXAMPLE — red sample maps to bit sequence '100111'.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → EXAMPLE → EXAMPLE
**Persuasive move:** reverse-engineer-the-PHY-pipeline.
**Sentence flow notes:** "As illustrated in Figure 9" anchors the first step; "Specifically" details mechanics; "To approximate" describes the QAM choice; "As the 'Reverse CSD+QAM mapper'…illustrates" instantiates the example.

---

## §5 Multi-stream CTC · p.6 · ¶3

**Quote:** "Since the interleaver shuffles the bit sequence deterministically, reversing the interleaver is performed via rearranging each bit sequence accordingly. The stream parser, cutting the serial encoded bits into multiple blocks and further feeding to multiple bit sequences, is reversed by assembling bits into the blocks and placed in the encoded bits alternatively. For instance, as 'Reverse Interleaver' in the Figure 9 illustrates, indices 3,4 and 1,5 in the red bit sequence '100111' are switched to yield '101011'. Then, every three bits in the red sequence are assembled into one block and further placed at the odd index while the blocks from the blue sequence are placed at the even index in the encoded bits. Consequently, the serial encoded bits '101010011101' are generated from the two streams."

1. METHODOLOGY (CAUSE) — deterministic interleaver implies invertibility by rearrangement.
2. METHODOLOGY — stream parser reversed by alternating block placement.
3. EXAMPLE — concrete bit-swap '100111' → '101011'.
4. EXAMPLE — block-alternation rule worked out.
5. CONSEQUENCE — produces serial encoded bits '101010011101'.

**Shape:** METHODOLOGY → METHODOLOGY → EXAMPLE → EXAMPLE → CONSEQUENCE
**Persuasive move:** worked-example-of-inverse-PHY.
**Sentence flow notes:** "Since" supplies the why-invertible cause; "For instance" launches the worked example; "Then" continues; "Consequently" lands the resulting bitstream.

---

## §5 Multi-stream CTC · p.6 · ¶4

**Quote:** "To finally obtain the payload for multi-stream CTC, we need to reverse the encoding and scrambling, which are the first two steps in 802.11n transmission. Since the principle of encoder and scrambler are equivalent for 802.11g and 802.11n, we adopt the design in WEBee [36] (designed for 802.11g) to convert the encoded bits to the payload, except that the scrambler seed, a 7-bits sequence controlling the scrambler, cannot be manually set in 802.11n. To resolve this issue, we take advantage of the predictable seed sequences (increments by one between two packets) in many commodity WiFi chips (e.g., AR9334 and AR9380) and the fixed initial scrambling seed (seed index 71 out of 128). Consequently, given the count of the transmitted WiFi packet, the scrambling seed in 802.11n is easily tracked 2. Upon finding the scrambling seed, the scrambler can be reversed to yield the payload for multi-stream CTC."

1. METHODOLOGY (SCOPE) — last steps: reverse encoding and scrambling.
2. METHODOLOGY (BRIDGE) — adopt WEBee design due to 802.11g/n equivalence; flags the scrambler-seed obstacle.
3. METHODOLOGY (CLAIM) — exploits predictable seed sequence and fixed initial seed.
4. CONSEQUENCE — given packet count, seed is trackable.
5. CONSEQUENCE — once seed known, scrambler invertible.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → CONSEQUENCE → CONSEQUENCE
**Persuasive move:** obstacle-then-vendor-feature-leverage.
**Sentence flow notes:** "Since the principle of encoder and scrambler are equivalent" justifies reuse; "except that" raises the obstacle; "To resolve this issue" introduces the fix; "Consequently" delivers tractability; "Upon finding" closes the loop.

---

## §5 Multi-stream CTC · p.7 · ¶1

**Quote:** "For a further intuition on the entire design, Figure 10 demonstrates the ZigBee signals at the two receivers in comparison to the ground truth. The slight difference indicates our design inherits the limitation of the state-of-the-art CTC – determined by the finite constellation points, the precision is degraded by emulation errors. Despite the inevitable error, the successful operation of X-MIMO is promised by the high redundancy in ZigBee's direct sequence spread spectrum (DSSS)."

1. EVIDENCE — Figure 10 compares received signals against ground truth.
2. CONCESSION (CAUSE) — slight differences reflect finite-constellation emulation error.
3. CLAIM — DSSS redundancy nonetheless guarantees X-MIMO operation.

**Shape:** EVIDENCE → CONCESSION → CLAIM
**Persuasive move:** acknowledge-error-then-cover-via-DSSS-redundancy.
**Sentence flow notes:** "The slight difference indicates" concedes; "Despite the inevitable error" pivots to the redundancy defence.

---

## §5.1 Spectral Efficient Emulation · p.7 · ¶1

**Quote:** "Due to the asymmetric bandwidth (20 MHz in WiFi and 2 MHz in ZigBee), the large spectral wastage (18 MHz) degrades the spectral efficiency. To resolve this issue, we select the weakest QAM samples in the constellation diagram (in \"Reverse CSD+QAM mapper) to suppress the power allocated to the subcarriers ( non-overlapped with ZigBee), thereby opening the unused frequencies for other wireless devices. As depicted in Figure 11, we discover that if the six bits allocated to this subcarrier are 'X10X10', where 'X' is an arbitrary '0' or '1', the generated constellation QAM sample (using QAM 64) are weakest. Therefore, we enforce the six bits on each non-overlapped subcarrier to follow the form of 'X10X10' in our Multi-stream CTC to minimize the energy leakage to the unused frequencies."

1. PREMISE — bandwidth asymmetry yields 18 MHz wastage.
2. METHODOLOGY (CLAIM) — pick weakest QAM samples to suppress unused subcarriers.
3. EVIDENCE (INTERPRETATION) — discovered pattern 'X10X10' minimises QAM-64 sample power.
4. METHODOLOGY (CONSEQUENCE) — enforce 'X10X10' on non-overlapped subcarriers to cap leakage.

**Shape:** PREMISE → METHODOLOGY → EVIDENCE → METHODOLOGY
**Persuasive move:** constellation-trick-to-recycle-bandwidth.
**Sentence flow notes:** "To resolve this issue" introduces the trick; "As depicted in Figure 11, we discover that" presents the pattern; "Therefore" enforces the rule.

---

## §5.1 Spectral Efficient Emulation · p.7 · ¶2

**Quote:** "Figure 12 shows an example of the power spectral density (denoted by Power Spec.) associated with the waterfall of the 802.11n packets, which emulate precoded signals on two ZigBee channels (±5 MHz apart from the center frequency). As demonstrated in Figure 12, we discover that the non-overlapped subcarriers' power (suppressed by emulation) is 15.68 dB less than the peak subcarriers' power, validating the effectiveness of our spectral efficient emulation. Although the pilot subcarriers (represented by the high power of four lines and peaks in Figures 12) are uncontrollable (thus, not suppressed), their impacts are negligible (demonstrated in Section 6.4)."

1. EVIDENCE (EXAMPLE) — Figure 12 example showing PSD and waterfall.
2. EVIDENCE (CLAIM) — 15.68 dB suppression validates effectiveness.
3. CONCESSION (CLAIM) — pilot subcarriers uncontrollable but impacts negligible (forward ref §6.4).

**Shape:** EVIDENCE → EVIDENCE → CONCESSION
**Persuasive move:** measurement-validates-with-honest-caveat.
**Sentence flow notes:** "As demonstrated in Figure 12, we discover that" delivers the validation number; "Although" raises and defuses the pilot-subcarrier caveat.

---

## §6 Evaluation · p.7 · ¶1

**Quote:** "This section discusses implementation of X-MIMO on commodity devices and the performance analysis under practical scenarios."

1. SCOPE — section preview.

**Shape:** SCOPE
**Persuasive move:** roadmap-sentence.
**Sentence flow notes:** Single-sentence section preview.

---

## §6.1 Implementation · p.7 · ¶1

**Quote:** "Figure 13 illustrates our implementation setup with two TP-link TL-WDR4300 wireless routers and three Telosb nodes. A router is running X-MIMO while the other operates as a common WiFi device, and the TelosB nodes are ZigBee nodes. The details of the X-MIMO implementation3 are as follows: X-MIMO modifies the ath9k-based Atheros CSI tool [56] to support the large maximum transmission unit (MTU) for emulating long ZigBee streams. We set another TP-link TL-WDR4300 wireless router to inject customized WiFi fragmented packets. To implement multi-stream emulation on top of 802.11n physical layer, we track the transmission of each WiFi packet ever since the WiFi is initialized on the TP-link WDR4300 router in order to track the scrambler seed. In our experiments, we upload these customized packets to the wireless router and inject them via lorcon. Since tracking the scrambler seed is just one step from system initialization while the TP-link WDR4300 router has a 560 MHz CPU and 128 MB RAM, the overhead is negligible."

1. METHODOLOGY (EXAMPLE) — testbed: 2 routers + 3 TelosB.
2. METHODOLOGY — role assignment of devices.
3. BRIDGE — handoff into detail list.
4. METHODOLOGY — modifies ath9k CSI tool for large MTU.
5. METHODOLOGY — second router injects custom WiFi fragments.
6. METHODOLOGY — tracks scrambler seed from initialisation.
7. METHODOLOGY — uses lorcon for injection.
8. CLAIM — seed tracking overhead negligible on WDR4300 hardware.

**Shape:** METHODOLOGY → METHODOLOGY → BRIDGE → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CLAIM
**Persuasive move:** reproducibility-via-commodity-only.
**Sentence flow notes:** "The details of the X-MIMO implementation are as follows" bridges to detail; "Since tracking the scrambler seed is just one step" defuses the overhead concern.

---

## §6.1 Implementation · p.8 · ¶1

**Quote:** "To evaluate the performance of X-MIMO, we modify TinyOS to access the raw received bits at two ZigBee nodes. We disable the CRC check via setting the register \"MODEMCTRL0.AUTOCRC\" [6], which is commonly supported on all TI serial radio chips and use Printf interface (in TinyOs library) to print all the received raw symbols. We note that disabling CRC is only for analysis purposes; i.e., obtaining the symbol error rate and calculating the corresponding throughput. In addition to the above implementations, we use USRPs as ZigBee nodes to test our ZigBee channel estimation and the performance of X-MIMO under different settings. We modify the 802.15.4 implementation [4] on GNURadio to further check the detailed ZigBee symbol error rate at USRP."

1. METHODOLOGY — TinyOS modified to expose raw received bits.
2. METHODOLOGY — CRC disabled via register write; Printf interface used.
3. SCOPE (CONCESSION) — CRC disable is for analysis only.
4. METHODOLOGY — USRPs added as ZigBee nodes for in-depth tests.
5. METHODOLOGY — GNURadio 802.15.4 modified for SER analysis.

**Shape:** METHODOLOGY → METHODOLOGY → SCOPE → METHODOLOGY → METHODOLOGY
**Persuasive move:** instrumentation-disclosure-with-scope-fence.
**Sentence flow notes:** "We note that disabling CRC is only for analysis purposes" fences scope; "In addition to the above implementations" stacks the USRP track.

---

## §6.2 X-MIMO Performance · p.8 · ¶1

**Quote:** "In this experiment, we evaluate the performance of X-MIMO under two TelosB nodes (commodity ZigBee) while X-MIMO with more ZigBee receivers is evaluated in Section 6.3. Figure 14 depicts the two scenarios in our evaluation: hallway and office. To evaluate the performance of X-MIMO in the line-of-sight (LOS) scenario, we deploy the X-MIMO device and ZigBee devices at different distances in the hallway. To evaluate X-MIMO in the non-line-of-sight (NLOS) scenario, we deploy ZigBee devices at four different positions in the office, which is shown in Figure 15. In this experiment, the transmission power of the ZigBee device is 0 dBm and the transmission power of the WiFi fragments is the default 17 dBm. We set the Tx power of X-MIMO according to our design in Section 4.3. The symbol error rate and throughput of X-MIMO are evaluated and compared with WEBee [36]."

1. SCOPE — two TelosB nodes here; more in §6.3.
2. METHODOLOGY — hallway + office scenarios.
3. METHODOLOGY — LOS deployment description.
4. METHODOLOGY — NLOS deployment description.
5. METHODOLOGY — TX power settings.
6. METHODOLOGY — X-MIMO power per §4.3 design.
7. SCOPE — metrics + WEBee baseline.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → SCOPE
**Persuasive move:** experiment-design-disclosure.
**Sentence flow notes:** "To evaluate the performance of X-MIMO in the line-of-sight (LOS) scenario" and "To evaluate X-MIMO in the non-line-of-sight (NLOS) scenario" pair the scenarios; closing sentence frames the comparison.

---

## §6.2 X-MIMO Performance · p.8 · ¶2 (SER)

**Quote:** "Symbol Error Rate (SER). The ZigBee symbol error rate of X-MIMO and WEBee in LOS and NLOS scenarios are shown in Figure 16. In our experiment, WEBee transmits to the two ZigBee devices alternatively and therefore its SER is the average of the SER at two ZigBee devices. The SER of X-MIMO for two ZigBee devices at position 1 is 1% and 27%, exhibiting a significant imbalance. This is because the channel from X-MIMO to ZigBee 2 is so weak that the signal for ZigBee 1 keeps dominating the ZigBee device 2. The SER of X-MIMO at the other three positions are (9.1%, 7.2%), (9%, 8.5%) and (9.4%, 8.6%) while the SER of WEBee is 7%, 10.1% and 10.2, respectively."

1. EVIDENCE (BRIDGE) — refers to Figure 16 for SER.
2. METHODOLOGY (PREMISE) — WEBee transmits alternately; SER averaged.
3. EVIDENCE (CONCESSION) — position 1 shows imbalanced SER (1% vs. 27%).
4. CAUSE (INTERPRETATION) — weak ZigBee 2 channel allows ZigBee 1 signal to dominate.
5. EVIDENCE — SER at the other three positions vs. WEBee.

**Shape:** EVIDENCE → METHODOLOGY → EVIDENCE → CAUSE → EVIDENCE
**Persuasive move:** numbers-with-honest-failure-case.
**Sentence flow notes:** "In our experiment" sets up baseline metric; "This is because" supplies the imbalance cause; the closing sentence stacks comparative numbers.

---

## §6.2 X-MIMO Performance · p.8 · ¶3

**Quote:** "In the LOS scenario, two ZigBee devices are placed at distances of 3 - 21 m away from the X-MIMO. For the distance of 12 and 18 m, SER of X-MIMO for two ZigBee devices is less than WEBee. Despite the small imbalance of SER at two ZigBee devices (6.8% at ZigBee 1 and 2.6% at ZigBee 2), the average SER of X-MIMO at 15 m distance is 4.7%, which is still less than the SER of WEBee (5.4%). As the distance between X-MIMO and ZigBee devices increases, the SNR of the overlapped ZigBee signal gets weaker while the SER of X-MIMO does not drop too much. X-MIMO still achieves ≤ 9% SER with ≤ 2% error at 21 m."

1. METHODOLOGY — LOS distance range 3–21 m.
2. EVIDENCE — at 12 and 18 m, X-MIMO beats WEBee.
3. EVIDENCE (CONCESSION) — slight imbalance at 15 m; still beats WEBee average.
4. INTERPRETATION — SER not strongly degraded by distance.
5. EVIDENCE — ≤9% SER (≤2% error) at 21 m.

**Shape:** METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION → EVIDENCE
**Persuasive move:** distance-robustness-stack.
**Sentence flow notes:** "For the distance of 12 and 18 m" foregrounds favourable distances; "Despite" concedes imbalance; "As the distance…increases" supplies trend interpretation.

---

## §6.2 X-MIMO Performance · p.8 · ¶4 (Throughput)

**Quote:** "Throughput. Figure 17 demonstrates the throughput of X-MIMO obtained from the SER. In the NLoS scenario, X-MIMO achieves 432, 466, 456, and 455 Kbps at four positions, outperforming WEBee by ×1.85 − 2.03. In the LOS scenario, the throughput of X-MIMO exhibits a more stable trend. At the distance of ≤ 12 meters, the throughput of X-MIMO is greater than 490 Kbps, which is almost doubling the throughput of legacy ZigBee. For the distance of 15, 18, and 21 meters, the throughput of X-MIMO (465, 471, and 455 Kbps) are almost twice of WEBee. This result shows that given the two-stream precoding in X-MIMO, the throughput of communication for IoT is significantly improved by the number of antennas."

1. EVIDENCE (BRIDGE) — Figure 17 reports throughput.
2. EVIDENCE — NLoS: 432–466 Kbps, 1.85–2.03× WEBee.
3. INTERPRETATION — LOS more stable.
4. EVIDENCE — ≤12 m: >490 Kbps, near double legacy ZigBee.
5. EVIDENCE — 15/18/21 m: 455–471 Kbps, ~2× WEBee.
6. INTERPRETATION (CLAIM) — two-stream precoding scales throughput with antennas.

**Shape:** EVIDENCE → EVIDENCE → INTERPRETATION → EVIDENCE → EVIDENCE → INTERPRETATION
**Persuasive move:** throughput-stack-vs-baseline.
**Sentence flow notes:** "In the NLoS scenario" / "In the LOS scenario" pair scenarios; "This result shows that" wraps with an interpretation.

---

## §6.3 Scalability of X-MIMO · p.8 · ¶1

**Quote:** "To demonstrate the scalability of X-MIMO, we extend the implementation of two-streams X-MIMO to support two parallel ZigBee channels and three streams (three ZigBee receivers)."

1. SCOPE — preview of two scalability extensions.

**Shape:** SCOPE
**Persuasive move:** roadmap-sentence.
**Sentence flow notes:** Single-sentence preview.

---

## §6.3 Scalability of X-MIMO · p.8 · ¶2 (Parallel X-MIMO)

**Quote:** "Parallel X-MIMO. The settings of parallel X-MIMO are illustrated in Figure 18(a): (i) X-MIMO is implemented on TP-link WDR4300 wireless router, which works on 2.46 GHz to cover ZigBee channel 21 (2.455 GHz) and 23 (2.465 GHz). Despite 2.46 GHz is not the center frequency of any WiFi channel, the ath9k WiFi driver and the commodity WiFi device allow us to set the center frequency to be an arbitrary value (1 MHz granularity) via controlling register 'channelSel' in the function 'ar9003_hw_set_channel'. We control the emulation process to transmit the precoded signals on subcarriers 13 - 19 and 45 - 52 to support two-stream MU-MIMO on ZigBee channels 21 and 23. (ii) Two ZigBee devices (USRPs) are deployed on each ZigBee channel. The distance between X-MIMO and the four ZigBee devices (Rx1 - Rx4) is 3 meters. The symbol error rate of parallel X-MIMO is demonstrated in Figure 18(b), showing that the SER of four devices is less than 1.7%. Hence, with the two-channel parallel X-MIMO, we enable MU-MIMO for four ZigBee devices with an aggregated throughput of 983 Kbps."

1. METHODOLOGY — X-MIMO set to 2.46 GHz covering ZigBee ch.21/23.
2. METHODOLOGY (CONCESSION) — non-standard centre frequency enabled by ath9k register.
3. METHODOLOGY — uses subcarriers 13–19 and 45–52 for two-channel MU-MIMO.
4. METHODOLOGY — two USRPs per ZigBee channel.
5. METHODOLOGY — 3 m TX-RX distance.
6. EVIDENCE — Figure 18(b): SER <1.7% across four devices.
7. CONSEQUENCE (CLAIM) — 983 Kbps aggregated throughput for four devices.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE → CONSEQUENCE
**Persuasive move:** numeric-scale-up demo.
**Sentence flow notes:** "Despite 2.46 GHz is not the center frequency of any WiFi channel" registers a caveat then a fix; "Hence" lands the aggregated-throughput conclusion.

---

## §6.3 Scalability of X-MIMO · p.9 · ¶1 (Three-stream X-MIMO)

**Quote:** "Three-stream X-MIMO. In this experiment, the performance of three-streams X-MIMO is evaluated on the TP-link 4300 wireless router with three antennas. Specifically, the TP-link WDR4300 router is equipped with two WNICs, i.e. AR9344 (2.4 GHz) and AR9580 (5 GHz). The AR9344 WNIC only supports up to two antennas while the AR9580 WNIC supports three antennas. Then, we implement X-MIMO on AR9580 (5 GHz) as a workaround to demonstrate the performance of three-streams X-MIMO. Specifically, as Figure 19(a) illustrates, we set X-MIMO (AR9580 WNIC on TP-link WDR4300) to work on WiFi channel 44 (i.e., 5.22 GHz) and deploy three USRPs, running ZigBee module, on 5.225 GHz as MU-MIMO receivers. The distance between X-MIMO and three ZigBee receivers (Rx1 - Rx3) is 2 meters. The performance of three-streams X-MIMO is shown in Figure 19(b). The symbol error rate at three ZigBee receivers is less than 7%, while the average SER is 6.1%."

1. SCOPE — three-stream evaluation on the three-antenna router.
2. METHODOLOGY — two WNICs onboard.
3. PREMISE — AR9344 ≤2 antennas; AR9580 supports 3.
4. METHODOLOGY (CONCESSION) — workaround: implement on 5 GHz AR9580.
5. METHODOLOGY — WiFi channel 44; receivers at 5.225 GHz.
6. METHODOLOGY — 2 m TX-RX distance.
7. EVIDENCE (BRIDGE) — Figure 19(b) carries results.
8. EVIDENCE — SER <7%, average 6.1%.

**Shape:** SCOPE → METHODOLOGY → PREMISE → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE → EVIDENCE
**Persuasive move:** scale-to-three-streams via 5 GHz workaround.
**Sentence flow notes:** "Then, we implement X-MIMO on AR9580 (5 GHz) as a workaround" labels the workaround; "Specifically, as Figure 19(a) illustrates" anchors the setup.

---

## §6.4 X-MIMO Spectral Efficiency · p.9 · ¶1

**Quote:** "We evaluate our proposed spectral efficient emulation to show the improvement in the spectral efficiency. Specifically, two ZigBee devices (Tx and Rx), working on X-MIMO's non-overlapping frequencies, are deployed 4.5 meters apart while a USRP (X-MIMO) is placed 2.35 meters away from ZigBee Rx. The Tx power of ZigBee and USRP is set to be 5 dBm. The spectral efficient emulation in practice is shown in Figure 20(a), where the leakage to the non-overlapped frequencies is -80 dBm. Such leakage is weaker than the default CCA threshold on the typical commodity ZigBee device such as CC2420 (i.e., -71 dBm)[6], thereby incurring zero impact on the ZigBee Rx's communication. The effectiveness of spectral reuse is shown in Figure 20(b) through the packet reception ratio under three wireless scenarios: (i) The USRP transmits regular WiFi packets, which overlap with ZigBee channel, (ii) The USRP (X-MIMO) transmits WiFi packets with spectral efficient emulation in the same frequency band, and (iii) The USRP does not transmit any wireless signals. While ZigBee Rx receives 53.92% packets due to severe interference under (i) WiFi scenario, where ZigBee Rx shows 99.28% and 99.50% of packet reception rate, under (ii) X-MIMO and (iii) none scenarios, respectively. This experiment shows that X-MIMO with spectral efficient emulation is able to avoid the spectral wastage on the non-overlapped subcarriers. Thus, the maximum spectral efficiency X-MIMO can achieve is to 0.36 bits/s/Hz, 3× of the legacy ZigBee, and 28.8× of the state-of-the-art design WEBee[36]."

1. SCOPE — purpose of the experiment.
2. METHODOLOGY — two ZigBees 4.5 m apart; USRP 2.35 m from Rx.
3. METHODOLOGY — TX power 5 dBm.
4. EVIDENCE — Figure 20(a): leakage at -80 dBm.
5. EVIDENCE (CLAIM) — -80 dBm below CC2420 CCA threshold (-71 dBm) ⇒ zero impact.
6. METHODOLOGY (SCOPE) — three scenarios for PRR comparison.
7. EVIDENCE — PRR: 53.92% (WiFi), 99.28% (X-MIMO), 99.50% (none).
8. INTERPRETATION — confirms emulation avoids spectral wastage.
9. CLAIM (CONSEQUENCE) — 0.36 bits/s/Hz, 3× legacy ZigBee, 28.8× WEBee.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → EVIDENCE → EVIDENCE → METHODOLOGY → EVIDENCE → INTERPRETATION → CLAIM
**Persuasive move:** spectral-reuse-pay-off quantified.
**Sentence flow notes:** "Such leakage is weaker than" compares to a hardware threshold; "While ZigBee Rx receives 53.92% packets…" couples disadvantage of (i) with advantage of (ii) and (iii); "Thus" lands the multiplicative gain.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.9 · ¶1

**Quote:** "In this Section, we measure and show the real traffic associated with and the performance of cross-technology channel estimation."

1. SCOPE — section preview.

**Shape:** SCOPE
**Persuasive move:** roadmap-sentence.
**Sentence flow notes:** Single-sentence preview.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.9 · ¶2 (Timing Control in Practice)

**Quote:** "Timing Control in Practice. We deploy one USRP to measure the signal generated at X-MIMO, ZigBee device, and the WiFi device. We set the fragmentation threshold to be 1898 Bytes (= 146 WiFi MCS 3 symbols), which consists of 1 symbol for MAC header, 120 symbols for emulating ZigBee ACK request, and 25 padded symbols. Then, the duration of Fragment 1 is 36 (preamble) + 146×4 (146 symbols in the payload) = 620 𝜇𝑠. Figure 21 depicts the traffic of cross-technology channel estimation consists of two WiFi fragments, the replied WiFi and ZigBee ACKs. After the WiFi device transmits the Fragment 1 (620 𝜇𝑠), the X-MIMO responses with a WiFi ACK, followed by Fragment 2, which collides with the replied ZigBee ACK (359 𝜇𝑠). Given that the inter-fragment interval is 60 𝜇𝑠, the whole time consumed for cross-technology channel estimation is 620 + 60 + 32 + 359 = 1071 𝜇𝑠, which is is negligible compared to the long coherence time for narrow-band ZigBee signal (over seconds[40, 49])."

1. METHODOLOGY — USRP placed to record signals.
2. METHODOLOGY — fragmentation threshold 1898 B = 146 symbols decomposed.
3. METHODOLOGY — Fragment 1 duration 620 μs derivation.
4. METHODOLOGY (EVIDENCE) — Figure 21 shows the timing trace.
5. METHODOLOGY — handshake sequence and collision with ZigBee ACK.
6. CONSEQUENCE (CLAIM) — total 1071 μs estimation, negligible vs. seconds coherence.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CONSEQUENCE
**Persuasive move:** measured-cost-versus-coherence margin.
**Sentence flow notes:** "Then, the duration of Fragment 1 is" sums to 620 μs; "After the WiFi device transmits the Fragment 1" sequences the handshake; "Given that" leads to the negligibility claim.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.10 · ¶1 (Cross-tech. Channel Estimation Precision)

**Quote:** "Cross-tech. Channel Estimation Precision. We use two USRPs in this experiment, where USRP 1 works as X-MIMO, and USRP 2 transmits customized WiFi fragments on antenna 1 and works as a ZigBee device on antenna 2. The HT-LTF field in the transmitted WiFi fragments and the ZigBee signal are perfectly aligned. To get the ground truth of ZigBee channels, we also control the USRP 2 to transmit ZigBee signal without the interference of WiFi fragments to USRP 1, where the ground truth ZigBee channel is obtained by comparing the first 3.2 𝜇s received ZigBee signal and the first 3.2 𝜇s transmitted ZigBee signal. The Tx power of WiFi fragments (USRP) is set to be 17 dBm and the Tx power of ZigBee (USRP) is set to be 0 dBm. We deploy the two USRPs in the same LOS and NLOS scenario as Figure 14 illustrates and compare the estimated ZigBee channel with ground truth."

1. METHODOLOGY — USRP 1 = X-MIMO; USRP 2 = dual-purpose TX.
2. METHODOLOGY — HT-LTF and ZigBee perfectly aligned.
3. METHODOLOGY — ground-truth obtained via clean ZigBee transmission.
4. METHODOLOGY — TX power settings.
5. METHODOLOGY — reuses LOS/NLOS scenarios from Figure 14.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY
**Persuasive move:** clean-setup-for-ground-truth-comparison.
**Sentence flow notes:** "To get the ground truth of ZigBee channels" introduces the calibration step; the rest enumerates standard experimental parameters.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.10 · ¶2

**Quote:** "In our experiment, the position of the WiFi device (transmits customized fragments) does not change with the ZigBee devices, thus leading to the constant amplitude of the WiFi channel. However, the ZigBee signal strength varies with its position. For example, Figure 22 shows two CSI values (with ZigBee overlapped) collected from the ZigBee 6 and 21 meters away from X-MIMO. The ZigBee signal at 21 meters is much weaker than the ZigBee signal at 6 meters."

1. METHODOLOGY (PREMISE) — WiFi device position fixed → constant WiFi-channel amplitude.
2. CONTRAST — ZigBee signal strength varies with position.
3. EVIDENCE (EXAMPLE) — Figure 22 contrasts 6 m and 21 m CSI.
4. INTERPRETATION — ZigBee at 21 m much weaker than at 6 m.

**Shape:** METHODOLOGY → CONTRAST → EVIDENCE → INTERPRETATION
**Persuasive move:** isolate-variable-then-show-effect.
**Sentence flow notes:** "However" contrasts the variable; "For example" supplies the illustration; the closing sentence delivers the qualitative read.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.10 · ¶3 (Method)

**Quote:** "Method. The absolute phase of the estimated ZigBee channel is affected by the hardware uncertainty, resulting in a time-variant estimation compared to the ground truth. Hence, in this experiment, we use the relative phase between two ZigBee channels as the metric to check the precision of the phase of the estimated ZigBee channel. Since the phase of two estimated ZigBee channels is affected simultaneously by the hardware uncertainty as described in Section 4.3, Eq. 6 indicates that the relative phase between two estimated ZigBee channels are immune to the hardware uncertainty. Specifically, the relative phase between the estimated channel 𝑝1ℎ11 and 𝑝1ℎ12 is identical with ∡(ℎ11,ℎ12). Thus, the relative phase is kept the same within channel coherent time."

1. PREMISE — absolute phase corrupted by hardware uncertainty.
2. METHODOLOGY (CONSEQUENCE) — therefore use relative phase as the metric.
3. INTERPRETATION (CAUSE) — Eq. 6 implies relative phase immune to uncertainty.
4. EXAMPLE — concrete relative-phase identity.
5. CONSEQUENCE — relative phase preserved within coherence time.

**Shape:** PREMISE → METHODOLOGY → INTERPRETATION → EXAMPLE → CONSEQUENCE
**Persuasive move:** metric-design-derived-from-immunity-theorem.
**Sentence flow notes:** "Hence" elevates premise to metric choice; "Since…Eq. 6 indicates" anchors in the prior derivation; "Specifically" instantiates; "Thus" lands the consequence.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.10 · ¶4

**Quote:** "To evaluate the precision of the amplitude of the estimated ZigBee channel, we utilize the amplitude ratio between two estimated channels as the metric. Specifically, the amplitude ratio between the estimated channel 𝑝1ℎ11 and 𝑝1ℎ12 is |ℎ11|/|ℎ12|, which removes the influence of hardware uncertainty 𝑝1 . Hence, the amplitude ratio is an indicator to check the precision of the estimated ZigBee channel, in terms of amplitude."

1. METHODOLOGY — amplitude ratio is the metric for amplitude precision.
2. INTERPRETATION — ratio cancels out hardware uncertainty.
3. CONSEQUENCE — ratio is a valid amplitude precision indicator.

**Shape:** METHODOLOGY → INTERPRETATION → CONSEQUENCE
**Persuasive move:** parallel-metric-design via ratio.
**Sentence flow notes:** "Specifically" elaborates the formula; "Hence" closes with the indicator claim.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.10 · ¶5 (Results)

**Quote:** "Results. Figure 23(a) illustrates the precision of the relative phase between two estimated ZigBee channels at four positions in the office. The error in the relative phase of the estimated ZigBee channel is ≤ 0.013 rad, with the maximum standard variance of 0.04 rad, indicating that the phase of the estimated ZigBee CSI is precise in the NLoS scenario. Figure 23(b) illustrates the error of amplitude ratio compared to the ground truth. The error in the amplitude ratio of the estimated ZigBee channel is ≤ 0.052, with the maximum standard variance of 0.07. Since both relative phase and amplitude ratio of the estimated channel are precise, our ZigBee channel estimation is precise in the NLoS scenario."

1. EVIDENCE (BRIDGE) — Figure 23(a) for NLoS relative phase.
2. EVIDENCE (INTERPRETATION) — ≤0.013 rad error, max stdev 0.04 rad ⇒ precise phase.
3. EVIDENCE (BRIDGE) — Figure 23(b) for amplitude ratio.
4. EVIDENCE — ≤0.052 error, stdev 0.07.
5. CLAIM — precise NLoS channel estimation overall.

**Shape:** EVIDENCE → EVIDENCE → EVIDENCE → EVIDENCE → CLAIM
**Persuasive move:** dual-metric-validation-NLoS.
**Sentence flow notes:** "Since both relative phase and amplitude ratio of the estimated channel are precise" combines the two readings into one verdict.

---

## §6.5 Cross-tech. Channel Estimation in Practice · p.10 · ¶6

**Quote:** "In the LOS scenario, we deploy the ZigBee (USRP) 3 - 21 meters away from X-MIMO and the results are shown in Figure 24. As the distance between ZigBee and X-MIMO increases, the error in the relative phase increases to up 0.11 rad at 21 meters because of the signal strength drop. Despite the SNR drops due to the distance, the error in the amplitude ratio is less than 0.035 and the maximum variance of this error is 0.16. Although the precision of ZigBee channel estimation in the LOS scenario is worse than that in the NLOS scenario, the errors in both phase and amplitude are still very small and negligible."

1. METHODOLOGY (EVIDENCE) — LOS deployment 3–21 m; results in Figure 24.
2. EVIDENCE (CAUSE) — phase error rises to 0.11 rad at 21 m due to signal drop.
3. EVIDENCE (CONCESSION) — amplitude ratio error ≤0.035 despite SNR drop.
4. CONCESSION (CLAIM) — LOS worse than NLoS but errors still negligible.

**Shape:** METHODOLOGY → EVIDENCE → EVIDENCE → CONCESSION
**Persuasive move:** distance-degradation-conceded-but-bounded.
**Sentence flow notes:** "As the distance…increases" gives the trend; "Despite" concedes SNR drop; "Although" frames the comparative concession and dismissal.

---

## §6.6 Obtaining WiFi-ZigBee Mixed Signal · p.10 · ¶1

**Quote:** "We manipulate the payload of WiFi fragmented 1 to trigger the ZigBee ACK, which overlaps with the WiFi fragmented 2. In practice, the robustness of the WiFi packet, colliding with ZigBee ACK, depends on the transmission rate of the WiFi packet. On one hand, setting a high transmission rate of WiFi fragments, i.e. applying higher resolution signal emulation, would provide us a higher possibility to successfully trigger the ZigBee ACK. On the other hand, the higher transmission rate indicates the fragment 2 is so vulnerable to the overlapped ZigBee signal that the fragment 2 might be corrupted and the CSI will not be recorded."

1. METHODOLOGY — fragment 1 manipulated to trigger ZigBee ACK overlapping fragment 2.
2. PREMISE — robustness depends on WiFi TX rate.
3. PREMISE (CONTRAST) — higher rate → higher chance to trigger ACK (finer emulation).
4. PREMISE (CONTRAST) — higher rate → fragment 2 fragile, may corrupt CSI.

**Shape:** METHODOLOGY → PREMISE → PREMISE → PREMISE
**Persuasive move:** trade-off setup for the next experiment.
**Sentence flow notes:** "On one hand…On the other hand" frames the trade-off.

---

## §6.6 Obtaining WiFi-ZigBee Mixed Signal · p.10 · ¶2

**Quote:** "To find out the optimal transmission rate of the WiFi fragments, we obtain the rate of successfully triggering ZigBee ACK at the ZigBee device and packet reception rate of the WiFi fragment 2 at X-MIMO in the settings of different tx rate. Specifically, the ZigBee devices are deployed at eight different positions in the office, as shown in Figure 25. The transmission rates we compare are 58.5 Mbps and 26 Mbps, which apply QAM 64 and QAM 16 mapper to emulate ZigBee. Since the QPSK and BPSK modulations are not suitable to emulate the ZigBee packet with ≥ 50% successful rate, their results are omitted in this section."

1. SCOPE (METHODOLOGY) — measure trigger rate and PRR across TX rates.
2. METHODOLOGY — eight positions in the office.
3. METHODOLOGY — compare 58.5 Mbps (QAM 64) vs. 26 Mbps (QAM 16).
4. SCOPE (CAUSE) — QPSK/BPSK omitted: <50% success.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → SCOPE
**Persuasive move:** experimental-trade-off enumeration.
**Sentence flow notes:** "Specifically" details position setup; "Since" supplies the omission rationale.

---

## §6.6 Obtaining WiFi-ZigBee Mixed Signal · p.11 · ¶1

**Quote:** "Figure 26(a) shows the fragmented WiFi packets with 58.5 Mbps tx rate trigger more ZigBee ACKs than 26 Mbps tx rate. This result is expected because the modulation of the 58.5 Mbps WiFi fragment is finer than the modulation of the 26 Mbps WiFi fragment. We also need to notice that, the percentage of successfully triggering ZigBee ACKs by 26 Mbps WiFi fragments is not significantly less than 58.5 Mbps fragments. Even at position 5, the 26 Mbps tx rate achieves a 72% successful rate while achieving more than 80% at other positions."

1. EVIDENCE — 58.5 Mbps triggers more ACKs than 26 Mbps.
2. CAUSE (INTERPRETATION) — finer modulation explains higher trigger rate.
3. CONTRAST (CLAIM) — 26 Mbps not significantly worse.
4. EVIDENCE (EXAMPLE) — worst case at position 5 still 72%; ≥80% elsewhere.

**Shape:** EVIDENCE → CAUSE → CONTRAST → EVIDENCE
**Persuasive move:** finer-mod-wins-but-coarser-still-good.
**Sentence flow notes:** "This result is expected because" supplies the cause; "We also need to notice that" pivots to the surprising near-equivalence; "Even at position 5" supplies the worst-case datum.

---

## §6.6 Obtaining WiFi-ZigBee Mixed Signal · p.11 · ¶2

**Quote:** "Figure 26(b) illustrates the percentage of the WiFi-ZigBee overlapped signal to be successfully received at X-MIMO device. As we can see from this result, since the ZigBee devices at positions 1, 3, and 6 are too close to X-MIMO device, the fragment 2 of 58.5 Mbps is easier to be corrupted than the 26 Mbps WiFi fragment. The average success rate of receiving the 26 Mbps WiFi-ZigBee overlapped signal at eight positions is 94.3% while the average success rate of receiving the 58.5 Mbps WiFi-ZigBee overlapped signal at eight positions is only 75.2%. Since the 26 Mbps WiFi packets could emulate ZigBee packets with a high success rate (≥ 80%) and 94.3% of replied ZigBee ACK could be captured in the WiFi CSI, we set the tx rate of WiFi fragments to be 26 Mbps."

1. EVIDENCE (BRIDGE) — Figure 26(b) on overlap reception.
2. CAUSE (EVIDENCE) — proximity makes 58.5 Mbps fragment 2 more corruptible.
3. EVIDENCE — average PRR: 26 Mbps 94.3% vs. 58.5 Mbps 75.2%.
4. CLAIM (CONSEQUENCE) — choose 26 Mbps as the WiFi fragment TX rate.

**Shape:** EVIDENCE → CAUSE → EVIDENCE → CLAIM
**Persuasive move:** quantitative-trade-off resolution.
**Sentence flow notes:** "As we can see from this result, since…" supplies the cause; "Since the 26 Mbps WiFi packets could emulate ZigBee packets with a high success rate" delivers the decision.

---

## §6.7 Impact of Transmission Power · p.11 · ¶1

**Quote:** "We test the impact of transmission power via the symbol error rate at ZigBee devices by controlling transmission power at the X-MIMO device. We deploy X-MIMO device, WiFi device (transmits customized fragments), and ZigBee devices (position 2) in the NLoS scenario, as Figure 15 depicts. We set the transmission power of X-MIMO to be 5 - 17 dBm via iw command. Since the two ZigBee devices have a similar symbol error rates, we plot the average SER of two ZigBee devices in Figure 27. When we set the transmission power of X-MIMO to be 11 dBm, the SER (6.8%) is the lowest compared to other settings. Since the default transmission power of the ZigBee device (TelosB mote) is 0 dBm, according to our design in Section 4.3, 8.45 dBm transmission power at each X-MIMO antenna would maintain relative amplitude. Then, the total transmission power of X-MIMO should be 11.45 dBm (=8.45+3). As WiFi hardware only allows us to set the integer transmission power, 11 dBm is the closest legitimate value, leading to the minimum SER."

1. SCOPE — test TX-power impact on ZigBee SER.
2. METHODOLOGY — NLoS deployment at position 2.
3. METHODOLOGY — power sweep 5–17 dBm via iw.
4. METHODOLOGY (PREMISE) — average SER plotted due to similar device behaviour.
5. EVIDENCE — minimum SER 6.8% at 11 dBm.
6. CAUSE (INTERPRETATION) — derivation: 8.45 dB/antenna ⇒ 11.45 dBm total.
7. CONSEQUENCE — 11 dBm is closest integer setting, hence the optimum.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE → CAUSE → CONSEQUENCE
**Persuasive move:** empirical-matches-theoretical-power-budget.
**Sentence flow notes:** "Since the default transmission power of the ZigBee device" tags the rationale; "As WiFi hardware only allows us to set the integer transmission power" lands the discretisation explanation.

---

## §6.8 Immunity to ZigBee ACK Jitter · p.11 · ¶1

**Quote:** "In this experiment, we evaluate the impact of the ZigBee ACK jitter. To start, the distribution of the measured jitter is illustrated in Figure 28(a), where the jitter is within the interval of [−75, 75] ns. Then, we use a USRP B210 to precisely control the timing of the WiFi fragments and ZigBee signal, such that the controlled jitter is imposed. By transmitting the customized WiFi and ZigBee signal with different jitter from USRP to X-MIMO device, X-MIMO estimates ZigBee channels and perform cross-technology precoding accordingly."

1. SCOPE — evaluates ACK jitter impact.
2. EVIDENCE (METHODOLOGY) — measured jitter spans [−75, 75] ns.
3. METHODOLOGY — USRP B210 used to impose controlled jitter.
4. METHODOLOGY — X-MIMO performs estimation and precoding per-jitter.

**Shape:** SCOPE → EVIDENCE → METHODOLOGY → METHODOLOGY
**Persuasive move:** stress-test-with-controlled-perturbation.
**Sentence flow notes:** "To start" sequences setup; "Then" introduces controllable jitter via USRP.

---

## §6.8 Immunity to ZigBee ACK Jitter · p.12 · ¶1

**Quote:** "The symbol error rate of ZigBee suffered from different jitters are plotted in Figure 28(b). The SER is random and ≤ 1% under the jitter from -0.1 𝜇s to 0.1 𝜇s, showing that the ZigBee jitter is negligible in our design. Since the USRP, transmitting customized ZigBee and WiFi fragments, in this experiment does not synchronize its clock to X-MIMO, this result has already involved the influence of CFO and hence no experiment of customizing CFO is conducted. Thus, this SER validates our assertion in Section 4.3 that our cross-technology precoding is immune to the phase uncertainties caused by jitter and CFO.

1. EVIDENCE (BRIDGE) — Figure 28(b) plots SER vs. jitter.
2. EVIDENCE (CLAIM) — SER ≤1% across ±0.1 μs jitter — jitter negligible.
3. METHODOLOGY (PREMISE) — USRP unsynchronised clock already includes CFO; no separate CFO test needed.
4. CLAIM (CONSEQUENCE) — empirically validates §4.3's immunity assertion.

**Shape:** EVIDENCE → EVIDENCE → METHODOLOGY → CLAIM
**Persuasive move:** empirical-closure-of-theoretical-claim.
**Sentence flow notes:** "Since the USRP…does not synchronize its clock to X-MIMO" justifies skipping a separate CFO sweep; "Thus" closes the loop with the §4.3 assertion.

---

## §7 Related Work · p.12 · ¶1

**Quote:** "MU-MIMO has been studied in many papers. Since they require very precise clock synchronization and precise channel estimation, most of their designs could only be implemented on the software-defined radio [18, 19, 28, 37, 41, 55] or customized hardware [11, 24, 54, 58]. For instance, MURS [19] utilizes an SDR to decode multiple packets simultaneously. Despite Surface MIMO [13] achieves up to 1.3 Gbps throughput on commodity WiFi devices, the design is hard to be applied on low-power devices because (i), the low-power devices cannot support the high-speed signal processing in consideration of energy consumption [20, 59, 60, 67], (ii) low-Power IoT does not support multiple antennas. To improve the spectral efficiency, a few works focus on concurrent communication for IoT [9, 12, 27]."

1. PREMISE — MU-MIMO is well-studied.
2. CAUSE (PREMISE) — strict synchronisation/estimation drives prior work to SDR/custom hardware.
3. EXAMPLE — MURS uses SDR to decode multiple packets.
4. CONCESSION (CONTRAST) — Surface MIMO works on commodity WiFi but not on low-power IoT; enumerates two reasons.
5. BRIDGE — flags concurrent-communication line of work as related.

**Shape:** PREMISE → CAUSE → EXAMPLE → CONCESSION → BRIDGE
**Persuasive move:** prior-work-on-the-wrong-hardware framing.
**Sentence flow notes:** "Since they require very precise clock synchronization" supplies the cause; "For instance" instantiates; "Despite Surface MIMO" concedes commodity progress while highlighting the IoT gap.

---

## §7 Related Work · p.12 · ¶2

**Quote:** "In this paper, we present X-MIMO to enable MU-MIMO from commodity WiFi to commodity ZigBee without any modification of hardware or firmware. X-MIMO explores the channel information in low-power commodity devices, offering more opportunities for wireless sensing and tracking [10, 14–17, 25, 26, 32, 35, 43, 50, 51, 57, 61, 63, 64]. Moreover, X-MIMO extends the single-stream signal emulation to multiple-streams signal emulation, which provides us an opportunity to further push the capacity of WiFi-ZigBee MU-MIMO up to the number of WiFi antennas. These unique features are not supported in existing CTC designs [22, 23, 29–31, 34, 38, 42, 52, 53, 62, 65, 66]."

1. CLAIM — restates X-MIMO contribution against the related-work background.
2. CONSEQUENCE — opens channel info on commodity for sensing/tracking.
3. CLAIM — extends single-stream CTC to multi-stream CTC capping at antenna count.
4. CLAIM (CONTRAST) — these features absent in prior CTC works.

**Shape:** CLAIM → CONSEQUENCE → CLAIM → CLAIM
**Persuasive move:** differentiation-against-CTC-corpus.
**Sentence flow notes:** "Moreover" stacks the second differentiator; "These unique features are not supported in existing CTC designs" closes with a citation-bracketed differentiation.

---

## §8 Conclusion · p.12 · ¶1

**Quote:** "This work presents X-MIMO, a cross-technology MU-MIMO on commodity devices. Utilizing cross-technology channel estimation and precoding, X-MIMO is the first work to offer cross-technology MU-MIMO on commodity devices. Our experiments demonstrate X-MIMO achieves the throughput of 495 Kbps, almost doubling the throughput of legacy ZigBee (250 Kbps), with 99% symbol reliability for two ZigBee receivers. X-MIMO achieves 704.24 Kbps for three ZigBee MU-MIMO and 983 Kbps for four ZigBee receivers (two on each ZigBee channel). Our evaluations also show that X-MIMO performs well in both LOS and NLOS scenarios, where the symbol error rate is ≤ 13.6%. Moreover, as the foundation of X-MIMO, cross-technology channel estimation is very precise on commodity devices, offering more opportunities for wireless sensing with low-power IoT devices."

1. CLAIM — summary statement of X-MIMO.
2. CLAIM (METHODOLOGY) — first cross-tech MU-MIMO on commodity, via estimation + precoding.
3. EVIDENCE — 495 Kbps (2× legacy), 99% symbol reliability.
4. EVIDENCE — 704.24 Kbps (three streams) and 983 Kbps (four receivers).
5. EVIDENCE (CLAIM) — LOS/NLOS robustness with SER ≤13.6%.
6. CONSEQUENCE — channel estimation foundation enables further IoT sensing.

**Shape:** CLAIM → CLAIM → EVIDENCE → EVIDENCE → EVIDENCE → CONSEQUENCE
**Persuasive move:** headline-restated-with-numbers-and-future-pull.
**Sentence flow notes:** "Utilizing cross-technology channel estimation and precoding" tags the mechanisms; consecutive sentences stack throughput numbers; "Moreover" pivots to a broader pay-off.

---

## §Appendix — Compensating HW Imperfections · p.12 · ¶1

**Quote:** "As in Figure 4, computing for 𝐻 assumes that CSIs are similar for the subcarriers that do not overlap with ZigBee (the shaded area), since these two CSIs are measured within the coherence time. However, as shown in Figure 29, it is typically not true. That is, CSI measurements suffer from phase distortion and amplitude offset which must be compensated prior to computing 𝐻 . In other words, the curves in the shaded area need to be matched."

1. PREMISE — original assumption: non-overlapped CSIs similar within coherence.
2. CONCESSION (CONTRAST) — Figure 29 shows assumption fails in practice.
3. CLAIM (INTERPRETATION) — phase distortion and amplitude offset must be compensated.
4. METHODOLOGY (CONSEQUENCE) — task is to match the shaded-area curves.

**Shape:** PREMISE → CONCESSION → CLAIM → METHODOLOGY
**Persuasive move:** ideal-assumption-corrected-with-compensation.
**Sentence flow notes:** "However" pivots assumption to reality; "That is" elaborates the discrepancy; "In other words" restates as a matching task.

---

## §Appendix — Compensating HW Imperfections · p.12 · ¶2 (Phase Compensation)

**Quote:** "Phase Compensation. Phase distortion stems from the packet boundary detection delay – a jitter in WiFi packet detection time that can be up to sampling duration. This incurs phase shift linearly to the subcarrier frequencies. Let 𝜏 be the difference in the boundary detection time for WiFi fragments 1 and 2. Then, the phase shift for 𝑘𝑡ℎ subcarrier becomes 2𝜋𝑘𝑓 𝛿 𝜏1 where 𝑓 𝛿 is the subcarrier spacing (=312.5KHz). This causes phase distortion linearly to 𝑘 (i.e., subcarrier index). As in Figure 30(a) the phase distortion is easily found via linear regression from the phase difference between two CSIs. The compensation would be adding the corresponding phase bias or equivalently, multiplying 𝑒𝑗2𝜋𝑘𝑓𝛿𝜏 to the CSI of WiFi fragment 1 for subcarrier 𝑘."

1. CAUSE — phase distortion from packet-boundary detection jitter.
2. CONSEQUENCE — phase shift linear in subcarrier frequency.
3. DEFINITION — 𝜏 = boundary-detection time difference.
4. METHODOLOGY — phase shift formula on subcarrier k.
5. INTERPRETATION — distortion linear in k.
6. METHODOLOGY (EVIDENCE) — detect via linear regression on phase difference.
7. METHODOLOGY (CONSEQUENCE) — compensation via complex-exponential multiplication.

**Shape:** CAUSE → CONSEQUENCE → DEFINITION → METHODOLOGY → INTERPRETATION → METHODOLOGY → METHODOLOGY
**Persuasive move:** identify-cause-then-derive-compensation.
**Sentence flow notes:** "This incurs phase shift linearly" couples cause to effect; "Let 𝜏 be" introduces notation; "Then" derives formula; "As in Figure 30(a)" supplies empirical handle; "The compensation would be" provides the fix.

---

## §Appendix — Compensating HW Imperfections · p.12 · ¶3 (Amplitude Compensation)

**Quote:** "Amplitude Compensation. Amplitude offset is caused by Automatic Gain Controller (AGC), a hardware component that dynamically scales the received signal to best fit the ADC range. This leads to an amplitude offset between CSI measurements. Meanwhile, AGC scales all subcarriers in a packet by the same amount. Therefore, as in Figure 30(b), the amplitude offset is simply the ratio between the CSI amplitude, which is consistent among all subcarriers. From this, the amplitude is compensated by multiplying the CSI of WiFi fragment 1 with the amplitude ratio averaged across non-overlapped subcarriers (under gray)."

1. CAUSE (DEFINITION) — AGC scales signals to fit ADC.
2. CONSEQUENCE — amplitude offset arises between CSIs.
3. PREMISE — AGC scales all subcarriers uniformly.
4. INTERPRETATION (CONSEQUENCE) — offset is a constant ratio across subcarriers.
5. METHODOLOGY — compensate by multiplying fragment 1's CSI by ratio averaged over non-overlapped subcarriers.

**Shape:** CAUSE → CONSEQUENCE → PREMISE → INTERPRETATION → METHODOLOGY
**Persuasive move:** AGC-uniformity-yields-simple-ratio-fix.
**Sentence flow notes:** "This leads to" introduces the consequence; "Meanwhile" adds the uniformity premise; "Therefore" combines into the constant-ratio interpretation; "From this" delivers the compensation rule.

---

## §Endnotes

**Paragraph count:** 60 prose paragraphs annotated (excluding figure/table captions, pure equation/pseudocode blocks, references, and acknowledgements).

**Sentence count:** 327 annotated sentences.

**Three most frequent paragraph shapes (top patterns by exact-tag composition):**
1. METHODOLOGY-only chains of varying length (e.g., METHODOLOGY → METHODOLOGY → METHODOLOGY, with or without additional METHODOLOGY tags) — dominant in §4.2, §5, §6.1, §6.2 (setup), §6.5 setup paragraphs.
2. EVIDENCE-then-CLAIM-or-INTERPRETATION patterns (e.g., EVIDENCE → EVIDENCE → CLAIM, EVIDENCE → INTERPRETATION → CONSEQUENCE) — dominant across §6 results paragraphs (§6.2, §6.4, §6.5 results, §6.6, §6.8, §8).
3. CLAIM-introducing-then-elaborating patterns mixing METHODOLOGY/CONSEQUENCE/EVIDENCE (e.g., CLAIM → METHODOLOGY → SCOPE → BRIDGE → METHODOLOGY → CLAIM → CONSEQUENCE for the intro proposal and CLAIM → EVIDENCE/CONSEQUENCE wrap-ups in §1.5 contribution bullets and §8).
