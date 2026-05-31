# Argumentation Annotation — Wang et al., "UMusic: In-car Occupancy Sensing via High-resolution UWB Power Delay Profile" (SenSys '25)

## Table of Contents
- 1. Introduction
- 2. Background and Motivation
  - 2.1 The Need of In-car Occupancy Sensing
  - 2.2 UWB Primer
  - 2.3 The Limitation of UWB CIR Amplitude
- 3. Design Overview
- 4. Main Design
  - 4.1 High-resolution Power Delay Profile in UWB
    - 4.1.1 Reverting Low-pass Filter
    - 4.1.2 Reflected Paths Separation
    - 4.1.3 CFR Transformation
    - 4.1.4 Immunity to Hardware Imperfections
  - 4.2 Car Occupancy Detection using PDP
- 5. Efficiency Enhancement
  - 5.1 Computational Cost Optimization
  - 5.2 Aliasing Avoidance
  - 5.3 PDP Synchronization
- 6. Evaluation
  - 6.1 Implementation
  - 6.2 Overall Performance
    - 6.2.1 Single-person Detection
    - 6.2.2 Multiple-people Detection
  - 6.3 Stationary vs Driving
  - 6.4 Impact of Out-car Environments
  - 6.5 Aggregated Performance
  - 6.6 Impact of Different Car Models and UWB Devices Deployment
  - 6.7 Impact of the Number of UWB Sensors
  - 6.8 Performance on Unseen Passenger
  - 6.9 Impact of Environment Augmentation
  - 6.10 PDP Calculation Precision
    - 6.10.1 Performance vs Number of Paths
    - 6.10.2 Performance under Hardware Imperfection
    - 6.10.3 With/Without Computational Cost Reduction
- 7. Discussion and Future Work
- 8. Related Work
- 9. Conclusion
- Endnotes

---

## §1. Introduction · p.116 · ¶1

**Verbatim:**
> The automotive industry has been undergoing a major transformation over the past century, shifting from engine-centric design to prioritizing passenger experience [14]. Modern cars are no longer just transport vehicles but intelligent ecosystems that enhance safety and comfort for users [4, 81, 86]. For instance, incorporating various sensors, automakers like Ford [13], Honda [19], and Tesla [59] are making significant progress in building advanced collision avoidance, theft protection, and keyless entry solutions [12, 21].

1. PREMISE — broad industry trend toward passenger-centric design.
2. CLAIM (PREMISE) — modern cars are "intelligent ecosystems".
3. EXAMPLE (EVIDENCE) — "For instance, … Ford, Honda, and Tesla".

**Shape:** PREMISE → CLAIM → EXAMPLE.
**Persuasive move:** stage-setting-by-industry-trend.
**Flow notes:** S1 paints a century-long trajectory; S2 escalates from "transport vehicles" to "intelligent ecosystems"; S3 "For instance" anchors the abstract claim in named manufacturers.

---

## §1. Introduction · p.116 · ¶2

**Verbatim:**
> Meanwhile, in-car occupancy sensing is gaining traction in the automotive industry, enhancing interior intelligence [4, 14, 32] for applications such as rear seat belt reminders, optimized airbag deployment [38], vital signs recognition [25, 29, 91], child (or pet) left behind detection [20, 85, 90], and personalized HVAC and stereo settings. Despite the promising applications, most of these techniques assume prior knowledge of the car's occupancy status. In other words, these sensing systems are unable to adaptively customize sensor parameters (e.g., camera rotation) to focus on the passenger for improved sensing performance when the occupancy status is unknown. While weight sensors are commonly used for in-car occupancy detection, they struggle to distinguish heavy luggage from human occupants [32]. Additionally, weight sensors are typically constrained to the front seats, while they are unavailable to the rear seats due to their high cost and installation complexity [35, 85].

1. PREMISE (EVIDENCE) — "Meanwhile, in-car occupancy sensing is gaining traction" with application catalog.
2. CONCESSION (CONTRAST) — "Despite the promising applications, most of these techniques assume prior knowledge".
3. INTERPRETATION — "In other words" restates the gap as inability to adapt.
4. CONCESSION (EVIDENCE) — weight sensors confuse luggage vs. occupants.
5. CONSEQUENCE (EVIDENCE) — weight sensors confined to front seats, high cost in rear.

**Shape:** PREMISE → CONCESSION → INTERPRETATION → CONCESSION → CONSEQUENCE.
**Persuasive move:** gap-in-prior-work.
**Flow notes:** S1 "Meanwhile" pivots to the focal topic; S2 "Despite" turns to limitation; S3 "In other words" rephrases; S4 "While weight sensors" introduces incumbent technology and its weakness; S5 "Additionally" stacks a second weakness.

---

## §1. Introduction · p.116 · ¶3

**Verbatim:**
> To address the need for in-car occupancy detection, this paper introduces UMusic, a precise in-car sensing solution leveraging Ultra-Wideband (UWB) technology for occupancy detection. UMusic utilizes UWB's channel estimation capabilities to analyze the in-car environment. Changes in passenger occupancy alter UWB signal propagation paths, referred to as the power delay profile (PDP), leading to variations in channel information, specifically the accessible channel impulse response (CIR) data [48, 49]. UMusic leverages the existing deployment of UWB technology for access control via digital key services provided by manufacturers such as Volkswagen [40], BMW [2], and Hyundai [5]. Unlike vision-based [12, 22, 37, 51, 62], acoustic-based [90], mmWave-based [16, 25, 58], and WiFi-based [20, 81, 85] approaches, UMusic offers enhanced privacy preservation, cost-effectiveness, high precision and a lightweight design.

1. CLAIM — paper introduces UMusic.
2. METHODOLOGY — UMusic uses UWB channel estimation.
3. PREMISE (DEFINITION) — occupancy changes PDP, manifest in CIR.
4. PREMISE — leverages existing access-control UWB deployment.
5. CONTRAST (CLAIM) — "Unlike vision/acoustic/mmWave/WiFi … UMusic offers …".

**Shape:** CLAIM → METHODOLOGY → PREMISE → PREMISE → CONTRAST.
**Persuasive move:** reuse-existing-infrastructure-positioning.
**Flow notes:** S1 introduces; S2 names the mechanism; S3 explains the physical chain (occupancy→PDP→CIR); S4 grounds in actual manufacturer deployments; S5 "Unlike" differentiates against all rival modalities at once.

---

## §1. Introduction · p.116 · ¶4

**Verbatim:**
> However, directly applying CIR data for in-car environment sensing, particularly for occupancy detection, presents unique challenges. The metal structure confines reflected UWB signals within a compact (2m×2m) space, leading to rich reflections with similar propagation path lengths. These reflections are difficult to differentiate in the CIR data due to limited spatial resolution; specifically, two paths need to differ by at least 0.6 m to be separated effectively, as validated in Section 2.3. To address this challenge, deep-learning-based approaches, such as CarOSense [32], are utilized. These methods typically involve complex deep learning models that demand extensive training and testing to generalize across different car models, which poses challenges for their widespread adoption across various vehicle types.

1. CLAIM (CONTRAST) — "However, directly applying CIR data … presents unique challenges".
2. CAUSE — metal confines signals → rich similar-length reflections.
3. CONSEQUENCE (EVIDENCE) — 0.6 m resolution floor referenced to §2.3.
4. BRIDGE — names DL approach (CarOSense) as the existing answer.
5. CONCESSION — DL methods need extensive training, hindering generalization.

**Shape:** CLAIM → CAUSE → CONSEQUENCE → BRIDGE → CONCESSION.
**Persuasive move:** problem-statement-with-prior-art-shortfall.
**Flow notes:** S1 "However" flips from the rosy positioning to the difficulty; S2 explains the physical cause; S3 "specifically" quantifies; S4 "To address this challenge" cues prior solution; S5 closes by undercutting it.

---

## §1. Introduction · p.116 · ¶5

**Verbatim:**
> UMusic takes a different approach by employing signal processing techniques combined with a simple classification model to precisely detect in-car occupancy via the following innovative technical highlights: (i), High-Resolution and Robust PDP Calculation - UMusic carefully extracts reflected signal paths to compute a high-resolution PDP from CIR data by decomposing in-car reflections. Additionally, we provide a formal proof demonstrating the robustness of our PDP calculation method against errors introduced by UWB hardware imperfections. (ii), Smart Occupancy Detection - UMusic effectively utilizes a phenomenon where the presence of a passenger only affects signal propagation paths longer than the line-of-sight (LoS) path (i.e., TX-Passenger-RX), while shorter paths remain unaffected. By excluding these longer paths from the PDP, UMusic reduces the complexity of occupancy detection. With the assistance of a simple machine learning model such as SVM, UMusic detects changes in the LoS paths, enabling accurate occupancy status detection. (iii), Computational Efficiency - To optimize performance for onboard computers, UMusic employs a downsampling strategy to lower the computational cost of calculating PDP, cutting it down to a brisk 0.125 ms. A formal proof also substantiates the method's capability to preserve PDP accuracy efficiently. In addition, we address CIR aliasing and misalignment issues, enhancing the practicality of UMusic. By incorporating these techniques, UMusic achieves a highly accurate, lightweight, cost-effective, and privacy-preserving design, making it suitable for deployment in various car models. We evaluate UMusic's performance on two car models across different scenarios, achieving an overall detection rate of 90.2% and an accuracy of 99.4% when aggregating the results of six consecutive estimations. To summarize, the contribution of this paper is threefold:

1. CLAIM — UMusic uses signal-processing + simple classifier; previews three highlights.
2. METHODOLOGY — (i) high-resolution PDP by decomposing reflections.
3. CLAIM — formal proof of robustness to hardware imperfections.
4. METHODOLOGY (PREMISE) — (ii) phenomenon: passenger affects only paths longer than LoS.
5. METHODOLOGY — excluding longer paths reduces complexity.
6. METHODOLOGY — SVM on LoS-path changes yields detection.
7. METHODOLOGY (EVIDENCE) — (iii) downsampling cuts cost to 0.125 ms.
8. CLAIM — formal proof preserves PDP accuracy.
9. METHODOLOGY — also address aliasing and misalignment.
10. CONSEQUENCE (CLAIM) — design is accurate, lightweight, cost-effective, privacy-preserving.
11. EVIDENCE — 90.2% single-shot, 99.4% with six aggregations.
12. BRIDGE — "the contribution of this paper is threefold".

**Shape:** CLAIM → METHODOLOGY → CLAIM → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → CLAIM → METHODOLOGY → CONSEQUENCE → EVIDENCE → BRIDGE.
**Persuasive move:** numbered-technical-highlights-with-proof-claims.
**Flow notes:** Three "(i)/(ii)/(iii)" labels segment the paragraph; "Additionally" extends each bullet with the proof move; "By incorporating these techniques" summarizes; "We evaluate" pivots to the empirical anchor; "To summarize" funnels into the bullet list that follows.

---

## §1. Introduction · p.116 · ¶6 (contribution bullet 1)

**Verbatim:**
> We present UMusic, an innovative in-car occupancy sensing system designed to detect passengers' seats accurately. UMusic envisions to utilize the UWB devices available in cars for access control, making it a cost-effective and easy-to-deploy solution.

1. CLAIM — presents UMusic.
2. CONSEQUENCE (CLAIM) — reuse of existing UWB makes it cost-effective and easy to deploy.

**Shape:** CLAIM → CONSEQUENCE.
**Persuasive move:** deployment-justification.
**Flow notes:** S1 introduces; S2 derives a deployment claim from the infrastructure-reuse premise.

---

## §1. Introduction · p.116 · ¶7 (contribution bullet 2)

**Verbatim:**
> UMusic introduces an innovative path decomposition technique to capture the high-resolution PDP of in-car signal propagation, which is then used to detect passenger occupancy. To implement UMusic effectively, we tackle UWB hardware imperfections such as carrier frequency offset, sampling time offset, and random initial phase. Additionally, we reduce computational complexity to ensure compatibility with various onboard computing systems.

1. CLAIM (METHODOLOGY) — path decomposition for high-resolution PDP, used for detection.
2. METHODOLOGY — addresses CFO, STO, random initial phase.
3. METHODOLOGY — reduces complexity for onboard compatibility.

**Shape:** CLAIM → METHODOLOGY → METHODOLOGY.
**Persuasive move:** technical-contribution-stack.
**Flow notes:** S1 names the core technique; S2 "To implement … effectively" lists practical hurdles addressed; S3 "Additionally" extends to compute compatibility.

---

## §1. Introduction · p.116 · ¶8 (contribution bullet 3)

**Verbatim:**
> We evaluate UMusic's performance on DW1000 UWB devices under various scenarios. The experimental results demonstrate that UMusic achieves 99.4% accuracy in occupancy detection. At the same time, the simulation shows the high precision of PDP calculation, indicating the potential for UMusic's application in more general scenarios.

1. METHODOLOGY — evaluation on DW1000 across scenarios.
2. EVIDENCE — 99.4% accuracy headline number.
3. INTERPRETATION — simulation hints at generalizable PDP precision.

**Shape:** METHODOLOGY → EVIDENCE → INTERPRETATION.
**Persuasive move:** evidence-with-extrapolation.
**Flow notes:** S1 sets the platform; S2 reports headline; S3 "At the same time" connects to simulation for breadth.

---

## §2. Background and Motivation · §2.1 The Need of In-car Occupancy Sensing · p.117 · ¶1

**Verbatim:**
> In recent years, there have been significant advancements in in-car human sensing [12, 14, 64, 81], which allows the car to obtain the passengers' positions [25, 32], presence [12, 21], and even vital signs [25, 55]. Of these sensing abilities, occupancy detection has become increasingly important for modern vehicles. For example, by assisting impact sensors, the in-car occupancy detection system can ensure that only specific airbags (i.e., the driver and passenger front airbags and corresponding side airbags) are deployed, while the rest remain deactivated to prevent unnecessary injury [23]. Additionally, the vehicle's interior lighting, audio system, air conditioning, and power accessories can be adapted to enhance the passenger experience. Furthermore, occupancy detection serves as the foundation for many existing in-car sensing systems, including those for detecting vital signs [25, 55], which can be used to enable child left behind and medical emergency detection.

1. PREMISE (EVIDENCE) — recent advances in in-car human sensing.
2. CLAIM — occupancy detection is the most important sub-ability.
3. EXAMPLE — airbag deployment refined via occupancy.
4. EXAMPLE — interior comfort customization.
5. EXAMPLE (CLAIM) — occupancy as foundation of downstream vital-sign / child-left-behind systems.

**Shape:** PREMISE → CLAIM → EXAMPLE → EXAMPLE → EXAMPLE.
**Persuasive move:** importance-by-stacked-examples.
**Flow notes:** S1 surveys; S2 narrows; "For example" / "Additionally" / "Furthermore" enumerate three benefit categories (safety, comfort, foundational).

---

## §2.1 · p.117 · ¶2

**Verbatim:**
> Despite the potential benefits, a deployable in-car occupancy detection system is not there yet. EU NCAP [38] and NHTSA [39] currently only mandate occupancy detection for the driver and front seats, leaving rear seat detection unregulated, while the ability to derive the potential presence of a subject or object inside the car based on pressure will not be rewarded from 2025 onwards. Consequently, weight/pressure sensors are primarily installed in front seats, as equipping all seats with dedicated sensors would increase hardware and installation costs [35, 85]. A car occupancy detection system, in general, should be highly accurate and commercially viable, prompting us to rethink the UWB technologies that are already integrated into the existing in-car systems [2, 5].

1. CLAIM (CONCESSION) — "Despite the potential benefits, … not there yet".
2. EVIDENCE — EU NCAP / NHTSA regulatory state; pressure-sensing reward sunsets 2025.
3. CONSEQUENCE (CAUSE) — "Consequently, weight/pressure sensors are primarily installed in front seats".
4. CLAIM (BRIDGE) — desired system must be accurate and commercially viable → rethink UWB.

**Shape:** CLAIM → EVIDENCE → CONSEQUENCE → CLAIM.
**Persuasive move:** regulatory-and-economic-justification.
**Flow notes:** S1 declares the gap; S2 backs it with regulators; S3 "Consequently" derives current incumbent practice; S4 transitions to "rethink UWB" framing.

---

## §2.2 UWB Primer · p.117 · ¶1

**Verbatim:**
> UWB is a wireless technology widely applied in real-time location systems, featured by NXP Semiconductors [53] and Qorvo [48, 49] radio chips as well as many famous vendors, including Apple, Samsung, and Xiaomi. Its high accuracy in distance ranging is essentially achieved by utilizing UWB CIR data to determine the packet arrival time, which is used to precisely estimate the distance between the transmitter and receiver. The UWB PHY packet begins with the preamble field, consisting of sequences of +/- pulses or no pulse, as illustrated in Figure 1(a). The preamble sequence is designed to maintain a perfect periodic autocorrelation, enabling the receiver to obtain the exact CIR using a correlator [34]. As a result, the receiver marks the timestamp with the arrival time of the first path (the first peak in CIR), as depicted in Figure 1(b), while the full CIR data is temporally saved in the UWB PHY layer. With a pulse duration of 2 ns, which is equivalent to the 60 cm spatial resolution, UWB can support large-area sensing tasks such as people counting [9] and localization [26, 60, 61, 63] in a large room, car localization [82] on the road, and enabling keyless entry for cars [21]. However, directly using CIR amplitude for small-area sensing such as car occupancy detection would result in ambiguity, as demonstrated in our preliminary study in the next section.

1. DEFINITION (EVIDENCE) — UWB widely adopted by named vendors.
2. DEFINITION (CAUSE) — CIR drives precise ranging.
3. DEFINITION — preamble structure (Figure 1(a)).
4. DEFINITION (CAUSE) — preamble autocorrelation enables CIR via correlator.
5. CONSEQUENCE — receiver marks arrival time, full CIR retained at PHY.
6. CLAIM (EVIDENCE) — 2 ns pulse → 60 cm resolution → supports large-area sensing.
7. CONTRAST (BRIDGE) — "However, directly using CIR amplitude for small-area sensing … would result in ambiguity".

**Shape:** DEFINITION → DEFINITION → DEFINITION → DEFINITION → CONSEQUENCE → CLAIM → CONTRAST.
**Persuasive move:** primer-with-pivot-to-limitation.
**Flow notes:** Stack of "is", "begins with", "is designed to", "As a result", "With a pulse duration" builds the technical primer chain; S7 "However" pivots to the precise limitation that motivates §2.3.

---

## §2.3 The Limitation of UWB CIR Amplitude · p.117 · ¶1

**Verbatim:**
> We deploy two UWB devices in the middle of a car, as depicted in Figure 2(a), to collect the CIR data when a person sits on seat 1 and seat 2. As a person sits in different seats, the signal propagation paths affected by the human body should change significantly. The essence of occupancy detection is to capture this change from the power delay profile. However, as shown in Figure 2(b), the CIR data (amplitude) collected on these two seats are quite similar (with a correlation 𝜌 of 0.96). Such a high correlation would eventually result in the ambiguity of directly applying CIR to detect the occupied seats. This happens because the signal propagation change caused by these two occupancy statuses is much less than the spatial resolution (60 cm). Since the CIR amplitude couldn't fully capture any signal propagation change within 60 cm, it is challenging to distinguish the two adjacent seats (20 to 30 cm away) occupancy. Previous work on using UWB for small-area sensing has either relied on heavy machine-learning models [32, 89, 92] or complex hardware setups [3, 71, 88]. However, these designs are impractical for low-cost UWB radio chips and onboard computers with limited computational resources. In contrast, UMusic is designed to work with commodity UWB devices via a novel signal-processing technique to extract a high-resolution PDP that can perceive small environment changes caused by different occupancy statuses. As illustrated in Figure 2(b), the PDP calculated by UMusic reflects signal propagation in finer granularity, with the common first peak corresponding to the Tx-Rx path, while other peaks containing the Tx-Body-Rx path are different for the two seats. With the help of a lightweight classification model, UMusic achieves a detection rate of 99.4%, aggregated from six consecutive CIR measurements. In the next section, we provide an overview of UMusic, followed by a detailed design of each technical highlight.

1. METHODOLOGY — preliminary deployment of two UWB devices.
2. PREMISE — expected paths should change significantly.
3. DEFINITION (CLAIM) — "essence of occupancy detection" is capturing this change.
4. EVIDENCE (CONTRAST) — "However, … CIR amplitudes … correlation ρ of 0.96".
5. CONSEQUENCE — high correlation creates ambiguity.
6. CAUSE — change is below 60 cm resolution.
7. INTERPRETATION — therefore distinguishing 20–30 cm seats is challenging.
8. CONCESSION (EVIDENCE) — prior small-area UWB relies on heavy ML or complex hardware.
9. CONTRAST — "However, these designs are impractical" for low-cost chips.
10. CLAIM — "In contrast, UMusic is designed to work with commodity UWB devices …".
11. EVIDENCE — Figure 2(b) shows finer-grained PDP with distinct Tx-Body-Rx peaks.
12. EVIDENCE — 99.4% accuracy with six aggregated measurements.
13. BRIDGE — preview of overview and design sections.

**Shape:** METHODOLOGY → PREMISE → DEFINITION → EVIDENCE → CONSEQUENCE → CAUSE → INTERPRETATION → CONCESSION → CONTRAST → CLAIM → EVIDENCE → EVIDENCE → BRIDGE.
**Persuasive move:** empirical-limitation-then-counter-positioning.
**Flow notes:** S1–S3 set expectation; S4 "However" delivers the disconfirming measurement; S5–S7 derive cause; S8 "Previous work" surveys prior remedies; S9 "However" undercuts them; S10 "In contrast" introduces UMusic; S11–S12 evidence; S13 bridges to next section.

---

## §3. Design Overview · p.118 · ¶1

**Verbatim:**
> As depicted in Figure 3, UMusic comprises three steps: (i) CIR data is collected from multiple links of transmitters and receivers installed in a vehicle. The multiple Tx-Rx links allow us to obtain the PDP from various angles, providing a better perception of the signal propagation changes caused by a human. (ii) UMusic calculates the high-resolution PDP from the collected CIR data. Finally, the PDPs obtained from multiple Tx-Rx links are fed into a simple classification model in Step (iii) to detect the car occupancy status. The resolution of CIR amplitude is limited by the UWB PHY layer design, posing significant challenges for PDP calculation. UMusic overcomes this limitation and achieves high-resolution PDP calculation and accurate car occupancy detection by fully investigating the CIR data's amplitude and phase information. However, due to hardware imperfections, the phase information in CIR is highly biased, leading to more practical issues in the design of UMusic. The following section presents our solutions for these challenges.

1. METHODOLOGY — (i) CIR collected from multiple Tx-Rx links.
2. INTERPRETATION (CAUSE) — multiple links enable better perception.
3. METHODOLOGY — (ii) calculate high-resolution PDP.
4. METHODOLOGY — (iii) PDPs feed simple classifier for occupancy status.
5. PREMISE — CIR-amplitude resolution is PHY-limited.
6. CLAIM — UMusic resolves this by exploiting amplitude + phase.
7. CONCESSION (CONTRAST) — "However, … phase is highly biased" → practical issues.
8. BRIDGE — next section presents solutions.

**Shape:** METHODOLOGY → INTERPRETATION → METHODOLOGY → METHODOLOGY → PREMISE → CLAIM → CONCESSION → BRIDGE.
**Persuasive move:** pipeline-walkthrough-with-flagged-challenges.
**Flow notes:** "(i)/(ii)/(iii)" structure narrates the steps; S5–S6 reframe the high-level challenge; S7 "However" warns of phase bias; S8 hands off to design.

---

## §4. Main Design · p.118 · ¶1

**Verbatim:**
> As the core of UMusic, decomposing the reflected path in high-resolution PDP calculation is demonstrated first in this section, followed by car occupancy detection.

1. BRIDGE — section roadmap.

**Shape:** BRIDGE.
**Persuasive move:** sectional-signposting.
**Flow notes:** Single-sentence prelude orienting the reader.

---

## §4.1 High-resolution Power Delay Profile in UWB · p.118 · ¶1

**Verbatim:**
> To illustrate the root cause of the ambiguity in CIR amplitude, we first formulate the CIR estimation process in the UWB PHY layer before demonstrating the detailed design for high-resolution PDP calculation. Let 𝑥(𝑡) denote the transmitted signal of the preamble field in the UWB packet. After the UWB signal is transmitted, it traverses through 𝑁 signal propagation paths, resulting in copies with a delay of 𝜏𝑖 for the 𝑖-th path. These delayed copies arrive at the receiver side consecutively, yielding the received signal 𝑦(𝑡):

1. METHODOLOGY (BRIDGE) — explanatory roadmap.
2. DEFINITION — defines x(t) as the preamble signal.
3. DEFINITION (CAUSE) — multipath delays produce delayed copies.
4. DEFINITION — these copies form received signal y(t).

**Shape:** METHODOLOGY → DEFINITION → DEFINITION → DEFINITION.
**Persuasive move:** formal-buildup.
**Flow notes:** "To illustrate the root cause" frames the upcoming formalism; subsequent sentences chain definitions for the equation that follows.

---

## §4.1 · p.118 · ¶2 (post-Eq.1)

**Verbatim:**
> where 𝑓𝑐 is the center frequency of the UWB signal, and 𝑎𝑖 is the attenuation of the 𝑖-th path. In addition, the corresponding power delay profile could be formulated as follows:

1. DEFINITION — notation for f_c and a_i.
2. BRIDGE — sets up Eq. 2 for PDP.

**Shape:** DEFINITION → BRIDGE.
**Persuasive move:** notational-handoff.
**Flow notes:** "In addition" daisy-chains into the next equation.

---

## §4.1 · p.118 · ¶3 (post-Eq.2)

**Verbatim:**
> where 𝛿(𝑡) is a Dirac's delta function [41]. The received signal 𝑦(𝑡) is fed into the UWB PHY layer to obtain the CIR data as an estimation of ℎ(𝑡), as illustrated in Figure 4.

1. DEFINITION — Dirac delta notation.
2. METHODOLOGY — y(t) feeds PHY layer to yield CIR as estimate of h(t).

**Shape:** DEFINITION → METHODOLOGY.
**Persuasive move:** notational-handoff.
**Flow notes:** S1 defines a symbol; S2 anchors the function of CIR estimation visually.

---

## §4.1 · p.118 · ¶4

**Verbatim:**
> Specifically, upon the arrival of the UWB signal, the Mixer first performs a passband to baseband conversion on the signal to obtain the baseband signal 𝑦(𝑡). Subsequently, the received signal is passed through a Low-pass Filter to eliminate unwanted parts and retain the middle 𝐵 Hz bandwidth (e.g, 500MHz), which is achieved mathematically by convolving 𝑦(𝑡) with the sinc-shaped filter, represented by sinc(𝐵𝑡) = sin(𝜋𝐵𝑡)/𝜋𝐵𝑡 [41]. The resulting signal is then sampled by the Analog-to-digital Converter (ADC) at every 𝑇𝑠 (e.g., 1 ns) interval to generate time-domain samples. Finally, the CIR Estimation applies a correlator to extract the CIR Data, which is denoted by ℎ[𝑛]:

1. METHODOLOGY — Mixer down-conversion.
2. METHODOLOGY (DEFINITION) — Low-pass filter, sinc shape.
3. METHODOLOGY — ADC sampling at T_s.
4. METHODOLOGY — correlator extracts h[n].

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY.
**Persuasive move:** pipeline-walkthrough.
**Flow notes:** "Specifically", "Subsequently", "then", "Finally" form a linear pipeline narration.

---

## §4.1 · p.118 · ¶5 (post-Eq.3)

**Verbatim:**
> By comparing ℎ[𝑛] in Equation 3 and ℎ(𝑡) in Equation 2, the precision of using CIR amplitude as PDP is determined by the sinc filter. To demonstrate these effects thoroughly, we provide an example with two signal propagation paths.

1. INTERPRETATION — precision of CIR-as-PDP is dictated by sinc filter.
2. BRIDGE — promises an illustrative two-path example.

**Shape:** INTERPRETATION → BRIDGE.
**Persuasive move:** analysis-then-example.
**Flow notes:** "By comparing" performs derivation; "To demonstrate" cues an example to come.

---

## §4.1 · p.118 · ¶6

**Verbatim:**
> Figure 5 illustrates an example where the channel includes two paths with propagation delays of 1 ns and 1.5 ns, respectively. When two copies of the transmitted signal arrive through these paths, they are shaped by the sinc filter with a main lobe of 2 ns, resulting in the corresponding two peaks merging into a single high peak at 1.25 ns. After ADC sampling, the resulting samples are further distorted, and the CIR amplitude does not accurately capture the two paths of 1 ns and 1.5 ns. This ambiguity in the CIR data can mislead car occupancy detection. To address this ambiguity, Caraokey [21] attempts to improve the precision of CIR amplitude through zero-padding-based upsampling. However, this approach only smooths the CIR amplitude, and the ambiguity remains unresolved. In the following section, we will demonstrate how UMusic overcomes the influence of LPF and achieves a high-resolution PDP.

1. EXAMPLE — Figure 5 two-path scenario.
2. CAUSE (EVIDENCE) — sinc 2 ns main lobe merges peaks at 1.25 ns.
3. CONSEQUENCE — ADC further distorts → CIR doesn't capture two paths.
4. CONSEQUENCE — ambiguity misleads detection.
5. CONCESSION (BRIDGE) — Caraokey tries zero-padding upsampling.
6. CONTRAST — "However, … smooths … ambiguity remains unresolved".
7. BRIDGE — preview of UMusic's LPF inversion.

**Shape:** EXAMPLE → CAUSE → CONSEQUENCE → CONSEQUENCE → CONCESSION → CONTRAST → BRIDGE.
**Persuasive move:** worked-example-rebutting-prior-fix.
**Flow notes:** "When two copies" walks the example; "After ADC" follows the pipeline; "To address" introduces a rival fix; "However" disqualifies it; final sentence promises UMusic's answer.

---

## §4.1.1 Reverting Low-pass Filter · p.118 · ¶1

**Verbatim:**
> In order to undo the LPF's sinc shaping, the time-domain CIR ℎ[𝑛] is transformed into the frequency domain, resulting in the channel frequency response (CFR):

1. METHODOLOGY — FFT of h[n] yields CFR to undo LPF.

**Shape:** METHODOLOGY.
**Persuasive move:** technique-introduction.
**Flow notes:** "In order to undo" frames the motivation in one sentence before the equation.

---

## §4.1.1 · p.119 · ¶2 (post-Eq.4)

**Verbatim:**
> where 𝑓0 represents the center frequency of the leftmost frequency bins, and 𝑓Δ denotes the channel spacing, which is typically 5 MHz in UWB. As the sinc function in the frequency domain is equivalent to a rectangular function [41], the CFR in Equation 4 is free of LPF's influence. While this calculation is simple, it is a crucial step for reverting the low-pass filter to obtain the reflected paths.

1. DEFINITION — f_0 and f_Δ.
2. INTERPRETATION (CAUSE) — sinc ↔ rectangle in frequency → CFR free of LPF.
3. CLAIM (CONCESSION) — "While this calculation is simple, it is a crucial step".

**Shape:** DEFINITION → INTERPRETATION → CLAIM.
**Persuasive move:** simple-but-essential.
**Flow notes:** "As the sinc function" justifies; "While this calculation is simple" preempts a "too easy" objection.

---

## §4.1.2 Reflected Paths Separation · p.119 · ¶1

**Verbatim:**
> However, transforming the CIR data into the frequency domain incorporates all path information into each frequency bin, making it challenging to distinguish individual paths. To clarify this problem, let us first establish the formulation. Specifically, we represent the phase increment between two consecutive frequency bins for the 𝑖-th path with Ω𝑖 = 𝑒−𝑗2𝜋𝑓Δ𝜏𝑖. Similarly, we use 𝛾𝑖 = 𝑎𝑖𝑒−𝑗2𝜋𝑓0𝜏𝑖 to denote the complex value of the 𝑖-th path on the first (leftmost) frequency bin. Using these notations, the received CFR of the 𝑁 paths are expressed as follows:

1. CLAIM (CONTRAST) — "However, … challenging to distinguish individual paths".
2. BRIDGE — establishes notation.
3. DEFINITION — Ω_i phase increment.
4. DEFINITION — γ_i complex value on first bin.
5. BRIDGE — leads to Eq. 5.

**Shape:** CLAIM → BRIDGE → DEFINITION → DEFINITION → BRIDGE.
**Persuasive move:** problem-then-formalize.
**Flow notes:** "However" sets the next sub-problem; "To clarify this problem" cues the formulation; "Specifically"/"Similarly" introduce notation.

---

## §4.1.2 · p.119 · ¶2 (post-Eq.5)

**Verbatim:**
> where 𝑀 is the number of frequency bins included in the PDP calculation; typically 100 in UWB. Essentially, the process of computing PDP from CIR data involves solving Equation 5 to derive the elements of matrix 𝛀. However, due to the presence of two unknowns (𝛀 and 𝚪) on the right-hand side and only one known (CFR vector H) on the left-hand side, this equation cannot be straightforwardly solved using standard linear algebra techniques.

1. DEFINITION — M is number of frequency bins, ~100.
2. INTERPRETATION — solving Eq. 5 for Ω = computing PDP.
3. CLAIM (CONTRAST) — "However, … cannot be straightforwardly solved".

**Shape:** DEFINITION → INTERPRETATION → CLAIM.
**Persuasive move:** identify-the-bottleneck.
**Flow notes:** "Essentially" recasts; "However" surfaces the algebraic obstruction.

---

## §4.1.2 · p.119 · ¶3

**Verbatim:**
> We note that the matrix 𝛀 exhibits the Vandermonde property, which motivates us to utilize the MUSIC algorithm [52] to solve Equation 5. In accordance with the MUSIC algorithm's convention, we refer to the matrix 𝛀 as the steering matrix, where each column is known as a steering vector. Originally developed for determining the angle-of-arrival (AoA) of each incident signal in a uniform linear antenna array, the MUSIC algorithm accepts the known left-hand side matrix (corresponding to H in Equation 5) as input and generates an estimate for every element in the steering matrix of Vandermonde shape (which corresponds to H in Equation 5), without requiring knowledge of 𝚪. It is worth pointing out that the MUSIC algorithm assumes uncorrelated incident signals and requires a sufficient number of signals to be collected from the antenna array. Consequently, to apply the MUSIC algorithm, H and 𝚪 must be matrices with a rank greater than the number of reflected paths 𝑁, as opposed to vectors in Equation 5. The next section illustrates how to transform H and 𝚪 into the required matrices, while preserving the same steering elements, to prepare them to be solved by the MUSIC algorithm.

1. CLAIM (CAUSE) — Vandermonde property motivates MUSIC.
2. DEFINITION — renaming convention for Ω as steering matrix.
3. DEFINITION (METHODOLOGY) — MUSIC origin and operation.
4. SCOPE — MUSIC's assumptions (uncorrelated signals, enough collected).
5. CONSEQUENCE — H and Γ must be rank > N.
6. BRIDGE — next section transforms H, Γ accordingly.

**Shape:** CLAIM → DEFINITION → DEFINITION → SCOPE → CONSEQUENCE → BRIDGE.
**Persuasive move:** borrow-known-algorithm-with-caveats.
**Flow notes:** "We note" presents insight; "In accordance" formalizes naming; "Originally developed" provides lineage; "It is worth pointing out" flags assumptions; "Consequently" derives constraint; final sentence bridges.

---

## §4.1.3 CFR Transformation · p.119 · ¶1

**Verbatim:**
> The transformation of the CFR vector is motivated by Spatial Smoothing [54], which rearranges the CFR vector H into a full-rank matrix without altering any of the elements in the steering matrix. The essential idea behind this procedure is that if we could identify several subsets of H with the same steering elements as the initial H, these subsets would be combined to form a full-rank matrix. The details are explained as follows:

1. METHODOLOGY (CAUSE) — motivated by Spatial Smoothing.
2. INTERPRETATION — sketches the rearrangement idea.
3. BRIDGE — promises details.

**Shape:** METHODOLOGY → INTERPRETATION → BRIDGE.
**Persuasive move:** technique-handoff.
**Flow notes:** "The transformation … is motivated by" credits prior tool; "The essential idea" recasts informally; "as follows" cues the details.

---

## §4.1.3 · p.119 · ¶2

**Verbatim:**
> The selection of subsets of p elements from H to form the columns of the new matrix H′ is depicted in Figure 6. The new steering matrix, 𝛀′, is identical to the original 𝛀. Since H′ is the product of the full-rank matrix 𝛀′ and 𝚪′, which is a scaled version of the Vandermonde matrix and also full-rank, UMusic successfully transforms the CFR vector H into a full-rank matrix H′. The size of the matrix 𝛀′ is 𝑝×𝑁, while the size of 𝚪′ is 𝑁×𝑞. As both matrices are full-rank, the rank of 𝚪′ is min(𝑞,𝑁), and the rank of H′ is min(𝑝,𝑞,𝑁). To meet the rank requirement for H′ and 𝚪′, which must exceed the number of reflected paths 𝑁, we select values of 𝑝 and 𝑞 that are greater than 𝑁. This ensures that the requirement for using the MUSIC algorithm to obtain the PDP is satisfied. We can then input the matrix H′ into the MUSIC algorithm to obtain the high-resolution PDP, consisting of 𝜏1 to 𝜏𝑁.

1. METHODOLOGY — subset selection forms H'.
2. CLAIM — Ω' identical to Ω.
3. CAUSE (CLAIM) — H' = Ω'·Γ' is full-rank → transformation succeeds.
4. DEFINITION — sizes p×N and N×q.
5. INTERPRETATION — rank min(p,q,N).
6. METHODOLOGY (CLAIM) — choose p, q > N to meet requirement.
7. CONSEQUENCE — requirement satisfied.
8. METHODOLOGY — feed H' into MUSIC to obtain τ_1…τ_N.

**Shape:** METHODOLOGY → CLAIM → CAUSE → DEFINITION → INTERPRETATION → METHODOLOGY → CONSEQUENCE → METHODOLOGY.
**Persuasive move:** constructive-proof-of-correctness.
**Flow notes:** "Since H' is the product …" delivers a one-sentence proof; "To meet the rank requirement" applies the constraint; "We can then" closes the algorithmic loop.

---

## §4.1.3 · p.119 · ¶3

**Verbatim:**
> In UWB, the number of frequency bins, 𝑀, is typically 100, and the number of reflected paths within the 2𝑚×2𝑚 in-car area is limited. This implies that there are many choices for the values of 𝑝 and 𝑞. However, if we carelessly select the combination of 𝑝 and 𝑞, it could lead to poor PDP performance. An example to illustrate this is provided next.

1. PREMISE — M~100, paths are limited in 2m×2m.
2. INTERPRETATION — many viable (p, q) choices.
3. CLAIM (CONTRAST) — "However, … careless selection … poor performance".
4. BRIDGE — example follows.

**Shape:** PREMISE → INTERPRETATION → CLAIM → BRIDGE.
**Persuasive move:** flag-design-knob.
**Flow notes:** "This implies" extends the premise; "However" warns; "An example … next" cues.

---

## §4.1.3 · p.119 · ¶4

**Verbatim:**
> Consider the two paths shown in Figure 5 as an example. The resulting PDP calculated by (𝑝=50,𝑞=50) has more significant peaks than the result obtained with (𝑝=90,𝑞=10), as illustrated in Figure 7. This is because the corresponding steering vectors Ω⃗1 = [1,Ω1,…,Ω𝑝−1 1]⊤ and Ω⃗2 = [1,Ω2,…,Ω𝑝−1 2]⊤ are less distinguishable under (𝑝=90,𝑞=10) than under (𝑝=50,𝑞=50). Specifically, the maximum phase change for each steering vector is given by Ω𝑝−1 1 and Ω𝑝−1 2, which are (𝑝−1)𝜋/100 and 1.5(𝑝−1)𝜋/100, respectively. When 𝑝=10, the difference between these two phases is only 0.095𝜋, which is insufficient to distinguish between the two paths. In a practical low SNR environment, the results could be even more degraded. Since the values of 𝑝 and 𝑞 are constrained by the number of frequency bins, 𝑀, such that 𝑝+𝑞≤𝑀, UMusic must carefully select 𝑝 and 𝑞 to maximize the phase change over the steering vectors. Therefore, by selecting 𝑝=𝑞=𝑀/2, UMusic maximizes the phase change over the steering vectors, which leads to the maximum achievable SNR.

1. EXAMPLE — two paths from Figure 5.
2. EVIDENCE — (50,50) yields better peaks than (90,10).
3. CAUSE (INTERPRETATION) — steering vectors less distinguishable for (90,10).
4. EVIDENCE (DEFINITION) — phase formulas.
5. EVIDENCE — at p=10, phase diff 0.095π insufficient.
6. INTERPRETATION — low-SNR worsens it.
7. CONSEQUENCE — p+q ≤ M constrains design; must maximize phase change.
8. CLAIM — "Therefore, by selecting p = q = M/2, … maximum achievable SNR".

**Shape:** EXAMPLE → EVIDENCE → CAUSE → EVIDENCE → EVIDENCE → INTERPRETATION → CONSEQUENCE → CLAIM.
**Persuasive move:** parameter-optimization-by-example.
**Flow notes:** "Consider … as an example" frames; "This is because" gives the mechanism; "Specifically" provides numbers; "Since … such that" applies the constraint; "Therefore" yields the optimum.

---

## §4.1.4 Immunity to Hardware Imperfections · p.119 · ¶1

**Verbatim:**
> As the CIR data is obtained from the UWB PHY layer, any hardware imperfections in the components of the PHY layer may result in an inaccurate matrix H′ and subsequently affect the PDP calculation. In this section, we will show that our PDP calculation is robust and can be immune to the hardware imperfections.

1. PREMISE (CAUSE) — PHY hardware imperfections may bias H'.
2. CLAIM (BRIDGE) — section claim: robust immunity.

**Shape:** PREMISE → CLAIM.
**Persuasive move:** thesis-statement-of-robustness.
**Flow notes:** S1 sets the threat; S2 announces the rebuttal.

---

## §4.1.4 · p.120 · ¶2

**Verbatim:**
> Essentially, hardware imperfections can introduce three major errors in the CIR data, including: (i) carrier frequency offset (CFO), denoted as 𝑓𝐶𝐹𝑂, which arises due to unsynchronized oscillators in the UWB transmitter and receiver, causing a slight mismatch between the mixer at the receiver side and the center frequency of the transmitted UWB signal; (ii) sampling time offset 𝜏Δ, which is caused by unsynchronized analog-to-digital converters (ADCs) that sample the signal with a random time shift; and (iii) initial phase offset 𝜃Δ, which is an inherent and unknown phase value imposed on RF devices when they are powered on. In the presence of these imperfections, the received CFR is formulated as Ĥ[𝑘]:

1. DEFINITION (METHODOLOGY) — enumerates three imperfections (CFO, STO, initial phase) and their causes.
2. BRIDGE — leads to Eq. 6 for distorted CFR.

**Shape:** DEFINITION → BRIDGE.
**Persuasive move:** taxonomy-then-model.
**Flow notes:** "Essentially" introduces the categorization; "In the presence of these imperfections" cues the modified equation.

---

## §4.1.4 · p.120 · ¶3 (post-Eq.6)

**Verbatim:**
> where Ω̂𝑖 = 𝑒−𝑗2𝜋𝑓Δ(𝜏𝑖+𝜏Δ) and 𝛾̂𝑖 = 𝑎𝑖𝑒−𝑗2𝜋(𝑓𝐶𝐹𝑂+𝑓0)(𝜏𝑖+𝜏Δ)+𝜃Δ. It is important to note that the effects of CFO and random initial phase are confined to 𝛾̂𝑖, which implies that these hardware imperfections do not affect the steering matrix. Additionally, since the MUSIC algorithm does not require knowledge of 𝛾̂𝑖 to calculate the steering matrix, any variations in 𝛾̂𝑖 will not alter the PDP calculation result. Therefore, our high-resolution PDP calculation is immune to the effects of CFO and random initial phase.

1. DEFINITION — Ω̂_i, γ̂_i.
2. CLAIM (INTERPRETATION) — CFO and initial phase confined to γ̂_i.
3. CAUSE (CLAIM) — MUSIC ignores γ̂_i → variations don't change PDP.
4. CONSEQUENCE — "Therefore, … immune to … CFO and random initial phase".

**Shape:** DEFINITION → CLAIM → CAUSE → CONSEQUENCE.
**Persuasive move:** algebraic-isolation-proof.
**Flow notes:** "It is important to note" highlights the algebraic trick; "Additionally" stacks rationale; "Therefore" closes.

---

## §4.1.4 · p.120 · ¶4

**Verbatim:**
> The sampling time offset affects all estimated paths simultaneously by introducing a delay of 𝜏Δ, which results in a shift of all estimated peaks by the same amount. As shown in Figure 8, all peaks have a delay of 0.1 ns compared to the case without sampling offset. The relative distance between two peaks (paths) remains unchanged, irrespective of the sampling offset, which is critical for aligning all calculated PDPs, as described in Section 5.3. With this analysis, it can be concluded that our high-resolution PDP is immune to the three hardware imperfections mentioned above. This immunity is a unique advantage of our approach compared to existing wireless sensing works, such as SpotFi [24] and Seirious [27], which are significantly impacted by these imperfections and require additional designs to compensate for their effects. Moreover, unlike SpotFi, which relies on a multi-antenna setup in WiFi to decompose reflected paths, UMusic requires only a single antenna configuration in UWB, thanks to its novel formulation for UWB signal decomposition and optimal solution derivation. In the next section, we will demonstrate how UMusic detects car occupancy by utilizing the high-resolution PDP.

1. INTERPRETATION — STO is a uniform shift on all peaks.
2. EVIDENCE — Figure 8 shows 0.1 ns shift.
3. CLAIM (CONSEQUENCE) — relative distances preserved → support PDP alignment in §5.3.
4. CONSEQUENCE — "it can be concluded that … immune to the three hardware imperfections".
5. CONTRAST (CLAIM) — unique advantage vs SpotFi, Seirious.
6. CONTRAST (CLAIM) — also unlike SpotFi, UMusic needs only a single antenna.
7. BRIDGE — preview of §4.2 detection.

**Shape:** INTERPRETATION → EVIDENCE → CLAIM → CONSEQUENCE → CONTRAST → CONTRAST → BRIDGE.
**Persuasive move:** robustness-superiority-claim.
**Flow notes:** "As shown in Figure 8" evidences; "With this analysis, it can be concluded" wraps up; "This immunity is a unique advantage … compared to" distinguishes; "Moreover, unlike SpotFi" stacks a second differentiator; final sentence bridges.

---

## §4.2 Car Occupancy Detection using PDP · p.120 · ¶1

**Verbatim:**
> The detection of car occupancy primarily relies on our observation that the human body only affects signal propagation paths that are longer than the LoS path (TX-body-RX), while the shorter paths remain unaffected. To illustrate this observation, we compare the PDP obtained under two conditions: an empty car (without people occupancy) and a car with a person sitting in the right back seat, as shown in Figure 9. The results in Figure 9(b) demonstrate the newly created reflection caused by human occupancy. Specifically, the newly created reflection has a delay of 6.4 ns, which is 3.1 ns later than the first path (3.3 ns), while the remaining three peaks remain unchanged. This delay of 3.1 ns corresponds to a distance of 0.93 m, which is the additional length of the Tx-body-Rx path compared to the Tx-Rx path (1 m), thereby confirming the validity of our observation.

1. CLAIM (PREMISE) — observation: body affects only paths longer than LoS.
2. METHODOLOGY — compare empty vs. occupied PDP.
3. EVIDENCE — Figure 9(b) shows new reflection.
4. EVIDENCE (DEFINITION) — 6.4 ns peak, 3.1 ns later than first path.
5. INTERPRETATION (CLAIM) — 3.1 ns ↔ 0.93 m matches expected geometry → observation confirmed.

**Shape:** CLAIM → METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION.
**Persuasive move:** observation-validated-by-measurement.
**Flow notes:** S1 declares observation; "To illustrate" sets up experiment; "Specifically" gives numbers; final sentence converts to physical distance to close the loop.

---

## §4.2 · p.120 · ¶2

**Verbatim:**
> Using this observation and the high-resolution PDP, car occupancy detection can be achieved by comparing the PDP collected without people to the PDP obtained under the current status. As the in-car layout is limited to a 2𝑚×2𝑚 area, we only compare the PDP within the first 4 meters for occupancy detection, while PDPs longer than that are disregarded as they correspond to non-Tx-body-Rx paths. To enhance occupancy detection, UMusic utilizes the deployment of multiple UWB devices and simple classification models. Our evaluation shows that, with the support of high-resolution PDP, even traditional classification models like SVM can reach an accuracy of 90.2% for in-car occupancy detection.

1. METHODOLOGY (CLAIM) — detection by comparing empty vs. current PDP.
2. METHODOLOGY (CAUSE) — 2m×2m layout → restrict comparison to first 4 m.
3. METHODOLOGY — multiple UWB devices and simple classifier.
4. EVIDENCE — SVM reaches 90.2%.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE.
**Persuasive move:** feature-engineering-justification.
**Flow notes:** "As the in-car layout … 2m×2m" justifies the windowing; "To enhance" adds multi-device; "Our evaluation shows … even traditional … SVM" anchors with empirical result.

---

## §5. Efficiency Enhancement · p.120 · ¶1

**Verbatim:**
> UMusic's efficiency is further enhanced by three additional features.

1. BRIDGE — previews three features in this section.

**Shape:** BRIDGE.
**Persuasive move:** sectional-signposting.
**Flow notes:** Single-sentence bridge.

---

## §5.1 Computational Cost Optimization · p.120 · ¶1

**Verbatim:**
> This section reduces computational costs to facilitate UMusic's integration into onboard systems without overburdening resources already running multiple applications. As the solver for high-resolution PDP calculation, the MUSIC algorithm takes 𝑂(𝑝3) time complexity, which is dominated by the heavy eigenstructure decomposition of the covariance matrix of H′ (size of 𝑝×𝑞) [18]. However, since 𝑝 and 𝑞 have already been optimized, directly reducing these parameters would lead to a decrease in the precision of the PDP.

1. SCOPE (CLAIM) — goal: lower cost for onboard deployment.
2. DEFINITION — MUSIC is O(p^3); dominated by eigendecomposition.
3. CONSEQUENCE (CONTRAST) — directly reducing p, q hurts precision.

**Shape:** SCOPE → DEFINITION → CONSEQUENCE.
**Persuasive move:** cost-tradeoff-framing.
**Flow notes:** S1 frames the goal; S2 quantifies cost; "However" forbids the obvious lever.

---

## §5.1 · p.120 · ¶2

**Verbatim:**
> Our approach to achieving complexity reduction without compromising PDP precision is depicted in Figure 10. Here, we reduce the size of H′ by a factor of 𝐷 by downsampling the matrix H′, resulting in a complexity reduction of 𝐷3. For ease of understanding, we assume that 𝑝 and 𝑞 are multiples of 𝐷. By evenly selecting one CFR element from every 𝐷 elements, we obtain a downsampled CFR matrix, referred to as H𝐷, which can be decomposed into two matrices, namely 𝛀𝐷 and 𝚪𝐷. Although the 𝑖-th downsampled steering vector Ω⃗𝐷_𝑖 = [1,Ω𝐷_𝑖,…,Ω𝑝−𝐷_𝑖]⊤ has only 1/𝐷 of the elements compared to the steering vector Ω⃗_𝑖 = [1,Ω_𝑖,…,Ω𝑝−1_𝑖]⊤, the PDP calculated from this downsampled matrix remains accurate. The high precision is ensured by the substantial phase change over the downsampled steering vector. The phase change over the downsampled steering vector Ω⃗𝐷_𝑖 is given by 2𝜋𝑓Δ(𝑝−𝐷)𝜏𝑖, which is 2𝜋𝑓Δ(𝐷−1)𝜏𝑖 less than the phase change over the original steering vector. The lost, 2𝜋𝑓Δ(𝐷−1)𝜏𝑖, is negligible, and thus downsampling does not alter the correlation between the steering vectors of different paths.

1. CLAIM (METHODOLOGY) — Figure 10 shows the approach.
2. METHODOLOGY — downsample by D → D^3 cost reduction.
3. SCOPE — assume p, q multiples of D for clarity.
4. METHODOLOGY — H_D = Ω_D · Γ_D.
5. CLAIM (CONCESSION) — downsampled steering vector is 1/D the size, yet PDP remains accurate.
6. CAUSE — high precision from substantial phase change.
7. EVIDENCE (DEFINITION) — phase change 2πf_Δ(p-D)τ_i; loss is 2πf_Δ(D-1)τ_i.
8. INTERPRETATION (CLAIM) — loss negligible → correlation preserved.

**Shape:** CLAIM → METHODOLOGY → SCOPE → METHODOLOGY → CLAIM → CAUSE → EVIDENCE → INTERPRETATION.
**Persuasive move:** asymptotic-savings-with-precision-proof.
**Flow notes:** "Here, we reduce" introduces the trick; "For ease of understanding" admits the assumption; "Although … remains accurate" preempts an objection; "The lost … is negligible, and thus" closes with a small-error claim.

---

## §5.1 · p.120 · ¶3

**Verbatim:**
> This is validated in Figure 11(a), where the correlation between Ω⃗𝐷_1 and Ω⃗𝐷_2 is compared at different downsampling factors, even a six-fold reduction. The results indicate that the correlation between Ω⃗𝐷_1 and Ω⃗𝐷_2 remains relatively close, even after downsampling. To provide a more comprehensive evaluation, we simulate the steering vectors of 1000 different paths and calculate their correlation for various downsampling factors. The results, shown in Figure 11(b), indicate that the accuracy of the PDP calculation remains unaffected and consistent across different D values. Since the MUSIC algorithm primarily relies on the correlation between different paths, preserving this correlation is crucial for downsampling to reduce computational costs without losing PDP precision.

1. EVIDENCE — Figure 11(a) validates correlation up to 6× downsampling.
2. INTERPRETATION — correlation remains close.
3. METHODOLOGY — 1000-path simulation for breadth.
4. EVIDENCE (INTERPRETATION) — Figure 11(b) shows accuracy unaffected across D.
5. CAUSE (CLAIM) — preserving correlation is what makes downsampling safe.

**Shape:** EVIDENCE → INTERPRETATION → METHODOLOGY → EVIDENCE → CAUSE.
**Persuasive move:** simulation-backed-validation.
**Flow notes:** "This is validated in Figure 11(a)" cues evidence; "The results indicate" interprets; "To provide a more comprehensive evaluation" widens; "Since the MUSIC algorithm primarily relies on …" derives the underlying rationale.

---

## §5.2 Aliasing Avoidance · p.120 · ¶1

**Verbatim:**
> Despite the effectiveness of computational cost reduction, the side effect of downsampling is aliasing, where two different path yields the same steering vector. This is formally defined as Ω⃗𝐷_1 = Ω⃗𝐷_2, while 𝜏1≠𝜏2. Then, the two aliased paths follow:

1. CONCESSION (CLAIM) — downsampling causes aliasing.
2. DEFINITION — formal aliasing condition.
3. BRIDGE — sets up Eq. 7.

**Shape:** CONCESSION → DEFINITION → BRIDGE.
**Persuasive move:** concede-side-effect-then-formalize.
**Flow notes:** "Despite the effectiveness" pivots; "This is formally defined as" formalizes; "Then" leads to derivation.

---

## §5.2 · p.121 · ¶2 (post-Eq.7)

**Verbatim:**
> where 𝑟 is an integer. Aliasing causes erroneous results in high-resolution PDP calculation because the longer path (i.e., 𝜏2) confuses the computation of the short path (i.e., 𝜏1) if they satisfy the relationship specified in Equation 7. In fact, this relationship is easy to meet in the in-car environment.

1. DEFINITION — r is integer.
2. CAUSE (CLAIM) — aliasing causes errors when relationship holds.
3. CLAIM (INTERPRETATION) — "easy to meet in the in-car environment".

**Shape:** DEFINITION → CAUSE → CLAIM.
**Persuasive move:** identify-real-world-trigger.
**Flow notes:** "Aliasing causes erroneous results … because" explains; "In fact" emphasizes the practical risk.

---

## §5.2 · p.121 · ¶3

**Verbatim:**
> Equation 7 implies that aliasing happens 𝜏1 = 𝜏2 − 50 ns, when 𝐷 = 4, as illustrated in Figure 12, where the aliasing problem between the two regions with 50 ns (taps) gap. Since in-car signal propagation has a fairly long tail due to the signal frequently bouncing in compact and metal car structures, the aliasing issue is inevitable if using downsampling.

1. EVIDENCE — at D=4, aliasing at 50 ns gap.
2. CAUSE (CLAIM) — long-tail in-car propagation makes aliasing inevitable.

**Shape:** EVIDENCE → CAUSE.
**Persuasive move:** anchor-problem-in-physics.
**Flow notes:** S1 quantifies; "Since in-car signal propagation has a fairly long tail" justifies inevitability.

---

## §5.2 · p.121 · ¶4

**Verbatim:**
> To overcome the aliasing issue, we exclude the regions that could alias the first eight CIR taps. This protects the PDP calculation for the first 2.4 meters environment because this range is mainly leveraged for in-car occupancy detection as illustrated in Section 4.2. As illustrated in Figure 13, by excluding the second region, FFT results only contain the paths within the first region, thereby free from aliasing. This also brings another benefit: involving less number of CIR taps into FFT calculation naturally expands the channel spacing, i.e., 𝑓Δ, which is equivalent to downsampling the CFR matrix.

1. METHODOLOGY — exclude regions that would alias the first eight CIR taps.
2. SCOPE (CAUSE) — protects 2.4 m, the range used for detection.
3. EVIDENCE — Figure 13 shows aliasing-free FFT.
4. CONSEQUENCE (CLAIM) — added benefit: wider f_Δ ≡ downsampling.

**Shape:** METHODOLOGY → SCOPE → EVIDENCE → CONSEQUENCE.
**Persuasive move:** fix-with-bonus.
**Flow notes:** "To overcome" introduces the fix; "This protects … because" scopes; "As illustrated" evidences; "This also brings another benefit" upsells.

---

## §5.3 PDP Synchronization · p.121 · ¶1

**Verbatim:**
> Affected by the sampling offset, the calculated PDP results suffer from severe misalignment, as depicted in Figure 14(a). As of 1ns sampling offset, the corresponding misalignment (30 cm) is significant enough to cause the wrong comparison of PDP, even misleading the occupancy detection result. To synchronize the PDP calculated from different UWB packets, we leverage the insight in Section 4.1.4, where the relative distance between the peaks in PDP is immune to the sampling offset. Specifically, the first peak in the PDP result corresponds to the Tx-Rx path, which is the shortest one compared with all other paths. This path length is fixed since the Tx and Rx are tightly attached to the car. Therefore, by aligning the first peak in the PDP results, all PDPs are synchronized. As Figure 14(b) shows, the second peaks in two PDPs are also aligned after the synchronization of the first peak. This concludes all designs of UMusic. Next, we will demonstrate the evaluation of UMusic.

1. EVIDENCE (CAUSE) — sampling offset → severe misalignment.
2. INTERPRETATION — 1 ns offset = 30 cm misalignment, big enough to mislead detection.
3. METHODOLOGY (CAUSE) — leverage §4.1.4 insight: relative distance immune to STO.
4. DEFINITION — first peak = Tx-Rx path = shortest.
5. PREMISE — Tx-Rx length is fixed (devices fixed to car).
6. CONSEQUENCE (CLAIM) — align first peak → all PDPs synchronized.
7. EVIDENCE — Figure 14(b) shows second peak alignment after fix.
8. BRIDGE — closes design section; cues evaluation.

**Shape:** EVIDENCE → INTERPRETATION → METHODOLOGY → DEFINITION → PREMISE → CONSEQUENCE → EVIDENCE → BRIDGE.
**Persuasive move:** anchor-alignment-on-physical-invariant.
**Flow notes:** "Affected by" introduces problem; "As of 1 ns" quantifies; "To synchronize" announces remedy; "Therefore" yields the alignment rule; "This concludes all designs" caps the design block.

---

## §6. Evaluation · §6.1 Implementation · p.121 · ¶1 (System Implementation)

**Verbatim:**
> System Implementation: We implement UMusic in Python codes (i), collect CIR data from Radino DW1000 modules [50], (ii), calculate high-resolution PDP. (iii), recognize the car occupancy status via representative classification models. Moreover, we implement CarOSense as the State-of-the-Art (SoTA). Our approach for processing the CFR vector H is based on the PyArgus tool [45]. We select an SVM model as an occupancy detection model, implemented by directly calling svm.SVC() by default provided by scikit-learn [44]. Moreover, UMusic is implemented on a Mac M1 computer configured to use a single CPU and a single thread, without relying on GPU acceleration or any specific hardware needs from the Mac M1.

1. METHODOLOGY — UMusic in Python; CIR via Radino DW1000; calculate PDP; classify.
2. METHODOLOGY — also implement CarOSense baseline.
3. METHODOLOGY — uses PyArgus for H.
4. METHODOLOGY — default scikit-learn SVM.
5. METHODOLOGY — Mac M1, single CPU/thread, no GPU.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY.
**Persuasive move:** reproducibility-statement.
**Flow notes:** Sequential implementation enumeration; "Moreover" stacks the SoTA reproduction and the hardware choice.

---

## §6.1 · p.121 · ¶2 (SoTA Approach)

**Verbatim:**
> SoTA Approach: CaroSense [32] feeds the raw CIR data collected from eight UWB sensors deployed in a car into a deep learning model, which incorporates MIMO and masking techniques to determine occupancy status.

1. DEFINITION — describes CarOSense pipeline.

**Shape:** DEFINITION.
**Persuasive move:** baseline-spec.
**Flow notes:** Single-sentence baseline definition.

---

## §6.1 · p.121 · ¶3 (Data Collection)

**Verbatim:**
> Data Collection: Our evaluations include comparing UMusic with SoTA system, CarOSense [32] utilizing the same setup and dataset, for fairness. As Figure 15 depicts, eight UWB devices are positioned in a sedan at the following locations: 1 (front-left ceiling), 2 (rear-view mirror), 3 (front-right ceiling), 4 (center panel), 5 (back-center ceiling), 6 (back-left ceiling), 7 (back-right ceiling), and 8 (trunk). We also evaluate UMusic in an SUV with only four UWB devices deployed to showcase its generalizability with fewer UWB devices in Section 6.6. We employ nine volunteers with different bio-metrics (height: [165cm, 182cm]; weight: [125lb, 185lb]). We evaluate UMusic and SoTA for one to four people occupancy detection, under various scenarios, including car status (stationary or driving), car models (sedan or SUV), and out-car environment (indoor or outdoor). Volunteers are instructed to sit in different positions within the car and are free to move their bodies, hands, and legs, allowing them to adopt relaxed postures, such as using smartphones or talking with other passengers. For example, when evaluating UMusic's performance for detecting 2 people, two volunteers occupy all possible combinations of two seats. The CIR data is collected in a round-robin manner, where each UWB device takes turns transmitting UWB packets, while the other devices collect CIR data. During each round of data collection, data is collected for 10 minutes, with 20 CIR sample data collected every second, resulting in 12K CIR sample data per round. The dataset is divided into training, validation and testing set in a 7:2:1 ratio. All experiments are approved by our institution's Institutional Review Board (IRB).

1. METHODOLOGY (SCOPE) — fair comparison with CarOSense via same setup/dataset.
2. METHODOLOGY — sedan deployment of eight UWB devices.
3. METHODOLOGY — SUV with four devices for generalization.
4. METHODOLOGY — nine volunteers, height/weight ranges.
5. METHODOLOGY (SCOPE) — scenarios: car status, model, out-car environment.
6. METHODOLOGY — relaxed postures allowed.
7. EXAMPLE — "For example, … two-person case, all combinations".
8. METHODOLOGY — round-robin transmission/collection.
9. METHODOLOGY — 10 min, 20 samples/s, 12K per round.
10. METHODOLOGY — 7:2:1 train/val/test split.
11. SCOPE — IRB approval.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → EXAMPLE → METHODOLOGY → METHODOLOGY → METHODOLOGY → SCOPE.
**Persuasive move:** evaluation-fairness-and-coverage.
**Flow notes:** Linear specification of setup; "For example" clarifies the multi-person protocol; final sentence states IRB compliance.

---

## §6.1 · p.122 · ¶4 (Evaluation Metric)

**Verbatim:**
> Evaluation Metric: We utilize accuracy to evaluate the performance of in-car occupancy detection. Since volunteers are instructed to occupy all possible seat combinations, we also measure accuracy per seat to evaluate performance in detail for each individual seat.

1. METHODOLOGY — accuracy metric.
2. METHODOLOGY (CAUSE) — per-seat accuracy because all combinations covered.

**Shape:** METHODOLOGY → METHODOLOGY.
**Persuasive move:** metric-justification.
**Flow notes:** "Since" justifies the second metric.

---

## §6.2 Overall Performance · p.122 · ¶1

**Verbatim:**
> This experiment evaluates UMusic and CarOSense with the car parked in a garage, while no other objects are outside the car. As illustrated in Figure 16, we compare the accuracy of UMusic and CarOSense across different configurations for the number of people. The median accuracy of UMusic for 1 to 4 people is 97.2%, 93.3%, 87.2%, and 83.1%, respectively, representing improvements of 16.8%, 17.9%, 14.3%, and 13.7% compared to CarOSense, which achieves median accuracies of 83.2%, 79.1%, 76.3%, and 73.1%, respectively. This result demonstrates that our high-resolution PDP effectively captures the significant changes in signal propagation due to human occupancy, allowing traditional classification models like SVM to achieve an overall accuracy of 90.2%, outperforming CarOSense by 15.7%. Moreover, the detailed FP and FN of UMusic and CarOSense are provided in Table 1, where UMusic achieves a lower error rate with respect to the number of people. Notably, this accuracy could be further improved by aggregating results from the CIR collected at multiple time slots, as discussed in Section 6.5. Moreover, although the posture of lying in the backseat is not included in the experiments, this posture should still signal to UMusic that both backseats are occupied. To illustrate the detailed accuracy per seat, we provide a breakdown of the overall evaluation next.

1. METHODOLOGY (SCOPE) — garage setup, isolated car.
2. METHODOLOGY — Figure 16 comparison.
3. EVIDENCE — median accuracies and improvements.
4. INTERPRETATION (CLAIM) — high-resolution PDP enables SVM to reach 90.2%, +15.7% over CarOSense.
5. EVIDENCE — Table 1 FP/FN.
6. CLAIM (BRIDGE) — aggregation in §6.5 improves further.
7. CLAIM (CONCESSION) — lying posture not tested but should still trigger detection.
8. BRIDGE — per-seat breakdown next.

**Shape:** METHODOLOGY → METHODOLOGY → EVIDENCE → INTERPRETATION → EVIDENCE → CLAIM → CLAIM → BRIDGE.
**Persuasive move:** headline-result-with-mechanism-and-caveat.
**Flow notes:** "This result demonstrates that … allowing traditional … SVM … outperforming CarOSense by 15.7%" packages the key claim; "Notably" and "Moreover" stack improvements and edge cases.

---

## §6.2.1 Single-person Detection · p.122 · ¶1

**Verbatim:**
> Detecting a single person is essential for identifying when a person is left in the car. This experiment demonstrates the accuracy of UMusic and CarOSense in detecting a single person per seat in the car. The results are shown in Figure 17, where seat index 0 indicates nobody is in the car. The accuracy varies across different seats but remains above 95%, with the highest accuracy of 98.6% achieved at Seat #3; this is a 28.2% improvement compared to CarOSense's 80.4%. Similarly, UMusic's accuracy is higher than CarOSense across all seats.

1. PREMISE — single-person detection critical for child-left-behind use case.
2. METHODOLOGY — per-seat single-person test.
3. EVIDENCE — Figure 17.
4. EVIDENCE — > 95% accuracy, 98.6% at Seat #3; +28.2% vs. CarOSense.
5. CLAIM — UMusic beats CarOSense at every seat.

**Shape:** PREMISE → METHODOLOGY → EVIDENCE → EVIDENCE → CLAIM.
**Persuasive move:** importance-then-comparative-evidence.
**Flow notes:** "Similarly" generalizes the per-seat dominance.

---

## §6.2.2 Multiple-people Detection · p.122 · ¶1

**Verbatim:**
> As shown in Figure 18, when the number of occupants in the vehicle gradually increases, the accuracy of UMusic experiences a decline. For Seat #1, the accuracy is 92.3% when there are two people in the vehicle, which drops to 86.3% with three occupants, and further decreases to 82.1% when the number of occupants increases to four. The results for other seats are similar to those for Seat #1, as the reflected paths in the confined space of the vehicle become more complex with an increasing number of occupants, leading to deviations in PDP estimation.

1. CLAIM (EVIDENCE) — accuracy declines with more occupants.
2. EVIDENCE — Seat #1: 92.3% → 86.3% → 82.1%.
3. INTERPRETATION (CAUSE) — more occupants → more complex reflections → PDP deviation.

**Shape:** CLAIM → EVIDENCE → INTERPRETATION.
**Persuasive move:** acknowledge-degradation-with-cause.
**Flow notes:** "As the reflected paths … become more complex … leading to" provides causal account.

---

## §6.3 Stationary vs Driving · p.123 · ¶1

**Verbatim:**
> To evaluate the robustness of UMusic, we conduct experiments in driving scenarios, where one person occupies the driver's seat while others take turns sitting in the other three seats. The results in Figure 19 indicate that driving and stationary (engine on) have a slight impact on occupancy detection accuracy compared to when the car is stationary and the engine is off. This is likely due to the random shakes experienced when during these two cases, which can affect the collection of CIR data since the attachment to in-car objects may not be entirely stable.

1. METHODOLOGY (SCOPE) — driving scenario with driver + rotating seat occupants.
2. EVIDENCE — slight accuracy drop in driving vs. stationary-engine-off.
3. CAUSE (INTERPRETATION) — random shakes affect CIR collection.

**Shape:** METHODOLOGY → EVIDENCE → CAUSE.
**Persuasive move:** robustness-evidence-with-mechanism.
**Flow notes:** "This is likely due to" provides the speculated cause.

---

## §6.4 Impact of Out-car Environments · p.123 · ¶1

**Verbatim:**
> The metal structure of a car confines wireless signals, preventing the weak in-car RF signal from leaking out of a car and the out-car signals from penetrating into the car. To apply UMusic to the practical scenarios, we evaluate the performance under different out-car environments, as depicted in Figure 20. Figure 21 illustrates the performance of UMusic under different out-car environments. When there is only one occupant in the car, UMusic can accurately detect the occupant, with detection accuracy in different scenarios all above 94%. As the out-car environment changes, there is no significant change in detection accuracy, indicating that the out-car environment has a negligible impact on UMusic for in-car occupancy detection.

1. PREMISE (CAUSE) — metal shields RF in both directions.
2. METHODOLOGY — test under multiple out-car environments.
3. EVIDENCE — Figure 21.
4. EVIDENCE — > 94% accuracy across scenarios for single occupant.
5. INTERPRETATION (CLAIM) — out-car environment has negligible impact.

**Shape:** PREMISE → METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION.
**Persuasive move:** physical-justification-then-empirical-confirmation.
**Flow notes:** "The metal structure … confines wireless signals" predicts robustness; experiments confirm.

---

## §6.5 Aggregated Performance · p.123 · ¶1

**Verbatim:**
> Although the CIR data is collected every 50 ms using the Radino UWB DW1000 module and UMusic can provide results within that interval, users may only require the in-car occupancy system to deliver results once per second. This allows us to aggregate detection results over multiple CIR data samples to improve overall performance during that period. In this experiment, we combine the results from multiple detections and use a majority vote to estimate the in-car occupancy more accurately. As shown in Figure 22, after aggregating two to six occupancy detection results, UMusic achieves the accuracy of 93.5%, 96.7%, 98%, 98.4%, and 99.4%, respectively. This also shows that UMusic can achieve over 98% occupancy detection accuracy with just three aggregated results.

1. PREMISE (CONCESSION) — CIR every 50 ms but users only need 1 Hz output.
2. CONSEQUENCE — slack permits aggregation.
3. METHODOLOGY — majority vote across detections.
4. EVIDENCE — 93.5% → 96.7% → 98% → 98.4% → 99.4% with 2–6 aggregations.
5. INTERPRETATION (CLAIM) — > 98% achievable with just 3 aggregations.

**Shape:** PREMISE → CONSEQUENCE → METHODOLOGY → EVIDENCE → INTERPRETATION.
**Persuasive move:** exploit-temporal-slack.
**Flow notes:** "Although … users may only require" sets up the opportunity; "This allows us to aggregate" leverages it; figures back up the gain.

---

## §6.6 Impact of Different Car Models and UWB Devices Deployment · p.123 · ¶1

**Verbatim:**
> To verify the versatility of UMusic, we evaluate occupancy detection in each seat in an SUV with only four UWB devices deployed, as depicted in Figure 23(a). Four UWB devices provide CIR data of 12 links, from which UMusic estimates the occupancy status. Figure 23 shows the occupancy detection results of aggregating six consecutive estimations. Although only four devices are employed in this experiment, the accuracy of occupancy detection remains high in the SUV, all above 93%, showing the effectiveness of UMusic.

1. METHODOLOGY — SUV test with four UWB devices.
2. DEFINITION — 4 devices → 12 links.
3. METHODOLOGY — six-aggregation evaluation.
4. CLAIM (EVIDENCE) — > 93% across all seats despite fewer devices.

**Shape:** METHODOLOGY → DEFINITION → METHODOLOGY → CLAIM.
**Persuasive move:** generalization-evidence.
**Flow notes:** "Although only four devices are employed" preempts skepticism; "showing the effectiveness" closes.

---

## §6.7 Impact of the Number of UWB Sensors · p.124 · ¶1

**Verbatim:**
> To evaluate the impact of the number of sensors on UMusic's performance, we gradually reduce the number of sensors deployed in the vehicle from eight to three, measuring its accuracy under different occupancy status. Specifically, we use data collected from different combinations of UWB sensors, as described in Section 6.1, to simulate varying numbers of UWB sensors. The sensor combinations (1,2,3,4,5,6,7), (1,2,3,5,6,7), (1,3,4,5,7), (1,3,5,7), and (1,3,6) correspond to setups with seven to three UWB sensors, respectively. As shown in Figure 24, when eight sensors are used, UMusic achieves the average accuracy rates of 99.6%, 95.8%, 90.4%, and 85.8% for different occupancy status. When the number of sensors is reduced to 4, the accuracy slightly decreases but remains high at 97.2%, 93.3%, 87.5%, and 83.1%, with the decline within 3%. However, when the number of sensors is further reduced to 3, the detection performance shows a more noticeable decline, due to the reduced spatial diversity in PDP data.

1. METHODOLOGY — vary sensors 8 → 3.
2. METHODOLOGY — use subsets to simulate.
3. METHODOLOGY (DEFINITION) — exact subsets per count.
4. EVIDENCE — eight-sensor accuracy: 99.6 / 95.8 / 90.4 / 85.8.
5. EVIDENCE — four-sensor accuracy within 3% of eight.
6. CONCESSION (CAUSE) — at three sensors, noticeable decline due to lost spatial diversity.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE → EVIDENCE → CONCESSION.
**Persuasive move:** sensitivity-analysis-with-limit.
**Flow notes:** "Specifically" gives sampling plan; "However" introduces the regime where degradation appears, naming the cause.

---

## §6.8 Performance on Unseen Passenger · p.124 · ¶1

**Verbatim:**
> To evaluate UMusic's performance on seen/unseen passengers, we adopt the following dataset splitting strategy: data from five randomly selected volunteers is designated as the seen dataset, while data from the remaining four volunteers constitutes the unseen dataset. The model is trained on the seen dataset, which is further split into a training set and a testing set in an 8:2 ratio. To evaluate unseen passengers, the trained model is applied to the unseen dataset, which contains data from four volunteers. Since the seen dataset includes five passengers and the unseen dataset includes four passengers, the evaluation encompasses all occupancy statuses (from 1 person to 4 people). This process is repeated for all (9 choose 5) possible combinations of seen and unseen datasets, along with the corresponding models and results. Figure 25 shows UMusic's accuracy across various in-vehicle occupancy status, where the average accuracy for seen passengers is 96.2%, 92.3%, 86.2%, and 83.5%, while the average accuracy for unseen passengers is 94.7%, 90.1%, 83.1%, and 78.2%, respectively. These results demonstrate that UMusic achieves high recognition accuracy even for unseen passengers, with only a slight reduction in accuracy compared to the performance on seen passengers.

1. METHODOLOGY (SCOPE) — split 5/4 seen/unseen.
2. METHODOLOGY — 8:2 train/test in seen.
3. METHODOLOGY — apply to unseen 4.
4. SCOPE (CAUSE) — split design covers 1–4 occupancy statuses.
5. METHODOLOGY — repeat over all (9 choose 5) splits.
6. EVIDENCE — seen averages 96.2/92.3/86.2/83.5; unseen 94.7/90.1/83.1/78.2.
7. INTERPRETATION (CLAIM) — high accuracy on unseen with only slight reduction.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → SCOPE → METHODOLOGY → EVIDENCE → INTERPRETATION.
**Persuasive move:** generalization-protocol-with-numbers.
**Flow notes:** "Since the seen dataset includes five passengers" justifies coverage; "These results demonstrate" closes with claim.

---

## §6.9 Impact of Environment Augmentation · p.124 · ¶1

**Verbatim:**
> This evaluation considers three scenarios: placing a small box or bag on each seat, folding down the backseats, and pushing the front passenger seat back. To evaluate UMusic in these scenarios, we leverage the phenomenon that environmental augmentations remain static, while passengers typically exhibit unpredictable motion. By concatenating multiple consecutive CIR measurements into the SVM model, UMusic identifies occupancy status by capturing path changes caused by passengers, as described in [21]. Figure 26 illustrates UMusic's performance under environmental augmentation, with each line representing the average accuracy across four different occupancy statuses (1 to 4 people). Specifically, under the seat-pushed-back scenario, the average accuracy of UMusic is 85%, 92.4%, 96.4%, 98.7%, 99.4%, and 99.8% when one to six consecutive CIR measurements are concatenated as features for the SVM model, respectively. Meanwhile, the other two scenarios are similar.

1. SCOPE — three environmental augmentation scenarios.
2. METHODOLOGY (PREMISE) — exploit static-vs-motion contrast.
3. METHODOLOGY — concatenate consecutive CIR for SVM.
4. EVIDENCE — Figure 26.
5. EVIDENCE — seat-pushed-back: 85% → 99.8% as concatenation grows 1→6.
6. INTERPRETATION — other two scenarios similar.

**Shape:** SCOPE → METHODOLOGY → METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION.
**Persuasive move:** feature-inversion-of-concession.
**Flow notes:** Static-augmentation concession is turned into a feature by exploiting motion as a distinguishing signal; "Meanwhile" generalizes.

---

## §6.9 · p.124 · ¶2

**Verbatim:**
> In addition to the three scenarios that modify the in-car environment, UMusic is also capable of handling changes in the out-car environment, such as the use of sunshades. These are typically installed when passengers exit the vehicle, at which point the car is turned off. Since UMusic can detect a pet or passenger left behind within a few seconds after the engine is switched off, it does not need to operate continuously afterward. As a result, the presence of foil-like sunshades does not affect the functionality of UMusic.

1. CLAIM — UMusic also handles out-car changes like sunshades.
2. PREMISE — sunshades used after passengers exit.
3. CAUSE (CLAIM) — detection within seconds after engine off → no need for continuous operation.
4. CONSEQUENCE — sunshades don't affect functionality.

**Shape:** CLAIM → PREMISE → CAUSE → CONSEQUENCE.
**Persuasive move:** preempt-corner-case.
**Flow notes:** "In addition" stacks another scenario; "Since … it does not need to operate continuously afterward" sidesteps the concern; "As a result" concludes.

---

## §6.10 PDP Calculation Precision · p.124 · ¶1

**Verbatim:**
> While our evaluations demonstrate UMusic's performance in real-world scenarios, this section presents simulations to showcase PDP calculation performance in a controlled setting, where obtaining ground truth path lengths is challenging in real-world experiments. Specifically, our simulations are conducted to answer three major questions: (i), how many paths could be precisely resolved by the high-resolution PDP calculation? Since the in-car signal propagation is very complex, manufacturers are concerning the limit of our PDP calculation. (ii), are the three hardware imperfections fully immune? The quality of UWB radio varies for different manufacturers. We should confirm the PDP is robust to these issues even for the hardware with the worst quality. (iii), how could the computational cost reduction affect the precision of PDP calculation? The onboard computer is less powerful than our laptops. It also runs many interior systems, leaving limited computational resources for UMusic. Our simulation should demonstrate the effectiveness of computational cost while maintaining the PDP calculation precision. To fit into the car's interior area (2m×2m), our simulation focuses on paths that are less than 1.8 meters. The detailed simulation for the above three aspects is shown in the following sections.

1. BRIDGE (SCOPE) — pivot to controlled simulation due to lack of real-world ground truth.
2. QUESTION — (i) how many paths can be resolved?
3. PREMISE — manufacturers concerned with this limit.
4. QUESTION — (ii) full immunity to hardware imperfections?
5. PREMISE — worst-quality hardware must still be supported.
6. CLAIM (SCOPE) — confirm robustness across hardware variants.
7. QUESTION — (iii) impact of compute reduction on precision?
8. PREMISE — onboard computers are weaker and share resources.
9. CLAIM (SCOPE) — simulation should validate cost reduction without precision loss.
10. SCOPE — restrict to paths < 1.8 m.
11. BRIDGE — details in following sections.

**Shape:** BRIDGE → QUESTION → PREMISE → QUESTION → PREMISE → CLAIM → QUESTION → PREMISE → CLAIM → SCOPE → BRIDGE.
**Persuasive move:** stakeholder-questions-framing.
**Flow notes:** Three explicit "(i)/(ii)/(iii)" questions, each followed by a manufacturer-perspective premise — a stakeholder-driven structure.

---

## §6.10.1 Performance vs Number of Paths · p.125 · ¶1

**Verbatim:**
> In this simulation, we uniformly insert different paths into the 1.8 m compact in-car area. For instance, when we insert eight paths into the 1.8 m range, we control the average interval between two consecutive paths to be 1.8/8 = 0.225 meters. We also consider the regulation of the FCC about the maximum SNR of UWB signal (≤ −40dB/MHz [48]) in our simulation. Figure 27 depicts an example PDP calculation result of 8 paths and the corresponding groundtruth PDP used for that simulation. This result intuitively shows that the calculation error is quite small, indicating the high precision of our design.

1. METHODOLOGY — uniformly insert paths in 1.8 m.
2. EXAMPLE — 8 paths → 0.225 m spacing.
3. METHODOLOGY (SCOPE) — comply with FCC SNR regulation.
4. EVIDENCE — Figure 27 example PDP vs ground truth.
5. INTERPRETATION (CLAIM) — small error → high precision.

**Shape:** METHODOLOGY → EXAMPLE → METHODOLOGY → EVIDENCE → INTERPRETATION.
**Persuasive move:** controlled-illustration.
**Flow notes:** "For instance" exemplifies; "We also consider" widens realism; "This result intuitively shows" interprets.

---

## §6.10.1 · p.125 · ¶2

**Verbatim:**
> We also summarize the simulation results into two statistics shown in Figure 28. The result in Figure 28(a) shows the PDP calculation error when we insert seven, eight, and nine paths into the 1.8 m range. It is worth mentioning that for a 500MHz bandwidth UWB channel, the 1.8 m range corresponds to only six CIR taps, which is saturated by our 7-9 paths for pushing the PDP calculation into the limit. For each result, we repeatedly generate the uniformly distributed paths 1000 times to guarantee the effectiveness of this simulation. In specific, the average interval between two consecutive paths for the seven, eight, and nine paths scenarios are 1.8/7 = 0.257, 1.8/8 = 0.225, and 1.8/9 = 0.2 meters.

1. METHODOLOGY — Figure 28 summary statistics.
2. EVIDENCE — error for 7/8/9 paths.
3. INTERPRETATION (CLAIM) — 6 CIR taps saturated by 7–9 paths → stress test.
4. METHODOLOGY — 1000 random iterations.
5. DEFINITION — average intervals 0.257 / 0.225 / 0.2 m.

**Shape:** METHODOLOGY → EVIDENCE → INTERPRETATION → METHODOLOGY → DEFINITION.
**Persuasive move:** stress-test-with-replication.
**Flow notes:** "It is worth mentioning" interprets the difficulty; "For each result" describes replication; "In specific" gives the spacings.

---

## §6.10.1 · p.125 · ¶3

**Verbatim:**
> Due to the large gap between the two paths, the average calculation error for seven paths is 0.0001 m. When we insert eight paths, the error increase to 0.015m, while the error becomes 0.036 m if we insert nine paths. This is also validated by the CDF shown in Figure 28(b). This result shows that UMusic is able to recognize 7-9 paths which mainly reside in the six CIR taps (1.8 m).

1. EVIDENCE — 7 paths → 0.0001 m error.
2. EVIDENCE — 8 paths → 0.015 m; 9 → 0.036 m.
3. EVIDENCE — CDF in Figure 28(b) confirms.
4. INTERPRETATION (CLAIM) — UMusic resolves 7–9 paths in six CIR taps.

**Shape:** EVIDENCE → EVIDENCE → EVIDENCE → INTERPRETATION.
**Persuasive move:** quantitative-resolution-claim.
**Flow notes:** Three sentences of numbers chain into a one-sentence interpretation.

---

## §6.10.2 Performance under Hardware Imperfection · p.125 · ¶1

**Verbatim:**
> Our simulation for hardware imperfection strictly follows the parameters, specifically CFO, provided by the commodity device's datasheet. As specified in DW1000 datasheet [48], the oscillator has a maximum drift of ±20ppm, resulting in the CFO maximum at the ±69.8KHz. In our simulation, we manually insert the 11 different CFO from −100KHz to 100KHz to cover all potential cases even much worse than the datasheet specifies. Moreover, we impose the sampling offset and random initial phase offset to the simulated CIR data to complete the verification for hardware immunity. CFO, initial offset, random sampling time offset. The overall PDP calculation error is shown in Figure 29(a), where the average error is 0.016 m, which matches with the 8 paths calculation result in Section 6.10.1 obtained under no hardware imperfections. In specific, we show the impact of each hardware imperfection in Figure 29(b), 29(c), and 29(d), respectively. These results demonstrate that the distribution of PDP calculation error is constant even with various hardware imperfections, which verifies the immunity of our high-resolution PDP calculation.

1. METHODOLOGY (SCOPE) — simulations follow datasheet parameters.
2. DEFINITION — DW1000 drift ±20 ppm → CFO max ±69.8 kHz.
3. METHODOLOGY — sweep CFO from −100 to +100 kHz (worse than datasheet).
4. METHODOLOGY — impose STO and initial phase as well.
5. (Fragment — list of imperfections.)
6. EVIDENCE — average error 0.016 m, matches §6.10.1.
7. EVIDENCE — per-imperfection breakdown in Figure 29.
8. INTERPRETATION (CLAIM) — error distribution constant → immunity verified.

**Shape:** METHODOLOGY → DEFINITION → METHODOLOGY → METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION.
**Persuasive move:** stress-beyond-datasheet.
**Flow notes:** "even much worse than the datasheet specifies" stresses worst case; "These results demonstrate" closes the claim.

---

## §6.10.3 With/Without Computational Cost Reduction · p.125 · ¶1

**Verbatim:**
> We evaluate the time consumption for high-resolution PDP calculation. In this experiment, we reuse the simulated CIR data and their corresponding groundtruth PDP information for eight paths within 1.8 m range. We implement the downsampling design to reduce the computational cost on a Mac M1 computer, which is set to use a single CPU and a single thread for PDP calculation. This experiment compares the PDP calculation time and precision under the downsampling factors 1, 2, and 4. When we downsample by a factor of 4, we set 𝑝 and 𝑞 to be 48 in order to make them the multipliers of 𝐷 = 4. The detailed results are shown in Figure 30, where the average time consumption is 3.38 ms, 1.15 ms, and 0.125 ms under the downsampling factor of 1, 2, and 4 respectively. Compared to 𝐷=1, the time consumption is reduced by 2.94 and 27.1 times. Meanwhile, the accuracy experiences a negligible degradation due to downsampling, as shown in Figure 30(b). This result supports UMusic to operate in real-time since the interval between consecutive CIR collections is 50 ms. As UMusic is intended for in-car systems powered by the alternator, its energy consumption of UMusic is relatively low compared to other in-car systems.

1. METHODOLOGY — measure time consumption.
2. METHODOLOGY — reuse 8-path simulation data.
3. METHODOLOGY — Mac M1 single-thread.
4. METHODOLOGY (SCOPE) — compare D = 1, 2, 4.
5. METHODOLOGY (DEFINITION) — set p = q = 48 for D = 4.
6. EVIDENCE — 3.38 ms / 1.15 ms / 0.125 ms.
7. EVIDENCE — speedup of 2.94× and 27.1×.
8. INTERPRETATION (EVIDENCE) — negligible accuracy degradation.
9. CONSEQUENCE (CLAIM) — supports real-time (50 ms interval).
10. CLAIM (SCOPE) — alternator-powered ⇒ low energy concern.

**Shape:** METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION → CONSEQUENCE → CLAIM.
**Persuasive move:** speed-up-with-real-time-anchor.
**Flow notes:** "Meanwhile, the accuracy experiences a negligible degradation" couples cost with precision; "This result supports UMusic to operate in real-time" ties to the 50 ms budget; final sentence relaxes the energy concern.

---

## §7. Discussion and Future Work · p.126 · ¶1 (Impact of tall passenger)

**Verbatim:**
> Impact of tall passenger. In the experiment, UWB devices are mounted on the car's ceiling to minimize the impact of tall passengers. Even if an exceptionally tall passenger blocks the LoS path, UMusic can handle this effectively. Although PDP synchronization may be affected, this does not hinder occupancy detection, as the blockage also impacts longer paths, making the shortest affected path closer in length to the LoS path. Subsequently, this could be captured by UMusic to detect the exceptionally tall passenger.

1. METHODOLOGY (CAUSE) — ceiling-mounted sensors minimize tall-passenger impact.
2. CLAIM — UMusic handles even very tall passengers.
3. CONCESSION (CAUSE) — synchronization may be affected, but longer paths also blocked → shortest affected path approaches LoS.
4. CONSEQUENCE — UMusic can detect the tall passenger.

**Shape:** METHODOLOGY → CLAIM → CONCESSION → CONSEQUENCE.
**Persuasive move:** edge-case-conversion-into-feature.
**Flow notes:** "Although … this does not hinder" turns the concession; "Subsequently" derives a positive consequence.

---

## §7. · p.126 · ¶2 (Distinguishability between the passenger and large luggage)

**Verbatim:**
> Distinguishability between the passenger and large luggage. We note that luggage remains stationary, while passengers typically display unpredictable motion. This distinction allows UMusic to differentiate large luggage from passengers by tracking changes in propagation paths over time, reusing PDP data collected for aggregated detection, as studied in [21, 32, 87].

1. PREMISE — luggage static vs. passengers move.
2. CONSEQUENCE (CLAIM) — UMusic differentiates by tracking temporal path changes, reusing aggregated PDP.

**Shape:** PREMISE → CONSEQUENCE.
**Persuasive move:** distinguishability-from-motion.
**Flow notes:** "This distinction allows UMusic to" turns the contrast into a discriminator.

---

## §7. · p.126 · ¶3 (Extension to general sensing applications)

**Verbatim:**
> Extension to general sensing applications. While UMusic is tailored for in-car occupancy detection due to strong signal reflections from the vehicle's metal and compact structure, it can also be adapted for broader sensing applications. Its core feature, high-resolution PDP estimation, can be utilized in tasks like localization. By using multiple UWB sensors as anchors, the precise PDP of sensor links can enable accurate triangulation for localization.

1. CONCESSION (CLAIM) — tailored for in-car but adaptable.
2. CLAIM — core feature reusable for localization.
3. METHODOLOGY — multi-sensor anchors enable triangulation.

**Shape:** CONCESSION → CLAIM → METHODOLOGY.
**Persuasive move:** broaden-applicability.
**Flow notes:** "While … it can also be adapted" turns scope into extension; subsequent sentences sketch the extension.

---

## §7. · p.126 · ¶4 (Full support for HVAC and vital sign applications)

**Verbatim:**
> Full support for HVAC and vital sign applications. UMusic is designed to provide occupancy status, a prerequisite for HVAC systems, vital sign monitoring, and detect children left alone. Currently, UMusic excels at occupancy detection, identifying the number of passengers and their seating arrangements, which is suitable for HVAC applications. However, UMusic requires further enhancements to detect physiological signals such as heart rate, respiration rate, and body temperature for vital sign monitoring [83, 91] and other applications, which would be addressed in future work.

1. DEFINITION (CLAIM) — occupancy is a prerequisite for HVAC, vital signs, child-left-alone.
2. CLAIM — UMusic suffices for HVAC use cases today.
3. CONCESSION (BRIDGE) — vital-sign monitoring requires future work.

**Shape:** DEFINITION → CLAIM → CONCESSION.
**Persuasive move:** present-strength-with-future-scope.
**Flow notes:** "Currently … excels at" stakes today's claim; "However … requires further enhancements" delineates future work.

---

## §8. Related Work · p.126 · ¶1

**Verbatim:**
> Wireless sensing has been studied in numerous papers [27, 73] for indoor localization [24, 56, 72, 80], location tracking [47, 73, 79], floor mapping [31, 46], and motion tracking [84]. These designs, which are categorized by the wireless techniques utilized, have different advantages and limitations: (i), WiFi-based designs [24, 47, 56, 66, 68, 78] are easy for users to accept as the result of the popularity of WiFi [69, 70]. (ii), Acoustic [7, 90] and vision [12] based approaches have high precision while the privacy concerns are yet to be addressed. (iii), mmWave radar is also applied for sensing [30, 31, 36, 42, 58, 67, 73] and achieved both strong privacy reservation and high effectiveness. (iv), UWB is recently applied for precise localization [3, 43, 57, 60, 61, 63] and sensing [21, 32]. For instance, TALLA [63] achieves decimal-level localization and tracking precision using time difference of arrival (TDoA), derived from UWB communication. (v), Bluetooth [33, 65], LoRa [6, 15, 27, 74, 77], RFID [10, 75, 76] are also leveraged for sensing in various scenarios, where the granularity is not strictly required.

1. PREMISE — wireless sensing covers localization/tracking/mapping.
2. SCOPE (BRIDGE) — categorize by technique.
3. DEFINITION (CLAIM) — (i) WiFi: user-accepted due to popularity.
4. DEFINITION (CONCESSION) — (ii) acoustic/vision: precise but privacy-concerning.
5. DEFINITION (CLAIM) — (iii) mmWave: private and effective.
6. DEFINITION — (iv) UWB used for localization and sensing.
7. EXAMPLE — TALLA's TDoA precision.
8. DEFINITION — (v) Bluetooth/LoRa/RFID for coarser sensing.

**Shape:** PREMISE → SCOPE → DEFINITION → DEFINITION → DEFINITION → DEFINITION → EXAMPLE → DEFINITION.
**Persuasive move:** taxonomy-survey.
**Flow notes:** Five "(i)–(v)" tags categorize approaches by technique with paired advantages/limitations.

---

## §8. · p.126 · ¶2

**Verbatim:**
> For in-car scenarios, mmWave Radar [28, 30, 31, 42, 46, 73], vision-based [12] and acoustic-based [90] approaches have high sensitivity for occupancy detection. Vision-based solutions suffer from occlusions, while acoustic-based solutions still face privacy leakage issues. For instance, VeCare proposes the first Child Presence Detection (CPD) system that only utilizes car speakers and microphones. To ensure a robust solution, these approaches require additional hardware and associated installation costs. In addition, customizing a low-cost tag is an effective solution for in-car sensing [8]. This paper leverages UWB devices installed in the car for occupancy detection. Existing UWB sensing solutions [21, 32] have primarily focused on utilizing machine learning techniques, making them sensitive to changes in the in-car environment. For instance, CarOSense investigates the reuse of UWB keyless infrastructure through a novel deep-learning model called MaskMIMO to detect occupancy in each seat of a car [32]. UMusic, on the other hand, utilizes signal processing techniques making the solution more adaptive to the environmental effects. A combination of advanced ML techniques with UMusic can potentially create more robust hybrid models to detect and classify multiple occupancies in the various car models. Development and analysis of such models are left for future work.

1. PREMISE — mmWave / vision / acoustic offer high sensitivity for in-car detection.
2. CONCESSION (CONTRAST) — vision suffers occlusion; acoustic suffers privacy.
3. EXAMPLE — VeCare's CPD system using car speakers/microphones.
4. CONSEQUENCE — additional hardware/installation costs.
5. EXAMPLE (CLAIM) — low-cost custom tags also viable.
6. CLAIM — this paper uses installed UWB.
7. CONCESSION — existing UWB sensing uses ML, hence environment-sensitive.
8. EXAMPLE — CarOSense's MaskMIMO.
9. CONTRAST (CLAIM) — UMusic uses signal processing → more adaptive.
10. CLAIM (BRIDGE) — hybrid ML+UMusic could be more robust, future work.

**Shape:** PREMISE → CONCESSION → EXAMPLE → CONSEQUENCE → EXAMPLE → CLAIM → CONCESSION → EXAMPLE → CONTRAST → CLAIM.
**Persuasive move:** position-against-incumbents.
**Flow notes:** "Vision-based solutions suffer … while acoustic-based … face privacy" balances two flaws; "UMusic, on the other hand, utilizes signal processing" stakes the differentiating claim; final sentence opens hybrid future work.

---

## §9. Conclusion · p.126 · ¶1

**Verbatim:**
> This paper introduces UMusic, a system that uses commodity UWB devices to precisely detect car occupancy via lightweight signal processing techniques. UMusic converts CIR data into the frequency domain to obtain the channel frequency response, which is used to calculate the high-resolution PDP via the MUSIC algorithm. Through the comparison between the PDP of empty and occupied environments, UMusic is able to detect the occupancy status. We evaluate UMusic in a car with one or more passengers under various scenarios, including stationary and driving conditions. The experiments show that UMusic achieves an aggregated accuracy of 99.4%, highlighting its effectiveness in practical scenarios.

1. CLAIM — introduces UMusic for commodity UWB occupancy detection.
2. METHODOLOGY — CIR → CFR → MUSIC-based high-resolution PDP.
3. METHODOLOGY (CLAIM) — empty-vs-occupied PDP comparison yields detection.
4. METHODOLOGY — evaluated in stationary and driving scenarios.
5. EVIDENCE (CLAIM) — 99.4% aggregated accuracy demonstrates effectiveness.

**Shape:** CLAIM → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE.
**Persuasive move:** restate-thesis-with-headline-number.
**Flow notes:** Standard conclusion pipeline: introduce → mechanism → result; closes with the headline accuracy.

---

## Endnotes

- **Paragraph count:** 60 prose paragraphs annotated (Introduction 8; §2.1 2; §2.2 1; §2.3 1; §3 1; §4 prelude 1; §4.1 prose paragraphs 6 [excluding pure equation continuations were grouped where needed]; §4.1.1 2; §4.1.2 3; §4.1.3 4; §4.1.4 4; §4.2 2; §5 prelude 1; §5.1 3; §5.2 4; §5.3 1; §6.1 4; §6.2 1; §6.2.1 1; §6.2.2 1; §6.3 1; §6.4 1; §6.5 1; §6.6 1; §6.7 1; §6.8 1; §6.9 2; §6.10 1; §6.10.1 3; §6.10.2 1; §6.10.3 1; §7 4; §8 2; §9 1).
- **Sentence count:** ~290 sentences annotated across the body (each numbered within its paragraph).
- **Three most frequent paragraph shapes:**
  1. METHODOLOGY-led sequential pipelines (METHODOLOGY → METHODOLOGY → … → EVIDENCE / CONSEQUENCE / CLAIM) — dominant in §4.1.x, §5.x, §6.1, §6.10.3.
  2. EVIDENCE / METHODOLOGY → INTERPRETATION / CLAIM closing pattern (e.g., METHODOLOGY → EVIDENCE → INTERPRETATION) — recurrent in §6.x evaluation paragraphs.
  3. CLAIM → CONCESSION → CONTRAST / CONSEQUENCE — the rhetorical move that introduces a positioning then preempts an objection — used in §1¶2/¶4, §2.3, §4.1.4, §6.7, §6.9, §7 paragraphs.
