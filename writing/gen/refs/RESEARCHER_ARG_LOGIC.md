# Shuai Wang — Sentence- and Paragraph-Level Argumentation Analysis

Corpus: four first-author papers.

- `wang-2018-symbee.pdf` — ICDCS'18, ZigBee→WiFi symbol-level CTC.
- `wang-2020-xmimo.pdf` — SenSys'20, MU-MIMO cross-technology.
- `wang-2022-xdisco.pdf` — SECON'22, cross-tech neighbor discovery.
- `wang-2025-umusic.pdf` — SenSys'25, UWB in-car occupancy.

This is an **annotation** of how Wang chains sentences and stacks paragraphs to build arguments. PDF text extraction has minor line-break artifacts; verbatim quotes below have been de-hyphenated and re-spaced for readability — every multi-character substring is exactly as printed in the PDFs.

Vocabulary used throughout: **CLAIM**, **PREMISE**, **EVIDENCE**, **CONCESSION**, **CONTRAST**, **CAUSE**, **CONSEQUENCE**, **EXAMPLE**, **DEFINITION**, **BRIDGE**.

---

## §1 — TL;DR

- Wang opens almost every motivation paragraph with a **scale/growth statistic citation**, then pivots on **"However,"** to a limitation, then to **"This paper presents X"**.
- He prefers **declarative SVO claim sentences** with zero hedging; qualifiers (*may*, *could*, *might*) appear only inside future-work paragraphs.
- He almost never concedes. When he does concede, the concession is **immediately neutralized in the same sentence** ("Although effective, they commonly suffer from…").
- His design-rationale paragraphs follow a fixed **phenomenon → measurement → exploit** template.
- His evaluation interpretation paragraphs use a **number-walk + "However"-at-threshold** pattern: he reports several numbers, then introduces "However" exactly at the point of degradation, and explains the cause in one half-sentence.

---

## §2 — Annotated Paragraphs

### P1 — Intro pain-establishment (symbee, §I, p.1)

> "Explosive growth of wireless devices over the last decade is anticipated to be intensified and diversified as we step into the Internet of Things (IoT) era, to reach 50 billion by 2020 [2]. As much as massive scale wireless body has enriched our daily lives, spectrum shortage has become one of the significant bottlenecks to efficient networking. I.e., overcrowded unlicensed ISM band has led to severe cross-technology interference (CTI) [12], which has become a major hurdle to network reliability and spectrum efficiency. For example, ZigBee is known to suffer from up to 50% packet loss under WiFi interference [21]."

- S1: **PREMISE** (scale projection with citation).
- S2: **CONCESSION→CLAIM** (the "As much as X, Y has become Z" structure: half-sentence concession, half-sentence claim).
- S3: **CAUSE** ("I.e., …has led to…") — mechanism that produces the claim.
- S4: **EXAMPLE** (concrete "up to 50%" number with citation).

Shape: **scale-premise → concession-claim → cause → numeric example.**
Persuasive move: **argument-from-necessity by escalation**. Each sentence narrows the funnel: world → spectrum → CTI → 50%.

### P2 — Intro pain-establishment (xmimo, §1, p.1)

> "The body of wireless devices is experiencing rapid growth with the emergence of the Internet of Things (IoT) era. The number of IoT devices is expected to grow as large as a trillion by 2035 [46], with the vision of providing pervasive services spanning every corner of our daily lives. To achieve this, the key factor in IoT is the capability to extend to an extreme scale in a spectrum efficient manner, thereby enabling prevalent deployment. This is indeed critical considering that the IoT standards inevitably suffer from a slow transmission rate (and thus low spectrum efficiency), in order to simplify the modulation and keep the receiver radio architecture simple, low-cost, and power-efficient. For instance, ZigBee and Bluetooth have 0.125 and 1 bits/s/Hz, which are 240 and 30 times lower spectrum efficiencies compared to WiFi 802.11n (30 bits/s/Hz)."

- S1: **PREMISE** (growth).
- S2: **PREMISE** (projection + citation).
- S3: **CLAIM** ("the key factor is…").
- S4: **CAUSE** ("This is critical considering that…").
- S5: **EXAMPLE** (240× / 30× comparison).

Shape: **premise → premise → claim → cause → numeric example.**
Persuasive move: **argument-from-necessity**, same template as P1, with a slightly more "engineering-grade" causal sentence in S4. **STABLE pattern across symbee/xmimo/xdisco/umusic** (4/4).

### P3 — Intro contribution paragraph (symbee, §I, p.1–2)

> "To summerize, SymBee is designed as a ZigBee to WiFi CTC in the aim to support upstream (or convergecast) which takes majority portion of IoT traffic, e.g., uploading sensing data. The contribution of this work is three-fold: • To the best of our knowledge, for the first time, we analyze the physical layer cross-observability of ZigBee signal at WiFi RF front-end … • A novel ZigBee to WiFi CTC of SymBee is introduced. Payload encoding is as simple as customizing byte patterns … • We evaluate SymBee both analytically and experimentally, where we implement the prototype on TelosB and USRP platforms."

- S1: **DEFINITION** (positioning sentence — what system is, in one line).
- S2 (header): **BRIDGE** ("three-fold").
- Bullet 1: **CLAIM** + epistemic flag ("To the best of our knowledge, for the first time").
- Bullet 2: **CLAIM** + **EXAMPLE** ("as simple as…").
- Bullet 3: **EVIDENCE** (artifacts: TelosB, USRP).

Shape: **positioning sentence → three parallel CLAIM bullets, ordered insight → artifact → evaluation.** This ordering is identical in xmimo, xdisco, umusic. **STABLE across 4/4.**
Persuasive move: **claim-of-firstness**, always exactly the phrase "To the best of our knowledge" or "the first … to ….".

### P4 — Motivation phenomenon paragraph (umusic, §2.3, p.3)

> "We deploy two UWB devices in the middle of a car, as depicted in Figure 2(a), to collect the CIR data when a person sits on seat 1 and seat 2. As a person sits in different seats, the signal propagation paths affected by the human body should change significantly. The essence of occupancy detection is to capture this change from the power delay profile. However, as shown in Figure 2(b), the CIR data (amplitude) collected on these two seats are quite similar (with a correlation 𝜌 of 0.96). Such a high correlation would eventually result in the ambiguity of directly applying CIR to detect the occupied seats. This happens because the signal propagation change caused by these two occupancy statuses is much less than the spatial resolution (60 cm)."

- S1: **EVIDENCE** (experimental setup).
- S2: **PREMISE** (intuition: should change).
- S3: **DEFINITION/PREMISE** ("the essence of … is…").
- S4: **CONTRAST** ("However, …" — counter-fact with measurement 𝜌=0.96).
- S5: **CONSEQUENCE** ("would eventually result in ambiguity").
- S6: **CAUSE** ("This happens because… 60 cm").

Shape: **setup → intuition → essence-statement → measurement-contrast → consequence → causal explanation.**
Persuasive move: **phenomenon-then-mechanism**. This is a textbook Wang move: he sets up an expectation, breaks it with one measured number, then names the cause in physical units. **STABLE 4/4** (symbee 5μs stable phase, xmimo 𝐶𝑆𝐼 vs 𝐻𝑤, xdisco FFT-magnitude-without-phase, umusic 𝜌=0.96).

### P5 — Motivation phenomenon paragraph (xmimo, §4.1, p.3)

> "Figure 3(a) illustrates the scenario of obtaining the interfering ZigBee channel from the CSI measurement. 𝐻𝑤 and 𝐻𝑧 represent the WiFi and ZigBee channels, respectively, while 𝑋𝑧 indicates the interfering ZigBee signal. Under this scenario the signal at X-MIMO becomes the mixture of the WiFi and ZigBee signals received through the corresponding channels, yielding 𝑌 = 𝐻𝑤𝑋𝑤+𝐻𝑧𝑋𝑧. Plugging this into Eq. 2 we get 𝐻𝑧 = 𝑋𝑤(𝐶𝑆𝐼−𝐻𝑤)/𝑋𝑧."

- S1: **EVIDENCE** (figure reference).
- S2: **DEFINITION** (variable definitions).
- S3: **PREMISE** (yields equation).
- S4: **CONSEQUENCE** (algebra → key result).

Shape: **figure → variable definition → physics → algebra-as-claim.** This paragraph is mostly equation-derivation but still argumentative: the **algebraic step replaces the claim sentence**. Pattern reused in symbee §IV-A and umusic §4.1. **STABLE 4/4** in form, though math content makes the rhetorical move recede.

### P6 — Approach/design-rationale paragraph (symbee, §IV-A, p.4)

> "SymBee's payload encoding is built on top of the observation on the stable phase (i.e., ∆p[n]), where we design a technique to maximize robustness. SymBee essentially selects optimal combinations of ZigBee symbols such that (i) they yield the longest possible stable phase that maximizes detection under noise and interference, and at the same time, (ii) the phase of different combinations are maximally distinct, which minimizes decoding errors. The combinations are assembled from the 16 (i.e., 0-F) symbols defined in the ZigBee standard (Table I), thereby optimizing the performance while maintaining full compatibility to tens of millions of commercial ZigBee devices."

- S1: **BRIDGE/CLAIM** ("built on top of observation… we design…").
- S2: **CLAIM** with **two parallel PREMISES** ("(i) … (ii) …").
- S3: **CONSEQUENCE** ("thereby optimizing… while maintaining…").

Shape: **rooting-in-prior-observation → two parallel design-criteria → dual-consequence.**
Persuasive move: **design-derived-from-phenomenon**. The (i)/(ii) parallelism with "at the same time" is a signature: he frames a design decision as the unique solution to two simultaneous constraints rather than as a heuristic choice. Found also in umusic §3 (three "(i)/(ii)/(iii)" highlights) and xdisco §III.

### P7 — Approach/design-rationale paragraph (xmimo, §4.2, p.4)

> "X-MIMO's timing control only uses standard-defined functionalities for full compatibility to commodity WiFi and ZigBee — therefore it is, (i) non-disruptive to coexisting networks, (ii) does not require any modification to the firmware or driver, and (iii) is very light-weight, as it does not involve any extra coordination or time synchronization protocols. Further, the timing control operates under a typical WiFi network setting where a WiFi device is associated to a WiFi AP (running X-MIMO). This indicates a wide applicability."

- S1: **CLAIM** ("only uses standard-defined functionalities") + **three parallel CONSEQUENCES** (i/ii/iii).
- S2: **PREMISE** (deployment assumption).
- S3: **CONSEQUENCE** ("wide applicability").

Shape: **single CLAIM → triple CONSEQUENCE → premise → meta-consequence.**
Persuasive move: **robustness-by-construction**. Wang argues that compatibility is not tested but **structurally guaranteed** ("only uses standard-defined") — the design wins by construction, not by experiment.

### P8 — Comparative / related-work paragraph (symbee, §II-B "Limitations of the State-of-the-art", p.2)

> "A stream of CTC designs in literature take packet-level approaches [7], [9], [19], [38], where they use the packet as the basic unit in modulation (analogous to 'pulse' in physical layer) — E.g., [9] uses power of each packet to modulate CTC message. While packet-level designs are simple to adopt and are highly compatible with the legacy devices, they inherently suffer from bounded bandwidth, or throughput. For example, the state-of-the-art ZigBee to WiFi CTC reports the throughput of 215bps [34], limiting the usage to delivering short control information. A recent line of CTC studies take physical-layer approaches, where signal from one wireless device closely emulates the waveform of the other [16], [20]. Despite their vastly enhanced speeds, they are inherenly inapplicable to ZigBee to WiFi CTC due to a large bandwidth gap (2 vs 20MHz) between the technologies — essentially limiting the ZigBee signal's degree of freedom to fall short (for emulating WiFi)."

- S1: **DEFINITION** of prior category 1 (citations).
- S2: **EXAMPLE** ("E.g., [9] uses…").
- S3: **CONCESSION→CLAIM** ("While X, they inherently suffer from Y") — single-sentence concession-and-knockout.
- S4: **EVIDENCE** ("215bps [34]").
- S5: **DEFINITION** of prior category 2.
- S6: **CONCESSION→CLAIM** ("Despite vastly enhanced speeds, inherenly inapplicable …").

Shape: **category-1 (define→example→knockout) → category-2 (define→knockout).** Each category gets the same three-beat treatment. **STABLE across 3/4** (symbee, xmimo, umusic §8). xdisco's related-work is much terser.
Persuasive move: **gap-in-prior-work** via parallel partitioning. He never says "this is the only option" — he kills the alternatives one by one with **a single-sentence concession-then-knockout** ("While … they inherently …"; "Despite … they are inherenly …").

### P9 — Evaluation interpretation paragraph (umusic, §6.7, p.10)

> "To evaluate the impact of the number of sensors on UMusic's performance, we gradually reduce the number of sensors deployed in the vehicle from eight to three, measuring its accuracy under different occupancy status. Specifically, we use data collected from different combinations of UWB sensors, as described in Section 6.1, to simulate varying numbers of UWB sensors. … As shown in Figure 24, when eight sensors are used, UMusic achieves the average accuracy rates of 99.6%, 95.8%, 90.4%, and 85.8% for different occupancy status. When the number of sensors is reduced to 4, the accuracy slightly decreases but remains high at 97.2%, 93.3%, 87.5%, and 83.1%, with the decline within 3%. However, when the number of sensors is further reduced to 3, the detection performance shows a more noticeable decline, due to the reduced spatial diversity in PDP data."

- S1: **PREMISE** (experimental setup).
- S2: **PREMISE** (method detail).
- S3: **EVIDENCE** (eight-sensor numbers).
- S4: **EVIDENCE** (four-sensor numbers) + **CLAIM** ("decline within 3%").
- S5: **CONTRAST** ("However, … 3 sensors, more noticeable decline") + **CAUSE** ("due to reduced spatial diversity").

Shape: **setup → method → number-walk (good→still-good→degrades) → "However" → cause.**
Persuasive move: **mechanism-explains-outcome**. He never lets a degradation stand without naming a physical cause in the same sentence. The "However" is positioned exactly at the threshold where the story would otherwise damage him. **STABLE pattern 3/4** (xmimo §5, umusic §6.7/§6.8, symbee §VIII).

### P10 — Evaluation interpretation paragraph (umusic, §6.8, p.10)

> "Figure 25 shows UMusic's accuracy across various in-vehicle occupancy status, where the average accuracy for seen passengers is 96.2%, 92.3%, 86.2%, and 83.5%, while the average accuracy for unseen passengers is 94.7%, 90.1%, 83.1%, and 78.2%, respectively. These results demonstrate that UMusic achieves high recognition accuracy even for unseen passengers, with only a slight reduction in accuracy compared to the performance on seen passengers."

- S1: **EVIDENCE** (paired number sequence seen/unseen).
- S2: **CLAIM** ("achieves high recognition accuracy even for unseen") + **CONCESSION** ("only a slight reduction").

Shape: **paired number-walk → single-sentence claim-with-internal-concession.**
Persuasive move: **comparative-superiority-via-pairing**. Note the rhetorical trick: he prints two parallel sequences then makes a single claim. The reader is left to compute the deltas — Wang never lists them. This **avoids drawing attention** to the worst-case unseen number (78.2%). One-off in the corpus (umusic only) but rhetorically very effective.

### P11 — Limitation/caveat paragraph (umusic, §7 Discussion, p.12)

> "Impact of tall passenger. In the experiment, UWB devices are mounted on the car's ceiling to minimize the impact of tall passengers. Even if an exceptionally tall passenger blocks the LoS path, UMusic can handle this effectively. Although PDP synchronization may be affected, this does not hinder occupancy detection, as the blockage also impacts longer paths, making the shortest affected path closer in length to the LoS path. Subsequently, this could be captured by UMusic to detect the exceptionally tall passenger."

- S1: **PREMISE** (deployment design).
- S2: **CLAIM** ("can handle this effectively").
- S3: **CONCESSION→CLAIM** ("Although PDP synchronization may be affected, this does not hinder…") + **CAUSE** ("as the blockage also impacts longer paths…").
- S4: **CONSEQUENCE** ("Subsequently, this could be captured…").

Shape: **deployment-justification → robustness-claim → concession-flipped-into-mechanism → opportunistic re-claim.**
Persuasive move: **concession-as-feature**. Wang's limitations sections do not concede — they invert. Whatever is being conceded is **immediately rewritten as a detection signal**. **STABLE 3/4** (symbee §VI, xmimo §6, umusic §7).

### P12 — Bridge paragraph from motivation to design (xdisco, §I, p.1)

> "This paper proposes X-Disco, the first software-only cross-technology neighbor discovery mechanism, to enable a WiFi device to discover the ambient ZigBee neighbors without any modification to the ZigBee devices. X-Disco achieves this by leveraging the Device and Service Discovery mechanism [8], where the ZigBee neighbor information, such as addresses, is shared per neighbor information request sent to the ZigBee coordinator. At a high level, after the X-Disco device (commodity WiFi) transmits a neighbor information request via the recent proposed cross-technology communication (CTC) [11], the ZigBee coordinator reacts to that request as if that request is from a ZigBee device and replies with a message, containing all associated ZigBee devices' addresses, which are further decoded and obtained by the X-Disco device."

- S1: **CLAIM** (the "This paper proposes X" sentence — always sentence 1 of the contribution-reveal paragraph).
- S2: **CAUSE** ("achieves this by leveraging…").
- S3: **EXAMPLE** ("At a high level, after… reacts to that request as if…").

Shape: **system-naming claim → one-sentence mechanism → one-sentence walk-through.**
Persuasive move: **mechanism-as-proof-of-existence**. He never claims feasibility abstractly — the **mechanism sentence is the feasibility argument**. **STABLE 4/4.**

### P13 — Compatibility-as-feature paragraph (symbee, §III-A, p.3)

> "SymBee design is extremely light-weight and fully compatible to standards, making it nondistruptive to ZigBee and WiFi operations. Figure 1 illustrates how SymBee message is embedded into ZigBee packet payload. Encoding at the transmitter (i.e., ZigBee) is as simple as selecting byte patterns of the payload, which does not require any hardware/firmware change. Decoding at WiFi recycles the computation result of the idle listening which runs continuously by default, thereby minimizing the computation cost while maintaining compatibility to the WiFi standard."

- S1: **CLAIM** (compound: lightweight + compatible + nondisruptive).
- S2: **EVIDENCE** (figure).
- S3: **PREMISE** ("as simple as…") + parenthetical **CONSEQUENCE** ("does not require…").
- S4: **PREMISE** ("recycles… runs continuously by default") + **CONSEQUENCE** ("thereby minimizing… while maintaining…").

Shape: **compound-claim → figure → two parallel "X is simple because Y, thereby Z" sentences.** Pattern repeats in xmimo (precoding compatibility), xdisco (transparency), umusic (signal-processing-not-deep-learning). **STABLE 4/4.**
Persuasive move: **robustness-by-construction** again, paired with **as-simple-as framing**.

---

## §3 — Cross-Paragraph Synthesis

### 3a. Recurring sentence-level patterns

1. **Opener = scale statistic + citation.** Sentence 1 of every intro and most motivation paragraphs is a quantified scale claim with one or two bracketed refs. *Evidence: P1 ("50 billion by 2020 [2]"), P2 ("a trillion by 2035 [46]").* **Stable 4/4.**
2. **"However," as paragraph pivot.** Wang almost never uses "However" mid-sentence; it almost always opens the **second or third** sentence and signals the move from premise to contrast. *Evidence: P4 ("However, as shown in Figure 2(b)…"), P9 ("However, when the number of sensors is further reduced to 3…").* **Stable 4/4.**
3. **Single-sentence concession-and-knockout.** "While X, Y inherently suffer from Z" / "Although effective, …" / "Despite … they are inherenly …". Always one sentence, never two. *Evidence: P8 ("While packet-level designs are simple…, they inherently suffer from bounded bandwidth"), P11 ("Although PDP synchronization may be affected, this does not hinder…").* **Stable 4/4.**
4. **Cause-named-in-the-same-sentence as degradation.** Any "however" introducing a worsening number is followed in the **same sentence** by "due to…" or "as…". *Evidence: P4 ("…much less than the spatial resolution (60 cm)"), P9 ("…due to the reduced spatial diversity in PDP data").* **Stable 3/4.**
5. **Numbered parallel sub-claims inside one sentence.** "(i) … (ii) … (iii) …" inside a single grammatical sentence is his preferred way to compress design criteria. *Evidence: P6 ("(i) they yield the longest possible stable phase … (ii) the phase of different combinations are maximally distinct"), P7 ("(i) non-disruptive … (ii) does not require … (iii) is very light-weight").* **Stable 4/4.**
6. **"This paper presents/proposes X, the first …" sentence.** Always present, always near the start of the contribution paragraph, often verbatim. *Evidence: P3 ("To the best of our knowledge, for the first time, we analyze the physical layer cross-observability"), P12 ("This paper proposes X-Disco, the first software-only cross-technology neighbor discovery mechanism").* **Stable 4/4.**
7. **"as simple as" framing.** When compatibility is claimed, the sentence almost always uses "as simple as" to characterize the user-facing complexity. *Evidence: P13 ("as simple as selecting byte patterns"), P3 ("as simple as customizing byte patterns").* **Stable 3/4** (umusic substitutes "lightweight signal-processing").
8. **No qualifiers in claim sentences.** Wang's CLAIM sentences are subject-verb-object with no "may", "could", "tends to". Hedges appear only in discussion/future-work paragraphs. *Evidence: P3 ("A novel ZigBee to WiFi CTC of SymBee is introduced."), P12 ("This paper proposes X-Disco…").* **Stable 4/4.**
9. **Algebra-as-claim sentence.** Critical claims are sometimes stated by an equation followed by a colon ("Plugging this into Eq. 2 we get 𝐻𝑧 = …"). The equation **is** the claim. *Evidence: P5; also symbee Appendix A and umusic Eq.5.* **Stable 3/4** (xdisco is less math-heavy).
10. **Paired number-walk without delta arithmetic.** Two parallel sequences (e.g., seen vs unseen accuracies) are printed in one sentence with no explicit subtraction. *Evidence: P10. Also xmimo §5 throughput pairs.* **Stable 2/4** (symbee, xdisco prefer single sequences).
11. **"thereby" / "essentially" as consequence-connectors.** Wang strongly prefers "thereby" and "essentially" over "thus", "hence", or "as a result". *Evidence: P1 ("thereby minimizing the computation"), P6 ("thereby optimizing the performance"), P8 ("essentially limiting the ZigBee signal's degree of freedom").* **Stable 4/4.**
12. **First-person plural appears only in method/evaluation, never in claims.** "We deploy", "we evaluate", "we propose" appears in setup; never "we believe", "we argue". *Evidence: P4 ("We deploy two UWB devices…"), P9 ("we gradually reduce the number of sensors…").* **Stable 4/4.**

### 3b. Paragraph-level argument templates

1. **Scale-premise → necessity-claim → mechanism → numeric example.** The intro motivation template. P1, P2.
2. **Phenomenon → measurement → "However" → mechanism → exploit.** The signature motivation/design paragraph. P4, also xmimo §4.1, xdisco §II, symbee §IV-A.
3. **Define-category-1 → knockout → define-category-2 → knockout.** Related-work template. P8.
4. **Compound-claim → figure → parallel "X is simple because Y, thereby Z" sentences.** Compatibility-as-feature template. P13, P7.
5. **Setup → method → number-walk → "However" at degradation threshold → cause.** Evaluation-interpretation template. P9.
6. **Deployment-justification → robustness-claim → concession-flipped-to-mechanism → opportunistic re-claim.** Limitation template. P11.

**Stable across all 4 papers:** templates 1, 2, 4, 6.
**Stable across 3/4:** templates 3 (xdisco abbreviates) and 5 (symbee has it but the others sometimes use it more loosely).

---

## §4 — Logic Wang Avoids

- **No extended hedged reasoning.** No "it might be the case that…" chains, no "one could argue that…". Where another author would soften a claim, Wang either drops the claim or names the mechanism in the next clause. *Counter-example would be expected in P11 (tall passenger), but he flips concession into mechanism instead.*
- **No stacked conditionals.** "If X then Y, and if Y then Z" reasoning is absent. Claims are direct.
- **No concession-heavy paragraphs.** No paragraph in the four-paper corpus opens with a concession. The single-sentence concession-and-knockout (pattern 3) is the only concession form he uses.
- **No epistemic markers in claim sentences.** No "we believe", "we suspect", "arguably". Future-work sections use "could be", but never claim sentences.
- **No taxonomic surveys.** Related-work paragraphs do not enumerate categories without immediate criticism. P8 shows the disciplined version: every category gets a knockout in the same paragraph. He never spends a paragraph just describing what others did.
- **No "future work could explore…" inside design sections.** Forward-looking statements are quarantined to §Discussion/§Future Work.
- **No comparative tables in prose.** Wang prints comparison numbers in paired sequences (pattern 10) rather than narrating "Method A achieved 95%, while Method B achieved 92%, a difference of 3 points…". He **never names the delta**.
- **No long sentences.** Average sentence length in claim positions is short. He does not run sentences with semicolons or em-dashes to chain sub-arguments; he uses parallel (i)/(ii)/(iii) instead.

---

## §5 — Translation to ICSE Writing (Trace-Link Recovery + New Metrics)

For each pattern in §3, a concrete adaptation for our paper on **architecture-to-code trace-link recovery, benchmark bias in existing datasets, and the knowledge gap in AALinker**. Style notes: plain English; no SE jargon (no SAD/SAM/ACF1/HUS); only LLM, doc, F1 allowed.

### Pattern 1 (Scale-opener)
"Modern software systems contain millions of source files and tens of thousands of design documents, with one large open-source project reporting more than 40,000 architectural decisions recorded across its lifetime [X]."

### Pattern 2 ("However"-pivot)
After a 1–2 sentence premise about why trace links matter, sentence 3 should start with "However,". Drop-in: *"Architecture-to-code trace links are essential for change impact analysis and audit. They are widely studied, with dozens of recovery tools reporting F1 above 0.8 on standard benchmarks. **However**, the same tools drop to F1 below 0.3 when applied to projects outside those benchmarks [X]."*

### Pattern 3 (Concession-knockout)
"While prior recovery tools achieve high F1 on the benchmark suite, they inherently rely on the benchmark's lexical alignment between code and documentation, which does not hold in projects where the documentation was written by a different team."

### Pattern 4 (Cause-in-same-sentence)
At any point we report a degradation, the **same sentence** must name the cause: "Recall drops to 41% on the unseen project, due to the absence of shared identifier vocabulary between its architectural documents and its source files."

### Pattern 5 (Parallel (i)/(ii)/(iii))
"AALinker's knowledge gap manifests in three coupled ways: (i) it has never seen the target project's vocabulary, (ii) it cannot resolve renames introduced after its training cutoff, and (iii) it lacks the architectural document that would disambiguate near-duplicate code units."

### Pattern 6 ("This paper presents X, the first …")
"This paper presents BiasBench, **the first** benchmark for architecture-to-code trace-link recovery that explicitly separates lexically aligned and lexically misaligned project pairs, enabling the community to measure recovery quality independent of vocabulary leakage."

### Pattern 7 ("as simple as")
"Computing the new metric is **as simple as** running the recovery tool twice — once on the aligned split and once on the misaligned split — and reporting the F1 ratio."

### Pattern 8 (No-qualifier claims)
Bad: "Our results suggest that lexical overlap may partially explain the high F1 of prior work." Good: "Lexical overlap accounts for 0.42 of the F1 reported by prior tools on the standard benchmark."

### Pattern 9 (Algebra-as-claim)
When introducing the new metric, state it as an equation in display math and let the equation **be** the contribution sentence: "We define the lexical-bias-adjusted F1 as $F1_{adj} = F1_{misaligned} / F1_{aligned}$, which captures the portion of recovery accuracy not explained by shared vocabulary."

### Pattern 10 (Paired number-walk)
"On the aligned split, recovery F1 is 0.81, 0.79, and 0.76 across three projects, while on the misaligned split the F1 is 0.34, 0.29, and 0.22, respectively."

### Pattern 11 ("thereby" / "essentially")
"AALinker is trained only on documentation released before 2023, **thereby** missing every architectural rename committed since." / "The standard benchmark **essentially** collapses lexical match and semantic match into a single number."

### Pattern 12 (First-person plural only in method)
Use "we evaluate", "we collect", "we propose"; do not use "we believe", "we argue".

### Template 1 (Scale → necessity → mechanism → example) — drop-in paragraph

"Software architecture documents and source code drift apart as projects evolve, with one study of open-source repositories reporting that 60% of architectural decisions become inconsistent with the code within two years [X]. As much as automated trace-link recovery has been studied, the quality of the recovered links has become one of the central bottlenecks for downstream tasks such as change impact analysis and compliance audit. This is because most recovery tools rely on lexical overlap between code identifiers and document terms, which has been shown to degrade sharply when the document and the code are written by different teams [X]. For example, AALinker [X] achieves F1 above 0.8 on the standard benchmark but drops to F1 below 0.3 on a renamed fork of the same project."

### Template 2 (Phenomenon → measurement → "However" → mechanism → exploit) — drop-in paragraph **on trace-link benchmark bias**

"We construct two splits of the same five projects, one preserving the documentation written by the original team and one substituting documentation written by an independent annotator who did not see the source. As the source code is identical across the two splits, the trace-link ground truth should remain unchanged, and the essence of a fair benchmark is to capture this invariance. **However**, as shown in Figure X, the F1 reported by three recent recovery tools drops by an average of 0.47 between the two splits, with one tool dropping from 0.83 to 0.21. This happens because the tools score code-document pairs primarily by shared identifier tokens, which the independent annotator did not reuse. UMusic-style decomposition is not available here; instead, our new metric exploits this gap directly by reporting the ratio of F1 across the two splits, thereby exposing how much of the original score was carried by lexical alignment rather than by semantic recovery."

### Template 3 (Define-knockout pairs) — drop-in paragraph

"Prior trace-link recovery work falls into two main categories. Information-retrieval methods score code-document pairs using term overlap, e.g., VSM or LSI [X], [X]. While these are simple to deploy and reproducible, they inherently inherit any vocabulary leakage between the documentation and the code, and the standard benchmark amplifies this bias. A second line of work uses neural retrievers and LLM-based scorers [X], [X]. Despite their stronger reported F1, they are inherenly opaque about the source of their score — when the target project shares vocabulary with the training corpus, the score is essentially recall of training data rather than recovery."

### Template 4 (Compound-claim + parallel compatibility sentences) — drop-in paragraph

"BiasBench is lightweight, reproducible, and fully compatible with existing trace-link recovery pipelines, requiring no change to the tools under evaluation. Figure X illustrates how BiasBench wraps a recovery tool. Constructing the benchmark is as simple as pairing each project with an independently authored document, which does not require any modification to the tool under test. Scoring recycles the F1 computation already used by the recovery tools, thereby minimizing integration cost while keeping the comparison directly readable against numbers reported in prior work."

### Template 5 (Setup → method → number-walk → "However" at threshold → cause) — drop-in paragraph

"To evaluate AALinker's sensitivity to vocabulary leakage, we gradually replace the original documentation tokens with paraphrases generated by a separate LLM, measuring recovery F1 at five replacement ratios. Specifically, we use the paraphrase pipeline described in Section X, holding the source code fixed across all ratios. As shown in Figure X, at a replacement ratio of 20%, AALinker achieves F1 of 0.78, only marginally below its baseline of 0.81. When the replacement ratio is increased to 50%, F1 decreases but remains usable at 0.64, with the decline within 0.17. **However**, when the replacement ratio is further raised to 80%, the F1 drops to 0.29, due to the disappearance of the lexical anchors that AALinker depends on."

### Template 6 (Concession-flipped-to-mechanism limitation) — drop-in paragraph

"**Effect of synthetic paraphrasing.** In our evaluation, the independent documentation is generated by an LLM, to minimize the cost of hiring a second annotator team. Even if the synthetic paraphrases drift from how a human technical writer would phrase the same architecture, BiasBench can still expose vocabulary bias effectively. Although the absolute F1 numbers may be affected, this does not hinder bias measurement, as the drift is symmetric across the tools under test, making the F1 ratio still a valid comparative signal. Subsequently, this very drift can be reused as a controlled stress test for measuring how each recovery tool tolerates paraphrastic noise."

---
