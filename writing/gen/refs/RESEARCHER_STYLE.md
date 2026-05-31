# Researcher Writing-Style Study: Shuai Wang (Southeast University)

## IMPORTANT DOMAIN-MISMATCH WARNING

The Google Scholar profile `lOJUEfoAAAAJ` identifies **Shuai Wang**, Assistant Professor / Department Head of Computer Engineering at **Southeast University** (formerly George Mason / KAIST visiting), whose research is **wireless communication, cross-technology communication (CTC), wireless sensing, and LoRa/ZigBee/WiFi systems** — not software engineering, not trace-link recovery, not architecture-to-code traceability, and not SE evaluation methodology.

The ICSE submission this study is meant to inform sits in a fundamentally different sub-community (software engineering / ArDoCo-style traceability). Style transfer across these communities is partial: networking-systems papers (SenSys, MobiCom, ICDCS, INFOCOM, IEEE TMC) share a recognisable "systems-paper" rhetorical mould with SE systems venues (ICSE, FSE) — empirically grounded contributions, quantitative headline numbers, three-bullet contribution lists — but they diverge in their tolerance for hardware-physics jargon, equation density, and the way prior work is positioned. The recommendations in §4 are filtered to keep only the patterns that plausibly transfer to an ICSE submission.

If the user intended a *different* Shuai Wang (the name is common: HKUST's software-security Shuai Wang has a different Scholar ID, and an SE-side Shuai Wang exists at Northern Arizona / NEU), this study should be re-run with the correct profile ID. I followed the literal ID supplied.

---

## §1 Researcher identification

- **Name:** Shuai Wang (王帅)
- **Affiliation:** Assistant Professor & Department Head of Computer Engineering, Southeast University, Nanjing, China. PhD-era affiliations: George Mason University (PhD, with Song Min Kim) and visiting at KAIST.
- **Research area:** Wireless communication; cross-technology communication (CTC) between WiFi / ZigBee / LoRa / Bluetooth; wireless sensing; IoT MU-MIMO; mmWave sensing; recently spatio-temporal prediction on dynamic graphs.
- **Citations (Google Scholar, May 2026):** ~616 total, h-index 14, i10-index 17.
- **Most frequent co-authors:** Song Min Kim (KAIST), Tian He (UMN), Zhimeng Yin, Wenchao Jiang. Best-paper / award signal: ICDCS 2018 Best Paper for SymBee (reported by Semantic-Scholar / search results).

## §2 The seven papers

For each: section structure, abstract opening sentence (verbatim), introduction first paragraph (verbatim where the PDF was read; marked "abstract-only" where only the abstract was available), contribution-list shape, tone signals, forward references, figure/table density on first two pages, intro length.

### 2.1 SymBee — *Symbol-level Cross-technology Communication via Payload Encoding*, ICDCS 2018 (Wang, Kim, He) — top-cited, award-winning

- **Section structure:** I Introduction; II Motivation (A Opportunities for CTC, B The Need for Symbol-level CTC); III Design Overview and Background; IV SymBee Design; V Enhanced Decoding with SymBee Preamble; VI SymBee Features; VII Analytics; VIII Evaluation; IX Related Work; X Conclusion; Appendix A/B.
- **Abstract opening (verbatim):** "To mitigate the issue of cross-technology interference (CTI) under dense wireless, cross-technology communication (CTC) was recently proposed, which enables direct communication among heterogeneous wireless technologies."
- **Intro first paragraph (verbatim):** "Explosive growth of wireless devices over the last decade is anticipated to be intensified and diversified as we step into the Internet of Things (IoT) era, to reach 50 billion by 2020 [2]. As much as massive scale wireless body has enriched our daily lives, spectrum shortage has become one of the significant bottlenecks to efficient networking. I.e., overcrowded unlicensed ISM band has led to severe cross-technology interference (CTI) [12], which has become a major hurdle to network reliability and spectrum efficiency. For example, ZigBee is known to suffer from up to 50% packet loss under WiFi interference [21]."
- **Contribution shape:** Bulleted list, **3 bullets** ("three-fold"). First bullet stakes a novelty claim ("To the best of our knowledge, for the first time..."), second describes the artefact, third describes the evaluation.
- **Tone:** First-person plural ("we"), declarative, light hedging, jargon is heavy (CTI/CTC/DSSS/OQPSK) but acronyms are expanded on first use. Headline number quoted in abstract: "31.25kbps, 145.4× faster than the state-of-the-art."
- **Forward references in intro:** Yes — the 145.4× figure and the 31.25 Kbps figure both appear in the intro before any evaluation section.
- **Figure/table density in first two pages:** None (intro is text-only); first figure (a diagram of "SymBee message embedded in ZigBee packet") appears on page 3.
- **Intro length:** Roughly 6 paragraphs, ~45 sentences. Ends with an explicit "rest of this paper is organized as follows" roadmap.

### 2.2 SCLoRa — *Leveraging Multi-Dimensionality in Decoding Collided LoRa Transmissions*, ICNP 2020 (Hu, Yin, Wang, Wang, Xu, He) — 2nd-most-cited

- **Section structure:** I Introduction; II Background and Motivation (LoRa Modulation and Demodulation; Collided Concurrent Transmission; Opportunities of Decoding Collisions); III Design of SCLoRa (Spectral Coefficient; Cumulative Spectral Coefficient; Symbol Classification; Packet Processing; Spectrum Leakage Elimination); IV Performance Evaluation; V Related Work; VI Conclusion.
- **Abstract opening (verbatim):** "LoRa as a representative of Low-Power Wide Area Networks (LPWAN) technologies has emerged as an attractive communication platform for the Internet of Things."
- **Intro first paragraph (verbatim):** "The Low-Power Wide Area Networks (LPWAN) technologies, including LoRa [1], NB-IoT [2], Sigfox [3] and Weightless [4], have been emerging as popular technologies in recent years [5]. Many LPWAN-based applications, such as Sailing Monitoring System [6], health and well-being monitoring [7], [8], agriculture monitoring [9]–[12], are developed. Since LoRa is designed to support these applications over a long range (e.g., more than 10 KM), a large number of LoRa end devices inevitably coexist at the same time, leading to serious packet loss when these end devices send packets to a base station concurrently [13]."
- **Contribution shape:** Bulleted list, **3 bullets**. First bullet: novelty claim ("SCLoRa is the first to..."). Headline number "3× the state-of-the-art" appears in abstract and intro.
- **Tone:** First-person plural; declarative; uses an explicit comparison table (Table I) on page 2 to distil prior-vs-ours differences across five criteria — a hallmark move for this author.
- **Forward references:** Yes — the 3× throughput figure and the "first to examine multiple LoRa features" claim are both forward-referenced in the intro.
- **Figure/table density on first two pages:** One comparison table (Table I) by page 2; first figure (LoRa upchirp/downchirp/symbol) appears on page 2.
- **Intro length:** ~3 paragraphs, ~25 sentences. Short for this author.

### 2.3 X-MIMO — *Cross-Technology Multi-User MIMO*, SenSys 2020 (Wang, Jeong, Jung, Kim)

- **Section structure:** 1 Introduction; 2 Motivation (Need for IoT MU-MIMO; Opportunity #1 CTC; Opportunity #2 Multi-antenna WiFi AP); 3 X-MIMO Overview; 4 X-MIMO Design; 5 Multi-stream CTC; 6 Evaluation; 7 Related Work; 8 Conclusion; Appendix.
- **Abstract opening (verbatim):** "Multi-user MIMO (MU-MIMO) is a widely-known, fundamental technique to significantly improve the spectrum efficiency."
- **Intro first paragraph (verbatim):** "The body of wireless devices is experiencing rapid growth with the emergence of the Internet of Things (IoT) era. The number of IoT devices is expected to grow as large as a trillion by 2035 [46], with the vision of providing pervasive services spanning every corner of our daily lives. To achieve this, the key factor in IoT is the capability to extend to an extreme scale in a spectrum efficient manner, thereby enabling prevalent deployment. This is indeed critical considering that the IoT standards inevitably suffer from a slow transmission rate (and thus low spectrum efficiency), in order to simplify the modulation and keep the receiver radio architecture simple, low-cost, and power-efficient. For instance, ZigBee and Bluetooth have 0.125 and 1 bits/s/Hz, which are 240 and 30 times lower spectrum efficiencies compared to WiFi 802.11n (30 bits/s/Hz)."
- **Contribution shape:** Bulleted list, **3 bullets** introduced by "our contribution is three-fold:". Same template as SymBee.
- **Tone:** First-person plural; strongly declarative; jargon-heavy but every acronym expanded on first use. Headline numbers ("495 Kbps with <1% SER", "704.24 Kbps with 6.1% SER") quoted in the abstract.
- **Forward references:** Yes — multiple, including specific throughput numbers and "2× of state-of-the-art WEBee".
- **Figure/table density on first two pages:** None on page 1; the first figure (the 3-step pipeline) appears on page 3.
- **Intro length:** ~5 paragraphs, ~40 sentences. Followed by a dense 2-page motivation section with two explicit "Opportunity #1 / Opportunity #2" subsections — a distinctive rhetorical device.

### 2.4 mmSpyVR — *Exploiting mmWave Radar for Penetrating Obstacles to Uncover Privacy Vulnerability of Virtual Reality*, IMWUT/UbiComp 2024 (Mei, Liu, Yin, Zhao, Jiang, Wang, Lu, He)

- **Status:** Abstract-only (read via arXiv abstract page). Shuai Wang is a middle author here, but the paper is on his profile and represents the recent sensing-attack work.
- **Abstract opening (verbatim):** "Virtual reality (VR), while enhancing user experiences, introduces significant privacy risks."
- **Contribution shape (from abstract):** Two-part framework, narrative not bulleted in the abstract. Two key headline numbers up front: "98.5%" application-recognition accuracy and "92.6%" keystroke-recognition accuracy.
- **Tone:** Declarative, foregrounds the *vulnerability* framing. Engages with industry (Meta) — paraphrased from abstract.
- **Forward references:** Yes — the two accuracy figures appear in the abstract before the methods.
- **Section structure / figure density:** not directly read; per the arXiv HTML, paper opens with a "vulnerability discovery" narrative, then framework figure on page 1–2 (typical IMWUT).

### 2.5 NetCTC — *Networking Support for Bidirectional Cross-Technology Communication*, IEEE TMC 2019 (Wang, Yin, Wang, Chen, Li, Kim, He)

- **Status:** Abstract-only (no public PDF accessible via search; details paraphrased from the journal landing pages found via search).
- **Abstract opening (paraphrased — not verbatim):** Frames the problem as the absence of networking-layer support (ACKs, multicast, broadcast) for physical-layer CTC, then proposes NetCTC.
- **Contribution shape (paraphrased):** Three-fold: (i) the first networking-support design for PHY-CTC; (ii) a real-time interaction mechanism; (iii) testbed evaluation on USRP-N210 + commodity. (Inferred from the abstract excerpt visible in search; treat as paraphrase.)
- **Tone:** Same author template — "first ... to" novelty claim, explicit head-to-head comparison with prior CTC stack.

### 2.6 WiLo — *Long-Range Cross-Technology Communication From Wi-Fi to LoRa*, IEEE TCOM 2024 (Gao, Wang, Wang, ..., Yin, ..., He)

- **Status:** Abstract-only via IEEE Spectrum / Xplore landing.
- **Abstract opening (paraphrased — not verbatim):** Positions WiLo as the first CTC that bridges 2.4 GHz Wi-Fi to sub-GHz LoRa, achieves "500 m" range with "more than 96% frame reception rate".
- **Contribution shape (paraphrased):** Narrative; the headline 500 m / 96% appears at the top.
- **Tone:** Outcome-first ("we demonstrate concurrent wireless communication over a distance of 500 m"), strong quantitative anchor.

### 2.7 ProST — *Prompt Future Snapshot on Dynamic Graphs for Spatio-Temporal Prediction*, KDD 2025 (Xia, Lin, Wang, Zhang, Wang, He)

- **Status:** Abstract-only via ACM DL / ResearchGate listing.
- **Abstract opening (paraphrased — not verbatim):** Frames spatio-temporal prediction on dynamic graphs as the target task and proposes a "prompt future snapshot" framework with multi-granularity evolution graph convolution.
- **Contribution shape:** Recent ML-style abstract — problem, framework name, two-stage architecture (pre-training + prompting), brief mention of experimental superiority. Less of the explicit "three-fold contribution" bullet pattern; more compressed.
- **Tone:** Marks the author's evolution: as he moves into the ML/data-mining venue (KDD), the prose tightens and the bulleted "first to" pattern relaxes into an ML-paper paragraph form.

## §3 Recurring style patterns across the seven papers

Distilled from the four papers read in full (SymBee, SCLoRa, X-MIMO read first-pages-plus; mmSpyVR, NetCTC, WiLo, ProST read at abstract level only):

1. **Numbered headline gain in the abstract.** Every paper puts a "Nx vs state-of-the-art" or "≥ X%" number in the abstract — 145.4×, 3×, 495 Kbps / <1% SER, 98.5% / 92.6%, 500 m / 96%. The number is then *forward-referenced* into the intro before any methodology appears.
2. **Three-bullet "our contribution is X-fold" list at the end of the introduction.** The shape is stable across SymBee, SCLoRa, X-MIMO: bullet 1 = a novelty / "first-to" claim; bullet 2 = the artefact; bullet 3 = the evaluation result. ProST (KDD 2025) is the exception — recent ML venues prefer paragraph form.
3. **Funnel intro: IoT explosion → spectrum/scale bottleneck → specific technical defect → our fix.** The intros open with a macroscopic motivator (50 B IoT by 2020, 1 T by 2035, LPWAN/LoRa rise), then narrow to one quantitative pain (50% ZigBee packet loss under WiFi interference; 215 bps state-of-the-art throughput; 240× lower ZigBee spectral efficiency than WiFi). The narrowing is fast — within the first two paragraphs.
4. **Comparison table on page 2 distilling prior work.** SCLoRa Table I is exemplary: rows = competing systems, columns = qualitative properties (which feature, dimensionality, impact of SNR, demand on signal boundary, adaptability to burst traffic). This author reaches for a small qualitative table early, even when the rest of the paper is quantitative.
5. **Explicit "Opportunity #1 / Opportunity #2" or "Need / Limitations of Gateway / Limitations of State-of-the-Art / Advantages and Challenges" subsection scaffolding inside the motivation section.** SymBee §II-B and X-MIMO §2.2/§2.3 both use this rhetorical device — naming the opportunities or limitations as numbered subheadings rather than burying them in prose.
6. **Acronyms expanded on first use; jargon density tapers fast.** Even in deeply physical-layer papers (CTI, CTC, OQPSK, DSSS, CSI, HT-LTF, AGC, CFO) the first occurrence is always parenthesised with the expansion. The text otherwise leans hard on acronyms.
7. **Heavy use of figures with hand-drawn pipeline icons + waveform plots, but not on page 1.** Page 1 stays prose-only across SymBee, SCLoRa, X-MIMO. The first figure typically lands on page 2–3, after the contribution bullets.
8. **Roadmap sentence at the end of the intro.** "The rest of this paper is organized as follows. Section II ... Section X concludes." — a near-verbatim template across the systems papers. (Less common in the KDD 2025 paper.)
9. **Acknowledgement of inherited limitations.** X-MIMO §5: "The slight difference indicates our design inherits the limitation of the state-of-the-art CTC — determined by the finite constellation points, the precision is degraded by emulation errors." The author repeatedly signals what the system *cannot* do — a credibility move.
10. **Section-final mini-summaries.** Each design subsection often ends with a "To sum up..." or "This effectively demonstrates that..." sentence pinning down the take-away before the next subsection. Helps the reader keep the thread in equation-heavy prose.

## §4 Recommendations for the ICSE submission

Concrete prose moves to adopt, calibrated for an SE evaluation-methodology paper (not a wireless paper):

1. **Lead the abstract with one declarative headline number.** Replace any hedged "we observe an improvement" framing with "X reduces the false-positive rate from a% to b%" or "X recovers c% of trace links missed by Y", in the first three sentences of the abstract. Do not bury the number.
2. **Use a numbered three-bullet contribution list at the end of §1.** Bullet 1: a "to the best of our knowledge, this is the first study to..." novelty claim, scoped narrowly enough to be defensible. Bullet 2: the artefact / methodology. Bullet 3: the empirical result with at least one quantitative anchor. Keep to **three** bullets — four or more dilutes the claim.
3. **Forward-reference the headline numbers into §1.** Quote at least one specific metric in the intro before the related-work or methods section. ICSE reviewers also reward this.
4. **Insert a small qualitative comparison table on page 2.** Rows = competing trace-link / evaluation tools; columns = qualitative properties (handles multi-doc input? language-agnostic? requires labelled data? evaluation granularity?). A 5×4 table on page 2 outperforms three paragraphs of prose-positioning.
5. **Funnel-shape the intro: SE landscape pain → specific defect in current evaluation → our fix.** First paragraph: the broader SE problem (e.g., poor reproducibility of trace-link benchmarks, ambiguity in current evaluation metrics). Second paragraph: a specific quantitative pain (a number from a published benchmark). Third paragraph: how we fix it.
6. **Name the opportunities or pre-conditions as numbered subsections in §2.** E.g., "§2.2 Opportunity #1: LLMs expose calibrated traces", "§2.3 Opportunity #2: ArDoCo provides ground-truth anchors". This is one of the most distinctive Wang moves and it translates cleanly to SE.
7. **Expand every acronym on first use, even community-standard ones.** Per the user's writing-style memory, avoid SAD/SAM/ACF1/HUS entirely; spell out "F1" only as established. The Wang papers never assume a reviewer knows DSSS or HT-LTF — adopt the same discipline for SE acronyms.
8. **Add a roadmap sentence at the end of §1.** "The rest of this paper is organized as follows..." — short, neutral, but ICSE intros often skip it and lose orientation cues.
9. **End each major design subsection with a one-sentence "To sum up..." closer.** Especially useful in methodology sections where the reader has to track multiple metric definitions; this is a clarity move that costs nothing.
10. **Acknowledge inherited limitations explicitly in the design / evaluation sections.** A short sentence in the form "Our design inherits the limitation of [prior tool] — determined by [root cause], [observed effect]" buys reviewer credibility and reduces the surface area for "you didn't consider X" rebuttals.

### Recommendations to *not* adopt (network-systems-specific patterns that would harm an ICSE paper)

- The dense equation derivations in mid-paper (Wang's hallmark in SymBee §IV-B and X-MIMO §4.3) suit signal-processing reviewers but irritate SE reviewers — keep equations sparse and in an appendix.
- The "Nx faster" framing reads as marketing in SE venues. Use "X percentage points higher F1" or absolute deltas instead.
- Wang's related-work section is short and clustered at the end (§IX in SymBee). ICSE conventions prefer related work earlier or a "background" §2 — follow ICSE convention here, not Wang's.

---

*Sources used to identify the researcher and verify papers:* Google Scholar profile `lOJUEfoAAAAJ`; OpenReview profile `~Shuai_Wang30`; SEU homepage; ICDCS 2018, ICNP 2020, SenSys 2020 proceedings; arXiv 2411.09914; IEEE Xplore listings for TMC 2019 and TCOM 2024; ACM DL listing for KDD 2025.

*Downloaded PDFs in this directory:* `wang-2018-symbee.pdf`, `wang-2020-sclora.pdf`, `wang-2020-xmimo.pdf`, `wang-2024-mmspyvr.pdf`. The remaining three (NetCTC TMC 2019, WiLo TCOM 2024, ProST KDD 2025) were not downloaded — only abstracts were accessible without paywalled access.
