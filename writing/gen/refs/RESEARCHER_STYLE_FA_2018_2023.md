# Researcher Writing Style Study — First-Author Papers 2018–2023

## §1 Researcher identification

- **Name:** Shuai Wang
- **Current affiliation:** Assistant Professor, School of Computer Science and Engineering, Southeast University (Nanjing, China). Previously: George Mason University (PhD), with visiting/intern positions at KAIST (SMILE Lab) and Mitsubishi Electric Research Laboratories (MERL).
- **Research area:** Wireless networking and IoT — specifically Cross-Technology Communication (CTC), cross-technology MIMO, neighbor discovery across heterogeneous wireless protocols (WiFi/ZigBee/LoRa), and wireless sensing.
- **Identifier:** Google Scholar `lOJUEfoAAAAJ` — h-index 14, ~616 citations.

**Domain caveat for the caller.** This researcher works on wireless systems / physical-layer networking, *not* software-engineering trace-link recovery or architecture-to-code traceability. The §4 ICSE recommendations below transfer the *prose-level moves* that are portable across CS subfields (framing, contribution lists, motivation structure); they do not assume the ICSE paper is on wireless topics. If the caller intended a different Scholar profile, the style study should be redone with the correct ID.

## §2 Verified first-author papers, 2018–2023

Four first-author papers in window were verified against author homepage (cs.seu.edu.cn/shuaiwang_iot), the published PDFs, and IEEE/ACM metadata. One Google-Scholar-listed item (SCLoRa, ICNP 2020) was dropped because Shuai Wang is third author, not first. The TMC paper appears as "[TMC'19]" on the homepage but the formal IEEE pub year is 2021 (vol. 20, issue 1) — both endpoints fall inside the 2018–2023 window so it is retained.

### 2.1 SymBee — Symbol-Level Cross-Technology Communication via Payload Encoding (ICDCS 2018, Best Paper Award)

Authors in order: **Shuai Wang**, Song Min Kim, Tian He.

- **Section structure (verbatim from PDF):** I. Introduction; II. Motivation (A. Opportunities for CTC; B. The Need for Symbol-level CTC); III. Design Overview and Background (A. SymBee in a Nutshell); IV. (technical details); V. (robust technique); VI. (other notable features); VII. Analytical performance; VIII. Experimental performance; IX. Related Work; X. Conclusion.
- **Abstract opening (verbatim):** "To mitigate the issue of cross-technology interference (CTI) under dense wireless, cross-technology communication (CTC) was recently proposed, which enables direct communication among heterogeneous wireless technologies."
- **Intro first paragraph (verbatim):** "Explosive growth of wireless devices over the last decade is anticipated to be intensified and diversified as we step into the Internet of Things (IoT) era, to reach 50 billion by 2020 [2]. As much as massive scale wireless body has enriched our daily lives, spectrum shortage has become one of the significant bottlenecks to efficient networking. I.e., overcrowded unlicensed ISM band has led to severe cross-technology interference (CTI) [12], which has become a major hurdle to network reliability and spectrum efficiency. For example, ZigBee is known to suffer from up to 50% packet loss under WiFi interference [21]."
- **Contribution list shape:** Bulleted, three items, signposted with "The contribution of this work is three-fold:". Each bullet opens with a strong claim ("To the best of our knowledge, for the first time...", "A novel ZigBee to WiFi CTC of SymBee is introduced", "We evaluate SymBee both analytically and experimentally").
- **Tone signals:** First-person plural ("we present", "we evaluate"); declarative; only mild hedging ("To the best of our knowledge"); heavy use of "novel", "uniquely", "for the first time"; sentences are medium length (~22 words avg); jargon is field-standard (CTC, CTI, ZigBee, WiFi, ISM band).
- **Forward references — key numbers in intro:** Yes — 31.25 kbps throughput, 145.4× improvement over state-of-the-art, are both stated in the introduction.
- **Figure/table density in first 2 pages:** Zero figures or tables in pages 1–2; first figures appear later. Pure prose intro and motivation.
- **Intro length:** 1 page; introduction proper is ~5 paragraphs and contains an explicit "rest of this paper is organized as follows" roadmap.

### 2.2 NetCTC — Networking Support for Bidirectional Cross-Technology Communication (IEEE TMC 2021; accepted 2019)

Authors in order: **Shuai Wang**, Zhimeng Yin, Shuai Wang (homonym), Zhijun Li, Yongrui Chen, Song Min Kim, Tian He.

PDF not retrievable (IEEE paywall; author homepage no longer hosts). Analysis below is from the verified abstract and published metadata only — marked as **paraphrase** where the source is secondary.

- **Abstract opening (verbatim, from search-engine excerpt):** "Recent research on physical layer cross technology communication (PHY-CTC) brings a timely answer for escalated wireless coexistence and open spectrum movement."
- **Framing (paraphrase):** Abstract follows a recognisable Problem → Gap → Solution → Mechanism shape: PHY-CTC is exciting, but it has transmission-failure and asymmetric-link gaps; NetCTC fills them by adding feedback/ACK support for unicast, multicast, broadcast.
- **Contribution shape (paraphrase from abstract):** Single named artifact ("NetCTC – the first networking support design for PHY-CTC"); explicit "first" claim.
- **Tone signals:** Declarative, claim-forward, "first" used as a positioning device — consistent with §2.1.

### 2.3 X-MIMO — Cross-Technology Multi-User MIMO (ACM SenSys 2020)

Authors in order: **Shuai Wang**, Woojae Jeong, Jinhwan Jung, Song Min Kim.

- **Section structure (verbatim from PDF):** 1 INTRODUCTION; 2 MOTIVATION (2.1 The Need for IoT MU-MIMO; 2.2 Opportunity #1: CTC; ...); then design / evaluation sections.
- **Abstract opening (verbatim):** "Multi-user MIMO (MU-MIMO) is a widely-known, fundamental technique to significantly improve the spectrum efficiency."
- **Intro first paragraph (verbatim):** "The body of wireless devices is experiencing rapid growth with the emergence of the Internet of Things (IoT) era. The number of IoT devices is expected to grow as large as a trillion by 2035 [46], with the vision of providing pervasive services spanning every corner of our daily lives. To achieve this, the key factor in IoT is the capability to extend to an extreme scale in a spectrum efficient manner, thereby enabling prevalent deployment. This is indeed critical considering that the IoT standards inevitably suffer from a slow transmission rate (and thus low spectrum efficiency), in order to simplify the modulation and keep the receiver radio architecture simple, low-cost, and power-efficient. For instance, ZigBee and Bluetooth have 0.125 and 1 bits/s/Hz, which are 240 and 30 times lower spectrum efficiencies compared to WiFi 802.11n (30 bits/s/Hz)."
- **Contribution list shape:** Bulleted, three items, signposted with "our contribution is three-fold:". Same triadic structure as §2.1.
- **Tone signals:** First-person plural, declarative. Identical lexical tics — "novel", "uniquely", "first work", "zero-cost", "software-only" — strong positioning words.
- **Forward references — key numbers in intro:** Yes — 495 Kbps, <1% symbol error rate, 704.24 Kbps with 6.1% SER, 2× WEBee — all in intro.
- **Figure/table density in first 2 pages:** Zero figures/tables on pages 1–2.
- **Intro length:** ~1.5 pages, ~5 paragraphs. Section 2 ("Motivation") immediately follows intro with explicit numbered "Opportunity #1, #2" headings.

### 2.4 X-Disco — Cross-technology Neighbor Discovery (IEEE SECON 2022)

Authors in order: **Shuai Wang**, Jianlin Guo, Pu Wang, Kieran Parsons, Philip Orlik, Yukimasa Nagai, Takenori Sumi, Parth Pathak.

- **Section structure (verbatim from PDF):** I. INTRODUCTION; II. MOTIVATION (A. The Need for Cross-Technology Neighbor Discovery; B. Opportunities — 1) Cross-technology Communication; 2) Fine-grained PHY-layer Information at WiFi); III. (etc.).
- **Abstract opening (verbatim):** "With the explosive proliferation of wireless devices, our lives are improved by various applications supported by heterogeneous wireless technologies, such as WiFi and ZigBee."
- **Intro first paragraph (verbatim):** "We have witnessed the explosive growth of IoT devices, including WiFi, ZigBee, and Bluetooth, along with various applications supported by heterogeneous wireless technologies in the past decades. As half billion ZigBee chips sold [1] and over three billion WiFi devices shipped annually [2], WiFi and ZigBee coexist densely on the 2.4 GHz ISM spectrum and physical places such as smart homes and factories, raising critical coexistence issues such as cross-technology interference (CTI) [3], [4]. To avoid such interference, cross-technology coordination [5], [6] and cooperation [7] are proposed for better accommodating WiFi and ZigBee devices. Nevertheless, the coordination across multiple wireless technologies inevitably requires wireless devices to maintain the cross-technology neighbor information. Therefore, this paper focuses on enabling a universal neighbor discovery mechanism for a WiFi device to detect the ambient ZigBee neighbors, namely cross-technology neighbor discovery."
- **Contribution list shape:** Bulleted, three items, signposted with "the contribution of this paper is three-fold:". Triadic again.
- **Tone signals:** First-person plural; declarative; "first" claim ("the first software-only cross-technology neighbor discovery mechanism"); careful enumeration of challenges as "(i)... (ii)..." inline.
- **Forward references — key numbers in intro:** Yes — "successfully detects nine ZigBee neighbors within 70ms in the office".
- **Figure/table density in first 2 pages:** Figure 1 (the two-step X-Disco protocol diagram) appears on page 2; this is the earliest figure across the four papers.
- **Intro length:** ~1 page, ~5 paragraphs, ending with a triadic contribution bullet block and an enumerated technique list ("(i) ZigBee Symbol Extraction, (ii) ZigBee Coordinator Detection, (iii) Neighbor Information Acquisition").

## §3 Recurring style patterns

1. **Triadic, bulleted contribution lists, every time.** All three readable papers signpost contributions with the exact phrase "...is three-fold:" followed by three bullets (§2.1 SymBee, §2.3 X-MIMO, §2.4 X-Disco). The first bullet is always the headline claim of novelty; the second names the mechanism; the third is the evaluation. This is a near-template.

2. **Macro-statistic opening, drilling to a specific pain number.** Every intro opens with an industry-scale statistic ("50 billion IoT by 2020" §2.1; "trillion by 2035" §2.3; "half billion ZigBee chips sold, three billion WiFi devices shipped annually" §2.4) then narrows within 2–3 sentences to a sharply quantified pain ("50% ZigBee packet loss under WiFi interference"; "240× lower spectrum efficiency"). The funnel is mechanical and effective.

3. **Numeric forward references in the introduction.** Final-paper headline numbers appear in the intro itself — 31.25 kbps / 145.4× (§2.1), 495 Kbps / <1% SER / 2× (§2.3), "9 neighbors in 70 ms" (§2.4). The reader knows the punchline before §2.

4. **Positioning vocabulary is consistent and load-bearing.** "Novel", "uniquely", "first" (often "the first" or "to the best of our knowledge, for the first time"), "zero-cost", "software-only", "fully compatible with commodity devices". These words recur in abstract, intro, and bullets (§2.1, §2.3, §2.4). They do real framing work, not decorative.

5. **Named artifact carries the paper.** Every paper has a one-word, hyphenated brand (SymBee, NetCTC, X-MIMO, X-Disco) introduced in the abstract and threaded through every section heading and figure caption (§2.1, §2.2, §2.3, §2.4). The brand becomes the subject of declarative sentences ("SymBee reaches the throughput of...", "X-MIMO achieves 495 Kbps...").

6. **Two-part Motivation section that splits "Need" from "Opportunity".** §2.3 X-MIMO uses "2.1 The Need for IoT MU-MIMO" / "2.2 Opportunity #1: CTC" / "Opportunity #2:...". §2.4 X-Disco uses "A. The Need for Cross-Technology Neighbor Discovery" / "B. Opportunities — 1) ... 2) ...". §2.1 SymBee uses "A. Opportunities for CTC" / "B. The Need for Symbol-level CTC". Same template, slight reordering.

7. **Inline enumerated lists with "(i)... (ii)... (iii)..." inside dense paragraphs.** Used to compress challenge lists or technique lists without breaking into a bullet block (§2.3 lists three intrinsic IoT limitations inline; §2.4 lists three new technical highlights inline). This keeps prose moving while still being scannable.

8. **Sparse figures in the first 2 pages — prose carries the framing.** §2.1 and §2.3 have no figures on the first two pages. §2.4 has one protocol-overview figure on page 2. Heavy machinery (signal-processing block diagrams, PHY-layer figures) is held back until after the intro and motivation are done.

9. **First-person plural throughout, but no first-person singular and almost no hedging.** "We present", "we evaluate", "we propose" recur across all four. Hedging is reserved for the standard escape clause "to the best of our knowledge" before a "first" claim (§2.1, §2.4).

10. **Explicit "rest of the paper is organized as follows" roadmap in the intro.** §2.1 closes its intro with one; this is a standard ICDCS/IEEE move and the author keeps it.

## §4 Recommendations for our ICSE paper

These are prose-level moves grounded in this researcher's first-author voice. They are portable to an ICSE submission on trace-link recovery / SE evaluation methodology.

1. **Adopt a triadic, bulleted contribution list signposted with "the contribution of this paper is three-fold:"** — exactly as in §2.1, §2.3, §2.4. The triadic shape forces discipline: one bullet for novelty, one for mechanism, one for evaluation. Resist a four- or five-bullet list.

2. **Open the introduction with a macro-statistic, then funnel within three sentences to a sharply quantified pain number.** SymBee (§2.1) goes from "50 billion IoT devices" to "50% ZigBee packet loss" in five sentences. For ICSE: open with the scale of the software-architecture problem (millions of SLOC in studied systems, X% of effort spent on traceability maintenance), then drill to a single recall/precision pain number from prior work.

3. **Forward-reference your headline result number in the introduction.** All three readable papers do this — 145.4× (§2.1), 495 Kbps / <1% (§2.3), 9 neighbors in 70 ms (§2.4). Pick the single most compelling number in your results section and put it in paragraph 2 of the intro, with the same level of specificity (decimals included).

4. **Name your artifact and brand it consistently.** SymBee, X-MIMO, X-Disco (§2.1, §2.3, §2.4) all become the grammatical subject of declarative sentences across the paper. Give the ICSE artifact a short hyphenated or single-token name and use it as the subject — not "our approach" or "the proposed method".

5. **Split the Motivation section into "The Need for X" and "Opportunities".** This is template across §2.1, §2.3, §2.4. For ICSE: "2.1 The Need for Robust Trace-Link Recovery" (problem framing, why current SE workflows fail) and "2.2 Opportunities" (what recent advances — LLMs, embeddings — make this newly tractable). The split signals to reviewers that the problem is real AND the moment is right.

6. **Use inline "(i)... (ii)... (iii)..." enumerations for challenges and contributions inside flowing paragraphs.** §2.3 lists three IoT limitations inline; §2.4 lists three technical highlights inline. This is reviewer-friendly: scannable when skimmed, prose when read. Use it for stating threats to validity and technique components.

7. **Earn each use of "first" and "novel" by pairing with "to the best of our knowledge".** §2.1 and §2.4 each use this construction once, attached to a specific testable claim ("first to analyze the physical layer cross-observability of ZigBee at WiFi"). ICSE reviewers punish unsupported novelty claims; this hedge is the standard cover and it is doing real work in these papers.

8. **Keep figures out of pages 1–2; let prose do the framing.** §2.1 and §2.3 have zero figures in the first two pages; §2.4 has only a protocol-overview diagram. Resist the urge to put your architecture diagram on page 2. The intro and motivation should land before any visual.

9. **Use a Problem → Gap → Solution → Mechanism shape in the abstract, in exactly that order.** NetCTC's abstract (§2.2) is the cleanest example: PHY-CTC is exciting (problem context), but transmission failure / asymmetric link gaps remain (gap), NetCTC adds feedback support (solution), via a real-time interaction mechanism (mechanism). Four sentences, four jobs. Mirror this for the ICSE abstract.

10. **Close the introduction with an explicit roadmap sentence ("The rest of this paper is organized as follows...").** §2.1 keeps this old-school move; it is undervalued for fast reviewer skimming, especially under ICSE page pressure.
