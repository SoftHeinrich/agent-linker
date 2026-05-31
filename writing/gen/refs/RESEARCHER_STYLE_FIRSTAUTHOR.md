# Researcher Writing Style: First-Author Papers Only

## 1. Researcher identification

- **Name:** Shuai Wang
- **Google Scholar ID:** `lOJUEfoAAAAJ`
- **Current affiliation:** Assistant Professor, Department of Computer Science & Engineering, Southeast University, Nanjing, China. PhD George Mason University (2023), advised by Song Min Kim and Parth Pathak.
- **Research area:** Wireless communication, cross-technology communication (CTC), wireless sensing, IoT systems. *Note:* This area is unrelated to software-engineering trace-link recovery / architecture-to-code traceability. There is **no domain-relevant first-author paper** for an ICSE SE-traceability submission; the "most domain-relevant first-author paper" selection criterion has no satisfiable candidate in this researcher's catalogue.

### Selection caveats

Across the full Scholar profile (45 items), the following are first-author publications:

| Year | Title (short) | Venue | Cites | PDF? |
|------|---------------|-------|-------|------|
| 2014 | Distributed energy-efficient power control | IEEE DASC | 5 | none found |
| 2018 | Symbol-Level Cross-Technology Communication via Payload Encoding (SymBee) | ICDCS | 96 | yes |
| 2019/21 | Networking Support For Bidirectional CTC (NetCTC) | IEEE TMC | 48 | none found |
| 2020 | X-MIMO: Cross-technology multi-user MIMO | ACM SenSys | 29 | yes |
| 2022 | X-Disco: Cross-technology neighbor discovery | IEEE SECON | 16 | yes |
| 2023 | Cross-Technology Communication and Sensing Using Low-Power IoT | GMU dissertation | 0 | not found (a Google hit was a different Shuai Wang's PSU dissertation on binary code security — discarded) |
| 2025 | UMusic: In-car occupancy sensing via UWB PDP | ACM SenSys (Best Paper nominee) | 10 | yes |

Per the task's hard rule ("drop any you can't verify"), Section 2 below studies only the **4 first-author papers whose PDFs were located and whose first-author position was directly verified from the PDF title page**: SymBee 2018, X-MIMO 2020, X-Disco 2022, UMusic 2025. The other three (2014 DASC, 2019 TMC, 2023 dissertation) are listed for trajectory completeness but not analysed at the prose level. This is fewer than 7 because (a) the researcher's first-author corpus is small (~7 in 11 years), (b) two are paywalled with no open preprint, and (c) the dissertation could not be located (a same-named PSU graduate's dissertation was mis-suggested and verified to be a different person).

The 4 analysed papers still span the career: earliest open paper (2018, GMU PhD year 1), mid-PhD (2020), late-PhD (2022), and post-PhD as faculty (2025). All four are about CTC/wireless sensing; the prose patterns are highly stable across them.

---

## 2. The first-author papers, paper-by-paper

### 2.1 Wang 2018 — SymBee (ICDCS '18)

- **Full title (verbatim):** "Symbol-level Cross-technology Communication via Payload Encoding"
- **Authors verbatim:** Shuai Wang∗, Song Min Kim ∗‡, and Tian He †‡ — Wang first author, with co-corresponding authors marked.
- **Section structure (verbatim):** I. Introduction; II. Motivation; III. Design Overview and Background; IV. (Design — heading lost in extract but present); V. Enhanced Decoding with SymBee Preamble; VI. SymBee Features; VII. Analytics; VIII. Evaluation; IX. Related Work; X. Conclusion.
- **Abstract opening sentence (verbatim):** "To mitigate the issue of cross-technology interference (CTI) under dense wireless, cross-technology communication (CTC) was recently proposed, which enables direct communication among heterogeneous wireless technologies."
- **Introduction first paragraph (verbatim):** "Explosive growth of wireless devices over the last decade is anticipated to be intensified and diversified as we step into the Internet of Things (IoT) era, to reach 50 billion by 2020 [2]. As much as massive scale wireless body has enriched our daily lives, spectrum shortage has become one of the significant bottlenecks to efficient networking. I.e., overcrowded unlicensed ISM band has led to severe cross-technology interference (CTI) [12], which has become a major hurdle to network reliability and spectrum efficiency. For example, ZigBee is known to suffer from up to 50% packet loss under WiFi interference [21]."
- **Contributions:** narrative bullet list (3 bullets, ending with introduction of system, technical novelty, evaluation). Bullets are introduced after the contribution paragraph with explicit "•" markers.
- **Tone signals:** first-person plural ("We present", "We note", "We evaluate"); declarative; quantitative front-loading ("145.4×", "31.25 kbps") in both abstract and intro; light hedging ("Although effective, they commonly suffer from").
- **Forward references:** YES — headline number (145.4× improvement, 31.25 kbps) appears in abstract, intro paragraph 3, and contribution bullet 3.
- **Figure/table density first 2 pages:** very low — no figures in intro; first figure later in design section.
- **Intro length:** 4 paragraphs, ~28 sentences total.

### 2.2 Wang 2020 — X-MIMO (SenSys '20)

- **Full title (verbatim):** "X-MIMO: Cross-Technology Multi-User MIMO"
- **Authors verbatim:** Shuai Wang§†, Woojae Jeong†, Jinhwan Jung†, and Song Min Kim†∗ — Wang first author; Kim is corresponding.
- **Section structure (verbatim):** 1 Introduction; 2 Motivation; 3 X-MIMO Overview; 4 X-MIMO Design; 5 Multi-stream CTC; 6 Evaluation; 7 Related Work; 8 Conclusion.
- **Abstract opening sentence (verbatim):** "Multi-user MIMO (MU-MIMO) is a widely-known, fundamental technique to significantly improve the spectrum efficiency."
- **Introduction first paragraph (verbatim):** "The body of wireless devices is experiencing rapid growth with the emergence of the Internet of Things (IoT) era. The number of IoT devices is expected to grow as large as a trillion by 2035 [46], with the vision of providing pervasive services spanning every corner of our daily lives. To achieve this, the key factor in IoT is the capability to extend to an extreme scale in a spectrum efficient manner, thereby enabling prevalent deployment. This is indeed critical considering that the IoT standards inevitably suffer from a slow transmission rate (and thus low spectrum efficiency), in order to simplify the modulation and keep the receiver radio architecture simple, low-cost, and power-efficient. For instance, ZigBee and Bluetooth have 0.125 and 1 bits/s/Hz, which are 240 and 30 times lower spectrum efficiencies compared to WiFi 802.11n (30 bits/s/Hz)."
- **Contributions:** explicit "our contribution is three-fold:" followed by 3 bulleted contributions. Each bullet starts with a verb ("We design", "To apply", "We implement and evaluate").
- **Tone signals:** "To the best of our knowledge, X-MIMO is the first…", "Rigorous derivation shows…"; first-person plural; declarative; quantitative claims very prominent ("495 Kbps", "<1% SER", "704.24 Kbps", "near-linear").
- **Forward references:** YES — abstract states final numbers; intro recites them again with the "first of its kind" framing.
- **Figure/table density first 2 pages:** none in introduction; the first figure (system overview, three-step) lives in Section 3.
- **Intro length:** 5 paragraphs, ~35 sentences.

### 2.3 Wang 2022 — X-Disco (SECON '22)

- **Full title (verbatim):** "X-Disco: Cross-technology Neighbor Discovery"
- **Authors verbatim:** Shuai Wang∗, Jianlin Guo†, Pu Wang†, Kieran Parsons†, Philip Orlik†, Yukimasa Nagai‡, Takenori Sumi‡, Parth Pathak∗ — Wang first author.
- **Section structure (verbatim):** I. Introduction; II. Motivation; III. Overview of X-Disco and Background; IV. (Design of X-Disco); V. Advanced Features of X-Disco; VI. Evaluation; VII. Related Work; VIII. Conclusion.
- **Abstract opening sentence (verbatim):** "With the explosive proliferation of wireless devices, our lives are improved by various applications supported by heterogeneous wireless technologies, such as WiFi and ZigBee."
- **Introduction first paragraph (verbatim):** "We have witnessed the explosive growth of IoT devices, including WiFi, ZigBee, and Bluetooth, along with various applications supported by heterogeneous wireless technologies in the past decades. As half billion ZigBee chips sold [1] and over three billion WiFi devices shipped annually [2], WiFi and ZigBee coexist densely on the 2.4 GHz ISM spectrum and physical places such as smart homes and factories, raising critical coexistence issues such as cross-technology interference (CTI) [3], [4]. To avoid such interference, cross-technology coordination [5], [6] an[d]…"
- **Contributions:** "To summarize, the contribution of this paper is three-fold:" followed by 3 bullets — verbatim a near-identical scaffold to X-MIMO.
- **Tone signals:** "the first cross-technology neighbor discovery mechanism"; "To the best of our knowledge, X-Disco is the first design to…"; declarative; quantitative ("nine ZigBee neighbors within 70ms").
- **Forward references:** YES — 70 ms latency repeated in abstract, intro, and bullet 3.
- **Figure/table density first 2 pages:** none.
- **Intro length:** ~3 paragraphs, ~22 sentences.

### 2.4 Wang 2025 — UMusic (SenSys '25, Best Paper nominee)

- **Full title (verbatim):** "UMusic: In-car Occupancy Sensing via High-resolution UWB Power Delay Profile"
- **Authors verbatim:** Shuai Wang (Southeast University), Yunze Zeng (Bosch Research), Vivek Jain (Bosch Research), Parth Pathak (GMU). Wang first author; Zeng corresponding.
- **Section structure (verbatim):** 1 Introduction; 2 Background and Motivation; 3 Design Overview; 4 Main Design; 5 Efficiency Enhancement; 6 Evaluation; 7 Discussion and Future Work; 8 Related Work; 9 Conclusion. (First time a "Discussion and Future Work" section appears.)
- **Abstract opening sentence (verbatim):** "Occupancy sensing is essential for vehicle safety and security applications such as seat belt reminders, airbag deployment, intrusion detection, and child-left-behind alerts."
- **Introduction first paragraph (verbatim):** "The automotive industry has been undergoing a major transformation over the past century, shifting from engine-centric design to prioritizing passenger experience [14]. Modern cars are no longer just transport vehicles but intelligent ecosystems that enhance safety and comfort for users [4, 81, 86]. For instance, incorporating various sensors, automakers like Ford [13], Honda [19], and Tesla [59] are making significant progress in building advanced collision avoidance, theft protection, and keyless entry solutions [12, 21]."
- **Contributions:** "the contribution of this paper is threefold:" followed by 3 bullets. Identical scaffold to X-MIMO/X-Disco.
- **Tone signals:** "novel … system", "innovative path decomposition technique", "to the best of our knowledge"-style positioning throughout; declarative; quantitative ("90.2% detection rate", "99.4% accuracy", "50 ms", "0.125 ms").
- **Forward references:** YES — both 90.2% and 99.4% appear in abstract, intro contribution paragraph, and contribution bullets.
- **Figure/table density first 2 pages:** Figure 1 (UWB PHY illustration) appears at the bottom of page 2 — the first paper in the set to put a figure before the design section.
- **Intro length:** ~5 paragraphs.

---

## 3. Recurring style patterns (Wang's own first-author voice)

1. **Three-fold contribution scaffold, verbatim across papers.** The phrase "the contribution of this paper is three-fold" (or "our contribution is three-fold") appears in X-MIMO, X-Disco, and UMusic. Bullet 1 = "We design/present [SYSTEM], the first …". Bullet 2 = "[SYSTEM] introduces [N] techniques: …". Bullet 3 = "We implement and evaluate [SYSTEM] on commodity devices …". This is a rigid template.
2. **"First-of-its-kind" framing.** Every analysed paper claims a "first" — "first symbol-level CTC", "first MU-MIMO on commodity IoT", "first cross-technology neighbor discovery mechanism", "novel in-car occupancy sensing system that reuses UWB". The phrase "to the best of our knowledge … the first" is a recurring construction.
3. **Headline-number forward references.** A single quantitative claim is repeated in abstract → intro → contribution bullet → conclusion: 145.4× (SymBee), 495 Kbps + <1% SER (X-MIMO), 70 ms / 9 neighbors (X-Disco), 99.4% / 50 ms / 0.125 ms (UMusic). Reader sees the headline number 3–4 times in the first two pages.
4. **"Explosive growth" opening.** SymBee, X-MIMO and X-Disco intros all open by invoking IoT/wireless explosion ("Explosive growth", "rapid growth … emergence of IoT", "explosive growth of IoT devices"). UMusic varies this to "major transformation" of the automotive industry — same rhetorical move, different domain.
5. **Citation-dense opening sentences.** Wang stacks numbered citations into the first paragraph: e.g., UMusic uses 6 distinct refs in its first 3 sentences ([14], [4,81,86], [13], [19], [59], [12,21]). The pattern is "every factual claim is referenced", which raises the citation density of intros well above the section average.
6. **Declarative, present-tense, first-person plural.** "We present", "We design", "We propose", "We evaluate", "We note that". No first-person singular. Hedging is rare and short ("Although effective, …"; "may", "could" used sparingly).
7. **No figures or tables in the introduction.** SymBee, X-MIMO, and X-Disco place the first figure inside Section 3 (Design Overview). UMusic is a slight exception, placing Figure 1 near the end of the intro. Wang relies on prose, not visuals, to motivate.
8. **Tight motivation section preceding overview.** Every paper has a dedicated §II/§2 "Motivation" with two consistent subsections: "The Need for X" and "Opportunity #1 / Challenges". This is essentially a SenSys/ICDCS systems template, but Wang follows it strictly across venues.
9. **Acronyms defined inline on first use, then heavily reused.** CTC, CTI, MU-MIMO, CSI, PDP, CIR, SER are all expanded once and then used as bare acronyms — Wang trusts the reader after first definition, and the resulting prose is dense.
10. **Compatibility / zero-cost / commodity messaging.** A recurring value proposition: "fully compatible with commodity ZigBee/WiFi", "zero-cost", "software-only", "without modification to hardware or firmware". This is essentially a brand claim in every first-author paper.

---

## 4. Recommendations for our ICSE paper

These are concrete prose-level moves transferable to an SE traceability paper. Each cites the first-author paper(s) where the move is best demonstrated. (Caveat: this researcher is not an SE author; we are borrowing rhetorical moves, not domain framing.)

1. **Adopt the "three-fold contribution" scaffold with parallel verb structure.** Bullet 1 = system claim ("We present X, the first …"). Bullet 2 = technique enumeration ("X introduces three components: A, B, C"). Bullet 3 = evaluation summary with a headline number. Demonstrated by X-MIMO, X-Disco, UMusic — all three use the identical pattern, which suggests it's load-bearing for systems/CS venues.
2. **Forward-reference your top result number three times in the first two pages.** Abstract last sentence → intro penultimate paragraph → final contribution bullet. SymBee does this with "145.4×"; UMusic with "99.4%". For our paper, pick one trace-link recovery F1 (or comparable headline) and surface it in all three slots.
3. **Open the intro with a citation-dense statement of the problem domain's scale or importance.** UMusic packs 6 citations into the first 3 sentences; X-Disco opens with concrete industry numbers ("half billion ZigBee chips", "three billion WiFi devices shipped"). For an SE paper, this maps to: open with concrete adoption numbers for the SE artifacts you target (LOC, repo counts, citation counts of the baseline tools).
4. **Claim a "first-of-its-kind" positioning explicitly using "to the best of our knowledge".** Every Wang paper does this. For us: name what makes our trace-link recovery approach the first of its kind (e.g., first to combine X and Y, first to evaluate on Z) in a single sentence within the intro.
5. **Use a dedicated §2 Motivation with two subsections: "The Need for …" and "Opportunity / Challenge".** All four Wang papers do this. It separates problem framing from technical background and avoids cluttering the introduction.
6. **Defer the first figure until Section 3 (Design Overview) rather than placing it in the intro.** SymBee, X-MIMO, X-Disco follow this. The intro becomes pure prose and reads faster. UMusic is a counter-example showing the trade-off (Figure 1 in intro improves comprehension of UWB physical layer) — only adopt if a single illustration genuinely unlocks the rest.
7. **Repeat the same rhetorical claim ("commodity", "zero-cost", "compatible") consistently across abstract, intro, and contributions.** Wang turns "compatibility with commodity devices" into a brand line repeated 3–5 times per paper. For SE, the analogous brand line might be "training-free", "model-agnostic", "no labelled traces required" — pick one and repeat it deliberately.
8. **Keep the contribution-bullet count fixed at three.** Wang never uses 4 or 5. Three forces the writing to compress sub-contributions inside each bullet. Demonstrated in X-MIMO bullet 2, which lists three sub-challenges within a single bullet rather than promoting them to standalone bullets.
9. **Use "We note that …" as a soft signpost for placing caveats and minor observations.** Used in SymBee ("We note that there has been a recent advancement …") and X-MIMO ("We note that a large number of ZigBee/802.15.4 IoT devices …"). It's a low-key way to insert qualifying information without breaking the declarative tone.
10. **Avoid first-person singular and hedging adverbs ("perhaps", "possibly", "we believe"). Use first-person plural and declaratives.** Wang's prose never hedges its central claims; hedges appear only when discussing baselines ("Although effective, they commonly suffer from limited data rate"). For ICSE, this matches reviewer expectations for a strong claims-first systems paper.

---

*End of style study. Files saved in this directory:*
- `wang-2018-symbee.pdf`
- `wang-2020-xmimo.pdf`
- `wang-2022-xdisco.pdf`
- `wang-2025-umusic.pdf`
