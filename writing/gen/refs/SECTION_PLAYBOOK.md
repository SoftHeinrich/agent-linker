# SECTION_PLAYBOOK — Section-by-Section Writing Plan for our ICSE Paper

Wang exemplars come from four annotation files (4 papers, ~217 prose paragraphs,
~1066 annotated sentences):
- `ARG_FULL_symbee.md` (Wang et al., SymBee, ICDCS 2018)
- `ARG_FULL_xmimo.md` (Wang et al., X-MIMO, 2020)
- `ARG_FULL_xdisco.md` (Wang et al., X-Disco, 2022)
- `ARG_FULL_umusic.md` (Wang et al., UMusic, SenSys 2025)

ICSE expectations come from prior knowledge of the venue's review tradition
(formal RQs, explicit Threats to Validity, ablation tables, replication
package, three-axis baseline comparison, theme-organised related work).

Verified data points (drop-in only; anything else is `\todo{}`):
- Doc-to-code file F1: TransArc 0.803 → AALinker 0.931 (+12.9pp).
- Doc-to-code decision F1: TransArc 0.596 → AALinker 0.823 (+22.7pp).
- Doc-to-code component F1: TransArc 0.714 → AALinker 0.817 (+10.3pp aggregate;
  Teammates regresses ~−9pp; other four projects gain +4pp to +35pp).
- Doc-to-code weighted F1: TransArc 0.594 → AALinker 0.821 (+22.7pp).
- Doc-to-model F1: TransArc 0.799 → AALinker 0.951 (+15.2pp).
- Random predictor avg file F1 ≈ 0.155; oracle-subset avg F1 ≈ 0.987;
  oracle-component avg F1 ≈ 0.906.
- 525 raw human decisions → 18,660 file-level pairs. Per-project factor:
  MediaStore 1.0×, TeaStore 10.1×, Teammates 35.5×, BBB 11.6×, JabRef 217.6×.
- JabRef: top-3 components hold 98.6% of file-level gold; top component = 47%.
- Block homogeneity 96%–100% across the five projects.
- LiSSA: single-pass LLM baseline, weaker than TransArc (numbers `\todo{}`).
- AALinker: multi-agent pipeline; knowledge layer (alias table, ambiguity
  classification); four linker agents (explicit / contextual / anaphoric /
  abbreviated); two validation patterns (multi-pass consensus, citation-grounded
  verification).

The metric story is balanced: AALinker wins decisively under both old and new
rulers; the new metric (decision F1) reports +22.7pp where the standard ruler
reports +12.9pp; 0.164 of decision-F1 headroom remains to oracle. Future work
targets this remaining gap.

---

## How to read this playbook

Each of the eleven sections below contains four sub-documents:

- **§A — How Wang writes this section.** Verbatim exemplars from the four
  annotation files (cited as `paper · §section · ¶index`), the paragraph
  shape, the persuasive move, tone notes, and common opener / closer patterns.
- **§B — What ICSE expects.** What reviewers look for, and where Wang
  diverges from ICSE convention (helpfully or not).
- **§C — Hybrid recommendation.** 4–8 numbered rules per section, each pinned
  to either a (Wang exemplar) or an (ICSE expectation).
- **§D — Drop-in adaptations.** One or two complete drop-in paragraphs in
  Wang's argumentative style, written on our topic from the verified data.
  Missing facts become `\todo{}` — no inventions.

At the end: `§§Authoring sequence` lists which section to write first, and
which sections depend on which.

---

# §1 Abstract

### §1.A How Wang writes this section

Wang does not present abstracts in the annotation files (the four ARG_FULL
documents start at §I Introduction). The Wang model for an abstract is
recoverable from his §I.¶3 + §I.¶6 + §X.¶1 trio: an artefact name + identity
tagline in one sentence, three mechanisms in one sentence, one headline number
in one sentence. Tone: declarative, no hedging, vendor-anchored when possible.

The closest verbatim model from the corpus is the X-MIMO conclusion paragraph
(X-MIMO · §8 · ¶1):

> "This work presents X-MIMO, a cross-technology MU-MIMO on commodity devices.
> Utilizing cross-technology channel estimation and precoding, X-MIMO is the
> first work to offer cross-technology MU-MIMO on commodity devices. Our
> experiments demonstrate X-MIMO achieves the throughput of 495 Kbps, almost
> doubling the throughput of legacy ZigBee (250 Kbps), with 99% symbol
> reliability for two ZigBee receivers."

Shape: `CLAIM → CLAIM (mechanism) → EVIDENCE`.

And the UMusic conclusion (UMusic · §9 · ¶1):

> "This paper introduces UMusic, a system that uses commodity UWB devices to
> precisely detect car occupancy via lightweight signal processing techniques.
> UMusic converts CIR data into the frequency domain to obtain the channel
> frequency response, which is used to calculate the high-resolution PDP via
> the MUSIC algorithm. … The experiments show that UMusic achieves an
> aggregated accuracy of 99.4%, highlighting its effectiveness in practical
> scenarios."

Shape: `CLAIM → METHODOLOGY → METHODOLOGY → METHODOLOGY → EVIDENCE`.

Tone notes for both: no hedge words, no "we believe", no "promising". Every
sentence either names the artefact, names a mechanism, or quotes a number.
Common opener: "This paper presents/introduces X." Common closer: a single
headline number paired with the word "achieves" or "demonstrates" or
"reaching".

### §1.B What ICSE expects

ICSE abstracts are 150–250 words; reviewers expect (1) problem context in 1–2
sentences, (2) the gap or failure mode of prior work in 1 sentence, (3) the
contribution (the artefact and what it does) in 1–2 sentences, (4) the
evaluation setup in 1 sentence, (5) the headline result in 1 sentence, and
optionally (6) one sentence on threats or replication. Reviewers also expect
the abstract to read as a stand-alone artefact retrievable by search.

Where Wang diverges from ICSE:
- Wang abstracts (inferred from his conclusion-style) under-explain the
  evaluation setup. ICSE reviewers want "we evaluate on five open-source
  systems" stated explicitly.
- Wang front-loads a single headline number (e.g. "145.4× over the
  state-of-the-art" / "99.4% accuracy"). ICSE allows two or three numbers in
  the abstract if they map to distinct contributions — and our paper has two
  co-equal contributions (AALinker + metric suite), so we will need two
  headline numbers, not one.

Where Wang exceeds ICSE convention helpfully:
- The "system + mechanism + headline number" sequence is more memorable than
  the generic ICSE abstract. We should keep it.

### §1.C Hybrid recommendation

1. Open with the contribution sentence, not the context. "This paper
   presents AALinker, a multi-agent doc-to-code linker, together with a
   six-metric evaluation suite for trace-link recovery." (Wang exemplar:
   X-MIMO §8¶1.)
2. State the gap in one sentence. "Existing pipelines reach high file-level
   F1 but score only 0.596 at the decision level — leaving more than 40 % of
   linking decisions wrong." (ICSE expectation: gap statement before
   contribution.)
3. State the artefact mechanism in one compact sentence with `(i)/(ii)/(iii)`
   enumeration. (Wang exemplar M5: X-MIMO §1¶2 "(i)/(ii)" inline; UMusic
   §1¶5 three-highlight list.)
4. Name the evaluation setup explicitly: "five open-source Java systems
   (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef)" (ICSE
   expectation).
5. Quote two headline numbers (one per contribution): "+12.9pp file F1 over
   TransArc; +22.7pp at the decision-level F1 introduced by the new metric
   suite" (Wang exemplar M11: no-qualifier declarative).
6. Close with a one-sentence framing of the metric story: "the standard
   ruler underreports the gain" — co-equal stance, not modest hedge (Wang
   exemplar: UMusic §9¶1 closes with "highlighting its effectiveness").
7. No hedge words. Forbidden: "we believe", "may", "might", "promising",
   "step toward" (Wang avoidance A1; ICSE neutral).
8. Length budget 200–240 words.

### §1.D Drop-in adaptation

> This paper presents AALinker, a multi-agent doc-to-code trace-link recovery
> system, together with a six-metric evaluation suite for the same task. On
> the established five-project benchmark (MediaStore, TeaStore, Teammates,
> BigBlueButton, JabRef), the strongest prior pipeline TransArc reaches a
> file-level F1 of 0.803, but only 0.596 at the decision level — leaving
> more than 40 % of linking decisions wrong. AALinker combines a
> per-project knowledge layer (alias table, ambiguity classification), four
> linker agents specialised by reference type (explicit, contextual,
> anaphoric, abbreviated), and two validation patterns (multi-pass
> consensus, citation-grounded verification). Evaluated against TransArc,
> SWATTR, and LiSSA on the same benchmark, AALinker reaches file F1 0.931
> (+12.9pp), decision F1 0.823 (+22.7pp), component F1 0.817 aggregate
> (+10.3pp), and doc-to-model F1 0.951 (+15.2pp). The same benchmark also
> drives a six-metric evaluation suite that corrects three structural
> biases — enrollment inflation (1.0×–217.6× per project), block correlation
> (96–100 % within directory), and gold concentration (top-3 components hold
> 98.6 % of JabRef's gold) — under which the standard ruler underreports the
> doc-to-code gain by an order of magnitude. All prompts, agent code,
> per-project results, and metric implementations are available in the
> replication package \todo{cite}.

---

# §2 Introduction

### §2.A How Wang writes this section

The introduction is Wang's most stylised section. The 4-paper corpus reveals a
canonical 6–7 paragraph layout:

1. **Problem-staircase opener (T1).** A numerical fact in sentence 1 or 2;
   pivot on a single connector ("However", "Despite", "As much as") by
   sentence 3.
   Exemplar (SymBee · §I · ¶1):
   > "Explosive growth of wireless devices over the last decade is anticipated
   > to be intensified and diversified … to reach 50 billion by 2020 [2]. As
   > much as massive scale wireless body has enriched our daily lives,
   > spectrum shortage has become one of the significant bottlenecks … For
   > example, ZigBee is known to suffer from up to 50% packet loss under WiFi
   > interference [21]."
   Shape: `PREMISE → CONTRAST → CAUSE → EXAMPLE`. Move: problem-by-escalation.

   Exemplar (X-MIMO · §1 · ¶1):
   > "The number of IoT devices is expected to grow as large as a trillion by
   > 2035 [46] … For instance, ZigBee and Bluetooth have 0.125 and 1
   > bits/s/Hz, which are 240 and 30 times lower spectrum efficiencies
   > compared to WiFi 802.11n."

2. **Gap-via-two-limits (T2).** Names the prior-art category, praises it
   briefly, then closes two alternative escapes.
   Exemplar (X-MIMO · §1 · ¶2):
   > "However, achieving this in the IoT domain is challenging due to the
   > following intrinsic limitations: (i) Most of IoT devices are equipped
   > with a single antenna, while MU-MIMO needs a multi-antenna transmitter.
   > (ii) While channel estimation is an essential part of MU-MIMO, it is
   > typically unavailable in IoT … unachievable with commodity IoT."
   Shape: `DEFINITION → CLAIM → CONTRAST → PREMISE → CAUSE → CONCESSION`.

   Exemplar (UMusic · §1 · ¶2):
   > "Despite the promising applications, most of these techniques assume
   > prior knowledge of the car's occupancy status. In other words, these
   > sensing systems are unable to adaptively customize sensor parameters …
   > when the occupancy status is unknown."

3. **Headline-and-mechanism proposal (T3).** Names the artefact, identity
   tagline, mechanism, headline number.
   Exemplar (X-MIMO · §1 · ¶3):
   > "This paper presents X-MIMO, the first work to bring MU-MIMO into the
   > picture of commodity IoT networking. X-MIMO is a zero-cost, software-only
   > solution that uses pervasively-deployed commodity WiFi APs as the IoT
   > MU-MIMO transmitter … X-MIMO does not require additional hardware or
   > modification of firmware or driver."

   Exemplar (X-Disco · §I · ¶3):
   > "This paper proposes X-Disco, the first software-only cross-technology
   > neighbor discovery mechanism, to enable a WiFi device to discover the
   > ambient ZigBee neighbors without any modification to the ZigBee devices."

4. **Novelty stamp + technical highlights paragraph.** Single "to the best of
   our knowledge, … the first …" sentence (M7), then optionally `(i)/(ii)/(iii)`
   technical highlights inline.
   Exemplar (X-Disco · §I · ¶4):
   > "X-Disco is built with three new technical highlights: (i) ZigBee Symbol
   > Extraction, (ii) ZigBee Coordinator Detection, and (iii) Neighbor
   > Information Acquisition … To the best of our knowledge, X-Disco is the
   > first design to discover cross-technology neighbors using commodity WiFi
   > devices."

5. **Contribution bullets (T4).** Three bullets, each `CLAIM → METHODOLOGY →
   CLAIM/CONSEQUENCE`, ≤3 sentences.
   Exemplar (X-Disco · §I · ¶5):
   > "• We design X-Disco, the first cross-technology neighbor discovery
   > mechanism for a commodity WiFi device to detect ambient ZigBee
   > neighbors. The full compatibility with commodity WiFi and ZigBee
   > hardware and protocol ensures X-Disco's wide and practical deployment."

6. **Roadmap (T5).** One sentence per section.
   Exemplar (SymBee · §I · ¶7):
   > "Section II introduces the motivation, where Section III provides a
   > design overview. Section IV presents technical details of our design …
   > Finally, Section X concludes the paper."

**Tone notes for the introduction.** Active voice ("This paper presents",
"X-MIMO does not require"); short sentences (mean ≈ 18 words); zero hedging
in CLAIM sentences (M11). Hedging is reserved for §Discussion. Wang's
introductions almost never contain the words "we believe", "may", "might",
"promising", or "step toward" in a CLAIM position. Forward-pointers
"(details in §X)" appear three or four times across the intro (M12), so the
reader can keep moving.

**Common opener.** A numerical or vendor-anchored statistic in sentence 1:
"50 billion by 2020", "a trillion by 2035", "half billion ZigBee chips
sold", "century-long automotive transformation". Common closer: a roadmap
sentence (T5) preceded by the contribution bullets.

**Figure placement.** Figure 1 (architecture/pipeline diagram) often does
NOT appear in §I in Wang's papers — it appears in §III "Overview" (SymBee
§III.A; X-Disco §III.A; UMusic §3). The introduction is text-only in 4/4
papers.

### §2.B What ICSE expects

ICSE reviewers expect:
- A clear statement of the research problem in the first paragraph.
- A statement of prior-art limitations specific enough to anchor the
  contribution.
- An explicit statement of contributions, usually as a bullet list (Wang
  agrees here).
- An explicit roadmap (Wang agrees).
- Some ICSE introductions include explicit research questions ("This paper
  answers the following research questions: RQ1 …"); this is more common in
  empirical-study papers than design papers, but it is venue-acceptable.
  Wang never does this (A3).
- ICSE introductions typically include a "replication package available at …"
  pointer in the intro or contributions list. Wang does not, but we should.

Where Wang's style EXCEEDS ICSE convention helpfully:
- His T3 "headline-and-mechanism" paragraph (with identity tagline like
  "zero-cost, software-only") is more memorable than the generic ICSE
  pattern. Keep it.
- His one-sentence novelty stamp (M7 "to the best of our knowledge, … the
  first …") is more disciplined than the typical ICSE intro that scatters
  novelty claims across paragraphs. Keep it, used exactly once.

Where Wang diverges:
- No explicit RQs in §I (A3). For our paper this is fine — the RQs live in
  §Experiment Design, which is standard ICSE practice.
- Wang's intros do not always cite the replication package up front. ICSE
  reviewers reward early replication-package pointers.

### §2.C Hybrid recommendation

1. Paragraph 1 must contain a quantitative anchor in sentence 1 or 2 (Wang
   exemplar M1: SymBee §I¶1, X-MIMO §1¶1). For us, the anchor is either "525
   raw annotator decisions expand into 18,660 file-level pairs" or the
   headline F1 contrast 0.803 → 0.931.
2. Paragraph 2 must close two prior-art doors in one paragraph — lexical /
   transitive (TransArc, SWATTR) and single-pass LLM (LiSSA). (Wang exemplar
   T2: SymBee §II.B¶2, X-MIMO §1¶2.)
3. Paragraph 3 names AALinker with an identity tagline and the three-part
   mechanism inside a single `(i)/(ii)/(iii)` sentence (Wang exemplar M5:
   X-MIMO §1¶2, UMusic §1¶5). End with one headline number.
4. Add a metric-glimpse paragraph (already drafted in `gen/intro.tex`) — this
   is a CrystalBLEU-shaped six-sentence move and has no Wang analogue
   because Wang papers have one contribution, not two. Treat it as our
   structural innovation (ICSE expectation for co-equal contributions:
   BALANCE_ANALYSIS.md §5).
5. One single-sentence novelty stamp with "to the best of our knowledge"
   (Wang exemplar M7), placed once and only once in the paper.
6. Contribution bullets: four bullets (AALinker, metric suite, empirical
   study, replication package). Each bullet ≤3 sentences (Wang exemplar T4).
   Include the replication-package bullet (ICSE expectation).
7. Single-paragraph roadmap, one sentence per section (Wang exemplar T5:
   SymBee §I¶7).
8. No hedge words anywhere in §I (Wang avoidance A1). Reserve hedging for
   §Discussion.

### §2.D Drop-in adaptations

**§2.D.1 — Paragraph 1 (problem staircase).**

> Software projects routinely ship documentation alongside source code, and
> trace links between the two are needed for impact analysis, change
> propagation, and regulatory review \todo{cite trace-link survey}. On the
> five-project benchmark released with TransArc~\cite{transarc,replication},
> 525 raw human annotator decisions expand into 18,660 file-level data
> points, with per-project expansion factors ranging from 1.0× on MediaStore
> to 217.6× on JabRef. The strongest published pipeline reaches a file-level
> F1 of 0.803 across these five projects. However, at the granularity of a
> single human decision the same pipeline scores only 0.596, leaving more
> than 40 % of the linking decisions wrong — far below the 0.987 file-level
> F1 achievable on the oracle subset of the benchmark.

**§2.D.2 — Paragraph 3 (headline-and-mechanism).**

> This paper presents AALinker, the first multi-agent doc-to-code linker
> built on commodity LLMs. AALinker is a software-only pipeline that
> reuses the documentation and source code already on disk: it (i) builds a
> per-project knowledge layer — an alias table and an ambiguity classifier —
> in a single pass over the project, (ii) runs four linker agents
> specialised by reference type (explicit, contextual, anaphoric,
> abbreviated) on every documentation sentence, and (iii) validates every
> candidate link through two patterns — multi-pass consensus for extraction
> agents and citation-grounded verification for the anaphoric agent. On the
> same five-project benchmark, AALinker raises the average file F1 from
> 0.803 to 0.931 (+12.9pp) and the decision F1 from 0.596 to 0.823 (+22.7pp).

---

# §3 Motivation / Background

### §3.A How Wang writes this section

Wang's §II (Motivation) is consistently the section where he stacks named
industrial deployments against a quantified harm. The shape is `PREMISE →
EXAMPLE → EXAMPLE → CAUSE → CLAIM → CLAIM`, often with single-sentence SCOPE
paragraphs (T5) marking subsection boundaries.

Exemplar (X-MIMO · §2.1 · ¶1):
> "We note that a large number of ZigBee/802.15.4 IoT devices are widely
> deployed to support variant applications across different sectors
> including smart homes and factories. Amazon Echo Plus, Samsung SmartThings,
> Philips Hue, Hive, Xiaomi Mijia, and IKEA Tradfri are among a large body
> of smart home gadgets. Smart factories often operate under 802.15.4-based
> protocols, such as WirelessHART, ISA100.11a, and TSCH. For instance,
> Emerson's smart factory IoT network using WirelessHART is deployed at 54K
> smart factories worldwide, serving over 19 billion operating hours [5]."
Shape: `CLAIM → PREMISE → PREMISE → EVIDENCE → EVIDENCE → EVIDENCE → PREMISE
→ PREMISE → BRIDGE`. Move: traffic-pressure stack-up.

Exemplar (X-Disco · §II.A · ¶1):
> "For instance, 53 million Amazon Echo devices, equipped with WiFi and
> ZigBee transceivers, Philips Hue Smart Bulb, and Samsung SmartThings, were
> shipped in 2020 to support smart homes. … In such a dense WiFi and ZigBee
> coexisting environment, severe ZigBee transmission loss (≥50% ZigBee
> packets [4]) … degrades spectral efficiency … Therefore, we present
> X-Disco for commodity WiFi devices to detect the ZigBee neighbors."
Shape: `PREMISE → EXAMPLE → EXAMPLE → CAUSE → CONTRAST → CLAIM → CLAIM`.

Exemplar (UMusic · §2.1 · ¶2):
> "Despite the potential benefits, a deployable in-car occupancy detection
> system is not there yet. EU NCAP and NHTSA currently only mandate
> occupancy detection for the driver and front seats … Consequently,
> weight/pressure sensors are primarily installed in front seats … A car
> occupancy detection system, in general, should be highly accurate and
> commercially viable, prompting us to rethink the UWB technologies that are
> already integrated into the existing in-car systems."

Exemplar (UMusic · §2.3 · ¶1) — the empirical-limitation-then-counter move:
> "However, as shown in Figure 2(b), the CIR data (amplitude) collected on
> these two seats are quite similar (with a correlation 𝜌 of 0.96). Such a
> high correlation would eventually result in the ambiguity of directly
> applying CIR to detect the occupied seats. This happens because the signal
> propagation change caused by these two occupancy statuses is much less
> than the spatial resolution (60 cm). … In contrast, UMusic is designed to
> work with commodity UWB devices via a novel signal-processing technique to
> extract a high-resolution PDP …"

**Tone notes.** Sentence length grows in §Motivation (mean ≈ 24 words) vs §I
(mean ≈ 18). Wang uses passive voice ("is widely deployed", "is mandated")
more here than in §I. Heavy use of "For instance" (M10) to anchor abstract
claims in concrete brands or measurements. Heavy use of "Therefore" /
"Consequently" (M4) to close paragraphs with a verdict.

**Common opener.** A single-sentence SCOPE paragraph (T5) introduces the
section and each subsection — e.g. SymBee §II¶1 "This section illustrates
the values that SymBee would bring in diverse domains for wireless
networking …".

**Common closer.** The last paragraph of §Motivation always lands on
"Therefore, we present X-System for …" (X-Disco §II.A¶1) or "prompting us to
rethink …" (UMusic §2.1¶2) — a forward-pointer to the proposal.

**Figure placement.** Often one or two background figures appear in §Motivation
(e.g. UMusic Figure 2 on PDP resolution, X-Disco Figure 1 on Spectral Scan).
For our paper, the bias-illustration figure (or the enrollment table) sits
here naturally.

### §3.B What ICSE expects

ICSE expects §Background and §Motivation to do four things:
1. Define enough terminology that the contribution is intelligible (e.g.,
   "trace link", "architecture document", "component").
2. Establish that the problem is real and economically important — vendor
   anchors and named systems help.
3. Explain why prior approaches fail in a concrete, measurable way.
4. End with a clear bridge to the proposal.

Where Wang's style is well-aligned with ICSE: stacked named-systems
evidence (X-MIMO §2.1, X-Disco §II.A) is exactly what ICSE Empirical
Software Engineering reviewers want.

Where Wang diverges from ICSE: ICSE reviewers in the trace-link / SE
analytics community also want a quantified empirical bias diagnostic — e.g.
"on JabRef, three components produce 98.6 % of the gold standard". Wang
papers rarely use this kind of self-diagnostic on the benchmark itself
because his benchmarks are testbeds, not data corpora. For our paper this is
where we deploy the structural-inequality story, and it has no direct Wang
analogue. Treat it as a new move grafted onto his motivation template.

### §3.C Hybrid recommendation

1. Open each subsection with a single-sentence SCOPE (Wang exemplar T5:
   SymBee §II¶1, §III¶1).
2. Pair every abstract claim ("trace-link recovery matters for impact
   analysis") with a vendor anchor or quantified harm ("on JabRef, 525
   annotator decisions explode into 18,660 file pairs") in the same
   paragraph (Wang exemplar M10: X-MIMO §2.1¶1).
3. Use the structural-inequality move as the bias-diagnostic paragraph
   (already drafted in `gen/metric.tex` §Background) — three short paragraphs
   tagged "Enrollment inflation", "Block correlation", "Gold concentration"
   in italics. (No Wang analogue; ICSE expectation for empirical-bias
   evidence.)
4. End the section with a `Therefore` sentence pointing to the proposal —
   for us, the structural-inequality observation calls both for AALinker and
   for the metric suite (Wang exemplar M4 + the closing-bridge move at
   X-Disco §II.A¶1 last sentence).
5. Single Figure or Table in this section (the enrollment table). Keep it.
6. Resist the urge to recap contributions here (Wang avoidance A6).
7. Length budget: 1.25 pages (per `PAPER_PLAN.md`).
8. Don't introduce the six metrics yet — that is §Metric Suite's job.

### §3.D Drop-in adaptation

> Software documentation routinely accumulates over the lifetime of a
> project: textbook descriptions, ADRs, design notes, and architecture
> overviews accrete into hundreds of pages, while the source tree grows
> independently. On the established five-project benchmark for doc-to-code
> trace-link recovery~\cite{transarc,replication}, the gold standard is
> built from 525 raw human annotator decisions on five Java open-source
> systems — MediaStore, TeaStore, Teammates, BigBlueButton, and JabRef. To
> compute file-level precision, recall, and F1, each directory-granularity
> decision is expanded to every file inside that directory, turning the 525
> decisions into 18{,}660 file-level data points with per-project expansion
> factors of 1.0× (MediaStore) to 217.6× (JabRef).
>
> This expansion is not uniform across the benchmark. On JabRef, three
> components alone produce 98.6\% of the file-level gold standard, and the
> single largest component covers 47\% of the gold; files inside the same
> directory share their outcome 96\%–100\% of the time across the five
> projects. The file-level F1 thus reports the result of perhaps three
> decisions on JabRef, masquerading as the result of 8{,}268. A pipeline
> that wins on the three largest components scores above 0.94 file F1
> regardless of its behaviour on the other twelve. Therefore, fairly
> reporting the gap between today's pipelines and the oracle (file F1
> $\approx0.987$ on the oracle subset; decision F1 ceiling $\approx 1.0$)
> requires both a better linker and a metric set whose units match a single
> human decision, not a single enrolled file.

---

# §4 Approach (AALinker — multi-agent linker)

### §4.A How Wang writes this section

Wang's §III ("Overview / Background") and §IV ("Main Design") have a
distinctive two-stage rhythm: a `T5 SCOPE` sentence opens the section, then a
`T6 background walkthrough` paragraph chains METHODOLOGY sentences and lands
on an "exploitable property" sentence at the end, then the §IV.x subsections
each name one component and execute another walkthrough.

Exemplar (SymBee · §III.A "SymBee in a Nutshell" · ¶1):
> "SymBee is a ZigBee to WiFi CTC technique that vastly improves the data
> rate of the state-of-the-art designs by exploiting physical layer
> signatures. This is effectively achieved by the two unique features: (i)
> ZigBee's physical layer signature is indirectly controlled by manipulating
> the payload bytes, which we call payload encoding, such that (ii) the
> payload exposes intended (i.e., message-bearing) patterns when … In
> other words, SymBee is carefully designed not only considering the
> physical layer properties of ZigBee and WiFi, but also their
> cross-observability …"
Shape: `CLAIM → METHODOLOGY → INTERPRETATION`. Move: design-in-a-nutshell.

Exemplar (X-Disco · §III.A · ¶1):
> "X-Disco is a two-step approach, containing four messages (M1 to M4)
> exchanged between an X-Disco device and ambient ZigBee coordinators. As
> illustrated in Figure 1, in Step (a), the X-Disco device transmits an
> emulated ZigBee broadcast packet in message M1 … In Step (b), the X-Disco
> device requests the ZigBee neighbor information … By leveraging the
> ZigBee Device and Service Discovery mechanism, X-Disco detects ZigBee
> neighbors via fetching the neighbor information from the ZigBee
> coordinator, with only four messages exchanged. As ZigBee coordinators
> are always in active mode, the exchanged messages are naturally immune
> to the duty-cycle related problems, thereby achieving the minimum
> overhead."
Shape: `DEFINITION → METHODOLOGY → METHODOLOGY → CLAIM → CONSEQUENCE`.

Exemplar (UMusic · §3 "Design Overview" · ¶1):
> "As depicted in Figure 3, UMusic comprises three steps: (i) CIR data is
> collected from multiple links … (ii) UMusic calculates the high-resolution
> PDP from the collected CIR data. Finally, the PDPs obtained from multiple
> Tx-Rx links are fed into a simple classification model in Step (iii) to
> detect the car occupancy status. … However, due to hardware imperfections,
> the phase information in CIR is highly biased, leading to more practical
> issues in the design of UMusic. The following section presents our
> solutions for these challenges."

Exemplar (X-Disco · §III.B.2 · ¶1) — the climactic "exploitable property"
sentence at the end of a background walkthrough:
> "As Figure 4 depicts, in Step (i), the mixer shifts the passband signal to
> the baseband … Finally, in Step (iv), the rest 64 samples are fed into
> FFT calculation, which outputs the corresponding magnitude while the
> phase information is left out. Since this process does not require the
> received signal to be WiFi, an arbitrary signal (e.g., ZigBee) will be
> reflected in FFT magnitude if Spectral Scan mode is on."

**Tone notes.** Sentence length is shortest in this section (mean ≈ 16
words). Heavy use of step enumerators ("(i)/(ii)/(iii)", "Initially / Then /
Finally / Since" — M5). Active voice. Forward pointers to figures (M12) are
ubiquitous: "As illustrated in Figure X", "As depicted in Figure Y".

**Common opener.** Single-sentence SCOPE (T5). Example (SymBee · §IV · ¶1):
"This section provides technical details and insights on SymBee."

**Common closer.** A `CONSEQUENCE` sentence that names the practical payoff
of the design: "thereby achieving the minimum overhead" (X-Disco §III.A¶1)
or "thereby minimizing the computation cost while maintaining compatibility
to the WiFi standard" (SymBee §III.A¶2). The connective "thereby" is the
single most characteristic word of Wang's approach section.

**Figure placement.** Figure 1 (the architecture/pipeline diagram) appears
in §III.A "X in a Nutshell" in all four Wang papers. Subsequent figures (Fig
2, 3) appear in §III.B / §IV.A background walkthroughs to anchor each
mechanism.

### §4.B What ICSE expects

ICSE expects an approach section to:
- State the inputs and outputs of the system precisely.
- Present an overview figure / pipeline diagram early (usually Figure 1).
- Justify each design choice — not just describe it.
- Include enough detail that a reader could reimplement (or describe what is
  in the replication package).
- Use formal notation when appropriate (definitions, algorithms), but not
  gratuitously.

Where Wang's style matches ICSE: the pipeline figure in §III, the
background-walkthrough paragraph (T6) that lands on the exploitable
property, the "thereby" consequence closer.

Where Wang's style EXCEEDS ICSE convention helpfully:
- His "X in a Nutshell" subsection is a small ritual most ICSE approach
  sections lack. It gives reviewers a 200-word summary of the whole pipeline
  before the technical detail starts. Adopt it.
- His M6 "as simple as Y" simplicity tagline is rare in ICSE approach
  sections but invaluable for managing a multi-agent system that looks
  complex on the surface. For us: "Building the knowledge layer is as
  simple as one LLM pass over the source tree and one pass over the
  documentation."

Where Wang diverges:
- He rarely uses pseudocode. ICSE reviewers in the SE-tooling community
  often expect Algorithm 1 (compact pseudocode) somewhere in §IV. If the
  AALinker pipeline benefits from one, include it. If not, the four-agent
  table from `approach.tex` (`tab:strategies`) does the job.

### §4.C Hybrid recommendation

1. Open §Approach with a single-sentence SCOPE paragraph (Wang exemplar T5).
2. Make §4.A "AALinker in a Nutshell" a short subsection (≤200 words) with
   the pipeline figure (Wang exemplar: SymBee §III.A, X-Disco §III.A,
   UMusic §3). Three sentences max: claim, three-step mechanism via
   `(i)/(ii)/(iii)` (M5), and one CONSEQUENCE sentence with "thereby" (M14).
3. For each of the three layers (knowledge layer, four linker agents, two
   validators), run a T6 background-walkthrough paragraph and land on the
   exploitable property (Wang exemplar: X-Disco §III.B.2 climax).
4. Use one M6 "as simple as Y" tagline per subsection — e.g. "Building the
   knowledge layer is as simple as one LLM pass per project" (Wang exemplar:
   SymBee §I¶4).
5. Place Table~\ref{tab:strategies} (already in `approach.tex`) at the
   boundary between §Agents and §Validators (ICSE expectation: visual
   summary of design decisions).
6. Close every subsection with a "thereby / essentially / uniquely"
   consequence sentence (Wang exemplar M14).
7. Resist re-stating contributions inside §Approach (Wang avoidance A6).
8. Length budget: 2.5 pages (per `PAPER_PLAN.md`).

### §4.D Drop-in adaptations

**§4.D.1 — "AALinker in a Nutshell" subsection skeleton.**

> AALinker is a multi-agent doc-to-code linker that combines a per-project
> knowledge layer with four specialised linker agents and two validation
> patterns. As illustrated in Figure~\ref{fig:pipeline}, AALinker proceeds
> in three steps: (i) one LLM pass over the source tree and one pass over
> the documentation build an alias table and an ambiguity classifier
> covering the project's named entities; (ii) for every documentation
> sentence, four linker agents — explicit (canonical names), contextual
> (paraphrased descriptions), anaphoric (pronouns and demonstratives), and
> abbreviated (partial names) — propose candidate links against the
> knowledge layer; (iii) every candidate link is then validated through one
> of two patterns: extraction agents pass through multi-pass consensus
> across two LLM samples, and the anaphoric agent passes through
> citation-grounded verification that demands a quoted antecedent. The four
> agents share the same knowledge layer, thereby keeping the per-project
> setup cost to a single pair of LLM passes regardless of how many
> documentation sentences are linked downstream.

**§4.D.2 — Knowledge-layer subsection paragraph (T6 walkthrough).**

> The knowledge layer turns a project into a vocabulary the four linker
> agents can share. Given a source tree, AALinker first scans every file
> header and identifier (\todo{define which scopes}) to build a canonical
> name set, then runs a single LLM pass over the documentation to extract
> every named entity referenced in the text. The two sets are then
> reconciled into an alias table (mapping documentation references to
> canonical names) and an ambiguity classification (flagging mentions whose
> documentation form maps to multiple canonical names). Because every
> linker agent reads from this table rather than re-resolving aliases
> on its own, AALinker resolves the same alias the same way for every
> sentence in the document — thereby eliminating the per-sentence aliasing
> error that single-pass LLM linkers like LiSSA accumulate. Building the
> knowledge layer is as simple as two LLM passes per project, executed once
> before any linking decision is made.

---

# §5 Metric Suite (six metrics)

### §5.A How Wang writes this section

Wang does not present metric-suite sections. None of his four papers
introduces a new evaluation methodology as a contribution; they all use
standard SER, BER, throughput, or accuracy. The closest analogues in his
corpus are the brief metric-definition paragraphs inside §VI / §VIII (Wang
spends ≤ 1 paragraph defining each metric, then moves on).

Exemplar (UMusic · §6.10.2 · ¶1) — Wang's defining-a-metric move:
> "The overall PDP calculation error is shown in Figure 29(a), where the
> average error is 0.016 m, which matches with the 8 paths calculation
> result in Section 6.10.1 obtained under no hardware imperfections."
This is a sub-sub-section. The metric (error in m) is barely defined before
results are reported.

Exemplar (X-MIMO · §6.5 · ¶3 "Method") — closest Wang move to a metric
definition:
> "The absolute phase of the estimated ZigBee channel is affected by the
> hardware uncertainty, resulting in a time-variant estimation compared to
> the ground truth. Hence, in this experiment, we use the relative phase
> between two ZigBee channels as the metric to check the precision of the
> phase of the estimated ZigBee channel … Specifically, the relative phase
> between the estimated channel 𝑝1ℎ11 and 𝑝1ℎ12 is identical with
> ∡(ℎ11,ℎ12). Thus, the relative phase is kept the same within channel
> coherent time."
Shape: `PREMISE → METHODOLOGY → INTERPRETATION → EXAMPLE → CONSEQUENCE`.
Move: metric-design-derived-from-immunity-theorem.

**This is the most important "Wang weakness" finding.** §Metric Suite has
no Wang model. We must lean on ICSE convention and on the CrystalBLEU
template (which is closer to our shape).

The transferable Wang moves are:
- M8 "This is because" pattern — every metric definition should be followed
  by a one-sentence "this is because" mechanism (i.e., why the standard F1
  misses what this metric catches).
- M5 `(i)/(ii)/(iii)` compressed enumeration — list the six metrics inline
  in one sentence before defining them.
- M11 no-qualifier declarative — name each metric and what it measures
  without "we believe this captures …".
- M12 forward pointers — every metric definition closes with a forward
  pointer to the RQ where it is used.

### §5.B What ICSE expects

ICSE expects metric-suite sections to do four things:
1. Justify why each metric is needed (typically by pointing at a deficiency
   of the standard metric).
2. Define each metric precisely — symbolic formula or equation if it is
   non-trivial.
3. State the range and the interpretation of high/low values.
4. Pair each metric with sanity baselines (random predictor, oracle) when
   the metric is not absolutely-scaled (this is the CrystalBLEU and SARI
   pattern; see BALANCE_ANALYSIS §3 and §7.1).

Where Wang's style is too thin: Wang gives one paragraph per metric. ICSE
reviewers in the empirical SE community want at least one paragraph PLUS
a formula PLUS the interpretation paragraph.

Where the CrystalBLEU / SARI pattern helps: both papers organise their
metric section as "background → flaw → proposal → validation". Our
`gen/metric.tex` already follows this shape.

### §5.C Hybrid recommendation

1. Open with a single-sentence SCOPE (Wang exemplar T5).
2. State the six metrics inline in one `(i)–(vi)` sentence (Wang exemplar
   M5) before defining any of them. Names: per-component F1, per-sentence
   F1, sentence coverage, noise rate, coverage-and-purity, skill score —
   plus the decision-level F1 already promised in the intro (so the suite
   is actually seven if decision-F1 is counted; rationalise in
   `gen/metric.tex` review).
3. For each metric, run a four-sentence block: (a) name and one-sentence
   purpose; (b) formula or one-sentence definition; (c) range and
   interpretation; (d) `This is because …` sentence linking the metric to
   one of the three structural-inequality biases from §Motivation (Wang
   exemplar M8).
4. Bracket the suite with sanity baselines: random predictor avg file F1
   ≈ 0.155; oracle-subset file F1 ≈ 0.987; oracle-component F1 ≈ 0.906
   (ICSE expectation; CrystalBLEU pattern).
5. Resist defining metrics that are not used in §Results (ICSE expectation
   "every metric must do work").
6. Close §Metric Suite with a one-sentence forward pointer to §Results
   (Wang exemplar M12).
7. Length budget: 1.75 pages (per `PAPER_PLAN.md`); already 80 lines of
   `metric.tex` is drafted, of which §5.2 (definitions) is a stub.
8. Plain English: avoid SAD, SAM, ACF1, HUS — established terms only (Wang
   discipline; user-preference rule).

### §5.D Drop-in adaptation

**§5.D.1 — Six-metric inline preview sentence (M5 compressed enumeration).**

> Our suite contains six metrics chosen to correct one or more of the three
> structural biases of Section~\ref{sec:metric:bias} or to expose a
> property of trace-link quality that the standard file-level F1
> structurally cannot: (i) per-component F1 equalises the weight of small
> and large components; (ii) per-sentence F1 averages F1 over gold
> sentences to expose concentration on a few sentences; (iii) sentence
> coverage and (iv) noise rate together quantify the developer-facing
> behaviour (how many sentences receive at least one correct link, and how
> many wrong links arrive per sentence); (v) coverage-and-purity is the
> harmonic mean of the two and gives a single developer-facing number; and
> (vi) the skill score rescales F1 between a random predictor
> (avg.\ file F1 $\approx 0.155$ on our benchmark) and the oracle
> (avg.\ subset file F1 $\approx 0.987$, oracle-component F1
> $\approx 0.906$) so that projects of different difficulty become
> commensurable.

**§5.D.2 — One full metric block (per-component F1, with M8 cause).**

> \textbf{Per-component F1.} For each project we compute precision, recall,
> and F1 over the file-level (sentence, file) gold pairs once per component
> in the architecture model, then average across components rather than
> across files. This gives every component the same weight, so a system that
> gets the small components right is rewarded as much as one that gets the
> three largest components right. The metric ranges in $[0,1]$ and reaches
> $0.906$ on the oracle-component split of our benchmark — meaning per-
> component F1 is not absolute, and an oracle ceiling is the right
> reference. This is because the standard file-level F1 is, on JabRef,
> essentially the F1 over three components (which alone hold 98.6\% of the
> gold), while per-component F1 reports the average over all components and
> exposes the regression on the smaller ones — as we will show in
> Section~\ref{sec:exp:rq2}.

---

# §6 Experiment Design (RQs, setup, dataset)

### §6.A How Wang writes this section

Wang has no §Experiment Design section. He has §VIII Evaluation (SymBee) /
§6 Evaluation (X-MIMO, UMusic) / §VI Evaluation (X-Disco), all of which open
with an §A.1 "Implementation" subsection and then jump to results. The setup
material is distributed: implementation in §VI.A, dataset in §VI.A (one
paragraph), and the metric is defined inline at the first result that uses
it.

Exemplar (X-Disco · §VI · ¶1):
> "We build X-Disco on USRP B210 and TP-link WDR 4300 WiFi router … The
> primary metric to evaluate X-Disco is the time consumed for discovering
> all the ZigBee neighbors. We evaluate X-Disco in the office (None
> Line-of-sight) and the hallway (Line-of-sight). We also evaluate the
> advanced features of X-Disco in the office."
Shape: `METHODOLOGY → CLAIM → METHODOLOGY × n → DEFINITION → SCOPE → SCOPE`.

Exemplar (UMusic · §6.10 · ¶1) — the rare Wang move that uses explicit
"questions":
> "Specifically, our simulations are conducted to answer three major
> questions: (i), how many paths could be precisely resolved …? (ii), are
> the three hardware imperfections fully immune? … (iii), how could the
> computational cost reduction affect the precision …?"
This is the closest Wang gets to RQs. They are not labelled "RQ1/RQ2" and
they live inside §6.10, not as a standalone subsection.

**Wang has no analogue for ICSE-style numbered RQ environments.** This is
the section where ICSE convention dominates.

### §6.B What ICSE expects

ICSE expects:
1. Explicit RQs in a numbered environment ("RQ1: Does AALinker recover
   better trace links than existing approaches?"). For empirical-study
   papers this is mandatory; for design papers it is recommended.
2. Per-RQ motivation paragraph (1–4 sentences).
3. Per-RQ experimental design subsection.
4. A §Dataset subsection naming every benchmark project with citations.
5. Statement of statistical analysis where applicable (CI, significance
   tests).
6. Replication-package pointer.

Our current `gen/eval.tex` already follows the ICSE pattern faithfully. The
RQ environments use `\begin{enumerate}` with `series=researchQuestions` and
labels, and each RQ has its own motivation paragraph. This section should
stay close to its current form.

### §6.C Hybrid recommendation

1. Keep the formal RQ environments — this is ICSE convention; Wang has no
   competing model (ICSE expectation; Wang absence A3).
2. Open §6 with a single-sentence SCOPE paragraph (Wang exemplar T5:
   SymBee §III¶1, X-Disco §VI¶1).
3. Use a `(i)/(ii)/(iii)/(iv)` enumeration of the four experiment families
   in the SCOPE paragraph (Wang exemplar M5).
4. For each RQ motivation paragraph, run a four-sentence T7-like shape
   (`METHODOLOGY → PREMISE → CLAIM → CONSEQUENCE`) — see the current
   `gen/eval.tex` which already does this.
5. Name the five projects exactly once, in §Dataset, with citations (Wang
   exemplar M10 vendor-anchor — for us, the projects are the vendors).
6. Quote the enrollment expansion 525 → 18,660 once in §Dataset; do not
   repeat it (Wang discipline; the number lives in §Motivation already).
7. Justify the choice of baselines (TransArc, SWATTR, LiSSA) in one
   sentence each, with a forward pointer to §RW (Wang exemplar M12).
8. Length budget: 1.5 pages (per `PAPER_PLAN.md`).

### §6.D Drop-in adaptation

**§6.D.1 — SCOPE paragraph for §Experiment Design.**

> This section evaluates AALinker on the established five-project doc-to-code
> trace-link benchmark across four experiment families: (i) a head-to-head
> comparison with TransArc, SWATTR, and LiSSA under the standard ruler
> (\ref{RQ1 SOTA comparison}); (ii) a re-evaluation of the same four systems
> under the architecture-driven metric suite of
> Section~\ref{sec:metric:suite} (\ref{RQ2 metrics}); (iii) a
> validator-ablation study isolating the contribution of the two
> validation patterns (\ref{RQ3 validators}); and (iv) a per-agent ablation
> isolating the contribution of each of the four linker agents
> (\ref{RQ4 ablation}). All AALinker runs use a fixed model configuration
> (Section~\ref{sec:hyperparameters}) so that the results are reproducible
> despite LLM non-determinism.

---

# §7 Results

### §7.A How Wang writes this section

Wang's evaluation section is the most stable across the four papers. Every
result subsection follows pattern T7: `METHODOLOGY → EVIDENCE → CAUSE → CLAIM`
(occasionally `METHODOLOGY → EVIDENCE → EVIDENCE → INTERPRETATION`). Every
number has a one-sentence mechanism attached via M8 ("This is because", "due
to", "Since"). Every paragraph closes with a verdict sentence (M4
"Therefore / Thus / As a result").

Exemplar (UMusic · §6.2.1 / §6.2 · ¶1, the most representative T7):
> "The median accuracy of UMusic for 1 to 4 people is 97.2%, 93.3%, 87.2%,
> and 83.1%, respectively, representing improvements of 16.8%, 17.9%,
> 14.3%, and 13.7% compared to CarOSense, which achieves median accuracies
> of 83.2%, 79.1%, 76.3%, and 73.1%, respectively. This result demonstrates
> that our high-resolution PDP effectively captures the significant changes
> in signal propagation due to human occupancy …"
Shape: `EVIDENCE → CLAIM`. Move: numbers-then-mechanism.

Exemplar (SymBee · §VIII.A · ¶3) — the head-to-head with named baselines:
> "Throughput of C-morse is 215bps when distance between sender and receiver
> is 1.5 meters in the office scenario. Figure.16 indicates SymBee
> outperforms C-Morse, the state-of-the-art ZigBee to WiFi cross-technology
> approach, by 145.4×."
Shape: `METHODOLOGY → METHODOLOGY → EVIDENCE → CLAIM`.

Exemplar (X-MIMO · §6.2 · ¶2) — the **T8 honest-failure-case**:
> "The SER of X-MIMO for two ZigBee devices at position 1 is 1% and 27%,
> exhibiting a significant imbalance. This is because the channel from
> X-MIMO to ZigBee 2 is so weak that the signal for ZigBee 1 keeps
> dominating the ZigBee device 2."
Shape: `EVIDENCE → CONCESSION → CAUSE → CLAIM`. Move: own the bad number.

Exemplar (SymBee · §VIII.C · ¶1) — the other honest-failure case:
> "Even though S3 is closer to R than S2, throughput of S3 is lower than
> S2 due to more blockages from walls. This indicates the walls decreases
> the throughput of SymBee severely along with the distance between sender
> and receiver."

Exemplar (UMusic · §6.10.3 · ¶1) — multi-data-point paragraph with verdict:
> "The detailed results are shown in Figure 30, where the average time
> consumption is 3.38 ms, 1.15 ms, and 0.125 ms under the downsampling
> factor of 1, 2, and 4 respectively. Compared to 𝐷=1, the time
> consumption is reduced by 2.94 and 27.1 times. Meanwhile, the accuracy
> experiences a negligible degradation due to downsampling, as shown in
> Figure 30(b). This result supports UMusic to operate in real-time since
> the interval between consecutive CIR collections is 50 ms."

**Tone notes.** Sentence length is shortest in §Results (mean ≈ 14 words).
Active voice throughout. Headline numbers always paired with their absolute
pair (e.g. "0.803 → 0.931", not "+12.9pp" alone). No hedge words.

**Common opener.** Single-sentence SCOPE for each result subsection
(SymBee §VIII.B¶1 "We present robustness of our design by evaluating bit
error rate (BER).") or one METHODOLOGY sentence introducing the experiment.

**Common closer.** A "Therefore / Thus / This indicates" sentence that
delivers the verdict (Wang exemplar M4).

**Figure placement.** One figure per evaluation subsection in 4/4 papers.
Figures are named first ("Figure 16 indicates …"), then numbers cited from
them.

**Honest-failure-case rule.** Wang reserves exactly one T8 paragraph per
evaluation section for the single worst datum. He owns it, names the cause,
and closes with the aggregate gain. This is the move we adopt for the
Teammates component-level regression (≈ −9pp).

### §7.B What ICSE expects

ICSE expects:
1. Each RQ to be answered explicitly — usually with a labelled
   "**RQ1 answer:** …" paragraph or a `\paragraph{Answer to RQ1}` block.
2. Tables for head-to-head baseline comparisons; numbers in tables, not in
   prose.
3. Statistical significance tests where applicable.
4. A discussion of any negative or surprising results — Wang's T8 pattern
   is the right move here.
5. Ablation tables for both per-validator and per-agent studies.
6. A summary figure or paragraph at the end of §Results that compresses
   the four RQ answers into one paragraph.

Where Wang's style is well-aligned: T7 paragraphs are exactly the right
shape for RQ result paragraphs; T8 is exactly the right honesty move.

Where Wang's style is too thin: he does not write `\paragraph{Answer to
RQX}` labels. We should add them for ICSE reviewer ergonomics.

### §7.C Hybrid recommendation

1. Open each RQ subsection with the table reference and a one-sentence
   summary, then run T7 paragraphs underneath (Wang exemplar T7).
2. Every numerical claim must be paired with its absolute pair: "0.803 →
   0.931 (+12.9pp)" not "+12.9pp" alone (Wang exemplar M11).
3. Every result paragraph must contain a CAUSE sentence (M8 "This is
   because") — no number stands alone.
4. Use exactly one T8 honest-failure-case paragraph: the Teammates
   component-level regression at ≈ −9pp (Wang exemplar T8: X-MIMO §6.2¶2,
   SymBee §VIII.C¶1).
5. Add a `\paragraph{Answer to RQX}` label at the end of each RQ subsection
   summarising the verdict in one sentence (ICSE expectation; Wang's style
   doesn't do this but it costs nothing and helps reviewers).
6. Close each RQ paragraph with "Therefore / Thus" (Wang exemplar M4).
7. Tables for head-to-head numbers; grid figure for the metric-suite
   matrix (RQ2). The four-systems × five-projects × full-metric panel is
   Figure~2 in the page budget.
8. Length budget: 2.5 pages (per `PAPER_PLAN.md`).

### §7.D Drop-in adaptations

**§7.D.1 — RQ1 headline result paragraph (T7).**

> Table~\ref{tab:rq1-doc-to-code} reports the doc-to-code performance of
> AALinker against TransArc, SWATTR, and LiSSA on the five-project
> benchmark. AALinker reaches an average file F1 of 0.931, against
> TransArc's 0.803 — a gain of +12.9 percentage points. The same picture
> holds at every tested granularity: doc-to-model F1 rises from 0.799 to
> 0.951 (+15.2pp); decision-level F1 from 0.596 to 0.823 (+22.7pp); and
> aggregate component F1 from 0.714 to 0.817 (+10.3pp). LiSSA, the single-
> pass LLM baseline, scores below TransArc on all four metrics
> (\todo{LiSSA file F1 number}, \todo{LiSSA decision F1}). This is because
> single-pass classification cannot resolve an alias once and reuse it
> across the document — every sentence renegotiates the linking decision
> for the same name. Therefore, AALinker closes most of the gap to the
> oracle ceiling at every granularity (file F1 $0.987$ oracle subset,
> component F1 $0.906$ oracle-component), with the largest residual on the
> decision-level F1 axis (0.164 to the absolute oracle).
>
> \paragraph{Answer to \ref{RQ1 SOTA comparison}}
> AALinker improves over the strongest published baseline by +12.9pp file
> F1, +22.7pp decision F1, +15.2pp doc-to-model F1, and +10.3pp aggregate
> component F1; LiSSA's single-pass design does not reach TransArc.

**§7.D.2 — Honest-failure-case paragraph (T8) for Teammates component F1.**

> The component-level result is the least uniform of the four metrics. On
> four of the five projects AALinker improves component F1 by between +4pp
> (JabRef) and +35pp (MediaStore); on Teammates alone it regresses by about
> $-9$pp. This is because \todo{name the specific component, e.g.\
> ``the controller component, where the alias table merges two distinct
> controllers under a common English name''}, and because the knowledge
> layer is built once per project, the aliasing error propagates to every
> later agent. The other four projects show consistent gains, and the
> aggregate component F1 still rises from $0.714$ to $0.817$. We discuss
> this regression and its implications for the design of the knowledge
> layer in Section~\ref{sec:discussion}.

---

# §8 Discussion

### §8.A How Wang writes this section

Wang has a §7 / §VII / §Discussion section in only one of the four papers
(UMusic §7 "Discussion and Future Work"). The other three jump from §VIII
Evaluation directly to §IX Related Work. Where he does write one, the
paragraphs are short (≤ 4 sentences) and reserved for edge cases and
extensions, not threats to validity.

Exemplar (UMusic · §7 · ¶1 "Impact of tall passenger"):
> "In the experiment, UWB devices are mounted on the car's ceiling to
> minimize the impact of tall passengers. Even if an exceptionally tall
> passenger blocks the LoS path, UMusic can handle this effectively.
> Although PDP synchronization may be affected, this does not hinder
> occupancy detection, as the blockage also impacts longer paths, making
> the shortest affected path closer in length to the LoS path."
Shape: `METHODOLOGY → CLAIM → CONCESSION → CONSEQUENCE`. Move: edge-case
conversion into feature.

Exemplar (UMusic · §7 · ¶4 "Full support for HVAC and vital sign
applications"):
> "UMusic is designed to provide occupancy status, a prerequisite for HVAC
> systems, vital sign monitoring, and detect children left alone.
> Currently, UMusic excels at occupancy detection … However, UMusic
> requires further enhancements to detect physiological signals … which
> would be addressed in future work."
Shape: `DEFINITION → CLAIM → CONCESSION`. Move: present-strength with
future-scope.

**The Wang pattern: concessions modulate the proposal only here.** §I, §IV,
§VIII are concession-free; §Discussion is the one section where "Although",
"While", "However" land on the proposal itself.

**Wang doesn't write classical threats to validity.** §Threats is an ICSE
ritual he avoids — there is no exemplar to follow.

### §8.B What ICSE expects

ICSE expects two distinct things often labelled together as "Discussion and
Threats":
1. **Discussion proper** — Why does the system work? What does the
   non-obvious finding mean? What are the implications for the field? (Wang
   has a model here: edge-case conversion, future-work pointers.)
2. **Threats to Validity** — Standard four-category structure:
   - *Construct validity* (the metric measures what it should).
   - *Internal validity* (the experiment isolates the cause).
   - *External validity* (the result generalises).
   - *Conclusion validity* (the statistical inference is sound).
   Wang has no model for this; ICSE convention dominates.

In our paper, given two co-equal contributions and one shared benchmark,
the §Discussion should also include a "why AALinker wins more on the new
metrics" paragraph and a "Teammates regression at component level"
paragraph (already planned in `PAPER_PLAN.md` §7).

### §8.C Hybrid recommendation

1. Open §Discussion with the "why AALinker wins more on the new metrics"
   paragraph — this is where the metric story pays off. One paragraph,
   shape `CLAIM → MECHANISM → CONSEQUENCE`. (No Wang exemplar; this is the
   load-bearing paragraph for the metric contribution.)
2. The Teammates regression paragraph belongs here, in T8 shape (Wang
   exemplar T8). It is the same content as the T8 paragraph in §Results
   but framed as design implication ("the knowledge layer must be made
   ambiguity-aware") rather than as result.
3. §Threats to Validity sub-section with four subsubsections (Construct,
   Internal, External, Conclusion). Each ≤ 2 sentences. (ICSE expectation;
   no Wang model.)
4. Future-work pointer in the closing sentence — close the decision-F1
   gap to oracle (0.164 headroom remains). Use Wang's `CLAIM → CONCESSION
   → BRIDGE` shape (UMusic §7¶4).
5. Do NOT recap contributions here (Wang avoidance A6; ICSE neutral).
6. Length budget: 1 page (per `PAPER_PLAN.md` §7).

### §8.D Drop-in adaptation

**§8.D.1 — "Why AALinker wins more on the new metrics" paragraph.**

> AALinker's gain over TransArc grows as the metric moves closer to the
> human decision. On the standard file-level F1, AALinker improves by
> +12.9 percentage points; on the decision-level F1 matched to a single
> human annotator decision the improvement is +22.7 percentage points.
> This is because the standard ruler enrols every directory decision into
> all of its files, so a correct decision on JabRef's three largest
> components (98.6\% of the file-level gold) is rewarded thousands of
> times while a correct decision on a small component is rewarded once.
> AALinker concentrates its gains on the harder, smaller decisions —
> resolving aliased mentions, anaphoric references, and abbreviated
> names — exactly the decisions the standard F1 is least sensitive to.
> The metric suite of Section~\ref{sec:metric:suite} therefore does not
> change the ranking, but it reports the gain at its actual size.

**§8.D.2 — Threats to Validity subsection skeleton.**

> \paragraph{Construct validity.} The new metrics in
> Section~\ref{sec:metric:suite} are designed against the structural
> inequality of the benchmark, not against AALinker's results, and each
> metric is bracketed by a random predictor (file F1 $\approx 0.155$) and
> an oracle (file F1 $\approx 0.987$, component F1 $\approx 0.906$). LLM
> non-determinism is mitigated by the multi-pass consensus validator and
> by fixing model and seed (Section~\ref{sec:hyperparameters}).
>
> \paragraph{Internal validity.} All baselines are taken from the original
> papers to avoid re-tuning bias~\cite{transarc,lissa}. The knowledge
> layer is built from the source tree and documentation alone — no
> benchmark-specific tuning is performed (\todo{cite BENCHMARK\_TABOO.md
> protocol}).
>
> \paragraph{External validity.} The benchmark contains five Java
> open-source projects with hand-curated architecture documents. The
> metric suite is task-agnostic but AALinker's validation patterns are
> only evaluated on this corpus.
>
> \paragraph{Conclusion validity.} Block homogeneity within a directory
> (96\%--100\% across the five projects) means file-level F1 has many
> fewer effective data points than the 18{,}660 raw count suggests; we
> therefore report decision-level F1 alongside file-level F1 throughout
> Section~\ref{sec:results}.

---

# §9 Threats to Validity

(Embedded in §Discussion above; this section in our paper may be merged
into §8 to save page budget. The drop-in template is in §8.D.2.)

### §9.A How Wang writes this section

**Wang does not write threats-to-validity sections.** None of the four
papers (SymBee, X-MIMO, X-Disco, UMusic) contains a §Threats to Validity.
This is the section where Wang's style is least useful as a model. Adopt
ICSE convention.

### §9.B What ICSE expects

The four-category structure (Construct / Internal / External / Conclusion)
is standard at ICSE. Reviewers expect each category to be addressed in 1–3
sentences, with mitigation strategies listed where relevant.

### §9.C Hybrid recommendation

1. If page budget allows a standalone §Threats subsection, give it ≤ 0.5
   pages.
2. Otherwise merge into §Discussion (per `PAPER_PLAN.md` §7.3).
3. Use the four-category structure (ICSE expectation).
4. State mitigations, not just concessions.
5. Length budget: 0.5 page.

### §9.D Drop-in adaptation

(See §8.D.2 above for the threats subsection skeleton.)

---

# §10 Related Work

### §10.A How Wang writes this section

Wang's §Related Work sections are short, theme-organised, and disciplined.
Three short paragraphs per paper on average (SymBee §IX has 4, X-MIMO §7
has 2, X-Disco §VII has 3, UMusic §8 has 2). Never one paragraph per
competitor — always one paragraph per *theme*, with named competitors
inside.

Exemplar (X-Disco · §VII · ¶2):
> "Based on CTC, many works improve the performance of channel
> coordination [5] and cooperation [7]. Two papers [10], [24] claim they
> focus on the cross-technology neighbors discovery. However, applying the
> WiFi to ZigBee CTC to assist ZigBee devices in detecting ZigBee neighbors,
> NewBee [24] is still for discovering homogeneous wireless neighbors.
> SERVOUS [10] is using ZigBee device to detect BLE neighbors while it
> requires modification at both ZigBee and BLE sides, incurring
> unaffordable costs at deploying that design into practice. Compared to
> SERVOUS, X-Disco is transparent to the ZigBee network, at the zero cost
> for installing X-Disco to the WiFi device without any modification to
> the existing ZigBee devices and ZigBee network."
Shape: `PREMISE → CONCESSION → CONTRAST → CONTRAST → CLAIM`. Move:
rebuttal-of-closest-prior-work.

Exemplar (SymBee · §IX · ¶2 "Cross-technology communication"):
> "Most of the CTC work take the packet-level approach where a packet
> serves as the unit of modulation … SymBee takes a unique approach of
> symbol-level CTC for throughput breakthrough. GapSense introduces a
> fine-grained physical layer design, where it requires a special hardware.
> The latest work of WEBee and BlueBee, with the physical layer approach
> and high-throughput, are most similar, but are infeasible for ZigBee to
> WiFi communication."
Shape: `DEFINITION → CLAIM → CONCESSION → CONCESSION`.

Exemplar (UMusic · §8 · ¶2):
> "Vision-based solutions suffer from occlusions, while acoustic-based
> solutions still face privacy leakage issues. For instance, VeCare
> proposes the first Child Presence Detection (CPD) system that only
> utilizes car speakers and microphones. … UMusic, on the other hand,
> utilizes signal processing techniques making the solution more adaptive
> to the environmental effects."

Exemplar (X-MIMO · §7 · ¶1):
> "Despite Surface MIMO achieves up to 1.3 Gbps throughput on commodity
> WiFi devices, the design is hard to be applied on low-power devices
> because (i), the low-power devices cannot support the high-speed signal
> processing in consideration of energy consumption, (ii) low-Power IoT
> does not support multiple antennas."

**Tone notes.** Sentence length is medium (mean ≈ 22 words). Heavy use of
"Despite" / "Although" / "However" (M9 single-sentence
concession+knockout). Wang's §RW closer is consistently a CLAIM sentence
that names the closest competitor and the exact axis of difference
("Compared to SERVOUS, X-Disco is transparent …" — X-Disco §VII¶2).

**Common opener.** A single-sentence SCOPE preview (SymBee §IX¶1 "This
work lies in the intersection of three areas: cross-technology
communication, interference mitigation, and heterogeneous collaboration.")
that lists the themes.

**Common closer.** No separate "X is the best because" paragraph. The
contrast is embedded in the last sentence of each theme paragraph.

### §10.B What ICSE expects

ICSE expects:
1. Coverage of three to four threads of related work, by theme rather than
   by paper (Wang agrees).
2. Each thread compares concrete competitors with the current paper on a
   specific axis (Wang agrees).
3. Less acceptable at ICSE than at SenSys: a one-paragraph §RW. ICSE
   reviewers want more thematic discrimination.
4. Citation-rich. Each theme paragraph should cite at least 5–10 papers.

Where Wang's style EXCEEDS ICSE convention:
- His M9 single-sentence concession+knockout ("Despite X is high-throughput,
  it is hard to apply on low-power devices") is a more compressed move than
  most ICSE related-work paragraphs achieve. Adopt it.

Where Wang's style is too thin for ICSE:
- 2-paragraph §RW (X-MIMO §7, UMusic §8) is too short for an ICSE
  full-paper RW. ICSE expects 3–4 paragraphs.

### §10.C Hybrid recommendation

1. Three theme paragraphs (per `PAPER_PLAN.md` §8): (a) architecture-to-code
   traceability — SWATTR, TransArc; (b) LLM-based trace-link recovery —
   LiSSA and contemporaneous LLM TLR work; (c) evaluation critique and
   metric proposals — CrystalBLEU, pass@k, CodeBLEU, Allamanis dedup,
   Papadakis mutation, CheckList. (Wang exemplar T9 shape per paragraph.)
2. Each paragraph follows T9: `PREMISE → CONCESSION → CONTRAST → CLAIM`
   (Wang exemplar: X-Disco §VII¶2, SymBee §IX¶2).
3. Every paragraph closes with the closest-competitor sentence and the
   exact axis of difference (Wang exemplar).
4. Use M9 "Despite … inherently" / "However … cannot" once per paragraph
   (Wang exemplar M9).
5. Do NOT repeat "to the best of our knowledge" here — that phrase is
   already spent in §I (Wang exemplar M7: one stamp per paper).
6. Length budget: 0.75 page (per `PAPER_PLAN.md` budget).

### §10.D Drop-in adaptations

**§10.D.1 — LLM-based trace-link recovery theme paragraph (T9 shape).**

> A recent line of work applies LLM classifiers directly to trace-link
> recovery. LiSSA~\cite{lissa} runs a single-pass LLM classification per
> (sentence, candidate) pair and was proposed to close the lexical gap
> that limits SWATTR-style systems. On our five-project benchmark, however,
> LiSSA remains weaker than TransArc on every metric we report
> (\todo{LiSSA file F1, decision F1, doc-to-model F1}). This is because a
> single LLM pass cannot resolve an alias once and reuse it across the
> document — every sentence renegotiates the same name. Compared to LiSSA,
> AALinker builds the alias table and the ambiguity classifier in a single
> pre-pass before any linking decision, and then specialises four agents to
> the four reference types — thereby raising the average decision F1 from
> $0.596$ to $0.823$ on the same benchmark, with no change to the source
> projects.

**§10.D.2 — Evaluation critique and metric proposals theme paragraph.**

> A parallel line of work critiques the evaluation metrics used in
> SE-tooling research and proposes corrections. CrystalBLEU~\cite{eghbali2022crystalbleu}
> shows that BLEU's distinguishing power on code is obscured by trivially
> shared n-grams and proposes a recalibration; Allamanis~\cite{allamanis2019dedup}
> shows that code duplication inflates ML-on-code accuracy by up to 100\%;
> pass@k~\cite{chen2021passk} replaces BLEU with functional-correctness
> sampling for code generation; CheckList~\cite{ribeiro2020checklist}
> replaces held-out accuracy with capability-by-capability behavioural
> testing for NLP. These works share a single move: identify a structural
> bias in the standard ruler, then propose a metric that corrects one
> source of bias at a time. Our six-metric suite is a direct application
> of this move to doc-to-code trace-link recovery, instantiated against
> three structural biases (enrollment inflation, block correlation, gold
> concentration) specific to the benchmark.

---

# §11 Conclusion

### §11.A How Wang writes this section

Wang's conclusion is a six-sentence paragraph in 4/4 papers. Pattern T10:
`CLAIM → METHODOLOGY → EVIDENCE` (with one optional CONSEQUENCE sentence).

Exemplar (SymBee · §X · ¶1):
> "We propose SymBee, a cross-technology communication framework that aims
> to bridge capacity and compatibility by customizing ZigBee packets.
> SymBee's encoding is as simple as putting specific byte patterns in the
> ZigBee packet payload, maximizing its applicability. This generates
> pattern at the PHY layer that can easily be detected at the WiFi idle
> listening. Theoretical analysis and extensive testbed experiments on
> TelosB nodes and USRP B210 reveal that SymBee is a reliable and
> efficient under various practical settings with the throughput up to
> 31.25Kbps, 145.4× of the state-of-the-art."

Exemplar (UMusic · §9 · ¶1):
> "This paper introduces UMusic, a system that uses commodity UWB devices
> to precisely detect car occupancy via lightweight signal processing
> techniques. UMusic converts CIR data into the frequency domain to obtain
> the channel frequency response, which is used to calculate the
> high-resolution PDP via the MUSIC algorithm. Through the comparison
> between the PDP of empty and occupied environments, UMusic is able to
> detect the occupancy status. We evaluate UMusic in a car with one or
> more passengers under various scenarios, including stationary and
> driving conditions. The experiments show that UMusic achieves an
> aggregated accuracy of 99.4%, highlighting its effectiveness in
> practical scenarios."

Exemplar (X-MIMO · §8 · ¶1):
> "This work presents X-MIMO, a cross-technology MU-MIMO on commodity
> devices. Utilizing cross-technology channel estimation and precoding,
> X-MIMO is the first work to offer cross-technology MU-MIMO on commodity
> devices. Our experiments demonstrate X-MIMO achieves the throughput of
> 495 Kbps, almost doubling the throughput of legacy ZigBee (250 Kbps),
> with 99% symbol reliability for two ZigBee receivers."

Exemplar (X-Disco · §VIII · ¶1):
> "In this paper, we present X-Disco to enable a WiFi device to detect
> the ambient ZigBee neighbors. We demonstrate the feasibility that a
> commodity WiFi device is capable of decoding the ZigBee packets just
> using the FFT magnitude extracted from WiFi Spectral Scan. … Evaluated
> in the office (LoS and NLoS), X-Disco discovers nine ZigBee neighbors
> within 70ms, demonstrating its efficacy in discovering the
> cross-technology neighbors."

**Tone notes.** Sentence length medium-short (mean ≈ 19 words). No hedge
words. Headline number(s) appear in the last two sentences in 4/4 papers.

**Common opener.** "This paper presents/introduces/proposes X" (4/4).

**Common closer.** A headline number paired with "achieves" /
"demonstrates" / "reaching" / "showing".

**Length.** 5–6 sentences in 4/4 papers.

### §11.B What ICSE expects

ICSE expects the conclusion to be short (≤ 0.5 page), to restate the
contributions, and to deliver the headline numbers one more time.
Optionally include a future-work pointer (Wang does not).

### §11.C Hybrid recommendation

1. One paragraph, 5–6 sentences, T10 shape (Wang exemplar T10: 4/4 papers).
2. Restate AALinker's identity (multi-agent linker with knowledge layer +
   four agents + two validators) in the first sentence (Wang exemplar M11
   no-qualifier declarative).
3. Re-quote the four headline numbers (file 0.931, decision 0.823,
   component 0.817, doc-to-model 0.951) with their absolute pairs (Wang
   exemplar M11).
4. State the metric story in one sentence: standard ruler +12.9pp;
   decision-level ruler +22.7pp.
5. Do NOT introduce new contributions, new arguments, or new future-work
   threads (Wang avoidance A4, A6).
6. Length budget: ≤ 6 sentences ≈ 0.25 page (per `PAPER_PLAN.md`).

### §11.D Drop-in adaptation

> This paper presents AALinker, a multi-agent doc-to-code trace-link recovery
> system that combines a per-project knowledge layer with four mention-type
> linker agents and two validation patterns over commodity LLMs. Together
> with AALinker we introduce a six-metric evaluation suite that corrects
> three structural biases of the established benchmark — enrollment
> inflation, block correlation, and gold concentration. Evaluated against
> TransArc, SWATTR, and LiSSA on the five-project benchmark (MediaStore,
> TeaStore, Teammates, BigBlueButton, JabRef), AALinker reaches an average
> file F1 of 0.931, decision F1 of 0.823, component F1 of 0.817, and
> doc-to-model F1 of 0.951 — improving over TransArc by +12.9, +22.7,
> +10.3, and +15.2 percentage points respectively. The standard ruler
> reports the gain at +12.9pp; the decision-level ruler matched to a
> single human annotator decision reports it at +22.7pp, showing that
> decision-level doc-to-code linking is reachable on commodity tooling and
> that the choice of ruler matters as much as the choice of linker.

---

# §§Authoring sequence

The right order is determined by data dependencies and by the difficulty
of each section. Recommended sequence:

**Stage 1 (now — no external blockers, polish only):**
1. **§7 Results** — RQ1 numbers are verified (file 0.803→0.931, decision
   0.596→0.823, component 0.714→0.817 aggregate, doc-to-model 0.799→0.951).
   Write RQ1 result paragraph (T7) and the Teammates T8 paragraph. RQ2
   results follow as soon as the metric definitions in §Metric Suite are
   finalised.
2. **§5 Metric Suite** §5.2 (the definitions stub) — port from
   `evaluation/writing/eval.tex` in plain English. Once the six metrics
   are defined, RQ2 result paragraphs can be written.
3. **§2 Introduction** — paragraphs 1, 2, 3 (the `\todo{}` blocks in
   `gen/intro.tex`). The metric-glimpse paragraph and the contributions
   bullets are already drafted.

**Stage 2 (depends on §5 and §7):**
4. **§3 Motivation / Background** — much of this is already drafted in
   `gen/metric.tex` §Background. Re-paragraph so that the
   structural-inequality story flows into the metric suite. The motivation
   should not duplicate the bias paragraphs in §5; one canonical place
   per fact.
5. **§4 Approach** — `writen-paper/sections/approach.tex` already exists
   at ~70%. Apply T6 walkthroughs in each subsection; add the "AALinker
   in a Nutshell" sub-subsection (§4.D.1 above). Fill `\todo{}` markers
   in the human draft.
6. **§6 Experiment Design** — already drafted; needs LiSSA decision and
   dataset-stat / enrollment tables (`PAPER_PLAN.md` §5 gaps).

**Stage 3 (depends on results being final):**
7. **§7 Results** RQ2, RQ3, RQ4 — written once §5 (RQ2), validator-ablation
   data (RQ3), and per-agent ablation data (RQ4) are in place. `PAPER_PLAN.md`
   §6 flags that RQ3 NoConsensus/NoCitation variants may need to be run.
8. **§8 Discussion** — depends on §7 being final. The "why AALinker wins
   more on the new metrics" paragraph (§8.D.1) and the Teammates
   regression paragraph re-cast as design implication.
9. **§9 Threats to Validity** — depends on §7. Pure ICSE template.
10. **§10 Related Work** — can be drafted in parallel with §8; depends only
    on §2 contribution stance being final.
11. **§11 Conclusion** — written last. Depends on all four headline
    numbers being final and on the metric-story stance being locked.
12. **§1 Abstract** — written last. Depends on §2 contributions list, §7
    headline numbers, and §11 stance.

**Dependency summary (most → least depended upon):**
- §7 Results depends on: §5 Metric Suite definitions, §6 Experiment Design
  baselines, §4 Approach for ablation variants.
- §1 Abstract depends on: everything (write last).
- §11 Conclusion depends on: §7 Results (headline numbers), §8 Discussion
  (metric stance).
- §8 Discussion depends on: §7 Results, §5 Metric Suite, §3 Motivation.
- §10 Related Work depends on: §2 Introduction stance.
- §2 Introduction depends on: §5 + §7 (already drafted with placeholders).
- §3 Motivation depends on: §5 Metric Suite (canonical bias paragraphs).
- §5 Metric Suite depends on: nothing (write second).
- §6 Experiment Design depends on: LiSSA data (`PAPER_PLAN.md` §5).
- §4 Approach depends on: human author for `\todo{}` content; otherwise
  independent.
- §9 Threats depends on: §7 Results, §5 Metric Suite, §4 Approach.

**Single-page execution order, optimised for unblocked-first:**
§7-RQ1 → §5-§5.2 → §2-¶1,2,3 → §7-RQ2 → §3 → §4-Nutshell → §7-RQ3,RQ4 →
§8 → §9 → §10 → §11 → §1 Abstract.
