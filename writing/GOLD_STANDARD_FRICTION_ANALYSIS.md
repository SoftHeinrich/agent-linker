# Gold Standard Friction Analysis: Transitive Consistency of the ARDoCo Benchmark

## Executive Summary

We analyzed all three gold standard layers (SAD-SAM, SAM-CODE, SAD-CODE) across 5 benchmark projects for transitive consistency. The key question: **if (sentence, component) is in SAD-SAM gold and (component, file) is in SAM-CODE gold, is (sentence, file) always in SAD-CODE gold?**

### Key Findings

| Finding | Impact | Severity |
|---------|--------|----------|
| **Gold standards are CONTRADICTORY for Teammates** | SAD-SAM says 51 sentences are NOT architecture-related; SAD-CODE says they ARE linked to component code | **Critical** |
| **Teammates has 128 non-transitive SAD-CODE entries** (56% of its gold) | Gold-constrained transitive product only covers 78.8% of SAD-CODE gold | Critical |
| **SWATTR's SAD-SAM "FPs" become SAD-CODE TPs** | Perverse incentive: over-linking at doc→model level HELPS at doc→code level | Critical |
| **BBB has 6 friction entries** (transitive says yes, gold says no) | FreeSWITCH S59 is correctly linked but gold omits it | Moderate |
| **3 projects are perfectly transitive** (MS, TS, JAB) | No friction, no non-transitive links — gold is consistent | Good |
| **3 ghost elements** across projects have SAD-SAM links but no code | GAE Datastore, kurento, unnamed MS interface — infrastructure with no source | Minor |
| **42 orphan elements** have SAM-CODE entries but no SAD-SAM links | Mostly interfaces — documented implicitly through their components | Expected |

---

## 1. Aggregate Results

### Raw-Level (Directory Entries as-is)

| Project | SAD-SAM | SAM-CODE | SAD-CODE | Transitive | Friction | Non-Trans | Ceiling_R |
|---------|:-------:|:--------:|:--------:|:----------:|:--------:|:---------:|:---------:|
| mediastore | 31 | 55 | 57 | 57 | 0 | 0 | 1.000 |
| teastore | 27 | 37 | 70 | 70 | 0 | 0 | 1.000 |
| **teammates** | **57** | **22** | **228** | **91** | **0** | **128** | **0.788** |
| bigbluebutton | 62 | 64 | 132 | 135 | **6** | **3** | 0.995 |
| jabref | 18 | 11 | 38 | 38 | 0 | 0 | 1.000 |
| **Total** | **195** | **189** | **525** | **391** | **6** | **131** | |

**Definitions:**
- **Transitive product** = {(s,f) : ∃ component c s.t. (s,c) ∈ SAD-SAM gold ∧ (c,f) ∈ SAM-CODE gold}
- **Friction** = transitive product entries NOT in SAD-CODE gold (A→B ∧ B→C but ¬A→C)
- **Non-transitive** = SAD-CODE gold entries NOT reachable transitively (A→C but ¬∃B: A→B ∧ B→C)
- **Ceiling_R** = |gold ∩ transitive| / |gold| = maximum recall achievable by any transitive approach

### Enrolled (File-Level, After Directory Expansion via ACM)

| Project | Gold (enrolled) | Trans (enrolled) | Match | Friction | Non-Trans | Ceiling_R |
|---------|:---------------:|:----------------:|:-----:|:--------:|:---------:|:---------:|
| mediastore | 59 | 59 | 59 | 0 | 0 | 1.000 |
| teastore | 707 | 707 | 707 | 0 | 0 | 1.000 |
| **teammates** | **8,097** | **6,380** | **6,380** | **0** | **1,717** | **0.788** |
| bigbluebutton | 1,529 | 1,615 | 1,521 | 94 | 8 | 0.995 |
| jabref | 8,268 | 8,268 | 8,268 | 0 | 0 | 1.000 |
| **Total** | **18,660** | **17,029** | **16,935** | **94** | **1,725** | |

---

## 2. Deep Dive: Teammates (Critical Inconsistency)

### The Problem

Teammates SAD-CODE gold has **228 raw entries** but only **91 are transitively reachable** (40%). The remaining **137 entries** (128 truly non-transitive + 9 explained by directory containment) cannot be recovered by ANY approach that works through SAD-SAM × SAM-CODE.

### Root Cause: 51 Sentences Annotated Directly to Code

The SAD-CODE gold standard contains 93 unique sentences. The SAD-SAM gold standard contains only 45 unique sentences. **51 sentences appear in SAD-CODE but NOT in SAD-SAM.** These sentences were annotated as having code-level trace links without corresponding architecture-model links.

| Metric | Value |
|--------|-------|
| SAD-SAM sentences | 45 |
| SAD-CODE sentences | 93 |
| In SAD-CODE but NOT SAD-SAM | **51** (55% of SAD-CODE sentences) |
| In SAD-SAM but NOT SAD-CODE | 3 (S137, S138, S141 — linked to GAE Datastore, no code) |
| Non-transitive raw entries | 128 |
| Ceiling_R | 0.788 (21.2% structural FN rate) |
| Enrolled: impossible FNs | 1,717 / 8,097 (21.2%) |

### What These 51 Sentences Look Like

These are sentences annotated directly to specific files/subdirectories within a component, but without a SAD-SAM link to the component itself:

- **S22** → `logic/`, `ui/webapi/` — discusses logic and UI but no SAD-SAM link
- **S32** → `OriginCheckFilter.java` — specific servlet filter class
- **S39-S40** → `WebPageServlet.java` — specific servlet class
- **S43-S45** → `WebApiServlet.java` — specific servlet class
- **S44** → `ActionFactory.java`, `Action.java` — specific web API classes
- **S75** → `Const.java` — specific utility class
- **S84-S86** → `logic/api/`, `logic/core/` — specific logic subpackages
- **S87** → `Logic.java`, `EmailGenerator.java`, `EmailSender.java`, `TaskQueuer.java` — specific classes
- **S90-S91** → `EmailGenerator.java`, `EmailSender.java` — specific classes
- **S100-S110** → Exception classes, entity attribute classes — specific data types
- **S125-S133** → Storage subpackages and entity classes — specific storage internals
- **S146-S160** → Common subpackages — utilities, exceptions, data transfer objects
- **S172** → Test directories across multiple components
- **S178-S179** → Storage search, UI webapi test directories
- **S187-S198** → E2E, LNP, client subdirectories

### Do These Code Files Have SAM-CODE Links?

**YES — 119 out of 121 entries (98.3%) are covered by SAM-CODE.** Every code file/directory in these 51 sentences' SAD-CODE gold entries maps to a known architecture component via SAM-CODE. Only 2 entries are not in SAM-CODE: `src/lnp/java/teammates/lnp/` (Load & Performance testing, no component mapping).

This means the B→C link EXISTS for virtually all of them. The ONLY missing piece is the A→B link (SAD-SAM). The gold standard is saying:
- "This sentence (A) does NOT discuss component B" (SAD-SAM)
- "Component B maps to these code files (C)" (SAM-CODE) ✓
- "This sentence (A) DOES link to these code files (C)" (SAD-CODE)

All three layers agree on B→C. The contradiction is purely about whether A→B should exist.

### Two Categories of Non-Transitive Entries

**Category 1: No SAD-SAM link (46 sentences, 121 code entries)**
Sentences were annotated to code but never annotated to the architecture model. These discuss implementation details (specific classes, subpackages) below the component abstraction level.

**Category 2: Cross-component link (5 sentences, 7 code entries)**
Sentence is linked to component A in SAD-SAM but has code links to component B in SAD-CODE:
- **S47**: SAD-SAM → Logic, but SAD-CODE includes `ui/webapi/BaseActionTest.java` (UI component)
- **S85**: SAD-SAM → UI, but SAD-CODE includes `logic/api/` (Logic component)
- **S101**: SAD-SAM → Storage, but SAD-CODE includes `common/exception/EntityAlreadyExistsException.java` (Common component)
- **S131**: SAD-SAM → Logic, but SAD-CODE includes `storage/api/` (Storage component)
- **S163**: SAD-SAM → Test Driver, but SAD-CODE includes `common/datatransfer/DataBundle.java` (Common component)

### Ghost Element: GAE Datastore

Element `_KGVMcKETEeu-mYqkDskRow` = **"GAE Datastore"** (Google App Engine Datastore) appears in SAD-SAM gold (5 links: S9, S122, S137, S138, S141) but has NO SAM-CODE mapping. It's an external infrastructure dependency with no source code in the repository.

- S137, S138, S141: Have SAD-SAM link to GAE Datastore but NO SAD-CODE entry (correct — no code to link to)
- S9, S122: Have SAD-SAM link to GAE Datastore AND other components, so they still have transitive SAD-CODE paths

### Implication for Evaluation

When evaluating transitive approaches on Teammates:
- **Raw SAD-CODE**: 228 gold entries, but only 91 are reachable → **perfect system gets R=0.399**
- **Enrolled SAD-CODE**: 8,097 gold file pairs, but only 6,380 reachable → **perfect system gets R=0.788**
- **Standard evaluation penalizes** transitive approaches for 1,717 file-level FNs that are structurally impossible

---

## 3. Deep Dive: BigBlueButton (Friction + Non-Transitive)

### The FreeSWITCH / FSESL Confusion

BBB has both **friction** (6 entries) and **non-transitive** links (3 entries), all centered on the FreeSWITCH/FSESL relationship.

**Context**: The architecture model has both:
- **Component: FreeSWITCH** — the FreeSWITCH telephony server component
- **Component: FSESL** — the FreeSWITCH Event Socket Library client

These are architecturally distinct but share code directories (e.g., `akka-bbb-fsesl/`, `bbb-fsesl-client/`).

### Friction: S59 (Transitive says YES, gold says NO)

S59 is linked to **Component: FreeSWITCH** in SAD-SAM. FreeSWITCH's SAM-CODE mapping includes:
- `akka-bbb-fsesl/src/main/java/org/bigbluebutton/freeswitch/`
- `bbb-fsesl-client/src/main/java/org/freeswitch/`
- `bbb-fsesl-client/src/test/java/org/freeswitch/`
- `build/packages-template/bbb-freeswitch-core/`
- `build/packages-template/bbb-freeswitch-sounds/`
- `freeswitch.placeholder.sh`

The transitive product says S59 should link to all 6, but the SAD-CODE gold standard OMITS S59 entirely. After enrollment, this becomes **94 file-level friction entries** (94 transitive predictions that the gold standard rejects).

**This is a gold standard annotation gap**: S59 is correctly linked to FreeSWITCH in SAD-SAM, FreeSWITCH correctly maps to these code files, but the annotator did not include S59 in SAD-CODE.

### Non-Transitive: S58 → FSESL Code

S58 is linked to **Component: FreeSWITCH** in SAD-SAM. But the SAD-CODE gold has S58 linking to FSESL code:
- `akka-bbb-fsesl/`
- `bbb-fsesl-client/`
- `build/packages-template/bbb-fsesl-akka/`

These are mapped to **Component: FSESL** in SAM-CODE, not FreeSWITCH. So S58 has a cross-component non-transitive link: SAD-SAM says FreeSWITCH, SAD-CODE says FSESL code. The annotator apparently considered FSESL code as relevant to a FreeSWITCH discussion.

### Ghost Element: kurento

Element `_oN4CMFkHEeyewPSmlgszyA` = **"kurento"** (WebRTC media server) appears in SAD-SAM gold (3 links: S67, S68, S69) but has NO SAM-CODE mapping. Like GAE Datastore, it's an external infrastructure dependency.

### Impact

- **Ceiling_R = 0.995** (near-perfect, only 8/1,529 enrolled links are non-transitive)
- **Friction = 94 enrolled file-level links** (6.1% of transitive product)
- A perfect transitive system would report ~94 FPs that the gold standard rejects but are actually correct

---

## 4. The Gold Standard Contradiction (Critical Finding)

### The Paradox

TransArC's Java test thresholds assert Teammates SAD-CODE recall ≥ 0.90. But our gold-constrained ceiling_R = 0.788 (using SAD-SAM gold × SAM-CODE gold). How can recall exceed the ceiling?

**Answer**: TransArC's SWATTR component links MORE sentences to components than the SAD-SAM gold standard annotates. The 51 "extra" sentences — which are FPs at the SAD-SAM level — become TPs at the SAD-CODE level because the SAD-CODE gold DOES include these sentences.

### The Contradiction in Detail

For each of the 51 sentences that appear in SAD-CODE but NOT in SAD-SAM gold:
- **SAD-SAM gold says**: "This sentence does NOT discuss any architecture component"
- **SAD-CODE gold says**: "This sentence DOES link to component code files"
- **Both can't be true**: if a sentence links to UI code files, it discusses the UI component

The contradiction is concentrated in two components:
- **Common**: 17 sentences, ~49 SAD-CODE entries (exception classes, DTOs, utility classes)
- **UI**: 16 sentences, ~31 SAD-CODE entries (servlets, web API actions, filters)
- **Storage**: 6 sentences, ~14 entries (entity classes, search API)
- **Logic**: 6 sentences, ~13 entries (logic API, core logic)
- **Client**: 4 sentences, ~6 entries (client scripts, connectors)
- **E2E**: 2 sentences, ~7 entries (test framework)

### The Perverse Incentive

This creates a perverse evaluation dynamic:

1. **SWATTR links S22→UI component** (correctly identifying the sentence as discussing UI)
2. At SAD-SAM level, this is an **FP** (S22 is not in SAD-SAM gold)
3. But at SAD-CODE level, S22→UI code files are **TPs** (S22 IS in SAD-CODE gold)
4. A system optimized for SAD-SAM precision would NOT link S22 → misses these SAD-CODE TPs
5. A system optimized for SAD-CODE recall would over-link SAD-SAM → appears to have poor SAD-SAM precision

**The two evaluation levels reward OPPOSITE behaviors.**

### Quantitative Impact

If a transitive system matches the SAD-SAM gold perfectly (adding no extra links):
- **SAD-SAM**: P=1.0, R=1.0, F1=1.0 (perfect)
- **SAD-CODE Ceiling_R**: 0.788 (cannot recover 1,717 of 8,097 enrolled gold entries)
- **SAD-CODE maximum F1**: ≤0.881 (assuming perfect precision with R=0.788)

If TransArC's SWATTR over-links (producing FPs at SAD-SAM level for these 51 sentences):
- **SAD-SAM**: P drops (extra links are FPs), R stays (gold links still found)
- **SAD-CODE**: R increases above 0.788 ceiling, potentially reaching ≥0.90

This explains why TransArC's Teammates SAD-CODE F1 can be reasonable despite the 21% structural gap.

### Implication for Our Pipeline

Our pipeline (which replaces SWATTR with an LLM-based SAD-SAM linker) faces the same dilemma:
- If we optimize for SAD-SAM precision → we'll miss the 51 sentences → SAD-CODE recall capped at ~0.788
- If we link those 51 sentences (correctly, from the SAD-CODE perspective) → our SAD-SAM precision drops
- **The "correct" behavior depends on which gold standard you evaluate against**

### Recommendation

Report SAD-CODE results **both ways**:
1. **Standard**: compare against full SAD-CODE gold (for comparability with TransArC)
2. **Gold-consistent**: compare against only the transitively reachable subset (gold ∩ transitive product from gold × gold)

Disclose the contradiction explicitly and recommend that future benchmark versions harmonize the three gold standard layers.

---

## 5. Perfectly Transitive Projects

### MediaStore (Ceiling_R = 1.000)
- 57 raw SAD-CODE entries, all transitively reachable
- 59 enrolled file-level pairs, all match
- **1 ghost element**: An unnamed interface (`_qxAiILg7EeSNPorBlo7x9g`) with 3 SAD-SAM links (S33, S35, S36) but no SAM-CODE mapping
- **10 orphan elements**: All interfaces (IDownload, IFacade, IMediaAccess, etc.) — have SAM-CODE but no SAD-SAM links
- **Enrollment**: Almost no inflation (55/55 = 1.0x). Only 3 directory entries, 2 of which expanded to 0 files (AudioWatermarking and Cache directories not in ACM — or empty)
- **Observation**: S35, S36 link to the ghost interface → they're in SAD-SAM but not SAD-CODE (no code to link to)

### TeaStore (Ceiling_R = 1.000)
- 70 raw SAD-CODE entries, all transitively reachable
- 707 enrolled file-level pairs, all match
- No ghost elements, 13 orphan elements (all interfaces)
- **Enrollment**: 37 raw SAM-CODE entries expand to 707 files (19.1x)
- Perfectly consistent gold standard

### JabRef (Ceiling_R = 1.000)
- 38 raw SAD-CODE entries, all transitively reachable
- 8,268 enrolled file-level pairs, all match
- No ghost elements, 1 orphan element (Component: globals)
- **Enrollment**: 11 raw SAM-CODE entries, 10 are directories (91%) → massive expansion
- **Warning**: JabRef ACM enrollment was 0 for all but 1 component in our initial run — the SAM-CODE gold uses paths like `src/main/java/org/jabref/logic/` but ACM paths include `buildSrc/` prefix for some. After fixing ACM parsing, 8,268 files enrolled correctly.

---

## 6. SAD-SAM Distribution Analysis

### Per-Element Link Counts

| Project | Elements | Total Links | Min | Max | Mean | Median | Gini |
|---------|:--------:|:-----------:|:---:|:---:|:----:|:------:|:----:|
| mediastore | 10 | 31 | 1 | 7 | 3.1 | 3.0 | 0.27 |
| teastore | 6 | 27 | 2 | 8 | 4.5 | 4.0 | 0.20 |
| teammates | 8 | 57 | 4 | 15 | 7.1 | 5.0 | 0.23 |
| bigbluebutton | 11 | 62 | 1 | 15 | 5.6 | 5.0 | 0.30 |
| jabref | 5 | 18 | 1 | 5 | 3.6 | 4.0 | 0.18 |
| **Total** | **40** | **195** | | | **4.9** | | |

### Multi-Component Sentences (Sentence Links to 2+ Elements)

| Project | Multi-comp | Total | % | Max components per sentence |
|---------|:----------:|:-----:|:-:|:---------------------------:|
| mediastore | 4 | 27 | 15% | 2 |
| teastore | 4 | 23 | 17% | 2 |
| teammates | 7 | 45 | 16% | 7 (S1 links to ALL 7 components) |
| bigbluebutton | 12 | 48 | 25% | 4 |
| jabref | 5 | 10 | 50% | 4 |

**Observation**: JabRef has 50% multi-component sentences — half its sentences discuss multiple components. BBB has 25%. This makes precision harder: a system that links a sentence to ONE correct component will also be tested against its OTHER component links.

### Element Coverage: SAD-SAM vs SAM-CODE

| Project | SAD-SAM elems | SAM-CODE elems | Ghosts | Orphans |
|---------|:-------------:|:--------------:|:------:|:-------:|
| mediastore | 10 | 19 | 1 | 10 |
| teastore | 6 | 19 | 0 | 13 |
| teammates | 8 | 14 | 1 | 7 |
| bigbluebutton | 11 | 22 | 1 | 12 |
| jabref | 5 | 6 | 0 | 1 |

**Pattern**: SAD-SAM covers far fewer elements than SAM-CODE. The gap is mostly **interfaces** — SAM-CODE maps both components and interfaces to code, but SAD-SAM only links sentences to components (interfaces are implicitly covered through their implementing components).

### Ghost Elements (In SAD-SAM but NOT in SAM-CODE)

| Project | Element | SAD-SAM Links | Reason |
|---------|---------|:-------------:|--------|
| mediastore | Unknown interface | 3 (S33, S35, S36) | No code in repository |
| teammates | **GAE Datastore** | 5 (S9, S122, S137-S141) | External infrastructure dependency |
| bigbluebutton | **kurento** | 3 (S67-S69) | External infrastructure dependency |

Ghost elements create **impossible FNs at SAD-CODE level** for sentences that ONLY link to the ghost element (S35, S36 in MS; S137, S138, S141 in TM; S68, S69 in BBB). However, sentences that link to BOTH a ghost element AND a regular component (e.g., TM S9 → GAE Datastore + Storage) still have transitive paths through the regular component.

---

## 7. Dark Matter Analysis

### SAD-SAM vs SAD-CODE Sentence Coverage

| Project | SAD-SAM sents | SAD-CODE sents | In CODE not SAM | In SAM not CODE |
|---------|:-------------:|:--------------:|:---------------:|:---------------:|
| mediastore | 27 | 25 | 0 | 2 (ghost elem) |
| teastore | 23 | 23 | 0 | 0 |
| **teammates** | **45** | **93** | **51** | **3** (ghost elem) |
| bigbluebutton | 48 | 45 | 0 | 3 (ghost + FreeSWITCH friction) |
| jabref | 10 | 10 | 0 | 0 |

**The Teammates anomaly**: 51 sentences (55% of SAD-CODE sentences) were annotated to code but NEVER annotated to the architecture model. This is the primary source of the 21.2% structural FN ceiling.

### Why This Happened

The Teammates SAD-CODE gold standard appears to have been annotated **independently** from SAD-SAM, using a different (finer-grained) annotation strategy:
1. SAD-SAM annotators linked sentences to high-level components (UI, Logic, Storage, etc.)
2. SAD-CODE annotators linked sentences to specific classes, subpackages, and test files
3. Many implementation-detail sentences (discussing specific servlets, entity classes, exception types) got SAD-CODE links but were not considered architecturally relevant for SAD-SAM

This is not an "error" — it reflects different annotation scopes. But it means **SAD-CODE evaluation of transitive approaches on Teammates is fundamentally unfair** without ceiling adjustment.

---

## 8. SAM-CODE Enrollment Impact

### Directory vs File Granularity

| Project | Raw entries | Dir entries | File entries | Dir % | Enrolled files | Inflation |
|---------|:----------:|:-----------:|:------------:|:-----:|:--------------:|:---------:|
| mediastore | 55 | 3 | 52 | 5% | 59* | 1.1x |
| teastore | 37 | 29 | 8 | 78% | 707 | 19.1x |
| teammates | 22 | 22 | 0 | **100%** | 1,616 | **73.5x** |
| bigbluebutton | 64 | 32 | 32 | 50% | 730 | 11.4x |
| jabref | 11 | 10 | 1 | 91% | 8,268† | **751.6x** |

*MediaStore has some empty directories (AudioWatermarking, Cache, UserManagement)
†JabRef's massive inflation comes from only 5 components mapping to 8,268 files

### Per-Component Enrollment Extremes

**Highest inflation**:
- JabRef all components → 751.6x (11 entries → 8,268 files)
- Teammates Interface: UI → 174.0x (2 dirs → 348 files)
- Teammates Interface: E2E → 123.0x (1 dir → 123 files)
- TeaStore Interface: WebUI → 80.0x (1 dir → 80 files)
- BBB Interface: Presentation Conversion → 70.0x (1 dir → 70 files)

**Lowest inflation**:
- MediaStore individual files → 1.0x
- BBB Redis DB → 1.0x (3 individual files)
- BBB Recording Service → 1.9x (7 entries → 13 files)

---

## 9. Implications for the ICSE Paper

### For Evaluation Framework (Section 5)

1. **Report Ceiling_R per project**: Any paper evaluating transitive approaches MUST disclose what fraction of the gold standard is structurally reachable. Teammates Ceiling_R = 0.788 means raw recall is capped at 78.8%.

2. **Use ceiling-adjusted metrics**:
   ```
   Adjusted_R = |recovered ∩ gold ∩ transitive_product| / |gold ∩ transitive_product|
   ```
   This gives the recall among structurally reachable links only.

3. **Report enrollment inflation factors**: The 73.5x–751.6x variation across projects makes file-level F1 incomparable.

### For Results (Section 6)

1. **Teammates results need asterisks**: Any reported recall/F1 on Teammates is artificially depressed by 21.2% structural ceiling.

2. **BBB FreeSWITCH friction**: 94 file-level "FPs" from S59 are actually correct predictions that the gold standard missed. Report these separately.

3. **Cross-project comparison is unfair** without normalization: JabRef's high F1 is partly because it's perfectly transitive with massive enrollment (low difficulty); Teammates' low F1 is partly because 21% of its gold is structurally unreachable.

### For Threats to Validity (Section 8)

1. **Gold standard contradiction**: The Teammates SAD-SAM and SAD-CODE gold standards are CONTRADICTORY for 51 sentences. SAD-SAM says they're not architecture-related; SAD-CODE says they are. This creates a perverse incentive where SAD-SAM FPs improve SAD-CODE recall.

2. **Ghost elements**: 3 elements across projects have SAD-SAM links but no code. Sentences exclusively linked to these elements contribute impossible FNs.

3. **Interface orphans**: 42 elements (mostly interfaces) exist in SAM-CODE but not in SAD-SAM. These never contribute to transitive evaluation but do expand the enrolled file count.

### For Future Work (Section 7)

1. **Gold standard harmonization**: SAD-SAM and SAD-CODE annotations MUST be checked for transitive consistency. The 51 Teammates sentences with SAD-CODE but no SAD-SAM links should either get SAD-SAM annotations (likely correct) or be removed from transitive evaluation.

2. **Per-project annotation metadata**: Report whether each gold entry is (a) transitively consistent, (b) cross-component, or (c) direct (no architectural mediation). This enables fair comparison between transitive and direct approaches.

3. **BBB FreeSWITCH audit**: The 6 friction entries (94 enrolled FPs) and 3 non-transitive entries for FreeSWITCH/FSESL suggest annotator confusion between these related components. Recommend clarification.

---

## 10. Benchmark Version Verification

**Our version**: ardoco monorepo commit `0bce0f848` (2026-01-22), benchmark subtree from `ardoco/Benchmark` commit `2449c851`.

**Latest version**: `ardoco/Benchmark` commit `e60ddb26` (2025-08-05, "Cleanup repository").

**Changes between our version and latest**:
- **Teammates**: Gold standards (SAD-SAM, SAM-CODE, SAD-CODE) **UNCHANGED**. Only cleanup (removed old 2015 versions, renamed images).
- **BBB**: Text was edited (lines added/removed), causing sentence renumbering across both SAD-SAM and SAD-CODE gold standards. **1 structural change**: FreeSWITCH link to old S56 was REMOVED (-1 link net). All other changes are renumbering.
- **MediaStore, TeaStore, JabRef**: No gold standard changes.
- **ardoco/ardoco `origin/main`**: No benchmark data changes vs our local copy. Latest remote commits (post 2026-01-22) are all dependency bumps.

**Conclusion**: Our analysis uses the latest gold standard data. The Teammates contradiction is NOT an artifact of an outdated benchmark version — it's present in the current canonical gold standard.

---

## 11. Raw Data Files

- **Analysis script**: `writing/analyze_gold_friction.py`
- **Full output**: `writing/gold_friction_results.txt` (985 lines)
- **Gold standard source**: `ardoco/core/tests-base/src/main/resources/benchmark/*/goldstandards/`
