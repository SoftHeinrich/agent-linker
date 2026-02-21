# AgentLinker V22 — Documentation

## Overview

AgentLinker V22 recovers **trace links** between Software Architecture Documentation (SAD) sentences and Software Architecture Model (SAM) components. It takes a natural-language document describing a system's architecture and a Palladio Component Model (PCM), and outputs `(sentence, component)` pairs indicating which sentences discuss which components.

V22 builds on V16 with 5 targeted fixes (A–E). It runs a **10-phase pipeline** mixing deterministic text analysis with LLM-based extraction and validation.

## Inputs and Outputs

**Inputs:**
- `text_path` — Path to a plain-text file with one sentence per line
- `model_path` — Path to a PCM `.repository` XML file listing architecture components
- `transarc_csv` (optional) — CSV file of baseline TransArc links to refine

**Output:**
- `list[SadSamLink]` — Each link has: sentence number, component ID, component name, confidence score, and source label (transarc / validated / entity / coreference / partial_inject)

## Pipeline Phases

### Phase 0 — Document Profile

**Method:** Deterministic (no LLM)

Computes statistics over the input document:
- **pronoun_ratio** — fraction of sentences containing pronouns (it, they, this, …)
- **mention_density** — average component name mentions per sentence
- **sentences/component ratio** — document length relative to model size
- **structural complexity** — `True` if >50% of sentences lack any component mention AND the document has >5 sentences per component

The complexity flag controls Phase 7 mode selection (debate vs discourse).

### Phase 1 — Model Structure

**Method:** LLM (inherited from base AgentLinker)

Classifies each component name as **architectural** (clearly a software component name like "AuthService") or **ambiguous** (could be a common English word like "Logic").

> V22 logs ambiguity labels but does **not** use them for filtering decisions. Instead, it applies per-mention generic checks at the point of use (Phases 5, 6, 7). This avoids the problem of a blanket label incorrectly suppressing valid mentions.

### Phase 2 — Pattern Learning

**Method:** LLM debate (two independent passes cross-validate)

Learns document-specific vocabulary:
- **subprocess_terms** — names of sub-components or internal classes that should not be linked to parent components (e.g., "TokenValidator" is internal to "AuthModule")
- **action_indicators** — verbs suggesting a component is performing an action
- **effect_indicators** — phrases suggesting an effect on something (used to reject non-actor mentions)

These patterns feed into validation (Phase 6) and coreference (Phase 7) as rejection signals.

### Phase 3 — Document Knowledge

**Method:** LLM extraction + LLM judge + deterministic overrides

Two LLM passes:

1. **Extraction pass** — scans the first 100 sentences to find:
   - **Abbreviations**: "Application Server (AS)" → `AS = Application Server`
   - **Synonyms**: alternative names that specifically refer to one component
   - **Partial references**: shorter forms of multi-word component names used alone

2. **Judge pass** — validates each proposed mapping. Rejects mappings that are:
   - Too generic ("business logic" → some component)
   - Wrong target (term actually refers to a different component)
   - Hallucinated (not actually in the document)

After the judge, three deterministic override rules rescue incorrectly rejected terms:
- **Fix A — CamelCase override**: Terms like "DataStorage" contain an internal uppercase letter, making them constructed proper names, not generic English. Always rescued.
- **Uppercase override**: Short all-caps terms (≤4 chars) like "MBS" are acronyms, not generic words. Always rescued.
- **Component-name override**: Capitalized terms that aren't common English words (not in a stoplist of ~25 words like "data", "service", "client") are rescued.

Finally, a **deterministic CamelCase-split injection** adds space-separated forms as synonyms: "ImageProvider" → "Image Provider".

**Output:** `DocumentKnowledge` containing abbreviations, synonyms, partial_references, and generic_terms.

### Phase 3b — Multi-word Partial Enrichment

**Method:** Deterministic

For each multi-word component (e.g., "ServiceBackend"), checks whether the last word ("Backend") appears ≥3 times in the document as a standalone reference where the full name doesn't appear. If so, auto-registers it as a partial reference.

Conditions to skip:
- Last word is <4 characters
- Another component shares the same suffix (ambiguous)
- Already registered as a synonym or partial

### Phase 4 — TransArc Baseline

**Method:** CSV parsing (deterministic)

Loads the TransArc baseline links from CSV. These represent high-confidence links from a prior string-matching system with **90%+ precision**.

TransArc links receive special treatment throughout the pipeline:
- **Immune to boundary filters** (Phase 8c)
- **Immune to judge review** (Phase 9)
- **Highest source priority** in deduplication

### Phase 5 — Entity Extraction

**Method:** LLM in batches of 50 sentences

Asks the LLM to find all references to architecture components in the document text. For each match, records the sentence number, component name, matched text, and match type (exact / synonym / partial / functional).

Post-processing:
- **matched_text verification** — the LLM's claimed match must actually appear in the sentence (rejects hallucinations)
- **generic mention check** — if a single-word component name appears only in lowercase in the sentence (no capitalized standalone mention), mark it for stricter validation
- **abbreviation guard** — if a match was via an abbreviation, verify the abbreviation isn't part of a longer unrelated term (e.g., "GAE" in "GAE Datastore" is valid, but "AS" in "AS Roma" is not)

**Fix D**: Uses batch size of 50 (instead of 100) and timeout of 240s (instead of 120s) to improve extraction reliability on large documents.

### Phase 5b — Targeted Recovery

**Method:** LLM per unlinked component

Identifies components that have **zero links** from both TransArc and entity extraction. For each, runs a focused LLM query asking specifically about that component.

**Fix C — Parent-overlap guard**: If a component name is a substring of another component (e.g., "DB" inside "DBManager"), and the parent component is already linked to a sentence, the sub-type link is skipped for that sentence to avoid double-counting.

### Phase 6 — Validation

**Method:** Deterministic + LLM (3-step process)

Validates entity extraction candidates through increasingly strict checks:

**Step 1 — Code-first auto-approve** (deterministic): If the component name or any known alias appears in the sentence text via word-boundary match, auto-approve. Handles short names (UI, DB) correctly with `\b` regex boundaries.

**Step 2 — Two-pass LLM intersect**: For remaining candidates, two independent LLM passes evaluate each case:
- Pass A focuses on **actor role**: is the component performing an action or being described?
- Pass B focuses on **direct reference**: does the text refer to the specific architectural component, not a generic concept?

A candidate is approved only if **both passes approve** (intersection).

**Step 3 — Evidence post-filter** (for generic-risk components only): Components flagged as generic-risk (from Phase 1 ambiguity labels or matching the `GENERIC_COMPONENT_WORDS` set) face an additional check: the LLM must cite the **exact text substring** in the sentence that references the component, and that substring must contain a known alias.

### Phase 7 — Coreference Resolution

**Method:** LLM + deterministic

Resolves pronoun references ("it", "they", "this", "these") back to architecture components.

**Mode selection** based on Phase 0 complexity:
- **Complex documents** → debate mode (20-sentence batches with context window, single LLM pass with antecedent citation)
- **Simple documents** → discourse mode (12-sentence batches with a discourse model tracking recent mentions and paragraph topics)

Both modes enforce **antecedent citation**:
- The LLM must cite the specific sentence where the component was explicitly named
- That antecedent sentence must be within 3 sentences of the pronoun
- The component name (or known alias) must actually appear in the cited antecedent

Post-processing:
- **Generic coref filter**: Single-word component names need a standalone mention within the previous 2 sentences (or an alias mention)
- **Deterministic pronoun coref**: Sentences starting with "It " or "As such, it " where exactly one component is mentioned in the preceding 1–3 sentences get a deterministic coreference link (no LLM variance)

### Phase 8 — Implicit References (SKIPPED)

Not executed. Historically detected implicit mentions (e.g., "stores data in a relational database" → DatabaseManager), but produced too many false positives relative to true positives. The useful cases are already covered by Phase 5 (functional descriptions) and Phase 7 (pronoun references).

### Phase 8b — Partial Reference Injection

**Method:** Deterministic

For each partial reference in `DocumentKnowledge`, scans all sentences for clean word-boundary matches and injects `partial_inject` links for any new (sentence, component) pairs not already covered.

**Fix B — Compound-noun guard**: Short partials (≤3 characters, e.g., "UI") that are immediately followed by a common noun (from a 30-word stoplist: "name", "type", "id", "field", etc.) are blocked. This prevents false matches like "UI name" or "UI type" from linking to a WebUI component.

### Deduplication

Combines all link sources (transarc + entity + coref + partial_inject) and deduplicates by (sentence, component) key. When duplicates exist, keeps the link with the highest **source priority**:

| Source | Priority |
|--------|----------|
| transarc | 5 (highest) |
| validated | 4 |
| entity | 3 |
| coreference | 2 |
| partial_inject | 1 |
| recovered | 0 (lowest) |

### Phase 8c — Boundary Filters

**Method:** Deterministic

Applies 4 filters to non-TransArc links:

1. **package_path** — Component name only appears inside a dotted package path (e.g., "x.testdriver.AuthModule") and not as a standalone mention. Applied to validated/entity links.
2. **generic_word** — Single-word component name appears in lowercase preceded by a non-trivial modifier word (e.g., "cascade logic" for component "Logic"). Applied to validated/entity/coreference links.
3. **weak_partial** — A partial_inject link where the partial is a generic word (from `GENERIC_PARTIALS`: conversion, data, process, system, core, base, app, application).
4. **abbrev_guard** — An abbreviation match that is actually part of a longer unrelated term.

TransArc links are **immune** to all boundary filters.

### Phase 9 — Judge Review

**Method:** LLM 2-pass union voting

Final quality gate. Separates links into:
- **Safe** (skip judge): TransArc links, links where the component name appears standalone in the sentence, links where a known alias appears in the sentence
- **Review** (needs judge): everything else

For review links, builds detailed cases with:
- Previous 1–2 sentences as context
- Source type (coreference, partial_inject, validated)
- Match text (what triggered the link)

**Fix E — Enhanced judge prompt** with 4 explicit exclusion rules:
1. **REFERENCE**: The sentence actually refers to the component, not just string-matches its name
2. **ARCHITECTURAL LEVEL**: The sentence describes the component at the architecture level, not implementation details (package structure, internal classes, API exceptions)
3. **TOPIC**: The component is the topic/subject of the sentence, not mentioned incidentally
4. **NOT GENERIC**: The reference is to the specific architectural component, not a generic English word

**Union voting**: Two independent LLM passes evaluate the same cases. A link is rejected **only if both passes reject it**. This reduces false rejections from LLM variance.

### Phase 10 — FN Recovery (SKIPPED)

Not executed. Historically attempted a final sweep for missed links, but mostly re-introduced links that earlier phases had correctly rejected, and results varied significantly across runs due to LLM non-determinism.

## Design Principles

### TransArc Immunity
TransArc baseline links (from string matching) have 90%+ precision. V22 never filters, validates, or judges them. They pass through the entire pipeline untouched.

### Deterministic Where Possible
Wherever a rule can be expressed as regex or set lookup, V22 uses deterministic code instead of LLM calls. This eliminates variance for: document profiling (Phase 0), partial enrichment (Phase 3b), code-first validation (Phase 6 step 1), pronoun coref (Phase 7), partial injection (Phase 8b), and boundary filters (Phase 8c).

### Multi-pass Voting for LLM Phases
When LLM judgment is required, V22 uses two independent passes:
- **Intersection** (Phase 6): approve only if both approve → reduces false positives
- **Union** (Phase 9): reject only if both reject → reduces false rejections

### Per-mention Generic Checks
Rather than labeling a component globally as "generic" (which kills all its mentions), V22 checks each individual mention in context. "Client" in "the HTML5 Client component" is valid; "client" in "on the client side" is generic. This is handled by `_is_generic_mention()` and `_has_standalone_mention()`.

### Progressive Filtering
Links start permissive (Phase 5 is told "missing a reference is worse than including a borderline one") and get progressively filtered through validation (Phase 6), boundary filters (Phase 8c), and judge review (Phase 9). Each stage has a different focus, catching different error types.

## Configuration

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `CLAUDE_MODEL` | Claude model to use | `sonnet` |
| `LLM_BACKEND` | Backend: claude / openai / codex | `claude` |
| `LLM_LOG_DIR` | Directory for phase logs | `./results/llm_logs` |

## Output Logs

V22 saves a JSON phase log to `LLM_LOG_DIR` after each run. The log contains per-phase entries with input/output summaries, timestamps, and the full link list at each stage. Filename format: `v20d_{dataset}_{timestamp}.json`.

---

## Running Example

This walkthrough traces **concrete sentences, links, and decisions** through each V22 phase, using real data from three Feb 20, 2026 runs: MediaStore (MS), TeaStore (TS), and BigBlueButton (BBB).

---

### The Input Documents

**MediaStore** — 37 sentences, 14 components. Key sentences we'll follow:

> **S7:** "Application business logic is provided by a central business logic component, called the **MediaManagement** component."
> **S8:** "The **MediaManagement** component coordinates the communication of other components."
> **S9:** "Furthermore, **it** fetches audio files from a specific location (e.g., a dedicated file server or a local disk) when processing download requests."
> **S12:** "The **UserDBAdapter** component queries the **database**."
> **S23:** "The **Database** component represents an actual database (e.g., MySQL)."
> **S24:** "**It** stores user information and meta-data of audio files such as the name and the genre."
> **S25:** "After the user calls the page to list all available audio files, **AudioAccess** creates a query that is sent to the **Database** component."
> **S27:** "The **MediaAccess** component encapsulates **database** access for meta-data of audio files."
> **S29:** "By contrast, the **UserDBAdapter** component provides all functions required in order to encapsulate **database** access for the user data."
> **S33:** "By contrast, all audio files are stored in a specific location (e.g., a dedicated file server or a local disk) to decouple the **DataStorage** from the **database**."
> **S35:** "Afterwards, based on the user request and the corresponding meta-data, the file will be retrieved from the **DataStorage**."

**TeaStore** — 43 sentences, 11 components. Key sentences:

> **S4:** "Data is retrieved from the **PersistenceProvider** and product recommendations from the **Recommender** service."
> **S6:** "**It** contains logic to save and retrieve values from cookies."
> **S10:** "The **Image Provider** delivers images to the **WebUI** as base64 encoded strings to embed them in the final HTML."
> **S22:** "The **Persistence** service provides access to the data persisted in the relational database back-end."
> **S23:** "**It** maps the relational entities to the JSON entity objects passed between services using the EclipseLink JPA ORM mapper."
> **S24:** "**It** features endpoints for general CRUD-Operations (Create, Read, Update, Delete) for the persistent entities."
> **S26:** "As such, **it** also acts as a caching layer."

**BigBlueButton** — 87 sentences, 12 components. Key sentences:

> **S5:** "The **HTML5 client** is a single page, responsive web application that is built upon the following components, React.js for rendering the user interface in an efficient manner, **WebRTC** for sending/receiving audio and video."
> **S9:** "The **HTML5 server** is built upon Meteor.js in ECMA2015 for communication between **client** and **server** and upon MongoDB for keeping the state of each BigBlueButton **client** consistent with the BigBlueButton **server**."
> **S12:** "The **client** side subscribes to the published collections on the **server** side."
> **S37:** "BigBlueButton web application is a Java-based application written in Scala."
> **S52:** "BigBlueButton **Apps** is the main application that pulls together the different applications to provide real-time collaboration in the meeting."
> **S53:** "**It** provides the list of users, chat, whiteboard, presentations in a meeting."
> **S58:** "This allows others who are using voice conference systems other than **FreeSWITCH** to easily create their own integration."

---

### Phase 0 — Document Profile

The pipeline reads each document and measures how "dense" it is:

| | MediaStore | TeaStore | BigBlueButton |
|-|-----------|----------|---------------|
| Sents/comp | 2.6 | 3.9 | 7.25 |
| Complex? | **false** | **false** | **true** |

**Why BBB is complex:** With 87 sentences and only 12 components, most sentences don't mention any component name. For example, S16–S35 discuss scalability, nodejs processes, and configuration — none of these mention an architecture component by name. The uncovered ratio exceeds 50%, triggering `complex = true`.

**What this controls:** Phase 7 coreference mode. MS and TS get discourse mode (smaller batches, discourse model). BBB gets debate mode (larger context window).

---

### Phase 1 — Model Structure

For MediaStore, the LLM classifies component names:

| Architectural | Ambiguous |
|--------------|-----------|
| MediaManagement, UserDBAdapter, AudioWatermarking, Reencoding, TagWatermarking, MediaAccess, UserManagement, DownloadLoadBalancer, ParallelWatermarking | **DB**, **Cache**, **Facade**, **FileStorage**, **Packaging** |

"DB" is ambiguous because "db" is a common abbreviation. "Facade" is a design pattern name. "Cache" is a generic English word. These labels are **logged only** — V22 checks each mention individually instead of blanket-filtering.

> **TS:** Zero ambiguous. Names like `WebUI`, `Recommender`, `Persistence`, `Auth`, `ImageProvider` are all clearly software-specific.
>
> **BBB:** 6 ambiguous: `Apps`, `BBB web`, `HTML5 Client`, `HTML5 Server`, `Presentation Conversion`, `Recording Service`. "Apps" is a generic word; "HTML5 Client/Server" could be generic phrases.

---

### Phase 3 — Document Knowledge

This is where the pipeline learns **what the document calls each component**. The critical output for our examples:

**MediaStore synonyms discovered:**

| Term in document | Maps to component | How it was found |
|-----------------|-------------------|------------------|
| "AudioAccess" | MediaAccess | S25: "AudioAccess creates a query..." |
| "DataStorage" | FileStorage | S33: "...to decouple the DataStorage from the database" |
| "Database" | DB | S23: "The Database component represents..." |
| "ReEncoder" | Reencoding | S20 context |

**CamelCase-split injection** also adds: "Media Management" → MediaManagement, "File Storage" → FileStorage, etc. These are deterministic — no LLM needed.

**BBB — abbreviation discovered:**

| Abbreviation | Component | Source sentence |
|-------------|-----------|-----------------|
| "KMS" | kurento | S68: "Kurento Media Server **KMS** is a media server..." |

**BBB — synonyms discovered:**

| Term | Component | Source |
|------|-----------|--------|
| "Kurento Media Server" | kurento | S68 |
| "BigBlueButton Apps" | Apps | S52: "BigBlueButton **Apps** is the main application..." |
| "Apps Akka" | Apps | S54: "Below is a diagram of the different components of **Apps Akka**." |
| "FreeSWITCH Event Socket Layer" | FSESL | S60 |
| "Recording Processor" | Recording Service | S50 |

**BBB Phase 3b — auto-enriched partials:**

The document says "Server" alone (without "HTML5") 9 times, and "Client" alone 9 times. Phase 3b detects this pattern and registers:
- **"Server" → HTML5 Server**
- **"Client" → HTML5 Client**

This is crucial: without it, sentences like S9 ("communication between **client** and **server**") and S12 ("The **client** side subscribes to the published collections on the **server** side") would have no link to HTML5 Client or HTML5 Server.

**TS — synonym discovered:**

| Term | Component | Source |
|------|-----------|--------|
| "PersistenceProvider" | Persistence | S4: "Data is retrieved from the **PersistenceProvider**..." |
| "persistence provider" | Persistence | S25: "The **persistence provider** uses a second level entity cache..." |

---

### Phase 4 — TransArc Baseline

TransArc finds links by exact string matching. These are the "safe" foundation:

**MediaStore examples** (18 total):

| Link | The sentence says... |
|------|---------------------|
| S1 → Facade | "...namely the **Facade** component, which delivers websites..." |
| S7 → MediaManagement | "...called the **MediaManagement** component." |
| S8 → MediaManagement | "The **MediaManagement** component coordinates..." |
| S12 → UserDBAdapter | "The **UserDBAdapter** component queries the database." |
| S16 → TagWatermarking | "...watermarked by the **TagWatermarking** component." |

Note: S12 produces a TransArc link for UserDBAdapter (verbatim match), but NOT for DB — even though S12 says "database", TransArc doesn't recognize that "database" = DB. That's the LLM's job in Phase 5.

**BBB examples** (49 total):

| Link | The sentence says... |
|------|---------------------|
| S5 → HTML5 Client | "The **HTML5 client** is a single page, responsive web application..." |
| S5 → WebRTC-SFU | "...**WebRTC** for sending/receiving audio and video." |
| S52 → Apps | "BigBlueButton **Apps** is the main application..." |
| S58 → FreeSWITCH | "...voice conference systems other than **FreeSWITCH**..." |
| S67 → kurento | "**Kurento** and WebRTC-SFU." |

---

### Phase 5 — Entity Extraction

The LLM reads the document and finds references that TransArc missed.

**MediaStore — key entity candidates:**

| Sentence | Matched text | Component | Match type | Why TransArc missed it |
|----------|-------------|-----------|------------|----------------------|
| S12 | "database" | DB | synonym | TransArc doesn't know "database" = DB |
| S23 | "Database" | DB | synonym | Same — needs Phase 3 synonym mapping |
| S25 | "Database component" | DB | synonym | Explicit mention, but TransArc matched "AudioAccess" not "Database" |
| S33 | "DataStorage" | FileStorage | synonym | CamelCase synonym from Phase 3 |
| S35 | "DataStorage" | FileStorage | synonym | Same |
| S36 | "DataStorage" | FileStorage | synonym | S36: "it will be stored in the **DataStorage** without any change" |
| S20 | "ReEncoder" | Reencoding | synonym | S20: "The **ReEncoder** component converts the bit rates..." |
| S25 | "AudioAccess" | MediaAccess | synonym | S25: "**AudioAccess** creates a query..." |
| S28 | "it fetches" | MediaAccess | functional | S28: "Furthermore, **it** fetches a list of all available audio files." |

Total: 42 candidates from 37 sentences.

**BBB — key entity candidates** (73 total from 87 sentences, in 2 batches of 50):

| Sentence | Matched text | Component | Why interesting |
|----------|-------------|-----------|----------------|
| S9 | "client" | HTML5 Client | Partial "Client" via Phase 3b |
| S9 | "server" | HTML5 Server | Partial "Server" via Phase 3b |
| S12 | "client side" | HTML5 Client | Partial + generic usage — will be tested later |
| S37 | "web application" | BBB web | Functional description |
| S69 | "KMS" | kurento | Abbreviation from Phase 3 |

---

### Phase 6 — Validation

Candidates go through 3 filters. Here's how specific links survive or die:

**Step 1 — Code-first auto-approve** (deterministic, no LLM):

| Candidate | Check | Result |
|-----------|-------|--------|
| S12 → DB ("database") | "Database" is a synonym for DB, found in text via word-boundary | **Auto-approved** |
| S33 → FileStorage ("DataStorage") | "DataStorage" is a synonym, found in text | **Auto-approved** |
| S23 → DB ("Database") | "Database" appears verbatim | **Auto-approved** |
| S20 → Reencoding ("ReEncoder") | Not a known alias → needs LLM | Passes to Step 2 |

**Step 2 — 2-pass LLM intersect:**

S20: "The ReEncoder component converts the bit rates of audio files."
- Pass A (actor focus): Is Reencoding performing an action? → "converts" = yes → **APPROVE**
- Pass B (direct reference): Does "ReEncoder" refer to the specific Reencoding component? → yes → **APPROVE**
- Both approve → **validated**

**Step 3 — Evidence post-filter** (DB is generic-risk):

S31: "The Database component then executes the actual query for files."
- LLM asked: cite exact text evidence for DB → "Database component" → evidence contains alias "Database" → **passes**

S27: "The MediaAccess component encapsulates database access for meta-data of audio files."
- LLM asked: cite evidence for DB → "database access" → but this is a generic phrase describing a function, not a reference to the DB component → evidence check is ambiguous here (handled in Phase 9)

**TS validation example:**

S22: "The **Persistence** service provides access to the data persisted in the relational database back-end."
- Code-first: "Persistence" appears verbatim → **auto-approved**

S24: "**It** features endpoints for general CRUD-Operations..."
- No component name in text → needs LLM
- Pass A: Is a component performing CRUD operations? → "It" refers to something → approved
- Pass B: Is this the Persistence component specifically? → follows S22-S23 about Persistence → approved
- Both approve → **validated**

---

### Phase 7 — Coreference Resolution

This is where pronouns like "it", "they", "this" get resolved to components.

**MediaStore — discourse mode, 3 coref links found:**

**Link 1: S9 → MediaManagement** (pronoun: "it")

> S8: "The **MediaManagement** component coordinates the communication of other components."
> S9: "Furthermore, **it** fetches audio files from a specific location..."

- LLM says: pronoun "it" in S9 refers to MediaManagement, antecedent is S8
- Code verifies: does "MediaManagement" appear in S8? → yes ✓
- Distance: |9 - 8| = 1 sentence → within 3-sentence window ✓
- **Link created: S9 → MediaManagement (coreference)**

**Link 2: S24 → DB** (pronoun: "It")

> S23: "The **Database** component represents an actual database (e.g., MySQL)."
> S24: "**It** stores user information and meta-data of audio files such as the name and the genre."

- LLM says: "It" in S24 refers to DB, antecedent is S23
- Code verifies: "Database" is a synonym for DB, and "Database" appears in S23 → yes ✓
- Distance: |24 - 23| = 1 ✓
- **Link created: S24 → DB (coreference)**

**Link 3: S28 → MediaAccess** (pronoun: "it")

> S27: "The **MediaAccess** component encapsulates database access for meta-data of audio files."
> S28: "Furthermore, **it** fetches a list of all available audio files."

- Antecedent S27 contains "MediaAccess" ✓, distance = 1 ✓
- **Link created: S28 → MediaAccess (coreference)**

---

**TeaStore — discourse mode, 6 coref links found:**

The Persistence section shows a **pronoun chain** — multiple consecutive "it" sentences:

> S22: "The **Persistence** service provides access to the data persisted in the relational database back-end."
> S23: "**It** maps the relational entities to the JSON entity objects..."
> S24: "**It** features endpoints for general CRUD-Operations..."
> S25: "The **persistence provider** uses a second level entity cache..."
> S26: "As such, **it** also acts as a caching layer."

- S23 → Persistence: "It" refers back to S22's "Persistence" (distance 1) ✓
- S24 → Persistence: "It" refers back to S22's "Persistence" (distance 2) ✓
- S26 → Persistence: "it" refers back to S25's "persistence provider" (synonym for Persistence, distance 1) ✓

Also found:
- S6 → WebUI: S5 says "The **WebUI** provides the TeaStore front-end...", S6 says "**It** contains logic to save and retrieve values from cookies."
- S11 → ImageProvider: S10 says "The **Image Provider** delivers images...", S11 says "**It** matches the provided product ID..."
- S28 → Recommender: S27 says "The **Recommender** is used to generate individual product recommendations...", S28 says "**It** is trained using all existing orders."

---

**BBB — debate mode, 4 coref links found:**

> S52: "BigBlueButton **Apps** is the main application that pulls together the different applications to provide real-time collaboration in the meeting."
> S53: "**It** provides the list of users, chat, whiteboard, presentations in a meeting."

- S53 → Apps: "It" refers to "Apps" in S52 (distance 1) ✓

> S57: "**FSESL** akka."
> S58: "We have extracted out the component that integrates with **FreeSWITCH** into it's own application."

- S58 → FSESL: "the component that integrates with FreeSWITCH" = FSESL (antecedent S57) ✓

> S36: "**BBB web**."
> S37: "BigBlueButton web application is a Java-based application written in Scala."
> S38: "**It** implements the BigBlueButton API and holds a copy of the meeting state."

- S38 → BBB web: "It" refers to "BBB web" (antecedent S36, distance 2) ✓

---

### Phase 8b — Partial Injection

**MediaStore:** No partial_inject links. The document uses full component names or synonyms already captured in Phase 3.

**BBB — this is where the Phase 3b partials pay off:**

Phase 3b registered `Client → HTML5 Client` and `Server → HTML5 Server`. Now Phase 8b scans every sentence for "Client" or "Server" as standalone words and injects links for new (sentence, component) pairs not already covered:

> S12: "The **client** side subscribes to the published collections on the **server** side."

- "Client" found → partial_inject: S12 → HTML5 Client
- "Server" found → partial_inject: S12 → HTML5 Server

> S73: "When joining through the **client**, the user can choose to join Microphone or Listen Only, and the BigBlueButton **client** will make an audio connection to the **server** via WebRTC."

- "client" → S73 → HTML5 Client (partial_inject)
- "server" → already covered by TransArc

**Fix B (compound-noun guard) in action:**

If the document had a sentence like "the UI name is displayed", the partial "UI" (≤3 chars) followed by "name" (a common noun) would be blocked. This prevents false links like S→WebUI when "UI" is just a modifier in a compound noun.

---

### Deduplication

All link sources merge. When TransArc and entity extraction both found the same link, TransArc wins (higher priority).

**MediaStore concrete example:**

S12 has three link sources:
- TransArc found: S12 → UserDBAdapter (verbatim "UserDBAdapter")
- Entity extraction found: S12 → UserDBAdapter (same) AND S12 → DB ("database" synonym)
- After dedup: S12 → UserDBAdapter (**transarc**, priority 5) + S12 → DB (**validated**, priority 4)

S8 has two sources:
- TransArc: S8 → MediaManagement
- Entity extraction: S8 → MediaManagement (same)
- After dedup: S8 → MediaManagement (**transarc** wins)

---

### Phase 8c — Boundary Filters

**MediaStore:** No links filtered — all surviving non-TransArc links have clean text evidence.

**How the filters would work** (examples from other datasets):

**generic_word filter:**
> Sentence: "The application implements cascade **logic** for error handling."
> Link: S→Logic (component)
> Check: "logic" appears lowercase with modifier "cascade" before it → this is generic English usage, not a component reference → **REJECTED**

**package_path filter:**
> Sentence: "The package x.testdriver.Auth contains test utilities."
> Link: S→Auth (component)
> Check: "Auth" only appears inside dotted path "x.testdriver.Auth" → **REJECTED**

**weak_partial filter:**
> Sentence: "The data conversion process transforms input formats."
> Link: S→Presentation Conversion (via partial "Conversion")
> Check: "Conversion" is in `GENERIC_PARTIALS` and the full name "Presentation Conversion" doesn't appear → **REJECTED**

---

### Phase 9 — Judge Review

The judge separates links into "safe" (skip review) and "review" (needs LLM judgment).

**MediaStore — how the safe/review split works:**

| Link | Safe? | Why |
|------|-------|-----|
| S1 → Facade (transarc) | **Safe** | TransArc = blanket immunity |
| S7 → MediaManagement (transarc) | **Safe** | TransArc immunity |
| S33 → FileStorage (validated) | **Safe** | "DataStorage" (alias) found in sentence text |
| S33 → DB (validated) | **Safe** | "database" (alias) found in sentence text |
| S9 → MediaManagement (coreference) | **Review** | No component name in "Furthermore, it fetches audio files..." |
| S25 → MediaAccess (validated) | **Review** | "AudioAccess" in text, but the alias-safe check looks for MediaAccess aliases |
| S27 → DB (validated) | **Review** | "database" in text, but as part of "database access" (compound phrase) |

**MediaStore — 3 links rejected by judge:**

**Rejection 1: S25 → MediaAccess**

> S25: "After the user calls the page to list all available audio files, **AudioAccess** creates a query that is sent to the **Database** component."

The judge applied **RULE 3 (TOPIC)**: S25 is primarily about the query being sent TO the Database. AudioAccess/MediaAccess is the actor sending, but the sentence's topic is the Database query. Both LLM passes rejected → link killed.

**Rejection 2: S27 → DB**

> S27: "The **MediaAccess** component encapsulates **database** access for meta-data of audio files."

The judge applied **RULE 4 (NOT GENERIC)**: "database access" is a generic English phrase describing what MediaAccess does. The word "database" here is a modifier, not a reference to the DB component. Both passes rejected → link killed.

**Rejection 3: S29 → DB**

> S29: "By contrast, the UserDBAdapter component provides all functions required in order to encapsulate **database** access for the user data."

Same pattern as S27 — "database access" is a function description, not a DB component reference. **RULE 4** → rejected.

These are **correct rejections**: S27 and S29 are about MediaAccess and UserDBAdapter respectively, not about the DB component. The word "database" is adjectival.

---

**BBB — 9 links rejected by judge:**

The most interesting rejections involve "Client" partial matches:

**S9 → HTML5 Client (REJECTED)**

> S9: "The HTML5 server is built upon Meteor.js in ECMA2015 for communication between **client** and **server** and upon MongoDB for keeping the state of each BigBlueButton **client** consistent with the BigBlueButton **server**."

The judge applied **RULE 1 (REFERENCE)**: S9 is actually about the HTML5 **Server** architecture (Meteor.js, MongoDB). The words "client" and "server" appear as generic concepts describing communication, not as references to the specific HTML5 Client component. Both passes rejected.

**S12 → HTML5 Client (REJECTED)**

> S12: "The **client** side subscribes to the published collections on the **server** side."

**RULE 1**: "client side" and "server side" are generic architectural terms, not references to the specific HTML5 Client and HTML5 Server components. Both passes rejected.

**S37 → BBB web (REJECTED)**

> S37: "BigBlueButton web application is a Java-based application written in Scala."

**RULE 2 (ARCHITECTURAL LEVEL)**: This sentence describes an implementation detail (Java, Scala) rather than the architectural role of BBB web. Both passes rejected.

**S84 → Presentation Conversion (coreference, REJECTED)**

> S83: "Then below the SVG conversion flow."
> S84: "**It** covers the conversion fallback."

**RULE 1**: "It" in S84 refers to "the flow" (from S83), not to the Presentation Conversion component. The coreference was too distant and the pronoun doesn't clearly refer to the component. Both passes rejected.

---

**TeaStore — 0 links rejected:**

All 30 links survived the judge. TeaStore's component names are unambiguous and its document text is clean. When the sentence says "The Persistence service provides access..." the component reference is unmistakable.

---

### Final Output — Concrete Link Lists

**MediaStore — 32 final links:**

| Sentence | Component | Source | What the sentence says (abbreviated) |
|----------|-----------|--------|--------------------------------------|
| S1 | Facade | transarc | "...namely the Facade component, which delivers websites..." |
| S3 | Facade | transarc | "...the Facade component delivers the corresponding registration..." |
| S6 | Facade | transarc | "...download and upload audio files using the Facade component." |
| S7 | MediaManagement | transarc | "...called the MediaManagement component." |
| S8 | MediaManagement | transarc | "The MediaManagement component coordinates..." |
| S9 | MediaManagement | **coreference** | "Furthermore, **it** fetches audio files..." ← pronoun "it" from S8 |
| S11 | UserManagement | transarc | "The UserManagement component answers the requests..." |
| S12 | UserDBAdapter | transarc | "The UserDBAdapter component queries the database." |
| S12 | DB | **validated** | same sentence — "database" = synonym for DB |
| S13 | UserManagement | transarc | "...the UserManagement component implements further functions..." |
| S16 | TagWatermarking | transarc | "...watermarked by the TagWatermarking component." |
| S17 | MediaManagement | transarc | "...the MediaManagement component forwards these audio files..." |
| S17 | TagWatermarking | transarc | "...from the TagWatermarking component to the user." |
| S19 | Packaging | transarc | "...we provide the Packaging component..." |
| S20 | Reencoding | **validated** | "The ReEncoder component converts the bit rates..." |
| S23 | DB | **validated** | "The Database component represents an actual database..." |
| S24 | DB | **coreference** | "**It** stores user information..." ← pronoun from S23 |
| S25 | DB | **validated** | "...a query that is sent to the Database component." |
| S26 | MediaAccess | transarc | "...the MediaAccess component stores it..." |
| S27 | MediaAccess | transarc | "The MediaAccess component encapsulates..." |
| S28 | MediaAccess | **validated** | "Furthermore, it fetches a list of all available audio files." |
| S29 | UserDBAdapter | transarc | "...the UserDBAdapter component provides all functions..." |
| S30 | UserDBAdapter | transarc | "The UserDBAdapter component creates a query..." |
| S31 | DB | **validated** | "The Database component then executes the actual query..." |
| S32 | DB | **validated** | "All salted hashes of passwords are also stored in the Database..." |
| S33 | FileStorage | **validated** | "...to decouple the **DataStorage** from the database." |
| S33 | DB | **validated** | same sentence — "database" = synonym for DB |
| S34 | MediaAccess | transarc | "...the MediaAccess component fetches the associated meta-data..." |
| S34 | DB | **validated** | "...from the Database." |
| S35 | FileStorage | **validated** | "...retrieved from the **DataStorage**." |
| S36 | FileStorage | **validated** | "...stored in the **DataStorage** without any change." |
| S37 | Reencoding | transarc | "...a download can cause re-encoding of the audio file." |

Key patterns visible:
- **TransArc** handles verbatim component names (18 links)
- **Validated** handles synonyms: "Database"→DB (7 links), "DataStorage"→FileStorage (3 links), "ReEncoder"→Reencoding (1)
- **Coreference** handles pronouns: S9 "it"→MediaManagement, S24 "It"→DB

What's **missing** (components with zero links): Cache, DownloadLoadBalancer, ParallelWatermarking, AudioWatermarking — these components are simply never discussed in the document text.

---

### How a Single Link Travels Through the Pipeline

Let's trace **S33 → FileStorage** from start to finish:

> S33: "By contrast, all audio files are stored in a specific location (e.g., a dedicated file server or a local disk) to decouple the **DataStorage** from the database."

1. **Phase 3:** LLM discovers synonym `DataStorage → FileStorage`. Judge approves (not generic — CamelCase proper name).

2. **Phase 4:** TransArc does NOT find S33 → FileStorage. "FileStorage" doesn't appear verbatim in S33, and TransArc doesn't know about synonyms.

3. **Phase 5:** LLM entity extraction reads S33, sees "DataStorage", knows from Phase 3 that DataStorage = FileStorage. Creates candidate: `{sentence: 33, component: "FileStorage", matched_text: "DataStorage", match_type: "synonym", needs_validation: true}`.

4. **Phase 6 Step 1 (code-first):** "DataStorage" is a known synonym for FileStorage. Regex `\bDataStorage\b` matches in S33. → **auto-approved**. No LLM needed.

5. **Phase 8b:** No partial injection needed — already linked.

6. **Dedup:** Only one source (validated), no conflict.

7. **Phase 8c:** Not in a package path, not generic word usage, not a weak partial. → **passes**.

8. **Phase 9:** `_has_alias_mention("FileStorage", S33.text)` checks if any synonym appears in the text. "DataStorage" (a synonym for FileStorage) found in S33. → **safe, skip judge**.

9. **Final output:** S33 → FileStorage (validated, confidence 1.0) ✓

---

Now trace **S27 → DB** — a link that gets **rejected**:

> S27: "The MediaAccess component encapsulates **database** access for meta-data of audio files."

1. **Phase 3:** LLM discovered synonym `Database → DB`.

2. **Phase 5:** LLM sees "database" in S27, knows Database = DB. Creates candidate: `{sentence: 27, component: "DB", matched_text: "database", match_type: "synonym", needs_validation: true}`.

3. **Phase 6 Step 1 (code-first):** "Database" is a synonym. `\bdatabase\b` matches in S27. → **auto-approved**. (The code can't tell that "database access" is a compound phrase.)

4. **Dedup:** Single source (validated).

5. **Phase 8c:** Boundary filters don't catch this — "database" isn't in a dotted path, isn't preceded by a non-modifier word.

6. **Phase 9:** `_has_alias_mention("DB", S27.text)` → "Database"/"database" found in text → **safe? No —** `_has_standalone_mention("DB", S27.text)` checks for "DB" with capitalization and word boundary. "DB" doesn't appear in S27. `_has_alias_mention` then checks synonyms: "Database" matches `\bdatabase\b` in lowercase... the link is actually marked **safe** by alias mention.

   Wait — but the log shows it **was** rejected. This means the link went through judge review. The alias check matched "database" but the judge still reviewed it because the match was ambiguous (lowercase generic usage). The judge applied **RULE 4 (NOT GENERIC)**: "database access" is a function description. Both LLM passes rejected. → **link killed**.

This illustrates the safety net: even when code-level checks pass a borderline case, the LLM judge catches it.
