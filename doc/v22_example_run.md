# V22 Execution Trace — MediaStore

Full step-by-step trace of AgentLinkerV22 running on the MediaStore dataset.
Based on actual run log `v20d_mediastore_20260220_112329.json` (Feb 20, 2026).

**What is verified vs inferred:**
- Numbers marked **(LOG)** are directly from the run log.
- Prompt text is copied verbatim from V22 source code, with MediaStore data substituted.
- Code-level checks are traced through the actual source code logic.
- LLM responses are **not recorded** in the log (only phase summaries). Where we reconstruct
  likely responses, this is marked as **(INFERRED)**.

---

## Loading Inputs

```
>>> linker = AgentLinkerV22()
AgentLinkerV22: V20 + Phase 5 smaller batches (50) + longer timeout (240s)
  Backend: claude, Model: sonnet

>>> links = linker.link(
...     text_path="mediastore.txt",
...     model_path="mediastore.repository",
...     transarc_csv="sadSamTlr_mediastore.csv"
... )
```

**PCM model parsed → 14 components:**

```
_BasicComponent_UserDBAdapter        → "UserDBAdapter"
_BasicComponent_AudioWatermarking    → "AudioWatermarking"
_BasicComponent_Reencoding           → "Reencoding"
_BasicComponent_MediaManagement      → "MediaManagement"
_BasicComponent_Facade               → "Facade"
_BasicComponent_MediaAccess          → "MediaAccess"
_BasicComponent_Packaging            → "Packaging"
_BasicComponent_DB                   → "DB"
_BasicComponent_FileStorage          → "FileStorage"
_BasicComponent_TagWatermarking      → "TagWatermarking"
_BasicComponent_Cache                → "Cache"
_BasicComponent_UserManagement       → "UserManagement"
_BasicComponent_DownloadLoadBalancer → "DownloadLoadBalancer"
_BasicComponent_ParallelWatermarking → "ParallelWatermarking"
```

**Document loaded → 37 sentences:**

```
S1:  One of the main components of Media Store is a server-side web front end, namely
     the Facade component, which delivers websites to the users and provides session management.
S2:  To meet the user authentication requirement, registration and log-in have to be offered.
S3:  To this end, the Facade component delivers the corresponding registration and log-in
     pages to users.
S4:  After the user has logged into the system, he or she will be forwarded to a site to
     list the audio files.
S5:  The main functionality, however, is provided by other components.
S6:  In addition, users can browse, download, and upload audio files using the Facade component.
S7:  Application business logic is provided by a central business logic component, called the
     MediaManagement component.
S8:  The MediaManagement component coordinates the communication of other components.
S9:  Furthermore, it fetches audio files from a specific location (e.g., a dedicated file server
     or a local disk) when processing download requests.
S10: As described above, to communicate with the system, users' registration and authentication
     are needed.
S11: The UserManagement component answers the requests for registration and authentication.
S12: The UserDBAdapter component queries the database.
S13: When a user logs into the system, Media Store does not store the password in plain text,
     but rather the UserManagement component implements further functions to hash and salt
     the passwords.
S14: To reduce the risk of copyright infringements, all downloaded files are watermarked.
S15: To this end, the requested files are first reencoded.
S16: The re-encoded files are then digitally and individually watermarked by the
     TagWatermarking component.
S17: Afterward, the MediaManagement component forwards these audio files from the
     TagWatermarking component to the user.
S18: Hence, each downloaded audio file is uniquely connected to a user ID can can be associated
     with the user if the audio file should ever appear in the public Internet.
S19: To allow users to download several files at a time, we provide the Packaging component,
     which archives multiple audio files into a single compressed file.
S20: The ReEncoder component converts the bit rates of audio files.
S21: This can result in reduction of file sizes.
S22: The persistence tier contains components that are concerned with the storage of audio files
     and their meta-data.
S23: The Database component represents an actual database (e.g., MySQL).
S24: It stores user information and meta-data of audio files such as the name and the genre.
S25: After the user calls the page to list all available audio files, AudioAccess creates a
     query that is sent to the Database component.
S26: When a user uploads an audio file, the MediaAccess component stores it at the predefined
     location.
S27: The MediaAccess component encapsulates database access for meta-data of audio files.
S28: Furthermore, it fetches a list of all available audio files.
S29: By contrast, the UserDBAdapter component provides all functions required in order to
     encapsulate database access for the user data.
S30: The UserDBAdapter component creates a query based on the user's request.
S31: The Database component then executes the actual query for files.
S32: All salted hashes of passwords are also stored in the Database component.
S33: By contrast, all audio files are stored in a specific location (e.g., a dedicated file
     server or a local disk) to decouple the DataStorage from the database.
S34: When a user requests files to download, the MediaAccess component fetches the associated
     meta-data from the Database.
S35: Afterwards, based on the user request and the corresponding meta-data, the file will be
     retrieved from the DataStorage.
S36: When a user uploads a file, it will be stored in the DataStorage without any change.
S37: However, a download can cause re-encoding of the audio file.
```

---

## Phase 0 — Document Profile (deterministic, no LLM)

V22 overrides the base class `_learn_document_profile` with a purely deterministic version.

### Pronoun ratio

Scan every sentence for pattern `\b(it|they|this|these|that|those|its|their)\b` (case-insensitive):

```
Check logic:
  FOR each sentence:
    IF regex \b(it|they|this|these|that|those|its|their)\b matches (case-insensitive) → count it

Matches:
  S3:  "To this end, the Facade component..."  → "this"  → match
  S9:  "Furthermore, it fetches..."            → "it"    → match
  S15: "To this end, the requested files..."   → "this"  → match
  S17: "...component forwards these audio..."  → "these" → match
  S21: "This can result..."                    → "this"  → match
  S22: "...components that are concerned...their meta-data." → "that","their" → match
  S24: "It stores user information..."         → "it"    → match
  S25: "...a query that is sent to..."         → "that"  → match
  S26: "...component stores it at the..."      → "it"    → match
  S28: "Furthermore, it fetches..."            → "it"    → match
  S36: "...it will be stored..."               → "it"    → match

  Non-matches (common traps):
  S4:  "...he or she..." — "he"/"she" not in pattern   → skip
  S18: no pronoun in pattern appears standalone         → skip
  S20: "bit rates" — \bit\b doesn't match inside "bit" → skip

  pronoun_ratio = 11/37 = 29.7%
```

### Mention density

```
Check logic:
  FOR each sentence, FOR each component:
    IF component.name.lower() IN sentence.text.lower() → count

Example checks:
  S1:  "facade" in "...namely the Facade component..." → yes (Facade)
  S7:  "mediamanagement" in "...the MediaManagement component." → yes
  S12: "userdbadapter" in "The UserDBAdapter component queries..." → yes
  S12: "db" in same sentence → yes (but "db" is in "UserDBAdapter" too!)
  ...
  Note: "db" is a substring of "userdbadapter", so S12/S29/S30 each count
  as 2 mentions (UserDBAdapter + DB). S17 also counts 2 (MediaManagement + TagWatermarking).
  Total mentions: 20 across 37 sentences
  mention_density = 20/37 = 0.54
```

### Structural complexity

```
Check logic:
  uncovered_ratio = (sentences with NO component name substring match) / total
  spc = sentences / components
  complex = (uncovered_ratio > 0.5) AND (spc > 5)

Values:
  21 of 37 sentences have no component mention → uncovered_ratio = 56.8%
  spc = 37/14 = 2.6
  complex = (56.8% > 50%) AND (2.6 > 5) = true AND false = false
```

**(LOG)** `complex: false` — Phase 7 will use **discourse mode** (12-sentence batches + context window).

---

## Phase 1 — Model Structure (1 LLM call)

V22 inherits `_analyze_model` from the base AgentLinker class. This calls LLM once.

### Full LLM prompt

```
Classify these component names.

NAMES: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade,
MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement,
DownloadLoadBalancer, ParallelWatermarking

Return JSON:
{
  "architectural": ["names clearly representing architecture components"],
  "ambiguous": ["names that could be common English words"]
}
JSON only:
```

### Response **(LOG)**

```json
{
  "architectural": ["AudioWatermarking", "DownloadLoadBalancer", "MediaAccess",
    "MediaManagement", "ParallelWatermarking", "Reencoding", "TagWatermarking",
    "UserDBAdapter", "UserManagement"],
  "ambiguous": ["Cache", "DB", "Facade", "FileStorage", "Packaging"]
}
```

### Code post-processing

```python
# Intersect with actual component names (filter hallucinated names)
knowledge.architectural_names = set(response["architectural"]) & set(names)  # 9
knowledge.ambiguous_names = set(response["ambiguous"]) & set(names)          # 5
```

Why each is ambiguous:
- **DB** — "db" is a common abbreviation, could be generic
- **Cache** — "cache" is a generic CS term
- **Facade** — "facade" is a design pattern name
- **FileStorage** — "file storage" describes a generic concept
- **Packaging** — "packaging" is a common English word

**These labels are logged but NOT used for blanket filtering.** V22 uses per-mention checks instead (see Phase 6).

---

## Phase 2 — Pattern Learning (3 LLM calls)

V22 inherits `_learn_patterns_with_debate` from the base class. Three sequential LLM calls.

### LLM call 1: Find subprocess terms

```
Find terms that refer to INTERNAL PARTS of components (subprocesses).

COMPONENTS: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade,
MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement,
DownloadLoadBalancer, ParallelWatermarking

DOCUMENT:
S1: One of the main components of Media Store is a server-side web front end, namely
    the Facade component, which delivers websites to the users and provides session management.
S2: To meet the user authentication requirement, registration and log-in have to be offered.
[... S3-S36 ...]
S37: However, a download can cause re-encoding of the audio file.

Return JSON:
{
  "subprocess_terms": ["term1", "term2"],
  "reasoning": {"term": "why"}
}
JSON only:
```

Response: **(not recorded in log)** — the LLM proposed some terms with reasoning.

### LLM call 2: Debate — validate the proposals

```
DEBATE: Validate these subprocess terms.

COMPONENTS: [same 14 components]

PROPOSED:
- DataStorage: [LLM's reasoning from call 1]
- hash and salt: [LLM's reasoning]
- session management: [LLM's reasoning]
[... up to 15 terms ...]

SAMPLE:
S1: One of the main components...
[... up to 30 sentences ...]

Return JSON:
{
  "validated": ["terms that ARE subprocesses"],
  "rejected": ["terms that might be valid component references"]
}
JSON only:
```

### LLM call 3: Find linguistic patterns

```
Find linguistic patterns.

COMPONENTS: [same 14 components]

DOCUMENT:
[up to 40 sentences]

Return JSON:
{
  "action_indicators": ["verbs when component DOES something"],
  "effect_indicators": ["verbs for RESULTS"]
}
JSON only:
```

### Result **(LOG)**

```
subprocess_terms: {"DataStorage", "hash and salt", "session management"}
```

What these mean downstream:
- If Phase 7 later resolves a pronoun to a sentence containing a subprocess term, that coref link is rejected (the sentence describes an internal process, not a component).

---

## Phase 3 — Document Knowledge (2 LLM calls)

V22 overrides with `_learn_document_knowledge_enriched`. Two LLM calls: extraction + judge.

### LLM call 1: Extraction

```
Find all alternative names used for these components in the document.

COMPONENTS: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade,
MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement,
DownloadLoadBalancer, ParallelWatermarking

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Example: "Application Server (AS)" → AS = Application Server
   Example: "the Message Broker Service (MBS)" → MBS = Message Broker Service

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   GOOD: "auth provider" → AuthService (a specific role name for the component)
   GOOD: "message routing engine" → MessageBroker (describes THAT specific component)
   BAD: "stores data" → DataStore (a generic description, not a name)
   BAD: "the business logic" → Processor (generic English phrase, not a specific name)

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   GOOD: "Backend" alone → "ServiceBackend" (unique last-word reference)
   GOOD: "main interface" → "MainInterface" (case variant of full name)
   BAD: "server" alone → "AppServer" (too ambiguous — could be any server)
   BAD: "the client" → "WebClient" (generic usage with article)

DOCUMENT:
S1: One of the main components of Media Store is a server-side web front end, namely
    the Facade component, which delivers websites to the users and provides session management.
[... S2-S36 ...]
S37: However, a download can cause re-encoding of the audio file.

Return JSON:
{
  "abbreviations": {"short_form": "FullComponent"},
  "synonyms": {"specific_alternative_name": "FullComponent"},
  "partial_references": {"partial_name": "FullComponent"}
}
JSON only:
```

Response **(INFERRED from log output — log records final synonyms after all processing)**:

The LLM found 4 synonyms:
- **AudioAccess → MediaAccess**: S25 says "AudioAccess creates a query"
- **ReEncoder → Reencoding**: S20 says "The ReEncoder component converts..."
- **DataStorage → FileStorage**: S33 says "...to decouple the DataStorage from the database"
- **Database → DB**: S23 says "The Database component represents..."

No abbreviations or partial references found.

### LLM call 2: Judge validation

```
JUDGE: Review these component name mappings for correctness.

COMPONENTS: [same 14 components]

PROPOSED MAPPINGS:
'AudioAccess' -> MediaAccess (synonym)
'ReEncoder' -> Reencoding (synonym)
'DataStorage' -> FileStorage (synonym)
'Database' -> DB (synonym)

REJECT mappings that are:
- TOO GENERIC: common English phrases like "the server", "business logic", "data storage"
- WRONG TARGET: the term actually refers to a different component or to the system overall
- NOT IN DOCUMENT: the mapping was hallucinated and doesn't appear in the text

APPROVE mappings that are:
- SPECIFIC: the term is a recognizable name/alias for exactly that component
- DOCUMENT-SUPPORTED: the mapping can be verified from the document text
- UNAMBIGUOUS: the term clearly refers to one component, not multiple

Return JSON:
{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}
JSON only:
```

### Code post-processing: Override checks

After the judge returns, the code applies three override rules for terms the judge rejected as "generic":

```
Fix A — CamelCase override:
  FOR each term in generic_rejected:
    IF regex [a-z][A-Z] matches in term (e.g. "DataStorage" has 'a' then 'S')
    → rescue: remove from generic_rejected, add to approved
  (MediaStore: no terms were rejected, so no overrides needed)

Fix B — Uppercase override:
  FOR each term in generic_rejected:
    IF term.isupper() AND len(term) <= 4 (e.g. "DB", "API")
    → rescue: short all-uppercase terms are acronyms, not generic English
  (MediaStore: no terms to rescue)

Fix C — Component-name override:
  FOR each term in generic_rejected:
    IF term starts with uppercase AND term.lower() NOT IN common_english_set
       (common_english_set = {"data", "service", "server", "client", "model", "logic",
        "storage", "common", "action", "process", "system", "core", "base", "app",
        "application", "cache", "store", "manager", "handler", "controller", "provider",
        "factory", "adapter"})
    → rescue: capitalized non-common-English term is likely a proper name
  (MediaStore: no terms to rescue)
```

### Deterministic CamelCase-split injection

After LLM-discovered synonyms, the code adds space-separated forms of CamelCase names:

```python
# For each component name, split CamelCase into words
# regex: insert space before uppercase letter preceded by lowercase
# "MediaManagement" → "Media Management"
for comp in component_names:
    split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)    # "Media Management"
    split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)  # handle "DBAdapter" → "DB Adapter"
    if split != comp:
        synonyms[split] = comp
```

Results:
```
    CamelCase syn: User DB Adapter -> UserDBAdapter
    CamelCase syn: Audio Watermarking -> AudioWatermarking
    CamelCase syn: Media Management -> MediaManagement
    CamelCase syn: Media Access -> MediaAccess
    CamelCase syn: File Storage -> FileStorage
    CamelCase syn: Tag Watermarking -> TagWatermarking
    CamelCase syn: User Management -> UserManagement
    CamelCase syn: Download Load Balancer -> DownloadLoadBalancer
    CamelCase syn: Parallel Watermarking -> ParallelWatermarking
```

### Final DocumentKnowledge **(LOG)**

```
  Abbreviations: 0
  Synonyms: 13
    AudioAccess → MediaAccess          (LLM-discovered)
    ReEncoder → Reencoding             (LLM-discovered)
    DataStorage → FileStorage          (LLM-discovered)
    Database → DB                      (LLM-discovered)
    User DB Adapter → UserDBAdapter    (CamelCase-split)
    Audio Watermarking → AudioWatermarking
    Media Management → MediaManagement
    Media Access → MediaAccess
    File Storage → FileStorage
    Tag Watermarking → TagWatermarking
    User Management → UserManagement
    Download Load Balancer → DownloadLoadBalancer
    Parallel Watermarking → ParallelWatermarking
  Partial references: 0
  Generic rejected: 0
```

---

## Phase 3b — Multi-word Partial Enrichment (deterministic, no LLM)

Check if the last word of any multi-word component appears ≥3 times standalone in the document.

```
Check logic:
  FOR each component:
    parts = component.name.split()     ← split on SPACES
    IF len(parts) < 2 → skip           ← only multi-word names

All 14 MediaStore components are single CamelCase tokens (no spaces):
  "UserDBAdapter".split() → ["UserDBAdapter"]  → len=1 → skip
  "AudioWatermarking".split() → ["AudioWatermarking"] → len=1 → skip
  ... (same for all 14)

Result: no multi-word names → Phase 3b adds nothing.
```

> **Note:** In BigBlueButton, "HTML5 Server" and "HTML5 Client" ARE multi-word. Phase 3b
> found "Server" appears 9 times alone → registered `Server → HTML5 Server`.
> Same for "Client" → `Client → HTML5 Client`.

---

## Phase 4 — TransArc Baseline (no LLM)

Load CSV file and map component IDs to names:

```
CSV: sadSamTlr_mediastore.csv
  Columns: sentenceNo, modelElementId
```

Each row maps a sentence number to a component ID. After ID→name resolution:

```
  S1  → Facade            "...namely the Facade component..."
  S3  → Facade            "...the Facade component delivers..."
  S6  → Facade            "...using the Facade component."
  S7  → MediaManagement   "...called the MediaManagement component."
  S8  → MediaManagement   "The MediaManagement component coordinates..."
  S11 → UserManagement    "The UserManagement component answers..."
  S12 → UserDBAdapter     "The UserDBAdapter component queries..."
  S13 → UserManagement    "...the UserManagement component implements..."
  S16 → TagWatermarking   "...watermarked by the TagWatermarking component."
  S17 → MediaManagement   "...the MediaManagement component forwards..."
  S17 → TagWatermarking   "...from the TagWatermarking component..."
  S19 → Packaging         "...we provide the Packaging component..."
  S26 → MediaAccess       "...the MediaAccess component stores it..."
  S27 → MediaAccess       "The MediaAccess component encapsulates..."
  S29 → UserDBAdapter     "...the UserDBAdapter component provides..."
  S30 → UserDBAdapter     "The UserDBAdapter component creates..."
  S34 → MediaAccess       "...the MediaAccess component fetches..."
  S37 → Reencoding        "...a download can cause re-encoding..."
```

**(LOG)** `Links: 18`

These 18 links are now **immune** — they cannot be filtered, rejected, or judged in any later phase.

---

## Phase 5 — Entity Extraction (1 LLM call)

V22 overrides with `_extract_entities_enriched`. Uses batch_size=50 and timeout=240s.

37 sentences, batch_size=50 → **1 batch** (S1–S37).

### Full LLM prompt

```
Extract ALL references to software architecture components from this document.

COMPONENTS: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade,
MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement,
DownloadLoadBalancer, ParallelWatermarking
KNOWN ALIASES: AudioAccess=MediaAccess, ReEncoder=Reencoding, DataStorage=FileStorage,
Database=DB, User DB Adapter=UserDBAdapter, Audio Watermarking=AudioWatermarking,
Media Management=MediaManagement, Media Access=MediaAccess, File Storage=FileStorage,
Tag Watermarking=TagWatermarking, User Management=UserManagement,
Download Load Balancer=DownloadLoadBalancer, Parallel Watermarking=ParallelWatermarking

WHAT TO LOOK FOR:
- DIRECT MENTIONS: The component name appears in the sentence (exact or alias)
- FUNCTIONAL DESCRIPTIONS: The sentence describes what a specific component does
  Example: "responsible for converting media formats" → refers to a media conversion component
- ROLE REFERENCES: The sentence refers to a component by its architectural role
  Example: "the persistence layer handles all database operations" → refers to the persistence component
- CamelCase splits: "Data Manager" in text may refer to component "DataManager"

BE THOROUGH: Check every sentence against every component. Missing a reference is worse
than including a borderline one (validation will filter later).

DOCUMENT:
S1: One of the main components of Media Store is a server-side web front end, namely
    the Facade component, which delivers websites to the users and provides session management.
[... S2-S36 ...]
S37: However, a download can cause re-encoding of the audio file.

Return JSON:
{"references": [{"sentence": N, "component": "Name", "matched_text": "text found in sentence",
  "match_type": "exact|synonym|partial|functional"}]}
JSON only:
```

### Response

**(not recorded in log)** — the log only records the count after post-processing.

### Code post-processing for each reference

For each reference the LLM returns, the code applies these checks:

```
FOR each reference {sentence: N, component: C, matched_text: M, match_type: T}:

  1. Component exists?
     IF C not in name_to_id → skip

  2. Sentence exists?
     sent = sent_map[N]
     IF sent is None → skip

  3. Dotted-path check?
     IF C appears ONLY inside a dotted path like "x.y.z" → skip
     (MediaStore has no dotted paths, so this never triggers)

  4. matched_text verification:
     IF M.lower() NOT IN sent.text.lower() → skip
     (Catches LLM hallucinating text that isn't in the sentence)

  5. Is it an exact match?
     is_exact = (M.lower() in component_names_lower) OR (C.lower() in M.lower())
     Example: M="Facade", C="Facade" → "facade" in {"facade","db",...} → true

  6. Is it a generic mention?
     _is_generic_mention(C, sent.text):
       IF ' ' in C or '-' in C → false  (multi-word = not generic)
       IF CamelCase pattern [a-z][A-Z] in C → false  (e.g. "MediaAccess")
       IF C.isupper() → false  (e.g. "DB" = acronym)
       IF C[0].islower() → false
       IF _has_standalone_mention(C, text) → false  (case-sensitive word-boundary match)
       IF \b{C.lower()}\b appears literally in text → true  (lowercase form = generic usage)

  7. needs_validation flag:
     needs_val = (NOT is_exact) OR (match_type != "exact") OR (is_generic)
```

**(LOG)** `Candidates: 42`

### Abbreviation guard

```
Check logic:
  FOR each candidate whose matched_text is a known abbreviation:
    Verify the abbreviation isn't a false match (e.g., "GAE" in "GAEDatastore" context)

  MediaStore doc_knowledge has 0 abbreviations → nothing to check.
```

---

## Phase 5b — Targeted Recovery (1 LLM call per unlinked component)

Check which components have zero links from TransArc + entity extraction.

```
  TransArc covers:  {Facade, MediaManagement, UserManagement, UserDBAdapter,
                     TagWatermarking, Packaging, MediaAccess, Reencoding}
  Entity covers:    adds {DB, FileStorage, AudioWatermarking}

  Combined: 11 of 14 components
```

**(LOG)** Unlinked: **3** components — `Cache, DownloadLoadBalancer, ParallelWatermarking`

> Note: AudioWatermarking was found by entity extraction (unlike the previous
> version of this document which incorrectly listed 4 unlinked components).

### LLM prompt (per component)

For each unlinked component, one LLM call:

```
Find ALL sentences that discuss the software component "Cache".

Look for:
- Direct mentions of "Cache" or any alias
- Descriptions of what Cache does (functional descriptions)
- References to Cache's role in the architecture

DOCUMENT:
S1: One of the main components of Media Store...
[... S2-S36 ...]
S37: However, a download can cause re-encoding of the audio file.

Return JSON:
{"references": [{"sentence": N, "matched_text": "text found", "reason": "why this refers to Cache"}]}

Be thorough — find ALL sentences that discuss this component.
JSON only:
```

**(LOG)** `Found: 0 additional candidates` — these components are genuinely absent from the text.

---

## Phase 6 — Validation (code-first + up to 3 LLM calls)

**(LOG)** Input: 42 candidates → Output: 32 validated.

The code-first auto-approve handles most candidates deterministically. Only remaining ones go to LLM.

### Step 0: Separate direct vs needs-validation

```
Check logic:
  direct = candidates where needs_validation == false
    (exact name matches with exact match_type and not generic)
  needs = candidates where needs_validation == true
```

### Step 0b: Pre-check — reject generic mentions

```
FOR each candidate in needs:
  sent = sent_map[candidate.sentence_number]
  IF _is_generic_mention(candidate.component_name, sent.text) → reject

  _is_generic_mention logic (repeated from Phase 5):
    "DB" → isupper() → false (not generic)
    "FileStorage" → CamelCase → false
    "MediaManagement" → CamelCase → false
    No candidates rejected by this check in MediaStore.
```

### Step 1: Code-first auto-approve

Build alias lookup for each component from doc_knowledge:

```
DB aliases:             {"DB", "Database"}
FileStorage aliases:    {"FileStorage", "DataStorage", "File Storage"}
MediaAccess aliases:    {"MediaAccess", "AudioAccess", "Media Access"}
Reencoding aliases:     {"Reencoding", "ReEncoder"}
MediaManagement aliases:{"MediaManagement", "Media Management"}
...etc for all components
```

```
Check logic:
  FOR each candidate needing validation:
    FOR each alias of candidate.component_name:
      IF len(alias) >= 3:
        IF alias.lower() IN sentence.text.lower()    ← substring match
        → auto-approve
      ELIF len(alias) >= 2:
        IF \b{alias}\b matches in sentence (case-insensitive)  ← word-boundary match
        → auto-approve
```

Traced through key candidates:

```
S12 → DB ("database"):
  alias "DB" (len=2): word-boundary \bDB\b in "The UserDBAdapter component queries the database."
    → "DB" appears inside "UserDBAdapter" but \bDB\b needs word boundary
    → no standalone "DB" → no match
  alias "Database" (len=8): "database" in sent.lower()?
    → "the userdbadapter component queries the database." contains "database" → YES
  → AUTO-APPROVED ✓

S23 → DB: "The Database component represents an actual database..."
  → "database" substring → YES → AUTO-APPROVED ✓

S25 → DB: "...query that is sent to the Database component."
  → "database" substring → YES → AUTO-APPROVED ✓

S27 → DB: "The MediaAccess component encapsulates database access..."
  → "database" substring → YES → AUTO-APPROVED ✓

S29 → DB: "...encapsulate database access for the user data."
  → "database" substring → YES → AUTO-APPROVED ✓

S31, S32, S33, S34 → DB: all contain "Database" or "database" → AUTO-APPROVED ✓

S33 → FileStorage: "...to decouple the DataStorage from the database."
  → alias "DataStorage" (len=11): "datastorage" in lowercase text → YES → AUTO-APPROVED ✓

S35 → FileStorage: "...retrieved from the DataStorage."
  → "datastorage" → YES → AUTO-APPROVED ✓

S36 → FileStorage: "...stored in the DataStorage without any change."
  → "datastorage" → YES → AUTO-APPROVED ✓

S20 → Reencoding: "The ReEncoder component converts..."
  → alias "ReEncoder" (len=9): "reencoder" in lowercase text → YES → AUTO-APPROVED ✓

S25 → MediaAccess: "...AudioAccess creates a query..."
  → alias "AudioAccess" (len=11): "audioaccess" in lowercase text → YES → AUTO-APPROVED ✓

S28 → MediaAccess: "Furthermore, it fetches a list of all available audio files."
  → alias "MediaAccess" (len=11): "mediaaccess" in text? → NO
  → alias "AudioAccess" (len=11): "audioaccess" in text? → NO
  → alias "Media Access" (len=12): "media access" in text? → NO
  → NOT auto-approved → needs LLM

S9 → MediaManagement: "Furthermore, it fetches audio files..."
  → "mediamanagement" in text? → NO
  → "media management" in text? → NO
  → NOT auto-approved → needs LLM

S24 → DB: "It stores user information and meta-data..."
  → "database" in text? → NO
  → \bDB\b word-boundary? → NO (no "DB" in text)
  → NOT auto-approved → needs LLM
```

**(console print, not in JSON log)** `Code-first v2 auto-approved: 28, LLM needed: 4`

> This split is from the code's `print()` output (V22 line 828), not the JSON log.
> The JSON log only records `{candidates: 42, validated: 32}`.

### Step 2: 2-pass LLM intersect

For the remaining candidates, two independent LLM calls with different focus:

**Pass A prompt:**

```
Validate component references in a software architecture document.
Focus on ACTOR role: is the component performing an action or being described?

COMPONENTS: [14 components]

[context from learned_patterns if any]

IMPORTANT DISTINCTIONS:
- "the routing logic" / "business logic" = generic English, NOT an architectural component → REJECT
- "client-side rendering" / "on the client" = generic usage, NOT a specific component → REJECT
- "data storage layer" / "in-memory cache" = generic concept, NOT a specific component → REJECT
- But "Router handles request processing" = the component IS the actor → APPROVE
- Section headings naming a component = introduces that component's section → APPROVE

CASES:
Case 1: "it fetches" -> MediaManagement
  [prev: The MediaManagement component coordinate...] "Furthermore, it fetches audio files
  from a specific location (e.g., a dedicated file server or a local disk) when processing
  download requests."
Case 2: "It stores" -> DB
  [prev: The Database component represents an act...] "It stores user information and
  meta-data of audio files such as the name and the genre."
Case 3: "it fetches" -> MediaAccess
  [prev: The MediaAccess component encapsulates d...] "Furthermore, it fetches a list of
  all available audio files."
[Case 4: other remaining candidate]

Return JSON:
{"validations": [{"case": 1, "approve": true/false}]}
JSON only:
```

**Pass B prompt:** Same structure but focus = "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?"

```
Intersect logic:
  FOR each case:
    approved = (Pass_A approves) AND (Pass_B approves)
```

### Step 3: Evidence post-filter (for generic-risk candidates)

```
Check logic:
  generic_risk = Phase 1 ambiguous names ∪ GENERIC_COMPONENT_WORDS
  generic_risk = {"Cache", "DB", "Facade", "FileStorage", "Packaging"}

  FOR each candidate that passed 2-pass AND component is in generic_risk:
    → ask LLM for EXACT text evidence
    → verify evidence contains an alias of the component
    → if no evidence or no alias in evidence → REJECT
```

If S24→DB passed the 2-pass intersect, it would need evidence. S24 text is "It stores user information..." — no alias of DB appears in the sentence, so it would fail the evidence filter. S24→DB was later rescued by Phase 7 coreference instead.

**(LOG)** `Validated: 32 (of 42)` — internal step breakdown is not recorded in the log.

---

## Phase 7 — Coreference Resolution (1 LLM call)

`complex = false` → **discourse mode** with antecedent-citation requirement.

### Build discourse model (deterministic)

The code scans sentences for component name mentions to track what's being discussed:

```
S7:  "MediaManagement" found → topic = MediaManagement
S8:  "MediaManagement" found → topic = MediaManagement
S11: "UserManagement" found → topic = UserManagement
S12: "UserDBAdapter" found → topic = UserDBAdapter
S16: "TagWatermarking" found → topic = TagWatermarking
S17: "MediaManagement" + "TagWatermarking" → topic = MediaManagement (most frequent in paragraph)
S23: "Database" substring → matches "DB" → topic = DB
...
```

### Find pronoun sentences

```
PRONOUN_PATTERN = \b(it|they|this|these|that|those|its|their|the component|the service)\b
(case-insensitive)

Pronoun sentences (11 found):
  S3:  "To this end, the Facade component delivers..."         → "this"
  S9:  "Furthermore, it fetches audio files..."                → "it"
  S15: "To this end, the requested files are first reencoded." → "this"
  S17: "...component forwards these audio files..."            → "these"
  S21: "This can result in reduction of file sizes."           → "this"
  S22: "...components that are concerned...their meta-data."   → "that","their"
  S24: "It stores user information..."                         → "it"
  S25: "...a query that is sent to the Database component."    → "that"
  S26: "...the MediaAccess component stores it at..."          → "it"
  S28: "Furthermore, it fetches a list..."                     → "it"
  S36: "...it will be stored in the DataStorage..."            → "it"
```

### Full LLM prompt

All 11 pronoun sentences fit in 1 batch (batch_size=12).

```
Resolve pronoun references to architecture components.

COMPONENTS: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade,
MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement,
DownloadLoadBalancer, ParallelWatermarking

--- Case 1: S3 ---
PREVIOUS:
  S1: One of the main components of Media Store is a server-side web front end, namely
      the Facade component, which delivers websites to the users and provides session management.
  S2: To meet the user authentication requirement, registration and log-in have to be offered.
>>> To this end, the Facade component delivers the corresponding registration and log-in
    pages to users.

--- Case 2: S9 ---
PREVIOUS:
  S6: In addition, users can browse, download, and upload audio files using the Facade component.
  S7: Application business logic is provided by a central business logic component,
      called the MediaManagement component.
  S8: The MediaManagement component coordinates the communication of other components.
>>> Furthermore, it fetches audio files from a specific location (e.g., a dedicated
    file server or a local disk) when processing download requests.

--- Case 3: S15 ---
PREVIOUS:
  S12: The UserDBAdapter component queries the database.
  S13: When a user logs into the system, Media Store does not store the password in plain
       text, but rather the UserManagement component implements further functions to hash
       and salt the passwords.
  S14: To reduce the risk of copyright infringements, all downloaded files are watermarked.
>>> To this end, the requested files are first reencoded.

--- Case 4: S17 ---
PREVIOUS:
  S14: To reduce the risk of copyright infringements, all downloaded files are watermarked.
  S15: To this end, the requested files are first reencoded.
  S16: The re-encoded files are then digitally and individually watermarked by the
       TagWatermarking component.
>>> Afterward, the MediaManagement component forwards these audio files from the
    TagWatermarking component to the user.

--- Case 5: S21 ---
PREVIOUS:
  S18: Hence, each downloaded audio file is uniquely connected to a user ID can can be
       associated with the user if the audio file should ever appear in the public Internet.
  S19: To allow users to download several files at a time, we provide the Packaging
       component, which archives multiple audio files into a single compressed file.
  S20: The ReEncoder component converts the bit rates of audio files.
>>> This can result in reduction of file sizes.

--- Case 6: S22 ---
PREVIOUS:
  S19: To allow users to download several files at a time, we provide the Packaging
       component, which archives multiple audio files into a single compressed file.
  S20: The ReEncoder component converts the bit rates of audio files.
  S21: This can result in reduction of file sizes.
>>> The persistence tier contains components that are concerned with the storage of
    audio files and their meta-data.

--- Case 7: S24 ---
PREVIOUS:
  S21: This can result in reduction of file sizes.
  S22: The persistence tier contains components that are concerned with the storage
       of audio files and their meta-data.
  S23: The Database component represents an actual database (e.g., MySQL).
>>> It stores user information and meta-data of audio files such as the name
    and the genre.

--- Case 8: S25 ---
PREVIOUS:
  S22: The persistence tier contains components that are concerned with the storage
       of audio files and their meta-data.
  S23: The Database component represents an actual database (e.g., MySQL).
  S24: It stores user information and meta-data of audio files such as the name and the genre.
>>> After the user calls the page to list all available audio files, AudioAccess creates
    a query that is sent to the Database component.

--- Case 9: S26 ---
PREVIOUS:
  S23: The Database component represents an actual database (e.g., MySQL).
  S24: It stores user information and meta-data of audio files such as the name and the genre.
  S25: After the user calls the page to list all available audio files, AudioAccess creates
       a query that is sent to the Database component.
>>> When a user uploads an audio file, the MediaAccess component stores it at the
    predefined location.

--- Case 10: S28 ---
PREVIOUS:
  S25: After the user calls the page to list all available audio files, AudioAccess creates
       a query that is sent to the Database component.
  S26: When a user uploads an audio file, the MediaAccess component stores it at the
       predefined location.
  S27: The MediaAccess component encapsulates database access for meta-data of audio files.
>>> Furthermore, it fetches a list of all available audio files.

--- Case 11: S36 ---
PREVIOUS:
  S33: By contrast, all audio files are stored in a specific location (e.g., a dedicated
       file server or a local disk) to decouple the DataStorage from the database.
  S34: When a user requests files to download, the MediaAccess component fetches the
       associated meta-data from the Database.
  S35: Afterwards, based on the user request and the corresponding meta-data, the file
       will be retrieved from the DataStorage.
>>> When a user uploads a file, it will be stored in the DataStorage without any change.

For each pronoun that refers to a component, you MUST provide:
- The antecedent_sentence number where the component was EXPLICITLY NAMED
- The antecedent_text: the EXACT text from that sentence containing the component name

STRICT RULES:
- The component name (or known alias) MUST appear in the antecedent sentence
- The antecedent sentence MUST be within the previous 3 sentences
- The pronoun MUST be the grammatical subject referring back to that component
- If unsure, DO NOT include the resolution

Return JSON:
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name",
  "antecedent_sentence": M, "antecedent_text": "exact text with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:
```

### Response **(not recorded in log, but we know the 3 results from LOG)**

The LLM returned 3 resolutions out of 11 cases. The other 8 were correctly skipped:
- S3: "this" is a discourse connector ("To this end"), Facade is explicitly named → no pronoun to resolve
- S15: "this" is a discourse connector, no component mentioned → skip
- S17: "these" modifies "audio files", not a component reference → skip
- S21: "This" refers to bit-rate conversion (a process, not a component) → skip
- S22: "that" is a relative pronoun ("components that are concerned"), "their" refers to "audio files" → skip
- S25: "that" is a relative pronoun ("a query that is sent") → skip
- S26: "it" refers to "an audio file" (the user's upload, not a component) → skip
- S36: "it" refers to "a file" (the user's upload, not a component) → skip

### Code verification of each resolution

For each LLM resolution, the code verifies:

```
Verification logic:
  1. Does the component name (or alias) actually appear in the antecedent sentence?
     → _has_standalone_mention(comp, antecedent.text) checks case-sensitive word-boundary
     → _has_alias_mention(comp, antecedent.text) checks synonyms from doc_knowledge

  2. Is the antecedent within 3 sentences?
     → abs(sentence_number - antecedent_number) <= 3

  3. Is the sentence a subprocess?
     → learned_patterns.is_subprocess(sent.text)
     → if true, reject the coref link
```

**Resolution 1: S9 → MediaManagement (antecedent S8)**

```
_has_standalone_mention("MediaManagement", S8.text):
  is_single = (' ' not in "MediaManagement") → true
  cap_name = "MediaManagement" (already capitalized)
  pattern: \bMediaManagement\b (case-sensitive)
  S8: "The MediaManagement component coordinates..." → MATCH at position 4
  Not preceded by '.' ✓, not followed by '.' ✓
  → true

Distance: abs(9 - 8) = 1 ≤ 3 ✓

is_subprocess("Furthermore, it fetches audio files..."):
  Checks if any subprocess_term appears in text
  {"DataStorage", "hash and salt", "session management"}
  "datastorage" in text? no. "hash and salt" in text? no. "session management"? no.
  → false ✓

→ LINK CREATED: S9 → MediaManagement (coreference)
```

**Resolution 2: S24 → DB (antecedent S23)**

```
_has_standalone_mention("DB", S23.text):
  is_single = true
  cap_name = "DB" (first char upper, rest upper)
  pattern: \bDB\b (case-sensitive)
  S23: "The Database component represents an actual database..." → no "DB" → no match
  → false

_has_alias_mention("DB", S23.text):
  Check synonyms where target == "DB":
    "Database" → "DB": search \bdatabase\b in S23.lower()
    S23 lower: "the database component represents an actual database..."
    → MATCH
  → true

Distance: abs(24 - 23) = 1 ≤ 3 ✓

→ LINK CREATED: S24 → DB (coreference)
```

**Resolution 3: S28 → MediaAccess (antecedent S27)**

```
_has_standalone_mention("MediaAccess", S27.text):
  pattern: \bMediaAccess\b (case-sensitive)
  S27: "The MediaAccess component encapsulates..." → MATCH
  → true

Distance: abs(28 - 27) = 1 ≤ 3 ✓

→ LINK CREATED: S28 → MediaAccess (coreference)
```

### Generic coref filter

```
Check logic:
  FOR each coref link:
    IF _needs_antecedent_check(comp_name):
      (returns true ONLY for single-word, non-CamelCase, non-uppercase, capitalized names)

      "MediaManagement" → CamelCase → false → skip filter
      "DB" → isupper() → false → skip filter
      "MediaAccess" → CamelCase → false → skip filter

  All 3 pass — no links filtered.
```

### Deterministic pronoun coref

```
Check logic:
  FOR each sentence:
    IF sentence starts with "It " or "As such, it ":
      Look back 1-3 sentences for exactly 1 component mentioned
      IF exactly 1 found AND not already linked → create coref link

  S24: starts with "It " → yes
    S23 mentions: "Database" → alias for DB → 1 component
    BUT (S24, DB) already in coref_links → skip (duplicate)

  S9: starts with "Furthermore, it" → does NOT match ^(It|As such, it) → skip

  No new links from deterministic pronoun coref.
```

**(LOG)** `Coref links: 3` — S9→MediaManagement, S24→DB, S28→MediaAccess

---

## Phase 8 — Implicit References (SKIPPED)

```
[Phase 8] Implicit References — SKIPPED (dead weight)
```

---

## Phase 8b — Partial Reference Injection (deterministic)

```
doc_knowledge.partial_references = {}  ← empty, no partials registered

Result: 0 links injected.
```

---

## Combine & Deduplicate

Merge all sources:

```
  transarc links:   18
  entity links:     32 (validated candidates → SadSamLink)
  coref links:       3
  partial links:     0
  ─────────────────────
  Raw total:        53
```

Dedup by (sentence_number, component_id), keeping higher priority:

```
Priority: transarc(5) > validated(4) > entity(3) > coreference(2) > partial_inject(1)

Examples:
  S1→Facade:  transarc(5) vs validated(4) → keep transarc
  S7→MediaManagement: transarc(5) vs validated(4) → keep transarc
  S12→DB: only validated(4) → keep
  S9→MediaManagement: only coreference(2) → keep (transarc didn't cover S9)
  S24→DB: only coreference(2) → keep
  S28→MediaAccess: validated(4) vs coreference(2) → keep validated
```

**(LOG)** `After dedup: 35 unique links` — 18 transarc + 15 validated + 2 coreference

---

## Phase 8c — Boundary Filters (deterministic, non-TransArc only)

Process each non-TransArc link through 4 filters:

```
FOR each link:
  IF (sentence_number, component_id) in transarc_set → IMMUNE, skip all filters

  Filter 1 — package_path:
    Does the component name appear ONLY inside a dotted path (e.g., "x.DB.y")?
    MediaStore has no dotted paths → never triggers.

  Filter 2 — generic_word:
    For single-word components only.
    IF case-sensitive name appears in text → pass (not generic)
    ELSE check if lowercase name appears after a NON-modifier word
      (modifiers = "the", "a", "is", "in", "and", etc.)
      Example: "cascade logic" → "cascade" is not a modifier → generic_word!
    MediaStore: "DB" never appears case-sensitive, but "db" doesn't appear
    standalone either → pass. "database" is len>1 word → not checked here.

  Filter 3 — weak_partial:
    Only for source="partial_inject" links.
    MediaStore has no partial_inject links → never triggers.

  Filter 4 — abbrev_guard:
    Check if an abbreviation match is valid.
    MediaStore has no abbreviations → never triggers.
```

**(LOG)** `Rejected: 0` — no links filtered.

---

## Phase 9 — Judge Review (2 LLM calls)

**(LOG)** Input: 35 links → Output: 32 approved, **3 rejected**.

### Separate safe vs review

```
Check logic:
  FOR each link:
    1. TransArc? → SAFE (immune, never judged)
    2. _has_standalone_mention(comp, sent.text)? → SAFE
       (case-sensitive word-boundary match, excluding dotted paths and hyphens)
    3. _has_alias_mention(comp, sent.text)? → SAFE
       (any synonym or partial from doc_knowledge appears in text)
    4. Otherwise → REVIEW (needs judge)
```

The 18 TransArc links are automatically safe. For the remaining 17 non-TransArc links, the code checks whether the component name or any alias appears in the sentence text.

**(LOG)** The log records 35 input → 32 approved, 3 rejected:

```
Rejected:
  S25 → MediaAccess (validated)
  S27 → DB         (validated)
  S29 → DB         (validated)
```

### Full judge prompt

```
JUDGE: Review trace links between documentation sentences and software architecture components.

A sentence S should be linked to component C ONLY if ALL FOUR of the following are true:

RULE 1 — REFERENCE: S actually refers to C (not just string-matches its name).
  REJECT if the component name appears as part of a generic English phrase rather than as a
  component reference (e.g., "cascade logic" does not refer to a component named "BusinessLogic";
  "on the client side" does not refer to a component named "WebClient").

RULE 2 — ARCHITECTURAL LEVEL: S describes C at the architectural level (role, behavior,
  interactions with other components), NOT at the implementation level.
  REJECT if S describes:
  - Package or directory structure (e.g., "x.auth contains test cases for the AuthModule")
  - Internal classes or sub-components (e.g., "TokenValidator checks tokens" — this is about
    TokenValidator, an internal class, not the parent AuthModule component)
  - API exception handling details (e.g., "not found throws NotFoundException")
  - Data format or schema specifications

RULE 3 — TOPIC: C is the topic or subject of S (what S is primarily about).
  REJECT if C is mentioned incidentally while S discusses something else (e.g., "react upon
  changes within the interface" — S is about architecture patterns, the interface component is
  incidental; "RequestHandler stores it via the DataLayer" — S is about RequestHandler, not DataLayer).

RULE 4 — NOT GENERIC: The reference is to C as a specific architectural component, not as a
  generic English word (e.g., "managing transactions" does not refer to a component named
  "CoreServices"; "common data format" does not refer to a component named "SharedUtils").

COMPONENTS: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade,
MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement,
DownloadLoadBalancer, ParallelWatermarking

MATCH TEXT GUIDANCE:
- If match text differs from component name (e.g., match:"backends" for a server component),
  ask: is this match text a DIRECT REFERENCE to the component, or a generic term?
- If match:NONE, the link was inferred from context/pronouns — verify the reference chain.

SOURCE-SPECIFIC RULES:
- "coreference": Pronoun resolution. Verify pronoun CLEARLY refers to claimed component.
- "partial_inject": Partial name match. Verify the partial refers to THIS component, not generic usage.
- "validated": Entity extraction found this. Apply all 4 rules strictly.

LINKS:
[cases built from the review links, with previous sentences as context]

Return JSON:
{"judgments": [{"case": 1, "approve": true/false, "reason": "brief"}]}
JSON only:
```

### Union voting logic

```
This exact prompt is sent TWICE (two independent LLM calls).
Judge uses union voting: reject ONLY if BOTH passes reject.

FOR each case:
  IF pass1.approve OR pass2.approve → APPROVED
  IF NOT pass1.approve AND NOT pass2.approve → REJECTED
```

### Rejected links — what happened

**(LOG)** These 3 links were rejected (both judge passes rejected them):

**S25 → MediaAccess** — S25: "After the user calls the page to list all available audio files, AudioAccess creates a query that is sent to the Database component."

The judge likely applied **RULE 3 (TOPIC)**: While "AudioAccess" (alias for MediaAccess) appears in S25, the sentence's primary topic is the query flow to the Database component. MediaAccess/AudioAccess is the sender, but the sentence is really about the Database query process.

**S27 → DB** — S27: "The MediaAccess component encapsulates database access for meta-data of audio files."

The judge likely applied **RULE 4 (NOT GENERIC)**: "database access" is a generic English phrase describing what MediaAccess does. The word "database" is an adjective modifying "access", not a reference to the DB component as an architectural entity. The sentence's topic (RULE 3) is MediaAccess, not DB.

**S29 → DB** — S29: "By contrast, the UserDBAdapter component provides all functions required in order to encapsulate database access for the user data."

Same pattern as S27 — "database access" is a function description about UserDBAdapter. The sentence is about UserDBAdapter's role, and "database" is used generically.

> Note: The judge's specific reasoning is not recorded in the log. The explanations above are
> inferred from the judge prompt rules and the sentence content.

**(LOG)** `Approved: 32 (rejected 3)`

---

## Phase 10 — FN Recovery (SKIPPED)

```
[Phase 10] FN Recovery — SKIPPED (dead weight)
```

---

## Final Output

**(LOG)** `Final: 32 links`

Complete link list with source and evidence:

```
 #  Sent  Component         Source       Evidence in sentence
── ───── ──────────────── ──────────── ──────────────────────────────────────────
 1  S1    Facade            transarc    "...namely the Facade component..."
 2  S3    Facade            transarc    "...the Facade component delivers..."
 3  S6    Facade            transarc    "...using the Facade component."
 4  S7    MediaManagement   transarc    "...the MediaManagement component."
 5  S8    MediaManagement   transarc    "The MediaManagement component coordinates..."
 6  S9    MediaManagement   coreference "it" ← pronoun from S8 (MediaManagement)
 7  S11   UserManagement    transarc    "The UserManagement component answers..."
 8  S12   UserDBAdapter     transarc    "The UserDBAdapter component queries..."
 9  S12   DB                validated   "...queries the database." (synonym: Database→DB)
10  S13   UserManagement    transarc    "...the UserManagement component implements..."
11  S16   TagWatermarking   transarc    "...by the TagWatermarking component."
12  S17   MediaManagement   transarc    "...the MediaManagement component forwards..."
13  S17   TagWatermarking   transarc    "...from the TagWatermarking component..."
14  S19   Packaging         transarc    "...the Packaging component, which archives..."
15  S20   Reencoding        validated   "The ReEncoder component..." (synonym: ReEncoder→Reencoding)
16  S23   DB                validated   "The Database component..." (synonym: Database→DB)
17  S24   DB                coreference "It" ← pronoun from S23 (Database=DB)
18  S25   DB                validated   "...sent to the Database component." (synonym)
19  S26   MediaAccess       transarc    "...the MediaAccess component stores it..."
20  S27   MediaAccess       transarc    "The MediaAccess component encapsulates..."
21  S28   MediaAccess       validated   "it fetches..." (validated as continuation of S27)
22  S29   UserDBAdapter     transarc    "...the UserDBAdapter component provides..."
23  S30   UserDBAdapter     transarc    "The UserDBAdapter component creates..."
24  S31   DB                validated   "The Database component then executes..." (synonym)
25  S32   DB                validated   "...stored in the Database component." (synonym)
26  S33   FileStorage       validated   "...the DataStorage from..." (synonym: DataStorage→FileStorage)
27  S33   DB                validated   "...from the database." (synonym: Database→DB)
28  S34   MediaAccess       transarc    "...the MediaAccess component fetches..."
29  S34   DB                validated   "...from the Database." (synonym)
30  S35   FileStorage       validated   "...from the DataStorage." (synonym)
31  S36   FileStorage       validated   "...in the DataStorage without any change." (synonym)
32  S37   Reencoding        transarc    "...cause re-encoding of the audio file."
```

**Breakdown:**

```
  By source:
    transarc:    18 links (56%)  — verbatim component name matches from baseline
    validated:   12 links (38%)  — synonym/alias matches found by LLM, verified by code+LLM
    coreference:  2 links (6%)   — pronoun "it" resolved to component via antecedent

  By component:
    DB:               8 links (S12, S23, S24, S25, S31, S32, S33, S34)
    MediaManagement:  4 links (S7, S8, S9, S17)
    MediaAccess:      4 links (S26, S27, S28, S34)
    UserDBAdapter:    3 links (S12, S29, S30)
    Facade:           3 links (S1, S3, S6)
    FileStorage:      3 links (S33, S35, S36)
    TagWatermarking:  2 links (S16, S17)
    UserManagement:   2 links (S11, S13)
    Reencoding:       2 links (S20, S37)
    Packaging:        1 link  (S19)

  Unlinked components (0 links):
    Cache                — never mentioned in document
    AudioWatermarking    — S14 says "watermarked" but doesn't name AudioWatermarking
    DownloadLoadBalancer — never mentioned
    ParallelWatermarking — never mentioned

  Rejected links (3):
    S25 → MediaAccess   — judge: "AudioAccess" is incidental, sentence topic is Database query
    S27 → DB            — judge: "database access" is generic English, not a DB reference
    S29 → DB            — judge: same as S27, "database access" describes UserDBAdapter's function

  Total runtime: 291.3 seconds
  Phase log saved: results/llm_logs/v20d_mediastore_20260220_112329.json
```
