# Challenge 1: what reference form actually carries each gold link

## Method

The population is every gold doc-to-model trace link in the benchmark: **195
(sentence, component) pairs over 5 projects**, derived by `audit.py --sheets`
into `ch1_gold_sheet.csv`. That sheet carries an *automatic* guess at the
reference form (`auto_surface`), computed by contiguous-token containment.

For this report every one of the 195 links was re-read by hand in the SAD
(`benchmark/<project>/text_*/<name>.txt`, one sentence per line), together with
its neighbours wherever the sentence carries no name. The verdicts are recorded
as data in `ch1_annotations.py`, which also defines the four categories; this
script only joins and counts them, and asserts that the annotation set is exactly
the population (no row annotated twice, none missing).

- `canonical` — the catalog name itself (modulo case/space/hyphen/camelCase/plural)
- `alias` — a different document string (expansion, contraction, package or substitute name)
- `partial` — only part of a multi-word name (distinctive word or generic head noun)
- `pronoun_or_implicit` — no form of the name; pronoun, demonstrative or elided subject
- `other` — none of the above

The automatic classifier's five labels are collapsed onto these four for
comparison: `partial_d`/`partial_g` → `partial`, `none` → `pronoun_or_implicit`
(`none` is its stand-in for "no surface form at all").

A system "got" a link if SWATTR emitted it, or if Artemis / our approach / our
approach without the document-alias table emitted it in **≥2 of 3 runs**.

## Corrected counts per form

| form | manual | share | automatic | delta |
|---|---:|---:|---:|---:|
| `canonical` | 140 | 71.8% | 135 | +5 |
| `alias` | 17 | 8.7% | 22 | -5 |
| `partial` | 22 | 11.3% | 18 | +4 |
| `pronoun_or_implicit` | 16 | 8.2% | 20 | -4 |
| **total** | **195** | | **195** | |

### Per project

| project | links | canonical | alias | partial | pronoun/implicit | non-canonical share |
|---|---:|---:|---:|---:|---:|---:|
| mediastore | 31 | 17 | 11 | 0 | 3 | 45.2% |
| teastore | 27 | 20 | 0 | 1 | 6 | 25.9% |
| teammates | 57 | 49 | 0 | 3 | 5 | 14.0% |
| bigbluebutton | 62 | 36 | 6 | 18 | 2 | 41.9% |
| jabref | 18 | 18 | 0 | 0 | 0 | 0.0% |
| **all** | **195** | **140** | **17** | **22** | **16** | **28.2%** |

Projects in which each form occurs at all (out of 5):

| form | projects | present in |
|---|---:|---|
| `canonical` | 5 | mediastore, teastore, teammates, bigbluebutton, jabref |
| `alias` | 2 | mediastore, bigbluebutton |
| `partial` | 3 | teastore, teammates, bigbluebutton |
| `pronoun_or_implicit` | 4 | mediastore, teastore, teammates, bigbluebutton |

## How concentrated is each non-canonical form

### `alias` — 17 links

Top project supplies **11/17 (64.7%)** of them.

| project | links | share of this form | supplying components |
|---|---:|---:|---|
| mediastore | 11 | 64.7% | DB (6), FileStorage (3), Reencoding (1), MediaAccess (1) |
| bigbluebutton | 6 | 35.3% | HTML5 Server (3), BBB web (1), FSESL (1), kurento (1) |

2 of the 8 distinct components carrying this form account for at least half of it (9/17).

### `partial` — 22 links

Top project supplies **18/22 (81.8%)** of them.

| project | links | share of this form | supplying components |
|---|---:|---:|---|
| bigbluebutton | 18 | 81.8% | HTML5 Client (9), HTML5 Server (7), WebRTC-SFU (2) |
| teammates | 3 | 13.6% | GAE Datastore (3) |
| teastore | 1 | 4.5% | WebUI (1) |

2 of the 5 distinct components carrying this form account for at least half of it (16/22).

### `pronoun_or_implicit` — 16 links

Top project supplies **6/16 (37.5%)** of them.

| project | links | share of this form | supplying components |
|---|---:|---:|---|
| teastore | 6 | 37.5% | Persistence (3), WebUI (1), ImageProvider (1), Recommender (1) |
| teammates | 5 | 31.2% | Storage (2), Client (1), Logic (1), Test Driver (1) |
| mediastore | 3 | 18.8% | MediaManagement (1), DB (1), MediaAccess (1) |
| bigbluebutton | 2 | 12.5% | BBB web (1), Apps (1) |

5 of the 13 distinct components carrying this form account for at least half of it (8/16).

## Recall per corrected form

| form | links | SWATTR | Artemis | ours | ours-noknow |
|---|---:|---:|---:|---:|---:|
| `canonical` | 140 | 139 (99.3%) | 127 (90.7%) | 138 (98.6%) | 128 (91.4%) |
| `alias` | 17 | 0 (0.0%) | 15 (88.2%) | 17 (100.0%) | 2 (11.8%) |
| `partial` | 22 | 9 (40.9%) | 3 (13.6%) | 15 (68.2%) | 15 (68.2%) |
| `pronoun_or_implicit` | 16 | 0 (0.0%) | 12 (75.0%) | 16 (100.0%) | 12 (75.0%) |
| **all** | **195** | 148 (75.9%) | 157 (80.5%) | 186 (95.4%) | 157 (80.5%) |

Recall on the non-canonical links only (the links Challenge 1 is about):

| system | non-canonical links recovered |
|---|---:|
| SWATTR | 9/55 (16.4%) |
| Artemis | 30/55 (54.5%) |
| ours | 48/55 (87.3%) |
| ours-noknow | 29/55 (52.7%) |

Same split per project, for our approach:

| project | non-canonical links | ours ≥2/3 | SWATTR |
|---|---:|---:|---:|
| mediastore | 14 | 14 | 0 |
| teastore | 7 | 7 | 0 |
| teammates | 8 | 7 | 0 |
| bigbluebutton | 26 | 20 | 9 |
| jabref | 0 | — | — |

### Where the partial-form misses happen

**All 7 partial-form misses are judge rejections. None is a scan-coverage gap.**
Each of the 7 was proposed by the partial-name scan in all three runs and then
rejected by a judge in all three runs. Source: the recorded decisions in
`phase_states/s_linker126/openai/<project>/linker_{name,coreference}.pkl`.

| link | what the sentence writes | name judge | coref judge |
|---|---|---|---|
| teammates s122 → `GAE Datastore` | "the complexities of datastore" | rejected 3/3 | rejected 3/3 |
| bigbluebutton s6 → `HTML5 Server` | "the BigBlueButton server" | rejected 3/3 | not proposed |
| bigbluebutton s39 → `HTML5 Server` | "control the BigBlueButton server" | rejected 3/3 | not proposed |
| bigbluebutton s47 → `HTML5 Server` | "running on the BigBlueButton server" | rejected 3/3 | not proposed |
| bigbluebutton s65 → `WebRTC-SFU` | "connecting using WebRTC" | rejected 3/3 | not proposed |
| bigbluebutton s73 → `WebRTC-SFU` | "an audio connection ... via WebRTC" | rejected 3/3 | not proposed |
| bigbluebutton s73 → `HTML5 Server` | "an audio connection to the server" | rejected 3/3 | not proposed |

Every name-judge decision is `naming: "word only"`, `path:
partial_name_rejected`. In most of them the judge recorded `claim: none` — it
found no phrase in the sentence that names the component. The scan's whole-name
skip is not involved: it is per *component*, so s6 writing "HTML5 client" in full
does not stop the scan from proposing `HTML5 Server` for the same sentence.

The three misses that form the weakest cell of the paper's form-by-form table
(s6, s65, s73) behave the same way, and the no-knowledge arm differs on one of
them:

| link | full arm (3 runs) | no-knowledge arm (3 runs) |
|---|---|---|
| s6 → `HTML5 Server` | rejected 3/3 | rejected 1/3, approved 2/3 |
| s65 → `WebRTC-SFU` | rejected 3/3 | rejected 3/3 |
| s73 → `WebRTC-SFU` | rejected 3/3 | rejected 3/3 |

In the two approvals the judge quoted "connects directly with the BigBlueButton
server over port 443 (SSL)". So the alias table costs this link: the two arms do
not recover the *same* 15 partial links although both recover 15 — the table wins
teastore s8 (`WebUI` as "The UI") and loses bigbluebutton s6. The equal totals in
the recall table above are a swap, not a tie.

```python
kept, rej = A.judged(Path(RUN), "bigbluebutton", "linker_name")
assert (6,  "_yGgUMFkHEeyewPSmlgszyA") in rej   # HTML5 Server
assert (65, "_s0aIcFkHEeyewPSmlgszyA") in rej   # WebRTC-SFU
```

Consequence for the paper: the partial-form recall shortfall is a *judging*
result, not a linker-coverage result. Raising recall on this form means changing
how the name judge treats a generic head noun ("the server", "the datastore") in
a sentence that carries no other form of the name, not adding scan coverage.

## All 195 gold links

`auto` = the automatic classifier's label (raw). `verdict` = the manual one.
`got by` lists SWATTR / Artemis / ours / ours-noknow on the ≥2-of-3 rule.

| # | project | s | component | auto | verdict | evidence | got by |
|---:|---|---:|---|---|---|---|---|
| 1 | mediastore | 1 | Facade | canonical | **canonical** | 'namely the Facade component' | SWATTR+Artemis+ours+ours-noknow |
| 2 | mediastore | 3 | Facade | canonical | **canonical** | 'the Facade component delivers' | SWATTR+Artemis+ours+ours-noknow |
| 3 | mediastore | 6 | Facade | canonical | **canonical** | 'using the Facade component' | SWATTR+Artemis+ours+ours-noknow |
| 4 | mediastore | 7 | MediaManagement | canonical | **canonical** | 'called the MediaManagement component' | SWATTR+Artemis+ours+ours-noknow |
| 5 | mediastore | 8 | MediaManagement | canonical | **canonical** | 'The MediaManagement component coordinates' | SWATTR+Artemis+ours+ours-noknow |
| 6 | mediastore | 9 | MediaManagement | none | **pronoun_or_implicit** | 'Furthermore, it fetches audio files' - 'it' = the MediaManagement component of s8 | Artemis+ours+ours-noknow |
| 7 | mediastore | 11 | UserManagement | canonical | **canonical** | 'The UserManagement component answers' | SWATTR+Artemis+ours+ours-noknow |
| 8 | mediastore | 12 | UserDBAdapter | canonical | **canonical** | 'The UserDBAdapter component queries' | SWATTR+Artemis+ours+ours-noknow |
| 9 | mediastore | 13 | UserManagement | canonical | **canonical** | 'the UserManagement component implements' | SWATTR+Artemis+ours+ours-noknow |
| 10 | mediastore | 16 | TagWatermarking | canonical | **canonical** | 'watermarked by the TagWatermarking component' | SWATTR+Artemis+ours+ours-noknow |
| 11 | mediastore | 17 | TagWatermarking | canonical | **canonical** | 'from the TagWatermarking component' | SWATTR+Artemis+ours+ours-noknow |
| 12 | mediastore | 17 | MediaManagement | canonical | **canonical** | 'the MediaManagement component forwards' | SWATTR+Artemis+ours+ours-noknow |
| 13 | mediastore | 19 | Packaging | canonical | **canonical** | 'we provide the Packaging component' | SWATTR+Artemis+ours+ours-noknow |
| 14 | mediastore | 20 | Reencoding | alias | **alias** | 'The ReEncoder component converts' - catalog name is Reencoding; the doc never writes that form, it uses the agent noun | Artemis+ours |
| 15 | mediastore | 23 | DB | alias | **alias** | 'The Database component represents an actual database' - expansion of the catalog abbreviation DB | Artemis+ours |
| 16 | mediastore | 24 | DB | none | **pronoun_or_implicit** | 'It stores user information' - 'It' = the Database component of s23 | Artemis+ours |
| 17 | mediastore | 25 | DB | alias | **alias** | 'a query that is sent to the Database component' | Artemis+ours |
| 18 | mediastore | 25 | MediaAccess | none | **alias** ⚠ | 'AudioAccess creates a query' - document-only name; no AudioAccess exists in the catalog, the component is MediaAccess | Artemis+ours |
| 19 | mediastore | 26 | MediaAccess | canonical | **canonical** | 'the MediaAccess component stores it' | SWATTR+Artemis+ours+ours-noknow |
| 20 | mediastore | 27 | MediaAccess | canonical | **canonical** | 'The MediaAccess component encapsulates' | SWATTR+Artemis+ours+ours-noknow |
| 21 | mediastore | 28 | MediaAccess | none | **pronoun_or_implicit** | 'Furthermore, it fetches a list' - 'it' = the MediaAccess component of s27 | Artemis+ours+ours-noknow |
| 22 | mediastore | 29 | UserDBAdapter | canonical | **canonical** | 'the UserDBAdapter component provides' | SWATTR+Artemis+ours+ours-noknow |
| 23 | mediastore | 30 | UserDBAdapter | canonical | **canonical** | 'The UserDBAdapter component creates' | SWATTR+Artemis+ours+ours-noknow |
| 24 | mediastore | 31 | DB | alias | **alias** | 'The Database component then executes' | Artemis+ours |
| 25 | mediastore | 32 | DB | alias | **alias** | 'also stored in the Database component' | Artemis+ours |
| 26 | mediastore | 33 | DB | alias | **alias** | 'decouple the DataStorage from the database' - lower-case use of the same expansion | ours |
| 27 | mediastore | 33 | FileStorage | none | **alias** ⚠ | 'decouple the DataStorage' - document-only name for the FileStorage component | Artemis+ours |
| 28 | mediastore | 34 | DB | alias | **alias** | 'fetches the associated meta-data from the Database' | Artemis+ours |
| 29 | mediastore | 34 | MediaAccess | canonical | **canonical** | 'the MediaAccess component fetches' | SWATTR+Artemis+ours+ours-noknow |
| 30 | mediastore | 35 | FileStorage | none | **alias** ⚠ | 'retrieved from the DataStorage' | Artemis+ours |
| 31 | mediastore | 36 | FileStorage | none | **alias** ⚠ | 'stored in the DataStorage without any change' | Artemis+ours |
| 32 | teastore | 1 | Registry | canonical | **canonical** | 'a single Registry instance' | SWATTR+Artemis+ours+ours-noknow |
| 33 | teastore | 2 | WebUI | canonical | **canonical** | 'The WebUI service retrieves' | SWATTR+Artemis+ours+ours-noknow |
| 34 | teastore | 2 | ImageProvider | alias | **canonical** ⚠ | 'from the Image Provider' - the catalog name ImageProvider written with a space; normalisation alone resolves it | SWATTR+Artemis+ours |
| 35 | teastore | 3 | Auth | canonical | **canonical** | 'authenticated by the Auth service' | SWATTR+Artemis+ours+ours-noknow |
| 36 | teastore | 4 | Persistence | alias | **canonical** ⚠ | 'retrieved from the PersistenceProvider' - compound CONTAINING the catalog name Persistence | SWATTR+Artemis+ours |
| 37 | teastore | 4 | Recommender | canonical | **canonical** | 'from the Recommender service' | SWATTR+Artemis+ours+ours-noknow |
| 38 | teastore | 5 | WebUI | canonical | **canonical** | 'The WebUI provides the TeaStore front-end' | SWATTR+Artemis+ours+ours-noknow |
| 39 | teastore | 6 | WebUI | none | **pronoun_or_implicit** | 'It contains logic to save and retireve values from cookies' - 'It' = the WebUI of s5 | Artemis+ours+ours-noknow |
| 40 | teastore | 7 | WebUI | canonical | **canonical** | 'not provides by the WebUi' - case variant | SWATTR+Artemis+ours+ours-noknow |
| 41 | teastore | 7 | ImageProvider | alias | **canonical** ⚠ | 'from the Image Provider service' | SWATTR+Artemis+ours |
| 42 | teastore | 8 | WebUI | alias | **partial** ⚠ | 'The UI provides a status page' - only the second half of WebUI; note that teammates has a component actually NAMED UI | Artemis+ours |
| 43 | teastore | 10 | WebUI | canonical | **canonical** | 'delivers images to the WebUI' | SWATTR+ours+ours-noknow |
| 44 | teastore | 10 | ImageProvider | alias | **canonical** ⚠ | 'The Image Provider delivers images' | SWATTR+Artemis+ours |
| 45 | teastore | 11 | ImageProvider | none | **pronoun_or_implicit** | 'It matches the provided product ID' - 'It' = the Image Provider of s10 | Artemis+ours |
| 46 | teastore | 12 | ImageProvider | alias | **canonical** ⚠ | 'not available to the Image Provider' | SWATTR+Artemis+ours |
| 47 | teastore | 18 | Auth | canonical | **canonical** | 'The Auth service handles' | SWATTR+Artemis+ours+ours-noknow |
| 48 | teastore | 22 | Persistence | canonical | **canonical** | 'The Persistence service provides' | SWATTR+Artemis+ours+ours-noknow |
| 49 | teastore | 23 | Persistence | none | **pronoun_or_implicit** | 'It maps the relational entities' - 'It' = the Persistence service of s22 | Artemis+ours+ours-noknow |
| 50 | teastore | 24 | Persistence | none | **pronoun_or_implicit** | 'It features endpoints for general CRUD-Operations' | Artemis+ours+ours-noknow |
| 51 | teastore | 25 | Persistence | canonical | **canonical** | 'The persistence provider uses a second level entity cache' - lower-case spaced variant | SWATTR+Artemis+ours+ours-noknow |
| 52 | teastore | 26 | Persistence | none | **pronoun_or_implicit** | 'As such, it also acts as a caching layer' | Artemis+ours+ours-noknow |
| 53 | teastore | 27 | Recommender | canonical | **canonical** | 'The Recommender is used to generate' | SWATTR+Artemis+ours+ours-noknow |
| 54 | teastore | 28 | Recommender | none | **pronoun_or_implicit** | 'It is trained using all existing orders' | Artemis+ours+ours-noknow |
| 55 | teastore | 37 | Registry | canonical | **canonical** | 'The Registry provides information' | SWATTR+Artemis+ours+ours-noknow |
| 56 | teastore | 38 | Registry | canonical | **canonical** | 'register themselves at the registry on startup' | SWATTR+Artemis+ours+ours-noknow |
| 57 | teastore | 41 | Registry | canonical | **canonical** | 'uses one single registry' | SWATTR+Artemis+ours+ours-noknow |
| 58 | teastore | 43 | Registry | canonical | **canonical** | 'By limiting it to a single registry instance' | SWATTR+Artemis+ours+ours-noknow |
| 59 | teammates | 1 | UI | canonical | **canonical** | 'Architecture contains UI Component' | SWATTR+Artemis+ours+ours-noknow |
| 60 | teammates | 1 | Logic | canonical | **canonical** | 'Logic Component' | SWATTR+Artemis+ours+ours-noknow |
| 61 | teammates | 1 | Storage | canonical | **canonical** | 'Storage Component' | SWATTR+Artemis+ours+ours-noknow |
| 62 | teammates | 1 | Test Driver | canonical | **canonical** | 'Test Driver Component' | SWATTR+Artemis+ours+ours-noknow |
| 63 | teammates | 1 | E2E | canonical | **canonical** | 'E2E Component' | SWATTR+Artemis+ours+ours-noknow |
| 64 | teammates | 1 | Client | canonical | **canonical** | 'Client Component' | SWATTR+Artemis+ours+ours-noknow |
| 65 | teammates | 1 | Common | canonical | **canonical** | 'Common Component' | SWATTR+Artemis+ours+ours-noknow |
| 66 | teammates | 4 | UI | canonical | **canonical** | 'The UI Browser seen by users' - compound containing the name | SWATTR+Artemis+ours |
| 67 | teammates | 5 | UI | canonical | **canonical** | 'This UI is a single HTML page' | SWATTR+Artemis+ours+ours-noknow |
| 68 | teammates | 7 | UI | canonical | **canonical** | 'In the UI Server the entry point' | SWATTR+Artemis |
| 69 | teammates | 7 | Logic | canonical | **canonical** | 'the application back end logic' - the name word, used as a common noun | SWATTR+ours |
| 70 | teammates | 8 | Logic | canonical | **canonical** | 'The main logic of the application is in POJOs' - common-noun use of the name | SWATTR+ours |
| 71 | teammates | 9 | Storage | canonical | **canonical** | 'The storage layer of the application' - common-noun use of the name | SWATTR+ours |
| 72 | teammates | 9 | GAE Datastore | canonical | **canonical** | 'provided by GAE Datastore, a NoSQL database' | SWATTR+ours+ours-noknow |
| 73 | teammates | 10 | Test Driver | canonical | **canonical** | 'The following explains the use of the Test Driver' | SWATTR+Artemis+ours+ours-noknow |
| 74 | teammates | 15 | E2E | canonical | **canonical** | 'The E2E end-to-end component' | SWATTR+Artemis+ours+ours-noknow |
| 75 | teammates | 16 | E2E | canonical | **canonical** | 'Its primary function is for E2E tests' - the name occurs; the component-denoting subject is nevertheless the anaphor 'Its' | SWATTR+Artemis+ours+ours-noknow |
| 76 | teammates | 18 | Client | canonical | **canonical** | 'The Client component can connect' | SWATTR+Artemis+ours+ours-noknow |
| 77 | teammates | 19 | Client | none | **pronoun_or_implicit** | 'It is used for administrative purposes' - 'It' = the Client component of s18 | ours+ours-noknow |
| 78 | teammates | 20 | Common | canonical | **canonical** | 'The Common component contains utility code' | SWATTR+ours+ours-noknow |
| 79 | teammates | 25 | UI | canonical | **canonical** | 'the object structure of the UI component' | SWATTR+Artemis+ours+ours-noknow |
| 80 | teammates | 29 | UI | canonical | **canonical** | 'The UI component is the first stop' | SWATTR+Artemis+ours+ours-noknow |
| 81 | teammates | 47 | Logic | canonical | **canonical** | 'interacting with the Logic component as necessary' | SWATTR+Artemis+ours+ours-noknow |
| 82 | teammates | 68 | Logic | canonical | **canonical** | 'interacting with the Logic component as necessary' | SWATTR+Artemis+ours+ours-noknow |
| 83 | teammates | 77 | Logic | canonical | **canonical** | 'The Logic component handles the business logic' | SWATTR+Artemis+ours+ours-noknow |
| 84 | teammates | 78 | Logic | none | **pronoun_or_implicit** | 'In particular, it is responsible for the following' - 'it' = the Logic component of s77 | ours+ours-noknow |
| 85 | teammates | 81 | UI | canonical | **canonical** | 'received from the UI component' | SWATTR+Artemis+ours+ours-noknow |
| 86 | teammates | 85 | UI | canonical | **canonical** | 'to be accessed by the UI' | SWATTR+ours+ours-noknow |
| 87 | teammates | 87 | Logic | canonical | **canonical** | 'Logic API is represented by the classes Logic, GateKeeper' - the name also happens to be a class name here | SWATTR+Artemis+ours+ours-noknow |
| 88 | teammates | 88 | Logic | canonical | **canonical** | 'connects to the several Logic classes' - 'Logic' here names a Facade CLASS inside the component | SWATTR+Artemis+ours+ours-noknow |
| 89 | teammates | 88 | Storage | canonical | **canonical** | 'to access data from the Storage component' | SWATTR+Artemis+ours+ours-noknow |
| 90 | teammates | 97 | UI | canonical | **canonical** | 'The UI is expected to check access control' | SWATTR+Artemis+ours+ours-noknow |
| 91 | teammates | 97 | Logic | canonical | **canonical** | 'before calling a method in the Logic' | SWATTR+ours+ours-noknow |
| 92 | teammates | 101 | Storage | canonical | **canonical** | '(escalated from Storage level)' | SWATTR+ours+ours-noknow |
| 93 | teammates | 118 | Storage | canonical | **canonical** | 'The Storage component performs CRUD' | SWATTR+Artemis+ours+ours-noknow |
| 94 | teammates | 119 | Storage | none | **pronoun_or_implicit** | 'It contains minimal logic beyond what is directly relevant to CRUD' - 'It' = the Storage component of s118 | ours+ours-noknow |
| 95 | teammates | 120 | Storage | none | **pronoun_or_implicit** | 'In particular, it is reponsible for the following' | ours+ours-noknow |
| 96 | teammates | 122 | Logic | canonical | **canonical** | 'from the Logic component' | SWATTR+Artemis+ours+ours-noknow |
| 97 | teammates | 122 | GAE Datastore | alias | **partial** ⚠ | 'Hiding the complexities of datastore' - only the head word of the two-word name GAE Datastore | — |
| 98 | teammates | 123 | Storage | canonical | **canonical** | 'contained inside the Storage component' | SWATTR+Artemis+ours+ours-noknow |
| 99 | teammates | 128 | Storage | canonical | **canonical** | 'The Storage component does not perform' | SWATTR+Artemis+ours+ours-noknow |
| 100 | teammates | 129 | Logic | canonical | **canonical** | 'handled by the Logic component' | SWATTR+Artemis+ours+ours-noknow |
| 101 | teammates | 131 | Logic | canonical | **canonical** | 'to be accessed by the logic component' | SWATTR+ours+ours-noknow |
| 102 | teammates | 137 | GAE Datastore | canonical | **canonical** | 'act as the bridge to the GAE Datastore' | SWATTR+ours+ours-noknow |
| 103 | teammates | 138 | GAE Datastore | alias | **partial** ⚠ | 'until data is persisted in the datastore' - head word only | ours+ours-noknow |
| 104 | teammates | 141 | GAE Datastore | alias | **partial** ⚠ | "across all serves of the Google's distributed datastore" - head word plus a descriptive paraphrase of GAE | ours+ours-noknow |
| 105 | teammates | 155 | Common | canonical | **canonical** | 'The Common component contains common utilities' | SWATTR+Artemis+ours+ours-noknow |
| 106 | teammates | 163 | Test Driver | canonical | **canonical** | 'Test Driver can use the DataBundle' | SWATTR+Artemis+ours+ours-noknow |
| 107 | teammates | 168 | Test Driver | none | **pronoun_or_implicit** | 'This component automates the testing of TEAMMATES' - demonstrative shell; the component is identified only by the section (s169 'test.driver, test.cases') | Artemis+ours |
| 108 | teammates | 174 | Common | canonical | **canonical** | 'the datatransfer objects from the Common component' | SWATTR+Artemis+ours+ours-noknow |
| 109 | teammates | 175 | Common | canonical | **canonical** | 'the utility classes from the Common component' | SWATTR+Artemis+ours+ours-noknow |
| 110 | teammates | 176 | Logic | canonical | **canonical** | 'for testing the Logic component' | SWATTR+Artemis+ours+ours-noknow |
| 111 | teammates | 177 | Storage | canonical | **canonical** | 'for testing the Storage component' | SWATTR+Artemis+ours+ours-noknow |
| 112 | teammates | 185 | Logic | canonical | **canonical** | 'REST API calls for the back-end logic' - common-noun use of the name | SWATTR |
| 113 | teammates | 185 | E2E | canonical | **canonical** | 'The E2E component has no knowledge' | SWATTR+Artemis+ours+ours-noknow |
| 114 | teammates | 186 | E2E | canonical | **canonical** | 'Its primary function is for E2E tests and L&P tests' - name present, subject is the anaphor 'Its' | SWATTR+Artemis+ours+ours-noknow |
| 115 | teammates | 194 | Client | canonical | **canonical** | 'The Client component contains scripts' | SWATTR+Artemis+ours+ours-noknow |
| 116 | bigbluebutton | 4 | HTML5 Client | canonical | **canonical** | 'HTML5 client.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 117 | bigbluebutton | 5 | HTML5 Client | canonical | **canonical** | 'The HTML5 client is a single page, responsive web application' | SWATTR+Artemis+ours+ours-noknow |
| 118 | bigbluebutton | 6 | HTML5 Client | canonical | **canonical** | 'The HTML5 client connects directly' | SWATTR+Artemis+ours+ours-noknow |
| 119 | bigbluebutton | 6 | HTML5 Server | partial_d | **partial** | 'with the BigBlueButton server over port 443' - generic head noun only; the 'html5' the auto classifier matched belongs to the CLIENT mention in the same sentence | SWATTR+ours-noknow |
| 120 | bigbluebutton | 8 | HTML5 Server | canonical | **canonical** | 'The HTML5 server sits behind nginx' | SWATTR+Artemis+ours+ours-noknow |
| 121 | bigbluebutton | 9 | HTML5 Client | partial_d | **partial** | 'the state of each BigBlueButton client' - generic head noun; the sentence's 'HTML5' belongs to the SERVER mention | Artemis+ours+ours-noknow |
| 122 | bigbluebutton | 9 | HTML5 Server | canonical | **canonical** | 'The HTML5 server is built upon Meteor.js' | SWATTR+Artemis+ours+ours-noknow |
| 123 | bigbluebutton | 10 | HTML5 Client | partial_g | **partial** | 'each client connected to a meeting' - generic head noun only | ours+ours-noknow |
| 124 | bigbluebutton | 10 | HTML5 Server | partial_g | **partial** | 'all meetings on the server' - generic head noun only | SWATTR+ours+ours-noknow |
| 125 | bigbluebutton | 11 | HTML5 Client | partial_g | **partial** | "Each user's client is only aware of the their meeting's state" | ours+ours-noknow |
| 126 | bigbluebutton | 12 | HTML5 Client | partial_g | **partial** | 'The client side subscribes' | ours+ours-noknow |
| 127 | bigbluebutton | 12 | HTML5 Server | partial_g | **partial** | 'the published collections on the server side' | SWATTR+ours+ours-noknow |
| 128 | bigbluebutton | 13 | HTML5 Client | partial_g | **partial** | 'pushed to MiniMongo on the client side' | ours+ours-noknow |
| 129 | bigbluebutton | 13 | HTML5 Server | partial_g | **partial** | 'Updates to MongoDB on the server side' | SWATTR+ours+ours-noknow |
| 130 | bigbluebutton | 14 | HTML5 Client | canonical | **canonical** | 'the architecture of the HTML5 client' | SWATTR+Artemis+ours+ours-noknow |
| 131 | bigbluebutton | 15 | HTML5 Server | canonical | **canonical** | 'Scalability of HTML5 server component.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 132 | bigbluebutton | 19 | HTML5 Client | partial_d | **partial** | 'handling incoming messages from clients' - generic head noun, plural; the 'html5' in this sentence is inside 'bbb-html5', which names the SERVER | ours+ours-noknow |
| 133 | bigbluebutton | 19 | HTML5 Server | alias | **alias** | 'a single nodejs process for bbb-html5' - the deployment/package name, never the catalog form | Artemis+ours |
| 134 | bigbluebutton | 20 | HTML5 Server | alias | **alias** | 'bbb-html5 could use multiple CPU cores' | Artemis+ours |
| 135 | bigbluebutton | 21 | HTML5 Server | alias | **alias** | 'bbb-html5 uses 2 "frontend" and two "backend" processes' | Artemis+ours |
| 136 | bigbluebutton | 26 | Apps | canonical | **canonical** | 'send events to akka-apps' - compound containing the catalog name Apps | SWATTR+Artemis+ours+ours-noknow |
| 137 | bigbluebutton | 30 | BBB web | canonical | **canonical** | 'bbb-web splits the load in round-robin fashion' - hyphenated form of the two-word name BBB web | SWATTR+Artemis+ours+ours-noknow |
| 138 | bigbluebutton | 36 | BBB web | canonical | **canonical** | 'BBB web.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 139 | bigbluebutton | 37 | BBB web | alias | **alias** | 'BigBlueButton web application is a Java-based application' - expansion of the BBB abbreviation | Artemis+ours+ours-noknow |
| 140 | bigbluebutton | 38 | BBB web | none | **pronoun_or_implicit** | 'It implements the BigBlueButton API' - 'It' = the BigBlueButton web application of s37 | Artemis+ours |
| 141 | bigbluebutton | 39 | HTML5 Server | partial_g | **partial** | 'an endpoint to control the BigBlueButton server' - generic head noun; arguably the whole deployment rather than the HTML5 server component, so the gold link itself is debatable | SWATTR |
| 142 | bigbluebutton | 46 | Redis PubSub | canonical | **canonical** | 'Redis PubSub.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 143 | bigbluebutton | 47 | Redis PubSub | canonical | **canonical** | 'Redis PubSub provides a communication channel' | SWATTR+Artemis+ours+ours-noknow |
| 144 | bigbluebutton | 47 | HTML5 Server | partial_g | **partial** | 'running on the BigBlueButton server' - generic head noun; same debatable gold reading as s39 | SWATTR |
| 145 | bigbluebutton | 48 | Redis DB | canonical | **canonical** | 'Redis DB.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 146 | bigbluebutton | 49 | Redis DB | canonical | **canonical** | 'all events are stored in Redis DB' | SWATTR+Artemis+ours+ours-noknow |
| 147 | bigbluebutton | 51 | Apps | canonical | **canonical** | 'Apps akka.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 148 | bigbluebutton | 52 | Apps | canonical | **canonical** | 'BigBlueButton Apps is the main application' | SWATTR+Artemis+ours+ours-noknow |
| 149 | bigbluebutton | 53 | Apps | none | **pronoun_or_implicit** | 'It provides the list of users, chat, whiteboard' - 'It' = BigBlueButton Apps of s52 | Artemis+ours+ours-noknow |
| 150 | bigbluebutton | 54 | Apps | canonical | **canonical** | 'the different components of Apps Akka' | SWATTR+Artemis+ours+ours-noknow |
| 151 | bigbluebutton | 57 | FSESL | canonical | **canonical** | 'FSESL akka.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 152 | bigbluebutton | 58 | FreeSWITCH | canonical | **canonical** | 'the component that integrates with FreeSWITCH' | SWATTR+Artemis+ours+ours-noknow |
| 153 | bigbluebutton | 59 | FreeSWITCH | canonical | **canonical** | 'voice conference systems other than FreeSWITCH' | SWATTR+Artemis+ours+ours-noknow |
| 154 | bigbluebutton | 60 | Redis PubSub | canonical | **canonical** | 'uses messages through redis pubsub' | SWATTR+Artemis+ours+ours-noknow |
| 155 | bigbluebutton | 60 | FSESL | alias | **alias** | 'FreeSWITCH Event Socket Layer (fsels)' - the expansion; the parenthetical abbreviation is even mis-spelled ('fsels' vs FSESL), so the catalog form never appears | Artemis+ours |
| 156 | bigbluebutton | 60 | Apps | canonical | **canonical** | 'Communication between apps and FreeSWITCH' | SWATTR+Artemis+ours+ours-noknow |
| 157 | bigbluebutton | 61 | FreeSWITCH | canonical | **canonical** | 'FreeSWITCH.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 158 | bigbluebutton | 62 | FreeSWITCH | canonical | **canonical** | 'We think FreeSWITCH is an amazing piece of software' | SWATTR+Artemis+ours+ours-noknow |
| 159 | bigbluebutton | 63 | FreeSWITCH | canonical | **canonical** | 'FreeSWITCH provides the voice conferencing capability' | SWATTR+Artemis+ours+ours-noknow |
| 160 | bigbluebutton | 65 | WebRTC-SFU | partial_d | **partial** | 'by connecting using WebRTC' - the distinctive half of WebRTC-SFU, but here it denotes the PROTOCOL; the gold link to the component is debatable | SWATTR |
| 161 | bigbluebutton | 66 | FreeSWITCH | canonical | **canonical** | 'FreeSWITCH can also be integrated with VOIP providers' | SWATTR+Artemis+ours+ours-noknow |
| 162 | bigbluebutton | 67 | kurento | canonical | **canonical** | 'Kurento and WebRTC-SFU.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 163 | bigbluebutton | 67 | WebRTC-SFU | canonical | **canonical** | 'Kurento and WebRTC-SFU.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 164 | bigbluebutton | 68 | kurento | canonical | **canonical** | 'Kurento Media Server KMS is a media server' | SWATTR+Artemis+ours+ours-noknow |
| 165 | bigbluebutton | 69 | kurento | alias | **alias** | 'KMS is responsible for streaming of webcams' - the abbreviation introduced in s68 | ours+ours-noknow |
| 166 | bigbluebutton | 70 | WebRTC-SFU | canonical | **canonical** | 'The WebRTC-SFU acts as the media controller' | SWATTR+ours+ours-noknow |
| 167 | bigbluebutton | 72 | HTML5 Client | canonical | **canonical** | 'from the BigBlueButton HTML5 client' | Artemis+ours+ours-noknow |
| 168 | bigbluebutton | 72 | FreeSWITCH | canonical | **canonical** | 'the voice conference (running in FreeSWITCH)' | SWATTR+Artemis+ours+ours-noknow |
| 169 | bigbluebutton | 73 | HTML5 Client | partial_g | **partial** | 'the BigBlueButton client will make an audio connection' - generic head noun | Artemis+ours+ours-noknow |
| 170 | bigbluebutton | 73 | WebRTC-SFU | partial_d | **partial** | 'an audio connection to the server via WebRTC' - protocol reading again; debatable gold link | SWATTR |
| 171 | bigbluebutton | 73 | HTML5 Server | partial_g | **partial** | 'an audio connection to the server' | SWATTR |
| 172 | bigbluebutton | 76 | HTML5 Client | partial_g | **partial** | 'to be displayed inside the client' | ours+ours-noknow |
| 173 | bigbluebutton | 78 | BBB web | canonical | **canonical** | 'converted into scalable vector graphics (SVG) via bbb-web' | SWATTR+Artemis+ours |
| 174 | bigbluebutton | 79 | HTML5 Client | partial_g | **partial** | 'sends progress messages to the client' | ours+ours-noknow |
| 175 | bigbluebutton | 79 | Redis PubSub | canonical | **canonical** | 'through the Redis pubsub' | SWATTR+Artemis+ours+ours-noknow |
| 176 | bigbluebutton | 80 | Presentation Conversion | canonical | **canonical** | 'Presentation conversion flow.' - section heading | SWATTR+Artemis+ours+ours-noknow |
| 177 | bigbluebutton | 81 | Presentation Conversion | canonical | **canonical** | 'the flow of the presentation conversion' | SWATTR+Artemis+ours+ours-noknow |
| 178 | jabref | 1 | gui | canonical | **canonical** | 'towards the gui which is the outer shell' | SWATTR+Artemis+ours+ours-noknow |
| 179 | jabref | 1 | logic | canonical | **canonical** | 'the logic as an intermediate layer' | SWATTR+Artemis+ours+ours-noknow |
| 180 | jabref | 1 | model | canonical | **canonical** | 'with the model in the center' | SWATTR+Artemis+ours+ours-noknow |
| 181 | jabref | 2 | cli | canonical | **canonical** | 'utility packages for preferences and the cli' | SWATTR+Artemis+ours+ours-noknow |
| 182 | jabref | 2 | preferences | canonical | **canonical** | 'utility packages for preferences' | SWATTR+Artemis+ours+ours-noknow |
| 183 | jabref | 4 | gui | canonical | **canonical** | '(between logic, model, and gui)' | SWATTR+Artemis+ours+ours-noknow |
| 184 | jabref | 4 | logic | canonical | **canonical** | '(between logic, model, and gui)' | SWATTR+Artemis+ours+ours-noknow |
| 185 | jabref | 4 | model | canonical | **canonical** | '(between logic, model, and gui)' | SWATTR+Artemis+ours+ours-noknow |
| 186 | jabref | 5 | model | canonical | **canonical** | 'The model represents the most important data structures' | SWATTR+Artemis+ours+ours-noknow |
| 187 | jabref | 6 | gui | canonical | **canonical** | 'an API the gui can call and use' | SWATTR+Artemis+ours+ours-noknow |
| 188 | jabref | 6 | logic | canonical | **canonical** | 'The logic is responsible for reading/writing' | SWATTR+Artemis+ours+ours-noknow |
| 189 | jabref | 6 | model | canonical | **canonical** | 'manipulating the model' | SWATTR+Artemis+ours+ours-noknow |
| 190 | jabref | 7 | gui | canonical | **canonical** | 'Only the gui knows the user' | SWATTR+Artemis+ours+ours-noknow |
| 191 | jabref | 9 | logic | canonical | **canonical** | 'the logic should only depend on model classes' | SWATTR+Artemis+ours+ours-noknow |
| 192 | jabref | 9 | model | canonical | **canonical** | 'The model should have no dependencies' | SWATTR+Artemis+ours+ours-noknow |
| 193 | jabref | 10 | cli | canonical | **canonical** | 'The cli package bundles classes' | SWATTR+Artemis+ours+ours-noknow |
| 194 | jabref | 11 | preferences | canonical | **canonical** | 'The preferences represents all information customizable by a user' | SWATTR+Artemis+ours+ours-noknow |
| 195 | jabref | 12 | model | canonical | **canonical** | 'publish events from the model to the other layers' | SWATTR+Artemis+ours+ours-noknow |

## Disagreements with the automatic classifier

13 of 195 links (6.7%) are labelled differently by
hand. (Comparison is at the collapsed level, so `partial_d` vs `partial_g`
mistakes are *not* counted here even where the automatic label matched the right
category for the wrong reason — those are noted in the evidence column instead.)

| project | s | component | auto | manual | why the automatic label is wrong |
|---|---:|---|---|---|---|
| mediastore | 25 | MediaAccess | none | **alias** | 'AudioAccess creates a query' - document-only name; no AudioAccess exists in the catalog, the component is MediaAccess |
| mediastore | 33 | FileStorage | none | **alias** | 'decouple the DataStorage' - document-only name for the FileStorage component |
| mediastore | 35 | FileStorage | none | **alias** | 'retrieved from the DataStorage' |
| mediastore | 36 | FileStorage | none | **alias** | 'stored in the DataStorage without any change' |
| teastore | 2 | ImageProvider | alias | **canonical** | 'from the Image Provider' - the catalog name ImageProvider written with a space; normalisation alone resolves it |
| teastore | 4 | Persistence | alias | **canonical** | 'retrieved from the PersistenceProvider' - compound CONTAINING the catalog name Persistence |
| teastore | 7 | ImageProvider | alias | **canonical** | 'from the Image Provider service' |
| teastore | 8 | WebUI | alias | **partial** | 'The UI provides a status page' - only the second half of WebUI; note that teammates has a component actually NAMED UI |
| teastore | 10 | ImageProvider | alias | **canonical** | 'The Image Provider delivers images' |
| teastore | 12 | ImageProvider | alias | **canonical** | 'not available to the Image Provider' |
| teammates | 122 | GAE Datastore | alias | **partial** | 'Hiding the complexities of datastore' - only the head word of the two-word name GAE Datastore |
| teammates | 138 | GAE Datastore | alias | **partial** | 'until data is persisted in the datastore' - head word only |
| teammates | 141 | GAE Datastore | alias | **partial** | "across all serves of the Google's distributed datastore" - head word plus a descriptive paraphrase of GAE |

Net effect on the headline split:

| direction | links |
|---|---:|
| `alias` → `canonical` | 5 |
| `pronoun_or_implicit` → `alias` | 4 |
| `alias` → `partial` | 4 |

## Attribution leaks

A *leak* is a link our pipeline emitted through a proposal form that does not
match the reference form actually in the sentence. Expected pairing: `canonical`
and `alias` → `full_name` (the alias table feeds the full-name scan), `partial` →
`partial_name`, `pronoun_or_implicit` → `coreference`.

6 of the 186 emitted links leak (3.2%).

| project | s | component | true form | emitted as | runs | evidence |
|---|---:|---|---|---|---:|---|
| teastore | 8 | WebUI | `partial` | `full_name` | 3/3 | 'The UI provides a status page' - only the second half of WebUI; note that teammates has a component actually NAMED UI |
| teammates | 138 | GAE Datastore | `partial` | `full_name` | 3/3 | 'until data is persisted in the datastore' - head word only |
| teammates | 141 | GAE Datastore | `partial` | `full_name` | 3/3 | "across all serves of the Google's distributed datastore" - head word plus a descriptive paraphrase of GAE |
| teammates | 168 | Test Driver | `pronoun_or_implicit` | `partial_name` | 2/3 | 'This component automates the testing of TEAMMATES' - demonstrative shell; the component is identified only by the section (s169 'test.driver, test.cases') |
| bigbluebutton | 30 | BBB web | `canonical` | `partial_name` | 2/3 | 'bbb-web splits the load in round-robin fashion' - hyphenated form of the two-word name BBB web |
| bigbluebutton | 78 | BBB web | `canonical` | `partial_name` | 3/3 | 'converted into scalable vector graphics (SVG) via bbb-web' |

| leak direction | links |
|---|---:|
| true `partial` emitted as `full_name` | 3 |
| true `canonical` emitted as `partial_name` | 2 |
| true `pronoun_or_implicit` emitted as `partial_name` | 1 |

**Characterisation.** The six leaks are not random; they are three mechanisms.

1. *The alias table absorbs partial references.* teastore s8 ("The UI" for
   `WebUI`) and teammates s138/s141 ("the datastore" for `GAE Datastore`) carry
   only part of the name, but the run's discovered alias table binds that part to
   the component, so the full-name scan fires and the link is booked as
   `full_name`. The alias table is therefore doing partial-reference resolution
   while being credited as name matching. Switching the table off does not leave
   these links alone: teastore s8 drops to 1/3 runs without it. An alias-table
   ablation is therefore partly measuring partial-reference handling.
2. *Hyphenated names fall to the partial scan.* bigbluebutton s30 and s78 write
   `BBB web` as "bbb-web". The full-name scan wants the catalog form, misses, and
   the partial scan picks up the fragment — so a plain name mention is booked as
   `partial_name`. Full-name coverage is understated for hyphenated names by
   exactly as much as partial coverage is overstated.
3. *One right-answer-wrong-reason.* teammates s168 ("This component automates the
   testing of TEAMMATES") contains no form of `Test Driver` at all; it is emitted
   by the partial scan, which can only have fired on "testing". The coreference
   scan, which is the component that is supposed to own implicit references,
   did not claim it.

The practical consequence: **per-proposal-form contribution numbers cannot be read
as per-reference-form numbers.** On these 195 links the `full_name` proposal form
is credited with 3 links it did not lexically earn and the `partial_name` form
with 3 more, while 7 of the 22 true partial references are not emitted
at all.

Cross-check — the full confusion between true form and emitted proposal form:

| true form \ emitted | full_name | partial_name | coreference | not emitted |
|---|---:|---:|---:|---:|
| `canonical` | 136 | 2 | 0 | 2 |
| `alias` | 17 | 0 | 0 | 0 |
| `partial` | 3 | 12 | 0 | 7 |
| `pronoun_or_implicit` | 0 | 1 | 15 | 0 |

## Threats to Challenge 1

**What holds.** Every one of the four reference forms the paper names does occur
in the gold standard. 55 of 195 gold links (28.2%) are carried by
something other than the catalog name: 17 alias, 22 partial,
16 pronoun/implicit. A name-matching linker cannot reach them, and
the measured recall table above shows the gap is real, not hypothetical.

**What does not hold: generality.** The variety is not spread over the benchmark.

- `alias` is a *mediastore* phenomenon: 11/17 alias links come from
  mediastore, and they come from four naming decisions in one document
  (`Database`/`DB`, `DataStorage`/`FileStorage`, `AudioAccess`/`MediaAccess`,
  `ReEncoder`/`Reencoding`). Remove that one SAD and the alias category is a
  handful of abbreviation expansions.
- `partial` is a *bigbluebutton* phenomenon: 18/22 partial links come from
  bigbluebutton, and 16 of those are the two components `HTML5 Client` and
  `HTML5 Server` being referred to as "the client" and "the server". That is one
  document's habit of dropping a shared qualifier, not a general property of
  architecture documentation.
- `pronoun_or_implicit` is the only form that is genuinely spread: it occurs in
  4 of 5 projects, but it is also the smallest
  category at 16 links, and it is almost entirely "It …" in the sentence
  immediately after the one that introduces the component.
- *jabref* contributes **zero** non-canonical links (18/18 canonical). A
  reader of Challenge 1 would not guess that one of the five projects exercises
  none of the challenge.

**What must not be claimed.**

1. Do not claim the four forms are *balanced* or *each common*. The corpus is
   71.8% plain catalog-name mentions; the whole challenge rests on
   55 links.
2. Do not claim per-form recall differences generalise. `alias` recall is
   measured on 17 links from effectively one document, and `partial` recall on 22
   links of which most are one project's two components. Per-form recall numbers
   are descriptive of these five documents, nothing more.
3. Do not present the alias table's contribution as a general mechanism on this
   evidence. The ours-vs-ours-noknow difference in the recall table is dominated
   by the same mediastore/bigbluebutton naming decisions.
4. Do not lean on the partial category without saying that some of its links are
   questionable gold: `WebRTC` at bigbluebutton s65/s73 denotes the *protocol*,
   and "the BigBlueButton server" at s39/s47 arguably denotes the deployment, not
   the `HTML5 Server` component.
5. Do not quote the automatic surface-form split. It is wrong on
   13/195 links, in both directions: it over-reports `alias` (spacing and
   compounding variants such as "Image Provider" and "PersistenceProvider" are
   the name, not an alias) and it under-reports it (a genuine alias is filed as
   `none` when run 1 happened not to discover it).
6. Do not describe the partial-form misses as links the pipeline never proposes.
   All three misses in the weakest cell were proposed by the partial-name scan
   and rejected by the name judge — see *Where the partial-form misses happen*.
   The cell measures judge calibration on generic head nouns, not scan coverage.
7. Do not present the alias table as purely additive on this form. Ours and
   ours-noknow both recover 15 of 22 partial links, but the table trades
   teastore s8 for bigbluebutton s6.

**Safe formulation.** Challenge 1 is supported as an *existence* claim — all four
forms occur, and the non-canonical quarter of the gold standard is where lexical
baselines lose recall — provided the paper says where the variety comes from
instead of implying it is uniform across the benchmark.
