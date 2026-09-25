# CH2 type-2 false positives: a one-by-one error analysis

**Type 2** = a false positive on a sentence that carries **no surface form of the
wrongly linked component** — no catalog name, no document alias, not even a word of
the name. It is the failure mode `approach.tex` does *not* name: Challenge 2 as
written describes a lexical trap (a known form is present but denotes something
else), which is type 1.

Population derived by `audit.py --sheets` (`ch2_mode2_sheet.csv`, 106 items over 5
projects, pooled across 3 runs). Verdicts recorded item-by-item in
`ch2_mode2_annotations.py`; this file only joins and counts them. Each verdict was
reached by reading the sentence and its neighbours in the SAD.

Counts below are DISTINCT items, not per-run rates.

## Who produces type-2 errors

| producer | type-2 items |
|---|---:|
| Artemis (LLM baseline, ≥2 of 3 runs) | 35 |
| SWATTR (lexical baseline) | 1 |
| our scans proposed it (either judge saw it ≥2 runs) | 55 |
| our pipeline EMITTED it (≥2 of 3 runs) | 1 |

## Sub-modes

| sub-mode | items | share |
|---|---:|---:|
| `a` topic continuation (named in an adjacent sentence) | 46 | 43.4% |
| `b` long-range topical drift | 25 | 23.6% |
| `d` sibling / parent / child confusion | 14 | 13.2% |
| `e` role-word confusion (a document word, not the component) | 11 | 10.4% |
| `g` CLASSIFIER ARTIFACT -- a surface form is present; really a type-1 FP | 8 | 7.5% |
| `f` component carries no gold link anywhere | 2 | 1.9% |
| **total** | **106** | |

## Is the gold standard, not the system, what is wrong?

| the sentence does describe that component's responsibility | items | share |
|---|---:|---:|
| yes | 23 | 21.7% |
| borderline | 24 | 22.6% |
| no | 59 | 55.7% |

Cross-tabulated by producer:

| producer | gold-silent `yes` | `borderline` | `no` |
|---|---:|---:|---:|
| Artemis | 20 | 5 | 10 |
| our scans (pre-judge) | 4 | 19 | 32 |
| our final output | 0 | 1 | 0 |

## Per-project

| project | items | dominant sub-mode | gold-silent `yes` |
|---|---:|---|---:|
| mediastore | 9 | `d` (5) | 0 |
| teastore | 26 | `a` (20) | 19 |
| teammates | 39 | `b` (19) | 3 |
| bigbluebutton | 28 | `e` (11) | 1 |
| jabref | 4 | `a` (4) | 0 |

## Every item

`dist` = sentences between this one and the nearest sentence the gold DOES link to
that component (blank = the component has no gold link anywhere).

| # | project | s | wrongly linked to | dist | sub-mode | gold-silent | produced by | why it is wrong |
|---:|---|---:|---|---:|---|---|---|---|
| 1 | bigbluebutton | 7 | HTML5 Client | 1 | `a` | no | corefJudge 0/3 | sentence is about nginx; the client is named in s6 |
| 2 | bigbluebutton | 16 | HTML5 Server | 1 | `a` | borderline | Artemis(1/3) | s15 is the heading 'Scalability of HTML5 server component'; s16 describes its 2.2 history |
| 3 | bigbluebutton | 17 | HTML5 Server | 2 | `a` | borderline | Artemis(1/3), corefJudge 0/3 | same section; describes the nodejs bottleneck of that server |
| 4 | bigbluebutton | 23 | HTML5 Client | 4 | `e` | no | corefJudge 0/3 | 'front-end/back-end' here are nodejs process roles inside the HTML5 server |
| 5 | bigbluebutton | 24 | HTML5 Client | 5 | `e` | no | corefJudge 0/3 | 'Frontends' = nodejs frontend processes, not the HTML5 Client component |
| 6 | bigbluebutton | 25 | HTML5 Client | 6 | `e` | no | corefJudge 0/3 | 'Frontends collect subscriptions' |
| 7 | bigbluebutton | 26 | HTML5 Client | 7 | `e` | no | corefJudge 0/3 | 'Frontends receive other DDP events'; gold here is Apps |
| 8 | bigbluebutton | 27 | HTML5 Client | 8 | `e` | no | corefJudge 0/3 | 'Frontends handle the Streamer redis events' |
| 9 | bigbluebutton | 28 | HTML5 Client | 9 | `e` | no | corefJudge 0/3 | 'Frontends still require MeetingStarted' |
| 10 | bigbluebutton | 29 | HTML5 Client | 10 | `e` | no | corefJudge 0/3 | 'Backends handle all the non-streamer events' |
| 11 | bigbluebutton | 30 | kurento | 37 | `b` | no | Artemis | sentence is bbb-web load splitting; kurento is 37 sentences away |
| 12 | bigbluebutton | 31 | Apps | 5 | `e` | no | corefJudge 0/3 | 'individual backends' = process roles |
| 13 | bigbluebutton | 31 | BBB web | 1 | `a` | no | corefJudge 0/1 | bbb-web named in s30; this sentence is about backends |
| 14 | bigbluebutton | 32 | Apps | 6 | `e` | no | corefJudge 0/3 | 'passed to backends as well' |
| 15 | bigbluebutton | 32 | HTML5 Client | 13 | `e` | no | corefJudge 0/2 | 'no frontends' |
| 16 | bigbluebutton | 41 | BBB web | 3 | `b` | no | corefJudge 0/1 | third-party integrations, not the web application |
| 17 | bigbluebutton | 42 | BBB web | 4 | `e` | no | corefJudge 0/1 | 'its own front-end called Greenlight' is a third-party portal |
| 18 | bigbluebutton | 42 | WebRTC-SFU | 23 | `b` | no | Artemis | unrelated; 23 sentences from the nearest SFU sentence |
| 19 | bigbluebutton | 43 | Recording Service | — | `f` | no | nameJudge 0/3 | Recording Service carries no gold link anywhere; the NAME scan proposed it with no surface form |
| 20 | bigbluebutton | 44 | BBB web | 6 | `b` | no | corefJudge 0/1 | 'simple API demos' |
| 21 | bigbluebutton | 45 | BBB web | 7 | `b` | borderline | corefJudge 0/2 | 'they all use the API under the hood' -- the API is bbb-web's |
| 22 | bigbluebutton | 49 | Recording Service | — | `f` | borderline | nameJudge 0/3 | 'when a meeting is recorded'; the component is never in gold |
| 23 | bigbluebutton | 53 | Presentation Conversion | 27 | `g` | no | Artemis, nameJudge 0/3 | 'presentations' is the plural of the name's head word; the tokenizer required the exact form |
| 24 | bigbluebutton | 56 | Apps | 2 | `a` | no | corefJudge 0/3 | MeetingActor named in s55 |
| 25 | bigbluebutton | 57 | FreeSWITCH | 1 | `d` | no | Artemis(1/3) | sibling of FSESL, which is what s57 heads |
| 26 | bigbluebutton | 58 | FSESL | 1 | `d` | yes | Artemis, corefJudge 0/3 | 'the component that integrates with FreeSWITCH' IS FSESL; gold assigns FreeSWITCH |
| 27 | bigbluebutton | 59 | FSESL | 1 | `d` | borderline | Artemis, corefJudge 0/3 | about replacing FreeSWITCH; gold assigns FreeSWITCH |
| 28 | bigbluebutton | 78 | Presentation Conversion | 2 | `d` | borderline | Artemis | PDF->SVG conversion is presentation conversion; gold assigns bbb-web |
| 29 | jabref | 3 | model | 1 | `a` | borderline | Artemis, corefJudge 2/3, **SURVIVED 2/3** | layer dependency direction; OUR ONLY SURVIVING TYPE-2 FP |
| 30 | jabref | 8 | gui | 1 | `a` | no | Artemis, corefJudge 0/1 | 'for each layer we form packages'; gui named in s7 |
| 31 | jabref | 8 | logic | 1 | `a` | no | Artemis | same sentence, same reason |
| 32 | jabref | 8 | model | 1 | `a` | no | Artemis, corefJudge 0/1 | same sentence, same reason |
| 33 | mediastore | 8 | UserManagement | 3 | `b` | no | corefJudge 0/1 | sentence is about MediaManagement coordinating others |
| 34 | mediastore | 15 | MediaManagement | 2 | `a` | no | corefJudge 0/3 | 'the requested files are first reencoded' |
| 35 | mediastore | 16 | AudioWatermarking | — | `d` | no | Artemis | sibling of TagWatermarking; never linked in gold |
| 36 | mediastore | 16 | ParallelWatermarking | — | `d` | no | Artemis | sibling of TagWatermarking; never linked in gold |
| 37 | mediastore | 17 | AudioWatermarking | — | `d` | no | Artemis | sibling of TagWatermarking |
| 38 | mediastore | 17 | ParallelWatermarking | — | `d` | no | Artemis | sibling of TagWatermarking |
| 39 | mediastore | 21 | Reencoding | 1 | `a` | borderline | Artemis, corefJudge 0/3 | 'this can result in reduction of file sizes' continues the ReEncoder description |
| 40 | mediastore | 26 | FileStorage | 7 | `d` | borderline | corefJudge 0/1 | 'stores it at the predefined location'; gold assigns MediaAccess |
| 41 | mediastore | 37 | Reencoding | 17 | `g` | no | SWATTR | 're-encoding' is the hyphenated form of the catalog name -- SWATTR's only type-2 item |
| 42 | teammates | 3 | UI | 1 | `a` | no | corefJudge 0/3 | 'overview of the main components' |
| 43 | teammates | 11 | Test Driver | 1 | `a` | borderline | nameJudge 0/3 | s10 heads 'the use of the Test Driver'; s11 is that section |
| 44 | teammates | 13 | Test Driver | 3 | `a` | borderline | nameJudge 0/3 | testing-framework list under the Test Driver section |
| 45 | teammates | 14 | Test Driver | 4 | `a` | borderline | nameJudge 0/3 | same section |
| 46 | teammates | 16 | Test Driver | 6 | `d` | no | nameJudge 0/3 | gold assigns E2E, a sibling test component |
| 47 | teammates | 17 | Test Driver | 7 | `d` | borderline | nameJudge 0/3 | Selenium E2E automation; gold assigns E2E |
| 48 | teammates | 21 | Common | 1 | `a` | no | corefJudge 0/1 | 'the diagram below shows how code is organized' |
| 49 | teammates | 27 | UI | 2 | `a` | borderline | corefJudge 0/3 | 'written in Angular' continues the ui.website description |
| 50 | teammates | 30 | UI | 1 | `a` | no | corefJudge 0/2 | 'such a request will go through the following steps' |
| 51 | teammates | 36 | Client | 17 | `b` | no | corefJudge 0/2 | Client is the admin CLI; this is browser traffic, 17 sentences away |
| 52 | teammates | 36 | UI | 7 | `b` | borderline | corefJudge 0/1 | user requests from the web browser |
| 53 | teammates | 38 | UI | 9 | `b` | no | corefJudge 0/1 | 'the initial request for the web page' |
| 54 | teammates | 40 | UI | 11 | `b` | borderline | corefJudge 0/3 | 'WebPageServlet returns the built single web page' |
| 55 | teammates | 41 | Client | 22 | `b` | no | corefJudge 0/1 | browser rendering, not the admin Client |
| 56 | teammates | 42 | GAE Datastore | 33 | `b` | no | corefJudge 0/1 | AJAX request processing, 33 sentences from the nearest datastore sentence |
| 57 | teammates | 45 | Logic | 2 | `a` | no | corefJudge 0/1 | 'WebApiServlet executes the action' |
| 58 | teammates | 47 | UI | 18 | `b` | no | Artemis(1/3) | gold assigns Logic; the sentence names Logic |
| 59 | teammates | 48 | Logic | 1 | `a` | no | corefJudge 0/1 | 'the Action packages the result' |
| 60 | teammates | 50 | Client | 31 | `b` | no | corefJudge 0/1 | 'sends the result back to the browser' |
| 61 | teammates | 52 | UI | 23 | `b` | no | corefJudge 0/3 | 'the Web API is protected by two layers' |
| 62 | teammates | 53 | UI | 24 | `b` | no | corefJudge 0/3 | 'origin check, authentication and authorization' |
| 63 | teammates | 55 | UI | 26 | `b` | no | corefJudge 0/3 | access-control description |
| 64 | teammates | 56 | Test Driver | 46 | `b` | no | nameJudge 0/3 | 'typically for testing purpose' 46 sentences away |
| 65 | teammates | 59 | GAE Datastore | 50 | `b` | no | corefJudge 0/3 | 'this type of request will be processed as follows' |
| 66 | teammates | 64 | Test Driver | 54 | `b` | no | nameJudge 0/3 | 'useful in testing the actions' 54 sentences away |
| 67 | teammates | 67 | Logic | 1 | `a` | no | corefJudge 0/1 | 'automatedServlet executes the action' |
| 68 | teammates | 72 | GAE Datastore | 50 | `b` | no | corefJudge 0/2 | 'configured in cron.xml' |
| 69 | teammates | 75 | GAE Datastore | 47 | `b` | no | corefJudge 0/2 | 'configured in queue.xml' |
| 70 | teammates | 92 | Logic | 4 | `a` | borderline | corefJudge 0/3 | EmailSender is a Logic class; gold silent on the elaboration |
| 71 | teammates | 94 | Logic | 3 | `a` | borderline | corefJudge 0/3 | TaskQueuer is a Logic class |
| 72 | teammates | 96 | Logic | 1 | `a` | yes | corefJudge 0/3 | 'this component provides methods to perform access control' refers to Logic |
| 73 | teammates | 121 | Storage | 1 | `a` | yes | corefJudge 0/2 | a responsibility bullet under the Storage section |
| 74 | teammates | 122 | Storage | 1 | `a` | yes | Artemis(1/3) | 'hiding the complexities of datastore from the Logic component' is Storage's own responsibility |
| 75 | teammates | 137 | Storage | 9 | `d` | borderline | corefJudge 0/3 | the Db classes are Storage; gold assigns GAE Datastore |
| 76 | teammates | 182 | Test Driver | 14 | `b` | borderline | nameJudge 0/3 | 'front-end files are tested with Jest' |
| 77 | teammates | 184 | Test Driver | 16 | `b` | borderline | nameJudge 0/3 | 'how TEAMMATES testing maps to standard types' |
| 78 | teammates | 186 | Test Driver | 18 | `d` | no | nameJudge 0/3 | gold assigns E2E |
| 79 | teammates | 188 | Test Driver | 20 | `d` | borderline | nameJudge 0/3 | 'e2e.util helpers for running E2E tests' |
| 80 | teammates | 193 | Test Driver | 25 | `b` | borderline | nameJudge 0/3 | 'load and performance tests' |
| 81 | teastore | 4 | WebUI | 1 | `a` | borderline | Artemis | the implicit retriever is the WebUI; gold assigns Persistence and Recommender |
| 82 | teastore | 9 | WebUI | 1 | `a` | yes | Artemis | 'the status view' is the status page the UI provides in s8 |
| 83 | teastore | 13 | ImageProvider | 1 | `a` | yes | Artemis | image lookup fallback -- an Image Provider responsibility |
| 84 | teastore | 14 | ImageProvider | 2 | `a` | yes | Artemis | image scaling -- an Image Provider responsibility |
| 85 | teastore | 15 | ImageProvider | 3 | `a` | yes | Artemis | 'the scaled image is stored for later use' |
| 86 | teastore | 16 | ImageProvider | 4 | `a` | yes | Artemis | the LFU image cache |
| 87 | teastore | 17 | ImageProvider | 5 | `a` | yes | Artemis | cache lookup before loading from disk |
| 88 | teastore | 19 | Auth | 1 | `a` | yes | Artemis | 'passwords are hashed using BCrypt' -- an Auth responsibility |
| 89 | teastore | 20 | Auth | 2 | `a` | yes | Artemis | session validation via SessionBlob |
| 90 | teastore | 21 | Auth | 3 | `a` | yes | Artemis | session-tampering check |
| 91 | teastore | 29 | Recommender | 1 | `a` | yes | Artemis | how recommendations are generated |
| 92 | teastore | 30 | Recommender | 2 | `a` | yes | Artemis | item rating basis |
| 93 | teastore | 31 | Recommender | 3 | `a` | yes | Artemis | fallback algorithm for unknown users |
| 94 | teastore | 32 | Recommender | 4 | `a` | yes | Artemis | Slope One collaborative filtering |
| 95 | teastore | 32 | SlopeOneRecommender | — | `g` | no | corefJudge 0/3 | 'Slope One' IS written; the tokenizer does not split the camelCase catalog name |
| 96 | teastore | 33 | Recommender | 5 | `a` | yes | Artemis | 'we implemented two versions of the algorithm' |
| 97 | teastore | 33 | SlopeOneRecommender | — | `g` | no | corefJudge 0/3 | camelCase catalog name vs 'Slope One' in the text |
| 98 | teastore | 34 | PreprocessedSlopeOneRecommender | — | `g` | no | corefJudge 0/1 | the two versions described ARE the preprocessed/on-the-go variants |
| 99 | teastore | 34 | Recommender | 6 | `a` | yes | Artemis | CPU- vs memory-intensive variants |
| 100 | teastore | 34 | SlopeOneRecommender | — | `g` | no | corefJudge 0/1 | camelCase catalog name |
| 101 | teastore | 35 | OrderBasedRecommender | — | `g` | no | corefJudge 0/3 | 'order-based nearest-neighbor approach' IS the name, hyphenated and split |
| 102 | teastore | 35 | Recommender | 7 | `a` | yes | Artemis | the order-based variant |
| 103 | teastore | 36 | OrderBasedRecommender | — | `g` | no | corefJudge 0/3 | 'its recommendation time' continues the order-based description |
| 104 | teastore | 36 | Recommender | 8 | `a` | yes | Artemis | recommendation timing |
| 105 | teastore | 39 | Registry | 1 | `a` | yes | Artemis, corefJudge 0/1 | 'services send a heartbeat by re-registering' -- a Registry responsibility |
| 106 | teastore | 40 | Registry | 1 | `a` | yes | Artemis, corefJudge 0/3 | heartbeat timeout handling |

## The items that survived our judges

- **jabref s3 → model** (`a`, gold-silent: borderline) — Artemis, corefJudge 2/3, **SURVIVED 2/3**
  - > The dependencies are only directed towards the center.
  - layer dependency direction; OUR ONLY SURVIVING TYPE-2 FP

## Exhibit: the cleanest type-2 case

Selection is a human judgement (unambiguous, zero surface trace, reproducible,
baseline-only); every fact below is computed from the data.

### PRIMARY — bigbluebutton s40-45, wrongly linked to `WebRTC-SFU`

- `s40` **[gold: —]** Every access to BigBlueButton comes through a front-end portal (we refer to as a third-party application).
- `s41` **[gold: —]** BigBlueButton integrates Moodle, Wordpress, Canvas, Sakai, and others (see third-party integrations).
- `s42` **[gold: —]** BigBlueButton comes with its own front-end called Greenlight.
- `s43` **[gold: —]** When using a learning management system (LMS) such as Moodle, teachers can setup BigBlueButton rooms within their course and students can access the rooms and their recordings.
- `s44` **[gold: —]** The BigBlueButton comes with some simple API demos.
- `s45` **[gold: —]** Regardless of which front-end you use, they all use the API under the hood.

- surface trace of `WebRTC-SFU` in the passage, including stems: **none at all**
- `WebRTC-SFU` is gold-linked at sentences [65, 67, 70, 73] — 25+ sentences away
- gold links anywhere in this passage: **0**
- Artemis, the three runs: [['s42->WebRTC-SFU'], ['s42->WebRTC-SFU'], ['s42->WebRTC-SFU']]
- ours, the three runs: [[], [], []]
- SWATTR: []

### RUNNER-UP — bigbluebutton s29-31, wrongly linked to `kurento`

- `s29` **[gold: —]** Backends handle all the non-streamer events.
- `s30` **[gold: BBB web]** If more than one backend is running, bbb-web splits the load in round-robin fashion by assigning an instanceId.
- `s31` **[gold: —]** So individual backends only process redis events for the meetings matching the associated instanceId.

- surface trace of `kurento` in the passage, including stems: **none at all**
- `kurento` is gold-linked at sentences [67, 68, 69] — 38+ sentences away
- gold links anywhere in this passage: **1**
- Artemis, the three runs: [['s30->BBB web', 's30->kurento'], ['s30->BBB web', 's30->kurento'], ['s30->BBB web', 's30->kurento']]
- ours, the three runs: [['s30->BBB web'], ['s30->BBB web'], []]
- SWATTR: ['s30->BBB web']

## Classifier artifacts

8 of the 106 items are not really type 2: a surface form of the
component IS present and the automatic tokenizer missed it. These should be counted
as type-1 (lexical) false positives, which makes the lexical mode slightly larger and
the inference mode slightly smaller than the automatic pass reported.

- bigbluebutton s53 → Presentation Conversion: 'presentations' is the plural of the name's head word; the tokenizer required the exact form
- mediastore s37 → Reencoding: 're-encoding' is the hyphenated form of the catalog name -- SWATTR's only type-2 item
- teastore s32 → SlopeOneRecommender: 'Slope One' IS written; the tokenizer does not split the camelCase catalog name
- teastore s33 → SlopeOneRecommender: camelCase catalog name vs 'Slope One' in the text
- teastore s34 → PreprocessedSlopeOneRecommender: the two versions described ARE the preprocessed/on-the-go variants
- teastore s34 → SlopeOneRecommender: camelCase catalog name
- teastore s35 → OrderBasedRecommender: 'order-based nearest-neighbor approach' IS the name, hyphenated and split
- teastore s36 → OrderBasedRecommender: 'its recommendation time' continues the order-based description
