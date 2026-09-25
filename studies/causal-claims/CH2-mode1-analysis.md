# CH2 type-1 false positives: a one-by-one error analysis

**Type 1** = a false positive on a sentence that DOES carry a surface form of the
wrongly linked component (`auto_surface != none`) but is about something else. This
is the failure mode `paper/sections/approach.tex` L20-22 and `sections/motivation.tex`
assert as Challenge 2: a link that *reads as plausible but has no support in the
sentence*, illustrated lexically. The complement (type 2, no surface form at all) is
covered in `CH2-mode2-analysis.md` and is not repeated here.

## Method

The population is **derived, never transcribed**: `ch2_mode1_report.py` reads
`ch2_fp_sheet.csv` (all 213 distinct false positives any system produced over the
three s126 terra runs) and subtracts `ch2_mode2_sheet.csv` (the 106 type-2 items),
joining on `(project, sentence, wrong_component)`. That leaves **107 items**, and
the script asserts that the manual verdict set covers them exactly -- no item
annotated that is not in the population, none in the population left unannotated.

Verdicts live as data in `ch2_mode1_annotations.py`: one sub-mode, one gold-silence
judgement, one `auto_surface` flag and one evidence quote per item. Every verdict was
reached by opening the SAD at `benchmark/<project>/text_*/…txt` and reading the
sentence with its neighbours. This file computes every aggregate below; nothing is
hand-tallied.

Counts are DISTINCT items, not per-run rates. `SWATTR` is single-shot, so its column
is 0/1; `Artemis` and our own stages are pooled over 3 runs.

### Sub-mode vocabulary

| code | sub-mode | reading |
|---|---|---|
| `r` | other component's responsibility (the paper's exact case) | |
| `q` | qualifier / compound points elsewhere | |
| `o` | ordinary English, not the component | |
| `t` | third-party technology the component is named after | |
| `s` | shared-token sibling is the real subject | |
| `p` | package-path / code-identifier inventory | |
| `l` | list / cross-reference to a different component | |
| `g` | NO TRAP -- genuine mention, gold gap | |

`r`, `q`, `o`, `t`, `s` are lexical traps in the paper's sense: a name form is
present and denotes something other than the component. `p` and `l` are weaker --
the string is a code identifier or an enumeration entry rather than a claim. `g` is
not a trap at all: the mention is genuine and the gold standard simply has no link.

## Sub-mode distribution

| sub-mode | items | share |
|---|---:|---:|
| `p` package-path / code-identifier inventory | 32 | 29.9% |
| `q` qualifier / compound points elsewhere | 20 | 18.7% |
| `t` third-party technology the component is named after | 15 | 14.0% |
| `o` ordinary English, not the component | 15 | 14.0% |
| `g` NO TRAP -- genuine mention, gold gap | 10 | 9.3% |
| `l` list / cross-reference to a different component | 9 | 8.4% |
| `s` shared-token sibling is the real subject | 4 | 3.7% |
| `r` other component's responsibility (the paper's exact case) | 2 | 1.9% |
| **total** | **107** | |

Genuine lexical traps (`r`+`q`+`o`+`t`+`s`): **56/107** (52.3%). 
Identifier/enumeration artefacts (`p`+`l`): **41** (38.3%). 
Not traps at all (`g`): **10** (9.3%).

### Per project

| project | items | `r` | `q` | `o` | `t` | `s` | `p` | `l` | `g` | dominant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mediastore | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | `q` (2) |
| teastore | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | `q` (4) |
| teammates | 71 | 1 | 4 | 10 | 12 | 0 | 32 | 8 | 4 | `p` (32) |
| bigbluebutton | 27 | 0 | 10 | 3 | 3 | 4 | 0 | 1 | 6 | `q` (10) |
| jabref | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | `o` (2) |

## Cross-tab by producer

| producer | type-1 items | `r` | `q` | `o` | `t` | `s` | `p` | `l` | `g` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWATTR (lexical baseline, single shot) | 39 | 0 | 2 | 7 | 2 | 0 | 26 | 0 | 2 |
| Artemis (LLM baseline, >=2 of 3 runs) | 9 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 5 |
| our scans pre-judge (either judge saw it >=2 runs) | 107 | 2 | 20 | 15 | 15 | 4 | 32 | 9 | 10 |
| our surviving output (>=2 of 3 runs) | 24 | 2 | 7 | 0 | 2 | 0 | 3 | 1 | 9 |

For scale, the same four producers on the **type-2** population were 1 / 35 / 55 / 1
(`CH2-mode2-analysis.md`).

The `our scans pre-judge` row is 107/107 **by construction, not as a finding**: a
type-1 item is one whose sentence carries a surface form of the component, which is
exactly the NameScanner's trigger, so it proposes all of them -- indeed all
107 were put to the NameValidator in all three runs. The informative numbers are the
baselines' and our surviving output's.

Per-project split of the same four columns:

| project | SWATTR | Artemis | our scans | our output |
|---|---:|---:|---:|---:|
| mediastore | 0 | 0 | 3 | 3 |
| teastore | 0 | 0 | 4 | 0 |
| teammates | 32 | 2 | 71 | 13 |
| bigbluebutton | 5 | 6 | 27 | 8 |
| jabref | 2 | 1 | 2 | 0 |
| **all** | 39 | 9 | 107 | 24 |

## Is the gold standard, not the system, what is wrong?

| the sentence does describe that component's responsibility | items | share |
|---|---:|---:|
| yes | 15 | 14.0% |
| borderline | 36 | 33.6% |
| no | 56 | 52.3% |

Gold-silent breakdown per producer:

| producer | items | gold-silent `yes` | `borderline` | `no` |
|---|---:|---:|---:|---:|
| SWATTR (lexical baseline, single shot) | 39 | 9 | 23 | 7 |
| Artemis (LLM baseline, >=2 of 3 runs) | 9 | 4 | 4 | 1 |
| our scans pre-judge (either judge saw it >=2 runs) | 107 | 15 | 36 | 56 |
| our surviving output (>=2 of 3 runs) | 24 | 9 | 6 | 9 |

## Where `auto_surface` overstates the evidence

`audit.py`'s `classify()` labels 107 of these items as carrying a surface form. On
**84** of them (78.5%) that label overstates the lexical evidence:
the matched string is an ordinary English word, a generic head noun promoted to
`partial_d`, a token borrowed from an unrelated compound, or an over-broad document
alias. Distribution of the flagged items by the automatic label they were given:

| auto_surface | items | of which flagged |
|---|---:|---:|
| `canonical` | 41 | 39 |
| `alias` | 27 | 16 |
| `partial_d` | 36 | 26 |
| `partial_g` | 3 | 3 |
| **total** | **107** | **84** |

Every flagged item:

| project | s | component | auto_surface | matched | why the label overstates it |
|---|---:|---|---|---|---|
| bigbluebutton | 5 | BBB web | `partial_d` | `web` | 'web' of 'web application' counted as a distinctive token of BBB web |
| bigbluebutton | 6 | BBB web | `partial_d` | `web` | 'web' of 'web socket' |
| bigbluebutton | 16 | HTML5 Client | `partial_g` | `client` | partial_g: the generic head noun 'client' inside 'client-side' |
| bigbluebutton | 18 | HTML5 Server | `partial_g` | `server` | partial_g: 'server' here is a physical machine |
| bigbluebutton | 19 | BBB web | `partial_d` | `bbb` | 'bbb' taken from the unrelated identifier 'bbb-html5' |
| bigbluebutton | 20 | BBB web | `partial_d` | `bbb` | 'bbb' taken from 'bbb-html5' |
| bigbluebutton | 21 | BBB web | `partial_d` | `bbb` | 'bbb' taken from 'bbb-html5' |
| bigbluebutton | 33 | BBB web | `partial_d` | `bbb` | 'bbb' taken from the CLI name 'bbb-conf' |
| bigbluebutton | 60 | FreeSWITCH | `canonical` | `FreeSWITCH` | canonical fired on the prefix of a LONGER alias bound to a different component ('FreeSWITCH Event Socket Layer' -> FSESL) |
| bigbluebutton | 68 | HTML5 Server | `partial_g` | `server` | partial_g: 'server' of 'media server' |
| bigbluebutton | 75 | Presentation Conversion | `partial_d` | `presentation` | 'presentation' of 'a presentation' treated as distinctive |
| bigbluebutton | 77 | Presentation Conversion | `partial_d` | `presentation` | 'presentation' of 'the uploaded presentation' |
| bigbluebutton | 82 | Presentation Conversion | `partial_d` | `conversion` | 'conversion' of 'SWF, SVG and PNG conversion' |
| bigbluebutton | 83 | Presentation Conversion | `partial_d` | `conversion` | 'conversion' of 'the SVG conversion flow' |
| bigbluebutton | 84 | Presentation Conversion | `partial_d` | `conversion` | 'conversion' of 'the conversion fallback' |
| jabref | 5 | logic | `canonical` | `logic` | canonical 'logic' is ordinary English here |
| jabref | 7 | preferences | `canonical` | `preferences` | canonical 'preferences' is ordinary English here |
| teammates | 2 | GAE Datastore | `alias` | `GAE` | alias 'GAE' is bound to GAE Datastore but GAE is the whole platform |
| teammates | 4 | Client | `canonical` | `Client` | canonical 'Client' fired on 'client-side' |
| teammates | 12 | Test Driver | `partial_d` | `test` | 'test' counted as a distinctive token of Test Driver |
| teammates | 18 | Logic | `alias` | `back end` | alias 'back end' -> Logic is over-broad (the back end is Logic+Storage+GAE) |
| teammates | 22 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on the package segment of 'logic, ui.website, ui.controller' |
| teammates | 22 | UI | `canonical` | `UI` | canonical 'UI' fired on 'ui.website'/'ui.controller' |
| teammates | 23 | UI | `canonical` | `UI` | canonical 'UI' fired on 'ui.website' |
| teammates | 26 | UI | `canonical` | `UI` | canonical 'UI' fired on 'ui.website' |
| teammates | 31 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore, but 'GAE server' is the app server |
| teammates | 35 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 49 | Common | `canonical` | `Common` | canonical 'Common' fired on the adjective 'common' |
| teammates | 58 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 62 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 69 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 73 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 74 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 79 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on 'cascade logic' |
| teammates | 83 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 84 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on 'logic.api, logic.core' |
| teammates | 85 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on 'logic.api' |
| teammates | 86 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on 'logic.core' and on the noun 'logic' |
| teammates | 94 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 117 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on 'cascade logic' |
| teammates | 119 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on the mass noun 'logic' |
| teammates | 125 | Storage | `canonical` | `Storage` | canonical 'Storage' fired on 'storage.entity' |
| teammates | 127 | Common | `canonical` | `Common` | canonical 'Common' fired on 'common.datatransfer' |
| teammates | 130 | Storage | `canonical` | `Storage` | canonical 'Storage' fired on 'storage.api, storage.entity, storage.search' |
| teammates | 131 | Storage | `canonical` | `Storage` | canonical 'Storage' fired on 'storage.api' |
| teammates | 132 | Storage | `canonical` | `Storage` | canonical 'Storage' fired on 'storage.entity' |
| teammates | 133 | Storage | `canonical` | `Storage` | canonical 'Storage' fired on 'storage.search' |
| teammates | 139 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 140 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 144 | GAE Datastore | `alias` | `GAE` | alias 'GAE' -> Datastore |
| teammates | 156 | Common | `canonical` | `Common` | canonical 'Common' fired on 'common.util, common.exceptions, common.datatransfer' |
| teammates | 157 | Common | `canonical` | `Common` | canonical 'Common' fired on 'common.util' |
| teammates | 158 | Common | `canonical` | `Common` | canonical 'Common' fired on 'common.exceptions' |
| teammates | 159 | Common | `canonical` | `Common` | canonical 'Common' fired on 'common.datatransfer' |
| teammates | 160 | Common | `canonical` | `Common` | canonical 'Common' fired on 'common.datatransfer' |
| teammates | 169 | Test Driver | `canonical` | `Test Driver` | canonical 'Test Driver' fired on the path 'test.driver' |
| teammates | 170 | Test Driver | `canonical` | `Test Driver` | canonical 'Test Driver' fired on 'test.driver' |
| teammates | 171 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 172 | Logic | `canonical` | `Logic` | canonical 'Logic' fired on the test package 'x.logic' |
| teammates | 172 | Storage | `canonical` | `Storage` | canonical 'Storage' fired on the test package 'x.storage' |
| teammates | 174 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 175 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 176 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 177 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 178 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 179 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 180 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 181 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 182 | UI | `alias` | `front-end` | alias 'front-end' matched inside 'Front-end files' |
| teammates | 183 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 187 | E2E | `canonical` | `E2E` | canonical 'E2E' fired on 'e2e.util, e2e.pageobjects, e2e.cases' |
| teammates | 188 | E2E | `canonical` | `E2E` | canonical 'E2E' fired on 'e2e.util' |
| teammates | 189 | E2E | `canonical` | `E2E` | canonical 'E2E' fired on 'e2e.pageobjects' |
| teammates | 190 | E2E | `canonical` | `E2E` | canonical 'E2E' fired on 'e2e.cases' |
| teammates | 190 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 191 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 192 | E2E | `canonical` | `E2E` | canonical 'E2E' fired on the test package 'x.e2e' |
| teammates | 192 | Test Driver | `partial_d` | `test` | 'test' counted as distinctive |
| teammates | 195 | Client | `canonical` | `Client` | canonical 'Client' fired on 'client.util, client.remoteapi, client.scripts' |
| teammates | 196 | Client | `canonical` | `Client` | canonical 'Client' fired on 'client.util' |
| teammates | 197 | Client | `canonical` | `Client` | canonical 'Client' fired on 'client.remoteapi' |
| teammates | 197 | Logic | `alias` | `back end` | alias 'back end' -> Logic is over-broad |
| teammates | 198 | Client | `canonical` | `Client` | canonical 'Client' fired on 'client.scripts' |
| teammates | 198 | Logic | `alias` | `back end` | alias 'back end' -> Logic is over-broad |

## Judge behaviour

- **Rejected by the NameValidator** (`name_judged>=2 and name_kept==0`): **82** items.
- **Survived to our final output** (`approach_final>=2`): **24** items.
- For contrast, the type-2 population had exactly **1** survivor out of 106.

What separates them. Rejected vs survived, by the automatic surface label, by
sub-mode, and by gold silence:

| split | `canonical` | `alias` | `partial_d` | `partial_g` |
|---|---:|---:|---:|---:|
| rejected (82) | 35 | 15 | 31 | 1 |
| survived (24) | 6 | 12 | 5 | 1 |

| split | `r` | `q` | `o` | `t` | `s` | `p` | `l` | `g` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rejected (82) | 0 | 13 | 13 | 13 | 4 | 30 | 8 | 1 |
| survived (24) | 2 | 7 | 0 | 2 | 0 | 3 | 1 | 9 |

| split | gold-silent `yes` | `borderline` | `no` | flagged `auto_surface` |
|---|---:|---:|---:|---:|
| rejected (82) | 7 | 30 | 45 | 69 |
| survived (24) | 9 | 6 | 9 | 14 |

So of the 24 survivors, **15** are gold-silent `yes`/`borderline` -- the
sentence arguably does describe the component and the reference gold simply has no
link there -- and **9** are false positives by any reading.

### What the judge lets through

Survival rate by the automatic surface label -- this is the sharpest signal in the
whole analysis:

| auto_surface | type-1 items | survived | survival rate |
|---|---:|---:|---:|
| `canonical` | 41 | 6 | 14.6% |
| `alias` | 27 | 12 | 44.4% |
| `partial_d` | 36 | 5 | 13.9% |
| `partial_g` | 3 | 1 | 33.3% |
| **total** | **107** | **24** | **22.4%** |

**The leak is the document-alias table.** 12 of the 24 survivors
(50.0%) were matched through a run-generated alias, although
aliases are only 27/107 of the population. A `partial_d` or `canonical` hit on
an ordinary word is something the NameValidator reliably talks itself out of; a hit on
a string the run itself put in the alias table is treated as settled, and the judge
never re-asks whether that string is being used in its component sense here. The
aliases doing the damage, with how many survivors each produced:

| alias | bound to | survivors | example |
|---|---|---:|---|
| `Database` | DB | 3 | The UserDBAdapter component queries the database. |
| `back end` | Logic | 3 | The Client component can connect to the back end directly |
| `front-end` | UI | 3 | It is a conceptual package representing the front-end of the application |
| `GAE` | GAE Datastore | 2 | eventual consistency involving multiple servers in the GAE production environment |
| `Recording Processor` | Recording Service | 1 | the Recording Processor will take all the recorded events ... for processing |

By sub-mode the same point: the judge removes **every** ordinary-English trap
(`o`: 15 items, 0 survivors) and most technology-name
traps (`t`: 15 items, 2 survivors), but keeps the compound/qualifier
trap (`q`: 20 items, 7 survivors) and almost every genuine mention the
gold happens not to cover (`g`: 10 items, 9 survivors). In short: it is a good detector of
*is this string even a name?* and a poor detector of *is this name the subject of the
claim?* -- which is precisely the discrimination Challenge 2 asks for.

### Every surviving item

- **bigbluebutton s16 -> HTML5 Client** (`q`, gold-silent: no, auto_surface `partial_g`/`client`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > BigBlueButton 2.2 used a single nodejs process for all client-side communication.
  - gold here: (none)
  - the subject is the HTML5 server's nodejs process; 'client-side' is an adjective
- **bigbluebutton s50 -> Recording Service** (`g`, gold-silent: yes, auto_surface `alias`/`Recording Processor`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > When the meeting ends, the Recording Processor will take all the recorded events as well as the different raw (PDF, WAV, FLV) files for processing.
  - gold here: (none)
  - 'Recording Processor' is the run's own alias for Recording Service and the sentence IS its responsibility; the component carries no gold link anywhere in the project
- **bigbluebutton s60 -> FreeSWITCH** (`q`, gold-silent: no, auto_surface `canonical`/`FreeSWITCH`) -- SWATTR, Artemis, nameJudge 2/3, **SURVIVED 2/3**
  - > Communication between apps and FreeSWITCH Event Socket Layer (fsels) uses messages through redis pubsub.
  - gold here: Apps;FSESL;Redis PubSub
  - THE textbook Challenge-2 item: the name FreeSWITCH is present but the phrase names FSESL; gold here is Apps+FSESL+Redis PubSub
- **bigbluebutton s76 -> Presentation Conversion** (`g`, gold-silent: yes, auto_surface `partial_d`/`conversion`) -- Artemis, nameJudge 3/3, corefJudge 0/1, **SURVIVED 3/3**
  - > Uploaded presentations go through a conversion process in order to be displayed inside the client.
  - gold here: HTML5 Client
  - this IS the presentation conversion; gold links only s80-s81 to the component and puts HTML5 Client here
- **bigbluebutton s79 -> Presentation Conversion** (`g`, gold-silent: yes, auto_surface `partial_d`/`conversion`) -- Artemis, nameJudge 3/3, corefJudge 0/2, **SURVIVED 3/3**
  - > The conversion process sends progress messages to the client through the Redis pubsub.
  - gold here: HTML5 Client;Redis PubSub
  - 'the conversion process' anaphorically IS the component; gold assigns only HTML5 Client + Redis PubSub
- **bigbluebutton s82 -> Presentation Conversion** (`g`, gold-silent: borderline, auto_surface `partial_d`/`conversion`) -- Artemis, nameJudge 3/3, **SURVIVED 3/3**
  - > We take in consideration the configuration for enabling and disabling SWF, SVG and PNG conversion.
  - gold here: (none)
  - inside the presentation-conversion-flow section; gold is silent
- **bigbluebutton s83 -> Presentation Conversion** (`l`, gold-silent: borderline, auto_surface `partial_d`/`conversion`) -- Artemis, nameJudge 3/3, **SURVIVED 3/3**
  - > Then below the SVG conversion flow.
  - gold here: (none)
  - a pointer to a diagram, no claim about any component
- **bigbluebutton s84 -> Presentation Conversion** (`g`, gold-silent: borderline, auto_surface `partial_d`/`conversion`) -- Artemis, nameJudge 3/3, **SURVIVED 3/3**
  - > It covers the conversion fallback.
  - gold here: (none)
  - describes the converter's fallback behaviour; gold is silent
- **mediastore s12 -> DB** (`r`, gold-silent: borderline, auto_surface `alias`/`Database`) -- nameJudge 2/3, **SURVIVED 2/3**
  - > The UserDBAdapter component queries the database.
  - gold here: UserDBAdapter
  - the claim is UserDBAdapter's responsibility; 'the database' is its object, lower-case and unspecific
- **mediastore s27 -> DB** (`q`, gold-silent: no, auto_surface `alias`/`Database`) -- nameJudge 2/3, **SURVIVED 2/3**
  - > The MediaAccess component encapsulates database access for meta-data of audio files.
  - gold here: MediaAccess
  - 'database access' -- literally the paper's own qualifier example; the responsibility is MediaAccess's
- **mediastore s29 -> DB** (`q`, gold-silent: no, auto_surface `alias`/`Database`) -- nameJudge 2/3, **SURVIVED 2/3**
  - > By contrast, the UserDBAdapter component provides all functions required in order to encapsulate database access for the user data.
  - gold here: UserDBAdapter
  - 'database access' again; the responsibility is UserDBAdapter's
- **teammates s17 -> E2E** (`g`, gold-silent: yes, auto_surface `canonical`/`E2E`) -- SWATTR, Artemis, nameJudge 3/3, **SURVIVED 3/3**
  - > Selenium Java is used to automate E2E testing with actual Web browsers.
  - gold here: (none)
  - gold links s15/s16 ('its primary function is for E2E tests') but not this one; same subject
- **teammates s18 -> Logic** (`r`, gold-silent: no, auto_surface `alias`/`back end`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > The Client component can connect to the back end directly without using a Web browser.
  - gold here: Client
  - the claim is the Client component's capability
- **teammates s22 -> Logic** (`p`, gold-silent: borderline, auto_surface `canonical`/`Logic`) -- SWATTR, nameJudge 3/3, **SURVIVED 3/3**
  - > logic, ui.website, ui.controller represent an application of Model-View-Controller pattern.
  - gold here: (none)
  - a package-level design claim, not a Logic responsibility
- **teammates s24 -> UI** (`g`, gold-silent: yes, auto_surface `alias`/`front-end`) -- Artemis(1/3), nameJudge 3/3, corefJudge 0/3, **SURVIVED 3/3**
  - > It is a conceptual package representing the front-end of the application.
  - gold here: (none)
  - 'front-end' is the run's alias for UI and it genuinely denotes it; gold starts linking UI at the next sentence
- **teammates s50 -> UI** (`g`, gold-silent: yes, auto_surface `alias`/`front-end`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > Sixth, WebApiServlet sends the result back to the browser which will then process it on the front-end.
  - gold here: (none)
  - 'the front-end' is the UI alias and the processing is genuinely the UI's; gold is silent
- **teammates s125 -> Storage** (`p`, gold-silent: yes, auto_surface `canonical`/`Storage`) -- SWATTR, Artemis(1/3), nameJudge 0/3, corefJudge 3/3, **SURVIVED 3/3**
  - > Classes in the storage.entity package are not visible outside this component to hide information specific to data persistence.
  - gold here: (none)
  - 'this component' IS Storage; gold links s123 and s128 but not this one
- **teammates s139 -> GAE Datastore** (`t`, gold-silent: no, auto_surface `alias`/`GAE`) -- nameJudge 2/3, **SURVIVED 2/3**
  - > This is not enough to compensate for eventual consistency involving multiple servers in the GAE production environment.
  - gold here: (none)
  - a deployment-environment remark
- **teammates s144 -> GAE Datastore** (`t`, gold-silent: no, auto_surface `alias`/`GAE`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > Implementation of Transaction Control has been minimized due to limitations of GAE environment and the nature of our data schema.
  - gold here: (none)
  - a platform limitation, not the datastore component
- **teammates s173 -> Test Driver** (`g`, gold-silent: yes, auto_surface `canonical`/`Test Driver`) -- SWATTR, nameJudge 3/3, **SURVIVED 3/3**
  - > x.testdriver contains component test cases for testing the test driver infrastructure and helpers.
  - gold here: (none)
  - the object under test is literally the Test Driver component; gold links Test Driver only at s1/s10/s163/s168
- **teammates s182 -> UI** (`q`, gold-silent: borderline, auto_surface `alias`/`front-end`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > Front-end files (particularly TypeScript) are tested separately with Jest.
  - gold here: (none)
  - the claim is about the testing setup; 'front-end' modifies 'files'
- **teammates s196 -> Client** (`p`, gold-silent: yes, auto_surface `canonical`/`Client`) -- SWATTR, nameJudge 3/3, **SURVIVED 3/3**
  - > client.util contains helpers needed for client scripts.
  - gold here: (none)
  - that package IS the Client component; gold links Client only at s1/s18/s19/s194
- **teammates s197 -> Logic** (`q`, gold-silent: no, auto_surface `alias`/`back end`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > client.remoteapi classes needed to connect to the back end directly.
  - gold here: (none)
  - 'the back end' is the whole server side; the claim is about client.remoteapi
- **teammates s198 -> Logic** (`q`, gold-silent: no, auto_surface `alias`/`back end`) -- nameJudge 3/3, **SURVIVED 3/3**
  - > client.scripts scripts that deal with the back end data for administrative purposes.
  - gold here: (none)
  - 'the back end data' is the datastore contents; the claim is about client.scripts

### Items the NameValidator rejected

By sub-mode: `q` 13, `o` 13, `t` 13, `s` 4, `p` 30, `l` 8, `g` 1.

| project | s | component | auto_surface | matched | sub-mode | gold-silent | evidence |
|---|---:|---|---|---|---|---|---|
| bigbluebutton | 5 | BBB web | `partial_d` | `web` | `q` | no | a single page, responsive web application |
| bigbluebutton | 5 | WebRTC-SFU | `partial_d` | `webrtc` | `t` | no | WebRTC for sending/receiving audio and video |
| bigbluebutton | 6 | BBB web | `partial_d` | `web` | `q` | no | making a web socket connection |
| bigbluebutton | 19 | BBB web | `partial_d` | `bbb` | `q` | no | moves away from a single nodejs process for bbb-html5 |
| bigbluebutton | 20 | BBB web | `partial_d` | `bbb` | `q` | no | this means that bbb-html5 could use multiple CPU cores |
| bigbluebutton | 20 | HTML5 Client | `partial_d` | `html5` | `s` | no | bbb-html5 could use multiple CPU cores for processing messages |
| bigbluebutton | 21 | BBB web | `partial_d` | `bbb` | `q` | no | bbb-html5 uses 2 'frontend' and two 'backend' processes |
| bigbluebutton | 21 | HTML5 Client | `partial_d` | `html5` | `s` | no | bbb-html5 uses 2 'frontend' and two 'backend' processes |
| bigbluebutton | 27 | Redis DB | `partial_d` | `redis` | `s` | no | Frontends handle completely the Streamer redis events |
| bigbluebutton | 27 | Redis PubSub | `partial_d` | `redis` | `q` | borderline | handle completely the Streamer redis events |
| bigbluebutton | 31 | Redis DB | `partial_d` | `redis` | `s` | no | individual backends only process redis events |
| bigbluebutton | 31 | Redis PubSub | `partial_d` | `redis` | `q` | borderline | only process redis events for the meetings matching the associated instanceId |
| bigbluebutton | 33 | BBB web | `partial_d` | `bbb` | `q` | no | sudo bbb-conf --setip <hostname> |
| bigbluebutton | 68 | HTML5 Server | `partial_g` | `server` | `o` | no | Kurento Media Server KMS is a media server |
| bigbluebutton | 68 | WebRTC-SFU | `alias` | `SFU` | `t` | no | implements both SFU and MCU models |
| bigbluebutton | 74 | WebRTC-SFU | `partial_d` | `webrtc` | `t` | borderline | WebRTC provides the user with high-quality audio with lower delay |
| bigbluebutton | 75 | Presentation Conversion | `partial_d` | `presentation` | `o` | no | Uploading a presentation. |
| bigbluebutton | 77 | Presentation Conversion | `partial_d` | `presentation` | `g` | borderline | it needs to be converted into PDF using LibreOffice |
| jabref | 7 | preferences | `canonical` | `preferences` | `o` | borderline | Only the gui knows the user and his preferences |
| teammates | 2 | GAE Datastore | `alias` | `GAE` | `t` | no | runs on Google App Engine (GAE) |
| teammates | 4 | Client | `canonical` | `Client` | `q` | no | JavaScript for client-side interactions such as sorting |
| teammates | 12 | Test Driver | `partial_d` | `test` | `o` | borderline | Test data is transmitted using JSON format. |
| teammates | 22 | UI | `canonical` | `UI` | `p` | borderline | logic, ui.website, ui.controller represent an application of Model-View-Controller pattern |
| teammates | 23 | UI | `canonical` | `UI` | `p` | borderline | ui.website is not a real package. |
| teammates | 26 | UI | `canonical` | `UI` | `p` | borderline | ui.website is not a Java package. |
| teammates | 31 | GAE Datastore | `alias` | `GAE` | `t` | no | First, request received by the GAE server. |
| teammates | 35 | GAE Datastore | `alias` | `GAE` | `t` | no | the automated (GAE server-invoked) requests |
| teammates | 49 | Common | `canonical` | `Common` | `o` | no | The most common format is JsonResult |
| teammates | 58 | GAE Datastore | `alias` | `GAE` | `t` | no | requests sent automatically by the GAE server |
| teammates | 62 | GAE Datastore | `alias` | `GAE` | `t` | no | requests generated by the GAE server are equipped with this privilege |
| teammates | 69 | GAE Datastore | `alias` | `GAE` | `t` | no | GAE server sends such automated requests through two different configurations |
| teammates | 73 | GAE Datastore | `alias` | `GAE` | `t` | no | queued by users ... but executed by GAE |
| teammates | 74 | GAE Datastore | `alias` | `GAE` | `t` | no | the 1 minute standard request processing limit imposed by GAE |
| teammates | 79 | Logic | `canonical` | `Logic` | `o` | yes | Managing relationships between entities, e.g. cascade logic |
| teammates | 83 | GAE Datastore | `alias` | `GAE` | `t` | no | Connecting to GAE-provided or third-party APIs |
| teammates | 84 | Logic | `canonical` | `Logic` | `p` | borderline | Package overview contains logic.api, logic.core. |
| teammates | 85 | Logic | `canonical` | `Logic` | `p` | yes | logic.api provides the API of the component to be accessed by the UI |
| teammates | 86 | Logic | `canonical` | `Logic` | `p` | yes | logic.core contains the core logic of the system |
| teammates | 94 | GAE Datastore | `alias` | `GAE` | `t` | no | It connects to GAE's task queue API. |
| teammates | 117 | Logic | `canonical` | `Logic` | `o` | borderline | Refer to the API for the cascade logic. |
| teammates | 119 | Logic | `canonical` | `Logic` | `o` | no | It contains minimal logic beyond what is directly relevant to CRUD operations |
| teammates | 125 | Storage | `canonical` | `Storage` | `p` | yes | Classes in the storage.entity package are not visible outside this component |
| teammates | 127 | Common | `canonical` | `Common` | `p` | borderline | These datatransfer classes are in common.datatransfer package |
| teammates | 130 | Storage | `canonical` | `Storage` | `p` | borderline | Package overview contains storage.api, storage.entity, storage.search. |
| teammates | 131 | Storage | `canonical` | `Storage` | `p` | yes | storage.api provides the API of the component to be accessed by the logic component |
| teammates | 132 | Storage | `canonical` | `Storage` | `p` | borderline | storage.entity contains classes that represent persistable entities |
| teammates | 133 | Storage | `canonical` | `Storage` | `p` | borderline | storage.search contains classes for dealing with searching and indexing |
| teammates | 140 | Test Driver | `partial_d` | `test` | `o` | no | it is expected to avoid test failures caused by eventual consistency |
| teammates | 156 | Common | `canonical` | `Common` | `p` | borderline | Package overview contains common.util, common.exceptions, common.datatransfer. |
| teammates | 157 | Common | `canonical` | `Common` | `p` | borderline | common.util contains utility classes. |
| teammates | 158 | Common | `canonical` | `Common` | `p` | borderline | common.exceptions contains custom exceptions. |
| teammates | 159 | Common | `canonical` | `Common` | `p` | borderline | common.datatransfer contains data transfer objects. |
| teammates | 160 | Common | `canonical` | `Common` | `p` | borderline | common.datatransfer package contains lightweight data transfer object classes |
| teammates | 169 | Test Driver | `canonical` | `Test Driver` | `p` | borderline | Package overview contains test.driver, test.cases and subpackages. |
| teammates | 170 | Test Driver | `canonical` | `Test Driver` | `p` | yes | test.driver contains infrastructure and helpers needed for running the tests |
| teammates | 171 | Test Driver | `partial_d` | `test` | `p` | borderline | test.cases contains test cases. |
| teammates | 172 | Logic | `canonical` | `Logic` | `p` | no | Sub-packages contains x.testdriver, x.datatransfer, x.util, x.logic, x.storage |
| teammates | 172 | Storage | `canonical` | `Storage` | `p` | no | Sub-packages contains ... x.logic, x.storage, x.search, x.webapi |
| teammates | 174 | Test Driver | `partial_d` | `test` | `l` | no | component test cases for testing the datatransfer objects from the Common component |
| teammates | 175 | Test Driver | `partial_d` | `test` | `l` | no | testing the utility classes from the Common component |
| teammates | 176 | Test Driver | `partial_d` | `test` | `l` | no | x.logic contains component test cases for testing the Logic component |
| teammates | 177 | Test Driver | `partial_d` | `test` | `l` | no | x.storage contains component test cases for testing the Storage component |
| teammates | 178 | Test Driver | `partial_d` | `test` | `l` | no | x.search contains component test cases for testing the search functions |
| teammates | 179 | Test Driver | `partial_d` | `test` | `l` | no | x.webapi contains system test cases for testing the user-invoked actions |
| teammates | 180 | Test Driver | `partial_d` | `test` | `l` | no | x.automated contains system test cases for testing the system-automated actions |
| teammates | 181 | Test Driver | `partial_d` | `test` | `o` | borderline | Some Component tests are pure unit tests |
| teammates | 183 | Test Driver | `partial_d` | `test` | `o` | no | The test cases are found in x.spec.ts files. |
| teammates | 187 | E2E | `canonical` | `E2E` | `p` | borderline | Package overview contains e2e.util, e2e.pageobjects, e2e.cases, x.util, x.e2e, x.lnp |
| teammates | 188 | E2E | `canonical` | `E2E` | `p` | yes | e2e.util contains helpers needed for running E2E tests |
| teammates | 189 | E2E | `canonical` | `E2E` | `p` | borderline | e2e.pageobjects contains abstractions of the pages as they appear on a Browser |
| teammates | 190 | E2E | `canonical` | `E2E` | `p` | borderline | e2e.cases contains test cases. |
| teammates | 190 | Test Driver | `partial_d` | `test` | `o` | no | e2e.cases contains test cases. |
| teammates | 191 | Test Driver | `partial_d` | `test` | `l` | no | x.util contains component test cases for testing the test helpers |
| teammates | 192 | E2E | `canonical` | `E2E` | `p` | borderline | x.e2e contains system test cases for testing the application as a whole |
| teammates | 192 | Test Driver | `partial_d` | `test` | `o` | no | x.e2e contains system test cases |
| teammates | 195 | Client | `canonical` | `Client` | `p` | borderline | Package overview contains client.util, client.remoteapi, client.scripts. |
| teammates | 197 | Client | `canonical` | `Client` | `p` | borderline | client.remoteapi classes needed to connect to the back end directly |
| teammates | 198 | Client | `canonical` | `Client` | `p` | borderline | client.scripts scripts that deal with the back end data |
| teastore | 11 | WebUI | `alias` | `UI` | `q` | no | It matches the provided product ID or UI name |
| teastore | 12 | WebUI | `alias` | `UI` | `q` | no | If the product ID or UI name is not available to the Image Provider |
| teastore | 13 | WebUI | `alias` | `UI` | `q` | no | If the product ID or UI name is found but not in the requested size |
| teastore | 15 | WebUI | `alias` | `UI` | `q` | no | If the product ID or UI name and size is found, the image will be loaded and delivered |

## Every item

| # | project | s | wrongly linked to | auto_surface | matched | sub-mode | gold-silent | `auto_surface` flagged | produced by | evidence | why it is wrong |
|---:|---|---:|---|---|---|---|---|---|---|---|---|
| 1 | bigbluebutton | 5 | BBB web | `partial_d` | `web` | `q` | no | yes | nameJudge 0/3 | a single page, responsive web application | the web application described IS the HTML5 client; 'BigBlueButton web application' is the alias of BBB web, so the generic phrase collides with it |
| 2 | bigbluebutton | 5 | WebRTC-SFU | `partial_d` | `webrtc` | `t` | no |  | SWATTR, nameJudge 0/3 | WebRTC for sending/receiving audio and video | WebRTC is the browser API the client is built on, not the WebRTC-SFU component |
| 3 | bigbluebutton | 6 | BBB web | `partial_d` | `web` | `q` | no | yes | nameJudge 0/3 | making a web socket connection | 'web socket' is a transport, not bbb-web |
| 4 | bigbluebutton | 16 | HTML5 Client | `partial_g` | `client` | `q` | no | yes | nameJudge 3/3, **SURVIVED 3/3** | a single nodejs process for all client-side communication | the subject is the HTML5 server's nodejs process; 'client-side' is an adjective |
| 5 | bigbluebutton | 18 | HTML5 Server | `partial_g` | `server` | `o` | no | yes | SWATTR, Artemis(1/3), nameJudge 1/3, **SURVIVED 1/3** | having a 16 or 32 CPU core server for BigBlueButton 2.2 | hardware, not the HTML5 server component |
| 6 | bigbluebutton | 19 | BBB web | `partial_d` | `bbb` | `q` | no | yes | nameJudge 0/3 | moves away from a single nodejs process for bbb-html5 | bbb-html5 is the HTML5 server; only the shared 'bbb' prefix matched |
| 7 | bigbluebutton | 20 | BBB web | `partial_d` | `bbb` | `q` | no | yes | nameJudge 0/3 | this means that bbb-html5 could use multiple CPU cores | same prefix collision |
| 8 | bigbluebutton | 20 | HTML5 Client | `partial_d` | `html5` | `s` | no |  | nameJudge 0/3 | bbb-html5 could use multiple CPU cores for processing messages | 'html5' is shared with the sibling HTML5 Server, whose alias bbb-html5 is the subject |
| 9 | bigbluebutton | 21 | BBB web | `partial_d` | `bbb` | `q` | no | yes | nameJudge 0/3 | bbb-html5 uses 2 'frontend' and two 'backend' processes | same prefix collision |
| 10 | bigbluebutton | 21 | HTML5 Client | `partial_d` | `html5` | `s` | no |  | nameJudge 0/3, corefJudge 0/3 | bbb-html5 uses 2 'frontend' and two 'backend' processes | sibling HTML5 Server (bbb-html5) is the subject |
| 11 | bigbluebutton | 27 | Redis DB | `partial_d` | `redis` | `s` | no |  | nameJudge 0/3 | Frontends handle completely the Streamer redis events | the redis traffic here is pubsub, not the Redis DB sibling |
| 12 | bigbluebutton | 27 | Redis PubSub | `partial_d` | `redis` | `q` | borderline |  | nameJudge 0/3 | handle completely the Streamer redis events | 'redis events' does denote pubsub traffic but the claim is about the frontends |
| 13 | bigbluebutton | 31 | Redis DB | `partial_d` | `redis` | `s` | no |  | nameJudge 0/3 | individual backends only process redis events | pubsub traffic, not the Redis DB sibling |
| 14 | bigbluebutton | 31 | Redis PubSub | `partial_d` | `redis` | `q` | borderline |  | nameJudge 0/3 | only process redis events for the meetings matching the associated instanceId | the claim is about backend process sharding |
| 15 | bigbluebutton | 33 | BBB web | `partial_d` | `bbb` | `q` | no | yes | nameJudge 0/3, corefJudge 0/2 | sudo bbb-conf --setip <hostname> | bbb-conf is a shell tool, not the web application |
| 16 | bigbluebutton | 50 | Recording Service | `alias` | `Recording Processor` | `g` | yes |  | nameJudge 3/3, **SURVIVED 3/3** | the Recording Processor will take all the recorded events ... for processing | 'Recording Processor' is the run's own alias for Recording Service and the sentence IS its responsibility; the component carries no gold link anywhere in the project |
| 17 | bigbluebutton | 60 | FreeSWITCH | `canonical` | `FreeSWITCH` | `q` | no | yes | SWATTR, Artemis, nameJudge 2/3, **SURVIVED 2/3** | Communication between apps and FreeSWITCH Event Socket Layer (fsels) | THE textbook Challenge-2 item: the name FreeSWITCH is present but the phrase names FSESL; gold here is Apps+FSESL+Redis PubSub |
| 18 | bigbluebutton | 68 | HTML5 Server | `partial_g` | `server` | `o` | no | yes | SWATTR, nameJudge 0/3 | Kurento Media Server KMS is a media server | a different server entirely; gold is kurento |
| 19 | bigbluebutton | 68 | WebRTC-SFU | `alias` | `SFU` | `t` | no |  | nameJudge 0/3 | implements both SFU and MCU models | SFU is the generic media-routing model here, paired with MCU; gold is kurento |
| 20 | bigbluebutton | 74 | WebRTC-SFU | `partial_d` | `webrtc` | `t` | borderline |  | SWATTR, nameJudge 0/3 | WebRTC provides the user with high-quality audio with lower delay | a claim about the WebRTC protocol's quality, not the SFU component |
| 21 | bigbluebutton | 75 | Presentation Conversion | `partial_d` | `presentation` | `o` | no | yes | nameJudge 0/3 | Uploading a presentation. | a section heading about the artefact, not the converter |
| 22 | bigbluebutton | 76 | Presentation Conversion | `partial_d` | `conversion` | `g` | yes |  | Artemis, nameJudge 3/3, corefJudge 0/1, **SURVIVED 3/3** | Uploaded presentations go through a conversion process | this IS the presentation conversion; gold links only s80-s81 to the component and puts HTML5 Client here |
| 23 | bigbluebutton | 77 | Presentation Conversion | `partial_d` | `presentation` | `g` | borderline | yes | nameJudge 0/3 | it needs to be converted into PDF using LibreOffice | a conversion pipeline step; gold is silent |
| 24 | bigbluebutton | 79 | Presentation Conversion | `partial_d` | `conversion` | `g` | yes |  | Artemis, nameJudge 3/3, corefJudge 0/2, **SURVIVED 3/3** | The conversion process sends progress messages to the client | 'the conversion process' anaphorically IS the component; gold assigns only HTML5 Client + Redis PubSub |
| 25 | bigbluebutton | 82 | Presentation Conversion | `partial_d` | `conversion` | `g` | borderline | yes | Artemis, nameJudge 3/3, **SURVIVED 3/3** | enabling and disabling SWF, SVG and PNG conversion | inside the presentation-conversion-flow section; gold is silent |
| 26 | bigbluebutton | 83 | Presentation Conversion | `partial_d` | `conversion` | `l` | borderline | yes | Artemis, nameJudge 3/3, **SURVIVED 3/3** | Then below the SVG conversion flow. | a pointer to a diagram, no claim about any component |
| 27 | bigbluebutton | 84 | Presentation Conversion | `partial_d` | `conversion` | `g` | borderline | yes | Artemis, nameJudge 3/3, **SURVIVED 3/3** | It covers the conversion fallback. | describes the converter's fallback behaviour; gold is silent |
| 28 | jabref | 5 | logic | `canonical` | `logic` | `o` | no | yes | SWATTR, nameJudge 1/3, corefJudge 0/1, **SURVIVED 1/3** | has only a little bit of logic attached | the claim is about the model layer; 'logic' is a mass noun, not the logic layer |
| 29 | jabref | 7 | preferences | `canonical` | `preferences` | `o` | borderline | yes | SWATTR, Artemis, nameJudge 0/3, corefJudge 0/2 | Only the gui knows the user and his preferences | 'his preferences' is the user's settings; the claim is about the gui, though the preferences package does hold them |
| 30 | mediastore | 12 | DB | `alias` | `Database` | `r` | borderline |  | nameJudge 2/3, **SURVIVED 2/3** | The UserDBAdapter component queries the database. | the claim is UserDBAdapter's responsibility; 'the database' is its object, lower-case and unspecific |
| 31 | mediastore | 27 | DB | `alias` | `Database` | `q` | no |  | nameJudge 2/3, **SURVIVED 2/3** | The MediaAccess component encapsulates database access for meta-data | 'database access' -- literally the paper's own qualifier example; the responsibility is MediaAccess's |
| 32 | mediastore | 29 | DB | `alias` | `Database` | `q` | no |  | nameJudge 2/3, **SURVIVED 2/3** | in order to encapsulate database access for the user data | 'database access' again; the responsibility is UserDBAdapter's |
| 33 | teammates | 2 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | runs on Google App Engine (GAE) | the hosting platform, not the datastore |
| 34 | teammates | 4 | Client | `canonical` | `Client` | `q` | no | yes | SWATTR, nameJudge 0/3 | JavaScript for client-side interactions such as sorting | the claim is about the UI; 'client-side' is an adjective |
| 35 | teammates | 12 | Test Driver | `partial_d` | `test` | `o` | borderline | yes | nameJudge 0/3 | Test data is transmitted using JSON format. | 'test data' is a format claim inside the Test Driver section |
| 36 | teammates | 17 | E2E | `canonical` | `E2E` | `g` | yes |  | SWATTR, Artemis, nameJudge 3/3, **SURVIVED 3/3** | Selenium Java is used to automate E2E testing with actual Web browsers | gold links s15/s16 ('its primary function is for E2E tests') but not this one; same subject |
| 37 | teammates | 18 | Logic | `alias` | `back end` | `r` | no | yes | nameJudge 3/3, **SURVIVED 3/3** | The Client component can connect to the back end directly | the claim is the Client component's capability |
| 38 | teammates | 22 | Logic | `canonical` | `Logic` | `p` | borderline | yes | SWATTR, nameJudge 3/3, **SURVIVED 3/3** | logic, ui.website, ui.controller represent an application of Model-View-Controller pattern | a package-level design claim, not a Logic responsibility |
| 39 | teammates | 22 | UI | `canonical` | `UI` | `p` | borderline | yes | SWATTR, Artemis(1/3), nameJudge 0/3 | logic, ui.website, ui.controller represent an application of Model-View-Controller pattern | same sentence, same reason |
| 40 | teammates | 23 | UI | `canonical` | `UI` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | ui.website is not a real package. | a claim about a package name, not about the UI component |
| 41 | teammates | 24 | UI | `alias` | `front-end` | `g` | yes |  | Artemis(1/3), nameJudge 3/3, corefJudge 0/3, **SURVIVED 3/3** | It is a conceptual package representing the front-end of the application | 'front-end' is the run's alias for UI and it genuinely denotes it; gold starts linking UI at the next sentence |
| 42 | teammates | 26 | UI | `canonical` | `UI` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | ui.website is not a Java package. | package-name claim |
| 43 | teammates | 31 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3, corefJudge 0/3 | First, request received by the GAE server. | the request pipeline, nothing to do with the datastore |
| 44 | teammates | 35 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3, corefJudge 0/2 | the automated (GAE server-invoked) requests | the app server invokes them, not the datastore |
| 45 | teammates | 49 | Common | `canonical` | `Common` | `o` | no | yes | nameJudge 0/3 | The most common format is JsonResult | pure ordinary English; the claim is about ActionResult formats |
| 46 | teammates | 50 | UI | `alias` | `front-end` | `g` | yes |  | nameJudge 3/3, **SURVIVED 3/3** | the browser which will then process it on the front-end | 'the front-end' is the UI alias and the processing is genuinely the UI's; gold is silent |
| 47 | teammates | 58 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | requests sent automatically by the GAE server | app server, not datastore |
| 48 | teammates | 62 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | requests generated by the GAE server are equipped with this privilege | app server, not datastore |
| 49 | teammates | 69 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | GAE server sends such automated requests through two different configurations | app server, not datastore |
| 50 | teammates | 73 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | queued by users ... but executed by GAE | the task-queue service of the platform |
| 51 | teammates | 74 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3, corefJudge 0/2 | the 1 minute standard request processing limit imposed by GAE | a platform quota |
| 52 | teammates | 79 | Logic | `canonical` | `Logic` | `o` | yes | yes | SWATTR, nameJudge 0/3 | Managing relationships between entities, e.g. cascade logic | the matched words are ordinary, but the bullet IS one of the Logic component's listed responsibilities (s77-s78); gold stops at s78 |
| 53 | teammates | 83 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | Connecting to GAE-provided or third-party APIs | a Logic responsibility bullet naming the platform |
| 54 | teammates | 84 | Logic | `canonical` | `Logic` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | Package overview contains logic.api, logic.core. | a package inventory of the Logic component; no responsibility claim |
| 55 | teammates | 85 | Logic | `canonical` | `Logic` | `p` | yes | yes | SWATTR, nameJudge 0/3, corefJudge 1/3, **SURVIVED 1/3** | logic.api provides the API of the component to be accessed by the UI | logic.api IS the Logic component's API; gold assigns only UI here |
| 56 | teammates | 86 | Logic | `canonical` | `Logic` | `p` | yes | yes | SWATTR, nameJudge 0/3 | logic.core contains the core logic of the system | the Logic component's own core package; gold is silent |
| 57 | teammates | 94 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 0/3 | It connects to GAE's task queue API. | the task queue, not the datastore |
| 58 | teammates | 117 | Logic | `canonical` | `Logic` | `o` | borderline | yes | SWATTR, nameJudge 0/3 | Refer to the API for the cascade logic. | a cross-reference; cascade logic is indeed handled by Logic (s129) but nothing is asserted here |
| 59 | teammates | 119 | Logic | `canonical` | `Logic` | `o` | no | yes | SWATTR, nameJudge 0/3 | It contains minimal logic beyond what is directly relevant to CRUD operations | the subject is the Storage component; gold assigns Storage |
| 60 | teammates | 125 | Storage | `canonical` | `Storage` | `p` | yes | yes | SWATTR, Artemis(1/3), nameJudge 0/3, corefJudge 3/3, **SURVIVED 3/3** | Classes in the storage.entity package are not visible outside this component | 'this component' IS Storage; gold links s123 and s128 but not this one |
| 61 | teammates | 127 | Common | `canonical` | `Common` | `p` | borderline | yes | SWATTR, nameJudge 0/3, corefJudge 0/3 | These datatransfer classes are in common.datatransfer package | a forward pointer inside the Storage section |
| 62 | teammates | 130 | Storage | `canonical` | `Storage` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | Package overview contains storage.api, storage.entity, storage.search. | package inventory of Storage |
| 63 | teammates | 131 | Storage | `canonical` | `Storage` | `p` | yes | yes | SWATTR, nameJudge 0/3, corefJudge 0/3 | storage.api provides the API of the component to be accessed by the logic component | storage.api IS Storage's API; gold assigns only Logic here |
| 64 | teammates | 132 | Storage | `canonical` | `Storage` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | storage.entity contains classes that represent persistable entities | package inventory |
| 65 | teammates | 133 | Storage | `canonical` | `Storage` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | storage.search contains classes for dealing with searching and indexing | package inventory |
| 66 | teammates | 139 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 2/3, **SURVIVED 2/3** | eventual consistency involving multiple servers in the GAE production environment | a deployment-environment remark |
| 67 | teammates | 140 | Test Driver | `partial_d` | `test` | `o` | no | yes | nameJudge 0/3 | it is expected to avoid test failures caused by eventual consistency | ordinary use of 'test'; the subject is datastore consistency |
| 68 | teammates | 144 | GAE Datastore | `alias` | `GAE` | `t` | no | yes | nameJudge 3/3, **SURVIVED 3/3** | due to limitations of GAE environment | a platform limitation, not the datastore component |
| 69 | teammates | 156 | Common | `canonical` | `Common` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | Package overview contains common.util, common.exceptions, common.datatransfer. | package inventory of Common |
| 70 | teammates | 157 | Common | `canonical` | `Common` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | common.util contains utility classes. | package inventory |
| 71 | teammates | 158 | Common | `canonical` | `Common` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | common.exceptions contains custom exceptions. | package inventory |
| 72 | teammates | 159 | Common | `canonical` | `Common` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | common.datatransfer contains data transfer objects. | package inventory |
| 73 | teammates | 160 | Common | `canonical` | `Common` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | common.datatransfer package contains lightweight data transfer object classes | package inventory |
| 74 | teammates | 169 | Test Driver | `canonical` | `Test Driver` | `p` | borderline | yes | nameJudge 0/3 | Package overview contains test.driver, test.cases and subpackages. | package inventory of the Test Driver component |
| 75 | teammates | 170 | Test Driver | `canonical` | `Test Driver` | `p` | yes | yes | nameJudge 0/3 | test.driver contains infrastructure and helpers needed for running the tests | that package IS the Test Driver component; gold is silent |
| 76 | teammates | 171 | Test Driver | `partial_d` | `test` | `p` | borderline | yes | nameJudge 0/3 | test.cases contains test cases. | package inventory |
| 77 | teammates | 172 | Logic | `canonical` | `Logic` | `p` | no | yes | nameJudge 0/3 | Sub-packages contains x.testdriver, x.datatransfer, x.util, x.logic, x.storage | x.logic is a TEST package, not the Logic component |
| 78 | teammates | 172 | Storage | `canonical` | `Storage` | `p` | no | yes | nameJudge 0/3 | Sub-packages contains ... x.logic, x.storage, x.search, x.webapi | x.storage is a TEST package, not the Storage component |
| 79 | teammates | 173 | Test Driver | `canonical` | `Test Driver` | `g` | yes |  | SWATTR, nameJudge 3/3, **SURVIVED 3/3** | x.testdriver contains component test cases for testing the test driver infrastructure and helpers | the object under test is literally the Test Driver component; gold links Test Driver only at s1/s10/s163/s168 |
| 80 | teammates | 174 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | component test cases for testing the datatransfer objects from the Common component | the claim is about Common; gold assigns Common |
| 81 | teammates | 175 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | testing the utility classes from the Common component | the claim is about Common; gold assigns Common |
| 82 | teammates | 176 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | x.logic contains component test cases for testing the Logic component | the claim is about Logic; gold assigns Logic |
| 83 | teammates | 177 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | x.storage contains component test cases for testing the Storage component | the claim is about Storage; gold assigns Storage |
| 84 | teammates | 178 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | x.search contains component test cases for testing the search functions | the claim is about the search functions |
| 85 | teammates | 179 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | x.webapi contains system test cases for testing the user-invoked actions | the claim is about the web API actions |
| 86 | teammates | 180 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | x.automated contains system test cases for testing the system-automated actions | the claim is about the automated actions |
| 87 | teammates | 181 | Test Driver | `partial_d` | `test` | `o` | borderline | yes | nameJudge 0/3 | Some Component tests are pure unit tests | a taxonomy of test kinds, not a claim about the Test Driver component |
| 88 | teammates | 182 | UI | `alias` | `front-end` | `q` | borderline | yes | nameJudge 3/3, **SURVIVED 3/3** | Front-end files (particularly TypeScript) are tested separately with Jest | the claim is about the testing setup; 'front-end' modifies 'files' |
| 89 | teammates | 183 | Test Driver | `partial_d` | `test` | `o` | no | yes | nameJudge 0/3 | The test cases are found in x.spec.ts files. | a file-location remark |
| 90 | teammates | 187 | E2E | `canonical` | `E2E` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | Package overview contains e2e.util, e2e.pageobjects, e2e.cases, x.util, x.e2e, x.lnp | package inventory of the E2E component |
| 91 | teammates | 188 | E2E | `canonical` | `E2E` | `p` | yes | yes | SWATTR, Artemis, nameJudge 0/3 | e2e.util contains helpers needed for running E2E tests | that package IS the E2E component; gold is silent |
| 92 | teammates | 189 | E2E | `canonical` | `E2E` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | e2e.pageobjects contains abstractions of the pages as they appear on a Browser | package inventory |
| 93 | teammates | 190 | E2E | `canonical` | `E2E` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | e2e.cases contains test cases. | package inventory |
| 94 | teammates | 190 | Test Driver | `partial_d` | `test` | `o` | no | yes | nameJudge 0/3 | e2e.cases contains test cases. | 'test cases' is ordinary; the package belongs to E2E |
| 95 | teammates | 191 | Test Driver | `partial_d` | `test` | `l` | no | yes | nameJudge 0/3 | x.util contains component test cases for testing the test helpers | the claim is about the test helpers of E2E |
| 96 | teammates | 192 | E2E | `canonical` | `E2E` | `p` | borderline | yes | Artemis(1/3), nameJudge 0/3 | x.e2e contains system test cases for testing the application as a whole | x.e2e is a test package for the E2E component |
| 97 | teammates | 192 | Test Driver | `partial_d` | `test` | `o` | no | yes | nameJudge 0/3 | x.e2e contains system test cases | ordinary use of 'test' |
| 98 | teammates | 195 | Client | `canonical` | `Client` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | Package overview contains client.util, client.remoteapi, client.scripts. | package inventory of the Client component |
| 99 | teammates | 196 | Client | `canonical` | `Client` | `p` | yes | yes | SWATTR, nameJudge 3/3, **SURVIVED 3/3** | client.util contains helpers needed for client scripts | that package IS the Client component; gold links Client only at s1/s18/s19/s194 |
| 100 | teammates | 197 | Client | `canonical` | `Client` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | client.remoteapi classes needed to connect to the back end directly | package inventory |
| 101 | teammates | 197 | Logic | `alias` | `back end` | `q` | no | yes | nameJudge 3/3, **SURVIVED 3/3** | classes needed to connect to the back end directly | 'the back end' is the whole server side; the claim is about client.remoteapi |
| 102 | teammates | 198 | Client | `canonical` | `Client` | `p` | borderline | yes | SWATTR, nameJudge 0/3 | client.scripts scripts that deal with the back end data | package inventory |
| 103 | teammates | 198 | Logic | `alias` | `back end` | `q` | no | yes | nameJudge 3/3, **SURVIVED 3/3** | scripts that deal with the back end data for administrative purposes | 'the back end data' is the datastore contents; the claim is about client.scripts |
| 104 | teastore | 11 | WebUI | `alias` | `UI` | `q` | no |  | nameJudge 0/3 | It matches the provided product ID or UI name | 'UI name' is an image FILENAME category; the claim is the Image Provider's lookup; gold assigns ImageProvider |
| 105 | teastore | 12 | WebUI | `alias` | `UI` | `q` | no |  | nameJudge 0/3 | If the product ID or UI name is not available to the Image Provider | 'UI name' again; gold assigns ImageProvider |
| 106 | teastore | 13 | WebUI | `alias` | `UI` | `q` | no |  | nameJudge 0/3 | If the product ID or UI name is found but not in the requested size | 'UI name' again; the claim is image scaling |
| 107 | teastore | 15 | WebUI | `alias` | `UI` | `q` | no |  | nameJudge 0/3 | If the product ID or UI name and size is found, the image will be loaded and delivered | 'UI name' again; the claim is image delivery |

## Does Challenge 2 hold?

**Yes as a phenomenon. But the paper's illustration is narrower than the data, the
mode is far bigger before our judges than after them, and among what survives it is
entangled with gold-standard coverage.**

Of the 213 distinct false positives in the study, 107 are type 1 -- the lexical
trap the paper describes -- against 106 type 2. So as a share of all false
positives the mode Challenge 2 names is 50.2%: real and frequent.

Per system it is lopsided:

- **SWATTR** (39 type-1 items, vs 1 type-2). Being purely lexical, SWATTR fails
  almost only in this mode -- but 26
  of its 39 items are package-path / enumeration matches in TEAMMATES
  (`logic.api`, `storage.entity`, `e2e.util`), and only
  11 are prose traps of the kind the paper draws. Challenge 2 is SWATTR's
  failure mode, but mostly in its dullest form.
- **Artemis** (9 type-1 items, vs 35 type-2). The LLM baseline barely falls for
  the lexical trap at all, and 5
  of its 9 items are genuine mentions the gold does not cover. Artemis fails by
  topical inference instead. **Challenge 2 as written does not characterise the LLM
  baseline.**
- **Our scans before judging** (107/107): the proposal stage walks into every
  instance of the trap, which is exactly the motivation for having a validator.
- **Our final output** (24 type-1 vs **1** type-2). The judges remove
  77.6% of the trap, but everything they fail to remove is type 1. For
  our own approach Challenge 2 is not just real, it is the only false-positive mode
  left standing.

The caveat: only 9 of our 24 survivors are wrong under any reading. The
other 15 are gold-silent `yes`/`borderline` -- package-inventory sentences
(*logic.api provides the API of the component*), section-internal elaborations (*the
conversion process sends progress messages*), or a component the gold never links
anywhere (BigBlueButton's Recording Service). Scoring them as errors understates our
precision and inflates how big Challenge 2 looks in our own results.

### What the paper should say instead

1. **Keep the challenge and keep the lexical illustration** -- it is attested, and two
   items are near-perfect exhibits. BigBlueButton s60: *"Communication between apps
   and FreeSWITCH Event Socket Layer (fsels) uses messages through redis pubsub"*,
   linked to **FreeSWITCH** although the phrase names **FSESL** -- and it is produced
   by SWATTR, by Artemis and by us. MediaStore s27/s29: *"encapsulates database
   access"* linked to the **Database** component, which is the paper's own
   "database access" example occurring verbatim, and it survives our pipeline.
2. **Do not present the trap as one thing.** It decomposes into at least five
   mechanisms and the two largest are not the one the paper draws: a *third-party
   technology* name the component is named after (`t`, 15 items -- "the GAE
   server", "WebRTC provides the user with...", "implements both SFU and MCU
   models") and a *qualifier inside a compound* (`q`, 20 items -- "database
   access", "client-side interactions", "UI name", "the back end"). The
   "responsibility of another component" reading on its own (`r`) is 2 items.
   It is the right frame; it is not the bulk.
3. **Concede the identifier case.** 41 items (38.3%) are
   not prose: the SAD lists package paths (`logic.api`, `storage.entity`, `e2e.util`,
   `x.logic`) and any tokenizer sees a component name in them. That is a
   document-genre artefact -- TEAMMATES' package overviews account for
   40 of them --
   not a plausibility problem. It deserves one sentence of its own rather than being
   folded into Challenge 2, and it is the main reason a purely lexical baseline looks
   bad on TEAMMATES.
4. **Say whose problem it is, and name the residual honestly.** The evidence supports
   the validator: 107 type-1 proposals -> 24 kept. What it lets through is not
   random: it is dominated by *document aliases* the run itself introduced
   (`database`, `back end`, `front-end`, `GAE`, `Recording Processor`), which the
   judge treats as settled evidence. If the paper wants to state a limitation that
   its own numbers still carry, that is the honest one -- alias-mediated lexical
   traps -- rather than the generic "plausible but unsupported" wording.
5. **Do not generalise Challenge 2 to LLM-based recovery.** On this benchmark the LLM
   baseline's false positives are overwhelmingly type 2 (35 vs 9). If the motivation
   section needs a challenge that describes LLM behaviour, it is topical inference
   over a neighbourhood, which the current text does not name.
