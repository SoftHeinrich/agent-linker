# Candidate rubric — large-scale OSS case for the s110 linker

The linker's input contract is small: (1) an architecture document as plain prose,
one sentence per line; (2) a flat list of named components (id, name). Output is a
set of (sentence, component) links. It has no access to code. So a candidate needs:

| # | Criterion | What to record |
|---|-----------|----------------|
| D1 | **Architecture prose** — natural-language document(s) describing the system's structure, naming its parts; NOT API reference, NOT per-function comments | path/URL, license, size in sentences (measured), how components are named in it (proper names? paths? lowercase nouns?), git-tracked? |
| D2 | **Component model** — an explicit, project-authored list of components (not one we invent); 15–80 entries ideal; hierarchical is fine if a level can be chosen | source artifact (MAINTAINERS, OWNERS, moz.build, sigs.yaml, module list, dir tree), count per level, are names the same strings the doc uses? |
| D3 | **Code→component map** — file/dir → component, project-authored | artifact, coverage (% of files mapped), granularity |
| S1 | **Self-supervised sentence-level gold** — any signal that ties a *sentence* to a component without human labelling: code references in the sentence (paths, `:c:func:`, class names) resolvable through D3; hyperlinks to a component's own page; doc file ownership (D3 covering Documentation/); co-change (doc line + code dir in the same commit/PR) | which signals exist, measured count of sentences carrying each, noise estimate (spot-check 10) |
| S2 | **Doc-level gold** (weaker) — file/section labelled with a component (owning-sig, BUG_COMPONENT, MAINTAINERS F:) | count |
| T1 | **Downstream task** the links enable and a way to score it from history: doc-staleness (doc names component removed from code — verify via git), doc-update recommendation (code change in C → which doc sentences; gold from co-change history), review routing, newcomer navigation | concrete protocol + gold source + rough size |
| K | **Killer-case pitch** — one sentence: why would a reader care that links exist for THIS system | |
| C | **Cost/feasibility** — sentences × components; any legal/licence issue; can we vendor the doc into the replication package? | |

Report format: one markdown file per candidate in `candidates/<name>.md`, sections
D1 D2 D3 S1 S2 T1 K C, every number measured (say how), every artifact with a
URL or path, and a final 3-line verdict: READY / READY-WITH-WORK / NOT-READY.
Keep raw artifacts you fetched under /tmp/oss-case/<name>/ (not in the repo).
