# FSE 2027 reviewer and related-work audit

Snapshot date: 2026-09-14 UTC.
Target paper: [\`paper/main.tex\`](../../paper/main.tex#L117-L135) — *ALinker: An LLM Workflow for Software Architectural Traceability Link Recovery*.

## Outcome

The official FSE 2027 Research Papers committee page was captured and parsed into an exhaustive roster of 461 unique public profiles:

- 2 Chairs
- 19 Area Chairs
- 440 Members

The public page identifies the program committee, not the reviewer assigned to any particular submission. FSE’s research-track page says submissions receive at least three reviews, may receive more based on expertise or disagreement, and use double-anonymous review. This sheet therefore clusters public topic-fit signals; it does not infer or expose reviewer assignments.

The actionable shortlist is in [\`reviewer_clusters.csv\`](reviewer_clusters.csv). It prioritizes:

- C1: architecture, architecture knowledge, and traceability;
- C2: requirements, NLP, ambiguity, and trace-link recovery;
- C3: LLM/AI4SE, empirical evidence, and workflow validation;
- C4: program analysis, testing, and software quality;
- C5: software evolution, mining, and adjacent empirical methods.

The citation inventory is in [\`citation_candidates.csv\`](citation_candidates.csv). The strongest direct-fit candidates are CIT-06 (anaphoric ambiguity), CIT-07 (trace-link recovery), CIT-08 (textual + structural recovery), and CIT-09 (query reformulation). Use CIT-01/02/04/18 for architecture/traceability foundations and CIT-11/12/14/15/16/17 only for LLM or empirical-workflow context.

## Files

- [\`fse2027_research_pc.csv\`](fse2027_research_pc.csv): complete official roster snapshot, with role, affiliation, country, and profile URL.
- [\`reviewer_clusters.csv\`](reviewer_clusters.csv): relevance-ranked, evidence-linked shortlist; fit tier A is direct, B is adjacent/contextual.
- [\`citation_candidates.csv\`](citation_candidates.csv): paper-level candidates, claim fit, PC author overlap, DOI/canonical URL, and local BibTeX status.
- [\`verify.py\`](verify.py): offline consistency checker plus optional live roster comparison.
- [\`verification/2026-09-14-fse-2027-reviewer-audit.md\`](../../verification/2026-09-14-fse-2027-reviewer-audit.md): preserved command/configuration/output evidence.

## How to use the sheet

1. Start with the target claim, not with a person. Pick a citation only when its title/abstract/full text supports the exact sentence.
2. Use C1/C2 for the related-work backbone; use C3–C5 to position LLM workflow design and evaluation.
3. Verify every DOI/title/venue against the linked publisher, author, institutional, or repository source before final submission.
4. Check the local BibTeX status. Existing keys are not automatically correct; inspect the entry and cite only after claim-level review.
5. Treat author overlap as a discovery aid, never as a reason to add a citation. Do not imply that any listed person reviewed this paper.
6. Re-run the live check close to submission because committee membership and page content can change.

## Suggested related-work mapping

| Claim in the paper | First candidates | Caution |
|---|---|---|
| Traceability definitions, scope, and open problems | CIT-01, CIT-04 | Keep architecture/lifecycle scope explicit. |
| Architecture knowledge and artifact relationships | CIT-02, CIT-03, CIT-18 | Do not conflate architecture-based testing with TLR. |
| Text/structure signals for TLR | CIT-07, CIT-08 | Compare artifact types and evaluation setting. |
| Vocabulary, glossary, and implicit references | CIT-05, CIT-06, CIT-09 | CIT-06 is requirements-domain evidence; explain the transfer. |
| LLM4SE positioning and empirical usage | CIT-11, CIT-12, CIT-14 | Separate survey/vision/usage evidence from method results. |
| Grounding, validation, and hallucination control | CIT-15, CIT-16, CIT-17 | These are adjacent code-LLM studies, not architecture TLR baselines. |

## Conflict and ethics check

The local paper materials contain prior work coauthored with Anne Koziolek in [\`paper/agent-linker.bib\`](../../paper/agent-linker.bib#L940-L1030); the sheet marks her as \`POTENTIAL_CHECK\`. This is not a determination that a conflict exists. Confirm the FSE conflict-of-interest definition, current-author list, institutional/collaboration history, and any submission-system declarations before submission. Do not use the public committee list to reverse-engineer reviewer identities.

## Reproducible checks

From the repository root:

\`\`\`bash
python3 research/fse-2027-reviewer-audit/verify.py
python3 research/fse-2027-reviewer-audit/verify.py --live
\`\`\`

The offline check verifies row counts, unique official profiles, role totals, cluster membership, citation-author membership, HTTPS evidence URLs, local BibTeX keys, and the explicit conflict-review flag. The live check fetches the official committee page and compares the profile set to this snapshot. Network failure should be recorded in the verification file rather than silently treated as current.

## Primary sources

- Official committee: https://conf.researchr.org/committee/fse-2027/fse-2027-papers-program-committee
- Official research-track scope/review policy: https://conf.researchr.org/track/fse-2027/fse-2027-papers
- FSE 2027 event site: https://conf.researchr.org/home/fse-2027
- Target-paper local source: [\`paper/main.tex\`](../../paper/main.tex#L117-L135)

## Limitations

This is a public-source, relevance-first audit as of 2026-09-14. It does not claim that the 33 shortlisted people are the only experts, that a public profile is current, or that a person will review this submission. Several B-tier rows are intentionally marked for a second publication-level screen. The citation list is a checked starting set, not permission to add all entries to the paper.
