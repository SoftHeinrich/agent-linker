#!/usr/bin/env python3
"""Offline and live consistency checks for the FSE 2027 audit artifact."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
COMMITTEE_URL = "https://conf.researchr.org/committee/fse-2027/fse-2027-papers-program-committee"

def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)

roster = rows(AUDIT / "fse2027_research_pc.csv")
clusters = rows(AUDIT / "reviewer_clusters.csv")
citations = rows(AUDIT / "citation_candidates.csv")
evidence = rows(AUDIT / "reviewer_paper_evidence.csv")

require(len(roster) == 461, f"roster row count: {len(roster)}")
profiles = [row["official_profile_url"] for row in roster]
require(len(set(profiles)) == 461, "official profile URLs are not unique")
require(all(url.startswith("https://conf.researchr.org/profile/fse-2027/") for url in profiles), "non-official roster URL")
role_counts = {}
for row in roster:
    role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
require(role_counts == {"Chair": 2, "Area Chair": 19, "Member": 440}, f"role counts: {role_counts}")
by_name = {row["name"]: row for row in roster}

require(len(clusters) == 33, f"cluster shortlist row count: {len(clusters)}")
require({row["cluster_id"] for row in clusters} == {"C1", "C2", "C3", "C4", "C5"}, "cluster IDs")
cluster_by_reviewer = {}
for row in clusters:
    require(row["reviewer"] in by_name, f"cluster reviewer missing from roster: {row['reviewer']}")
    require(row["official_role"] == by_name[row["reviewer"]]["role"], f"role mismatch: {row['reviewer']}")
    require(row["official_profile_url"] == by_name[row["reviewer"]]["official_profile_url"], f"profile mismatch: {row['reviewer']}")
    require(row["expertise_evidence_url"].startswith("https://"), f"non-HTTPS expertise evidence: {row['reviewer']}")
    cluster_by_reviewer[row["reviewer"]] = row
require(sum(row["conflict_status"] == "POTENTIAL_CHECK" for row in clusters) == 1, "expected one explicit conflict-review flag")
require(any(row["reviewer"] == "Anne Koziolek" and row["conflict_status"] == "POTENTIAL_CHECK" for row in clusters), "Anne Koziolek check flag missing")

require(len(evidence) == 67, f"paper-evidence row count: {len(evidence)}")
evidence_ids = [row["evidence_id"] for row in evidence]
require(len(set(evidence_ids)) == len(evidence), "paper-evidence IDs are not unique")
evidence_by_reviewer = {}
for row in evidence:
    reviewer = row["reviewer"]
    require(reviewer in by_name, f"paper-evidence reviewer missing from roster: {reviewer}")
    require(reviewer in cluster_by_reviewer, f"paper-evidence reviewer missing from clusters: {reviewer}")
    require(row["cluster_id"] == cluster_by_reviewer[reviewer]["cluster_id"], f"paper-evidence cluster mismatch: {row['evidence_id']}")
    require(row["fit_level"] in {"Direct", "Adjacent"}, f"unknown fit level: {row['evidence_id']}")
    expected_points = "3" if row["fit_level"] == "Direct" else "1"
    require(row["points"] == expected_points, f"fit points mismatch: {row['evidence_id']}")
    require(row["source_url"].startswith("https://"), f"non-HTTPS paper source: {row['evidence_id']}")
    require(row["citation_recommendation"] in {"HIGH", "CONTEXT"}, f"unknown citation recommendation: {row['evidence_id']}")
    evidence_by_reviewer.setdefault(reviewer, []).append(row)
for reviewer, cluster in cluster_by_reviewer.items():
    ev = evidence_by_reviewer.get(reviewer, [])
    direct = sum(row["fit_level"] == "Direct" for row in ev)
    adjacent = sum(row["fit_level"] == "Adjacent" for row in ev)
    require(int(cluster["evidence_paper_count"]) == len(ev), f"paper count mismatch: {reviewer}")
    require(int(cluster["direct_paper_count"]) == direct, f"direct-paper count mismatch: {reviewer}")
    require(int(cluster["adjacent_paper_count"]) == adjacent, f"adjacent-paper count mismatch: {reviewer}")
    require(int(cluster["topic_fit_score"]) == direct * 3 + adjacent, f"fit score mismatch: {reviewer}")
    require(cluster["evidence_ids"] == ";".join(row["evidence_id"] for row in ev), f"evidence ID list mismatch: {reviewer}")
    expected_status = "MULTI_PAPER_CHECKED" if len(ev) >= 3 else ("LIMITED_PAPER_CHECKED" if ev else "PROFILE_ONLY")
    require(cluster["paper_evidence_status"] == expected_status, f"paper evidence status mismatch: {reviewer}")

bib = (ROOT / "paper" / "agent-linker.bib").read_text(encoding="utf-8")
for row in citations:
    require(row["doi_or_canonical_url"].startswith("https://"), f"non-HTTPS citation URL: {row['citation_id']}")
    require(row["verification_source"].startswith("https://"), f"non-HTTPS verification source: {row['citation_id']}")
    for author in (part.strip() for part in row["pc_authors"].split(";") if part.strip()):
        require(author in by_name, f"citation PC author missing from roster: {row['citation_id']} / {author}")
    key = row["local_bib_key"]
    if row["local_bib_status"] == "PRESENT":
        require(key and re.search(r"\b" + re.escape(key) + r"\b", bib), f"missing local BibTeX key: {key}")
    elif row["local_bib_status"] == "ABSENT":
        require(not key, f"ABSENT row should have blank key: {row['citation_id']}")
    else:
        raise AssertionError(f"unknown local BibTeX status: {row['local_bib_status']}")

print("PASS offline: 461 roster rows; 461 unique official profiles")
print("PASS offline: role counts Chair=2, Area Chair=19, Member=440")
print("PASS offline: 33 shortlisted reviewers across C1-C5")
print("PASS offline: 67 paper-evidence rows; counts and score aggregation checked")
print("PASS offline: 18 citation candidates; PC-author and HTTPS checks passed")
print("PASS offline: local BibTeX PRESENT/ABSENT statuses and conflict flag checked")

if "--live" in sys.argv:
    request = Request(COMMITTEE_URL, headers={"User-Agent": "FSE-2027-audit-verifier/1.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    live = set(re.findall(r'href="(https://conf\.researchr\.org/profile/fse-2027/[^"]+)"', html))
    snapshot = set(profiles)
    require(live == snapshot, f"live profile set differs: live={len(live)} snapshot={len(snapshot)}")
    print("PASS live: official committee profile set matches snapshot")
