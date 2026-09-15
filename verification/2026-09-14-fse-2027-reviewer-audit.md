# FSE 2027 reviewer and related-work audit verification

Date: 2026-09-14 UTC

## Configuration

- Repository: `/mnt/hostshare/ardoco-home/agent-linker`
- Snapshot source: <https://conf.researchr.org/committee/fse-2027/fse-2027-papers-program-committee>
- Track policy source: <https://conf.researchr.org/track/fse-2027/fse-2027-papers>
- Expected public roster: 461 unique profiles
- Expected roles: 2 Chairs, 19 Area Chairs, 440 Members
- Offline artifact: `research/fse-2027-reviewer-audit/`
- Expanded evidence: 67 paper rows covering 17 shortlisted reviewers
- Fit score: `3 × direct papers + 1 × adjacent papers`

## Commands

```text
python3 research/fse-2027-reviewer-audit/verify.py
python3 research/fse-2027-reviewer-audit/verify.py --live
```

## Results

```text
PASS offline: 461 roster rows; 461 unique official profiles
PASS offline: role counts Chair=2, Area Chair=19, Member=440
PASS offline: 33 shortlisted reviewers across C1-C5
PASS offline: 67 paper-evidence rows; counts and score aggregation checked
PASS offline: 18 citation candidates; PC-author and HTTPS checks passed
PASS offline: local BibTeX PRESENT/ABSENT statuses and conflict flag checked
PASS offline: 461 roster rows; 461 unique official profiles
PASS offline: role counts Chair=2, Area Chair=19, Member=440
PASS offline: 33 shortlisted reviewers across C1-C5
PASS offline: 67 paper-evidence rows; counts and score aggregation checked
PASS offline: 18 citation candidates; PC-author and HTTPS checks passed
PASS offline: local BibTeX PRESENT/ABSENT statuses and conflict flag checked
PASS live: official committee profile set matches snapshot
```

The live check confirms that the public committee profile set still matches the
2026-09-14 snapshot. It does not identify reviewers assigned to a submission.
