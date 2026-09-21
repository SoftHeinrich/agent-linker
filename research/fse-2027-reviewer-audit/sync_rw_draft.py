#!/usr/bin/env python3
"""Render API-verified extra bibliography records into the standalone draft."""

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
START = "% BEGIN API RW EXTRA ROWS"
END = "% END API RW EXTRA ROWS"


def tex(value):
    return value.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")


def main():
    snapshot = json.loads((ROOT / "rw_draft_extra_api_records.json").read_text())
    records = snapshot["records"]
    with (ROOT / "reviewer_paper_evidence.csv").open(newline="") as f:
        clusters = {r["evidence_id"]: r["cluster_id"] for r in csv.DictReader(f)}
    draft_path = REPO / "draft.tex"
    draft = draft_path.read_text()
    intro = (
        "The 13 Crossref bibliography entries in \\texttt{draft-api.bib} are copied\n"
        "verbatim from Crossref's BibTeX transform API.  The five arXiv entries were\n"
        "mechanically constructed only from fields returned by the official arXiv API;\n"
        "their API URLs and response statuses are recorded above each entry.\n"
    )
    revised_intro = (
        "The original 18 entries in \\texttt{draft-api.bib} came from individual\n"
        "Crossref or arXiv API records. The 14 additional entries in\n"
        "\\texttt{draft-rw-extra.bib} were fetched individually from the same APIs.\n"
        "For Crossref records, the draft uses the API's print or published year when\n"
        "it differs from the BibTeX transform's online year; the differences and\n"
        "source responses are preserved in the metadata report.\n"
    )
    if intro in draft:
        draft = draft.replace(intro, revised_intro, 1)
    elif revised_intro not in draft:
        raise ValueError("Unexpected draft introduction; refusing to overwrite")
    rows = [START]
    for r in records:
        rows.append(r"\texttt{" + r["id"] + "} & " + tex(r["title"]) + " & " + clusters[r["id"]] + " " + chr(92) * 2)
    rows.append(END)
    block = "\n".join(rows)
    if START in draft:
        draft, n = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda _: block, draft, count=1, flags=re.S)
        if n != 1:
            raise ValueError("Could not replace generated rows")
    else:
        token = r"\texttt{CIT-18} & Using Software Architecture for Code Testing & C1 \\" + "\n"
        if token not in draft:
            raise ValueError("Could not find original citation-map ending")
        draft = draft.replace(token, token + block + "\n", 1)
    matches = list(re.finditer(r"\\nocite\{([^}]*)\}", draft, re.S))
    if len(matches) != 1:
        raise ValueError("Expected one nocite list")
    old_keys = [k.strip() for k in matches[0].group(1).split(",") if k.strip()]
    keys = list(dict.fromkeys(old_keys + [r["key"] for r in records]))
    if len(keys) != 32:
        raise ValueError(f"Expected 32 unique citations, found {len(keys)}")
    nocite = "\\nocite{" + ",\n".join(",".join(keys[i:i + 7]) for i in range(0, len(keys), 7)) + "}"
    draft = draft[:matches[0].start()] + nocite + draft[matches[0].end():]
    draft, n = re.subn(r"\\bibliography\{draft-api(?:,draft-rw-extra)?\}", r"\\bibliography{draft-api,draft-rw-extra}", draft, count=1)
    if n != 1:
        raise ValueError("Expected draft bibliography declaration")
    draft_path.write_text(draft)
    print(f"PASS: draft.tex cites {len(keys)} unique API-sourced entries; {len(records)} added rows")


if __name__ == "__main__":
    main()
