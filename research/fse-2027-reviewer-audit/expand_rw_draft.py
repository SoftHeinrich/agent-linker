#!/usr/bin/env python3
"""Add API-sourced bibliography records for the standalone related-work draft.

The evidence sheet supplies only record locators and IDs. Bibliographic fields
come from a separate Crossref or arXiv API request for each selected work.
"""

import csv
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
IDS = ("E007", "E009", "E010", "E022", "E030", "E031", "E039", "E040", "E041", "E059", "E060", "E062", "E063", "E065")
HEADERS = {"User-Agent": "ALinker-rw-draft-api/1.0 (research bibliography)"}
ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        return response.read().decode("utf-8")


def bib_escape(value):
    return value.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")


def crossref(evidence_id, locator):
    doi = locator.split("doi.org/", 1)[1]
    path = urllib.parse.quote(doi, safe="")
    base = "https://api.crossref.org/works/" + path
    record = json.loads(get(base))["message"]
    bib_url = base + "/transform/application/x-bibtex"
    bib = get(bib_url).strip()
    key = re.search(r"@\w+\{([^,]+),", bib)
    if not key:
        raise ValueError(f"No BibTeX key for {evidence_id}")
    published = record.get("published-print") or record.get("published") or record["issued"]
    year = published["date-parts"][0][0]
    bib_year_match = re.search(r"\byear=\{(\d{4})\}", bib)
    if not bib_year_match:
        raise ValueError(f"No BibTeX year for {evidence_id}")
    bib_year = int(bib_year_match.group(1))
    authored_bib = bib
    if bib_year != year:
        authored_bib = re.sub(r"\byear=\{\d{4}\}", f"year={{{year}}}", authored_bib, count=1)
    return {
        "id": evidence_id,
        "kind": "Crossref",
        "metadata_api": base,
        "bibtex_api": bib_url,
        "title": record["title"][0] + (": " + record["subtitle"][0] if record.get("subtitle") else ""),
        "authors": [" ".join(filter(None, (a.get("given"), a.get("family")))) for a in record["author"]],
        "year": year,
        "year_source_field": "published-print" if record.get("published-print") else "published" if record.get("published") else "issued",
        "transform_year": bib_year,
        "venue": (record.get("container-title") or [""])[0],
        "doi": record["DOI"],
        "key": key.group(1),
        "api_bibtex": bib,
        "bibtex": authored_bib,
    }


def arxiv(evidence_id, locator):
    identifier = locator.rsplit("/", 1)[1]
    url = "https://export.arxiv.org/api/query?id_list=" + identifier
    entry = ET.fromstring(get(url)).find(f"{{{ATOM}}}entry")
    if entry is None:
        raise ValueError(f"No arXiv entry for {evidence_id}")
    title = " ".join(entry.findtext(f"{{{ATOM}}}title").split())
    authors = [a.findtext(f"{{{ATOM}}}name") for a in entry.findall(f"{{{ATOM}}}author")]
    year = int(entry.findtext(f"{{{ATOM}}}published")[:4])
    category = entry.find(f"{{{ARXIV}}}primary_category")
    primary = category.attrib.get("term", "") if category is not None else ""
    key = "RW_" + evidence_id
    fields = [
        f"  author = {{{bib_escape(' and '.join(authors))}}}",
        f"  title = {{{bib_escape(title)}}}",
        f"  year = {{{year}}}",
        f"  eprint = {{{identifier}}}",
        "  archivePrefix = {arXiv}",
    ]
    if primary:
        fields.append(f"  primaryClass = {{{primary}}}")
    fields.append(f"  url = {{https://arxiv.org/abs/{identifier}}}")
    bib = "@misc{" + key + ",\n" + ",\n".join(fields) + "\n}"
    return {
        "id": evidence_id,
        "kind": "arXiv",
        "metadata_api": url,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": "arXiv preprint",
        "arxiv_id": identifier,
        "primary_class": primary,
        "key": key,
        "bibtex": bib,
    }


def main():
    with (ROOT / "reviewer_paper_evidence.csv").open(newline="") as f:
        evidence = {r["evidence_id"]: r for r in csv.DictReader(f)}
    records = []
    for evidence_id in IDS:
        row = evidence[evidence_id]
        locator = row["source_url"]
        if locator.startswith("https://doi.org/"):
            record = crossref(evidence_id, locator)
        elif locator.startswith("https://arxiv.org/abs/"):
            record = arxiv(evidence_id, locator)
        else:
            raise ValueError(f"No supported API locator for {evidence_id}: {locator}")
        record["search_sheet_title"] = row["title"]
        record["search_sheet_authors"] = row["authors"]
        records.append(record)
    existing = (REPO / "draft.tex").read_text()
    occupied = set(re.findall(r"@\w+\{([^,]+),", existing))
    for r in records:
        if r["key"] in occupied or r.get("transform_year") != r.get("year"):
            r["api_key"] = r["key"]
            r["key"] = "RW_" + r["id"]
            r["bibtex"] = re.sub(r"(@\w+\{)[^,]+,", r"\g<1>" + r["key"] + ",", r["bibtex"], count=1)
        if r["key"] in occupied:
            raise ValueError(f"Duplicate BibTeX key: {r['key']}")
        occupied.add(r["key"])
    bib = "\n\n".join(f"% {r['id']} | API: {r.get('bibtex_api', r['metadata_api'])}\n{r['bibtex']}" for r in records) + "\n"
    (REPO / "draft-rw-extra.bib").write_text(bib)
    snapshot = {
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": "python3 research/fse-2027-reviewer-audit/expand_rw_draft.py",
        "selection": list(IDS),
        "records": records,
    }
    (ROOT / "rw_draft_extra_api_records.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    lines = ["# API records added to the standalone RW draft", "", f"Command: `{snapshot['command']}`", "", "Each row was fetched independently. Crossref bibliography fields come from its BibTeX transform; when the transform year differs from Crossref's print/published year, the latter API field supplies the draft year. Colliding or year-mismatched local keys are renamed. arXiv entries use only Atom fields. The search sheet supplied locator and evidence ID only.", "", "| ID | API metadata | Title | Authors | Year | Key |", "|---|---|---|---|---:|---|"]
    for r in records:
        lines.append(f"| {r['id']} | [record]({r['metadata_api']}) | {r['title']} | {', '.join(r['authors'])} | {r['year']} | `{r['key']}` |")
    result_line = f"Result: {len(records)} API records fetched; {len([r for r in records if r['kind']=='Crossref'])} Crossref and {len([r for r in records if r['kind']=='arXiv'])} arXiv; all BibTeX keys unique."
    lines.extend(["", result_line, ""])
    lines.extend(["Crossref transform-year differences (online vs. print/published): " + ", ".join(f"{r['id']} {r['transform_year']}→{r['year']}" for r in records if "transform_year" in r and r["transform_year"] != r["year"]) + ".", ""])
    (ROOT / "rw_draft_extra_api_records.md").write_text("\n".join(lines))
    print(result_line)
    for r in records:
        print(r["id"], r["key"], r["title"])


if __name__ == "__main__":
    main()
