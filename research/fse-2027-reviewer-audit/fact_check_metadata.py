#!/usr/bin/env python3
"""Fetch independent Crossref/arXiv metadata for every citation candidate.

Usage: python3 research/fse-2027-reviewer-audit/fact_check_metadata.py
The JSON snapshot and text report are kept beside this script for review.
"""

import csv
import datetime as dt
import json
import re
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "ALinker-metadata-audit/1.0 (academic citation verification)"}
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        return response.read()


def normalize(value):
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in decomposed if not unicodedata.combining(c)))


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    m = json.loads(fetch(url))["message"]
    year = (m.get("published-print") or m.get("published") or m.get("issued"))["date-parts"][0][0]
    return {
        "source": url,
        "doi": m["DOI"],
        "title": m["title"][0] + (": " + m["subtitle"][0] if m.get("subtitle") else ""),
        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m["author"]],
        "year": year,
        "venue": (m.get("container-title") or [""])[0],
        "pages": m.get("page", ""),
        "type": m["type"],
    }


def arxiv(identifier):
    url = "https://export.arxiv.org/api/query?id_list=" + identifier
    entry = ET.fromstring(fetch(url)).find("a:entry", NS)
    if entry is None:
        raise ValueError(f"No arXiv entry for {identifier}")
    return {
        "source": url,
        "arxiv_id": identifier,
        "title": " ".join(entry.findtext("a:title", default="", namespaces=NS).split()),
        "authors": [a.findtext("a:name", default="", namespaces=NS) for a in entry.findall("a:author", NS)],
        "year": int(entry.findtext("a:published", namespaces=NS)[:4]),
        "venue": "arXiv preprint",
        "comment": entry.findtext("{http://arxiv.org/schemas/atom}comment", default=""),
    }


def main():
    with (ROOT / "citation_candidates.csv").open(newline="") as f:
        candidates = list(csv.DictReader(f))
    rows = []
    lines = [
        "# Independent citation metadata check",
        "",
        "Command: `python3 research/fse-2027-reviewer-audit/fact_check_metadata.py`",
        "Configuration: all 18 citation candidates; DOI records from Crossref; arXiv IDs from the official arXiv Atom API; exact title/ordered author/year comparison after punctuation and case normalization.",
        "",
        "| ID | Source | Title | Authors | Venue / pages | Year | Inventory differences |",
        "|---|---|---|---|---|---:|---|",
    ]
    for c in candidates:
        locator = c["doi_or_canonical_url"]
        if "arxiv.org" in locator or "arXiv." in locator:
            identifier = locator.rsplit("/", 1)[-1].removeprefix("arXiv.")
            record = arxiv(identifier)
        else:
            record = crossref(locator.split("doi.org/", 1)[1])
        listed_authors = [a.strip() for a in c["authors"].split(";")]
        discrepancies = []
        if normalize(c["title"]) != normalize(record["title"]):
            discrepancies.append("title")
        if [normalize(a) for a in listed_authors] != [normalize(a) for a in record["authors"]]:
            discrepancies.append("authors")
        if str(record["year"]) not in c["venue_year"]:
            discrepancies.append("year")
        rows.append({"id": c["citation_id"], "inventory": {"title": c["title"], "authors": listed_authors, "venue_year": c["venue_year"]}, "source_record": record, "discrepancies": discrepancies})
        venue_pages = record["venue"] + (", pp. " + record["pages"] if record.get("pages") else "")
        lines.append(f"| {c['citation_id']} | [record]({record['source']}) | {record['title']} | {', '.join(record['authors'])} | {venue_pages} | {record['year']} | {', '.join(discrepancies) or 'none'} |")
    output = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "source_policy": "Crossref DOI registry or official arXiv Atom record, fetched separately for each candidate", "records": rows}
    (ROOT / "citation_metadata_check.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    lines.extend(["", f"Result: {len(rows)} records fetched; {sum(bool(r['discrepancies']) for r in rows)} inventory rows with title, author, or year differences.", ""])
    (ROOT / "citation_metadata_check.md").write_text("\n".join(lines))
    print(lines[-2])
    for r in rows:
        if r["discrepancies"]:
            print(f"{r['id']}: {', '.join(r['discrepancies'])}")


if __name__ == "__main__":
    main()
