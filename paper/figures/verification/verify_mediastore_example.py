#!/usr/bin/env python3
"""Verify provenance and consistency of the paper's MediaStore example."""

import csv
import glob
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
DB_ID = "_5LN7MLg2EeSNPorBlo7x9g"
MEDIA_ACCESS_ID = "_9eK7YHDrEeSqnN80MQ2uGw"


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL'} {name}")
    return bool(condition)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    document = (ROOT / "benchmark/mediastore/text_2016/mediastore.txt").read_text().splitlines()
    expected = {
        23: "The Database component represents an actual database (e.g., MySQL).",
        24: "It stores user information and meta-data of audio files such as the name and the genre.",
        27: "The MediaAccess component encapsulates database access for meta-data of audio files.",
        28: "Furthermore, it fetches a list of all available audio files.",
    }
    checks = [
        check(
            "benchmark sentences S23/S24/S27/S28 are unchanged",
            all(document[number - 1] == sentence for number, sentence in expected.items()),
        )
    ]

    names = {DB_ID: "DB", MEDIA_ACCESS_ID: "MediaAccess"}
    gold_path = ROOT / "benchmark/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"
    with gold_path.open() as gold_file:
        gold = {
            (int(row["sentence"]), names.get(row["modelElementID"], row["modelElementID"]))
            for row in csv.DictReader(gold_file)
        }
    checks.extend(
        [
            check("gold has S23 -> DB only", (23, "DB") in gold and (23, "MediaAccess") not in gold),
            check("gold has S24 -> DB only", (24, "DB") in gold and (24, "MediaAccess") not in gold),
            check("gold has S27 -> MediaAccess, not DB", (27, "MediaAccess") in gold and (27, "DB") not in gold),
            check("gold has S28 -> MediaAccess only", (28, "MediaAccess") in gold and (28, "DB") not in gold),
        ]
    )

    code_path = ROOT / "benchmark/mediastore/goldstandards/goldstandard_sam_2016-code_2016.csv"
    with code_path.open() as code_file:
        code_rows = list(csv.DictReader(code_file))
    code_counts = {
        component_id: sum(row["ae_id"] == component_id for row in code_rows)
        for component_id in (DB_ID, MEDIA_ACCESS_ID)
    }
    checks.append(check("code gold maps DB to 4 files and MediaAccess to 2", code_counts == {DB_ID: 4, MEDIA_ACCESS_ID: 2}))

    s124_root = ROOT / ".claude/worktrees/coref-shortlist-annot"
    run_dir = s124_root / "results/shortlistmark_e2e_terra_r1_20260914"
    links_path = run_dir / "s_linker124_mediastore_links.csv"
    with links_path.open() as link_file:
        output = {(int(row["sentence"]), row["component_name"], row["source"]) for row in csv.DictReader(link_file)}
    checks.extend(
        [
            check("recorded s124 run links S23 -> DB by name", (23, "DB", "full_name") in output),
            check("recorded s124 run resolves S24 -> DB", (24, "DB", "coreference") in output),
            check(
                "recorded s124 run keeps S27 -> MediaAccess and rejects DB",
                (27, "MediaAccess", "full_name") in output and not any(s == 27 and c == "DB" for s, c, _ in output),
            ),
            check("recorded s124 run resolves S28 -> MediaAccess", (28, "MediaAccess", "coreference") in output),
        ]
    )

    call_paths = glob.glob(str(run_dir / "llm_logs/s_linker124_openai_mediastore_*_calls.json"))
    checks.append(check("one recorded s124 MediaStore call log found", len(call_paths) == 1))
    if len(call_paths) == 1:
        checks.extend(
            [
                check("recorded s124 call log hash matches", sha256(Path(call_paths[0])) == "c3d7bd95b191679a9938197728fee5381e3124c5166d6446b4ec4e1edea72bb3"),
                check("recorded s124 link CSV hash matches", sha256(links_path) == "40996051e6e63147f8f0d5114e8d07d382b60ac928c607eab72b9fd3c6f0f192"),
            ]
        )
    calls = json.loads(Path(call_paths[0]).read_text()) if len(call_paths) == 1 else []
    discovery = json.loads(calls[0]["response_text"]) if calls else {}
    alias_judge = json.loads(calls[1]["response_text"]) if len(calls) > 1 else {}
    proposed = {(item["term"], item["component"]) for group in discovery.values() for item in group}
    approved = {(item["term"], item["component"]) for item in alias_judge.get("approved", [])}
    resolutions = []
    s24_judgment = None
    for call in calls:
        try:
            response = json.loads(call["response_text"])
        except (KeyError, json.JSONDecodeError):
            continue
        resolutions.extend(response.get("resolutions", []))
        if "Case 5: pronoun/role-ref -> DB" in call.get("prompt", ""):
            s24_judgment = next(
                (item for item in response.get("validations", []) if item.get("case") == 5),
                None,
            )
    s24_resolution = next((item for item in resolutions if item.get("sentence") == 24), None)
    s27_rejected = any(
        'Case 19: "database" -> DB' in call.get("prompt", "")
        and any(
            item.get("case") == 19 and item.get("claim") == "none" and item.get("approve") is False
            for item in json.loads(call["response_text"]).get("validations", [])
        )
        for call in calls
    )
    s124_name_format = any(
        'Case 15: "Database" -> DB' in call.get("prompt", "")
        and "Evidence: written=short form" in call.get("prompt", "")
        and 'Case 19: "database" -> DB' in call.get("prompt", "")
        for call in calls
    )
    s124_shortlist = any(
        "TARGET S24:" in call.get("prompt", "")
        and "NAMED BEFORE THIS CASE: DB (S23, linked), Reencoding (S20, linked), Packaging (S19, linked), TagWatermarking (S17, linked), MediaManagement (S17, linked)" in call.get("prompt", "")
        for call in calls
    )
    checks.extend(
        [
            check("run proposes Database -> DB", ("Database", "DB") in proposed),
            check("alias judge approves Database -> DB", ("Database", "DB") in approved),
            check("name judge rejects S27 database -> DB", s27_rejected),
            check("s124 name cases use written=short form", s124_name_format),
            check("s124 S24 shortlist carries linked marks", s124_shortlist),
            check(
                "resolver records S24 It -> DB via S23",
                s24_resolution is not None
                and s24_resolution.get("reference") == "It"
                and s24_resolution.get("component") == "DB"
                and s24_resolution.get("antecedent_sentence") == 23,
            ),
            check(
                "S24 resolver emits the seven recorded bundle fields",
                s24_resolution is not None
                and set(s24_resolution) == {"case", "sentence", "reference", "candidates", "component", "antecedent_sentence", "antecedent_text"},
            ),
            check(
                "S24 coreference judge approves the resolver commitment",
                s24_judgment is not None
                and s24_judgment.get("approve") is True
                and s24_judgment.get("objection") == "none",
            ),
            check(
                "resolver records S28 it -> MediaAccess via S27",
                any(
                    item.get("sentence") == 28
                    and item.get("reference") == "it"
                    and item.get("component") == "MediaAccess"
                    and item.get("antecedent_sentence") == 27
                    for item in resolutions
                ),
            ),
        ]
    )

    live_paths = [
        "paper/sections/motivation.tex",
        "paper/sections/approach.tex",
        "paper/figures/jabref_trace_example.py",
        "paper/figures/coref-linker-illustration-jabref.html",
        "paper/figures/drawio/jabref_trace_example.drawio",
        "paper/figures/coref-linker-illustration.html",
    ]
    live = {path: (ROOT / path).read_text() for path in live_paths}
    prose_paths = ("paper/sections/motivation.tex", "paper/sections/approach.tex")
    old_examples = ("Data Model", "renders the main application window", "ImageProvider", "TeaStore benchmark")
    linker_source = (s124_root / "approach/src/llm_sad_sam/linkers/experimental/s_linker124.py").read_text()
    linker_parent_source = (s124_root / "approach/src/llm_sad_sam/linkers/experimental/s_linker123.py").read_text()
    checks.extend(
        [
            check("s124 source hash matches commit a738981b", hashlib.sha256(linker_source.encode()).hexdigest() == "bb1b83d0f9dc5fe2c93df7eafe48b29b4be65ff65e286d405c69f58e89e13bd0"),
            check("s123 bundle source hash matches commit a738981b", hashlib.sha256(linker_parent_source.encode()).hexdigest() == "404aeaacc45061a285b1681a45e1ae58419227897a59bde12adfbea97ba2060b"),
        ]
    )
    html = live["paper/figures/coref-linker-illustration-jabref.html"]
    coref_html = live["paper/figures/coref-linker-illustration.html"]
    approach = live["paper/sections/approach.tex"]
    checks.extend(
        [
            check(
                "motivation and approach use S23/S24/S27/S28",
                all(all(f"Sentence~{number}" in live[path] for number in expected) for path in prose_paths),
            ),
            check("older running examples removed from live assets", all(not any(old in value for old in old_examples) for value in live.values())),
            check(
                "generator and draw.io contain DB and MediaAccess",
                all("DB" in live[path] and "MediaAccess" in live[path] for path in ("paper/figures/jabref_trace_example.py", "paper/figures/drawio/jabref_trace_example.drawio")),
            ),
            check(
                "all visual sources use S23/S24/S27/S28",
                all(all(f"S{number}" in live[path] for number in expected) for path in live_paths[2:]),
            ),
            check(
                "table uses the implementation's name/coreference proposers",
                'class SLinker124(SLinker123)' in linker_source
                and "<th>proposer</th>" in html
                and '<td class="m">name</td>' in html
                and '<td class="m">coreference</td>' in html,
            ),
            check(
                "invented variant/ordinary form labels are absent",
                "<th>form</th>" not in html and ">variant<" not in html and ">ordinary<" not in html,
            ),
            check(
                "name bundle uses s124's real computed fields",
                'UNION_FIELDS = ("written", "competitors")' in linker_parent_source
                and all(f"\\texttt{{{field}}}" in approach for field in ("span", "written", "competitors", "naming"))
                and all(f"\\texttt{{{field}}}" not in approach for field in ("mention", "alternatives", "last\\_named", "anchors")),
            ),
            check(
                "HTML shows recorded s124 name cases and evidence field",
                all(token in html for token in ("Case 15:", "Case 19:", "written=short form"))
                and "writes=" not in html and "mention=" not in html,
            ),
            check(
                "HTML shows S24 resolver prompt and response bundle",
                all(token in html for token in ("LLM 1 &mdash; resolver prompt", "TARGET S24:", "NAMED BEFORE THIS CASE:", '"candidates":["DB"]', '"antecedent_sentence":')),
            ),
            check(
                "HTML shows S24 judge prompt and response",
                all(token in html for token in ("LLM 2 &mdash; judge prompt", "Case 5: pronoun/role-ref", '"objection":', '"approve":')),
            ),
            check(
                "coreference illustration shows the two recorded LLM prompts",
                "LLM 1 &mdash; resolver prompt" in coref_html
                and "LLM 2 &mdash; judge prompt" in coref_html
                and "Case 5: pronoun/role-ref -&gt; DB" in coref_html,
            ),
            check(
                "coreference illustration lists the seven resolver fields",
                "4 items" not in coref_html
                and "7 fields" in coref_html
                and all(f"<span class=\"mono\">{field}</span>" in coref_html for field in ("case", "sentence", "reference", "candidates", "component", "antecedent_sentence", "antecedent_text")),
            ),
            check(
                "coreference illustration uses the recorded S24 shortlist",
                "DB (S23, linked), Reencoding (S20, linked), Packaging (S19, linked), TagWatermarking (S17, linked), MediaManagement (S17, linked)" in coref_html,
            ),
            check("HTML illustration is structurally closed", "</figure>" in html and "</html>" in html),
            check(
                "generated PDF and PNG exist",
                all((ROOT / path).stat().st_size > 0 for path in ("paper/figures/jabref_trace_example.pdf", "paper/figures/jabref_trace_example.png")),
            ),
        ]
    )
    ET.parse(ROOT / "paper/figures/drawio/jabref_trace_example.drawio")
    checks.append(check("draw.io XML parses", True))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
