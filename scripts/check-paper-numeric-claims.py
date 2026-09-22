#!/usr/bin/env python3
"""Require an evidence-ledger entry for every numeric statement in the paper.

The checker is deliberately lexical.  It does not decide whether a claim is true;
it makes the review set complete and makes any wording change invalidate the old
review.  Review scopes and evidence are recorded in
verification/paper-numeric-claims-policy.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "verification" / "paper-numeric-claims-policy.json"

# These are the authored sources rendered when paper/main.tex has its current
# \showappendixfalse setting.  Generated tables are covered by their generators;
# their inclusion commands are audited here, while cell-by-cell checks remain in
# evaluation/mini-src/check.py and verification/2026-09-21-rq3-real-metrics-check.py.
SOURCES = (
    "paper/main.tex",
    "paper/sections/intro.tex",
    "paper/sections/motivation.tex",
    "paper/sections/approach.tex",
    "paper/sections/metric.tex",
    "paper/sections/eval.tex",
    "paper/sections/results.tex",
    "paper/sections/discussion.tex",
    "paper/sections/rw.tex",
    "paper/sections/conclusion.tex",
)

KINDS = {"measured", "reported", "setup", "method", "example", "notation"}
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred hundreds thousand thousands million millions "
    "billion billions first second third fourth fifth sixth seventh eighth ninth tenth "
    "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
    "nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth "
    "ninetieth hundredth thousandth millionth billionth single"
).split()
TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?:[+-]?(?:\d{1,3}(?:\{,\}\d{3})+|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\\,)?(?:\\%|%|pp|\\times|×|x)?|"
    + "|".join(NUMBER_WORDS)
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)
REMOVE_ARGUMENT_RE = re.compile(
    r"\\(?:cite|autoref|ref|label|input|includegraphics)\*?"
    r"(?:\[[^]]*\])?\{[^{}]*\}"
)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)


@dataclass(frozen=True)
class Finding:
    source: str
    line: int
    statement: str
    tokens: str
    fingerprint: str


def strip_comment(line: str) -> str:
    """Remove a TeX comment, retaining escaped percent signs."""
    for index, character in enumerate(line):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return line[:index]
    return line


def normalize(line: str) -> str:
    return " ".join(line.strip().split())


def fingerprint(source: str, statement: str) -> str:
    payload = f"{source}\0{statement}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def relevant_main_lines(text: str) -> set[int]:
    """Select the abstract only; main.tex's preamble numbers are formatting."""
    match = ABSTRACT_RE.search(text)
    if not match:
        raise ValueError("paper/main.tex has no abstract environment")
    first = text.count("\n", 0, match.start(1)) + 1
    last = first + match.group(1).count("\n")
    return set(range(first, last + 1))


def scan() -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for source in SOURCES:
        path = ROOT / source
        text = path.read_text(encoding="utf-8")
        selected = relevant_main_lines(text) if source == "paper/main.tex" else None
        for line_number, raw in enumerate(text.splitlines(), start=1):
            if selected is not None and line_number not in selected:
                continue
            statement = normalize(strip_comment(raw))
            if not statement:
                continue
            searchable = statement
            previous = None
            while previous != searchable:
                previous = searchable
                searchable = REMOVE_ARGUMENT_RE.sub("", searchable)
            tokens = tuple(match.group(0) for match in TOKEN_RE.finditer(searchable))
            if not tokens:
                continue
            key = (source, statement)
            if key in seen:
                raise ValueError(
                    f"duplicate numeric statement in {source}; split or distinguish it: {statement}"
                )
            seen.add(key)
            findings.append(
                Finding(
                    source=source,
                    line=line_number,
                    statement=statement,
                    tokens=" | ".join(tokens),
                    fingerprint=fingerprint(source, statement),
                )
            )
    return findings


def inventory_digest(findings: list[Finding]) -> str:
    payload = "\n".join(
        f"{finding.fingerprint}\t{finding.tokens}" for finding in findings
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_policy() -> dict[str, object]:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def evidence_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for item in value.split(";"):
        item = item.strip()
        if not item or item == "not-applicable":
            continue
        paths.append(ROOT / item.split("#", 1)[0])
    return paths


def is_tracked(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "paper":
        repository = ROOT / "paper"
        query = Path(*relative.parts[1:])
    else:
        repository = ROOT
        query = relative
    result = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--", str(query)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return bool(result.stdout.strip())


def matching_scopes(finding: Finding, scopes: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        scope
        for scope in scopes
        if scope["source"] == finding.source
        and int(scope["first_line"]) <= finding.line <= int(scope["last_line"])
    ]


def check(findings: list[Finding], policy: dict[str, object]) -> list[str]:
    errors: list[str] = []
    expected_digest = str(policy.get("inventory_sha256", ""))
    actual_digest = inventory_digest(findings)
    if actual_digest != expected_digest:
        errors.append(
            "numeric-statement inventory changed: "
            f"expected {expected_digest or '<unset>'}, found {actual_digest}; "
            "review the full --report output before accepting the new digest"
        )

    scopes = policy.get("scopes")
    if not isinstance(scopes, list):
        return errors + ["policy scopes must be a list"]
    scope_ids: set[str] = set()
    for index, scope in enumerate(scopes, start=1):
        if not isinstance(scope, dict):
            errors.append(f"scope {index}: must be an object")
            continue
        scope_id = str(scope.get("id", ""))
        if not scope_id or scope_id in scope_ids:
            errors.append(f"scope {index}: missing or duplicate id {scope_id!r}")
        scope_ids.add(scope_id)
        kind = str(scope.get("kind", ""))
        if kind not in KINDS:
            errors.append(f"scope {scope_id}: unknown kind {kind!r}")
        rationale = str(scope.get("rationale", ""))
        evidence = str(scope.get("evidence", ""))
        if not rationale.strip():
            errors.append(f"scope {scope_id}: rationale is empty")
        if kind == "notation" and evidence != "not-applicable":
            errors.append(f"scope {scope_id}: notation must use evidence=not-applicable")
        if kind != "notation" and not evidence.strip():
            errors.append(f"scope {scope_id}: {kind} scope has no evidence")
        for path in evidence_paths(evidence):
            if not path.exists():
                errors.append(f"scope {scope_id}: evidence path does not exist: {path.relative_to(ROOT)}")
            elif not is_tracked(path):
                errors.append(f"scope {scope_id}: evidence path is not tracked: {path.relative_to(ROOT)}")

    for finding in findings:
        location = f"{finding.source}:{finding.line}"
        matches = matching_scopes(finding, scopes)
        if not matches:
            errors.append(f"{location}: UNREVIEWED {finding.tokens}: {finding.statement}")
        elif len(matches) > 1:
            ids = ", ".join(str(scope["id"]) for scope in matches)
            errors.append(f"{location}: covered by multiple scopes: {ids}")
    return errors


def report(findings: list[Finding], policy: dict[str, object] | None) -> None:
    print(f"inventory_sha256={inventory_digest(findings)}")
    scopes = policy.get("scopes", []) if policy else []
    for finding in findings:
        matches = matching_scopes(finding, scopes) if isinstance(scopes, list) else []
        scope = str(matches[0]["id"]) if len(matches) == 1 else "UNREVIEWED"
        print(
            f"{finding.source}:{finding.line}\t{finding.fingerprint}\t"
            f"{finding.tokens}\t{scope}\t{finding.statement}"
        )


def self_test() -> None:
    assert strip_comment(r"value 12\% % note") == r"value 12\% "
    assert strip_comment("value 12% note") == "value 12"
    sample = REMOVE_ARGUMENT_RE.sub(
        "", r"$+6.9$pp, five runs, $3{,}622$, F_1, \cite{x2024}"
    )
    assert [m.group(0) for m in TOKEN_RE.finditer(sample)] == [
        "+6.9",
        "five",
        "3{,}622",
        "1",
    ]
    assert [m.group(0) for m in TOKEN_RE.finditer(
        "a single file, hundreds of links, and the thirteenth run"
    )] == ["single", "hundreds", "thirteenth"]
    assert REMOVE_ARGUMENT_RE.sub("", r"three runs~\cite{x2024}") == "three runs~"
    original = Finding("paper/test.tex", 1, "three runs", "three", "aaa")
    changed = Finding("paper/test.tex", 1, "four runs", "four", "bbb")
    assert inventory_digest([original]) != inventory_digest([changed])
    print("PASS self-test: TeX comments, numeric tokens, citation stripping, and change detection")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print every active numeric statement")
    parser.add_argument("--self-test", action="store_true", help="run scanner unit checks first")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    findings = scan()
    policy = load_policy() if POLICY.exists() else None
    if args.report:
        report(findings, policy)
    if policy is None:
        print(f"FAIL paper numeric-claim audit: missing {POLICY.relative_to(ROOT)}")
        return 1
    errors = check(findings, policy)
    counts: dict[str, int] = {kind: 0 for kind in sorted(KINDS)}
    for finding in findings:
        matches = matching_scopes(finding, policy["scopes"])
        if len(matches) == 1 and matches[0]["kind"] in counts:
            counts[str(matches[0]["kind"])] += 1
    if errors:
        print(f"FAIL paper numeric-claim audit: {len(errors)} error(s); {len(findings)} active statements")
        for error in errors:
            print(f"- {error}")
        return 1
    summary = ", ".join(f"{kind}={count}" for kind, count in counts.items())
    print(f"PASS paper numeric-claim audit: {len(findings)} active statements reviewed")
    print(f"PASS ledger classifications: {summary}")
    print("PASS every non-notation entry names existing tracked evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
