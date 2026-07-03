#!/usr/bin/env python3
"""Live pilot: can router behavior be integrated into s21's prompt structure?

This intentionally lives under pilot/ and does not modify canonical s21.

Design:
  * frozen s21 run outputs are the floor;
  * typed batch extraction prompts mirror s21 Phase 2 shape;
  * new AFFIRMATIVE candidates use the unchanged s21 P1/P2 entity gate;
  * new CONTRAST candidates use the measured contrast-specific gate;
  * CODEPATH/IMPLICIT/ANAPHORA are not accepted in the precision-safe variants.

The goal is to test the hypothesis that the effective router behavior is a
structured prompt/mode integration, not the current broad agentic router.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.llm_client import LLMBackend, LLMClient
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import (
    LAYERED_ENTITY_RULES,
    P1_FOCUS,
    P2_FOCUS,
)


PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
BENCH = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark",
))
EXTRACTS = ROOT / "results/v2.6.6_extracts_s21/gpt"
CACHE_DIR = ROOT / "pilot/cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = {
    "mediastore": "model_2016/pcm/ms.repository",
    "teastore": "model_2020/pcm/teastore.repository",
    "teammates": "model_2021/pcm/teammates.repository",
    "bigbluebutton": "model_2021/pcm/bbb.repository",
    "jabref": "model_2021/pcm/jabref.repository",
}
GOLD_FILE = {
    "mediastore": "goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore": "goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

MODES = ("AFFIRMATIVE", "CONTRAST", "IMPLICIT", "ANAPHORA", "CODEPATH")
PRECISION_SAFE_MODES = {"AFFIRMATIVE", "CONTRAST"}


VARIANTS = {
    # Closest to the measured GTP setup: ask all modes, but accept only named/contrast.
    "typed_all_filter_named": {
        "builder": "prompt_typed_all",
        "keep_modes": PRECISION_SAFE_MODES,
        "description": "all typed refs emitted; only AFFIRMATIVE/CONTRAST accepted",
    },
    # More direct s21 integration: ask only the deployable model-doc modes.
    "typed_named_only": {
        "builder": "prompt_named_only",
        "keep_modes": PRECISION_SAFE_MODES,
        "description": "extract only AFFIRMATIVE/CONTRAST plus CODEPATH rejection signal",
    },
    # Same scope as typed_named_only, with an explicit externalized scratchpad field.
    "scratchpad_named": {
        "builder": "prompt_scratchpad_named",
        "keep_modes": PRECISION_SAFE_MODES,
        "description": "named-only extraction with signal-before-mode scratchpad",
    },
    # Validator-only lower bound: no new extraction; contrast-gate s21 rejected candidates.
    "validator_contrast_only": {
        "builder": None,
        "keep_modes": PRECISION_SAFE_MODES,
        "description": "no extraction calls; revalidates s21 rejected contrast-like candidates",
    },
}


def load_env() -> None:
    for env_file in (ROOT / ".env", Path("/mnt/hostshare/ardoco-home/.env")):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    os.environ.pop("OPENAI_REASONING_EFFORT", None)
    os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")


def client() -> LLMClient:
    load_env()
    return LLMClient(
        backend=LLMBackend.OPENAI,
        model=os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4"),
        temperature=0.1,
        enable_logging=False,
    )


def parse_json_object(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        return {}
    try:
        return json.loads(text[a:b + 1])
    except Exception:
        return {}


def sentences(project: str) -> dict[int, str]:
    hits = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt"))
    out: dict[int, str] = {}
    if hits:
        with open(hits[0], encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                out[i] = line.strip()
    return out


def roster(project: str) -> dict[str, str]:
    path = BENCH / project / MODEL_FILE[project]
    return {c.name: c.id for c in parse_pcm_repository(path)}


def gold(project: str) -> set[tuple[int, str]]:
    out: set[tuple[int, str]] = set()
    with (BENCH / project / GOLD_FILE[project]).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.add((int(row["sentence"]), row["modelElementID"]))
    return out


def s21_extract(project: str, run: str) -> dict:
    return json.loads((EXTRACTS / run / f"{project}.json").read_text())


def s21_final(project: str, run: str) -> set[tuple[int, str]]:
    d = s21_extract(project, run)
    return {(link["s"], link["c"]) for link in d["final"]["links"]}


def s21_candidates(project: str, run: str) -> set[tuple[int, str]]:
    d = s21_extract(project, run)
    return {(link["s"], link["c"]) for link in d["entity"]["candidates"]}


def ambiguous_names(project: str) -> set[str]:
    out: set[str] = set()
    for run in RUNS:
        d = s21_extract(project, run)
        out.update(d["knowledge"]["model_knowledge"].get("ambiguous_names", []))
    return out


def standalone(name: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text,
                     re.IGNORECASE) is not None


def anchors(project: str, sentence_num: int, component: str, sents: dict[int, str]) -> list[str]:
    out = []
    for i in sorted(sents):
        if i == sentence_num:
            continue
        if standalone(component, sents[i]):
            out.append(f"S{i}: {sents[i]}")
        if len(out) >= 4:
            break
    return out


def catalog_block(names: Iterable[str]) -> str:
    return "\n".join(f"- {name}" for name in names)


def batch_lines(batch: list[tuple[int, str]], with_prev: bool = True) -> str:
    lines = []
    lookup = dict(batch)
    for number, text in batch:
        if with_prev and number - 1 in lookup:
            lines.append(f'S{number} PREV: "{lookup[number - 1]}"')
        lines.append(f'S{number}: "{text}"')
    return "\n".join(lines)


def prompt_typed_all(names: list[str], batch: list[tuple[int, str]]) -> str:
    return f"""Extract documentation references to architecture components.

COMPONENT CATALOG:
{catalog_block(names)}

Read each numbered sentence. Choose components only from the catalog. A sentence can
refer to a component by naming it, by a clear alias or role phrase, by contrast or
negation, by an anaphoric phrase pointing back, or only inside a code/package path.

For every reference, output:
- sentence: the sentence number
- component: exact catalog name
- quote: exact words in the sentence carrying the reference
- mode: one of {", ".join(MODES)}

Mode meanings:
AFFIRMATIVE = the component is plainly named or aliased as an architecture participant.
CONTRAST = the component is named or aliased inside contrast/negation/exclusion.
IMPLICIT = generic role/example phrase without the proper name.
ANAPHORA = pronoun or role phrase pointing back to a component.
CODEPATH = reference occurs only inside a code/package/member path.

Do not output a component unless specific quoted words in the sentence support it.

DOCUMENT:
{batch_lines(batch, with_prev=False)}

Return JSON:
{{"references":[{{"sentence":1,"component":"Name","quote":"exact words","mode":"AFFIRMATIVE"}}]}}
JSON only:"""


def prompt_named_only(names: list[str], batch: list[tuple[int, str]]) -> str:
    return f"""Extract deployable documentation-to-architecture component references.

COMPONENT CATALOG:
{catalog_block(names)}

Read each numbered sentence. Choose components only from the catalog.

Return only these model-doc modes:
AFFIRMATIVE - the component is named or clearly aliased and participates in the described architecture.
CONTRAST - the component is named or clearly aliased inside negation, contrast, or exclusion
  ("not X", "other than X", "unlike X", "rather than X"), but the sentence still says
  something about that component's system role.
CODEPATH - the only reference is a code/package/member path; include it so it can be rejected
  from model-doc scoring.

Do not output IMPLICIT or ANAPHORA references in this variant. Do not output generic
ordinary vocabulary unless the sentence's wording clearly aliases a catalog component.

For every reference, output sentence, component, exact quote, and mode.

DOCUMENT:
{batch_lines(batch, with_prev=False)}

Return JSON:
{{"references":[{{"sentence":1,"component":"Name","quote":"exact words","mode":"AFFIRMATIVE"}}]}}
JSON only:"""


def prompt_scratchpad_named(names: list[str], batch: list[tuple[int, str]]) -> str:
    return f"""Extract deployable documentation-to-architecture component references.

COMPONENT CATALOG:
{catalog_block(names)}

For each reference, FIRST write a short signal field naming the linguistic evidence,
then choose mode. The signal must be one of:
plain-name, alias-name, contrast-name, code-path, none.

Keep only real component references from the catalog:
AFFIRMATIVE - named or clearly aliased architecture participant.
CONTRAST - named or clearly aliased inside negation/contrast/exclusion, while the
  sentence still states a fact about that component.
CODEPATH - only a code/package/member path, not a model-doc link.

Do not output implicit role-only or anaphoric references in this variant.

DOCUMENT:
{batch_lines(batch, with_prev=False)}

Return JSON:
{{"references":[{{"sentence":1,"component":"Name","signal":"plain-name","quote":"exact words","mode":"AFFIRMATIVE"}}]}}
JSON only:"""


PROMPT_BUILDERS: dict[str, Callable[[list[str], list[tuple[int, str]]], str]] = {
    "prompt_typed_all": prompt_typed_all,
    "prompt_named_only": prompt_named_only,
    "prompt_scratchpad_named": prompt_scratchpad_named,
}


def normalize_refs(raw: list[dict], names_to_ids: dict[str, str], valid_sentences: set[int]) -> list[dict]:
    by_lower = {name.lower(): name for name in names_to_ids}
    out = []
    seen = set()
    for ref in raw:
        try:
            s = int(ref.get("sentence"))
        except Exception:
            continue
        if s not in valid_sentences:
            continue
        component_raw = str(ref.get("component", "")).strip()
        component = by_lower.get(component_raw.lower())
        if not component:
            continue
        mode = str(ref.get("mode", "AFFIRMATIVE")).upper().strip()
        if mode not in MODES:
            mode = "AFFIRMATIVE"
        key = (s, names_to_ids[component], mode)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "sentence": s,
            "component": component,
            "component_id": names_to_ids[component],
            "quote": str(ref.get("quote", component)).strip() or component,
            "mode": mode,
            "signal": str(ref.get("signal", "")).strip(),
        })
    return out


def run_extraction_variant(variant: str, batch_size: int) -> dict[str, list[dict]]:
    spec = VARIANTS[variant]
    if spec["builder"] is None:
        return {project: [] for project in PROJECTS}
    cache_path = CACHE_DIR / f"{variant}_extract_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cl = client()
    builder = PROMPT_BUILDERS[spec["builder"]]
    proposals: dict[str, list[dict]] = {}
    for project in PROJECTS:
        names_to_ids = roster(project)
        names = list(names_to_ids)
        sents = sentences(project)
        proposals[project] = []
        ordered = sorted(sents.items())
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            key = f"{project}|{start}|{batch_size}"
            if key not in cache:
                prompt = builder(names, batch)
                resp = cl.query(prompt, timeout=240)
                parsed = parse_json_object(resp.text if resp.success else "")
                cache[key] = parsed.get("references", [])
                cache_path.write_text(json.dumps(cache, indent=1))
                print(f"  [{variant}] extracted {project} batch {start // batch_size + 1}", file=sys.stderr)
            proposals[project].extend(
                normalize_refs(cache[key], names_to_ids, set(sents))
            )
        print(f"  [{variant}] {project}: {len(proposals[project])} grounded refs", file=sys.stderr)
    return proposals


def case_block(case: dict, i: int, *, anchors_on: bool = False, ambiguity_on: bool = False) -> str:
    lines = [f'Case {i}: "{case["quote"]}" -> {case["component"]}']
    if case.get("preceding"):
        lines.append(f'  [prev: "{case["preceding"]}"]')
    lines.append(f'  SENTENCE: "{case["sentence_text"]}"')
    if case.get("signal"):
        lines.append(f'  signal: {case["signal"]}')
    if ambiguity_on and case.get("is_ambiguous"):
        lines.append("  note: this component name is AMBIGUOUS (often an ordinary word)")
    if anchors_on:
        for anchor in case.get("anchors", [])[:4]:
            lines.append(f"  anchor: {anchor}")
    return "\n".join(lines)


def validation_prompt(cases: list[tuple[int, dict]], focus: str) -> str:
    body = "\n".join(case_block(case, i, anchors_on=True, ambiguity_on=True)
                     for i, case in cases)
    return f"""Validate components in a document. {focus}

{LAYERED_ENTITY_RULES}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none"), then decide approve
true/false based on that claim.

CASES:
{body}

Return JSON:
{{"validations":[{{"case":1,"claim":"<quote or none>","approve":true}}]}}
JSON only:"""


CONTRAST_RULES = (
    "The component appears inside a negation, contrast, or exclusion. Approve when "
    "the sentence still asserts a fact ABOUT THIS component's role in the system: "
    "it is compared against, excluded from, or offered as an alternative to something. "
    "Reject only when the sentence denies that this component is part of the system at "
    "all, or the token is a different entity / product-brand name."
)


def contrast_prompt(cases: list[tuple[int, dict]]) -> str:
    body = "\n".join(case_block(case, i, anchors_on=True, ambiguity_on=True)
                     for i, case in cases)
    return f"""Validate trace links where the component is named in CONTRAST or NEGATION.

{CONTRAST_RULES}

For each case, FIRST quote the exact contrast/negation words, THEN decide approve
true/false.

CASES:
{body}

Return JSON:
{{"validations":[{{"case":1,"claim":"<quote>","approve":true}}]}}
JSON only:"""


def parse_validations(text: str) -> dict[int, bool]:
    obj = parse_json_object(text)
    out: dict[int, bool] = {}
    for item in obj.get("validations", []):
        try:
            idx = int(item["case"])
            val = item.get("approve", item.get("keep"))
            out[idx] = val is True or (isinstance(val, str) and val.strip().lower() == "true")
        except Exception:
            pass
    return out


def build_case(project: str, proposal: dict, sents: dict[int, str], amb: set[str]) -> dict:
    s = proposal["sentence"]
    return {
        **proposal,
        "id": f"{project}|{s}|{proposal['component_id']}|{proposal['mode']}",
        "project": project,
        "sentence_text": sents.get(s, ""),
        "preceding": sents.get(s - 1, ""),
        "is_ambiguous": proposal["component"] in amb,
        "anchors": anchors(project, s, proposal["component"], sents),
    }


def validate_cases(variant: str, cases_by_project: dict[str, list[dict]]) -> dict[str, set[tuple[int, str]]]:
    cache_path = CACHE_DIR / f"{variant}_judge_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cl = client()
    kept: dict[str, set[tuple[int, str]]] = defaultdict(set)

    all_cases = []
    for project, cases in cases_by_project.items():
        for case in cases:
            all_cases.append((project, case))

    # AFFIRMATIVE: strict s21 P1 and P2.
    for mode in ("AFFIRMATIVE", "CONTRAST"):
        mode_cases = [(p, c) for p, c in all_cases if c["mode"] == mode]
        if not mode_cases:
            continue
        if mode == "AFFIRMATIVE":
            for pass_name, focus in (("p1", P1_FOCUS), ("p2", P2_FOCUS)):
                need = [(p, c) for p, c in mode_cases if f"{pass_name}|{c['id']}" not in cache]
                for start in range(0, len(need), 8):
                    batch = need[start:start + 8]
                    indexed = [(i, c) for i, (_p, c) in enumerate(batch, 1)]
                    resp = cl.query(validation_prompt(indexed, focus), timeout=180)
                    parsed = parse_validations(resp.text if resp.success else "")
                    for i, (p, c) in enumerate(batch, 1):
                        cache[f"{pass_name}|{c['id']}"] = bool(parsed.get(i, False))
                    cache_path.write_text(json.dumps(cache, indent=1))
        else:
            need = [(p, c) for p, c in mode_cases if f"contrast|{c['id']}" not in cache]
            for start in range(0, len(need), 8):
                batch = need[start:start + 8]
                indexed = [(i, c) for i, (_p, c) in enumerate(batch, 1)]
                resp = cl.query(contrast_prompt(indexed), timeout=180)
                parsed = parse_validations(resp.text if resp.success else "")
                for i, (p, c) in enumerate(batch, 1):
                    cache[f"contrast|{c['id']}"] = bool(parsed.get(i, False))
                cache_path.write_text(json.dumps(cache, indent=1))

    for project, case in all_cases:
        if case["mode"] == "AFFIRMATIVE":
            ok = bool(cache.get(f"p1|{case['id']}") and cache.get(f"p2|{case['id']}"))
        elif case["mode"] == "CONTRAST":
            ok = bool(cache.get(f"contrast|{case['id']}"))
        else:
            ok = False
        if ok:
            kept[project].add((case["sentence"], case["component_id"]))
    return kept


def validator_contrast_only_cases() -> dict[str, list[dict]]:
    """Lower bound: only s21 candidates rejected because they look contrast-like."""
    cues = re.compile(r"\b(not|no|other than|unlike|rather than|instead of|by contrast)\b", re.I)
    out: dict[str, list[dict]] = defaultdict(list)
    for project in PROJECTS:
        sents = sentences(project)
        amb = ambiguous_names(project)
        seen = set()
        for run in RUNS:
            d = s21_extract(project, run)
            validated = {(x["s"], x["c"]) for x in d["entity"]["validated"]}
            for cand in d["entity"]["candidates"]:
                key = (cand["s"], cand["c"])
                if key in validated or key in seen:
                    continue
                if not cues.search(cand["sentence_text"]):
                    continue
                seen.add(key)
                proposal = {
                    "sentence": cand["s"],
                    "component": cand["component_name"],
                    "component_id": cand["c"],
                    "quote": cand.get("matched_text") or cand["component_name"],
                    "mode": "CONTRAST",
                    "signal": "s21-rejected-contrast",
                }
                out[project].append(build_case(project, proposal, sents, amb))
    return out


def fbeta(p: float, r: float, beta: float = 2.0) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def prf(links: set[tuple[int, str]], truth: set[tuple[int, str]]) -> tuple[float, float, float, float, int, int, int]:
    tp = len(links & truth)
    fp = len(links - truth)
    fn = len(truth - links)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = fbeta(p, r, 2.0)
    return p, r, f1, f2, tp, fp, fn


def macro(link_fn: Callable[[str, str], set[tuple[int, str]]], golds: dict[str, set[tuple[int, str]]]):
    rows = []
    for run in RUNS:
        ps = []
        rs = []
        fs = []
        f2s = []
        for project in PROJECTS:
            p, r, f, f2, *_ = prf(link_fn(project, run), golds[project])
            ps.append(p)
            rs.append(r)
            fs.append(f)
            f2s.append(f2)
        rows.append((sum(ps) / len(ps), sum(rs) / len(rs),
                     sum(fs) / len(fs), sum(f2s) / len(f2s)))
    return rows


def avg(rows, idx: int) -> float:
    return sum(row[idx] for row in rows) / len(rows)


def score_variant(variant: str, proposals: dict[str, list[dict]], kept_new: dict[str, set[tuple[int, str]]]):
    golds = {project: gold(project) for project in PROJECTS}
    base_rows = macro(s21_final, golds)
    base_union = {project: set().union(*(s21_final(project, run) for run in RUNS))
                  for project in PROJECTS}

    prop_by_project = defaultdict(set)
    prop_mode_by_key = {}
    modes_by_project = defaultdict(Counter)
    for project, refs in proposals.items():
        for ref in refs:
            key = (ref["sentence"], ref["component_id"])
            prop_by_project[project].add(key)
            prop_mode_by_key.setdefault((project, *key), ref["mode"])
            modes_by_project[project][ref["mode"]] += 1

    prop_in_union = {
        project: prop_by_project[project] & base_union[project]
        for project in PROJECTS
    }

    def augmented(project: str, run: str) -> set[tuple[int, str]]:
        return s21_final(project, run) | prop_in_union[project] | kept_new.get(project, set())

    aug_rows = macro(augmented, golds)
    new_tp = sum(len(kept_new.get(project, set()) & golds[project]) for project in PROJECTS)
    new_fp = sum(len(kept_new.get(project, set()) - golds[project]) for project in PROJECTS)
    proposal_lookup = {}
    for project, refs in proposals.items():
        for ref in refs:
            proposal_lookup.setdefault((project, ref["sentence"], ref["component_id"]), ref)
    kept_details = []
    for project in PROJECTS:
        sents = sentences(project)
        for s, cid in sorted(kept_new.get(project, set())):
            ref = proposal_lookup.get((project, s, cid), {})
            kept_details.append({
                "project": project,
                "sentence": s,
                "component_id": cid,
                "component": ref.get("component", ""),
                "mode": ref.get("mode", ""),
                "quote": ref.get("quote", ""),
                "gold": (s, cid) in golds[project],
                "sentence_text": sents.get(s, ""),
            })

    extraction_diagnostics = {}
    validation_diagnostics = {}
    for project in PROJECTS:
        base = base_union[project]
        truth = golds[project]
        proposed = prop_by_project[project]
        marginal = proposed - base
        base_missed_gold = truth - base
        surfaced_gold = base_missed_gold & proposed
        kept = kept_new.get(project, set())
        extraction_diagnostics[project] = {
            "base_missed_gold": len(base_missed_gold),
            "proposal_total": len(proposed),
            "proposal_gold": len(proposed & truth),
            "proposal_fp_vs_gold": len(proposed - truth),
            "marginal_proposal_total": len(marginal),
            "marginal_proposal_gold": len(marginal & truth),
            "marginal_proposal_fp_vs_gold": len(marginal - truth),
            "base_missed_gold_surfaced": len(surfaced_gold),
            "base_missed_gold_surfaced_by_mode": dict(Counter(
                prop_mode_by_key.get((project, s, cid), "")
                for s, cid in sorted(surfaced_gold)
            )),
        }
        validation_diagnostics[project] = {
            "kept_total": len(kept),
            "kept_gold": len(kept & truth),
            "kept_fp_vs_gold": len(kept - truth),
            "surfaced_gold_kept": len(surfaced_gold & kept),
            "surfaced_gold_rejected_or_filtered": len(surfaced_gold - kept),
            "kept_by_mode": dict(Counter(
                prop_mode_by_key.get((project, s, cid), "")
                for s, cid in sorted(kept)
            )),
        }
    summary = {
        "variant": variant,
        "description": VARIANTS[variant]["description"],
        "baseline": {
            "P": avg(base_rows, 0), "R": avg(base_rows, 1),
            "F1": avg(base_rows, 2), "F2": avg(base_rows, 3),
        },
        "augmented": {
            "P": avg(aug_rows, 0), "R": avg(aug_rows, 1),
            "F1": avg(aug_rows, 2), "F2": avg(aug_rows, 3),
        },
        "delta": {
            "P": avg(aug_rows, 0) - avg(base_rows, 0),
            "R": avg(aug_rows, 1) - avg(base_rows, 1),
            "F1": avg(aug_rows, 2) - avg(base_rows, 2),
            "F2": avg(aug_rows, 3) - avg(base_rows, 3),
        },
        "new_TP": new_tp,
        "new_FP": new_fp,
        "proposal_counts": {project: len(prop_by_project[project]) for project in PROJECTS},
        "kept_new_counts": {project: len(kept_new.get(project, set())) for project in PROJECTS},
        "kept_new_links": kept_details,
        "mode_counts": {project: dict(modes_by_project[project]) for project in PROJECTS},
        "extraction_diagnostics": extraction_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "per_run_baseline": base_rows,
        "per_run_augmented": aug_rows,
    }
    return summary


def run_variant(variant: str, batch_size: int) -> dict:
    if variant == "validator_contrast_only":
        cases_by_project = validator_contrast_only_cases()
        proposals = {
            project: [
                {
                    "sentence": c["sentence"],
                    "component": c["component"],
                    "component_id": c["component_id"],
                    "quote": c["quote"],
                    "mode": c["mode"],
                    "signal": c.get("signal", ""),
                }
                for c in cases
            ]
            for project, cases in cases_by_project.items()
        }
    else:
        proposals = run_extraction_variant(variant, batch_size)
        keep_modes = VARIANTS[variant]["keep_modes"]
        cases_by_project = {}
        for project, refs in proposals.items():
            sents = sentences(project)
            amb = ambiguous_names(project)
            # judge only marginal proposals never in any s21 final; proposals already
            # present in at least one frozen s21 final are accepted through s21's floor.
            base_union = set().union(*(s21_final(project, run) for run in RUNS))
            marginal = []
            seen = set()
            for ref in refs:
                key = (ref["sentence"], ref["component_id"])
                if ref["mode"] not in keep_modes or key in base_union or key in seen:
                    continue
                seen.add(key)
                marginal.append(build_case(project, ref, sents, amb))
            cases_by_project[project] = marginal

    kept_new = validate_cases(variant, cases_by_project)
    summary = score_variant(variant, proposals, kept_new)
    (CACHE_DIR / f"{variant}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def print_summary(summary: dict) -> None:
    b = summary["baseline"]
    a = summary["augmented"]
    d = summary["delta"]
    print("\n" + "=" * 88)
    print(f"{summary['variant']} — {summary['description']}")
    print("=" * 88)
    print(f"baseline  P={b['P']:.4f} R={b['R']:.4f} F1={b['F1']:.4f} F2={b['F2']:.4f}")
    print(f"augmented P={a['P']:.4f} R={a['R']:.4f} F1={a['F1']:.4f} F2={a['F2']:.4f}")
    print(f"delta     P={d['P']:+.4f} R={d['R']:+.4f} F1={d['F1']:+.4f} F2={d['F2']:+.4f}")
    print(f"new kept: +{summary['new_TP']} TP / +{summary['new_FP']} FP")
    surfaced = sum(
        value["base_missed_gold_surfaced"]
        for value in summary["extraction_diagnostics"].values()
    )
    base_missed = sum(
        value["base_missed_gold"]
        for value in summary["extraction_diagnostics"].values()
    )
    kept_surfaced = sum(
        value["surfaced_gold_kept"]
        for value in summary["validation_diagnostics"].values()
    )
    print(f"extraction: surfaced {surfaced}/{base_missed} base-missed gold links")
    print(f"validation: kept {kept_surfaced}/{surfaced} surfaced base-missed gold links")
    print("proposal counts:", summary["proposal_counts"])
    print("kept_new counts:", summary["kept_new_counts"])
    print("mode counts:")
    for project, counts in summary["mode_counts"].items():
        print(f"  {project:<14} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["validator_contrast_only", "typed_named_only"])
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")
    summaries = []
    for variant in args.variants:
        summaries.append(run_variant(variant, args.batch_size))
        print_summary(summaries[-1])
    (CACHE_DIR / "latest_summaries.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
