"""Extract frozen s_linker20_union per-run phase_caches to neutral stdlib-loadable JSON.

Reads 30 cells (2 backends x 3 runs x 5 projects) from on-disk pickle checkpoints and
writes one deterministic JSON per cell to results/v2.6.6_extracts/<backend>/<run>/<project>.json.

Faithfulness oracle: each JSON's final.links set == that run's *_links.csv on
(sentence, component_id, source).

Exit code: 0 if all 30 cells extracted and PASS; nonzero if any cell is MISSING or FAILs.

No LLM, no network.  Deterministic: re-run produces byte-identical output.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

# Single registration import: registers AliasEntry and transitively pulls data_types_v2.
# Bare module import has no side effects — do NOT instantiate SLinker20Union
# (its __init__ builds an LLMClient; no network occurs on import).
import llm_sad_sam.linkers.experimental.s_linker20_union  # noqa: F401

# ── Matrix ────────────────────────────────────────────────────────────────────
# Directory identifiers — structural path components, NOT benchmark vocabulary (GATE-06).
PROJECTS = ["bigbluebutton", "jabref", "mediastore", "teammates", "teastore"]
RUNS = ["run1", "run2", "run3"]

# (results_root, backend_subdir, friendly_tag)
# Asymmetric: gpt root has an extra gpt/ level; sonnet does not.
# Subdirs differ: openai/ vs claude/  (Pitfall 2 + 3).
MATRIX = [
    ("results/v2.6.5_s20union/gpt", "openai", "gpt"),      # extra gpt/ in root
    ("results/v2.6.5_s20union_sonnet", "claude", "sonnet"),  # no extra level
]

# Cell pkl dir:  {root}/{run}/phase_cache/s_linker20_union/{subdir}/{project}/
# Oracle CSV:    {root}/{run}/{project}/s_linker20_union_{project}_links.csv
# Oracle JSON:   {root}/{run}/{project}/ablation_*.json  (glob; exactly one)

EXTRACT_ROOT = "results/v2.6.6_extracts"

# No-Knowledge matrix (51-04 sweep roots / 51-05 extract).  Same neutral-JSON shape as
# Full — only the results roots, the *_links.csv prefix, and the variant/ablation tags
# differ.  The cell pkl dir stays …/phase_cache/s_linker20_union/<subdir>/<project>/
# because the linker's _VARIANT_NAME is unchanged (Landmine 3), so cell_dir needs NO
# special-casing.  With empty knowledge, aliases serialize as an EMPTY list and
# ambiguous_names as an empty set — identical schema to Full, just empty.
NOKNOW_MATRIX = [
    ("results/v2.6.6_s20union_noknow/gpt", "openai", "gpt"),        # extra gpt/ in root
    ("results/v2.6.6_s20union_noknow_sonnet", "claude", "sonnet"),  # no extra level
]
NOKNOW_EXTRACT_ROOT = "results/v2.6.6_extracts_noknow"

# ── Load ──────────────────────────────────────────────────────────────────────

def load_cell(cell_dir: str) -> dict:
    """Load all 5 pickle layers for one (backend, run, project) cell.

    Args:
        cell_dir: path to the per-cell checkpoint directory
                  (…/phase_cache/s_linker20_union/<subdir>/<project>/).
    Returns:
        dict with keys layer1..layer4, final (each a plain dict).
    """
    cell: dict = {}
    for name in ("layer1", "layer2", "layer3", "layer4", "final"):
        path = os.path.join(cell_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            cell[name] = pickle.load(f)
    return cell


# ── Serialization helpers ─────────────────────────────────────────────────────

def keyed_to_records(d: dict) -> list:
    """Convert a dict[(s:int, c:str), value-dict] to an ordered list of records.

    Each output record carries {"s": s, "c": c, **value}.
    Py3.7+ dict insertion order is preserved.
    """
    return [{"s": s, "c": c, **v} for (s, c), v in d.items()]


def _candidate_link_to_record(cl) -> dict:
    """Serialize a CandidateLink dataclass to a JSON-native dict with s/c keys."""
    d = asdict(cl)
    return {
        "s": d["sentence_number"],
        "c": d["component_id"],
        "component_name": d["component_name"],
        "sentence_text": d["sentence_text"],
        "matched_text": d["matched_text"],
        "source": d["source"],
        "mention_type": d["mention_type"],
        "alias_used": d["alias_used"],
    }


def _sadsam_link_to_record(lk) -> dict:
    """Serialize a SadSamLink dataclass to a JSON-native dict with s/c keys."""
    d = asdict(lk)
    return {
        "s": d["sentence_number"],
        "c": d["component_id"],
        "component_name": d["component_name"],
        "confidence": d["confidence"],
        "source": d["source"],
    }


def _provenance_to_records(prov_dict: dict) -> list:
    """Serialize final_provenance dict[(s,c), value-dict] to an ordered list of records.

    Strips raw_resolution from coref_meta (bulky LLM JSON, audit-only; RESEARCH Open Q2).
    """
    records = []
    for (s, c), v in prov_dict.items():
        entry: dict = {"s": s, "c": c}
        for k, val in v.items():
            if k == "coref_meta" and val is not None:
                # Strip raw_resolution to keep files lean
                val = {mk: mv for mk, mv in val.items() if mk != "raw_resolution"}
            entry[k] = val
        records.append(entry)
    return records


def _coref_metadata_to_records(meta_dict: dict) -> list:
    """Serialize coref_metadata dict[(s,c), value-dict] to an ordered list of records.

    Strips raw_resolution (full LLM JSON, bulky, audit-only; RESEARCH Open Q2).
    """
    records = []
    for (s, c), v in meta_dict.items():
        entry: dict = {"s": s, "c": c}
        for k, val in v.items():
            if k == "raw_resolution":
                continue  # omit by default
            entry[k] = val
        records.append(entry)
    return records


def to_neutral(cell: dict, meta: dict) -> dict:
    """Serialize a loaded cell (dict of 5 pkl layers) to the neutral JSON schema.

    Schema top-level keys: meta, knowledge, entity, coref, final, audit.
    All values are JSON-native (no dataclasses, no tuple keys, no sets, no NaN).
    coref.raw and coref.validated are preserved as LISTS (never collapsed to a keyed
    map) to maintain dup-(s,c) fidelity (8/30 cells have duplicate coref keys).
    """
    l1 = cell["layer1"]
    l3 = cell["layer3"]
    l4 = cell["layer4"]
    lf = cell["final"]

    # ── Knowledge layer ───────────────────────────────────────────────────────
    mk = l1["model_knowledge"]
    dk = l1["doc_knowledge"]
    # doc_knowledge.aliases: dict[str, AliasEntry] at runtime (type hint says str but
    # values are AliasEntry(component, scope)).  Serialize as list of records (Pitfall 6).
    aliases_list = [
        {"term": term, "component": ae.component, "scope": ae.scope}
        for term, ae in dk.aliases.items()
    ]

    # ── Entity layer ──────────────────────────────────────────────────────────
    # entity.candidates / validated: list[CandidateLink] -> list of records
    # entity.decisions: dict[(s,c), {approved,p1,p2,path,stage}] -> list of records
    # entity.evidence_bundles: dict[(s,c), dict] (already plain dicts) -> list of records
    entity_decisions_records = keyed_to_records(l3["decisions"])
    evidence_bundles_records = keyed_to_records(l3["evidence_bundles"])

    # ── Coref layer ───────────────────────────────────────────────────────────
    # coref.raw / coref.validated: list[SadSamLink] -> list of records (NEVER dict-collapse)
    # coref.decisions: dict[(s,c), {approved,path}] -> list of records
    # coref.metadata: dict[(s,c), dict] -> list of records (raw_resolution stripped)
    coref_decisions_records = keyed_to_records(l4["coref_decisions"])
    coref_metadata_records = _coref_metadata_to_records(l4["coref_metadata"])

    # ── Final layer ───────────────────────────────────────────────────────────
    # final.links: list[SadSamLink] (authoritative; do NOT re-derive from decisions)
    # final.provenance: dict[(s,c), {from_coref,source,entity_decision,coref_decision,coref_meta}]
    final_provenance_records = _provenance_to_records(lf["final_provenance"])

    return {
        "meta": meta,
        "knowledge": {
            "model_knowledge": {
                "ambiguous_names": sorted(list(mk.ambiguous_names)),
            },
            "doc_knowledge": {
                "aliases": aliases_list,
            },
        },
        "entity": {
            "candidates": [_candidate_link_to_record(cl) for cl in l3["candidates"]],
            "validated": [_candidate_link_to_record(cl) for cl in l3["validated"]],
            "decisions": entity_decisions_records,
            "evidence_bundles": evidence_bundles_records,
        },
        "coref": {
            "raw": [_sadsam_link_to_record(lk) for lk in l4["coref_raw"]],
            "validated": [_sadsam_link_to_record(lk) for lk in l4["coref_validated"]],
            "decisions": coref_decisions_records,
            "metadata": coref_metadata_records,
        },
        "final": {
            "links": [_sadsam_link_to_record(lk) for lk in lf["final"]],
            "provenance": final_provenance_records,
        },
        "audit": {
            "phase_metrics": lf["phase_metrics"],
        },
    }


def write_json(obj: dict, path: str) -> None:
    """Write a neutral JSON extract with determinism and fail-loud on non-native types.

    Uses indent=2, sort_keys=True, allow_nan=False.  No default= argument:
    any non-native Python type will raise TypeError so serialization bugs surface
    immediately (PATTERNS JSON-writer rule).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)


# ── Faithfulness oracle (EXTRACT-03) ─────────────────────────────────────────

def faithfulness(neutral: dict, csv_path: str, ablation_path: str | None,
                 ablation_key: str = "s_linker20_union") -> tuple[bool, dict]:
    """Assert final.links set == *_links.csv on (sentence, component_id, source).

    Primary gate: set-equality between extract and CSV (authoritative model output).
    Secondary cross-check: ablation_*.json n_links and per-source counts (advisory).

    Args:
        neutral: the to_neutral() output dict.
        csv_path: path to s_linker20_union_<project>_links.csv.
        ablation_path: path to ablation_*.json (secondary cross-check); may be None.

    Returns:
        (ok: bool, detail: dict) — ok is True iff extract_set == csv_set.
        detail carries symmetric difference on failure + secondary check results.
    """
    # Build extract set from final.links (authoritative — NOT re-derived from decisions)
    extract_set = {(l["s"], l["c"], l["source"]) for l in neutral["final"]["links"]}

    # Read CSV oracle: sentence,component_id,component_name,confidence,source
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r, None)  # skip header
        csv_rows = list(r)
    csv_set = {(int(row[0]), row[1], row[4]) for row in csv_rows if len(row) >= 5}

    ok = (extract_set == csv_set)
    sym_diff = extract_set ^ csv_set

    # Secondary cross-check: ablation_*.json counts
    secondary_ok = True
    secondary_detail: dict = {}
    if ablation_path:
        try:
            with open(ablation_path) as f:
                ablation = json.load(f)
            # shape: {"<project>": {"s_linker20_union": {..., "n_links", "sources"}}}
            for _proj_key, proj_data in ablation.items():
                su = proj_data.get(ablation_key, {})
                n_links_expected = su.get("n_links")
                sources = su.get("sources", {})
                extract_n = len(neutral["final"]["links"])
                if n_links_expected is not None and n_links_expected != extract_n:
                    secondary_ok = False
                    secondary_detail["n_links_mismatch"] = {
                        "expected": n_links_expected, "got": extract_n
                    }
                entity_n = sum(1 for l in neutral["final"]["links"] if l["source"] == "entity")
                coref_n = sum(
                    1 for l in neutral["final"]["links"] if l["source"] == "coreference"
                )
                if sources.get("entity") is not None and sources["entity"] != entity_n:
                    secondary_ok = False
                    secondary_detail["entity_source_mismatch"] = {
                        "expected": sources["entity"], "got": entity_n
                    }
                if sources.get("coreference") is not None and sources["coreference"] != coref_n:
                    secondary_ok = False
                    secondary_detail["coref_source_mismatch"] = {
                        "expected": sources["coreference"], "got": coref_n
                    }
                break  # Only first project key
        except Exception as exc:
            secondary_detail["ablation_check_error"] = str(exc)

    detail = {
        "primary_ok": ok,
        "sym_diff": sorted(str(x) for x in sym_diff),
        "secondary_ok": secondary_ok,
        "secondary_detail": secondary_detail,
    }
    return ok, detail


def rederive_final(neutral: dict) -> set:
    """Re-derive the final link set from validated lists (entity-first setdefault).

    Mirrors Phase 6 dedup merge: entity validated first, coref validated second;
    first occurrence per (s,c) wins.  Asserts result equals neutral['final']['links'].

    This guard catches any serialization divergence between the stored final list
    and the validated lists (RESEARCH "Authoritative re-derivation").

    Returns:
        The derived {(s, c, source)} set.
    """
    seen: dict = {}
    for rec in neutral["entity"]["validated"]:
        seen.setdefault((rec["s"], rec["c"]), "entity")
    for rec in neutral["coref"]["validated"]:
        seen.setdefault((rec["s"], rec["c"]), "coreference")

    derived = {(s, c, src) for (s, c), src in seen.items()}
    stored = {(l["s"], l["c"], l["source"]) for l in neutral["final"]["links"]}
    assert derived == stored, (
        f"rederive_final mismatch: in_derived_not_stored={derived - stored}, "
        f"in_stored_not_derived={stored - derived}"
    )
    return derived


# ── Main driver ───────────────────────────────────────────────────────────────

def run_matrix(matrix: list, extract_root: str, csv_prefix: str,
               ablation_key: str, variant_label: str, total: int = 30) -> int:
    """Walk one matrix, extract each cell, run the faithfulness oracle.

    Shared by the Full and No-Knowledge paths so the proven
    load_cell -> to_neutral -> rederive_final -> write_json -> faithfulness flow is
    implemented ONCE.  The only differences between the two are passed in:

        csv_prefix    — *_links.csv filename prefix
                        ("s_linker20_union" | "s_linker20_union_noknow")
        ablation_key  — secondary-oracle key into ablation_*.json (same two values)
        variant_label — meta["variant"] tag (same two values)
        extract_root  — output tree (Full vs No-Knowledge)

    The cell pkl dir stays …/phase_cache/s_linker20_union/<subdir>/<project>/ for BOTH
    (Landmine 3 — _VARIANT_NAME is unchanged).

    For each cell:
      1. Assert all 5 pkls and the *_links.csv exist (missing -> failure).
      2. load_cell -> to_neutral with meta fields.
      3. rederive_final guard (internal consistency assertion).
      4. Write to extract_root/<friendly>/<run>/<project>.json (stable, non-timestamped).
      5. faithfulness() -> print PASS/FAIL with flush=True.

    Prints coverage line: "<N>/<total> cells extracted"
    Prints summary line: "<M>/<total> PASS"
    Returns 0 if all <total> cells extracted and all PASS; 1 otherwise.
    """
    n_extracted = 0
    n_pass = 0
    n_fail = 0
    any_missing = False

    for root, subdir, friendly in matrix:
        for run in RUNS:
            for project in PROJECTS:
                cell_dir = (
                    f"{root}/{run}/phase_cache/s_linker20_union/{subdir}/{project}"
                )
                csv_path = (
                    f"{root}/{run}/{project}/{csv_prefix}_{project}_links.csv"
                )
                ablation_candidates = glob.glob(
                    f"{root}/{run}/{project}/ablation_*.json"
                )
                ablation_path = ablation_candidates[0] if ablation_candidates else None

                # -- Check for missing cells ------------------------------------
                missing: list[str] = []
                for layer_name in ("layer1", "layer2", "layer3", "layer4", "final"):
                    pkl_path = os.path.join(cell_dir, f"{layer_name}.pkl")
                    if not os.path.exists(pkl_path):
                        missing.append(pkl_path)
                if not os.path.exists(csv_path):
                    missing.append(csv_path)

                if missing:
                    print(
                        f"MISSING  {friendly}/{run}/{project}  missing={missing}",
                        flush=True,
                    )
                    any_missing = True
                    continue

                # -- Load and serialize ----------------------------------------
                cell = load_cell(cell_dir)
                l1 = cell["layer1"]
                lf = cell["final"]

                meta = {
                    "backend": friendly,
                    "backend_subdir": subdir,
                    "backend_tag": lf.get("backend", subdir),
                    "run": run,
                    "project": project,
                    "variant": variant_label,
                    "n_sentences": l1["n_sentences"],
                    "n_components": l1["n_components"],
                    "elapsed_s": {
                        "final": lf.get("elapsed_s"),
                        "layer1": cell["layer1"].get("elapsed_s"),
                        "layer2": cell["layer2"].get("elapsed_s"),
                        "layer3": cell["layer3"].get("elapsed_s"),
                        "layer4": cell["layer4"].get("elapsed_s"),
                    },
                }

                neutral = to_neutral(cell, meta)

                # Internal consistency guard: rederive from validated lists
                rederive_final(neutral)

                # -- Write JSON ------------------------------------------------
                out_path = os.path.join(
                    extract_root, friendly, run, f"{project}.json"
                )
                write_json(neutral, out_path)
                n_extracted += 1

                # -- Faithfulness oracle ---------------------------------------
                ok, detail = faithfulness(
                    neutral, csv_path, ablation_path, ablation_key=ablation_key
                )
                status = "PASS" if ok else "FAIL"
                print(
                    f"{status}  {friendly}/{run}/{project}  "
                    f"links={len(neutral['final']['links'])}",
                    flush=True,
                )
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    print(f"  detail={detail}", flush=True)

    print(f"\n{n_extracted}/{total} cells extracted", flush=True)
    print(f"{n_pass}/{total} PASS", flush=True)

    if any_missing or n_fail > 0:
        return 1
    return 0


def main() -> int:
    """Full path: extract the 30 frozen Full cells (byte-identical to Phase-50)."""
    return run_matrix(
        MATRIX, EXTRACT_ROOT,
        "s_linker20_union", "s_linker20_union", "s_linker20_union", 30,
    )


def main_noknow() -> int:
    """No-Knowledge path: extract the 30 No-Knowledge cells (NOKNOW-02 / RQ4-02)."""
    return run_matrix(
        NOKNOW_MATRIX, NOKNOW_EXTRACT_ROOT,
        "s_linker20_union_noknow", "s_linker20_union_noknow",
        "s_linker20_union_noknow", 30,
    )


if __name__ == "__main__":
    if "--noknow" in sys.argv[1:]:
        raise SystemExit(main_noknow())
    raise SystemExit(main())
