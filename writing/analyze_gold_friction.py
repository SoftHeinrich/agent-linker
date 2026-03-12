#!/usr/bin/env python3
"""
Comprehensive gold standard analysis: link distribution and transitive friction.

Analyzes:
1. Single-layer distributions (SAD-SAM, SAM-CODE)
2. Transitive product: SAD-SAM × SAM-CODE → expected SAD-CODE
3. Gold friction: A→B ∧ B→C in gold but A→C NOT in gold
4. Non-transitive links: A→C in gold but ∄B s.t. A→B ∧ B→C in gold
5. Ceiling metrics for transitive approaches
6. Dark matter: sentences with component mentions but no gold link
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

BENCHMARK_DIR = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")

PROJECTS = {
    "mediastore": {
        "sad_sam": "goldstandard_sad_2016-sam_2016.csv",
        "sam_code": "goldstandard_sam_2016-code_2016.csv",
        "sad_code": "goldstandard_sad_2016-code_2016.csv",
        "acm": "model_2016/code/codeModel.acm",
    },
    "teastore": {
        "sad_sam": "goldstandard_sad_2020-sam_2020.csv",
        "sam_code": "goldstandard_sam_2020-code_2022.csv",
        "sad_code": "goldstandard_sad_2020-code_2022.csv",
        "acm": "model_2022/code/codeModel.acm",
    },
    "teammates": {
        "sad_sam": "goldstandard_sad_2021-sam_2021.csv",
        "sam_code": "goldstandard_sam_2021-code_2023.csv",
        "sad_code": "goldstandard_sad_2021-code_2023.csv",
        "acm": "model_2023/code/codeModel.acm",
    },
    "bigbluebutton": {
        "sad_sam": "goldstandard_sad_2021-sam_2021.csv",
        "sam_code": "goldstandard_sam_2021-code_2023.csv",
        "sad_code": "goldstandard_sad_2021-code_2023.csv",
        "acm": "model_2023/code/codeModel.acm",
    },
    "jabref": {
        "sad_sam": "goldstandard_sad_2021-sam_2021.csv",
        "sam_code": "goldstandard_sam_2021-code_2023.csv",
        "sad_code": "goldstandard_sad_2021-code_2023.csv",
        "acm": "model_2023/code/codeModel.acm",
    },
}


def norm_path(p):
    """Normalize a code path: strip Implementation/ prefix, ensure consistent trailing."""
    p = p.strip()
    if p.startswith("Implementation/"):
        p = p[len("Implementation/"):]
    return p


def load_sad_sam(project):
    """Load SAD-SAM gold: set of (sentence, modelElementID)."""
    path = BENCHMARK_DIR / project / "goldstandards" / PROJECTS[project]["sad_sam"]
    links = set()
    names = {}  # modelElementID -> name from SAM-CODE
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["sentence"])
            mid = row["modelElementID"]
            links.add((sid, mid))
    return links


def load_sam_code(project):
    """Load SAM-CODE gold: dict of modelElementID -> set of code paths."""
    path = BENCHMARK_DIR / project / "goldstandards" / PROJECTS[project]["sam_code"]
    mapping = defaultdict(set)
    names = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["ae_id"]
            name = row["ae_name"]
            # Handle column name variation
            code = row.get("ce_ids") or row.get("ce_id") or row.get("codeID", "")
            code = norm_path(code)
            mapping[mid].add(code)
            names[mid] = name
    return dict(mapping), names


def load_sad_code(project):
    """Load SAD-CODE gold: set of (sentence, code_path)."""
    path = BENCHMARK_DIR / project / "goldstandards" / PROJECTS[project]["sad_code"]
    links = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["sentenceID"])
            code = norm_path(row["codeID"])
            links.add((sid, code))
    return links


def is_subpath(file_path, dir_path):
    """Check if file_path is under dir_path (directory entry ending with /)."""
    if not dir_path.endswith("/"):
        return False
    return file_path.startswith(dir_path)


def load_acm_files(project):
    """Load all file paths from the .acm code model (JSON format)."""
    acm_path = BENCHMARK_DIR / project / PROJECTS[project]["acm"]
    if not acm_path.exists():
        return set()
    files = set()
    with open(acm_path) as f:
        data = json.load(f)
    repo = data.get("codeItemRepository", {}).get("repository", {})
    for item_id, item in repo.items():
        if item.get("type") == "CodeCompilationUnit":
            path_elements = item.get("pathElements", [])
            name = item.get("name", "")
            ext = item.get("extension", "")
            if path_elements and name:
                full_path = "/".join(path_elements) + "/" + name
                if ext:
                    full_path += "." + ext
                files.add(norm_path(full_path))
    return files


def enroll_directory(dir_path, acm_files):
    """Expand a directory path to individual files using ACM model."""
    if not dir_path.endswith("/"):
        return {dir_path}
    return {f for f in acm_files if f.startswith(dir_path)}


def enroll_set(link_set, acm_files):
    """Enroll all directory entries in a set of (sentence, code_path) pairs."""
    enrolled = set()
    for sid, code in link_set:
        if code.endswith("/"):
            for f in enroll_directory(code, acm_files):
                enrolled.add((sid, f))
        else:
            enrolled.add((sid, code))
    return enrolled


def compute_transitive_product(sad_sam, sam_code):
    """Compute SAD-SAM × SAM-CODE = {(sentence, code_path) : ∃ model s.t. (s,m) ∈ SAD-SAM and (m,f) ∈ SAM-CODE}."""
    product = set()
    # Also track which model element generated each link
    product_via = defaultdict(set)  # (sid, code) -> set of model elements
    for sid, mid in sad_sam:
        if mid in sam_code:
            for code in sam_code[mid]:
                product.add((sid, code))
                product_via[(sid, code)].add(mid)
    return product, product_via


def analyze_project(project):
    """Full analysis for one project."""
    print(f"\n{'='*80}")
    print(f"  PROJECT: {project.upper()}")
    print(f"{'='*80}")

    # Load data
    sad_sam = load_sad_sam(project)
    sam_code, elem_names = load_sam_code(project)
    sad_code_gold = load_sad_code(project)
    acm_files = load_acm_files(project)

    # ── LAYER 1: SAD-SAM Distribution ──
    print(f"\n── SAD-SAM Distribution ──")
    ss_sentences = {s for s, m in sad_sam}
    ss_elements = {m for s, m in sad_sam}
    ss_by_element = defaultdict(set)
    ss_by_sentence = defaultdict(set)
    for s, m in sad_sam:
        ss_by_element[m].add(s)
        ss_by_sentence[s].add(m)

    print(f"  Total links: {len(sad_sam)}")
    print(f"  Unique sentences: {len(ss_sentences)}")
    print(f"  Unique model elements: {len(ss_elements)}")
    print(f"  Avg links/element: {len(sad_sam)/len(ss_elements):.1f}")
    print(f"  Avg links/sentence: {len(sad_sam)/len(ss_sentences):.1f}")

    print(f"\n  Per-element breakdown:")
    for mid in sorted(ss_by_element, key=lambda m: len(ss_by_element[m]), reverse=True):
        name = elem_names.get(mid, mid[:20])
        sents = sorted(ss_by_element[mid])
        print(f"    {name:40s}  {len(sents):3d} sentences  [{', '.join(map(str, sents[:10]))}{'...' if len(sents) > 10 else ''}]")

    # Multi-component sentences
    multi = {s: ms for s, ms in ss_by_sentence.items() if len(ms) > 1}
    print(f"\n  Multi-component sentences: {len(multi)} / {len(ss_sentences)} ({100*len(multi)/len(ss_sentences):.0f}%)")
    for s in sorted(multi):
        names_list = [elem_names.get(m, m[:20]) for m in multi[s]]
        print(f"    S{s}: {', '.join(names_list)}")

    # ── LAYER 2: SAM-CODE Distribution ──
    print(f"\n── SAM-CODE Distribution ──")
    sc_total = sum(len(files) for files in sam_code.values())
    sc_elements_with_code = {m for m in sam_code if sam_code[m]}
    sc_dir_entries = sum(1 for m in sam_code for f in sam_code[m] if f.endswith("/"))
    sc_file_entries = sc_total - sc_dir_entries

    print(f"  Total entries: {sc_total}")
    print(f"  Directory entries: {sc_dir_entries} ({100*sc_dir_entries/sc_total:.0f}%)")
    print(f"  File entries: {sc_file_entries} ({100*sc_file_entries/sc_total:.0f}%)")
    print(f"  Model elements with code: {len(sc_elements_with_code)}")

    # Enrollment expansion
    if acm_files:
        enrolled_total = 0
        print(f"\n  Per-element enrollment:")
        for mid in sorted(sam_code, key=lambda m: elem_names.get(m, m)):
            name = elem_names.get(mid, mid[:20])
            raw_entries = sam_code[mid]
            enrolled = set()
            for entry in raw_entries:
                enrolled.update(enroll_directory(entry, acm_files))
            enrolled_total += len(enrolled)
            ratio = len(enrolled) / len(raw_entries) if raw_entries else 0
            print(f"    {name:40s}  raw={len(raw_entries):3d}  enrolled={len(enrolled):4d}  ratio={ratio:.1f}x")
        print(f"  Total enrolled files: {enrolled_total} (from {sc_total} raw entries, {enrolled_total/sc_total:.1f}x)")

    # ── SAD-SAM elements NOT in SAM-CODE (ghost elements) ──
    ghost_elements = ss_elements - set(sam_code.keys())
    if ghost_elements:
        print(f"\n── Ghost Elements (in SAD-SAM but NOT in SAM-CODE) ──")
        for mid in ghost_elements:
            name = elem_names.get(mid, mid)
            sents = sorted(ss_by_element[mid])
            print(f"    {name:40s}  {len(sents)} SAD-SAM links (sentences: {sents})")
    else:
        print(f"\n  No ghost elements (all SAD-SAM elements appear in SAM-CODE)")

    # SAM-CODE elements NOT in SAD-SAM (orphan code)
    orphan_elements = set(sam_code.keys()) - ss_elements
    if orphan_elements:
        print(f"\n── Orphan Code Elements (in SAM-CODE but NOT in SAD-SAM) ──")
        for mid in orphan_elements:
            name = elem_names.get(mid, mid)
            files = sam_code[mid]
            print(f"    {name:40s}  {len(files)} code entries")

    # ── TRANSITIVE ANALYSIS ──
    print(f"\n── Transitive Product Analysis ──")
    transitive_raw, transitive_via = compute_transitive_product(sad_sam, sam_code)

    # The transitive product at raw level (directories not expanded)
    print(f"  Transitive product (raw): {len(transitive_raw)} (sentence, code) pairs")
    print(f"  SAD-CODE gold (raw):      {len(sad_code_gold)} (sentence, code) pairs")

    # Raw-level comparison
    tp_match_raw = transitive_raw & sad_code_gold
    friction_raw = transitive_raw - sad_code_gold  # A→B ∧ B→C but NOT A→C
    non_transitive_raw = sad_code_gold - transitive_raw  # A→C but NOT via B

    print(f"\n  Raw-level set comparison:")
    print(f"    Match (in both):                {len(tp_match_raw)}")
    print(f"    FRICTION (transitive but ¬gold): {len(friction_raw)}")
    print(f"    NON-TRANSITIVE (gold but ¬trans): {len(non_transitive_raw)}")

    # ── DIRECTORY-AWARE COMPARISON ──
    # The key insight: SAD-CODE gold may have FINER-GRAINED entries than SAM-CODE
    # e.g., SAD-CODE has sentence→specific_file but SAM-CODE only has component→directory
    # We need to check containment, not just equality

    # For each non-transitive gold entry, check if it's CONTAINED in a transitive directory
    non_trans_explained = set()  # actually reachable via directory containment
    non_trans_unreachable = set()  # truly non-transitive
    for sid, code in non_transitive_raw:
        # Check if any transitive entry is a parent directory of this code path
        found = False
        for tid, tcode in transitive_raw:
            if tid == sid and tcode.endswith("/") and code.startswith(tcode):
                found = True
                break
            if tid == sid and tcode == code:
                found = True
                break
        if found:
            non_trans_explained.add((sid, code))
        else:
            non_trans_unreachable.add((sid, code))

    print(f"\n  Directory-aware non-transitive analysis:")
    print(f"    Explained by directory containment: {len(non_trans_explained)}")
    print(f"    TRULY non-transitive:               {len(non_trans_unreachable)}")

    # For friction: check if a gold entry covers the transitive entry
    friction_explained = set()
    friction_real = set()
    for sid, code in friction_raw:
        found = False
        for gid, gcode in sad_code_gold:
            if gid == sid and gcode.endswith("/") and code.startswith(gcode):
                found = True
                break
            if gid == sid and code.endswith("/") and gcode.startswith(code):
                found = True
                break
        if found:
            friction_explained.add((sid, code))
        else:
            friction_real.add((sid, code))

    print(f"\n  Directory-aware friction analysis:")
    print(f"    Explained by directory containment: {len(friction_explained)}")
    print(f"    REAL friction (trans ∧ ¬gold):      {len(friction_real)}")

    # ── ENROLLED COMPARISON ──
    if acm_files:
        print(f"\n── Enrolled (File-Level) Comparison ──")
        # Enroll both transitive product and gold standard
        trans_enrolled = enroll_set(transitive_raw, acm_files)
        gold_enrolled = enroll_set(sad_code_gold, acm_files)

        match_enrolled = trans_enrolled & gold_enrolled
        friction_enrolled = trans_enrolled - gold_enrolled
        non_trans_enrolled = gold_enrolled - trans_enrolled

        print(f"  Transitive product (enrolled): {len(trans_enrolled)} file-level pairs")
        print(f"  SAD-CODE gold (enrolled):      {len(gold_enrolled)} file-level pairs")
        print(f"  Match:                         {len(match_enrolled)}")
        print(f"  FRICTION (trans ∧ ¬gold):      {len(friction_enrolled)}")
        print(f"  NON-TRANSITIVE (gold ∧ ¬trans): {len(non_trans_enrolled)}")

        ceiling_r = len(match_enrolled) / len(gold_enrolled) if gold_enrolled else 1.0
        print(f"\n  Ceiling_R (enrolled): {ceiling_r:.4f} ({100*ceiling_r:.1f}%)")
        if gold_enrolled:
            print(f"  Structurally impossible FNs: {len(non_trans_enrolled)} / {len(gold_enrolled)} ({100*len(non_trans_enrolled)/len(gold_enrolled):.1f}%)")
        else:
            print(f"  Structurally impossible FNs: 0 / 0 (no enrolled files — ACM may not cover gold paths)")

    # ── DETAIL: Non-transitive gold entries ──
    if non_trans_unreachable:
        print(f"\n── TRULY Non-Transitive Gold Entries (detail) ──")
        # Group by sentence
        by_sentence = defaultdict(list)
        for sid, code in sorted(non_trans_unreachable):
            by_sentence[sid].append(code)

        for sid in sorted(by_sentence):
            # What components does this sentence link to in SAD-SAM?
            linked_components = ss_by_sentence.get(sid, set())
            comp_names = [elem_names.get(m, m[:20]) for m in linked_components]
            print(f"\n  S{sid} (SAD-SAM components: {', '.join(comp_names) if comp_names else 'NONE'})")
            for code in by_sentence[sid]:
                # What component does this code belong to in SAM-CODE?
                code_component = None
                for mid, files in sam_code.items():
                    for f in files:
                        if f == code or (f.endswith("/") and code.startswith(f)):
                            code_component = elem_names.get(mid, mid[:20])
                            break
                    if code_component:
                        break
                tag = f" [via {code_component}]" if code_component else " [NO SAM-CODE match]"
                print(f"    → {code}{tag}")
                # Why is this non-transitive?
                if sid not in ss_sentences:
                    print(f"      REASON: sentence {sid} has NO SAD-SAM link")
                elif code_component:
                    # Find the model element
                    for mid, name in elem_names.items():
                        if name == code_component:
                            if (sid, mid) not in sad_sam:
                                print(f"      REASON: S{sid} not linked to {code_component} in SAD-SAM (linked to: {', '.join(elem_names.get(m, m[:20]) for m in linked_components)})")
                            break

    # ── DETAIL: Friction entries ──
    if friction_real:
        print(f"\n── REAL Friction Entries (transitive product says YES, gold says NO) ──")
        by_sentence_f = defaultdict(list)
        for sid, code in sorted(friction_real):
            by_sentence_f[sid].append(code)

        for sid in sorted(by_sentence_f):
            linked_components = ss_by_sentence.get(sid, set())
            comp_names = [elem_names.get(m, m[:20]) for m in linked_components]
            print(f"\n  S{sid} (SAD-SAM: {', '.join(comp_names)})")
            for code in by_sentence_f[sid]:
                via_elements = transitive_via.get((sid, code), set())
                via_names = [elem_names.get(m, m[:20]) for m in via_elements]
                print(f"    → {code}")
                print(f"      VIA: {', '.join(via_names)}")

    # ── CEILING METRICS ──
    print(f"\n── Ceiling Metrics Summary ──")
    if acm_files:
        print(f"  Gold (enrolled):           {len(gold_enrolled)} file-level links")
        print(f"  Transitive reach:          {len(trans_enrolled)} file-level links")
        print(f"  Gold ∩ Trans (reachable):   {len(match_enrolled)}")
        print(f"  Gold \\ Trans (impossible):  {len(non_trans_enrolled)}")
        print(f"  Trans \\ Gold (friction):    {len(friction_enrolled)}")
        print(f"  Ceiling_R: {ceiling_r:.4f}")
    else:
        ceiling_r_raw = len(tp_match_raw) / len(sad_code_gold) if sad_code_gold else 1.0
        print(f"  (No ACM file — raw level only)")
        print(f"  Ceiling_R (raw): {ceiling_r_raw:.4f}")

    return {
        "project": project,
        "sad_sam_links": len(sad_sam),
        "sad_sam_sentences": len(ss_sentences),
        "sad_sam_elements": len(ss_elements),
        "sam_code_elements": len(sc_elements_with_code),
        "sam_code_entries": sc_total,
        "sam_code_dir_entries": sc_dir_entries,
        "sad_code_gold_raw": len(sad_code_gold),
        "transitive_raw": len(transitive_raw),
        "friction_raw": len(friction_raw),
        "friction_real": len(friction_real),
        "non_transitive_raw": len(non_transitive_raw),
        "non_transitive_unreachable": len(non_trans_unreachable),
        "ghost_elements": len(ghost_elements),
        "orphan_elements": len(orphan_elements),
        "multi_component_sentences": len(multi),
        "sad_code_enrolled": len(gold_enrolled) if acm_files else None,
        "trans_enrolled": len(trans_enrolled) if acm_files else None,
        "ceiling_r": ceiling_r if acm_files else None,
        "non_trans_enrolled": len(non_trans_enrolled) if acm_files else None,
        "friction_enrolled": len(friction_enrolled) if acm_files else None,
    }


def dark_matter_analysis(project):
    """Analyze dark matter: sentences with potential component references but no gold link."""
    print(f"\n── Dark Matter Analysis: {project.upper()} ──")

    sad_sam = load_sad_sam(project)
    sad_code_gold = load_sad_code(project)
    sam_code, elem_names = load_sam_code(project)

    # Load documentation text
    text_dir = BENCHMARK_DIR / project
    text_files = list(text_dir.glob("text_*/*/sentences.txt")) + list(text_dir.glob("text_*/*.txt"))
    if not text_files:
        text_files = list(text_dir.glob("text_*/*"))

    sad_sam_sentences = {s for s, m in sad_sam}
    sad_code_sentences = {s for s, f in sad_code_gold}

    print(f"  Sentences in SAD-SAM gold: {len(sad_sam_sentences)}")
    print(f"  Sentences in SAD-CODE gold: {len(sad_code_sentences)}")

    # Sentences in SAD-CODE but NOT in SAD-SAM
    code_only = sad_code_sentences - sad_sam_sentences
    if code_only:
        print(f"\n  Sentences in SAD-CODE but NOT SAD-SAM: {len(code_only)}")
        print(f"    Sentence IDs: {sorted(code_only)}")

    # Sentences in SAD-SAM but NOT in SAD-CODE
    sam_only = sad_sam_sentences - sad_code_sentences
    if sam_only:
        print(f"\n  Sentences in SAD-SAM but NOT SAD-CODE: {len(sam_only)}")
        print(f"    Sentence IDs: {sorted(sam_only)}")
        # For each, why no SAD-CODE link?
        for s in sorted(sam_only):
            components = {m for sid, m in sad_sam if sid == s}
            for m in components:
                name = elem_names.get(m, m[:20])
                has_code = m in sam_code and len(sam_code[m]) > 0
                if not has_code:
                    print(f"    S{s} → {name}: NO SAM-CODE mapping (element has no code)")
                else:
                    print(f"    S{s} → {name}: has SAM-CODE but SAD-CODE missing this sentence")


def summary_table(results):
    """Print aggregate summary."""
    print(f"\n{'='*100}")
    print(f"  AGGREGATE SUMMARY")
    print(f"{'='*100}")

    # Header
    fmt = "{:<15s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s} {:>10s} {:>8s}"
    print(fmt.format("Project", "SAD-SAM", "SAM-CODE", "SAD-CODE", "Trans", "Friction", "NonTrans", "CeilR", "Ghosts"))
    print("-" * 100)

    for r in results:
        print(fmt.format(
            r["project"],
            str(r["sad_sam_links"]),
            str(r["sam_code_entries"]),
            str(r["sad_code_gold_raw"]),
            str(r["transitive_raw"]),
            f"{r['friction_real']}",
            f"{r['non_transitive_unreachable']}",
            f"{r['ceiling_r']:.3f}" if r['ceiling_r'] is not None else "N/A",
            str(r["ghost_elements"]),
        ))

    print("-" * 100)
    # Totals
    tot_ss = sum(r["sad_sam_links"] for r in results)
    tot_sc = sum(r["sam_code_entries"] for r in results)
    tot_sdc = sum(r["sad_code_gold_raw"] for r in results)
    tot_trans = sum(r["transitive_raw"] for r in results)
    tot_fric = sum(r["friction_real"] for r in results)
    tot_nt = sum(r["non_transitive_unreachable"] for r in results)
    tot_ghost = sum(r["ghost_elements"] for r in results)
    enrolled = [r for r in results if r["ceiling_r"] is not None]
    avg_ceil = sum(r["ceiling_r"] for r in enrolled) / len(enrolled) if enrolled else 0

    print(fmt.format(
        "TOTAL/AVG", str(tot_ss), str(tot_sc), str(tot_sdc), str(tot_trans),
        str(tot_fric), str(tot_nt), f"{avg_ceil:.3f}", str(tot_ghost)
    ))

    # Enrolled summary
    print(f"\n── Enrolled (File-Level) Summary ──")
    fmt2 = "{:<15s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>8s}"
    print(fmt2.format("Project", "Gold(enr)", "Trans(enr)", "Match", "Friction", "NonTrans", "CeilR"))
    print("-" * 80)
    for r in results:
        if r["sad_code_enrolled"] is not None:
            print(fmt2.format(
                r["project"],
                str(r["sad_code_enrolled"]),
                str(r["trans_enrolled"]),
                str(r["sad_code_enrolled"] - r["non_trans_enrolled"]),
                str(r["friction_enrolled"]),
                str(r["non_trans_enrolled"]),
                f"{r['ceiling_r']:.3f}",
            ))

    # Multi-component sentence summary
    print(f"\n── Multi-Component Sentence Summary ──")
    for r in results:
        pct = 100 * r["multi_component_sentences"] / r["sad_sam_sentences"] if r["sad_sam_sentences"] else 0
        print(f"  {r['project']:15s}: {r['multi_component_sentences']:3d} / {r['sad_sam_sentences']:3d} sentences ({pct:.0f}%)")

    # Ghost + orphan summary
    print(f"\n── Element Coverage ──")
    for r in results:
        print(f"  {r['project']:15s}: SAD-SAM elements={r['sad_sam_elements']}, SAM-CODE elements={r['sam_code_elements']}, ghosts={r['ghost_elements']}, orphans={r['orphan_elements']}")


if __name__ == "__main__":
    results = []
    for project in PROJECTS:
        r = analyze_project(project)
        results.append(r)

    for project in PROJECTS:
        dark_matter_analysis(project)

    summary_table(results)
