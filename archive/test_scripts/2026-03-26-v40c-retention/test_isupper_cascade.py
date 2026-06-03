"""Test script: Investigate _is_generic_mention() isupper early-return.

Traces DB links through the V39a mediastore pipeline using saved checkpoints.
Tests what _is_generic_mention returns with and without the isupper check.
Reports which links would change behavior.

No LLM calls needed -- purely offline checkpoint analysis.
"""

import csv
import os
import pickle
import re
import sys

sys.path.insert(0, "src")

from llm_sad_sam.core.data_types import SadSamLink

# ── Paths ──────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "results/phase_cache/v39a/mediastore"
GS_FILE = (
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/"
    "benchmark/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"
)
TEXT_FILE = (
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/"
    "benchmark/mediastore/text_2016/mediastore.txt"
)

# ── Reimplementation of the relevant functions ─────────────────────────


def _has_standalone_mention(comp_name: str, text: str) -> bool:
    """Exact copy from ilinker2_v39a.py."""
    is_single = " " not in comp_name
    if is_single:
        cap_name = comp_name[0].upper() + comp_name[1:]
        pattern = rf"\b{re.escape(cap_name)}\b"
        flags = 0
    else:
        pattern = rf"\b{re.escape(comp_name)}\b"
        flags = re.IGNORECASE

    for m in re.finditer(pattern, text, flags):
        s, e = m.start(), m.end()
        if s > 0 and text[s - 1] == ".":
            continue
        if e < len(text) and text[e] == "." and e + 1 < len(text) and text[e + 1].isalpha():
            continue
        if s > 0 and text[s - 1] == "-":
            continue
        if e < len(text) and text[e] == "-" and "-" not in comp_name:
            continue
        return True
    return False


def _is_generic_mention_original(comp_name: str, sentence_text: str) -> bool:
    """Exact copy from ilinker2_v39a.py line 210-224."""
    if " " in comp_name or "-" in comp_name:
        return False
    if re.search(r"[a-z][A-Z]", comp_name):
        return False
    if comp_name.isupper():  # <-- THE CHECK UNDER INVESTIGATION
        return False
    if comp_name[0].islower():
        return False
    if _has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if re.search(rf"\b{re.escape(word_lower)}\b", sentence_text):
        return True
    return False


def _is_generic_mention_no_isupper(comp_name: str, sentence_text: str) -> bool:
    """Modified: isupper early-return REMOVED."""
    if " " in comp_name or "-" in comp_name:
        return False
    if re.search(r"[a-z][A-Z]", comp_name):
        return False
    # REMOVED: if comp_name.isupper(): return False
    if comp_name[0].islower():
        return False
    if _has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if re.search(rf"\b{re.escape(word_lower)}\b", sentence_text):
        return True
    return False


# ── Load data ──────────────────────────────────────────────────────────


def load_checkpoint(name: str):
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gold_standard():
    gs = set()
    with open(GS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gs.add((row["modelElementID"], int(row["sentence"])))
    return gs


def load_sentences():
    with open(TEXT_FILE) as f:
        lines = f.readlines()
    return {i + 1: line.strip() for i, line in enumerate(lines)}


# ── Main analysis ─────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("INVESTIGATION: _is_generic_mention() isupper early-return")
    print("=" * 78)

    # Load all data
    p1 = load_checkpoint("phase1")
    p3 = load_checkpoint("phase3")
    p4 = load_checkpoint("phase4")
    p5 = load_checkpoint("phase5")
    p6 = load_checkpoint("phase6")
    pj = load_checkpoint("pre_judge")
    final = load_checkpoint("final")
    gs = load_gold_standard()
    sentences = load_sentences()

    # ── Section 1: Understanding _is_generic_mention's logic ──────────

    print("\n" + "─" * 78)
    print("SECTION 1: Function Logic Analysis")
    print("─" * 78)
    print("""
_is_generic_mention(comp_name, sentence_text) checks if a component name
is used generically (as an ordinary English word) rather than as a component
reference. It returns True if the mention is generic (should be rejected).

Early-return conditions (return False = NOT generic):
  1. Multi-word or hyphenated name -> never generic
  2. CamelCase name -> never generic
  3. ALL-UPPERCASE name (isupper) -> never generic   <-- UNDER INVESTIGATION
  4. Lowercase first char -> never generic
  5. Standalone capitalized mention exists -> not generic here

If none of the above, checks if comp_name.lower() appears as a word in the text.
If yes -> generic mention (True). If no -> not generic (False).

Key insight: the function searches for \\b{comp_name.lower()}\\b in text.
For "DB", this is \\bdb\\b. The word "database" does NOT match \\bdb\\b.
""")

    # ── Section 2: All isupper components in the dataset ──────────────

    print("─" * 78)
    print("SECTION 2: ALL-UPPERCASE Components Across Datasets")
    print("─" * 78)
    print("""
  mediastore:     DB
  teammates:      UI, E2E
  bigbluebutton:  FSESL
  teastore:       (none)
  jabref:         (none)
""")

    # ── Section 3: DB FP cascade tracing ──────────────────────────────

    print("─" * 78)
    print("SECTION 3: Tracing DB FPs Through the Pipeline")
    print("─" * 78)

    db_id = "_5LN7MLg2EeSNPorBlo7x9g"
    db_gs_sents = sorted([s for mid, s in gs if mid == db_id])
    print(f"\nDB component (id={db_id})")
    print(f"Gold standard sentences: {db_gs_sents}")

    # Phase 3 partial references
    doc_knowledge = p3["doc_knowledge"]
    print(f"\nPhase 3 partial references to DB:")
    for partial, target in doc_knowledge.partial_references.items():
        if target == "DB":
            print(f'  "{partial}" -> DB')
    print(f"Phase 3 synonyms to DB:")
    for syn, target in doc_knowledge.synonyms.items():
        if target == "DB":
            print(f'  "{syn}" -> DB')

    # Phase 4 (ILinker2 seed)
    p4_db = [l for l in p4["transarc_links"] if l.component_name == "DB"]
    print(f"\nPhase 4 (ILinker2 seed) DB links: {sorted([l.sentence_number for l in p4_db])}")
    print(f"  Source: all 'transarc'")

    # Phase 5 (entity extraction)
    p5_db = [c for c in p5["candidates"] if c.component_name == "DB"]
    print(f"\nPhase 5 (entity extraction) DB candidates:")
    for c in p5_db:
        is_gs = (db_id, c.sentence_number) in gs
        label = "TP" if is_gs else "FP"
        print(f"  S{c.sentence_number} -> DB (matched='{c.matched_text}', "
              f"needs_val={c.needs_validation}) [{label}]")

    # Phase 6 (validation)
    p6_db = [l for l in p6["validated"] if l.component_name == "DB"]
    print(f"\nPhase 6 (validated) DB links:")
    for l in p6_db:
        is_gs = (db_id, l.sentence_number) in gs
        label = "TP" if is_gs else "FP"
        print(f"  S{l.sentence_number} -> DB [source={l.source}] [{label}]")

    # Pre-judge
    pj_db = [l for l in pj["preliminary"] if l.component_name == "DB"]
    print(f"\nPre-judge DB links:")
    for l in pj_db:
        is_gs = (db_id, l.sentence_number) in gs
        label = "TP" if is_gs else "FP"
        print(f"  S{l.sentence_number} -> DB [source={l.source}] [{label}]")

    # Final
    final_links = final["final"] if isinstance(final, dict) else final
    final_db = [l for l in final_links if l.component_name == "DB"]
    print(f"\nFinal DB links:")
    final_db_sents = set()
    for l in final_db:
        is_gs = (db_id, l.sentence_number) in gs
        label = "TP" if is_gs else "FP"
        final_db_sents.add(l.sentence_number)
        print(f"  S{l.sentence_number} -> DB [source={l.source}] [{label}]")

    fp_sents = sorted([l.sentence_number for l in final_db if (db_id, l.sentence_number) not in gs])
    tp_sents = sorted([l.sentence_number for l in final_db if (db_id, l.sentence_number) in gs])
    fn_sents = sorted([s for mid, s in gs if mid == db_id and s not in final_db_sents])
    print(f"\nDB Summary: {len(tp_sents)} TPs, {len(fp_sents)} FPs, {len(fn_sents)} FNs")
    print(f"  TPs: {tp_sents}")
    print(f"  FPs: {fp_sents}")
    print(f"  FNs: {fn_sents}")

    # ── Section 4: _is_generic_mention behavior ──────────────────────

    print("\n" + "─" * 78)
    print("SECTION 4: _is_generic_mention Behavior for DB Links")
    print("─" * 78)

    print("\nTesting _is_generic_mention('DB', sentence) for each FP and TP:")
    print()

    all_db_sents = sorted(set(
        [l.sentence_number for l in final_db] +
        [c.sentence_number for c in p5_db]
    ))

    for snum in all_db_sents:
        sent_text = sentences.get(snum, "")
        orig = _is_generic_mention_original("DB", sent_text)
        nois = _is_generic_mention_no_isupper("DB", sent_text)
        standalone = _has_standalone_mention("DB", sent_text)
        has_db_lower = bool(re.search(r"\bdb\b", sent_text))
        is_gs = (db_id, snum) in gs

        changed = "CHANGED" if orig != nois else "same"
        label = "TP" if is_gs else "FP"

        print(f"  S{snum} [{label}]: orig={orig}, no_isupper={nois} [{changed}]")
        print(f"    standalone('DB') = {standalone}")
        print(f"    \\bdb\\b in text = {has_db_lower}")
        print(f"    Text: {sent_text[:100]}")
        print()

    # ── Section 5: Root cause analysis ───────────────────────────────

    print("─" * 78)
    print("SECTION 5: Root Cause Analysis")
    print("─" * 78)
    print("""
FINDING: The isupper check is IRRELEVANT to the DB FP problem.

Even without the isupper check, _is_generic_mention("DB", text) returns False
for ALL sentences because:
  - \\bdb\\b does NOT match "database" (word boundary prevents it)
  - "db" never appears as a standalone word in the mediastore text

The DB FPs (S12, S27, S29) originate from Phase 5 entity extraction:
  - The LLM extracts "database" (lowercase) as matched_text for component "DB"
  - Phase 3 discovered "Database" -> DB as a partial reference
  - Phase 5 uses this alias to match sentences mentioning "database"
  - Phase 6 validates these matches (LLM confirms "database" refers to DB)
  - The links reach the final output

The real problem is the SEMANTIC gap: "database" (lowercase, generic concept)
vs "Database" (capitalized, component reference). The _is_generic_mention
function CAN detect this distinction when called with "Database" as comp_name
(see Section 6), but it is called with "DB" (the actual component name),
where the check is structurally impossible.
""")

    # ── Section 6: What WOULD work ───────────────────────────────────

    print("─" * 78)
    print("SECTION 6: _is_generic_mention with 'Database' (the partial)")
    print("─" * 78)
    print()
    print("If the function were called with the PARTIAL REFERENCE 'Database'")
    print("instead of the component name 'DB', it would correctly classify:")
    print()

    for snum in sorted(set(fp_sents + tp_sents + [s for _, s in gs if _ == db_id])):
        sent_text = sentences.get(snum, "")
        result = _is_generic_mention_original("Database", sent_text)
        is_gs = (db_id, snum) in gs
        label = "TP" if is_gs else "FP"
        correct = (result and not is_gs) or (not result and is_gs)
        verdict = "CORRECT" if correct else "WRONG"

        print(f"  S{snum} [{label}]: generic={result} [{verdict}]")
        print(f"    Text: {sent_text[:100]}")
        print()

    # ── Section 7: isupper removal impact ────────────────────────────

    print("─" * 78)
    print("SECTION 7: Impact of Removing isupper Check")
    print("─" * 78)
    print()

    # Test ALL links in the final output, not just DB
    final_links = final["final"] if isinstance(final, dict) else final

    changed_links = []
    for l in final_links:
        sent_text = sentences.get(l.sentence_number, "")
        if not sent_text:
            continue
        orig = _is_generic_mention_original(l.component_name, sent_text)
        nois = _is_generic_mention_no_isupper(l.component_name, sent_text)
        if orig != nois:
            is_gs = (l.component_id, l.sentence_number) in gs
            changed_links.append((l, orig, nois, is_gs))

    if changed_links:
        print(f"Links that would change behavior: {len(changed_links)}")
        for l, orig, nois, is_gs in changed_links:
            label = "TP" if is_gs else "FP"
            sent_text = sentences.get(l.sentence_number, "")
            print(f"  S{l.sentence_number} -> {l.component_name} [{label}]")
            print(f"    orig={orig} -> no_isupper={nois}")
            print(f"    Text: {sent_text[:100]}")
    else:
        print("NO links would change behavior in the mediastore dataset.")
        print()
        print("Reason: the only isupper component is 'DB'.")
        print("For 'DB', comp_name.lower() = 'db', and \\bdb\\b never matches")
        print("'database' in any sentence. So even without isupper, the function")
        print("returns the same result.")

    # ── Section 8: Teammates hypothetical analysis ───────────────────

    print()
    print("─" * 78)
    print("SECTION 8: Hypothetical Impact on Teammates (UI/E2E)")
    print("─" * 78)
    print()
    print("If isupper were removed, these teammates sentences WOULD be flagged:")
    print("(All are sub-package descriptions NOT in the gold standard)")
    print()

    tm_sents = {
        22: "logic, ui.website, ui.controller represent an application of Model-View-Controller pattern.",
        23: "ui.website is not a real package.",
        26: "ui.website is not a Java package.",
        187: "Package overview contains e2e.util, e2e.pageobjects, e2e.cases, x.util, x.e2e, x.lnp.",
        188: "e2e.util contains helpers needed for running E2E tests.",
        189: "e2e.pageobjects contains abstractions of the pages as they appear on a Browser (i.e. SUTs).",
        190: "e2e.cases contains test cases.",
        192: "x.e2e contains system test cases for testing the application as a whole.",
    }

    for snum, text in sorted(tm_sents.items()):
        comp = "UI" if "ui" in text.lower()[:10] or "ui." in text else "E2E"
        orig = _is_generic_mention_original(comp, text)
        nois = _is_generic_mention_no_isupper(comp, text)
        changed = "NEWLY FLAGGED" if not orig and nois else "same"
        print(f"  S{snum} comp={comp}: orig={orig}, no_isupper={nois} [{changed}]")
        print(f"    Text: {text[:100]}")
        print()

    print("These would be correctly flagged as generic (sub-package descriptions).")
    print("However, these sentences rarely survive to Phase 6 because the Phase 5")
    print("extraction prompt already excludes dotted-path references.")
    print("So the practical impact of removing isupper is NEAR ZERO.")

    # ── Section 9: Verdict ───────────────────────────────────────────

    print()
    print("=" * 78)
    print("VERDICT: Is the isupper early-return a BUG or INTENTIONAL DESIGN?")
    print("=" * 78)
    print("""
ANSWER: INTENTIONAL DESIGN, but with a documentation gap.

The isupper check is CORRECT defensive coding for these reasons:

1. ALL-UPPERCASE names (DB, UI, E2E, FSESL) are abbreviations/acronyms that
   should ALWAYS be treated as component references when they appear in text.
   They are never "generic English words" the way "Cache" or "Facade" might be.

2. The check is consistent with how isupper is used EVERYWHERE in the codebase:
   - Phase 1: ambiguous_names excludes isupper names (line ~270)
   - Phase 1: GENERIC_PARTIALS excludes isupper parts (line ~280)
   - _is_ambiguous_name_component: returns False for isupper (line ~1684)
   - _is_structural_name: returns True for isupper (line ~485)
   All of these encode the same design principle: acronyms are architectural.

3. Removing isupper would NOT fix the DB FP problem because \\bdb\\b does NOT
   match "database". The fix must be applied elsewhere (Phase 5 extraction
   or Phase 6 validation -- the LLM matching "database" to "DB" via the
   "Database" partial reference).

4. In the rare case where lowercase "ui" or "e2e" appears in text (teammates
   dotted paths), removing isupper would correctly flag those as generic.
   But this is a marginal benefit since Phase 5 already excludes dotted paths.

BOTTOM LINE: The isupper check is a REDUNDANT SAFETY NET, not a bug.
It happens to be a no-op for current data, but it encodes the correct
principle that acronyms are never generic words.

The DB FP problem is caused by Phase 5 entity extraction matching
"database" (generic concept) to "DB" (component) via the Phase 3
partial reference "Database" -> DB, NOT by the isupper check.
""")


if __name__ == "__main__":
    main()
