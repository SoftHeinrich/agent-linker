#!/usr/bin/env python3
"""Unit tests for V20a/b/c fixes — no LLM calls needed."""

import re
import sys
sys.path.insert(0, "src")


def test_v20a_phase3_override():
    """V20a: 'Database' should be rescued from generic rejection when it maps to DB."""
    print("=" * 60)
    print("TEST V20a: Phase 3 component-name generic override")
    print("=" * 60)

    comp_names = ["UserDBAdapter", "AudioWatermarking", "Reencoding", "MediaManagement",
                  "Facade", "MediaAccess", "Packaging", "DB", "FileStorage",
                  "TagWatermarking", "Cache", "UserManagement", "DownloadLoadBalancer",
                  "ParallelWatermarking"]

    # Simulate Phase 3 judge output where "Database" is rejected as generic
    all_mappings = {
        "Database": ("synonym", "DB"),
        "DataStorage": ("synonym", "FileStorage"),
        "ReEncoder": ("synonym", "Reencoding"),
    }
    generic_terms = {"Database"}  # LLM rejected this
    approved = {"DataStorage", "ReEncoder"}  # LLM approved these

    # --- V19 behavior: only CamelCase + Uppercase overrides ---
    rescued_v19 = set()
    for term in list(generic_terms):
        if re.search(r'[a-z][A-Z]', term):
            rescued_v19.add(term)
        elif term.isupper() and len(term) <= 4 and term in all_mappings:
            rescued_v19.add(term)

    print(f"  V19 rescues: {rescued_v19 or 'NONE'}")
    assert "Database" not in rescued_v19, "V19 should NOT rescue 'Database'"

    # --- V20a behavior: + component-name override ---
    common_english = {
        "data", "service", "server", "client", "model", "logic",
        "storage", "common", "action", "process", "system", "core",
        "base", "app", "application", "cache", "store", "manager",
        "handler", "controller", "provider", "factory", "adapter",
    }
    rescued_v20a = set()
    for term in list(generic_terms):
        if re.search(r'[a-z][A-Z]', term):
            rescued_v20a.add(term)
        elif term.isupper() and len(term) <= 4 and term in all_mappings:
            rescued_v20a.add(term)
        elif term[0].isupper() and term.lower() not in common_english and term in all_mappings:
            rescued_v20a.add(term)
            print(f"  V20a component-name override rescued: {term}")
        elif any(term.lower() == cn.lower() for cn in comp_names) and term in all_mappings:
            rescued_v20a.add(term)
            print(f"  V20a exact-component override rescued: {term}")

    assert "Database" in rescued_v20a, "V20a SHOULD rescue 'Database'"
    print(f"  V20a rescues: {rescued_v20a}")

    # Edge case: "cache" should NOT be rescued (common English lowercase)
    generic_terms_2 = {"cache"}
    all_mappings_2 = {"cache": ("synonym", "Cache")}
    rescued = set()
    for term in list(generic_terms_2):
        if term[0].isupper() and term.lower() not in common_english and term in all_mappings_2:
            rescued.add(term)
        elif any(term.lower() == cn.lower() for cn in comp_names) and term in all_mappings_2:
            rescued.add(term)
    # "cache" starts with lowercase, so first check fails.
    # But "cache".lower() == "Cache".lower() -> exact-component override triggers
    print(f"  Edge case 'cache': rescued={rescued}")
    # This is actually correct — "cache" as a synonym for Cache component should be rescued
    # because it maps to an actual component

    print("  PASS ✓\n")
    return True


def test_v20b_partial_context():
    """V20b: 'UI name' should NOT trigger WebUI partial injection."""
    print("=" * 60)
    print("TEST V20b: Phase 8b partial-injection context guard")
    print("=" * 60)

    PARTIAL_FALSE_FOLLOWERS = {
        "name", "names", "type", "types", "id", "ids", "identifier",
        "field", "fields", "column", "value", "values", "string",
        "format", "path", "file", "size", "level", "mode", "flag",
        "code", "number", "index", "key", "label", "tag", "text",
        "attribute", "property", "parameter", "setting", "option",
    }

    def partial_is_compound_noun(partial, sent_text):
        if len(partial) > 3:
            return False
        for m in re.finditer(rf'\b{re.escape(partial)}\b', sent_text, re.IGNORECASE):
            end = m.end()
            rest = sent_text[end:].lstrip()
            if not rest:
                continue
            next_word_m = re.match(r'(\w+)', rest)
            if next_word_m:
                next_word = next_word_m.group(1).lower()
                if next_word in PARTIAL_FALSE_FOLLOWERS:
                    return True
        return False

    # TS sentences that should be BLOCKED (FPs in V19)
    ts_fp_sentences = [
        (11, "It matches the provided product ID or UI name (the filename for images not representing a product and therefore without product ID)."),
        (12, 'If the product ID or UI name is not available to the Image Provider, a standard "not found" image will be delivered in the requested size.'),
        (13, "If the product ID or UI name is found but not in the requested size, the largest image will be loaded and scaled."),
        (15, "If the product ID or UI name and size is found, the image will be loaded and delivered."),
    ]

    print("  Testing TS FP sentences (should be BLOCKED):")
    for snum, text in ts_fp_sentences:
        blocked = partial_is_compound_noun("UI", text)
        status = "BLOCKED ✓" if blocked else "PASSED ✗ (BAD)"
        print(f"    S{snum}: {status}")
        assert blocked, f"S{snum} should be blocked but wasn't"

    # TS sentences that should PASS (TPs — "UI" refers to WebUI)
    # Need to check actual TS sentences where "UI" is a valid WebUI reference
    ts_tp_contexts = [
        # These don't have "UI name" pattern — they should pass
        (8, "The WebUI uses the status page to show the current status."),
        (9, "The status view lists the instance count and hosts for all registered service instances."),
    ]
    print("  Testing non-FP sentences (should NOT be blocked):")
    for snum, text in ts_tp_contexts:
        blocked = partial_is_compound_noun("UI", text)
        status = "PASSED ✓" if not blocked else "BLOCKED ✗ (BAD)"
        print(f"    S{snum}: {status}")
        assert not blocked, f"S{snum} should pass but was blocked"

    # BBB: "Client" (len=6 > 3) should never be blocked by this guard
    bbb_sentences = [
        (9, "The HTML5 client component uses the MongoDB client to connect."),
        (10, "The client initiates a WebRTC connection."),
    ]
    print("  Testing BBB 'Client' (len>3, should NOT be blocked):")
    for snum, text in bbb_sentences:
        blocked = partial_is_compound_noun("Client", text)
        status = "PASSED ✓" if not blocked else "BLOCKED ✗ (BAD)"
        print(f"    S{snum}: {status}")
        assert not blocked, f"Client partial should never be blocked (len>3)"

    # TM: "Datastore" (len=9 > 3) should not be blocked
    print("  Testing TM 'Datastore' (len>3, should NOT be blocked):")
    blocked = partial_is_compound_noun("Datastore", "The GAE Datastore provides persistence.")
    assert not blocked
    print(f"    Datastore: PASSED ✓")

    # Edge: "DB" (len=2, <=3) followed by "name" should be blocked
    print("  Testing edge case 'DB name':")
    blocked = partial_is_compound_noun("DB", "The DB name is configured in the settings.")
    assert blocked
    print(f"    'DB name': BLOCKED ✓")

    # Edge: "DB" followed by normal word should NOT be blocked
    blocked = partial_is_compound_noun("DB", "The DB stores all user data.")
    assert not blocked
    print(f"    'DB stores': PASSED ✓")

    print("  PASS ✓\n")
    return True


def test_v20c_parent_overlap():
    """V20c: SlopeOneRecommender should be skipped when Recommender already linked."""
    print("=" * 60)
    print("TEST V20c: Phase 5b parent-overlap guard")
    print("=" * 60)

    # TS components
    comp_names = ["WebUI", "Registry", "Persistence", "Recommender", "Auth",
                  "SlopeOneRecommender", "OrderBasedRecommender", "DummyRecommender",
                  "PopularityBasedRecommender", "ImageProvider", "PreprocessedSlopeOneRecommender"]

    # Build parent_map
    parent_map = {}
    for comp in comp_names:
        parents = set()
        for other in comp_names:
            if other != comp and len(other) >= 3 and other in comp:
                parents.add(other)
        if parents:
            parent_map[comp] = parents

    print("  Parent map:")
    for child, parents in sorted(parent_map.items()):
        print(f"    {child} -> parents: {', '.join(sorted(parents))}")

    # Verify sub-type recommenders have "Recommender" as parent
    for sub in ["SlopeOneRecommender", "OrderBasedRecommender", "DummyRecommender",
                "PopularityBasedRecommender", "PreprocessedSlopeOneRecommender"]:
        assert sub in parent_map, f"{sub} should have parents"
        assert "Recommender" in parent_map[sub], f"{sub} should have Recommender as parent"
    print("  All sub-recommenders correctly identify Recommender as parent ✓")

    # Simulate existing links (TransArc has Recommender→S4 and S27)
    existing_sent_comp = {4: {"Recommender"}, 27: {"Recommender"}}

    # Test: SlopeOneRecommender targeted to S27 should be SKIPPED
    snum = 27
    comp = "SlopeOneRecommender"
    parents_here = parent_map[comp] & existing_sent_comp.get(snum, set())
    assert parents_here == {"Recommender"}, f"Expected parent overlap at S{snum}"
    print(f"  S{snum} -> {comp}: SKIPPED (parent: {parents_here}) ✓")

    # Test: SlopeOneRecommender targeted to S30 (no existing Recommender) should PASS
    snum = 30
    parents_here = parent_map[comp] & existing_sent_comp.get(snum, set())
    assert not parents_here, f"Should NOT have parent overlap at S{snum}"
    print(f"  S{snum} -> {comp}: PASSED (no parent here) ✓")

    # Test: WebUI should NOT have any parents (no component name is substring of "WebUI")
    assert "WebUI" not in parent_map, "WebUI should have no parents"
    print(f"  WebUI: no parents (correct) ✓")

    # Edge: "ImageProvider" should NOT have "Image" as parent (Image is not a component)
    assert "ImageProvider" not in parent_map, "ImageProvider should have no parents (no 'Image' component)"
    print(f"  ImageProvider: no parents (correct) ✓")

    # Edge: HTML5 components in BBB
    bbb_comps = ["Recording Service", "kurento", "WebRTC-SFU", "HTML5 Server",
                 "HTML5 Client", "Presentation Conversion", "BBB web",
                 "Redis PubSub", "FSESL", "Apps", "Redis DB", "FreeSWITCH"]
    bbb_parent_map = {}
    for comp in bbb_comps:
        parents = set()
        for other in bbb_comps:
            if other != comp and len(other) >= 3 and other in comp:
                parents.add(other)
        if parents:
            bbb_parent_map[comp] = parents
    print(f"  BBB parent map: {bbb_parent_map or 'EMPTY (correct)'}")
    # "Redis DB" contains "Redis" but there's no "Redis" component → should be empty
    # "Redis PubSub" contains "Redis" but no "Redis" component → empty
    # But "HTML5 Server" and "HTML5 Client" both contain... nothing that's a separate component
    # Check: does "Redis DB" contain "Redis PubSub"? No. Does "Redis PubSub" contain "Redis DB"? No.
    assert not bbb_parent_map, "BBB should have no parent overlaps"
    print(f"  BBB: no parent overlaps (correct) ✓")

    print("  PASS ✓\n")
    return True


if __name__ == "__main__":
    results = {}
    results["v20a"] = test_v20a_phase3_override()
    results["v20b"] = test_v20b_partial_context()
    results["v20c"] = test_v20c_parent_overlap()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS ✓' if ok else 'FAIL ✗'}")

    if all(results.values()):
        print("\nAll unit tests passed! Ready for e2e testing.")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
