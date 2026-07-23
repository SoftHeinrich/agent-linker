#!/usr/bin/env python3
"""Spike 005 — Step 2 ($0, cache-only): classify HOW thinking-on extracts the
links effort-0 (nothink) never proposes.

Spike 004 showed ~1.1 macro-F1 is "extraction-bound": gold links in the thinking-on
candidate pool but NOT the nothink pool (no validator can recover them). The open
question (user): can the same reasoning-relocation mechanism that fixed the
validation gates also fix extraction? That depends on WHY thinking-on surfaces these
mentions and effort-0 does not. This harness reads the frozen thinking-on cache and,
for each extraction-bound link, dumps the mechanism thinking-on used:

  - literal      : the component name (or close variant) is literally in the sentence,
                   and thinking-on extracted it as an ENTITY mention. effort-0 simply
                   skipped an explicit/near-explicit mention -> a recall-expansion
                   scaffold (or case-insensitive sweep) can plausibly replicate it CHEAPLY.
  - coref        : surfaced only via coref discovery (anaphor -> antecedent). Needs an
                   anaphora-resolution scaffold, not a justification field.
  - indirect     : entity mention_type == indirect / alias_used set (role/participant
                   reference, name not verbatim) -> hardest; genuine inference.

Pure cache read, no LLM, $0. Run from repo root:
  python .planning/spikes/005-upstream-candidate-gap/harness/extraction_mechanism.py
"""
import os
import sys
from collections import defaultdict

H004 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "004-nogap-validator-ab", "harness")
sys.path.insert(0, os.path.abspath(H004))
import cache_io as C


def entity_index(cell):
    """(s, comp_id) -> entity CandidateLink for this cell's pool."""
    return {(c.sentence_number, c.component_id): c for c in cell["layer3"]["candidates"]}


def coref_index(cell):
    """(s, comp_id) -> coref SadSamLink for this cell's pool."""
    return {(lk.sentence_number, lk.component_id): lk for lk in cell["layer4"]["coref_raw"]}


def classify(key, ecand, cell, sent_map, comp_name):
    """Return (mechanism, detail-string) for one thinking-on extraction-bound link."""
    s_txt = sent_map[key[0]].text if key[0] in sent_map else ""
    name_in_text = comp_name.lower() in s_txt.lower()
    eb = cell["layer3"]["evidence_bundles"].get(key, {})
    cm = cell["layer4"]["coref_metadata"].get(key, {})

    if ecand is not None:
        mt = (ecand.mention_type or "").lower()
        matched = ecand.matched_text or eb.get("matched_span", "")
        alias = ecand.alias_used
        # indirect / alias = name not stated verbatim; the LLM inferred a role/participant
        if "indirect" in mt or alias:
            mech = "indirect" if not name_in_text else "literal"
        else:
            mech = "literal"
        detail = (f"entity matched='{matched}' mtype='{ecand.mention_type}' "
                  f"alias={alias!r} name_in_text={name_in_text}")
        return mech, detail, s_txt

    # coref-only
    if cm:
        return ("coref",
                f"coref ref='{cm.get('reference','')}' <- S{cm.get('antecedent_sentence','?')} "
                f"'{cm.get('antecedent_text','')}' via_alias={cm.get('antecedent_via_alias')} "
                f"name_in_text={name_in_text}",
                s_txt)
    return ("coref", f"coref (no metadata) name_in_text={name_in_text}", s_txt)


def main():
    # distinct extraction-bound link -> (#runs seen, ds, comp_name, mechanism, detail, s_txt)
    found = {}
    mech_count = defaultdict(lambda: defaultdict(int))   # ds -> mech -> n
    runs_seen = defaultdict(int)                          # key -> count

    for run in C.RUNS:
        for ds in C.DATASETS:
            nt = C.load_cell(C.NOTHINK_ROOT, run, ds)
            to = C.load_cell(C.THINKING_ROOT, run, ds)
            bench = C.load_benchmark(ds)
            gold, id2n, sent_map = bench["gold"], bench["id_to_name"], bench["sent_map"]

            nt_ei, nt_ci = entity_index(nt), coref_index(nt)
            to_ei, to_ci = entity_index(to), coref_index(to)
            nt_pool = set(nt_ei) | set(nt_ci)
            to_pool = set(to_ei) | set(to_ci)

            for key in (gold & to_pool) - nt_pool:      # extraction-bound this cell
                runs_seen[(ds, key)] += 1
                if (ds, key) in found:
                    continue
                comp_name = id2n.get(key[1], key[1])
                ecand = to_ei.get(key)                  # prefer entity evidence
                mech, detail, s_txt = classify(key, ecand, to, sent_map, comp_name)
                found[(ds, key)] = (ds, key[0], comp_name, mech, detail, s_txt)

    # tally mechanisms over distinct links
    for (ds, key), rec in found.items():
        mech_count[ds][rec[3]] += 1

    print("=" * 78)
    print("SPIKE 005 Step 2 — mechanism of extraction-bound links (thinking-on, $0)")
    print("=" * 78)
    print("\nDistinct (sentence, component) gold links thinking-on extracts but nothink never\n"
          "proposes, classified by HOW thinking-on surfaced them:\n")

    grand = defaultdict(int)
    print(f"  {'dataset':14s} {'literal':>8s} {'coref':>7s} {'indirect':>9s} {'total':>6s}")
    for ds in C.DATASETS:
        lit = mech_count[ds]["literal"]; cor = mech_count[ds]["coref"]; ind = mech_count[ds]["indirect"]
        tot = lit + cor + ind
        grand["literal"] += lit; grand["coref"] += cor; grand["indirect"] += ind
        print(f"  {ds:14s} {lit:8d} {cor:7d} {ind:9d} {tot:6d}")
    gt = sum(grand.values())
    print(f"  {'ALL':14s} {grand['literal']:8d} {grand['coref']:7d} {grand['indirect']:9d} {gt:6d}")
    if gt:
        print(f"\n  literal  (explicit/near-explicit mention effort-0 skipped) : "
              f"{grand['literal']:3d}  ({100*grand['literal']/gt:.0f}%)  <- cheap recall-scaffold target")
        print(f"  coref    (anaphora; needs resolution scaffold)             : "
              f"{grand['coref']:3d}  ({100*grand['coref']/gt:.0f}%)")
        print(f"  indirect (role/participant inference; hardest)             : "
              f"{grand['indirect']:3d}  ({100*grand['indirect']/gt:.0f}%)")

    # full per-link dump (bbb first — the dominant cell)
    print("\n" + "-" * 78)
    print("Per-link detail (robust = seen in all 3 runs):\n")
    for ds in ["bigbluebutton"] + [d for d in C.DATASETS if d != "bigbluebutton"]:
        links = sorted([k for k in found if k[0] == ds], key=lambda k: k[1][0])
        if not links:
            continue
        print(f"  [{ds}]  {len(links)} distinct extraction-bound links")
        for (d, key) in links:
            _, s_no, cname, mech, detail, s_txt = found[(d, key)]
            r = runs_seen[(d, key)]
            print(f"    S{s_no:<4d} {cname:18.18s} [{mech:8s}] runs={r}/3")
            print(f"          {detail}")
            print(f"          sent: {s_txt[:96]}")
        print()


if __name__ == "__main__":
    main()
