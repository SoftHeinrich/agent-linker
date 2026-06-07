"""Systematic scan for gold-standard SAD-SAM convention inconsistencies.

Three inconsistency patterns surfaced empirically:
  P1 (partial-name overlap): a surface form X is tagged as component X-Y in
      some sentences but not others, with no textual cue distinguishing them.
  P2 (heading vs prose): one-word / fragment sentences naming a component
      are tagged in some places, untagged in others.
  P3 (multi-mention sentence partial coverage): a sentence names ≥2
      components but gold tags only a subset.

For each project, scan the gold standard + document text and report:
  - All instances of each pattern with concrete examples
  - How our s_linker19 final + Artemis behave on each

Goal: enumerate the conventions the gold encodes but never documents.
"""
from __future__ import annotations

import csv
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

APPROACH = Path("/mnt/hostshare/ardoco-home/mono/approach")
EVAL_LIB = Path("/mnt/hostshare/ardoco-home/mono/evaluation/src/lib")
sys.path.insert(0, str(APPROACH / "src"))
sys.path.insert(0, str(EVAL_LIB))

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map  # noqa: E402
from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402
from transarc_error_analysis import PROJECTS, load_gs_sad_sam  # noqa: E402

BENCH = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
ARTEMIS_DIR = Path("/mnt/hostshare/ardoco-home/transarc-emp/results_artemis_gpt54")
CACHE = APPROACH / "results" / "phase_cache" / "s_linker19"

TEXT_YEAR = {"mediastore": "2016", "teastore": "2020", "teammates": "2021",
             "bigbluebutton": "2021", "jabref": "2021"}
PCM_FILE = {"mediastore": "ms", "teastore": "teastore", "teammates": "teammates",
            "bigbluebutton": "bbb", "jabref": "jabref"}


def text_path(p): return BENCH / p / f"text_{TEXT_YEAR[p]}" / f"{p}.txt"
def pcm_path(p):  return BENCH / p / f"model_{TEXT_YEAR[p]}" / "pcm" / f"{PCM_FILE[p]}.repository"


def load_artemis(p):
    out = set()
    with open(ARTEMIS_DIR / p / "sad-sam" / f"sadSamTlr_{p}.csv") as f:
        for r in csv.DictReader(f):
            out.add((str(r["sentence"]).strip(), r["modelElementID"].strip()))
    return out


def load_final(backend, project):
    with open(CACHE / backend / project / "final.pkl", "rb") as f:
        d = pickle.load(f)
    return {(str(l.sentence_number), l.component_id) for l in d["final"]}


def project_data(project):
    sentences = load_sentences(str(text_path(project)))
    sent_map = build_sent_map(sentences)
    components = parse_pcm_repository(str(pcm_path(project)))
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    gold = [(int(s), m) for (m, s) in load_gs_sad_sam(project)]
    gold_by_sent = defaultdict(set)
    for s, m in gold:
        gold_by_sent[s].add(m)
    return {
        "sentences": sentences, "sent_map": sent_map, "components": components,
        "name_to_id": name_to_id, "id_to_name": id_to_name,
        "gold_by_sent": gold_by_sent,
    }


# ── Pattern 1: partial-name overlap ──────────────────────────────────────────
# For each compound component name with a salient prefix or token (e.g.
# "WebRTC-SFU" -> "WebRTC", "GAE Datastore" -> "GAE"), find sentences where
# the prefix appears but the full canonical name doesn't, and check gold.

def partial_name_overlap(d, project):
    """Returns list of (token, component_name, sentences_tagged, sentences_untagged)."""
    results = []
    # Build "prefix → component" candidates from compound names — split ONLY on
    # whitespace / hyphen / dot, not camelCase (don't break "WebRTC" into "Web RTC").
    prefix_to_comp: dict[str, list[str]] = defaultdict(list)
    for c in d["components"]:
        name = c.name
        tokens = re.split(r"[\s\-.]+", name)
        tokens = [t for t in tokens if t and len(t) >= 3]
        if len(tokens) >= 2:
            for t in tokens:
                if t != name and t.lower() != name.lower():
                    prefix_to_comp[t].append(name)
    # For each prefix → component(s), scan sentences
    for prefix, comps in prefix_to_comp.items():
        if len(comps) != 1:
            continue  # ambiguous prefix
        comp = comps[0]
        cid = d["name_to_id"][comp]
        tagged = []; untagged = []
        for s in d["sentences"]:
            # Sentence contains the bare prefix but NOT the full canonical name?
            has_prefix = bool(re.search(rf'\b{re.escape(prefix)}\b', s.text))
            has_full = has_standalone_mention(comp, s.text)
            if not has_prefix or has_full:
                continue
            is_tagged = cid in d["gold_by_sent"].get(s.number, set())
            (tagged if is_tagged else untagged).append((s.number, s.text))
        if tagged and untagged:
            results.append((prefix, comp, tagged, untagged))
    return results


# ── Pattern 2: short/heading sentences with bare component name ──────────────

def heading_sentences(d, project):
    """Sentences ≤ 10 words containing a component name. Compare gold tagging."""
    flagged = []
    for s in d["sentences"]:
        wc = len(s.text.split())
        if wc > 10:
            continue
        names_present = [c.name for c in d["components"]
                         if has_standalone_mention(c.name, s.text)]
        for cn in names_present:
            cid = d["name_to_id"][cn]
            is_tagged = cid in d["gold_by_sent"].get(s.number, set())
            flagged.append((s.number, s.text, cn, is_tagged))
    return flagged


# ── Pattern 4: per-component cross-sentence tagging variance ─────────────────
# For each component, find all sentences with standalone name mentions, and
# compare tagged-rate. If a component is tagged in some "X-mentioning"
# sentences but not others, surface the untagged outliers (likely gold gaps).

def per_component_variance(d, project):
    """Returns list of (comp_name, tagged_sents, untagged_sents)."""
    out = []
    for c in d["components"]:
        cid = c.id
        tagged, untagged = [], []
        for s in d["sentences"]:
            if not has_standalone_mention(c.name, s.text):
                continue
            if cid in d["gold_by_sent"].get(s.number, set()):
                tagged.append((s.number, s.text))
            else:
                untagged.append((s.number, s.text))
        # Only interesting when both sets non-empty AND at least 2 sentences total
        if tagged and untagged and len(tagged) + len(untagged) >= 2:
            out.append((c.name, tagged, untagged))
    return out


# ── Pattern 3: multi-mention sentences with partial gold coverage ────────────

def multi_mention_partial_gold(d, project):
    """Sentences mentioning ≥2 components but with gold tagging only a subset."""
    out = []
    for s in d["sentences"]:
        present = []
        for c in d["components"]:
            if has_standalone_mention(c.name, s.text):
                present.append((c.name, c.id))
        if len(present) < 2:
            continue
        tagged_ids = d["gold_by_sent"].get(s.number, set())
        present_tagged = [p for p in present if p[1] in tagged_ids]
        present_untagged = [p for p in present if p[1] not in tagged_ids]
        if present_tagged and present_untagged:
            out.append((s.number, s.text, present_tagged, present_untagged))
    return out


# ── Cross-check with our system + Artemis to see who agrees with gold ────────

def check_predictions(project, sn, cid):
    sn_str = str(sn)
    cl_final = load_final("claude", project)
    oa_final = load_final("openai", project)
    art = load_artemis(project)
    return {
        "claude": (sn_str, cid) in cl_final,
        "openai": (sn_str, cid) in oa_final,
        "artemis": (sn_str, cid) in art,
    }


def render_pattern1(buf, d, proj, counts):
    p1 = partial_name_overlap(d, proj)
    if not p1: return
    buf.append("### Pattern 1 — partial-name overlap (`X` vs `X-Y`)")
    buf.append("")
    for prefix, comp, tagged, untagged in p1:
        buf.append(f"**Prefix `{prefix}` → component `{comp}`** — "
                   f"{len(tagged)} sentences tagged, {len(untagged)} untagged")
        buf.append("")
        buf.append("| sent | gold | text | cl | oa | art |")
        buf.append("|---|---|---|---|---|---|")
        cid = d["name_to_id"][comp]
        rows_total = ([(sn, "**G**", txt) for sn, txt in tagged]
                      + [(sn, "—", txt) for sn, txt in untagged])
        rows_total.sort(key=lambda r: r[0])
        for sn, marker, txt in rows_total:
            pred = check_predictions(proj, sn, cid)
            txt_short = txt if len(txt) < 120 else txt[:117] + "..."
            txt_safe = txt_short.replace("|", "\\|")
            buf.append(f"| S{sn} | {marker} | {txt_safe} | "
                       f"{'✓' if pred['claude'] else '✗'} | "
                       f"{'✓' if pred['openai'] else '✗'} | "
                       f"{'✓' if pred['artemis'] else '✗'} |")
        buf.append("")
        counts["tagged"] += len(tagged); counts["untagged"] += len(untagged)


def render_pattern2(buf, d, proj, counts):
    p2 = heading_sentences(d, proj)
    tagged_rows = [r for r in p2 if r[3]]
    untagged_rows = [r for r in p2 if not r[3]]
    if not (tagged_rows and untagged_rows): return
    buf.append("### Pattern 2 — short sentences (≤ 10 words) with bare component name")
    buf.append("")
    buf.append(f"**{len(tagged_rows)} tagged, {len(untagged_rows)} untagged**")
    buf.append("")
    buf.append("| sent | gold | text | component | cl | oa | art |")
    buf.append("|---|---|---|---|---|---|---|")
    for sn, txt, cn, is_tagged in sorted(p2):
        cid = d["name_to_id"][cn]
        pred = check_predictions(proj, sn, cid)
        marker = "**G**" if is_tagged else "—"
        txt_safe = txt.replace("|", "\\|")
        buf.append(f"| S{sn} | {marker} | {txt_safe} | {cn} | "
                   f"{'✓' if pred['claude'] else '✗'} | "
                   f"{'✓' if pred['openai'] else '✗'} | "
                   f"{'✓' if pred['artemis'] else '✗'} |")
    buf.append("")
    counts["tagged"] += len(tagged_rows); counts["untagged"] += len(untagged_rows)


def render_pattern3(buf, d, proj, counts):
    p3 = multi_mention_partial_gold(d, proj)
    if not p3: return
    buf.append("### Pattern 3 — multi-mention sentences with partial gold coverage")
    buf.append("")
    buf.append(f"{len(p3)} sentences mention ≥2 components; gold tags a subset.")
    buf.append("")
    buf.append("| sent | text | tagged | untagged |")
    buf.append("|---|---|---|---|")
    for sn, txt, tagged_pairs, untagged_pairs in p3[:15]:
        tag_str = ", ".join(n for n, _ in tagged_pairs)
        untag_str = ", ".join(n for n, _ in untagged_pairs)
        txt_safe = (txt if len(txt) < 110 else txt[:107] + "...").replace("|", "\\|")
        buf.append(f"| S{sn} | {txt_safe} | {tag_str} | {untag_str} |")
    if len(p3) > 15:
        buf.append("")
        buf.append(f"_… {len(p3) - 15} more cases truncated_")
    buf.append("")
    counts["sentences"] += len(p3)
    counts["tagged_total"] += sum(len(t) for _, _, t, _ in p3)
    counts["untagged_total"] += sum(len(u) for _, _, _, u in p3)


def render_pattern4(buf, d, proj, counts):
    p4 = per_component_variance(d, proj)
    if not p4: return
    buf.append("### Pattern 4 — same component, same surface form, different gold")
    buf.append("")
    buf.append("Components where the canonical name appears standalone in multiple "
               "sentences, with gold tagging some and not others.")
    buf.append("")
    # Show only components where the inconsistency is "interesting" — at least 1
    # tagged and 1 untagged, and at most 8 untagged sentences (to keep tractable).
    interesting = [(cn, t, u) for (cn, t, u) in p4 if u and len(u) <= 8]
    if not interesting:
        buf.append("_(many components affected; all examples below cap at 8 untagged sentences)_")
        buf.append("")
        interesting = p4[:5]
    for cn, tagged, untagged in interesting[:8]:
        cid = d["name_to_id"][cn]
        buf.append(f"**`{cn}`** — {len(tagged)} sentences tagged, "
                   f"{len(untagged)} untagged")
        buf.append("")
        buf.append("| sent | gold | text | cl | oa | art |")
        buf.append("|---|---|---|---|---|---|")
        rows_total = ([(sn, "**G**", txt) for sn, txt in tagged]
                      + [(sn, "—", txt) for sn, txt in untagged])
        rows_total.sort(key=lambda r: r[0])
        # Cap display
        for sn, marker, txt in rows_total[:10]:
            pred = check_predictions(proj, sn, cid)
            txt_short = txt if len(txt) < 110 else txt[:107] + "..."
            txt_safe = txt_short.replace("|", "\\|")
            buf.append(f"| S{sn} | {marker} | {txt_safe} | "
                       f"{'✓' if pred['claude'] else '✗'} | "
                       f"{'✓' if pred['openai'] else '✗'} | "
                       f"{'✓' if pred['artemis'] else '✗'} |")
        if len(rows_total) > 10:
            buf.append(f"")
            buf.append(f"_… {len(rows_total) - 10} more sentences truncated_")
        buf.append("")
        counts["components"] += 1
        counts["tagged_total"] += len(tagged)
        counts["untagged_total"] += len(untagged)


def main():
    out_md = APPROACH / "doc" / "gold_standard_inconsistencies.md"
    lines = [
        "# SAD-SAM gold-standard convention inconsistencies",
        "",
        "Systematic scan across the 5 ARDoCo benchmark projects (mediastore, "
        "teastore, teammates, bigbluebutton, jabref). Generated by "
        "`scripts/gold_inconsistency_scan.py`.",
        "",
        "Columns `cl`/`oa`/`art` show whether s_linker19 (claude), "
        "s_linker19 (openai), and Artemis (TAAS25, gpt-5.4 5x avg) emit the link. "
        "✓ = emitted, ✗ = not emitted. **G** marks the gold-standard verdict.",
        "",
        "**Patterns scanned:**",
        "- **Pattern 1** — partial-name overlap (`WebRTC` in sentences tagged "
        "as `WebRTC-SFU` in some places but not others)",
        "- **Pattern 2** — short / heading sentences (≤ 10 words) containing "
        "a bare component name, tagged inconsistently",
        "- **Pattern 3** — multi-mention sentences where gold tags only a "
        "subset of the named components",
        "- **Pattern 4** — same component, same canonical surface form, "
        "different gold verdicts across sentences",
        "",
    ]
    counts = {
        "p1": {"tagged": 0, "untagged": 0},
        "p2": {"tagged": 0, "untagged": 0},
        "p3": {"sentences": 0, "tagged_total": 0, "untagged_total": 0},
        "p4": {"components": 0, "tagged_total": 0, "untagged_total": 0},
    }

    for proj in PROJECTS:
        d = project_data(proj)
        proj_buf = []
        render_pattern1(proj_buf, d, proj, counts["p1"])
        render_pattern2(proj_buf, d, proj, counts["p2"])
        render_pattern3(proj_buf, d, proj, counts["p3"])
        render_pattern4(proj_buf, d, proj, counts["p4"])
        if proj_buf:
            lines.append(f"## {proj}")
            lines.append("")
            lines.extend(proj_buf)

    # Summary
    lines.append("## Summary across all projects")
    lines.append("")
    lines.append(f"- **Pattern 1 (partial-name overlap)**: "
                 f"{counts['p1']['tagged']} tagged + {counts['p1']['untagged']} untagged "
                 f"sentences. No textual cue distinguishes the groups.")
    lines.append(f"- **Pattern 2 (short/heading sentences)**: "
                 f"{counts['p2']['tagged']} tagged + {counts['p2']['untagged']} untagged "
                 f"sentences.")
    lines.append(f"- **Pattern 3 (multi-mention partial coverage)**: "
                 f"{counts['p3']['sentences']} sentences — "
                 f"{counts['p3']['tagged_total']} tagged / "
                 f"{counts['p3']['untagged_total']} untagged mentions.")
    lines.append(f"- **Pattern 4 (cross-sentence variance, same canonical name)**: "
                 f"{counts['p4']['components']} components affected — "
                 f"{counts['p4']['tagged_total']} tagged / "
                 f"{counts['p4']['untagged_total']} untagged sentences.")
    lines.append("")
    lines.append("## Implications")
    lines.append("")
    lines.append("Pattern 4 surfaces two distinct sub-cases that the scan does not "
                 "separate automatically:")
    lines.append("")
    lines.append("**4a — genuine gold gaps**: same canonical surface form, same "
                 "semantic role (architectural participation), different verdict. "
                 "Example: bbb `FreeSWITCH` — S58/S59/S62/S63/S66/S72 tagged, S60 "
                 "untagged, but S60 (\"Communication between apps and FreeSWITCH "
                 "Event Socket Layer (fsels) uses messages through redis pubsub\") "
                 "describes FreeSWITCH's role as cleanly as the tagged sentences. "
                 "Artemis catches S60 (✓); we don't (✗ ✗). Gold gap.")
    lines.append("")
    lines.append("**4b — legitimate semantic distinctions**: same canonical surface "
                 "form, *different* semantic role. Gold's untagged verdict is the "
                 "correct one — the bare name is incidental, not architectural. "
                 "Example: jabref S7 \"*Only the gui knows the user and his "
                 "preferences*\" — \"preferences\" here is the *user's* preferences, "
                 "not the `preferences` package. Our claude correctly rejects (✗), "
                 "our openai treats it as the package (✓ → FP).")
    lines.append("")
    lines.append("Distinguishing 4a from 4b requires per-sentence semantic reasoning "
                 "of exactly the kind a validator should do — so flagging Pattern 4 "
                 "cases does *not* automatically mean the gold is wrong. It means "
                 "the surface form alone is insufficient to predict the gold verdict.")
    lines.append("")
    lines.append("### Consequences for evaluation")
    lines.append("")
    lines.append("1. **F1 ceiling for any *surface-form-only* method is bounded "
                 "below 1.0** — Pattern 1 and Pattern 2 cases tag the same surface "
                 "pattern inconsistently. Artemis's NER-occurrence approach, our "
                 "string-matching extraction, and any rule-based linker hit this "
                 "ceiling.")
    lines.append("")
    lines.append("2. **Patterns 3 and 4 are the validator's actual job** — they "
                 "require reading the sentence semantics to distinguish architectural "
                 "participation from incidental mention. The genuine Pattern 4a gaps "
                 "(FreeSWITCH S60) reveal gold-tagger fatigue; the 4b cases reward "
                 "validators that do real reading (\"his preferences\" → user-owned, "
                 "not package).")
    lines.append("")
    lines.append("3. **Artemis's higher recall partly reflects matching the gold's "
                 "'name occurrence ⇒ link' convention.** Without a per-sentence "
                 "validator, Artemis emits every NER occurrence — which happens to "
                 "match how the gold was constructed for most sentences. Our system "
                 "trades that for stronger Pattern 3/4b discrimination (jabref S7 "
                 "FP avoidance), at the cost of missing some Pattern 4a tags.")
    lines.append("")
    lines.append("4. **For per-error attribution in the paper**, distinguish:")
    lines.append("   - **Surface-pattern FPs/FNs** (Patterns 1, 2) → benchmark "
                 "ceiling, document as such")
    lines.append("   - **Pattern 4a FNs** → gold gaps, document as such")
    lines.append("   - **Pattern 4b FPs** → real validator errors, fix or accept")
    lines.append("   - **Pattern 3 partial-tags** → ambiguous, case-by-case")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md} ({len(lines)} lines)")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
