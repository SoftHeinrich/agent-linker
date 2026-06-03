#!/usr/bin/env python3
"""Test seed reference disambiguation against gold standards.

Loads S-Linker11 checkpoint data, runs knowledge-aware LLM disambiguation
on raw seed links, and compares results against gold standards.

Design: per-component LLM pass with component dossier:
  - Ambiguity classification (from model_knowledge)
  - Known aliases (from doc_knowledge)
  - Anchor sentences (calibration)
  - Match context per seed (how the name appears in the sentence)
"""

import csv
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.llm_client import LLMClient
from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
CACHE_DIR = Path("results/phase_cache/s_linker11")
BENCHMARK = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

SEED_DISAMBIGUATION_RULES = """REFERENCE DISAMBIGUATION — determine what the name means in each sentence.

COMPONENT (approve): The sentence discusses this architectural component —
it performs actions, provides services, is described, configured, listed,
or referenced by name in any grammatical role.

OTHER (reject): The name clearly carries a different meaning:
- Code-level notation: the name appears inside a package path, qualified
  identifier, or a sentence that enumerates code-level identifiers
- Technique or methodology: the sentence describes an algorithm, pattern,
  or approach that shares the component's name — not what the component
  does as an architectural participant
- Embedded sub-entity: the name appears only as part of a longer proper
  name that denotes a different, more specific entity
- Different entity: the sentence refers to a similarly-named but distinct
  thing (the name partially overlaps but the full reference is different)
- Generic English: the word is used with its ordinary dictionary meaning

When uncertain, choose COMPONENT — these candidates passed independent extraction."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(dataset: str, phase: str):
    path = CACHE_DIR / dataset / f"{phase}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gold_standard(dataset: str) -> set[tuple[int, str]]:
    """Load gold standard as {(sentence_number, component_id)} set."""
    gs_dir = BENCHMARK / dataset / "goldstandards"
    # Exclude UME/MME variants — we want the base SAD-SAM gold standard
    candidates = [
        f for f in gs_dir.glob("goldstandard_sad*sam*.csv")
        if "UME" not in f.name and "MME" not in f.name
    ]
    assert candidates, f"No gold standard found in {gs_dir}"
    gs_path = candidates[0]

    links = set()
    with open(gs_path) as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            sent = int(row.get("sentence", row.get("sentenceNo", 0)))
            model_id = row.get("modelElementID", "")
            links.add((sent, model_id))
    return links


def has_standalone_mention(comp_name: str, text: str) -> bool:
    """Check for clean standalone mention (proper case, not in dotted path)."""
    if not comp_name:
        return False
    is_single = " " not in comp_name
    if is_single:
        if comp_name[0].islower():
            pattern = rf"\b{re.escape(comp_name)}\b"
            flags = 0
        else:
            cap = comp_name[0].upper() + comp_name[1:]
            pattern = rf"\b{re.escape(cap)}\b"
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


def classify_mention(comp_name: str, text: str, doc_knowledge: DocumentKnowledge) -> str:
    """Classify how the component name appears in the sentence.

    Returns a human-readable match description for the LLM prompt.
    """
    # Check exact proper-case standalone mention
    if has_standalone_mention(comp_name, text):
        # Check if it's embedded in a longer phrase
        name_esc = re.escape(comp_name)
        for m in re.finditer(rf"\b{name_esc}\b", text, re.IGNORECASE if " " in comp_name else 0):
            s, e = m.start(), m.end()
            # Check for longer compound: "FreeSWITCH Event Socket Layer"
            rest = text[e:].lstrip()
            if rest and rest[0].isupper():
                phrase_end = text.find(".", e)
                if phrase_end == -1:
                    phrase_end = min(e + 40, len(text))
                trailing = text[e:phrase_end].strip()
                if trailing and len(trailing.split()) <= 4:
                    return f"proper case, embedded in longer phrase: \"{comp_name}{trailing}\""
            break
        return "proper case, standalone"

    # Check lowercase mention
    comp_lower = comp_name.lower()
    has_lower = bool(re.search(rf"\b{re.escape(comp_lower)}\b", text))
    if has_lower:
        # In dotted path?
        for m in re.finditer(rf"\b{re.escape(comp_lower)}\b", text):
            s, e = m.start(), m.end()
            in_dotted = (s > 0 and text[s - 1] == ".") or (
                e < len(text) and text[e] == "." and e + 1 < len(text) and text[e + 1].isalpha()
            )
            if in_dotted:
                return "lowercase, inside dotted path"
        return "lowercase mention"

    # Check alias match
    for alias, target in doc_knowledge.abbreviations.items():
        if target == comp_name and re.search(rf"\b{re.escape(alias)}\b", text):
            return f"via known abbreviation \"{alias}\""
    for syn, target in doc_knowledge.synonyms.items():
        if target == comp_name and re.search(rf"\b{re.escape(syn)}\b", text, re.IGNORECASE):
            return f"via known synonym \"{syn}\""
    for partial, target in doc_knowledge.partial_references.items():
        if target == comp_name and re.search(rf"\b{re.escape(partial)}\b", text, re.IGNORECASE):
            return f"via known partial reference \"{partial}\""

    # Check hyphenated/compound
    if re.search(rf"{re.escape(comp_lower)}", text.lower()):
        return "embedded in compound word"

    return "indirect/unclear match"


def build_component_profile(
    comp_name: str,
    model_knowledge: ModelKnowledge,
    doc_knowledge: DocumentKnowledge,
) -> str:
    """Build a textual component profile for the LLM prompt."""
    lines = [f"- Name: {comp_name}"]

    # Ambiguity
    is_ambig = comp_name in model_knowledge.ambiguous_names
    if is_ambig:
        lines.append(f"- Classification: AMBIGUOUS — \"{comp_name}\" is a common English word")
    else:
        lines.append(f"- Classification: DISTINCTIVE — architecturally specific name")

    # Known aliases
    aliases = []
    for a, target in doc_knowledge.abbreviations.items():
        if target == comp_name:
            aliases.append(f"\"{a}\" (abbreviation)")
    for s, target in doc_knowledge.synonyms.items():
        if target == comp_name:
            aliases.append(f"\"{s}\" (synonym)")
    for p, target in doc_knowledge.partial_references.items():
        if target == comp_name:
            aliases.append(f"\"{p}\" (partial reference)")

    if aliases:
        lines.append(f"- Known aliases: {', '.join(aliases)}")
    else:
        lines.append("- Known aliases: none")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core disambiguation logic
# ─────────────────────────────────────────────────────────────────────────────

def disambiguate_seeds(
    raw_seeds: list[SadSamLink],
    model_knowledge: ModelKnowledge,
    doc_knowledge: DocumentKnowledge,
    sent_map: dict,
    llm: LLMClient,
) -> list[SadSamLink]:
    """Knowledge-aware seed reference disambiguation.

    Per-component LLM pass with:
    - Component profile (ambiguity, aliases from knowledge base)
    - Anchor sentences (proper-case mentions as calibration)
    - Match context per seed (how the name appears)
    """
    if not raw_seeds:
        return []

    # Group seeds by component
    by_comp: dict[str, list[SadSamLink]] = {}
    for sl in raw_seeds:
        by_comp.setdefault(sl.component_name, []).append(sl)

    verified = []
    total_calls = 0

    for comp_name, seeds in sorted(by_comp.items()):
        seed_sent_nums = {sl.sentence_number for sl in seeds}

        # ── Component profile ──
        profile = build_component_profile(comp_name, model_knowledge, doc_knowledge)

        # ── Anchor sentences (proper-case mentions NOT in seed set) ──
        anchor_lines = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number in seed_sent_nums:
                continue  # Don't use seed sentences as their own calibration
            if has_standalone_mention(comp_name, s.text):
                anchor_lines.append(f"  S{s.number}: \"{s.text}\"")
                if len(anchor_lines) >= 5:
                    break

        anchor_section = ""
        if anchor_lines:
            anchor_section = (
                f"KNOWN REFERENCES (these definitely refer to \"{comp_name}\"):\n"
                + "\n".join(anchor_lines)
                + "\n\n"
            )
        else:
            anchor_section = (
                f"NOTE: No standalone proper-case references to \"{comp_name}\" found "
                f"elsewhere in the document. This component may not be discussed "
                f"architecturally — be extra careful to verify each case.\n\n"
            )

        # ── Build cases with match context ──
        case_lines = []
        valid_seeds = []  # track which seeds have valid sentences
        for i, sl in enumerate(seeds):
            sent = sent_map.get(sl.sentence_number)
            if not sent:
                continue
            valid_seeds.append(sl)
            prev = sent_map.get(sl.sentence_number - 1)
            prev_text = f" [prev: \"{prev.text[:80]}\"]" if prev else ""

            match_ctx = classify_mention(comp_name, sent.text, doc_knowledge)
            case_lines.append(
                f"  Case {len(valid_seeds)} (S{sl.sentence_number}): \"{sent.text}\"{prev_text}\n"
                f"    Mention: {match_ctx}"
            )

        if not valid_seeds:
            continue

        # ── Build prompt ──
        prompt = f"""REFERENCE DISAMBIGUATION for component "{comp_name}"

COMPONENT PROFILE:
{profile}

{anchor_section}CASES TO VERIFY:
{chr(10).join(case_lines)}

{SEED_DISAMBIGUATION_RULES}

Return JSON:
{{"disambiguations": [{{"case": 1, "meaning": "component", "reason": "brief"}}]}}
JSON only:"""

        # ── Call LLM ──
        total_calls += 1
        response = llm.query(prompt, timeout=120)
        data = llm.extract_json(response)

        if not data:
            print(f"    [{comp_name}] LLM parse failed — keeping all {len(valid_seeds)} seeds")
            verified.extend(valid_seeds)
            continue

        results = {}
        for d in data.get("disambiguations", []):
            idx = d.get("case", 0) - 1
            results[idx] = d

        approved = 0
        rejected = 0
        for i, sl in enumerate(valid_seeds):
            r = results.get(i, {})
            meaning = (r.get("meaning", "component") or "component").lower().strip()
            if meaning == "other":
                reason = r.get("reason", "")
                print(f"    REJECT: S{sl.sentence_number} -> {comp_name} ({reason})")
                rejected += 1
            else:
                verified.append(sl)
                approved += 1

        print(f"    [{comp_name}] {approved} approved, {rejected} rejected")

    print(f"\n  Total LLM calls: {total_calls}")
    return verified


# ─────────────────────────────────────────────────────────────────────────────
# Main test driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.environ["CLAUDE_MODEL"] = "sonnet"
    llm = LLMClient()

    print("=" * 70)
    print("SEED REFERENCE DISAMBIGUATION TEST")
    print("=" * 70)

    results_table = []
    total_tp_killed = 0
    total_fp_caught = 0
    all_kills = []  # for detailed analysis

    for dataset in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset}")
        print(f"{'─' * 60}")

        # Load checkpoint data
        l1 = load_checkpoint(dataset, "layer1")
        l2 = load_checkpoint(dataset, "layer2")

        raw_seeds = l1["raw_seed_links"]
        model_knowledge = l1["model_knowledge"]
        doc_knowledge = l2["doc_knowledge"]  # enriched version

        # Load sentences and gold standard
        text_candidates = list((BENCHMARK / dataset).glob(f"text_*/{dataset}.txt"))
        assert text_candidates, f"No text file found for {dataset}"
        text_path = text_candidates[0]
        sentences = load_sentences(str(text_path))
        sent_map = build_sent_map(sentences)

        gold = load_gold_standard(dataset)

        # Classify raw seeds as TP/FP
        raw_tp = [(sl, True) for sl in raw_seeds if (sl.sentence_number, sl.component_id) in gold]
        raw_fp = [(sl, False) for sl in raw_seeds if (sl.sentence_number, sl.component_id) not in gold]

        print(f"  Raw seeds: {len(raw_seeds)} ({len(raw_tp)} TP, {len(raw_fp)} FP)")
        if raw_fp:
            for sl, _ in raw_fp:
                sent = sent_map.get(sl.sentence_number)
                print(f"    FP: S{sl.sentence_number} -> {sl.component_name}: \"{sent.text[:80] if sent else '?'}...\"")

        # Run disambiguation
        t0 = time.time()
        disambiguated = disambiguate_seeds(
            raw_seeds, model_knowledge, doc_knowledge, sent_map, llm
        )
        elapsed = time.time() - t0

        # Evaluate
        disambig_set = {(sl.sentence_number, sl.component_id) for sl in disambiguated}
        raw_set = {(sl.sentence_number, sl.component_id) for sl in raw_seeds}

        killed = raw_set - disambig_set
        tp_killed = []
        fp_caught = []
        for sl in raw_seeds:
            key = (sl.sentence_number, sl.component_id)
            if key in killed:
                if key in gold:
                    tp_killed.append(sl)
                    all_kills.append((dataset, sl, "TP_KILLED"))
                else:
                    fp_caught.append(sl)
                    all_kills.append((dataset, sl, "FP_CAUGHT"))

        dis_tp = len(raw_tp) - len(tp_killed)
        dis_fp = len(raw_fp) - len(fp_caught)

        total_tp_killed += len(tp_killed)
        total_fp_caught += len(fp_caught)

        results_table.append({
            "dataset": dataset,
            "raw": len(raw_seeds),
            "raw_tp": len(raw_tp),
            "raw_fp": len(raw_fp),
            "disambig": len(disambiguated),
            "dis_tp": dis_tp,
            "dis_fp": dis_fp,
            "killed_tp": len(tp_killed),
            "killed_fp": len(fp_caught),
            "time": elapsed,
        })

        if tp_killed:
            print(f"\n  ⚠ TP KILLED ({len(tp_killed)}):")
            for sl in tp_killed:
                sent = sent_map.get(sl.sentence_number)
                print(f"    S{sl.sentence_number} -> {sl.component_name}: \"{sent.text[:80] if sent else '?'}\"")

        if fp_caught:
            print(f"\n  ✓ FP CAUGHT ({len(fp_caught)}):")
            for sl in fp_caught:
                sent = sent_map.get(sl.sentence_number)
                print(f"    S{sl.sentence_number} -> {sl.component_name}: \"{sent.text[:80] if sent else '?'}\"")

        print(f"\n  Result: {len(disambiguated)}/{len(raw_seeds)} seeds kept "
              f"({dis_tp} TP, {dis_fp} FP) in {elapsed:.1f}s")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<15} {'Raw':>4} {'TP':>4} {'FP':>4} │ {'Dis':>4} {'TP':>4} {'FP':>4} │ {'K.TP':>5} {'K.FP':>5}")
    print("─" * 70)
    for r in results_table:
        print(f"{r['dataset']:<15} {r['raw']:>4} {r['raw_tp']:>4} {r['raw_fp']:>4} │ "
              f"{r['disambig']:>4} {r['dis_tp']:>4} {r['dis_fp']:>4} │ "
              f"{r['killed_tp']:>5} {r['killed_fp']:>5}")

    totals = {k: sum(r[k] for r in results_table)
              for k in ["raw", "raw_tp", "raw_fp", "disambig", "dis_tp", "dis_fp", "killed_tp", "killed_fp"]}
    print("─" * 70)
    print(f"{'TOTAL':<15} {totals['raw']:>4} {totals['raw_tp']:>4} {totals['raw_fp']:>4} │ "
          f"{totals['disambig']:>4} {totals['dis_tp']:>4} {totals['dis_fp']:>4} │ "
          f"{totals['killed_tp']:>5} {totals['killed_fp']:>5}")

    print(f"\n  TP preservation: {totals['dis_tp']}/{totals['raw_tp']} "
          f"({100*totals['dis_tp']/totals['raw_tp']:.1f}%)")
    print(f"  FP rejection:    {totals['killed_fp']}/{totals['raw_fp']} "
          f"({100*totals['killed_fp']/totals['raw_fp']:.1f}%)")
    print(f"  Kill ratio:      {totals['killed_tp']}:{totals['killed_fp']} TP:FP killed")

    if total_tp_killed == 0:
        print(f"\n  ✓ ZERO TP KILLS — safe to deploy")
    else:
        print(f"\n  ⚠ {total_tp_killed} TP kills — needs prompt tuning")


if __name__ == "__main__":
    main()
