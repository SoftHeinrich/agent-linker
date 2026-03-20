#!/usr/bin/env python3
"""Test partial validation variants on BBB from checkpoint.

Variant A: Soften P1 focus for alias cases (actor role → "mentioned or involved")
Variant B: Union instead of intersection for alias cases

Both load tier1+tier1_5 checkpoints, reconstruct partial candidates,
run ONLY the validation step. Zero full-pipeline cost.
"""
import copy, csv, os, sys, pickle, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.data_types import CandidateLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.linkers.experimental.prompts import VALIDATION_RULES

BENCH = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
DS = "bigbluebutton"

text_path = str(BENCH / DS / "text_2021" / f"{DS}.txt")
model_path = str(BENCH / DS / "model_2021" / "pcm" / "bbb.repository")
gold_path = str(BENCH / DS / "goldstandards" / "goldstandard_sad_2021-sam_2021.csv")

# Load gold standard
gold = set()
with open(gold_path) as f:
    for row in csv.DictReader(f):
        gold.add((int(row["sentence"]), row["modelElementID"]))

# Load data
loader = DocumentLoader()
sentences = loader.load_sentences(text_path)
components = parse_pcm_repository(model_path)
name_to_id = {c.name: c.id for c in components}
sent_map = {s.number: s for s in sentences}
comp_names = sorted(set(c.name for c in components))

# Load checkpoints
with open(f"results/phase_cache/s_linker9d/{DS}/tier1.pkl", "rb") as f:
    tier1 = pickle.load(f)
with open(f"results/phase_cache/s_linker9d/{DS}/tier1_5.pkl", "rb") as f:
    tier1_5 = pickle.load(f)

doc_knowledge = tier1_5.get("doc_knowledge") or tier1.get("doc_knowledge")

print(f"Partial references: {doc_knowledge.partial_references}")

# Build alias map
alias_map = {}
for c in components:
    aliases = {}
    if doc_knowledge:
        for a, cn in doc_knowledge.abbreviations.items():
            if cn == c.name:
                aliases[a] = "abbreviation"
        for s, cn in doc_knowledge.synonyms.items():
            if cn == c.name:
                aliases[s] = "synonym"
        for p, cn in doc_knowledge.partial_references.items():
            if cn == c.name:
                aliases[p] = "partial reference"
    alias_map[c.name] = aliases

# Reconstruct partial candidates using linker's own method
from llm_sad_sam.linkers.experimental.s_linker9d import SLinker9d
_tmp_linker = SLinker9d(backend=LLMBackend.CLAUDE)
_tmp_linker.doc_knowledge = doc_knowledge

partial_candidates = _tmp_linker._inject_partial_candidates(
    sentences, components, name_to_id, sent_map,
    seed_set=set(), validated_set=set(), coref_set=set()
)

print(f"Total partial candidates: {len(partial_candidates)}")
tp_total = sum(1 for c in partial_candidates if (c.sentence_number, c.component_id) in gold)
fp_total = len(partial_candidates) - tp_total
print(f"  Ground truth: {tp_total} TP, {fp_total} FP")
print()

# Build cases with alias hints (shared by both variants)
def build_cases(candidates):
    cases = []
    for i, c in enumerate(candidates):
        prev = sent_map.get(c.sentence_number - 1)
        p = f"[prev: {prev.text[:35]}...] " if prev else ""
        alias_hint = ""
        matched_lower = c.matched_text.lower() if c.matched_text else ""
        if matched_lower and matched_lower != c.component_name.lower():
            aliases = alias_map.get(c.component_name, {})
            for alias, atype in aliases.items():
                if alias.lower() in matched_lower or matched_lower in alias.lower():
                    alias_hint = f'\n  [KNOWN ALIAS: "{alias}" is a known {atype} for "{c.component_name}"]'
                    break
        cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}{alias_hint}\n  {p}"{c.sentence_text}"')
    return cases

llm = LLMClient(backend=LLMBackend.CLAUDE)

alias_rule = ("\n- When a KNOWN ALIAS is indicated, the word IS a reference to that component "
              "unless the sentence clearly uses it in an unrelated sense")

def run_validation_pass(cases, focus):
    prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{VALIDATION_RULES}{alias_rule}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""
    data = llm.extract_json(llm.query(prompt, timeout=120))
    results = {}
    if data:
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(cases):
                results[idx] = v.get("approve", False)
    return results

def score(validated, candidates, label):
    tp = fp = fn = 0
    validated_set = {(c.sentence_number, c.component_id) for c in validated}
    candidate_set = {(c.sentence_number, c.component_id) for c in candidates}
    for c in validated:
        if (c.sentence_number, c.component_id) in gold:
            tp += 1
        else:
            fp += 1
    fn = len(gold & candidate_set - validated_set)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    print(f"  {label}: TP={tp} FP={fp} FN={fn} | P={p:.1%} R={r:.1%} F1={f1:.1%}")
    return tp, fp, fn

cases = build_cases(partial_candidates)

# ======================================================================
# Variant A: Soften P1 focus for alias cases
# ======================================================================
print("=" * 60)
print("VARIANT A: Soften P1 (actor -> mentioned/involved)")
print("=" * 60)

r1a = run_validation_pass(cases,
    "Focus on INVOLVEMENT: is the component mentioned, involved in, or relevant to what the sentence describes?")
r2a = run_validation_pass(cases,
    "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

validated_a = []
for i, c in enumerate(partial_candidates):
    p1 = r1a.get(i, False)
    p2 = r2a.get(i, False)
    if p1 and p2:
        cc = copy.copy(c)
        cc.confidence = 1.0
        cc.source = "validated"
        validated_a.append(cc)
    elif p1 != p2:
        print(f"    Split: S{c.sentence_number} \"{c.matched_text}\"->{c.component_name} P1={'Y' if p1 else 'N'} P2={'Y' if p2 else 'N'}")

score(validated_a, partial_candidates, "Variant A")
print()

# ======================================================================
# Variant B: Union for alias cases (approve if EITHER pass says yes)
# ======================================================================
print("=" * 60)
print("VARIANT B: Union (approve if either pass approves)")
print("=" * 60)

r1b = run_validation_pass(cases,
    "Focus on ACTOR role: is the component performing an action or being described?")
r2b = run_validation_pass(cases,
    "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

validated_b = []
for i, c in enumerate(partial_candidates):
    p1 = r1b.get(i, False)
    p2 = r2b.get(i, False)
    if p1 or p2:
        cc = copy.copy(c)
        cc.confidence = 1.0
        cc.source = "validated"
        validated_b.append(cc)

score(validated_b, partial_candidates, "Variant B")
print()

# ======================================================================
# Summary
# ======================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Candidates: {len(partial_candidates)} ({tp_total} TP, {fp_total} FP)")
score(validated_a, partial_candidates, "A (soft P1 + intersect)")
score(validated_b, partial_candidates, "B (orig prompts + union)")
