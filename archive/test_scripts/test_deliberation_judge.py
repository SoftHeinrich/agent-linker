#!/usr/bin/env python3
"""Test deliberation-based TransArc judge (W24).

Loads REAL data from V24 logs + gold standards to test both strategies
on actual ambiguous-name TransArc links. Run each strategy separately.

Usage:
  python3 test_deliberation_judge.py advocate     # Strategy 1 only
  python3 test_deliberation_judge.py scratchpad   # Strategy 2 only
  python3 test_deliberation_judge.py              # Both (default)
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"
from llm_sad_sam.llm_client import LLMClient, LLMBackend

llm = LLMClient(backend=LLMBackend.CLAUDE)

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
LOG_DIR = Path("results/llm_logs")
RESULT_DIR = Path("results/ablation_results")

DATASETS = {
    "mediastore": {"text": "text_2016/mediastore.txt", "model_dir": "model_2016"},
    "teastore":   {"text": "text_2020/teastore.txt", "model_dir": "model_2020"},
    "teammates":  {"text": "text_2021/teammates.txt", "model_dir": "model_2021"},
    "bigbluebutton": {"text": "text_2021/bigbluebutton.txt", "model_dir": "model_2021"},
    "jabref":     {"text": "text_2021/jabref.txt", "model_dir": "model_2021"},
}


def load_sentences(ds_name):
    """Load sentences from text file. Returns {1: "text", 2: "text", ...}."""
    text_path = BENCHMARK_BASE / ds_name / DATASETS[ds_name]["text"]
    sents = {}
    with open(text_path) as f:
        for i, line in enumerate(f, 1):
            sents[i] = line.strip()
    return sents


def load_gold_standard(ds_name):
    """Load gold standard as set of (sentence_number, model_element_id)."""
    model_year = DATASETS[ds_name]["model_dir"].split("_")[1]
    text_year = DATASETS[ds_name]["text"].split("_")[0].replace("text", "").lstrip("_")
    if not text_year:
        text_year = DATASETS[ds_name]["text"].split("/")[0].split("_")[1]
    gs_dir = BENCHMARK_BASE / ds_name / "goldstandards"
    # Pattern: goldstandard_sad_YYYY-sam_YYYY.csv
    gs_file = None
    for f in gs_dir.glob("goldstandard_sad_*-sam_*.csv"):
        if "UME" not in f.name:
            gs_file = f
            break
    if not gs_file:
        raise FileNotFoundError(f"No SAD-SAM gold standard in {gs_dir}")
    gold = set()
    with open(gs_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Format: modelElementID,sentence
            eid = row.get("modelElementID", "")
            snum = int(row.get("sentence", 0))
            gold.add((snum, eid))
    return gold


def load_pipeline_log(ds_name):
    """Load the most recent pipeline log for a dataset. Tries V25, then V24 naming."""
    for prefix in ["v25", "v20d"]:
        logs = sorted(LOG_DIR.glob(f"{prefix}_{ds_name}_*.json"))
        if logs:
            with open(logs[-1]) as f:
                print(f"  Log: {logs[-1].name}")
                return json.load(f)
    return None


def load_components_from_pcm(ds_name):
    """Parse PCM to get component name -> id mapping."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from llm_sad_sam.pcm_parser import parse_pcm_repository
    model_dir = DATASETS[ds_name]["model_dir"]
    pcm_dir = BENCHMARK_BASE / ds_name / model_dir / "pcm"
    repo_files = list(pcm_dir.glob("*.repository"))
    if not repo_files:
        return {}, {}
    components = parse_pcm_repository(str(repo_files[0]))
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    return name_to_id, id_to_name


def get_ambiguous_names_from_log(log_data):
    """Extract ambiguous names from Phase 1 log entry."""
    for entry in log_data:
        if entry.get("phase") == "phase_1":
            return set(entry.get("out", {}).get("ambiguous", []))
    return set()


def get_transarc_links_from_log(log_data):
    """Extract TransArc links from Phase 4 log entry."""
    for entry in log_data:
        if entry.get("phase") == "phase_4":
            return entry.get("links", [])
    return []


def is_ambiguous_name(comp_name, ambiguous_names):
    """Check if component name would be reviewed by W24 deliberation."""
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    return comp_name in ambiguous_names


def build_test_cases():
    """Build test cases from real V24 data across all datasets."""
    all_cases = []

    for ds_name in DATASETS:
        print(f"\nLoading {ds_name}...")
        sentences = load_sentences(ds_name)
        gold = load_gold_standard(ds_name)
        log_data = load_pipeline_log(ds_name)
        name_to_id, id_to_name = load_components_from_pcm(ds_name)

        if not log_data:
            print(f"  No V24 log found for {ds_name}, skipping")
            continue

        ambiguous = get_ambiguous_names_from_log(log_data)
        ta_links = get_transarc_links_from_log(log_data)
        comp_names = sorted(name_to_id.keys())

        print(f"  Ambiguous: {sorted(ambiguous)}")
        print(f"  TransArc links: {len(ta_links)}")
        print(f"  Components: {comp_names}")

        # Filter to ambiguous-name TransArc links only
        reviewed = 0
        for lk in ta_links:
            comp = lk["c"]
            snum = lk["s"]
            if not is_ambiguous_name(comp, ambiguous):
                continue
            reviewed += 1

            comp_id = name_to_id.get(comp, "")
            is_tp = (snum, comp_id) in gold
            text = sentences.get(snum, "???")

            all_cases.append({
                "s": snum,
                "comp": comp,
                "text": text,
                "comps": comp_names,
                "dataset": ds_name,
                "gold": "TP" if is_tp else "FP",
            })

        tp_count = sum(1 for c in all_cases if c["dataset"] == ds_name and c["gold"] == "TP")
        fp_count = sum(1 for c in all_cases if c["dataset"] == ds_name and c["gold"] == "FP")
        print(f"  Ambiguous-name TransArc: {reviewed} reviewed ({tp_count} TP, {fp_count} FP)")

    return all_cases


# ═══════════════════════════════════════════════════════════════
# Strategy 1: Advocate-Prosecutor Deliberation
# ═══════════════════════════════════════════════════════════════

def run_advocate_prosecutor(case):
    """Run advocate-prosecutor-jury for one case with union voting."""
    snum = case["s"]
    comp = case["comp"]
    text = case["text"]
    comp_names_str = ', '.join(case["comps"])

    verdicts = []
    for pass_num in range(2):
        advocate_prompt = f"""You are the ADVOCATE for linking sentence S{snum} to component "{comp}".

CONTEXT: This link comes from a high-precision baseline (~90% accurate). The system has a
component literally named "{comp}". Most links are correct — your job is to defend valid ones.

SENTENCE: {text}
ALL COMPONENTS: {comp_names_str}

Your job: Find the STRONGEST evidence that this sentence discusses "{comp}" at the architectural level. Consider:
- Does the sentence describe {comp}'s role, behavior, interactions, or testing?
- Is "{comp.lower()}" used as a standalone noun/noun-phrase referring to a layer or part of the system?
- In architecture docs, even generic words like "the {comp.lower()} of the application" typically
  refer to the named component when such a component exists.

Provide your argument in 2-3 sentences. Then give your verdict.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

        prosecutor_prompt = f"""You are the PROSECUTOR arguing AGAINST linking sentence S{snum} to component "{comp}".

CONTEXT: This link comes from a baseline with ~90% precision. You should only argue REJECT
when there is CLEAR evidence the match is spurious — not just because the word has generic meanings.

SENTENCE: {text}
ALL COMPONENTS: {comp_names_str}

Your job: Find CLEAR evidence that this is a SPURIOUS match. Only these patterns warrant rejection:
1. "{comp.lower()}" is used as a modifier/adjective in a compound phrase (e.g., "cascade {comp.lower()}", "minimal {comp.lower()}") — NOT as a standalone noun referring to the component.
2. The sentence is primarily about a DIFFERENT component, and "{comp.lower()}" is purely incidental.
3. "{comp.lower()}" refers to a technology/protocol/tool, not the architecture component.
4. This is a package listing or dotted path (like x.foo.bar) where the match is coincidental.

If "{comp.lower()}" appears as a standalone noun describing a layer/part of the system, that is NOT spurious.

Provide your argument in 2-3 sentences. Then give your verdict.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

        adv_data = llm.extract_json(llm.query(advocate_prompt, timeout=60))
        pros_data = llm.extract_json(llm.query(prosecutor_prompt, timeout=60))

        adv_arg = adv_data.get("argument", "") if adv_data else ""
        adv_v = adv_data.get("verdict", "APPROVE").upper() if adv_data else "APPROVE"
        pros_arg = pros_data.get("argument", "") if pros_data else ""
        pros_v = pros_data.get("verdict", "REJECT").upper() if pros_data else "REJECT"

        print(f"    Pass {pass_num+1}: advocate={adv_v}, prosecutor={pros_v}")

        jury_prompt = f"""JURY: Decide if sentence S{snum} should be linked to component "{comp}".

CONTEXT: This link comes from a high-precision baseline (~90% accurate). The system has a component
literally named "{comp}". REJECT only with clear evidence — when in doubt, APPROVE.

SENTENCE: {text}

ADVOCATE argues: {adv_arg}

PROSECUTOR argues: {pros_arg}

Rule: APPROVE when "{comp.lower()}" is used as a standalone noun referring to a layer/part
of the system (its role, behavior, interactions, or testing). REJECT only when "{comp.lower()}"
is clearly used as a modifier in a compound phrase ("cascade {comp.lower()}"), a technology name,
or the sentence is entirely about a different component.

Return JSON: {{"verdict": "APPROVE" or "REJECT", "reason": "brief explanation"}}
JSON only:"""

        jury_data = llm.extract_json(llm.query(jury_prompt, timeout=60))
        v = True
        reason = ""
        if jury_data:
            v = jury_data.get("verdict", "APPROVE").upper() == "APPROVE"
            reason = jury_data.get("reason", "")
        verdicts.append(v)
        print(f"    Pass {pass_num+1} jury: {'APPROVE' if v else 'REJECT'} ({reason})")

    result = verdicts[0] or verdicts[1]  # Union
    if verdicts[0] != verdicts[1]:
        print(f"    Union: {'APPROVE (saved)' if result else 'REJECT'}")
    return result


# ═══════════════════════════════════════════════════════════════
# Strategy 2: Scratchpad Decomposition
# ═══════════════════════════════════════════════════════════════

def run_scratchpad(case):
    """Run scratchpad decomposition for one case with union voting."""
    snum = case["s"]
    comp = case["comp"]
    text = case["text"]
    comp_names_str = ', '.join(case["comps"])

    prompt = f"""Review: Should sentence S{snum} be linked to component "{comp}"?

CONTEXT: This link was produced by a high-precision baseline (~90% accurate). The system
has a component literally named "{comp}". Your job is to catch the ~10% of
false positives — only REJECT when there is strong evidence against the link.

SENTENCE: {text}
ALL COMPONENTS: {comp_names_str}

STEP 1 — OBSERVATION:
What is the grammatical role of "{comp.lower()}" in this sentence?
- Is it a standalone noun or noun phrase referring to a layer/part of the system?
- Or is it a modifier/adjective qualifying something else (e.g., "cascade X", "X pattern")?

STEP 2 — TWO INTERPRETATIONS:
A) COMPONENT REFERENCE: "{comp.lower()}" refers to the architectural component/layer "{comp}".
   Evidence includes: describes the component's role, behavior, data flow, interactions, or testing.
   In architecture docs, even generic-sounding words like "the X of the application" or "the X layer"
   typically refer to the named component when such a component exists.
B) GENERIC USAGE: "{comp.lower()}" is used purely as a common English word with NO
   connection to the component. Evidence: used as a modifier in a compound phrase ("cascade X",
   "minimal X"), refers to a technology/tool, or appears in a dotted package path.

STEP 3 — VERDICT: REJECT only if interpretation B is clearly stronger. When ambiguous, APPROVE.

Return JSON: {{"observation": "grammatical analysis", "interp_a": "evidence for component", "interp_b": "evidence for generic", "verdict": "APPROVE" or "REJECT", "reason": "brief"}}
JSON only:"""

    verdicts = []
    for pass_num in range(2):
        d = llm.extract_json(llm.query(prompt, timeout=60))
        v = True
        reason = ""
        if d:
            v = d.get("verdict", "APPROVE").upper() == "APPROVE"
            reason = d.get("reason", "")
        verdicts.append(v)
        obs = d.get("observation", "")[:60] if d else ""
        print(f"    Pass {pass_num+1}: {'APPROVE' if v else 'REJECT'} ({reason}) [{obs}]")

    result = verdicts[0] or verdicts[1]  # Union
    if verdicts[0] != verdicts[1]:
        print(f"    Union: {'APPROVE (saved)' if result else 'REJECT'}")
    return result


# ═══════════════════════════════════════════════════════════════
# Strategy 3: BATNA Negotiation (LLM-Deliberation inspired)
# ═══════════════════════════════════════════════════════════════

def run_batna(case):
    """Two-round negotiation: recall-agent vs precision-agent with BATNA threshold."""
    snum = case["s"]
    comp = case["comp"]
    text = case["text"]
    comp_names_str = ', '.join(case["comps"])

    # ── Round 1: Independent evaluation with private scratchpad ──
    recall_r1_prompt = f"""You are the RECALL AGENT. Your goal: avoid killing valid architectural links.
You are PENALIZED for rejecting links that are true positives. Be cautious about rejecting.

TASK: Should S{snum} be linked to component "{comp}"?

SENTENCE: {text}
ALL COMPONENTS: {comp_names_str}

SCRATCHPAD (think step by step):
1. What is "{comp.lower()}" doing in this sentence grammatically?
2. Could this plausibly refer to the architectural component "{comp}"?
3. What would be lost if we reject this link?

Your BATNA (minimum for rejection): You need OVERWHELMING evidence that
"{comp.lower()}" is NOT the architectural component to vote REJECT.

Return JSON: {{"scratchpad": "your reasoning", "argument": "your case in 2 sentences", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

    precision_r1_prompt = f"""You are the PRECISION AGENT. Your goal: avoid keeping spurious links.
You are PENALIZED for approving links that are false positives. Be skeptical.

TASK: Should S{snum} be linked to component "{comp}"?

SENTENCE: {text}
ALL COMPONENTS: {comp_names_str}

SCRATCHPAD (think step by step):
1. Is "{comp.lower()}" used as a common English word here (not the component)?
2. Is the sentence primarily about a DIFFERENT component?
3. Is "{comp.lower()}" a technology name, package path, or modifier?

Your BATNA (minimum for approval): You need CLEAR evidence that the sentence
discusses "{comp}" AS the architectural component to vote APPROVE.

Return JSON: {{"scratchpad": "your reasoning", "argument": "your case in 2 sentences", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

    recall_r1 = llm.extract_json(llm.query(recall_r1_prompt, timeout=60))
    precision_r1 = llm.extract_json(llm.query(precision_r1_prompt, timeout=60))

    recall_arg1 = recall_r1.get("argument", "") if recall_r1 else ""
    recall_v1 = recall_r1.get("verdict", "APPROVE").upper() if recall_r1 else "APPROVE"
    precision_arg1 = precision_r1.get("argument", "") if precision_r1 else ""
    precision_v1 = precision_r1.get("verdict", "REJECT").upper() if precision_r1 else "REJECT"

    print(f"    R1: recall={recall_v1}, precision={precision_v1}")

    # If both agree in Round 1, skip Round 2
    if recall_v1 == precision_v1:
        result = recall_v1 == "APPROVE"
        print(f"    R1 agreement: {'APPROVE' if result else 'REJECT'}")
        return result

    # ── Round 2: Each sees the other's argument and responds ──
    recall_r2_prompt = f"""You are the RECALL AGENT in Round 2. You previously argued:
"{recall_arg1}"

The PRECISION AGENT countered:
"{precision_arg1}"

SENTENCE: {text}
COMPONENT: {comp}

Consider the precision agent's points. Do they change your assessment?
Your BATNA: reject only with OVERWHELMING evidence this is NOT the component.

Return JSON: {{"response": "your counter-argument", "final_verdict": "APPROVE" or "REJECT"}}
JSON only:"""

    precision_r2_prompt = f"""You are the PRECISION AGENT in Round 2. You previously argued:
"{precision_arg1}"

The RECALL AGENT countered:
"{recall_arg1}"

SENTENCE: {text}
COMPONENT: {comp}

Consider the recall agent's points. Do they change your assessment?
Your BATNA: approve only with CLEAR evidence the sentence discusses this component.

Return JSON: {{"response": "your counter-argument", "final_verdict": "APPROVE" or "REJECT"}}
JSON only:"""

    recall_r2 = llm.extract_json(llm.query(recall_r2_prompt, timeout=60))
    precision_r2 = llm.extract_json(llm.query(precision_r2_prompt, timeout=60))

    recall_final = recall_r2.get("final_verdict", "APPROVE").upper() == "APPROVE" if recall_r2 else True
    precision_final = precision_r2.get("final_verdict", "REJECT").upper() == "APPROVE" if precision_r2 else False

    print(f"    R2: recall={'APPROVE' if recall_final else 'REJECT'}, "
          f"precision={'APPROVE' if precision_final else 'REJECT'}")

    # BATNA threshold: reject only if BOTH reject
    result = recall_final or precision_final
    if not result:
        print(f"    BATNA: both reject -> REJECT")
    return result


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def eval_results(cases, results, label):
    tp_kept = tp_killed = fp_kept = fp_killed = 0
    details = {"tp_killed": [], "fp_kept": []}
    for i, case in enumerate(cases):
        approved = results.get(i, True)
        if case["gold"] == "TP":
            if approved:
                tp_kept += 1
            else:
                tp_killed += 1
                details["tp_killed"].append(f"S{case['s']} -> {case['comp']} [{case['dataset']}]")
        else:
            if approved:
                fp_kept += 1
                details["fp_kept"].append(f"S{case['s']} -> {case['comp']} [{case['dataset']}]")
            else:
                fp_killed += 1

    total_tp = tp_kept + tp_killed
    total_fp = fp_kept + fp_killed

    print(f"\n{'='*60}")
    print(f"  {label} RESULTS")
    print(f"{'='*60}")
    print(f"  TP kept:   {tp_kept}/{total_tp} ({tp_kept/max(1,total_tp):.0%})")
    print(f"  TP killed: {tp_killed}/{total_tp} ({tp_killed/max(1,total_tp):.0%})  [target: 0%]")
    print(f"  FP killed: {fp_killed}/{total_fp} ({fp_killed/max(1,total_fp):.0%})  [target: >60%]")
    print(f"  FP kept:   {fp_kept}/{total_fp}")

    if details["tp_killed"]:
        print(f"\n  TP KILLS (BAD):")
        for d in details["tp_killed"]:
            print(f"    {d}")
    if details["fp_kept"]:
        print(f"\n  FP SURVIVORS (BAD):")
        for d in details["fp_kept"]:
            print(f"    {d}")

    # Per-dataset breakdown
    by_ds = defaultdict(lambda: {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0})
    for i, case in enumerate(cases):
        approved = results.get(i, True)
        ds = case["dataset"]
        if case["gold"] == "TP":
            if approved: by_ds[ds]["tp_kept"] += 1
            else: by_ds[ds]["tp_killed"] += 1
        else:
            if approved: by_ds[ds]["fp_kept"] += 1
            else: by_ds[ds]["fp_killed"] += 1

    print(f"\n  Per-dataset:")
    for ds in sorted(by_ds):
        d = by_ds[ds]
        ttp = d["tp_kept"] + d["tp_killed"]
        tfp = d["fp_kept"] + d["fp_killed"]
        print(f"    {ds:15s}: TP {d['tp_kept']}/{ttp}, FP killed {d['fp_killed']}/{tfp}")

    return {
        "tp_kept": tp_kept, "tp_killed": tp_killed,
        "fp_kept": fp_kept, "fp_killed": fp_killed,
        "label": label,
    }


def run_strategy(cases, strategy_name, strategy_fn):
    """Run one strategy on all cases."""
    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*60}")

    tp_cases = [c for c in cases if c["gold"] == "TP"]
    fp_cases = [c for c in cases if c["gold"] == "FP"]
    print(f"Cases: {len(tp_cases)} TP + {len(fp_cases)} FP = {len(cases)} total\n")

    results = {}
    t0 = time.time()
    for i, case in enumerate(cases):
        label = case["gold"]
        print(f"[{i+1}/{len(cases)}] {label} S{case['s']} -> {case['comp']} ({case['dataset']})")
        print(f"  \"{case['text'][:80]}{'...' if len(case['text']) > 80 else ''}\"")
        approved = strategy_fn(case)
        results[i] = approved
        tag = "OK" if (label == "TP" and approved) or (label == "FP" and not approved) else "WRONG"
        print(f"  -> {'APPROVED' if approved else 'REJECTED'} [{tag}]\n")

    elapsed = time.time() - t0
    stats = eval_results(cases, results, strategy_name)
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/max(1,len(cases)):.1f}s/case)")
    return stats


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Building test cases from pipeline logs + gold standards...")
    cases = build_test_cases()

    tp_count = sum(1 for c in cases if c["gold"] == "TP")
    fp_count = sum(1 for c in cases if c["gold"] == "FP")
    print(f"\nTotal: {tp_count} TP + {fp_count} FP = {len(cases)} cases")
    print(f"FP details:")
    for c in cases:
        if c["gold"] == "FP":
            print(f"  S{c['s']} -> {c['comp']} ({c['dataset']}): \"{c['text'][:60]}...\"")

    strategies = {
        "advocate": ("Advocate-Prosecutor", run_advocate_prosecutor),
        "scratchpad": ("Scratchpad", run_scratchpad),
        "batna": ("BATNA Negotiation", run_batna),
    }

    if strategy == "all":
        run_list = ["advocate", "scratchpad", "batna"]
    elif strategy == "both":
        run_list = ["advocate", "scratchpad"]
    else:
        run_list = [strategy]

    all_stats = []
    for s in run_list:
        if s in strategies:
            name, fn = strategies[s]
            stats = run_strategy(cases, name, fn)
            all_stats.append(stats)

    # Final summary
    if len(all_stats) > 1:
        print(f"\n\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for s in all_stats:
            ttp = s["tp_kept"] + s["tp_killed"]
            tfp = s["fp_kept"] + s["fp_killed"]
            print(f"  {s['label']:25s}: TP kill={s['tp_killed']}/{ttp} ({s['tp_killed']/max(1,ttp):.0%}), "
                  f"FP kill={s['fp_killed']}/{tfp} ({s['fp_killed']/max(1,tfp):.0%})")


if __name__ == "__main__":
    main()
