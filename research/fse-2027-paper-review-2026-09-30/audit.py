#!/usr/bin/env python3
"""Read-only checks supporting this review; no model calls or manuscript edits."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import pickle
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "evaluation/mini-src"))
import metrics as m
import rq34 as r
import rq34_rq2 as c


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    result = {}
    inputs = set()
    big_path = ROOT / "evaluation/reports/RQ12_BIGTABLE.csv"
    inputs.add(big_path)
    big = rows(big_path)
    means = {row["system"]: row for row in big if row["run"] in {"average", "single"}}
    keys = ["doc_to_model_link_f1", "doc_to_model_link_f2", "doc_to_code_file_f1",
            "doc_to_code_file_f2", "doc_to_code_worst_component_f1",
            "doc_to_code_harmonic_component_f1"]
    result["headline_deltas_pp"] = {
        backend: {key: 100 * (float(means[f"approach (GPT-5.6-{backend})"][key]) -
                            float(means[f"Artemis (GPT-5.6-{backend})"][key]))
                  for key in keys} for backend in ["terra", "luna"]}
    result["gold_counts"] = {p: len(m.load_gs_sad_sam(p)) for p in m.PROJECTS}
    result["composition_limits"] = {}
    for project in m.PROJECTS:
        gold = m.enroll(m.load_gs_sad_code_raw(project), m.load_code_model_files(project))
        closure = c.compose_doc_code(project, r.load_gold(project))
        result["composition_limits"][project] = {
            "gold_doc_code": len(gold), "gold_doc_model_composed": len(closure),
            "gold_outside_composition": len(gold - closure),
            "composition_outside_gold": len(closure - gold),
            "doc_code_recall_of_perfect_doc_model": len(gold & closure)/len(gold)}

    r.install_unpickler()
    gate_rows = []
    example_decisions = []
    non_earlier_antecedents = []
    for backend in r.BACKENDS:
        for run in r.RUNS:
            for project in r.PROJECTS:
                slot = Path("/nonexistent")
                pdir = r._phase_dir(slot, run, backend, project)
                cell = r.compute_cell(slot, run, backend, project)
                restored_llm = set(cell.final)
                gate_rejected = set()
                tag_counts = collections.Counter()
                for phase in r.PHASES:
                    path = pdir / phase["file"]
                    inputs.add(path)
                    with path.open("rb") as stream:
                        state = pickle.load(stream)
                    if backend == "terra" and project == "mediastore" and phase["key"] == "name":
                        example_decisions.extend(dict(run=run, **decision) for decision in
                            state.get("feedback", {}).get("judge_decisions", [])
                            if int(decision["sentence"]) == 27)
                    if phase["key"] == "coref":
                        accepted = {r._key(link) for link in state["links"]}
                        for record in state.get("feedback", {}).get("metadata", []):
                            if record.get("antecedent_sentence", -1) >= record["sentence"]:
                                non_earlier_antecedents.append({"backend": backend, "run": run,
                                    "project": project, "sentence": record["sentence"],
                                    "antecedent_sentence": record["antecedent_sentence"],
                                    "accepted": (int(record["sentence"]), str(record["component_id"])) in accepted})
                    kept = {r._key(link) for link in state["links"]}
                    llm_rejected = set()
                    for decision in state.get("feedback", {}).get("judge_decisions", []):
                        if decision.get("approved"):
                            continue
                        key = (int(decision["sentence"]), str(decision["component_id"]))
                        tag = decision.get("path", decision.get("stage", ""))
                        if tag in {"antecedent_form_rejected", "name_ambiguous_discard"}:
                            gate_rejected.add(key)
                            tag_counts[tag] += 1
                        else:
                            llm_rejected.add(key)
                    restored_llm |= llm_rejected - kept
                published = r.rq3_variant_sets(cell)["NoValidator"]
                rejected_union = set().union(*cell.rejected.values())
                gate_rows.append({"backend": backend, "run": run, "project": project,
                    "gate_decision_counts": dict(tag_counts),
                    "gate_only_restored_links": len(published - restored_llm),
                    "gate_only_restored_tp": len((published - restored_llm) & cell.gold),
                    "gate_only_restored_fp": len((published - restored_llm) - cell.gold),
                    "full": list(r.dm_vector(cell.final, cell.gold)),
                    "published_no_judge": list(r.dm_vector(published, cell.gold)),
                    "llm_judges_off_gates_retained": list(r.dm_vector(restored_llm, cell.gold)),
                    "rejected_fp_still_in_final": len((rejected_union - cell.gold) & cell.final)})
    result["judge_audit_per_project_run"] = gate_rows
    result["mediastore_s27_decisions"] = example_decisions
    result["non_earlier_antecedents"] = non_earlier_antecedents
    result["judge_audit_means"] = {}
    for backend in r.BACKENDS:
        selected = [row for row in gate_rows if row["backend"] == backend]
        result["judge_audit_means"][backend] = {
            key: [statistics.mean(row[key][i] for row in selected) for i in range(5)]
            for key in ["full", "published_no_judge", "llm_judges_off_gates_retained"]}
        for key in ["gate_only_restored_links", "gate_only_restored_tp", "gate_only_restored_fp",
                    "rejected_fp_still_in_final"]:
            result["judge_audit_means"][backend][key + "_five_project_total_mean"] = sum(
                row[key] for row in selected) / len(r.RUNS)

    cost_path = ROOT / "evaluation/reports/INFERENCE_COST_PERRUN.csv"
    inputs.add(cost_path)
    cost = rows(cost_path)
    result["cost_models"] = {system: sorted({row["model"] for row in cost if row["system"] == system})
                             for system in ["approach", "Artemis"]}

    # File-level predictions outside any evaluated gold-reachable component do not
    # contribute false positives to the component metrics.
    exclusions = []
    for project in m.PROJECTS:
        files = m.load_code_model_files(project)
        gold = m.enroll(m.load_gs_sad_code_raw(project), files)
        membership = m.load_file_to_comps(project, files)
        gold_components = {c for _, f in gold for c in membership.get(f, ())}
        for run in r.RUNS:
            path = ROOT / f"sota-links/doc-code/aalinker-composed/terra_s126/{run}/{project}.csv"
            inputs.add(path)
            pred = m.load_result(path, "sad-code")
            excluded = {(s, f) for s, f in pred if not (set(membership.get(f, ())) & gold_components)}
            exclusions.append({"project": project, "run": run,
                "predictions_outside_evaluated_components": len(excluded),
                "fp_outside_evaluated_components": len(excluded - gold),
                "all_fp": len(pred - gold)})
    result["component_metric_exclusions"] = exclusions

    def gini(xs):
        return sum(abs(a-b) for a in xs for b in xs) / (2*len(xs)*sum(xs))
    examples = [[1, 1, 1, 1, 15.8/1.05], [0.05, 0.10, 0.80, 1, 4.1525/1.05]]
    result["gini_counterexamples"] = [{"values": xs, "gini": gini(xs),
        "largest_over_bottom_80pct": max(xs)/(sum(xs)-max(xs))} for xs in examples]
    result["harmonic_example"] = 5 / (1/0.2 + 4/0.9)
    result["weighted_arithmetic_example"] = sum(a*b for a,b in zip(
        [20,40,1500,2800,3900], [.2,.9,.9,.9,.9]))/8260
    result["cmr_counterexample"] = {"gold_assignments": 1000,
        "components": 10, "correct_links_one_per_component": 10,
        "CMR_percent": 0, "recall": .01}
    result["input_hashes"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(inputs)}
    (HERE / "evidence/audit-results.json").write_text(json.dumps(result, indent=2) + "\n")
    for key in ["headline_deltas_pp", "gold_counts", "judge_audit_means", "cost_models",
                "component_metric_exclusions", "gini_counterexamples", "harmonic_example"]:
        print(key + ": " + json.dumps(result[key], sort_keys=True))
    print("PASS: read-only review computations completed; this does not certify the manuscript claims.")


if __name__ == "__main__":
    main()
