"""Ablation: isolate effects of two VALIDATION_RULES edits.

Variants tested (vs baseline already known):
  A - "or activity" only  (no "including counterparts")
  B - "including counterparts" only  (no "or activity")
  C - both  (already cached from axiom_edit_test run)

Strategy: patch axiom file on disk, launch subprocess (fresh module imports),
restore file. Sequential to avoid file conflicts.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent

AXIOM_FILE = _ROOT / "src" / "llm_sad_sam" / "linkers" / "experimental" / "prompts_v4_axiom.py"

VARIANTS = {
    "A_activity_only": (
        'VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant. '
        'Reject when the matching word is generic, names a different entity, or describes a technique or activity that merely shares the component\'s name."""'
    ),
    "B_counterparts_only": (
        'VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant, including counterparts. '
        'Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component\'s name."""'
    ),
}

BASELINE_KNOWN = {
    "mediastore":    {"F1": 0.8852, "FP": 13, "FN": 5},
    "teastore":      {"F1": 0.9474, "FP": 4,  "FN": 3},
    "teammates":     {"F1": 0.8160, "FP": 17, "FN": 6},
    "bigbluebutton": {"F1": 0.7288, "FP": 13, "FN": 19},
    "jabref":        {"F1": 0.9730, "FP": 1,  "FN": 0},
}
BOTH_KNOWN = {
    "mediastore":    {"F1": 0.9355, "FP": 2,  "FN": 2},
    "teastore":      {"F1": 0.9474, "FP": 3,  "FN": 0},
    "teammates":     {"F1": 0.7840, "FP": 19, "FN": 8},
    "bigbluebutton": {"F1": 0.7603, "FP": 13, "FN": 16},
    "jabref":        {"F1": 0.9474, "FP": 2,  "FN": 0},
}

ALL_PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]


def patch_axiom(new_validation_line: str) -> str:
    orig = AXIOM_FILE.read_text()
    patched = re.sub(
        r'^VALIDATION_RULES\s*=\s*""".*?"""',
        new_validation_line,
        orig,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert patched != orig, "Patch had no effect — regex didn't match"
    AXIOM_FILE.write_text(patched)
    return orig


def run_variant(name: str, new_line: str) -> dict:
    import tempfile
    print(f"\n[Ablation] Variant {name}")
    print(f"  Rule: {new_line[len('VALIDATION_RULES = '):][:90]}...")
    orig = patch_axiom(new_line)
    out_file = Path(tempfile.mktemp(suffix=f"_{name}.json"))
    try:
        # stderr streams live to terminal; results written to out_file (linker pollutes stdout)
        proc = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "run_ablation_variant.py"), name, str(out_file)],
            cwd=str(_ROOT),
            stderr=None,
            text=True,
            timeout=1200,
        )
        if proc.returncode != 0:
            print(f"  ERROR: subprocess exited {proc.returncode}")
            return {}
        if not out_file.exists():
            print(f"  ERROR: output file not created: {out_file}")
            return {}
        return json.loads(out_file.read_text())
    finally:
        AXIOM_FILE.write_text(orig)
        out_file.unlink(missing_ok=True)


def print_table(all_results: dict):
    projects = ALL_PROJECTS
    header_a = "A: +activity"
    header_b = "B: +counterp"
    print("\n" + "=" * 100)
    print(f"{'Project':<16} | {'Baseline':^22} | {header_a:^22} | {header_b:^22} | {'C: both':^22}")
    print("-" * 100)
    for p in projects:
        b = BASELINE_KNOWN[p]
        c = BOTH_KNOWN[p]

        def fmt(d, ref_f1):
            if not d:
                return f"{'N/A':^22}"
            delta = d["F1"] - ref_f1
            return f"{d['F1']:.4f}({delta:+.4f}) {d['FP']:2d}/{d['FN']:2d}"

        a = all_results.get("A_activity_only", {}).get(p, {})
        bv = all_results.get("B_counterparts_only", {}).get(p, {})
        b_fmt = f"{b['F1']:.4f}         {b['FP']:2d}/{b['FN']:2d}"
        c_fmt = fmt(c, b["F1"])
        print(f"{p:<16} | {b_fmt:^22} | {fmt(a, b['F1']):^22} | {fmt(bv, b['F1']):^22} | {c_fmt:^22}")

    print("-" * 100)
    macros = {}
    for tag, data in [("baseline", BASELINE_KNOWN), ("C_both", BOTH_KNOWN)]:
        macros[tag] = sum(v["F1"] for v in data.values()) / len(data)
    for vname in ["A_activity_only", "B_counterparts_only"]:
        vdata = all_results.get(vname, {})
        if vdata:
            macros[vname] = sum(v["F1"] for v in vdata.values()) / len(vdata)

    print(f"{'Macro':16} | {macros['baseline']:.4f}              | ", end="")
    for vn in ["A_activity_only", "B_counterparts_only"]:
        m = macros.get(vn, float("nan"))
        d = m - macros["baseline"]
        print(f"{m:.4f}({d:+.4f})         | ", end="")
    m = macros["C_both"]
    d = m - macros["baseline"]
    print(f"{m:.4f}({d:+.4f})        ")
    print("=" * 100)


def main():
    orig_content = AXIOM_FILE.read_text()
    all_results = {}

    try:
        for variant_name, new_line in VARIANTS.items():
            results = run_variant(variant_name, new_line)
            all_results[variant_name] = results
            if results:
                macro = sum(v["F1"] for v in results.values()) / len(results)
                print(f"  Macro: {macro:.4f}")
    finally:
        AXIOM_FILE.write_text(orig_content)
        print("\nAxiom file restored.")

    print_table(all_results)

    out_path = _ROOT / "results" / "voyager_v5" / "ablation_validation_rules.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "baseline": BASELINE_KNOWN,
        "C_both": BOTH_KNOWN,
        **all_results,
    }
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
