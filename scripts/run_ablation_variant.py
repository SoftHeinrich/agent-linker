"""Single-variant runner: called by ablation_validation_rules.py as subprocess.
Usage: python run_ablation_variant.py <variant_name>
Reads VALIDATION_RULES from current axiom file, runs all 5 projects, prints JSON.
"""
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# import training harness helpers
import runpy
vtrain_globals = runpy.run_path(
    str(_ROOT / "scripts" / "voyager_train_tlr_v5.py"),
    run_name="__ablation_runner__",  # prevents if __name__ == "__main__" block
)
import types
vtrain = types.SimpleNamespace(**vtrain_globals)

from llm_sad_sam.linkers.experimental.s_linker14_voyager import LLMBackend

ALL_PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

empty_bank = {"version": "v5", "slot_patterns": {s: [] for s in vtrain.SLOT_NAMES}}
results = {}
variant_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
print(f"[variant={variant_name}] running all 5 projects with empty bank", file=sys.stderr)
for project in ALL_PROJECTS:
    print(f"  {project}...", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    r = vtrain._run_linker_l(project, LLMBackend.OPENAI, "gpt-5.4", empty_bank, dry_run=False)
    elapsed = time.time() - t0
    results[project] = {"F1": round(r["F1"], 6), "FP": r["fp_count"], "FN": r["fn_count"]}
    print(f"F1={r['F1']:.4f} FP={r['fp_count']} FN={r['fn_count']} ({elapsed:.0f}s)", file=sys.stderr)

macro = sum(v["F1"] for v in results.values()) / len(results)
print(f"  macro={macro:.4f}", file=sys.stderr)
# Write JSON to file (not stdout — linker prints to stdout, contaminating it)
out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/ablation_{variant_name}.json"
Path(out_path).write_text(json.dumps(results))
print(f"  wrote: {out_path}", file=sys.stderr)
