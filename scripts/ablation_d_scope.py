"""Ablation: combined D call vs per-project D calls.

Uses oracle JSONs from the probe run (already on disk).
No L/O re-runs. Compares pattern proposals from:
  A) Combined: all 3 oracle outputs fed to D at once (current behaviour)
  B) Per-project: D run once per project (3 calls, each sees 1 project)

Usage:
    python scripts/ablation_d_scope.py [--backend openai] [--model gpt-5.4]
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from voyager_train_tlr_v4_beta import (
    LLMBackend,
    LLMClient,
    _load_dotenv,
    _run_distillator_d,
    _filter_proposals,
    OUT_ROOT,
    SLOT_NAMES,
)

ORACLE_DIR = OUT_ROOT / "mainline"
PROJECTS = ["mediastore", "teastore", "teammates"]

def _empty_bank() -> dict:
    return {"version": "v4b", "slot_patterns": {s: [] for s in SLOT_NAMES}}

def _print_patterns(label: str, patterns: list[dict]) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}  ({len(patterns)} proposed)")
    print('─'*60)
    for i, p in enumerate(patterns, 1):
        slot = p.get("slot", "?")
        title = p.get("title", p.get("description", str(p))[:80])
        rationale = p.get("rationale", "")[:100]
        print(f"  {i}. [{slot}] {title}")
        if rationale:
            print(f"     ↳ {rationale}")

def run_ablation(backend: LLMBackend, model: str | None) -> None:
    _load_dotenv()
    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"

    llm = LLMClient(backend=backend, model=model)

    # Load oracle outputs
    o_jsons: dict[str, dict] = {}
    for p in PROJECTS:
        path = ORACLE_DIR / f"pass1_{p}_oracle.json"
        if not path.exists():
            print(f"ERROR: missing {path}")
            sys.exit(1)
        o_jsons[p] = json.loads(path.read_text())
        fms = o_jsons[p].get("failure_modes", [])
        evs = [fm["evidence_count"] for fm in fms]
        print(f"  Loaded {p}: {len(fms)} FMs, evidence={evs}")

    bank = _empty_bank()

    print("\n" + "="*60)
    print("  CONDITION A: Combined D (all 3 projects → 1 D call)")
    print("="*60)
    all_o = list(o_jsons.values())
    # iter_num=91 avoids cache collision with probe run (iter1) and per-project calls below
    d_combined = _run_distillator_d(
        llm=llm,
        o_jsons=all_o,
        bank=bank,
        iter_num=91,
        backend_str=backend_str,
        model_str=model_str,
        dry_run=False,
    )
    combined_proposed = d_combined.get("patterns_proposed", [])
    combined_accepted, combined_rejected = _filter_proposals(combined_proposed)
    _print_patterns("Combined proposed", combined_proposed)
    print(f"\n  → GATE-06 accepted: {len(combined_accepted)}, rejected: {len(combined_rejected)}")

    # Save
    out_a = ORACLE_DIR / "ablation_d_combined.json"
    out_a.write_text(json.dumps(d_combined, indent=2))
    print(f"  Saved → {out_a}")

    print("\n" + "="*60)
    print("  CONDITION B: Per-project D (3 separate D calls)")
    print("="*60)
    per_project_results: dict[str, dict] = {}
    # iter_nums 92/93/94 for per-project: unique to avoid cache collisions with probe (iter1) and combined (91)
    per_project_iter_nums = {"mediastore": 92, "teastore": 93, "teammates": 94}
    for proj in PROJECTS:
        print(f"\n  [D-{proj}] running...")
        d_proj = _run_distillator_d(
            llm=llm,
            o_jsons=[o_jsons[proj]],
            bank=bank,
            iter_num=per_project_iter_nums[proj],
            backend_str=backend_str,
            model_str=model_str,
            dry_run=False,
        )
        per_project_results[proj] = d_proj
        proposed = d_proj.get("patterns_proposed", [])
        accepted, rejected = _filter_proposals(proposed)
        _print_patterns(f"Per-project D ({proj})", proposed)
        print(f"  → GATE-06 accepted: {len(accepted)}, rejected: {len(rejected)}")

        out_b = ORACLE_DIR / f"ablation_d_per_{proj}.json"
        out_b.write_text(json.dumps(d_proj, indent=2))
        print(f"  Saved → {out_b}")

    # Merge per-project accepted patterns (dedup by title prefix)
    all_per_project: list[dict] = []
    seen_titles: set[str] = set()
    for proj in PROJECTS:
        proposed = per_project_results[proj].get("patterns_proposed", [])
        accepted, _ = _filter_proposals(proposed)
        for p in accepted:
            key = p.get("title", str(p))[:40].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                all_per_project.append(p)

    print("\n" + "="*60)
    print("  COMPARISON")
    print("="*60)
    print(f"  Combined D:      {len(combined_proposed)} proposed, {len(combined_accepted)} accepted after GATE-06")
    total_per_raw = sum(len(per_project_results[p].get("patterns_proposed",[])) for p in PROJECTS)
    print(f"  Per-project D:   {total_per_raw} proposed (across 3 calls), {len(all_per_project)} unique accepted after GATE-06 + dedup")

    print(f"\n  Combined accepted slots: {[p.get('slot','?') for p in combined_accepted]}")
    print(f"  Per-project accepted slots: {[p.get('slot','?') for p in all_per_project]}")

    # Check slot diversity
    combined_slots = set(p.get("slot","?") for p in combined_accepted)
    per_slots = set(p.get("slot","?") for p in all_per_project)
    print(f"\n  Combined unique slots: {len(combined_slots)} → {combined_slots}")
    print(f"  Per-project unique slots: {len(per_slots)} → {per_slots}")

    print("\n  Done. Review ablation_d_combined.json and ablation_d_per_*.json in results/voyager_v4_beta/mainline/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="openai", choices=["openai", "claude"])
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()
    backend = LLMBackend.OPENAI if args.backend == "openai" else LLMBackend.CLAUDE
    run_ablation(backend, args.model)
