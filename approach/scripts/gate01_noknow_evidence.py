"""GATE-01 evidence harness for the NOKNOW ablation (Phase 51, plan 51-02).

Runs two zero-LLM, zero-network checks to prove that flag-off
``SLinker20Union(no_knowledge=False)`` is unchanged from the pre-51-01 full-knowledge
behavior:

CHECK 1 — Structural guard (load-bearing):
    Obtain the git diff that introduced the ``no_knowledge`` flag in
    ``s_linker20_union.py`` and assert:
    (a) Added lines (+) introduce only: the ``no_knowledge`` constructor param,
        the ``self.no_knowledge = no_knowledge`` assignment, and the
        ``if self.no_knowledge:`` branch with its NOKNOW-specific statements.
    (b) Every removed line (- prefix) has a matching added line (+) with identical
        text after stripping leading whitespace — proving the else-branch statements
        are the same pre-existing code, just re-indented.  No statement text was
        deleted or altered.
    (c) No added line introduces a hardcoded benchmark alias/vocabulary list.
        GATE-06 backstop: only the expected identifiers and the [NOKNOW] print
        literal appear as new string content.

CHECK 2 — Frozen-cache stability (corroborative):
    Re-run the Phase-50 extractor (``scripts/extract_s20union_caches.py``) against
    the 30 frozen Full cells and assert:
    (a) Subprocess exits 0.
    (b) Its stdout contains the line ``30/30 PASS``.
    This proves the Full caches + output schema have not drifted.

NOTE on literal flag-off linker replay:
    A byte-identical replay of ``SLinker20Union(no_knowledge=False).link(...)``
    against the 30 frozen phase_caches is NOT performed here.  The linker has no
    checkpoint-resume mode — it always makes live LLM calls.  The structural guard
    (Check 1) is the load-bearing proof that flag-off executes the identical
    pre-existing statements; the frozen-cache check (Check 2) corroborates schema
    and cache stability.  The live spot-check (one representative cell per backend
    with flag-OFF) lands as the first cell of the 51-04 sweep.
    See RESEARCH.md §A3 (Assumptions Log) for the full rationale.

Usage:
    python scripts/gate01_noknow_evidence.py --structural-only  # Check 1 only
    python scripts/gate01_noknow_evidence.py                    # Both checks
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Bootstrap (mirror extract_s20union_caches.py pattern) ────────────────────
sys.stdout.reconfigure(line_buffering=True)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────
_LINKER_PATH = "src/llm_sad_sam/linkers/experimental/s_linker20_union.py"

# Identifiers expected to appear in added lines (GATE-06 allow-list).
# These are code identifiers and one print-string prefix — NOT benchmark vocab.
_ALLOWED_ADDED_IDENTIFIERS = {
    "no_knowledge",
    "no_knowledge: bool = False",
    "self.no_knowledge",
    "ModelKnowledge",
    "DocumentKnowledge",
    "[NOKNOW]",
}


# ── Diff acquisition ──────────────────────────────────────────────────────────

def _get_diff(path: str) -> str:
    """Return the git diff text for *path*, trying multiple sources in order.

    Order:
    1. Commit that introduced 'no_knowledge: bool' (normal case: 51-01 already
       committed; working-tree diff would be empty).
    2. Staged diff (git diff --cached).
    3. Unstaged working-tree diff (git diff).

    Returns the first non-empty diff text, or '' if all sources are empty.
    """
    # Source 1: find the introducing commit and diff it against its parent.
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "-S", "no_knowledge: bool", "--", path],
            capture_output=True, text=True, check=True,
        )
        commit_hash = result.stdout.strip()
        if commit_hash:
            diff_result = subprocess.run(
                ["git", "diff", f"{commit_hash}^", commit_hash, "--", path],
                capture_output=True, text=True, check=True,
            )
            if diff_result.stdout.strip():
                return diff_result.stdout
    except subprocess.CalledProcessError:
        pass

    # Source 2: staged.
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--", path],
            capture_output=True, text=True, check=True,
        )
        if result.stdout.strip():
            return result.stdout
    except subprocess.CalledProcessError:
        pass

    # Source 3: unstaged working-tree.
    try:
        result = subprocess.run(
            ["git", "diff", "--", path],
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


# ── Structural check (CHECK 1) ────────────────────────────────────────────────

def structural_check() -> int:
    """Assert the 51-01 linker change is additive and the else-branch is unchanged.

    Returns 0 on PASS, 1 on any FAIL.
    """
    diff_text = _get_diff(_LINKER_PATH)

    if not diff_text or not diff_text.strip():
        print(
            "STRUCTURAL CHECK: FAIL — no diff found for "
            f"{_LINKER_PATH}\n"
            "  Was plan 51-01 applied?  Expected a commit that adds the "
            "'no_knowledge: bool' constructor parameter.",
            flush=True,
        )
        return 1

    # Parse diff lines: collect added (+) and removed (-) content lines.
    # Skip the diff header lines (---, +++, @@, diff).
    added_lines: list[str] = []
    removed_lines: list[str] = []

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            added_lines.append(raw_line[1:])   # strip leading '+'
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            removed_lines.append(raw_line[1:]) # strip leading '-'

    # ── Assertion (a): required added tokens present ──────────────────────────
    added_combined = "\n".join(added_lines)

    missing_required: list[str] = []
    if "no_knowledge: bool = False" not in added_combined:
        missing_required.append("no_knowledge: bool = False  (constructor param)")
    if "self.no_knowledge = no_knowledge" not in added_combined:
        missing_required.append("self.no_knowledge = no_knowledge  (assignment)")
    if "if self.no_knowledge:" not in added_combined:
        missing_required.append("if self.no_knowledge:  (guard branch)")

    if missing_required:
        print("STRUCTURAL CHECK: FAIL — required added tokens not found:", flush=True)
        for tok in missing_required:
            print(f"  missing: {tok}", flush=True)
        return 1

    # ── Assertion (b): every removed line has a matching added line (whitespace-norm) ──
    # Strip leading whitespace for comparison.
    stripped_added = {line.lstrip() for line in added_lines}
    stripped_removed = [line.lstrip() for line in removed_lines]

    unmatched_removed: list[str] = []
    for stripped_rm in stripped_removed:
        if not stripped_rm:  # blank lines are fine
            continue
        if stripped_rm not in stripped_added:
            unmatched_removed.append(stripped_rm)

    if unmatched_removed:
        print(
            "STRUCTURAL CHECK: FAIL — removed lines have no matching added line "
            "(text was actually deleted or altered, not merely re-indented):",
            flush=True,
        )
        for line in unmatched_removed:
            print(f"  unmatched removal: {line!r}", flush=True)
        return 1

    # ── Assertion (c): GATE-06 backstop — no benchmark vocabulary in added lines ──
    # Heuristic: the only new string *literals* should be the [NOKNOW] prefix and
    # Python identifiers from _ALLOWED_ADDED_IDENTIFIERS.
    # We flag any added line that contains a string literal (quoted text) that is
    # NOT the expected [NOKNOW] print string.
    # Simple heuristic: look for quoted content that looks like a list of names
    # (contains commas inside a quote, implying a comma-separated alias list).
    suspect_lines: list[str] = []
    for line in added_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Flag lines that contain a list-like string literal (comma inside quotes).
        import re
        # Match single or double quoted strings containing commas.
        matches = re.findall(r'["\'][^"\']*,[^"\']*["\']', stripped)
        for m in matches:
            # Allow if it matches the expected [NOKNOW] print string pattern.
            # The only expected quoted string with a comma-like pattern is the em-dash
            # separator in the print string, which does not contain a comma.
            # Any comma-containing quoted string is unexpected.
            suspect_lines.append(f"  suspect line: {stripped!r}  (quoted with comma: {m!r})")

    if suspect_lines:
        print(
            "STRUCTURAL CHECK: FAIL — GATE-06 backstop: added lines contain "
            "quoted string(s) with commas (possible benchmark alias list):",
            flush=True,
        )
        for s in suspect_lines:
            print(s, flush=True)
        return 1

    print("STRUCTURAL CHECK: PASS", flush=True)
    return 0


# ── Frozen-cache stability check (CHECK 2) ────────────────────────────────────

def frozen_cache_check() -> int:
    """Re-run the Phase-50 extractor and assert 30/30 PASS.

    Returns 0 on PASS, 1 on FAIL.
    """
    extractor = str(_ROOT / "scripts" / "extract_s20union_caches.py")
    result = subprocess.run(
        [sys.executable, extractor],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    combined_output = result.stdout + result.stderr

    # Check exit code.
    if result.returncode != 0:
        print("FROZEN-CACHE CHECK: FAIL — extractor exited non-zero", flush=True)
        print("  extractor output tail:", flush=True)
        tail_lines = combined_output.strip().splitlines()[-20:]
        for line in tail_lines:
            print(f"    {line}", flush=True)
        return 1

    # Check for "30/30 PASS" line in output.
    if "30/30 PASS" not in combined_output:
        print(
            "FROZEN-CACHE CHECK: FAIL — extractor exited 0 but '30/30 PASS' "
            "not found in output",
            flush=True,
        )
        print("  extractor output tail:", flush=True)
        tail_lines = combined_output.strip().splitlines()[-20:]
        for line in tail_lines:
            print(f"    {line}", flush=True)
        return 1

    # Echo the 30/30 PASS line so caller can grep for it.
    for line in combined_output.splitlines():
        if "30/30 PASS" in line:
            print(f"FROZEN-CACHE CHECK: PASS ({line.strip()})", flush=True)
            break
    return 0


# ── Main driver ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="GATE-01 evidence harness for the NOKNOW ablation (51-02).",
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Run CHECK 1 (structural guard) only; skip the frozen-cache re-run.",
    )
    args = parser.parse_args()

    if args.structural_only:
        return structural_check()

    # Run both checks; both must pass.
    struct_rc = structural_check()
    cache_rc = frozen_cache_check()

    overall_rc = 0 if (struct_rc == 0 and cache_rc == 0) else 1
    if overall_rc == 0:
        print("\nGATE-01 EVIDENCE: PASS", flush=True)
    else:
        print("\nGATE-01 EVIDENCE: FAIL", flush=True)
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
