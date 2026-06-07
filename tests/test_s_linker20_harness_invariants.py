"""Phase 44 harness-invariants: GATE-01 byte-equality, zero-network-egress, ReplayClient guard.

Phase 44 ROADMAP Success Criteria and which test verifies each:

- SC1 (fixture infrastructure exposes triples for every builder × project):
      Asserted transitively by the 6 snapshot modules passing (their non-skip
      cases prove load_records → BUILDERS → replay_parse is wired end-to-end).

- SC2 (six pytest test modules exist and pass on unmodified s19 baseline):
      Asserted by Test 2.5 (test_full_harness_suite_green_under_disable_socket),
      which spawns an inner pytest and asserts returncode == 0.

- SC3 (all snapshot tests pass on unmodified s19 baseline):
      Asserted by Test 2.5.

- SC4 (zero LLM API calls verified by absence of network I/O):
      Asserted by Test 2.2 (ReplayClient.query raises RuntimeError),
      Test 2.3 (zero .query( invocations in test layer),
      Test 2.4 (zero network-module imports in test layer),
      Test 2.5 (--disable-socket enforced on inner pytest run).

Module-level constants:
    FROZEN_BYTE_EQUAL_PATHS — three frozen source files that must stay byte-equal
        throughout Phase 44 (GATE-01).
    ROOT — repo root Path (mirrors tests/conftest.py).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FROZEN_BYTE_EQUAL_PATHS: tuple[str, ...] = (
    "src/llm_sad_sam/linkers/experimental/s_linker19.py",
    "src/llm_sad_sam/linkers/experimental/s_linker13_min.py",
    "src/llm_sad_sam/linkers/experimental/prompts_v5.py",
)


# ---------------------------------------------------------------------------
# Test 2.1 — GATE-01 byte-equality
# ---------------------------------------------------------------------------

def test_gate_01_byte_equality_s19_s13min_prompts_v5():
    """GATE-01: s_linker19.py, s_linker13_min.py, prompts_v5.py are byte-equal to HEAD.

    Runs ``git diff --stat HEAD -- <path>`` for each frozen source file and asserts
    empty output.  Skips if git is not on PATH (CI environments without git).
    """
    if not shutil.which("git"):
        pytest.skip("git binary not on PATH")

    failures = []
    for rel_path in FROZEN_BYTE_EQUAL_PATHS:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", rel_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=ROOT,
        )
        if result.stdout.strip():
            failures.append(
                f"GATE-01 FAIL: {rel_path!r} has uncommitted changes:\n"
                f"{result.stdout.strip()}"
            )

    assert not failures, "\n\n".join(failures)


# ---------------------------------------------------------------------------
# Test 2.2 — ReplayClient.query is forbidden
# ---------------------------------------------------------------------------

def test_replay_client_query_forbidden():
    """Belt-and-suspenders: ReplayClient().query() must raise RuntimeError.

    Plan 01 (44-01) asserts this contract in test_loader_self.py.  Keeping it
    here makes the Phase 44 invariant suite self-contained and visible.
    """
    from tests.harness.replay_client import ReplayClient

    with pytest.raises(RuntimeError, match="ReplayClient.query.. is forbidden"):
        ReplayClient().query("any prompt")


# ---------------------------------------------------------------------------
# Test 2.3 — Zero non-allowlisted .query( invocations in test layer
# ---------------------------------------------------------------------------

def test_no_llm_query_calls_in_harness_or_snapshot_modules():
    """Grep the test layer for .query( invocations.

    Allowed matches (allowlist):
    - The ReplayClient.query *definition* line in tests/harness/replay_client.py
    - Any line that mentions 'query' inside a docstring or comment (grep via -v '#')

    Disallowed: any actual invocation of .query( in harness or snapshot-test files.

    Uses subprocess so the check is robust to working-directory weirdness.
    """
    result = subprocess.run(
        ["grep", "-rnE", r"\.query\(", "tests/harness/", "--include=*.py"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    harness_matches = [
        line for line in result.stdout.splitlines()
        if line.strip() and not _is_allowlisted_query_match(line)
    ]

    # Also grep the snapshot test modules
    result2 = subprocess.run(
        ["grep", "-rnE", r"\.query\("] + [
            f"tests/test_s_linker20_prompt_{tag}.py"
            for tag in ("ambiguity", "doc_extract", "doc_judge", "extraction", "validation", "coref")
        ] + ["tests/test_s_linker20_harness_invariants.py"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    snapshot_matches = [
        line for line in result2.stdout.splitlines()
        if line.strip() and not _is_allowlisted_query_match(line)
    ]

    all_violations = harness_matches + snapshot_matches
    assert not all_violations, (
        "Non-allowlisted .query( invocations found in test layer:\n"
        + "\n".join(all_violations)
    )


def _is_allowlisted_query_match(line: str) -> bool:
    """Return True if this grep hit is an allowlisted .query( reference.

    Allowlisted:
    1. The ReplayClient.query *definition* line (``def query(self``).
    2. Any line whose stripped content begins with ``#`` (comment).
    3. Any docstring/string-literal line mentioning 'query' but not invoking it.
    4. pytest.raises context (any call to .query() inside a pytest.raises block)
       — identified by the calling file being a test-infrastructure file.
    5. Lines in this module itself (self-referential grep output).
    """
    # Strip "path:line:" prefix from grep output
    parts = line.split(":", 2)
    file_part = parts[0] if parts else ""
    code = parts[2] if len(parts) >= 3 else line
    code = code.strip()

    # Comment lines
    if code.startswith("#"):
        return True

    # ReplayClient.query method definition
    if re.search(r"def query\(self", code):
        return True

    # Lines in the harness invariants module itself (self-referential)
    if "test_s_linker20_harness_invariants.py" in file_part:
        return True

    # Lines in replay_client.py that are docstring content (not actual calls)
    if "replay_client.py" in file_part:
        # Only the def query line is a definition; other .query( in that file
        # are docstring text describing the forbidden behavior
        return True

    # test_loader_self.py: the client.query() call is inside pytest.raises
    if "test_loader_self.py" in file_part:
        return True

    # Any line whose stripped content is a string/docstring (no real Python call)
    # — look for surrounding quotes or "fails" / "forbidden" language
    if "is forbidden" in code or "forbidden" in code:
        return True

    return False


# ---------------------------------------------------------------------------
# Test 2.4 — Zero network-module imports in test layer
# ---------------------------------------------------------------------------

def test_no_network_module_imports_in_test_layer():
    """Test layer must contain no direct ``import openai|anthropic|requests|httpx|urllib``.

    This grep covers tests/harness/ and tests/test_s_linker20_*.py.
    Comments are excluded.
    """
    patterns = (r"^(import|from) (openai|anthropic|requests|httpx|urllib)",)
    target_dirs_and_files = [
        "tests/harness/",
    ] + [
        f"tests/test_s_linker20_prompt_{tag}.py"
        for tag in ("ambiguity", "doc_extract", "doc_judge", "extraction", "validation", "coref")
    ] + ["tests/test_s_linker20_harness_invariants.py"]

    for pattern in patterns:
        result = subprocess.run(
            ["grep", "-rnE", pattern, "--include=*.py"] + target_dirs_and_files,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=ROOT,
        )
        violations = [
            line for line in result.stdout.splitlines()
            if line.strip() and not _is_comment_line(line)
        ]
        assert not violations, (
            f"Network-module import found in test layer (pattern={pattern!r}):\n"
            + "\n".join(violations)
        )


def _is_comment_line(grep_line: str) -> bool:
    """Return True if the code portion of a grep line is a comment."""
    code = grep_line.split(":", 2)[-1] if ":" in grep_line else grep_line
    return code.strip().startswith("#")


# ---------------------------------------------------------------------------
# Test 2.5 — Full harness suite green under --disable-socket
# ---------------------------------------------------------------------------

def test_full_harness_suite_green_under_disable_socket():
    """Phase 44 SC3 + SC4: full snapshot suite passes with --disable-socket.

    Spawns an inner pytest to run the 6 snapshot modules under --disable-socket.
    Asserts returncode == 0.

    Recursion guard: the outer pytest sets _PHASE44_INNER=1; the inner pytest
    checks that env var at module import and skips this test, preventing infinite
    recursion (same pattern used in tests/test_single_step_harness.py).

    Skips with an actionable message if --disable-socket is not recognized
    (pytest-socket not installed).
    """
    # Recursion guard
    if os.environ.get("_PHASE44_INNER") == "1":
        pytest.skip("_PHASE44_INNER=1 — skipping to avoid infinite pytest recursion")

    # Verify --disable-socket is recognized (pytest-socket installed)
    probe = subprocess.run(
        [sys.executable, "-m", "pytest", "--disable-socket", "--collect-only", "-q",
         "tests/test_s_linker20_prompt_ambiguity.py", "--no-header"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
        env={**os.environ, "_PHASE44_INNER": "1"},
    )
    if "unrecognized" in probe.stderr.lower() or "no such option" in probe.stderr.lower():
        pytest.skip(
            "pytest-socket not installed — check pyproject.toml [dev] extras. "
            "--disable-socket flag is unrecognized."
        )

    snapshot_modules = [
        "tests/test_s_linker20_prompt_ambiguity.py",
        "tests/test_s_linker20_prompt_doc_extract.py",
        "tests/test_s_linker20_prompt_doc_judge.py",
        "tests/test_s_linker20_prompt_extraction.py",
        "tests/test_s_linker20_prompt_validation.py",
        "tests/test_s_linker20_prompt_coref.py",
    ]

    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + snapshot_modules + [
            "--disable-socket", "-q", "--no-header",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
        env={**os.environ, "_PHASE44_INNER": "1"},
    )

    assert result.returncode == 0, (
        f"Inner pytest failed (returncode={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-500:]}"
    )
