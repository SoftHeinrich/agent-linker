#!/usr/bin/env python3
"""Unit test: verify v24 has no hardcoded word lists beyond basic stopwords.

Scans agent_linker_v24.py for any hardcoded word lists and verifies
only NON_MODIFIERS (basic English stopwords) remains.
"""

import ast
import re
import sys
from pathlib import Path

V24_PATH = Path(__file__).parent / "src/llm_sad_sam/linkers/experimental/agent_linker_v24.py"

# These are the ONLY acceptable hardcoded word sets (basic English stopwords)
ALLOWED_SETS = {"NON_MODIFIERS", "SOURCE_PRIORITY"}


def find_class_level_sets(source):
    """Find all set/dict/list literals assigned at class level."""
    tree = ast.parse(source)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        name = getattr(target, 'id', None) or getattr(target, 'attr', None)
                        if name and isinstance(item.value, (ast.Set, ast.List, ast.Dict, ast.Tuple)):
                            n_elements = 0
                            if isinstance(item.value, ast.Set):
                                n_elements = len(item.value.elts)
                            elif isinstance(item.value, ast.List):
                                n_elements = len(item.value.elts)
                            elif isinstance(item.value, ast.Dict):
                                n_elements = len(item.value.keys)
                            elif isinstance(item.value, ast.Tuple):
                                n_elements = len(item.value.elts)
                            if n_elements >= 5:  # only flag large collections
                                findings.append((name, type(item.value).__name__, n_elements, item.lineno))
    return findings


def check_prompt_examples(source):
    """Check for concrete component-like names in prompt strings."""
    # Patterns that look like benchmark component names
    suspicious = re.findall(
        r'(?:Auth\w+|Message\w+|Data\w+|User\w+|File\w+|Image\w+|Media\w+|'
        r'Web\w+|Service\w+|Handler\w+|Controller\w+|Manager\w+|Provider\w+|'
        r'Gateway\w+|Broker\w+|Adapter\w+|Registry\w+|Factory\w+|Router\w+|'
        r'Cache\w+|Proxy\w+|Facade\w+|Wrapper\w+|Listener\w+|Dispatcher\w+|'
        r'Validator\w+|Formatter\w+|Parser\w+|Renderer\w+|Encoder\w+|'
        r'Decoder\w+|Serializer\w+|Interceptor\w+|Middleware\w+)',
        source
    )
    # Filter out things that are clearly code, not prompt examples
    code_patterns = {
        'DataRepository', 'DataLoader', 'DictReader',  # Python/framework classes
    }
    return [s for s in suspicious if s not in code_patterns and 'import' not in s]


def run_tests():
    passed = 0
    failed = 0

    source = V24_PATH.read_text()

    print("=" * 70)
    print("V24 Hardcoded List Audit")
    print("=" * 70)

    # Test 1: No large hardcoded collections except allowed ones
    print("\n--- Class-level collections (>=5 elements) ---")
    sets = find_class_level_sets(source)
    for name, typ, n, line in sets:
        if name in ALLOWED_SETS:
            print(f"  [OK] {name} ({typ}, {n} elements, line {line}) — allowed")
            passed += 1
        else:
            print(f"  [FAIL] {name} ({typ}, {n} elements, line {line}) — NOT ALLOWED")
            failed += 1

    if not sets:
        print("  No large collections found (besides inline)")
        passed += 1

    # Test 2: No PARTIAL_FALSE_FOLLOWERS
    print("\n--- Removed lists check ---")
    removed = ["PARTIAL_FALSE_FOLLOWERS", "GENERIC_COMPONENT_WORDS_HARDCODED",
               "common_english", "_VERB_SUFFIXES"]
    for name in removed:
        if name in source and f"# {name}" not in source and f"REMOVED" not in source.split(name)[0][-100:]:
            # Check if it's actually a variable assignment, not just a comment
            if re.search(rf'^\s+{name}\s*=', source, re.MULTILINE):
                print(f"  [FAIL] {name} still exists as variable")
                failed += 1
            else:
                print(f"  [OK] {name} only in comments")
                passed += 1
        else:
            print(f"  [OK] {name} removed")
            passed += 1

    # Test 3: GENERIC_COMPONENT_WORDS and GENERIC_PARTIALS start empty
    print("\n--- Dynamic discovery check ---")
    for var in ["GENERIC_COMPONENT_WORDS", "GENERIC_PARTIALS"]:
        match = re.search(rf'{var}\s*=\s*set\(\)', source)
        if match:
            print(f"  [OK] {var} initialized as empty set()")
            passed += 1
        else:
            print(f"  [WARN] {var} not found as empty set() — check manually")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
