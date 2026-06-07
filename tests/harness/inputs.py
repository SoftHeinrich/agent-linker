"""Inputs reconstruction for Phase 44 golden-replay snapshot harness.

Each per-builder reverse-extractor reads the prompt text from a _calls.json record
and reconstructs the original arguments that, when passed to the builder staticmethod,
produce a string byte-equal to ``record["prompt"]``.

This module's correctness is validated by the step-6 prompt-equality assertions in
the six ``test_s_linker20_prompt_*.py`` modules — if a reverse-extractor is wrong,
those tests fail loudly with a diff.

Public API
----------
reconstruct_inputs(builder_name, record, phase_tag) -> tuple
    Dispatch function called by every test module.  Returns the argument tuple
    to pass to ``BUILDERS[builder_name](*args)``.

Per-builder helpers (also importable directly for debugging):
    reconstruct_ambiguity_inputs(record) -> tuple[list[str]]
    reconstruct_doc_extract_inputs(record) -> tuple[list[str], list[str]]
    reconstruct_doc_judge_inputs(record) -> tuple[list[str], list[str]]
    reconstruct_extraction_inputs(record) -> tuple[list[str], list[str], list[Sentence]]
    reconstruct_validation_inputs(record, phase_tag) -> tuple[list[str], list[str], str]
    reconstruct_coref_inputs(record) -> tuple[list[str], list[dict]]

sys.path bootstrap: inherited from tests/conftest.py.
Do NOT modify sys.path in this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llm_sad_sam.core.data_types_v2 import ModelKnowledge, DocumentKnowledge
from llm_sad_sam.linkers.experimental.prompts_v5 import COREF_VALIDATION_FOCUS


# ---------------------------------------------------------------------------
# Sentence dataclass (used as the unit inside extraction / coref builders)
# ---------------------------------------------------------------------------

@dataclass
class Sentence:
    """Minimal Sentence representation used by prompt builders.

    Only ``.number`` and ``.text`` are used by the scaffolding; the full
    ``load_sentences`` dataclass may carry additional fields but the reverse-
    extractors only need these two.
    """
    number: int
    text: str


# ---------------------------------------------------------------------------
# Per-builder reverse-extractors
# ---------------------------------------------------------------------------

def reconstruct_ambiguity_inputs(record: dict) -> tuple[list[str]]:
    """Reconstruct ``(names,)`` from the phase_1_model prompt.

    The prompt prefix is::

        Classify these software architecture component names.\n\nNAMES: A, B, C

    Parse everything after ``NAMES: `` up to the first newline.
    """
    prompt: str = record["prompt"]
    m = re.search(r"^NAMES:\s+(.+)$", prompt, re.MULTILINE)
    if not m:
        raise ValueError(
            f"reconstruct_ambiguity_inputs: could not find 'NAMES: ...' line in prompt "
            f"(record phase={record.get('phase')!r})"
        )
    raw = m.group(1).strip()
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return (names,)


def reconstruct_doc_extract_inputs(record: dict) -> tuple[list[str], list[str]]:
    """Reconstruct ``(comp_names, doc_lines)`` from the phase_1_doc_extract prompt.

    Prompt structure::

        ...
        COMPONENTS: A, B, C
        ...
        DOCUMENT:\n<line1>\n<line2>\n...
        \nReturn JSON:

    Parse COMPONENTS line and DOCUMENT block (terminated by "\\nReturn JSON:" or end).
    """
    prompt: str = record["prompt"]

    # --- comp_names ---
    m = re.search(r"^COMPONENTS:\s+(.+)$", prompt, re.MULTILINE)
    if not m:
        raise ValueError(
            f"reconstruct_doc_extract_inputs: missing COMPONENTS line "
            f"(phase={record.get('phase')!r})"
        )
    comp_names = [n.strip() for n in m.group(1).split(",") if n.strip()]

    # --- doc_lines ---
    # The DOCUMENT: block is followed by raw sentence text (no S-number prefix)
    doc_start = prompt.find("\nDOCUMENT:\n")
    if doc_start == -1:
        raise ValueError(
            f"reconstruct_doc_extract_inputs: missing 'DOCUMENT:' block "
            f"(phase={record.get('phase')!r})"
        )
    doc_content_start = doc_start + len("\nDOCUMENT:\n")

    # The DOCUMENT block ends at "\n\nReturn JSON:" or end-of-string
    return_idx = prompt.find("\n\nReturn JSON:", doc_content_start)
    if return_idx != -1:
        doc_content = prompt[doc_content_start:return_idx]
    else:
        doc_content = prompt[doc_content_start:]

    doc_lines = [ln for ln in doc_content.split("\n") if ln]
    return (comp_names, doc_lines)


def reconstruct_doc_judge_inputs(record: dict) -> tuple[list[str], list[str]]:
    """Reconstruct ``(comp_names, mapping_list)`` from the phase_1_doc_judge prompt.

    Prompt structure::

        JUDGE: ...
        COMPONENTS: A, B, C
        PROPOSED MAPPINGS:
        'term1' -> Component1
        'term2' -> Component2
        ...
        <blank>
        <judge examples>

    Parse COMPONENTS line and PROPOSED MAPPINGS block (terminated by blank line).
    """
    prompt: str = record["prompt"]

    # --- comp_names ---
    m = re.search(r"^COMPONENTS:\s+(.+)$", prompt, re.MULTILINE)
    if not m:
        raise ValueError(
            f"reconstruct_doc_judge_inputs: missing COMPONENTS line "
            f"(phase={record.get('phase')!r})"
        )
    comp_names = [n.strip() for n in m.group(1).split(",") if n.strip()]

    # --- mapping_list ---
    # Find "PROPOSED MAPPINGS:" block, lines up to the first blank line
    mappings_start = prompt.find("\nPROPOSED MAPPINGS:\n")
    if mappings_start == -1:
        raise ValueError(
            f"reconstruct_doc_judge_inputs: missing 'PROPOSED MAPPINGS:' block "
            f"(phase={record.get('phase')!r})"
        )
    lines_start = mappings_start + len("\nPROPOSED MAPPINGS:\n")
    remaining = prompt[lines_start:]
    # Mappings end at first blank line
    blank_idx = remaining.find("\n\n")
    if blank_idx != -1:
        mappings_block = remaining[:blank_idx]
    else:
        mappings_block = remaining
    mapping_list = [ln for ln in mappings_block.split("\n") if ln.strip()]
    return (comp_names, mapping_list)


def reconstruct_extraction_inputs(
    record: dict,
) -> tuple[list[str], list[str], list[Sentence]]:
    """Reconstruct ``(comp_names, mappings, batch)`` from a phase_2_framing_c_pass* prompt.

    Prompt structure::

        Extract ALL references ...
        COMPONENTS: A, B, C
        KNOWN ALIASES: alias1=Comp1, alias2=Comp2    (optional)
        ...
        DOCUMENT:
        S1: sentence text
        S2: sentence text
        ...
        Return JSON:

    Mappings line is optional (empty when doc_knowledge has no global aliases).
    """
    prompt: str = record["prompt"]

    # --- comp_names ---
    m = re.search(r"^COMPONENTS:\s+(.+)$", prompt, re.MULTILINE)
    if not m:
        raise ValueError(
            f"reconstruct_extraction_inputs: missing COMPONENTS line "
            f"(phase={record.get('phase')!r})"
        )
    comp_names = [n.strip() for n in m.group(1).split(",") if n.strip()]

    # --- mappings (optional) ---
    mappings: list[str] = []
    m_aliases = re.search(r"^KNOWN ALIASES:\s+(.+)$", prompt, re.MULTILINE)
    if m_aliases:
        raw_aliases = m_aliases.group(1).strip()
        # Split on ", " (comma-space) — each alias is "alias=Component"
        mappings = [a.strip() for a in raw_aliases.split(", ") if a.strip()]

    # --- batch (Sentence objects) ---
    doc_start = prompt.find("\nDOCUMENT:\n")
    if doc_start == -1:
        raise ValueError(
            f"reconstruct_extraction_inputs: missing 'DOCUMENT:' block "
            f"(phase={record.get('phase')!r})"
        )
    doc_content_start = doc_start + len("\nDOCUMENT:\n")
    return_idx = prompt.find("\n\nReturn JSON:", doc_content_start)
    if return_idx != -1:
        doc_content = prompt[doc_content_start:return_idx]
    else:
        doc_content = prompt[doc_content_start:]

    # Each line is "S{N}: {text}"
    batch: list[Sentence] = []
    for line in doc_content.split("\n"):
        if not line.strip():
            continue
        m_sent = re.match(r"^S(\d+):\s+(.*)$", line)
        if m_sent:
            batch.append(Sentence(number=int(m_sent.group(1)), text=m_sent.group(2)))
        # Lines that don't match S{N}: format are ignored (shouldn't happen normally)

    return (comp_names, mappings, batch)


def reconstruct_validation_inputs(
    record: dict,
    phase_tag: str,
) -> tuple[list[str], list[str], str]:
    """Reconstruct ``(comp_names, cases, focus)`` from a validation-phase prompt.

    Handles all three validation phase tags:
    - ``phase_4_twopass_p1``  → focus = P1_FOCUS
    - ``phase_4_twopass_p2``  → focus = P2_FOCUS
    - ``phase_5_coref_validation`` → focus = COREF_VALIDATION_FOCUS

    The focus string is the prefix of the prompt up to the first blank line after the
    first sentence; we reconstruct it by reading it directly from the prompt rather than
    importing the constant (that would be circular with the step-6 byte-equality check).
    Actually, we reconstruct ``focus`` by reverse-parsing the first line of the prompt::

        "Validate component references in a software architecture document. {focus}\n"

    The rest of the prompt follows with COMPONENTS, VALIDATION_RULES, then CASES.

    Prompt structure::

        Validate component references ... {focus}

        COMPONENTS: A, B, C

        {VALIDATION_RULES}

        CASES:
        Case 1: ...
        Case 2: ...

        Return JSON:
    """
    prompt: str = record["prompt"]

    # --- focus: everything on the first line after the fixed prefix ---
    first_line = prompt.split("\n")[0]
    fixed_prefix = "Validate component references in a software architecture document."
    if not first_line.startswith(fixed_prefix):
        raise ValueError(
            f"reconstruct_validation_inputs: unexpected first line: {first_line[:80]!r} "
            f"(phase={phase_tag!r})"
        )
    focus = first_line[len(fixed_prefix):].strip()

    # --- comp_names ---
    m = re.search(r"^COMPONENTS:\s+(.+)$", prompt, re.MULTILINE)
    if not m:
        raise ValueError(
            f"reconstruct_validation_inputs: missing COMPONENTS line "
            f"(phase={phase_tag!r})"
        )
    comp_names = [n.strip() for n in m.group(1).split(",") if n.strip()]

    # --- cases: the CASES: block ---
    cases_start = prompt.find("\nCASES:\n")
    if cases_start == -1:
        raise ValueError(
            f"reconstruct_validation_inputs: missing 'CASES:' block "
            f"(phase={phase_tag!r})"
        )
    cases_content_start = cases_start + len("\nCASES:\n")
    return_idx = prompt.find("\n\nReturn JSON:", cases_content_start)
    if return_idx != -1:
        cases_block = prompt[cases_content_start:return_idx]
    else:
        cases_block = prompt[cases_content_start:]

    # Split on "Case N:" boundaries, but keep them as unit strings (the builder
    # receives a list of case strings, each one being a Case N: block)
    # The builder just does chr(10).join(cases) so each element is one case block.
    case_segments: list[str] = []
    current_lines: list[str] = []
    for line in cases_block.split("\n"):
        # Detect start of a new case: "Case N: ..."
        if re.match(r"^Case \d+:", line):
            if current_lines:
                # flush previous case (strip trailing blank lines)
                while current_lines and not current_lines[-1].strip():
                    current_lines.pop()
                case_segments.append("\n".join(current_lines))
            current_lines = [line]
        else:
            current_lines.append(line)
    # flush last case
    if current_lines:
        while current_lines and not current_lines[-1].strip():
            current_lines.pop()
        block = "\n".join(current_lines)
        if block.strip():
            case_segments.append(block)

    return (comp_names, case_segments, focus)


def reconstruct_coref_inputs(record: dict) -> tuple[list[str], list[dict]]:
    """Reconstruct ``(comp_names, cases)`` from a phase_5_coref prompt.

    The prompt structure is::

        Resolve anaphoric references ... (fixed preamble)

        COMPONENTS: A, B, C

        For each TARGET sentence ... (middle paragraph)

        --- Case 1: S{N1} ---
        CONTEXT:
        >>> S{N1}: {target sentence text}
            S{N2}: {context sentence text}
            ...
        TARGET: S{N1} (marked with >>>)

        --- Case 2: S{N2} ---
        ...

        {COREF_RULES}

        {ANTECEDENT_ALIAS_RULES}

        Return JSON:

    The builder ``_prompt_coref(comp_names, cases)`` expects::

        cases: list of dicts with:
            sent: Sentence(number=N, text="...")  -- the TARGET sentence
            context: list[str]                   -- the context lines (with >>> marker)

    We reconstruct by parsing each ``--- Case N: S{X} ---`` block.
    """
    prompt: str = record["prompt"]

    # --- comp_names ---
    m = re.search(r"^COMPONENTS:\s+(.+)$", prompt, re.MULTILINE)
    if not m:
        raise ValueError(
            f"reconstruct_coref_inputs: missing COMPONENTS line "
            f"(phase={record.get('phase')!r})"
        )
    comp_names = [n.strip() for n in m.group(1).split(",") if n.strip()]

    # --- cases ---
    # Locate each "--- Case N: S{X} ---" block
    case_header_pattern = re.compile(r"^--- Case \d+: S(\d+) ---$", re.MULTILINE)
    headers = list(case_header_pattern.finditer(prompt))

    if not headers:
        # No cases (shouldn't happen for a valid coref record)
        return (comp_names, [])

    # Terminal marker: COREF_RULES starts right after the last case block
    # We find the end of the last case block by looking for the first line that
    # starts with a known terminal pattern (blank line followed by the rules text)
    # Safer: use the start of the first non-case block after the last case header.
    terminal_markers = [
        "\nAre there any",      # part of ANTECEDENT_ALIAS_RULES pattern
        "\nFor each anaphoric", # part of COREF_RULES
        "\nReturn JSON:",
        "\n\nJSON only:",
    ]

    cases: list[dict] = []
    for idx, header_m in enumerate(headers):
        header_end = header_m.end()
        target_snum = int(header_m.group(1))

        # Determine block end: either next header start or terminal
        if idx + 1 < len(headers):
            block_end = headers[idx + 1].start()
        else:
            # Last case: find the terminal
            block_end = len(prompt)
            for marker in terminal_markers:
                ti = prompt.find(marker, header_end)
                if ti != -1 and ti < block_end:
                    block_end = ti

        block = prompt[header_end:block_end]

        # Extract CONTEXT lines and TARGET line
        # Block structure:
        #   \nCONTEXT:\n
        #   >>> S{N}: text   (target)
        #       S{M}: text   (context)
        #   TARGET: S{N} (marked with >>>)\n
        context_start = block.find("\nCONTEXT:\n")
        if context_start == -1:
            # Malformed block — skip
            continue
        context_content_start = context_start + len("\nCONTEXT:\n")

        target_line_start = block.find("\nTARGET: S", context_content_start)
        if target_line_start != -1:
            context_raw = block[context_content_start:target_line_start]
        else:
            context_raw = block[context_content_start:]

        # context lines: each starts with ">>> " (target) or "    " (context)
        context_lines: list[str] = []
        target_text: str = ""
        for line in context_raw.split("\n"):
            if not line.strip():
                continue
            context_lines.append(line)
            # The target sentence is marked with ">>>"
            m_target = re.match(r"^>>>\s+S(\d+):\s+(.*)$", line)
            if m_target and int(m_target.group(1)) == target_snum:
                target_text = m_target.group(2)

        cases.append({
            "sent": Sentence(number=target_snum, text=target_text),
            "context": context_lines,
        })

    return (comp_names, cases)


# ---------------------------------------------------------------------------
# Dispatch function — called by all 6 test modules
# ---------------------------------------------------------------------------

def reconstruct_inputs(
    builder_name: str,
    record: dict,
    phase_tag: str,
) -> tuple:
    """Reconstruct builder arguments from a _calls.json record.

    Args:
        builder_name: one of the 6 BUILDERS keys (e.g. "_prompt_ambiguity")
        record:       one element from load_records(project, phase_tag)
        phase_tag:    the phase tag the record came from (e.g. "phase_1_model")

    Returns:
        A tuple of arguments to pass to ``BUILDERS[builder_name](*args)``.
        The returned args, when passed to the builder, must produce a string
        byte-equal to ``record["prompt"]`` (asserted by the step-6 prompt-equality
        check in each test module).

    Raises:
        ValueError: if builder_name is not recognised.
    """
    dispatch: dict[str, Any] = {
        "_prompt_ambiguity": lambda: reconstruct_ambiguity_inputs(record),
        "_prompt_doc_knowledge_extract": lambda: reconstruct_doc_extract_inputs(record),
        "_prompt_doc_knowledge_judge": lambda: reconstruct_doc_judge_inputs(record),
        "_prompt_extraction": lambda: reconstruct_extraction_inputs(record),
        "_prompt_validation": lambda: reconstruct_validation_inputs(record, phase_tag),
        "_prompt_coref": lambda: reconstruct_coref_inputs(record),
    }
    if builder_name not in dispatch:
        raise ValueError(
            f"reconstruct_inputs: unknown builder_name={builder_name!r}. "
            f"Valid names: {sorted(dispatch)}"
        )
    return dispatch[builder_name]()
