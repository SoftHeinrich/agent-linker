---
phase: 10-scaffolding
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - src/llm_sad_sam/linkers/experimental/helper_v3.py
autonomous: true
requirements:
  - CLEAN-02
user_setup: []

must_haves:
  truths:
    - "A new module src/llm_sad_sam/linkers/experimental/helper_v3.py exists and is importable"
    - "helper_v3 exposes the helper functions that s_linker13_clean will import in plan 10-03 — extracted by copying the bodies verbatim from s_linker13.py, NOT by changing semantics"
    - "v2.0 helper modules (data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py) remain byte-identical"
    - "s_linker13.py remains byte-identical (it still inlines the helpers — that is intentional; s_linker13 is frozen)"
    - "All exported helper names have stable identifiers documented in a module docstring table so plan 10-03 can import them verbatim"
  artifacts:
    - path: "src/llm_sad_sam/linkers/experimental/helper_v3.py"
      provides: "Versioned sibling carrying extracted helpers for the v2.1 _clean variant chain"
      exports: ["coerce_mention_type", "format_mention_string", "has_standalone_mention", "build_component_profile", "parse_snum", "get_comp_names"]
  key_links:
    - from: "src/llm_sad_sam/linkers/experimental/helper_v3.py"
      to: "llm_sad_sam.core.data_types_v2"
      via: "import of EvidenceBundle, SadSamLink, CandidateLink as needed"
      pattern: "from llm_sad_sam\\.core\\.data_types_v2 import"
    - from: "src/llm_sad_sam/linkers/experimental/helper_v3.py"
      to: "(future) s_linker13_clean.py"
      via: "module-level function exports consumed by Plan 10-03"
      pattern: "def (coerce_mention_type|format_mention_string|has_standalone_mention|build_component_profile|parse_snum|get_comp_names)"
---

<objective>
Extract the pure-function helpers currently inlined inside s_linker13.py into a new versioned
sibling module helper_v3.py, without touching any frozen v2.0 file.

Purpose: CLEAN-02 prerequisite. Plan 10-03 (s_linker13_clean) will import these helpers
instead of inlining them, so the _clean variant becomes a thin orchestration class. The
v2.0 helper modules (data_types_v2, document_loader_v2, pcm_parser_v2) stay byte-identical
per the frozen-file contract.

Output: src/llm_sad_sam/linkers/experimental/helper_v3.py — a single versioned sibling
module exporting the helpers s_linker13_clean will consume. Code is copied verbatim from
s_linker13.py (no semantic changes — semantics-changing trims are deferred to Phase 12).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/10-scaffolding/10-CONTEXT.md
@src/llm_sad_sam/linkers/experimental/s_linker13.py
@src/llm_sad_sam/core/data_types_v2.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create helper_v3.py with extracted pure helpers</name>
  <files>src/llm_sad_sam/linkers/experimental/helper_v3.py</files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/s_linker13.py (FULL file — read it entirely so the extracted bodies are verbatim copies; the helpers to extract are at the following anchor lines per the existing structure: _build_component_profile near line 594, _parse_snum near line 1108, _has_standalone_mention near line 1120, _get_comp_names near line 1150; also locate _format_mention_string if present)
    - src/llm_sad_sam/linkers/experimental/s_linker13d.py (for the _coerce_mention_type and _format_mention_string canonical bodies — these were introduced for VAR-04 / Spike 003; the s_linker13 chain may have them inlined too)
    - src/llm_sad_sam/core/data_types_v2.py (to confirm the EvidenceBundle / SadSamLink / CandidateLink dataclass surface helper_v3 must import from — DO NOT modify this file)
    - .planning/phases/10-scaffolding/10-CONTEXT.md (the four helper concern groupings listed in <specifics>: a/doc knowledge, b/coref/alias mentions, c/ambiguity, d/misc — the CONTEXT block explicitly permits a single helper_v3.py if one file is enough; this plan ships a single file)
    - tests/test_s_linker13d_parity.py (read it for the byte-identical contract on _format_mention_string — extracted version must produce the same outputs)
  </read_first>
  <behavior>
    - Behavior 1 (extraction fidelity): for each helper function copied from s_linker13.py, applying it to the same inputs must return the identical output it returns inside s_linker13. No reformatting beyond stripping leading underscores and the `self` parameter when extracting static-methods / methods with no instance state.
    - Behavior 2 (no semantic drift): the only acceptable code changes are (a) removing the `self` parameter from methods that never reference self, (b) converting nested closures of self-references into local computation, (c) adding explicit `from __future__ import annotations` and type hints where missing — never logic changes.
    - Behavior 3 (no benchmark leakage): GATE-06 compliance — extracted helper bodies and any added docstrings contain zero benchmark component names (Reencoding, FreeSWITCH, kurento, Redis PubSub, Recording Service, etc.).
    - Behavior 4 (importable surface): `from llm_sad_sam.linkers.experimental.helper_v3 import coerce_mention_type, format_mention_string, has_standalone_mention, build_component_profile, parse_snum, get_comp_names` succeeds at the Python prompt.
    - Behavior 5 (frozen files untouched): `git diff --stat src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py` reports zero changes after this plan runs.
  </behavior>
  <action>
    Create src/llm_sad_sam/linkers/experimental/helper_v3.py.

    Required module docstring (the first triple-string in the file) must contain:
      - "helper_v3 — extracted helpers for v2.1 _clean variant chain (Plan 10-02)"
      - A frozen-file declaration: "v2.0 helpers (data_types_v2, document_loader_v2, pcm_parser_v2) and s_linker13.py are NOT modified by this extraction."
      - An export table listing every public function name with its origin line in s_linker13.py and a one-sentence purpose.

    Required imports (top of file):
      - from __future__ import annotations
      - import re (only if any extracted helper uses regex — has_standalone_mention does)
      - from typing import Iterable, Sequence (only the typing names actually used)
      - from llm_sad_sam.core.data_types_v2 import ... (only the names actually used)

    Required public functions (extract verbatim from s_linker13.py / s_linker13d.py, drop leading underscore, drop `self` where unused):

      1. coerce_mention_type(value: str) -> str
         Source: s_linker13d.SLinker13d._coerce_mention_type
         Strict-coercion (raises ValueError on unknown enum) — D-21a contract. Do NOT relax to silent fallback.

      2. format_mention_string(mention_type: str, alias: str | None) -> str
         Source: s_linker13d.SLinker13d._format_mention_string
         Byte-identical output contract — the test_s_linker13d_parity test will be cloned in a later plan to point at helper_v3 too. Six expected outputs (see test EXPECTED dict).

      3. has_standalone_mention(comp_name: str, text: str) -> bool
         Source: s_linker13._has_standalone_mention near line 1120
         Spike 002 RISKY primitive — KEEP. Do NOT replace with LLM here (EXT-01 / EXT-02 deferred to v2.2+ per PROJECT.md Key Decisions).

      4. build_component_profile(comp_name: str) -> str
         Source: s_linker13._build_component_profile near line 594
         If the original references self (e.g. self.components, self.knowledge), refactor signature to receive those values as explicit parameters: build_component_profile(comp_name: str, components, knowledge) — record the change in the docstring's export table.

      5. parse_snum(val) -> int | None
         Source: s_linker13._parse_snum near line 1108
         Pure utility, trivial to extract.

      6. get_comp_names(components) -> list[str]
         Source: s_linker13._get_comp_names near line 1150
         Pure utility, trivial to extract.

    Additional helpers to scan-and-extract if pure:
      - Any private staticmethod on SLinker13 that does not reference `self` or `cls` AND is called from inside the link() main loop. Add it to the export table.
      - SKIP: any method that reads/writes self.* state (LLMClient, document knowledge, checkpoint dirs, logging) — these belong inside the linker class, not in helper_v3.

    Code style:
      - No fenced code blocks anywhere in the file (it's a Python module — fenced blocks would break it; mentioned only to forbid them in any embedded docstring example).
      - Match existing project style: 4-space indent, double-quote strings, type hints required on new signatures.
      - Add a module-level constant MENTION_TYPES = frozenset({"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"}) mirroring SLinker13d.MENTION_TYPES so coerce_mention_type can validate against it without depending on the class.

    Do NOT:
      - Edit s_linker13.py (frozen)
      - Edit s_linker13d.py (frozen — exists in tree)
      - Edit prompts_v2.py (frozen)
      - Edit any module under src/llm_sad_sam/core/ (frozen v2.0 helpers)
      - Add any benchmark-derived word lists, component names, or tailored example phrasing
      - Add any logic-changing trims (those belong in Phase 12 PROMPT-02)
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && python -c "from llm_sad_sam.linkers.experimental import helper_v3 as h; assert h.MENTION_TYPES == frozenset({'proper_case','lowercase','dotted_path','via_alias','indirect'}); assert h.coerce_mention_type('lowercase')=='lowercase'; assert h.format_mention_string('lowercase', None)=='lowercase mention'; assert h.format_mention_string('via_alias','Dispatcher')=='via known alias \"Dispatcher\"'; assert h.format_mention_string('via_alias', None)=='via known alias'; assert h.format_mention_string('proper_case', None)=='proper case, standalone'; assert h.format_mention_string('dotted_path', None)=='lowercase, inside dotted path'; assert h.format_mention_string('indirect', None)=='indirect/unclear match'; import inspect; src=inspect.getsource(h); leaks=[w for w in ['Reencoding','FreeSWITCH','kurento','Recording Service','Redis PubSub'] if w in src]; assert not leaks, leaks; print('helper_v3 OK')" && git diff --quiet -- src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py && echo "frozen files untouched"</automated>
  </verify>
  <acceptance_criteria>
    - File src/llm_sad_sam/linkers/experimental/helper_v3.py exists
    - `python -c "from llm_sad_sam.linkers.experimental import helper_v3"` exits 0
    - Module exports MENTION_TYPES frozenset matching the SLinker13d set
    - All six byte-identical format_mention_string outputs match the EXPECTED table from tests/test_s_linker13d_parity.py (proper_case→"proper case, standalone", lowercase→"lowercase mention", dotted_path→"lowercase, inside dotted path", via_alias+"Dispatcher"→'via known alias "Dispatcher"', via_alias+None→"via known alias", indirect→"indirect/unclear match")
    - coerce_mention_type("garbage_enum_value") raises ValueError matching /Unknown mention_type/
    - Module docstring contains the literal strings "helper_v3 — extracted helpers for v2.1 _clean variant chain (Plan 10-02)" and "v2.0 helpers" and "are NOT modified"
    - `git diff --stat src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py` shows 0 changes
    - `grep -E "Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server" src/llm_sad_sam/linkers/experimental/helper_v3.py` returns nothing (GATE-06 clean)
    - Module exports at least the six required public names: coerce_mention_type, format_mention_string, has_standalone_mention, build_component_profile, parse_snum, get_comp_names
  </acceptance_criteria>
  <done>
    helper_v3.py exists with the six required helpers extracted verbatim, all parity outputs match, frozen files unchanged, GATE-06 clean.
  </done>
</task>

</tasks>

<verification>
1. `python -c "from llm_sad_sam.linkers.experimental import helper_v3; print(dir(helper_v3))"` exits 0 and prints a dir() including the six required names
2. `git diff --quiet -- src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py src/llm_sad_sam/linkers/experimental/prompts_v2.py` exits 0
3. `grep -E "Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server" src/llm_sad_sam/linkers/experimental/helper_v3.py` exits 1 (no matches)
</verification>

<success_criteria>
- helper_v3.py importable, all byte-identical format_mention_string outputs match the s_linker13d EXPECTED table
- Frozen v2.0 files (s_linker13.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py, prompts_v2.py) unchanged
- Zero benchmark component name leakage (GATE-06 clean)
- Plan 10-03 can import the helpers without further refactoring
</success_criteria>

<output>
After completion, create `.planning/phases/10-scaffolding/10-02-SUMMARY.md` recording:
- Final list of public names exported from helper_v3
- For each, the original location in s_linker13.py (line range) and any signature change (e.g. self removed, parameters added)
- Confirmation `git diff` shows zero changes to frozen files
- Confirmation GATE-06 grep is clean
</output>
