"""ILinker2 V33 — V32 + Pareto-safe cross-model prompt fixes.

Changes vs V32 (Pareto-tested: helps GPT, doesn't hurt Sonnet):

- Fix 1 (Phase 6): Pure-LLM generic mention handling. Disables the code-level
  _is_generic_mention filter entirely. Instead, the validation prompt teaches the
  LLM to distinguish architectural references from package-path/qualified-name
  contexts. Pareto test: +9 TP, +1 FP on Sonnet (the 1 FP is caught downstream).

- Fix 2 (Phase 9): Judge Rule 4 explicit exception for named components with
  generic names. If the system has a component literally named "X", architectural
  references to X ARE component-specific usage. Zero delta on Sonnet (it already
  gets this right), but helps GPT which interprets Rule 4 too literally.

Excluded (hurt Sonnet in Pareto test):
- Fix 3 (validation prompt exception): redundant with Fix 1
- Fix 4 (structural coref rules): -4 net TP on teammates

Checkpoint dir: v33.
"""

import os

from llm_sad_sam.linkers.experimental.ilinker2_v32 import ILinker2V32


class ILinker2V33(ILinker2V32):
    """V32 + Pareto-safe cross-model fixes (generic mention + judge Rule 4)."""

    # ── Checkpoint dir override ───────────────────────────────────────

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "v33", ds)
        os.makedirs(d, exist_ok=True)
        return d

    # ── Fix 1: Pure-LLM generic mention handling ──────────────────────

    def _is_generic_mention(self, comp_name, sentence_text):
        """Disabled — let the LLM validation prompt decide."""
        return False

    def _qual_validation_pass(self, comp_names, ctx, cases, focus):
        """V32 validation prompt + package-path awareness rule."""
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DECISION RULES:
APPROVE when:
- The component is the grammatical actor or subject (the sentence is ABOUT the component)
- A section heading names the component (introduces that component's topic)
- The sentence describes what the component does, provides, or interacts with

REJECT when:
- The name is used as an ordinary English word, not as a proper name
  (Like "proxy" in "proxy pattern" is the design pattern concept, not the Proxy component — reject the component link)
- The name is a modifier inside a larger phrase, not a standalone reference
  (Like "observer" in "observer pattern" modifies pattern — reject if Observer is a component)
- The sentence is about a subprocess, algorithm, or implementation detail — not the component itself

SINGLE-WORD COMPONENT NAMES (important for names like single common English words):
When the system has a component with a single-word name that is also an ordinary English
word, apply these rules:
- APPROVE when the word is used as a NOUN referring to a part of the system in an
  architectural context (describing system behavior, interactions, responsibilities).
  Even if lowercase, the word refers to the component when the sentence discusses
  the system's architecture.
- REJECT when the word appears inside a DOTTED PACKAGE PATH or QUALIFIED NAME
  (e.g., "x.utils", "x.api", "x.datatransfer"). The dotted path refers to a
  sub-package inside the component, not to the component itself.
- REJECT when the word is used as a plain English adjective, verb, or in an idiom
  unrelated to the system (e.g., "common ground", "persistent effort").

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = v.get("approve", False)
        return results

    # ── Fix 2: Judge Rule 4 exception ─────────────────────────────────

    def _build_judge_prompt(self, comp_names, cases):
        """4-rule judge with explicit Rule 4 exception for named components."""
        return f"""JUDGE: Validate trace links between documentation and software architecture components.

APPROVAL CRITERIA:
A link S→C is valid when the sentence satisfies all four conditions:

1. EXPLICIT REFERENCE
   The component name (or a direct reference to it) appears in the sentence as a clear
   entity being discussed. This distinguishes component-specific statements from
   incidental mentions or generic discussions where the component name appears but is
   not the subject of the statement.

2. SYSTEM-LEVEL PERSPECTIVE
   The sentence describes the component's role, responsibilities, interfaces, or
   interactions within the overall system architecture. Reject statements focused on
   internal implementation details (data structures, algorithms, code-level concerns)
   that are invisible at the architectural abstraction level.

3. PRIMARY FOCUS
   The component is the main subject of what the sentence conveys, not a secondary
   or incidental mention. The sentence is fundamentally about what the component does
   or how it relates to other system elements.

4. COMPONENT-SPECIFIC USAGE
   The reference is to the component as a named entity within the system architecture,
   not to a generic concept, pattern, or technology that happens to share a name.

   CRITICAL EXCEPTION: If the system has a component literally named "X" — even when
   X is a common word like "Scheduler", "Router", "Optimizer", "Parser", "Dispatcher",
   "Handler" — and the sentence describes what X does, how X interacts with other parts
   of the system, or X's responsibilities, then this IS component-specific usage.
   The existence of a named component means architectural references to that concept
   ARE about the component. Only reject Rule 4 when "X" is clearly used in a different
   domain (e.g., "scheduler" meaning "appointment planner" in a business context, not
   the software Scheduler component).

COMPONENTS: {', '.join(comp_names)}

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""
