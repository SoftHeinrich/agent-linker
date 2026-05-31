"""S-Linker13 Skill-Learned Clean — Voyager Pilot (Phase 12 EXTENSION).

Standalone variant for the Voyager-style train/test pilot. Replaces the 9
active prompt constants imported by `s_linker13_clean_v3` with axiom-only
counterparts from `prompts_v3_axiom`, then wraps each axiom prompt with a
LEARNED PATTERNS block sourced at __init__ time from a JSON skill bank.

PIPELINE: identical to SLinker13Clean (SAD-SAM 3-tier DAG). The ONLY change
is the prompt provenance — every other code path, helper, and dependency is
inherited from SLinker13Clean unchanged. This is a PROMPT-SOURCE swap, not
a pipeline change.

SKILL BANK FORMAT (JSON)
------------------------
    {
        "abstract_patterns": [
            {"prompt": "DOC_KNOWLEDGE_JUDGE_RULES",
             "pattern": "<one abstract sentence>"},
            ...
        ]
    }

Default path: ``./results/voyager_pilot/skill_bank.json``. If the file is
absent OR the field ``abstract_patterns`` is empty, the variant runs with
PURE axiom prompts (training mode, iter 0). Patterns whose ``prompt`` field
matches the constant name get appended to that prompt under a clearly
marked "LEARNED PATTERNS" section; patterns with an empty/unknown prompt
field get appended to every prompt (treated as global skills).

DISTILLED SKILLS FILE
---------------------
Test-time injection uses ``./results/voyager_pilot/distilled_skills.json``
with the same shape but the field name ``distilled_skills``. The two paths
are not interchangeable — training writes to one, test reads from the other,
so a misconfigured test run can never accidentally see un-distilled patterns.

The variant exposes ``skill_path`` as a constructor kwarg so the training
script can point it at either file deterministically.

GATE-06
-------
The variant itself contains NO benchmark terms (audited via the same regex
as scripts/audit_12_05_revisit.py). The skill bank IS allowed to contain
LLM-derived patterns; the training harness performs taboo grep on every
pattern before persisting and on every pattern in the distilled file before
test-time injection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental import prompts_v3_axiom as _axiom


# ---------------------------------------------------------------------------
# Skill-bank loading
# ---------------------------------------------------------------------------

DEFAULT_SKILL_PATH = "./results/voyager_pilot/skill_bank.json"
LEARNED_HEADER = (
    "\n\nLEARNED PATTERNS (apply when relevant; do not contradict the principles above):"
)

# Names of the 9 axiom prompts the variant can scope a learned pattern to.
PROMPT_CONSTANT_NAMES = (
    "AMBIGUITY_FEW_SHOT",
    "AMBIGUITY_RULES",
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "DOC_KNOWLEDGE_JUDGE_EXAMPLES",
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "ENTITY_EXTRACTION_RULES",
    "VALIDATION_RULES",
    "COREF_RULES",
    "SEED_DISAMBIGUATION_RULES",
)


def _load_patterns(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    """Load patterns from skill_bank.json OR distilled_skills.json.

    Accepts either ``{"abstract_patterns": [...]}`` (training format) or
    ``{"distilled_skills": [...]}`` (frozen format). Returns the list of
    pattern dicts. Returns ``[]`` if the file is missing OR empty.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
    patterns = data.get("abstract_patterns")
    if patterns is None:
        patterns = data.get("distilled_skills", [])
    out: list[dict[str, str]] = []
    for raw in patterns:
        if isinstance(raw, str):
            out.append({"prompt": "", "pattern": raw.strip()})
        elif isinstance(raw, dict):
            text = str(raw.get("pattern", "")).strip()
            if not text:
                continue
            scope = str(raw.get("prompt", "")).strip()
            out.append({"prompt": scope, "pattern": text})
    return out


def _wrap(base: str, prompt_name: str, patterns: list[dict[str, str]]) -> str:
    """Append LEARNED PATTERNS section to ``base`` if any apply.

    Patterns whose ``prompt`` field is empty are treated as global; patterns
    that name a specific constant are only appended to that constant.
    """
    applicable = [
        p["pattern"] for p in patterns
        if not p["prompt"] or p["prompt"] == prompt_name
    ]
    if not applicable:
        return base
    body = "\n".join(f"- {pat}" for pat in applicable)
    return f"{base}{LEARNED_HEADER}\n{body}"


# ---------------------------------------------------------------------------
# Module-level rewrite: ``prompts_v3`` is replaced on import for the LIFETIME
# of any imported SLinker13SkillLearned instance. The parent SLinker13Clean
# imports the 9 prompt constants AT MODULE IMPORT TIME from prompts_v3, so we
# cannot mutate them post-hoc on a single instance — we instead patch the
# constants on the parent's module reference for each instance's link() call.
# The patch is scoped (saved + restored) to prevent cross-variant bleed.
# ---------------------------------------------------------------------------

class SLinker13SkillLearned(SLinker13Clean):
    """Voyager-pilot variant: axiom prompts wrapped with learned patterns.

    Constructor accepts an optional ``skill_path`` kwarg. If not supplied, the
    variant reads from ``DEFAULT_SKILL_PATH``. The patterns are bound once at
    __init__; mutate ``self.patterns`` between runs (training loop) and call
    ``self._rebuild_prompts()`` to refresh.
    """

    _VARIANT_NAME = "s_linker13_skill_learned_clean"

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
        skill_path: str | None = None,
    ):
        super().__init__(
            backend=backend,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self.skill_path = str(skill_path or DEFAULT_SKILL_PATH)
        self.patterns: list[dict[str, str]] = _load_patterns(self.skill_path)
        self._wrapped_prompts: dict[str, str] = {}
        self._rebuild_prompts()
        print(
            f"  Voyager pilot: skill_path={self.skill_path} "
            f"patterns_loaded={len(self.patterns)}"
        )

    def _rebuild_prompts(self) -> None:
        """Compute the wrapped axiom + learned-patterns prompt strings.

        Stored in ``self._wrapped_prompts`` and applied by ``link()`` via a
        scoped monkey-patch of ``prompts_v3`` so the parent SLinker13Clean
        code path sees the wrapped values without being modified.
        """
        wrapped: dict[str, str] = {}
        for name in PROMPT_CONSTANT_NAMES:
            base = getattr(_axiom, name)
            wrapped[name] = _wrap(base, name, self.patterns)
        self._wrapped_prompts = wrapped

    def reload_skills(self, skill_path: str | None = None) -> int:
        """Reload patterns from disk (called between training iterations).

        Returns the new pattern count.
        """
        if skill_path is not None:
            self.skill_path = str(skill_path)
        self.patterns = _load_patterns(self.skill_path)
        self._rebuild_prompts()
        return len(self.patterns)

    # ------------------------------------------------------------------
    # Scoped prompt patching
    # ------------------------------------------------------------------

    def link(self, text_path, model_path, **kwargs):  # type: ignore[override]
        """Patch prompts_v3 + helper_v3 constants for the duration of link().

        SLinker13Clean imports prompt constants at module load time. To swap
        them per-instance we save the originals, overwrite, run, restore.
        This MUST be linear (no concurrent SLinker13SkillLearned instances
        sharing one Python process).
        """
        from llm_sad_sam.linkers.experimental import prompts_v3 as _v3
        from llm_sad_sam.linkers.experimental import s_linker13_clean as _parent_mod

        # Save originals from BOTH the source module and the bound names in
        # the parent module (the parent already imported them with `from ...
        # import NAME`, so a rebinding on prompts_v3 alone won't propagate).
        savepoints: list[tuple[Any, str, Any]] = []
        for name in PROMPT_CONSTANT_NAMES:
            new_val = self._wrapped_prompts[name]
            for mod in (_v3, _parent_mod):
                if hasattr(mod, name):
                    savepoints.append((mod, name, getattr(mod, name)))
                    setattr(mod, name, new_val)
        try:
            return super().link(text_path=text_path, model_path=model_path, **kwargs)
        finally:
            for mod, name, original in savepoints:
                setattr(mod, name, original)
