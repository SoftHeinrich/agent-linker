#!/usr/bin/env python3
"""GroundedTypedProposer (GTP) — the pilot's LLM/structure/context proposer.

This is the *proposer* the PROPOSAL.md argues for, built to be measured (not a
merged change; lives beside router_direct.py, deletes nothing). It replaces regex
identifier extraction with a single reasoning-off LLM read that is:

  GROUNDED   — a candidate survives only if its element resolves to a real name in
               the runtime catalog (component list here; a code index in the direct
               variant). Grounding is the FP floor — no stop-lists, no caps.
  CONTEXT-   — the read carries the previous sentence (constraining context that
   AUGMENTED   pins referents), never role justifications by default.
  TYPED      — each candidate is tagged with a reference MODE (AFFIRMATIVE /
               IMPLICIT / ANAPHORA / CONTRAST / CODEPATH); the mode IS the route.

Taboo-safe: the prompt template is generic English. The catalog (component names,
and optionally short role lines) is a RUNTIME input, exactly as s21's validator is
already handed runtime component names. No benchmark term is baked into the code.

ALL LLM calls reasoning-off (s21 is a no-reasoning config; the `quote` field is the
only justification, mirroring s21's claim-before-verdict). Decisions are cached.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
sys.path.insert(0, str(SRC))

# load OPENAI_API_KEY from approach/.env (same bootstrap as fn_judge/run_judges.py)
_envf = HERE.parents[1] / ".env"
if _envf.exists():
    for ln in _envf.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

MODEL = "gpt-5.4"
MODES = ("AFFIRMATIVE", "IMPLICIT", "ANAPHORA", "CONTRAST", "CODEPATH")

# ── prompt (generic; catalog injected at runtime) ────────────────────────────

_INSTR = (
    "You read one sentence from a software design document, with the sentence just "
    "before it for context, and list every architecture component the sentence "
    "refers to.\n\n"
    "Choose components ONLY from this catalog (copy the exact catalog name):\n"
    "{catalog}\n\n"
    "A sentence can refer to a component by (a) naming it directly, (b) describing "
    "its role/function with a generic noun or example (\"the storage\", \"a file "
    "server\"), (c) a pronoun or bare role phrase pointing back to it, or (d) naming "
    "it inside a contrast or negation (still a reference). For EACH component the "
    "sentence refers to, output its exact catalog name, the reference MODE, and the "
    "exact words in the sentence that carry the reference.\n"
    "MODE is one of: AFFIRMATIVE (named plainly) | IMPLICIT (role/generic/example, "
    "no proper name) | ANAPHORA (pronoun/role phrase) | CONTRAST (inside a "
    "negation/contrast) | CODEPATH (only inside a dotted code path).\n"
    "Do NOT list a component you cannot tie to specific quoted words. If the "
    "sentence refers to no catalog component, return an empty list.\n\n"
    'PREVIOUS: "{prev}"\n'
    'SENTENCE: "{sent}"\n\n'
    'Return JSON: {{"refs":[{{"component":"<exact catalog name>","mode":"<MODE>",'
    '"quote":"<exact words>"}}]}}\nJSON only:'
)


def _catalog_block(names, roles=None):
    """names-only (grounding by name) or name + one role line (context-augmented)."""
    if roles:
        return "\n".join(f"- {n}: {roles.get(n, '')}".rstrip(": ") for n in names)
    return "\n".join(f"- {n}" for n in names)


def build_prompt(sentence: str, prev: str, names, roles=None) -> str:
    return _INSTR.format(catalog=_catalog_block(names, roles),
                         prev=prev or "", sent=sentence)


# ── LLM plumbing ─────────────────────────────────────────────────────────────

def make_client():
    """Reasoning-off gpt-5.4 client (never set OPENAI_REASONING_EFFORT)."""
    os.environ.pop("OPENAI_REASONING_EFFORT", None)
    os.environ["OPENAI_MODEL_NAME"] = MODEL
    from llm_sad_sam.llm_client import LLMClient, LLMBackend
    return LLMClient(backend=LLMBackend.OPENAI, model=MODEL,
                     temperature=0.1, enable_logging=False)


def _parse(txt: str) -> list[dict]:
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return []
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return []
    out = []
    for r in obj.get("refs", []):
        try:
            out.append({"component": str(r["component"]).strip(),
                        "mode": str(r.get("mode", "AFFIRMATIVE")).strip().upper(),
                        "quote": str(r.get("quote", "")).strip()})
        except Exception:
            pass
    return out


def ground(refs: list[dict], names) -> tuple[list[dict], list[dict]]:
    """Keep refs whose element resolves to a catalog name (case-insensitive exact).
    Returns (grounded, dropped). Grounding is the entire precision floor."""
    lut = {n.lower(): n for n in names}
    kept, dropped = [], []
    for r in refs:
        canon = lut.get(r["component"].lower())
        if canon:
            kept.append({**r, "component": canon})
        else:
            dropped.append(r)
    return kept, dropped


class GroundedTypedProposer:
    """propose(sentence, prev, names, roles) -> list[{component, mode, quote}] (grounded)."""

    def __init__(self, client=None, cache_path: Path | None = None, catalog_mode="name"):
        self.client = client
        self.catalog_mode = catalog_mode          # "name" | "role"
        self.cache_path = cache_path
        self.cache = (json.loads(cache_path.read_text())
                      if cache_path and cache_path.exists() else {})
        self.dropped_total = 0

    def _client(self):
        if self.client is None:
            self.client = make_client()
        return self.client

    def propose(self, key: str, sentence: str, prev: str, names, roles=None) -> list[dict]:
        ck = f"{self.catalog_mode}|{key}"
        if ck in self.cache:
            raw = self.cache[ck]
        else:
            roles_in = roles if self.catalog_mode == "role" else None
            prompt = build_prompt(sentence, prev, names, roles_in)
            resp = self._client().query(prompt, timeout=180)
            raw = _parse(resp.text if resp.success else "")
            self.cache[ck] = raw
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self.cache, indent=1))
        grounded, dropped = ground(raw, names)
        self.dropped_total += len(dropped)
        return grounded
