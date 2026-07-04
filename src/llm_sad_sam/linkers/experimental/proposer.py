#!/usr/bin/env python3
"""GroundedTypedProposer (GTP) — the pilot's LLM/structure/context proposer.

This is the *proposer* the pilot proposal argues for, built to be measured (not a
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
import re
from pathlib import Path

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


# ── batched read (many numbered sentences per call) ──────────────────────────
# The per-sentence prompt above costs one LLM call per sentence, which is not
# acceptable. These builders read many numbered sentences in ONE call. A naive
# flat list ("plain") degrades recall — the model skims a long list — so we offer
# structural fixes that make a batched read keep per-sentence attention:
#   forced   — instruct the model to process each sentence independently;
#   coverage — REQUIRE one output row per sentence (forces it to walk every one);
#   blocks   — render each sentence as its own item with its prev-sentence context.
# The empirical batch-strategy sweep is `pilot/batch_strategy_compare.py`; these
# builders are its single source of truth (it imports them), so shipped == tested.

BATCH_STRATEGIES = ("plain", "forced", "coverage", "blocks", "residual")

_COMMON_REF_RULE = (
    "A sentence refers to a component when it names it, describes its role, refers "
    "back to it, or names it inside a contrast/negation. For each reference give the "
    "exact catalog name and the exact words carrying it; do not output a component you "
    "cannot tie to specific quoted words."
)

_FORCE_CLAUSE = (
    "\nProcess the sentences STRICTLY ONE AT A TIME, in order; re-read each on its own "
    "and list every catalog component it references as if it were the only sentence "
    "given. Do not let earlier sentences make you skim later ones.\n"
)


def filter_generic_aliases(pairs, sentences, max_df: int = 5):
    """Drop generic single-word aliases that inflate false positives by over-linking
    every incidental mention (the e2e teastore leak: ``UI``/``front-end`` -> WebUI;
    see pilot/KNOWLEDGE_PROPOSER_RESULTS.md). Rule: a MULTI-word term is specific ->
    always kept; a SINGLE-word term is kept only if it occurs (as a standalone token)
    in at most ``max_df`` sentences — rare single words are real handles (``KMS`` ->
    kurento), frequent ones are generic. Frequency-only, no vocabulary => GATE-06 safe.
    ``pairs`` is ``[(term, component), ...]``; returns the filtered list."""
    if not pairs:
        return pairs
    texts = [getattr(s, "text", str(s)).lower() for s in (sentences or [])]
    kept = []
    for term, comp in pairs:
        if len(term.split()) >= 2 or not texts:
            kept.append((term, comp))
            continue
        pat = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        if sum(1 for t in texts if pat.search(t)) <= max_df:
            kept.append((term, comp))
    return kept


# Generic architectural suffixes — shared across many component names, so NOT the
# distinctive token that defines a sibling family. Structural, not benchmark vocabulary.
_GENERIC_TOKENS = {
    "service", "server", "client", "manager", "component", "components", "system",
    "module", "provider", "engine", "adapter", "controller", "handler", "db",
    "database", "layer", "api", "app", "apps", "core", "web", "ui", "gui",
}


def _sibling_families(names):
    """Group catalog components that share a DISTINCTIVE (non-generic) token — e.g.
    {HTML5 Client, HTML5 Server} share "HTML5"; {Redis PubSub, Redis DB} share "Redis".
    Purely structural over the catalog names (no vocabulary), so GATE-06 safe."""
    from collections import defaultdict
    tok2names = defaultdict(set)
    for n in names:
        for w in n.split():
            wl = w.lower()
            if wl not in _GENERIC_TOKENS and len(wl) > 1:
                tok2names[wl].add(n)
    fams, seen = [], set()
    for ns in tok2names.values():
        if len(ns) >= 2:
            key = frozenset(ns)
            if key not in seen:
                seen.add(key)
                fams.append(sorted(ns))
    return fams


def _sibling_hint(names) -> str:
    """Prompt block that makes extraction sibling-aware. Root-cause fix for the
    dominant recall miss (pilot/ERROR_MODES.md): components sharing a base name
    (HTML5 Client vs HTML5 Server) referenced by a role word ("the client"/"the
    server") are never extracted because the reader won't commit to which sibling.
    This tells it to resolve the role/base reference to the specific sibling(s) from
    the sentence's cues. Generic English + structural families → GATE-06 safe."""
    fams = _sibling_families(names)
    if not fams:
        return ""
    lines = "\n".join("  - " + " / ".join(f) for f in fams)
    return ("\nSome catalog components share a base name and differ only by a "
            "qualifier:\n" + lines + "\nWhen a sentence refers to one of these by the "
            "shared base name or by a role word (e.g. \"the client\", \"the server\", "
            "\"the database\"), decide which specific qualified component it means from "
            "the sentence's cues; if it refers to more than one of them, list each.\n")


def _alias_block(aliases) -> str:
    """Render runtime doc-derived aliases (``[(term, component), ...]``) as a prompt
    block. Empirically (pilot/KNOWLEDGE_PROPOSER_RESULTS.md) this is the single lever
    that turns the blind blocks proposer into a recall SUPERSET of s21's alias-injected
    Framing-C pass — it recovers alias-mediated mentions (e.g. "bbb-html5" -> HTML5
    Server) the knowledge-blind read misses. Generic English; the alias pairs are the
    same runtime input s21 Framing-C already consumes (GATE-06 safe)."""
    if not aliases:
        return ""
    lines = "\n".join(f'  - "{t}" -> {c}' for t, c in aliases)
    return ("\nKnown alternative terms used in THIS document — if a SENTENCE uses the "
            "term (or its wording), it refers to the mapped catalog component (quote "
            "the term as the words):\n" + lines + "\n")


def build_batch_prompt(sentences, names, roles=None, strategy="coverage",
                       prev_of=None, base_of=None, aliases=None,
                       sibling_disambig=False) -> str:
    """Build a one-call prompt over ``sentences`` using ``strategy`` (see
    BATCH_STRATEGIES). ``prev_of`` maps sentence number -> previous-sentence text
    (blocks/residual). ``base_of`` maps sentence number -> list of component names
    a base system already linked to it; the ``residual`` strategy shows this as
    context and asks the model for what the base MISSED (LLM-side conditioning, no
    coded thresholds). ``aliases`` is an optional ``[(term, component), ...]`` list of
    runtime doc aliases injected into the blocks/residual read (see ``_alias_block``)."""
    catalog = _catalog_block(names, roles)
    alias_txt = _alias_block(aliases) + (_sibling_hint(names) if sibling_disambig else "")
    if strategy in ("plain", "forced"):
        body = "\n".join(f"S{s.number}: {s.text}" for s in sentences)
        clause = _FORCE_CLAUSE if strategy == "forced" else ""
        return (
            "Read the following numbered sentences from a software design document and "
            "list every architecture component each sentence refers to.\n\n"
            f"Choose components ONLY from this catalog (copy the exact name):\n{catalog}\n\n"
            f"{_COMMON_REF_RULE}\n{clause}\nSENTENCES:\n{body}\n\n"
            'Return JSON: {"refs":[{"sentence":<int>,"component":"<name>",'
            '"quote":"<words>"}]}\nJSON only:'
        )
    if strategy == "coverage":
        body = "\n".join(f"S{s.number}: {s.text}" for s in sentences)
        nums = ", ".join(f"S{s.number}" for s in sentences)
        return (
            "Read the numbered sentences from a software design document. Go through them "
            "ONE BY ONE and account for EVERY sentence.\n\n"
            f"Choose components ONLY from this catalog (copy the exact name):\n{catalog}\n\n"
            "Output exactly one entry per sentence, in order, covering all of: "
            f"{nums}. For each sentence list the catalog components it references (empty "
            f"list if none). {_COMMON_REF_RULE}\n\n"
            f"SENTENCES:\n{body}\n\n"
            'Return JSON: {"per_sentence":[{"sentence":<int>,"components":['
            '{"component":"<name>","quote":"<words>"}]}]}\nJSON only:'
        )
    if strategy == "blocks":
        prev_of = prev_of or {}
        blocks = "\n\n".join(
            f'ITEM {s.number}\n  PREVIOUS: "{prev_of.get(s.number, "")}"\n'
            f'  SENTENCE: "{s.text}"' for s in sentences)
        return (
            "Below are independent ITEMS. Treat each ITEM as a self-contained task: "
            "decide which catalog components its SENTENCE refers to, using its PREVIOUS "
            "line only as context. Give every item the same independent attention.\n\n"
            f"Choose components ONLY from this catalog (copy the exact name):\n{catalog}\n"
            f"{alias_txt}\n{_COMMON_REF_RULE}\n\n{blocks}\n\n"
            'Return JSON: {"items":[{"item":<int>,"refs":[{"component":"<name>",'
            '"quote":"<words>"}]}]}\nJSON only:'
        )
    if strategy == "residual":
        prev_of = prev_of or {}
        base_of = base_of or {}
        blocks = "\n\n".join(
            f'ITEM {s.number}\n  PREVIOUS: "{prev_of.get(s.number, "")}"\n'
            f'  SENTENCE: "{s.text}"\n'
            f'  ALREADY LINKED: {", ".join(base_of.get(s.number, [])) or "none"}'
            for s in sentences)
        return (
            "Below are independent ITEMS from a software design document. A base "
            "system has already linked some architecture components to each SENTENCE, "
            "shown as ALREADY LINKED. Your job is to find what the base system MISSED: "
            "for each item, list any catalog components the SENTENCE refers to that are "
            "NOT already in ALREADY LINKED. If the base already captured every component "
            "the sentence refers to, return an empty list for that item. Treat each item "
            "independently and use its PREVIOUS line only as context.\n\n"
            f"Choose components ONLY from this catalog (copy the exact name):\n{catalog}\n"
            f"{alias_txt}\n{_COMMON_REF_RULE}\n\n{blocks}\n\n"
            'Return JSON: {"items":[{"item":<int>,"refs":[{"component":"<name>",'
            '"quote":"<words>"}]}]}\nJSON only:'
        )
    raise ValueError(f"unknown batch strategy: {strategy}")


def _parse_batch(txt: str, strategy="coverage") -> list[dict]:
    """Parse a batched response into ``[{sentence, component, quote}]`` (ungrounded)."""
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return []
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return []
    out = []
    if strategy in ("plain", "forced"):
        for r in obj.get("refs", []):
            try:
                out.append({"sentence": int(r["sentence"]),
                            "component": str(r["component"]).strip(),
                            "quote": str(r.get("quote", "")).strip()})
            except Exception:
                pass
    elif strategy == "coverage":
        for e in obj.get("per_sentence", []):
            try:
                sn = int(e["sentence"])
                for r in e.get("components", []):
                    out.append({"sentence": sn, "component": str(r["component"]).strip(),
                                "quote": str(r.get("quote", "")).strip()})
            except Exception:
                pass
    elif strategy in ("blocks", "residual"):
        for e in obj.get("items", []):
            try:
                sn = int(e["item"])
                for r in e.get("refs", []):
                    out.append({"sentence": sn, "component": str(r["component"]).strip(),
                                "quote": str(r.get("quote", "")).strip()})
            except Exception:
                pass
    return out


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

    def propose_batch(self, sentences, names, roles=None, batch_size: int = 20,
                      strategy: str = "blocks", prev_of=None, base_of=None,
                      key_prefix: str = "", aliases=None,
                      sibling_disambig: bool = False) -> list[dict]:
        """Batched grounded read — ONE call per ``batch_size`` numbered sentences
        (never one per sentence). Returns grounded ``{sentence, component, quote}``.

        ``strategy`` (see ``BATCH_STRATEGIES``) controls how the batch is framed so
        it keeps per-sentence recall. ``blocks`` (each sentence rendered as its own
        context-carrying item) is the default and empirical winner: batching it does
        not dilute recall — ``pilot/batch_strategy_compare.py`` shows blocks@20 gives
        recall 1.000 on teammates (10 calls) / 0.742 on bbb (5 calls) vs 0.825/0.613
        for a naive flat one-call read. Calls are cached by (catalog_mode, strategy,
        batch_size, sentence-range).
        """
        roles_in = roles if self.catalog_mode == "role" else None
        out: list[dict] = []
        alias_tag = (f"a{len(aliases)}" if aliases else "") + ("s" if sibling_disambig else "")
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i:i + batch_size]
            ck = (f"{self.catalog_mode}|{strategy}|b{batch_size}|{key_prefix}{alias_tag}|"
                  f"{chunk[0].number}-{chunk[-1].number}")
            if ck in self.cache:
                raw = self.cache[ck]
            else:
                prompt = build_batch_prompt(chunk, names, roles_in,
                                            strategy=strategy, prev_of=prev_of,
                                            base_of=base_of, aliases=aliases,
                                            sibling_disambig=sibling_disambig)
                resp = self._client().query(prompt, timeout=240)
                raw = _parse_batch(resp.text if resp.success else "", strategy)
                self.cache[ck] = raw
                if self.cache_path:
                    self.cache_path.write_text(json.dumps(self.cache, indent=1))
            lut = {n.lower(): n for n in names}
            for r in raw:
                canon = lut.get(r["component"].lower())
                if canon:
                    out.append({**r, "component": canon})
                else:
                    self.dropped_total += 1
        return out
