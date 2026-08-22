"""ALinker/Core: the same workflow in three stages and one contract.

PRICED AND NOT ADOPTED.  This module is retained as the measured alternative, in
the spirit of ``s_linker76`` and ``s_linker79``, not as a candidate head.  Four of
its five collapses were already refuted by this branch's own rounds -- s26/s60
(alias proposal and judging in one call, F1 -2.7), s36 (the two focused full-name
judging calls merged, F1 -0.7), the s25 design law (showing the denotation judge
its target, -5.5 gold) and s46 (the alias table no longer suppressing
partial-name candidates, F1 -1.5) -- against the standing finding that every
consolidation of two LLM decisions into one call raises recall and lowers
precision, twelve variants and no exception.  See ``docs/CORE_ARCHITECTURE.md``
for what survives: the decision topology is load-bearing and stays at seven
points; only the code surface collapses.

The one element here that no round has tested is contrastive resolution -- ruling
on competing components for one sentence together.  Every merge the ledger
refused folded a judgment into the call that produced what it judged; none asked
one judge about two candidates side by side.

Read the document once, propose every claim anyone can make, then resolve each
sentence's competing claims together.  That is the whole architecture.

The three-linker head it replaces decomposed the work by *orthography*: a
writing relation over fidelity (as spelled / any case / any spelling) and extent
(whole name / one word), with one linker, one judge, one prompt and one evidence
format per cell.  That taxonomy is an artifact of how names happen to be typed,
not of how linking works, and it cost five prompts, two enums, three judging
paths and about 1,800 lines to express.

Core keeps one decomposition axis -- role -- and one data type.  Every stage is
a pure function over the same ``Claim``:

    READ     (document, catalog)      -> Dossier      one LLM call
    PROPOSE  (document, Dossier)      -> {Claim}      deterministic scan + one LLM call
    RESOLVE  (sentence, {Claim})      -> {Claim}      one LLM call per contested sentence

Nothing here is a heuristic about capitalization.  A proposer emits a claim; the
resolver, which sees every claim competing for the same sentence at once, decides
which survive.  Whether the name was proper-cased, lowercased, abbreviated or
absent is visible to the resolver in the sentence itself, where it belongs.

Three measurements on the head's own runs fixed this shape:

  * every gold link is already reachable, so the redesign may not spend anything
    on retrieval; the deterministic scan is kept precisely to preserve that;
  * 95% of the head's residual false negatives were never *proposed* by any
    stage, so proposal is unified and made permissive rather than cascaded;
  * 68% of its false positives pick the wrong component on a sentence that does
    link to a sibling, which no per-pair binary judge can see, so judging is
    replaced by per-sentence contrastive selection.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass


from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.llm_client import LLMBackend, LLMClient
from llm_sad_sam.pcm_parser import parse_pcm_repository

# Ordinary English function words.  This is the only fixed word list in the
# module and it carries no project vocabulary of any kind.
STOPWORDS = frozenset(
    "a an the and or of to in for on at by is are was were be been it its this that "
    "with as from into over under can will would should may might must not no".split()
)


# ─────────────────────────────────────────────────────────────────────────────
# The one contract
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Claim:
    """One assertion that a sentence discusses a component, and its warrant.

    ``span`` is the text the proposer read, always an exact substring of the
    claimed sentence.  ``anchor`` is the earlier sentence that supplies the
    component's name when the claimed sentence does not state it itself; it is
    ``None`` when the span carries the name.  Those two fields replace the head's
    ``NameForm`` enum, its ``MentionType`` enum, its ``EvidenceBundle`` and the
    three separate evidence formats its three judges each parsed.
    """

    sentence: int
    component: str          # component id
    span: str               # exact substring of sentence `sentence`
    anchor: int | None = None
    origin: str = ""        # provenance for logging; never read as control flow

    def key(self) -> tuple[int, str]:
        return (self.sentence, self.component)


@dataclass
class Entry:
    """What the reader learned about one component."""

    id: str
    name: str
    aliases: tuple[str, ...] = ()
    gloss: str = ""
    confusable_with: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        return (self.name,) + tuple(self.aliases)


Dossier = dict[str, Entry]          # keyed by component id


# ─────────────────────────────────────────────────────────────────────────────
# Reading names out of text.  One relation, used by one stage.
# ─────────────────────────────────────────────────────────────────────────────


def words(text: str) -> list[str]:
    """The word sequence of a name or a sentence, case- and separator-free.

    Splits camel case and any punctuation, so ``Image Provider``,
    ``image-provider`` and ``ImageProvider`` all read as ``["image","provider"]``.
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]


def find_name(sentence: str, name: str) -> str | None:
    """Return the exact substring of `sentence` that writes `name`, else None.

    Matches the name at a word boundary ignoring case, or as its word sequence
    ignoring separators and compound joining.  There is deliberately no third
    strictness level: the resolver reads the sentence and can see the case.
    """
    direct = re.search(
        r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", sentence, re.I
    )
    if direct:
        return direct.group(0)

    target = words(name)
    if not target:
        return None
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+", sentence)]
    flat = words(sentence)
    # `flat` and `tokens` can differ in length when a token is itself camel case;
    # fall back to a substring test in that case rather than mis-slicing.
    if len(flat) != len(tokens):
        return sentence if " ".join(target) in " ".join(flat) else None
    for i in range(len(flat) - len(target) + 1):
        if flat[i:i + len(target)] == target:
            return sentence[tokens[i][1]:tokens[i + len(target) - 1][2]]
    return None


def find_word(sentence: str, word: str) -> str | None:
    """Return the substring writing a single name word, under plain inflection."""
    for token in re.finditer(r"[A-Za-z0-9]+", sentence):
        lowered = token.group(0).lower()
        if lowered == word or lowered in (word + "s", word + "es"):
            return token.group(0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# The workflow
# ─────────────────────────────────────────────────────────────────────────────


class ALinkerCore:
    """Read, propose, resolve."""

    _VARIANT_NAME = "alinker_core"

    #: Sentences per RESOLVE call.  One number, not three batch sizes.
    RESOLVE_BATCH = 12
    #: Sentences per PROPOSE call.
    PROPOSE_BATCH = 60
    ASK_ATTEMPTS = 2

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
        no_knowledge: bool = False,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.6-terra")
        self.llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self.no_knowledge = no_knowledge
        self.calls = 0
        self.trace: list[dict] = []
        print("ALinker/Core (read -> propose -> resolve)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ── entry point ──────────────────────────────────────────────────────────

    def link(self, text_path, model_path, **_kwargs) -> list[SadSamLink]:
        started = time.time()
        self.calls = 0
        self.trace = []

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        sent_map = build_sent_map(sentences)
        texts = {s.number: s.text for s in sentences}
        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        dossier = self.read(sentences, components)
        claims = self.propose(sentences, dossier)
        kept = self.resolve(claims, texts, dossier)

        links = [
            SadSamLink(
                sentence_number=c.sentence,
                component_id=c.component,
                component_name=dossier[c.component].name,
                confidence=1.0,
                source=c.origin,
            )
            for c in sorted(kept, key=lambda c: (c.sentence, c.component))
        ]
        print(f"\nFinal: {len(links)} links "
              f"({time.time() - started:.1f}s, {self.calls} LLM calls)")
        return links

    # ── stage 1: READ ────────────────────────────────────────────────────────

    def read(self, sentences, components) -> Dossier:
        """One pass over the document that says who each component is.

        Absorbs the head's alias-extraction prompt, its alias-judging prompt and
        the model-understanding module it deleted.  The reader is asked for the
        aliases *and* for the siblings a reader could confuse a component with,
        because the sibling confusion is what the resolver is later asked to
        settle; discovering it here costs nothing extra and is project-specific
        knowledge, not a rule.
        """
        base = {c.id: Entry(id=c.id, name=c.name) for c in components}
        if self.no_knowledge:
            return base

        catalog = ", ".join(c.name for c in components)
        body = "\n".join(f"S{s.number}: {s.text}" for s in sentences)
        prompt = (
            "You are reading a system's documentation to learn who its components are.\n\n"
            f"COMPONENTS: {catalog}\n\n"
            f"DOCUMENT:\n{body}\n\n"
            "For each component, report:\n"
            "  aliases: other names this document actually uses for it -- an "
            "abbreviation it defines, a synonym it restates, or a shorthand it "
            "uses after introducing the full name. Only phrases present in the "
            "document. Omit a phrase that names the whole system, names a "
            "different component, or names a group rather than one component.\n"
            "  gloss: one clause naming what this component does, drawn from the document.\n"
            "  confusable_with: other components in the list a careful reader "
            "could mistake it for here, because they share a word or a role.\n\n"
            "Return JSON:\n"
            '  {"components": [{"component": "<exact name from COMPONENTS>", '
            '"aliases": [...], "gloss": "...", "confusable_with": [...]}]}'
        )
        data = self._ask(prompt, phase="core_read", require_present="components")

        by_name = {c.name: c.id for c in components}
        for item in data.get("components", []) or []:
            cid = by_name.get(str(item.get("component", "")).strip())
            if cid is None:
                continue
            aliases = tuple(
                a for a in (str(x).strip() for x in item.get("aliases", []) or [])
                if a and a.lower() != base[cid].name.lower() and a.lower() not in STOPWORDS
            )
            siblings = tuple(
                s for s in (str(x).strip() for x in item.get("confusable_with", []) or [])
                if s in by_name and s != base[cid].name
            )
            base[cid] = Entry(
                id=cid,
                name=base[cid].name,
                aliases=aliases,
                gloss=str(item.get("gloss", "")).strip(),
                confusable_with=siblings,
            )
        found = sum(len(e.aliases) for e in base.values())
        print(f"[read] {found} aliases over {len(base)} components")
        return base

    # ── stage 2: PROPOSE ─────────────────────────────────────────────────────

    def propose(self, sentences, dossier: Dossier) -> list[Claim]:
        """Every claim anyone can make, from two proposers of equal standing.

        The head ran three linkers in a fixed order and withheld from each the
        pairs the earlier ones had settled.  That ordering existed to stop a
        weaker form overruling a stronger one -- a job the resolver now does
        properly, by seeing the claims together.  So the proposers here are
        unordered and permissive, and neither admits anything on its own.
        """
        texts = {s.number: s.text for s in sentences}
        claims = self._scan(texts, dossier) + self._read_off(sentences, dossier)

        # One claim per (sentence, component): keep the one whose span is longest,
        # i.e. the proposer that read the most of the name.
        best: dict[tuple[int, str], Claim] = {}
        for c in claims:
            if not self._well_formed(c, texts, dossier):
                continue
            prior = best.get(c.key())
            if prior is None or len(c.span) > len(prior.span):
                best[c.key()] = c
        out = sorted(best.values(), key=lambda c: (c.sentence, c.component))
        print(f"[propose] {len(out)} claims over "
              f"{len({c.sentence for c in out})} sentences")
        return out

    def _scan(self, texts: dict[int, str], dossier: Dossier) -> list[Claim]:
        """Deterministic proposer: names, aliases, and uniquely-owned name words.

        This is the recall floor and the reason the redesign may delete the
        cascade without losing reachability.  It is free and it never varies.
        """
        owner: dict[str, set[str]] = {}
        for entry in dossier.values():
            for word in words(entry.name):
                if word in STOPWORDS or len(word) <= 2:
                    continue
                owner.setdefault(word, set()).add(entry.id)

        out: list[Claim] = []
        for snum, text in texts.items():
            for entry in dossier.values():
                hit = next((s for s in (find_name(text, n) for n in entry.names()) if s), None)
                if hit:
                    out.append(Claim(snum, entry.id, hit, None, "scan:name"))
                    continue
                for word in words(entry.name):
                    if owner.get(word) != {entry.id}:
                        continue        # a word two components share is evidence for neither
                    span = find_word(text, word)
                    if span:
                        out.append(Claim(snum, entry.id, span, None, "scan:word"))
                        break
        return out

    def _read_off(self, sentences, dossier: Dossier) -> list[Claim]:
        """LLM proposer: everything the scan cannot see, in the same contract.

        This one call replaces the head's named-reference extractor *and* its
        coreference resolver.  They were separate stages only because they
        returned different shapes; here both return a claim, and a claim whose
        sentence states no name simply carries an anchor.
        """
        catalog = ", ".join(e.name for e in dossier.values())
        known = "; ".join(
            f"{e.name}: {', '.join(e.aliases)}" for e in dossier.values() if e.aliases
        ) or "none"
        out: list[Claim] = []
        by_name = {e.name: e.id for e in dossier.values()}
        batches = [sentences[i:i + self.PROPOSE_BATCH]
                   for i in range(0, len(sentences), self.PROPOSE_BATCH)]

        for batch in batches:
            body = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = (
                "Find the sentences that discuss a component's responsibilities.\n\n"
                f"COMPONENTS: {catalog}\n"
                f"KNOWN ALIASES: {known}\n\n"
                f"DOCUMENT:\n{body}\n\n"
                "Report one entry per (sentence, component). For each:\n"
                "  span: the exact words of that sentence you read, copied "
                "character for character from it.\n"
                "  anchor: if the sentence names the component only through a "
                "pronoun or a phrase referring back, the number of the earlier "
                "sentence that states the name. Otherwise null.\n\n"
                "Report a sentence that refers back as readily as one that names "
                "the component outright. Do not decide borderline cases; report them.\n\n"
                "Return JSON:\n"
                '  {"claims": [{"sentence": <n>, "component": "<exact name>", '
                '"span": "...", "anchor": <n or null>}]}'
            )
            data = self._ask(prompt, phase="core_propose", require_present="claims")
            for item in data.get("claims", []) or []:
                cid = by_name.get(str(item.get("component", "")).strip())
                snum = _as_int(item.get("sentence"))
                if cid is None or snum is None:
                    continue
                out.append(Claim(
                    sentence=snum,
                    component=cid,
                    span=str(item.get("span", "")).strip(),
                    anchor=_as_int(item.get("anchor")),
                    origin="read",
                ))
        return out

    def _well_formed(self, c: Claim, texts: dict[int, str], dossier: Dossier) -> bool:
        """A claim is admissible only if its own warrant checks out.

        This is the head's substring check and its structural antecedent
        constraint, stated once for every claim instead of once per linker.
        """
        text = texts.get(c.sentence)
        if not text or not c.span:
            return False
        if c.span.lower() not in text.lower():
            return False                       # a fabricated span warrants nothing
        if c.anchor is None:
            return True
        if not (0 < c.anchor < c.sentence):
            return False
        anchor_text = texts.get(c.anchor)
        if not anchor_text:
            return False
        # The sentence a refer-back leans on must itself state the name.
        return any(find_name(anchor_text, n) for n in dossier[c.component].names())

    # ── stage 3: RESOLVE ─────────────────────────────────────────────────────

    def resolve(self, claims: list[Claim], texts: dict[int, str],
                dossier: Dossier) -> list[Claim]:
        """Decide each sentence's claims together, not one at a time.

        The head asked a judge, per candidate, whether that one link held. A
        judge asked that question cannot tell a component from its sibling,
        because it is never shown the sibling: two thirds of the head's false
        positives are a sentence given to the wrong one of two components that
        share a word. Here the unit of decision is the sentence, and competing
        claims are ruled on in each other's presence.
        """
        by_sentence: dict[int, list[Claim]] = {}
        for c in claims:
            by_sentence.setdefault(c.sentence, []).append(c)

        settled: list[Claim] = []
        contested: dict[int, list[Claim]] = {}
        for snum, group in by_sentence.items():
            if self._uncontested(group, dossier):
                settled.extend(group)
            else:
                contested[snum] = group

        print(f"[resolve] {len(settled)} claims settled without a call, "
              f"{sum(len(g) for g in contested.values())} contested "
              f"over {len(contested)} sentences")

        kept = list(settled)
        items = sorted(contested.items())
        for i in range(0, len(items), self.RESOLVE_BATCH):
            kept.extend(self._resolve_batch(items[i:i + self.RESOLVE_BATCH], texts, dossier))
        return kept

    def _uncontested(self, group: list[Claim], dossier: Dossier) -> bool:
        """A sentence needs no call when exactly one component can be meant.

        The single claim must state a full name of its component, that component
        must share no name word with a sibling, and the reader must not have
        flagged it as confusable. This is a routing rule about how many answers
        are possible, not a rule about capitalization.
        """
        if len(group) != 1:
            return False
        claim = group[0]
        entry = dossier[claim.component]
        if claim.anchor is not None or entry.confusable_with:
            return False
        if not any(find_name(claim.span, n) for n in entry.names()):
            return False
        mine = {w for w in words(entry.name) if w not in STOPWORDS}
        for other in dossier.values():
            if other.id != entry.id and mine & {w for w in words(other.name)}:
                return False
        return True

    def _resolve_batch(self, items, texts: dict[int, str], dossier: Dossier) -> list[Claim]:
        cases, index = [], {}
        for n, (snum, group) in enumerate(items, start=1):
            index[n] = group
            lines = [f"Case {n}: S{snum}: {texts.get(snum, '')}"]
            for claim in group:
                entry = dossier[claim.component]
                note = f" -- {entry.gloss}" if entry.gloss else ""
                lines.append(f"  candidate: {entry.name}{note}")
                lines.append(f"    read: \"{claim.span}\"")
                if claim.anchor is not None:
                    lines.append(f"    refers back to S{claim.anchor}: {texts.get(claim.anchor, '')}")
                if entry.confusable_with:
                    lines.append(f"    not to be confused with: {', '.join(entry.confusable_with)}")
            cases.append("\n".join(lines))

        prompt = (
            "For each case, decide which of the candidate components the sentence "
            "actually discusses.\n\n"
            "A sentence may discuss more than one, or none. Where two candidates "
            "share a word or a role, say which one this sentence is about and drop "
            "the other; do not keep both because both are plausible. Keep a "
            "candidate when the sentence says something of that component, "
            "including a bare mention. Drop it when the words you read are doing "
            "another job -- naming a different thing, part of a longer identifier, "
            "or an ordinary use of the word.\n\n"
            "Quote the words you ruled on before you rule.\n\n"
            f"CASES:\n" + "\n\n".join(cases) + "\n\n"
            "Return JSON:\n"
            '  {"cases": [{"case": <n>, "claim": "<words you ruled on>", '
            '"keep": ["<component name>", ...]}]}'
        )
        data = self._ask(prompt, phase="core_resolve", require_present="cases")

        kept: list[Claim] = []
        answered = set()
        for item in data.get("cases", []) or []:
            n = _as_int(item.get("case"))
            group = index.get(n) if n is not None else None
            if group is None:
                continue
            answered.add(n)
            names = {str(x).strip() for x in item.get("keep", []) or []}
            kept.extend(c for c in group if dossier[c.component].name in names)
        # A case the resolver did not answer is undecided, and an undecided claim
        # is not a link. Silence rejects; it must never approve.
        missing = set(index) - answered
        if missing:
            print(f"    resolve: {len(missing)} case(s) unanswered, dropped")
        return kept

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _ask(self, prompt: str, *, phase: str, require_present: str,
             timeout: int = 180) -> dict:
        for attempt in range(self.ASK_ATTEMPTS):
            self.calls += 1
            parsed = self.llm.extract_json(self.llm.query(prompt, timeout=timeout))
            self.trace.append({"phase": phase, "attempt": attempt,
                               "ok": bool(parsed) and require_present in (parsed or {})})
            if parsed and require_present in parsed:
                return parsed
            if attempt < self.ASK_ATTEMPTS - 1:
                print(f"    {phase}: unusable response, retrying...")
        return {}


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip().lstrip("Ss"))
    except (TypeError, ValueError):
        return None
