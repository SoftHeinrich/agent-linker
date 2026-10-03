import json
import re
from functools import lru_cache

from nltk.stem import WordNetLemmatizer

from .inputs import Link, load_components, load_sentences
from .llm import ask_json

ALIAS_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

ALIAS_EXCLUSION_RULES = """A fragment of a longer identifier is not an alias: if a term appears only as part of a compound or qualified name, do not include it."""

ALIAS_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. It is invalid when the phrase is generic vocabulary or identifies anything other than that one component. When uncertain, prefer APPROVE."""

TRACE_LINK_RULE = ("A trace link holds between a sentence and a component when the sentence makes an architectural claim about that component -- when it says something about that component as a participant in the system this document describes. Referring to the component as a participant is itself such a claim, even when the sentence says nothing further about it. "
                   """

Every case gives you the expression the sentence uses, the sentence itself, the evidence the document supplies, and the component whose name that expression reaches. The evidence says what the expression is doing here; none of it is a verdict.

  written -- how the sentence writes the component's name. Use exact when the component's full catalog name is written as a name; alias when the document established an alternate form for that component and this sentence writes it; part when only one word of the component's multi-word name is written; and qualified name when the full catalog name occurs only inside a longer joined or dotted identifier. A shorter surface leaves more readings open; it does not make the reading in front of you wrong. Where the sentence does not write the name as such, ask what the expression itself denotes in its local context: a participant in the system, or something merely associated with software.

Reject only on a positive ground -- that the sentence asserts nothing of this component, because the name is doing some other job here, or because the sentence denies what it would otherwise say of it.

Some sentences use an ordinary English word that happens to coincide with a component's name. Approve only when the sentence uses that word as the name of the component; if it is used in its ordinary sense and the component is not what the sentence is talking about, reject. Capitalization is evidence for a name and its absence is evidence against, but neither settles it on its own.

An expression that occurs only as part of a longer joined or dotted identifier is naming a piece of that identifier, not a participant in what the sentence describes. An expression denoting what a component acts on or produces refers to that thing and not to the component, however clearly the component is the one acting on it.""")

SURFACE_NOT_EVIDENCE = """Where the sentence does not write the name in full, that a surface can name this component is not evidence that it does here."""

NAME_DEMAND = """For each case, first quote the EXACT words from the sentence the verdict rests on -- the words that state the architectural claim about the component, or "none" if the sentence makes no such claim -- then decide approve true/false based on that quote."""

COREFERENCE_RULES = """Resolve when the surrounding sentences make one component the clear antecedent, under any form the document uses for it. Avoid resolving when two or more equally plausible antecedents exist."""

COREFERENCE_JUDGE_FOCUS = """Check coref resolution: does the referring expression in this sentence actually refer to the named component as an architectural participant?"""

COREFERENCE_JUDGE_RULES = """These are coreference links: a pronoun or noun phrase in the sentence is claimed to refer back to the component, which is NOT named in the sentence itself. Approve only when the sentence contains a genuine referring expression that unambiguously points to THIS component and makes an architectural claim about it. Reject when there is no such referring expression or when the antecedent could equally be a different component. An expression denoting what a component acts on or produces refers to that thing and not to the component, however clearly the component is the one acting on it. When uncertain, reject."""

WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"
CONTEXT = 5
JUDGE_BATCH = 25
COREFERENCE_BATCH = 10
LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=None)
def lemmas(word):
    word = word.casefold()
    return frozenset(LEMMATIZER.lemmatize(word, pos) for pos in ("n", "v"))


def name_spans(text, name):
    return [match.span() for match in
            re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)]


def word_spans(text, name):
    words = [lemmas(word) for word in re.findall(WORD, name)]
    return [match.span() for match in re.finditer(WORD, text)
            if any(lemmas(match.group(0)) & word for word in words)]


def in_dotted_path(text, start, end):
    before = start > 1 and text[start - 1] == "." and text[start - 2].isalnum()
    after = end + 1 < len(text) and text[end] == "." and text[end + 1].isalnum()
    return before or after


def only_in_identifier(text, name):
    spans = [match.span() for match in re.finditer(rf"\b{re.escape(name.lower())}\b", text)]
    return bool(spans) and all(in_dotted_path(text, *span) for span in spans)


def batches(items, size):
    return [items[start:start + size] for start in range(0, len(items), size)]


def sentence_number(value):
    if isinstance(value, str):
        value = value.lstrip("Ss")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def approves(item):
    value = item.get("approve", False)
    return value is True or (isinstance(value, str) and value.lower() == "true")


class AgentLinker:
    def __init__(self, chat, use_aliases=True):
        self.chat = chat
        self.use_aliases = use_aliases

    def link(self, text_path, model_path):
        self.sentences = load_sentences(text_path)
        self.components = load_components(model_path)
        self.by_number = {sentence.number: sentence for sentence in self.sentences}
        self.names = [component.name for component in self.components]
        self.aliases = self.learn_aliases() if self.use_aliases else {}
        links = {}
        for link in self.name_links() + self.coreference_links():
            links.setdefault((link.sentence, link.component_id), link)
        return list(links.values())

    def ask(self, phase, prompt, timeout=120, **requirement):
        return ask_json(self.chat, phase, prompt, timeout, **requirement)

    def aliases_of(self, name):
        return [term for term, owner in self.aliases.items() if owner == name]

    def written_as(self, text, name):
        if name_spans(text, name):
            return "qualified name" if only_in_identifier(text, name) else "exact"
        if any(name_spans(text, term) for term in self.aliases_of(name)):
            return "alias"
        return "part"

    def window(self, number):
        return [sentence.number for sentence in self.sentences
                if abs(sentence.number - number) <= CONTEXT]

    def table(self, numbers):
        return json.dumps([{"sentence": n, "text": self.by_number[n].text}
                           for n in sorted(numbers)])

    def previous(self, number):
        sentence = self.by_number.get(number - 1)
        return f"[prev: {sentence.text}] " if sentence else ""

    def learn_aliases(self):
        data = self.ask("alias_extract", f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(self.names)}

{ALIAS_EXTRACTION_RULES}

{ALIAS_EXCLUSION_RULES}

DOCUMENT:
{chr(10).join(sentence.text for sentence in self.sentences)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:""", timeout=300)
        proposals = []
        for key in ("abbreviations", "synonyms") if data else ():
            value = data.get(key, [])
            if isinstance(value, dict):
                value = [{"term": term, "component": name} for term, name in value.items()]
            for record in value if isinstance(value, list) else ():
                if not isinstance(record, dict):
                    continue
                pair = (record.get("term"), record.get("component"))
                if pair[0] and pair[1] in self.names and pair not in proposals:
                    proposals.append(pair)
        if not proposals:
            return {}

        data = self.ask("alias_judge", f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(self.names)}

PROPOSED MAPPINGS:
{json.dumps([{"term": term, "component": name} for term, name in proposals])}

{ALIAS_JUDGE_RULES}

Return JSON, echoing each approved mapping in full:
{{"approved": [{{"term": "term1", "component": "FullComponent"}}]}}
JSON only:""", require_present="approved")
        verdicts = data.get("approved") if isinstance(data, dict) else None
        if isinstance(verdicts, list):
            pairs = [(item.get("term"), item.get("component"))
                     for item in verdicts if isinstance(item, dict)]
            approved = [pair for index, pair in enumerate(pairs)
                        if pair in proposals and pair not in pairs[:index]]
        else:
            approved = proposals
        owners = {}
        for term, name in approved:
            owners.setdefault(term, []).append(name)
        return {term: names[0] for term, names in owners.items() if len(names) == 1}

    def name_candidates(self):
        candidates = {}
        for sentence in self.sentences:
            for component in self.components:
                for name in (component.name, *self.aliases_of(component.name)):
                    spans = name_spans(sentence.text, name)
                    if spans:
                        surface = sentence.text[spans[0][0]:spans[0][1]]
                        candidates[(sentence.number, component.id)] = (
                            sentence, component, surface, "full_name")
                        break
        for sentence in self.sentences:
            for component in self.components:
                key = (sentence.number, component.id)
                spans = word_spans(sentence.text, component.name)
                if key in candidates or not spans or self.owned_elsewhere(sentence.text, component.name):
                    continue
                surface = sentence.text[spans[-1][0]:spans[-1][1]]
                candidates[key] = (sentence, component, surface, "partial_name")
        return [candidates[key] for key in sorted(candidates)]

    def owned_elsewhere(self, text, name):
        mine = word_spans(text, name)
        others = [span for other in self.names if other != name
                  for span in name_spans(text, other)]
        return bool(mine and others) and all(
            any(start <= a and b <= end and end - start > b - a for start, end in others)
            for a, b in mine)

    def name_links(self):
        groups = {}
        for candidate in self.name_candidates():
            sentence, _, surface, _ = candidate
            groups.setdefault((sentence.number, surface.casefold()), []).append(candidate)
        single = [group[0] for group in groups.values() if len(group) == 1]
        links = []
        for batch in batches(single, JUDGE_BATCH):
            cases, context = [], set()
            for index, (sentence, component, surface, _) in enumerate(batch, 1):
                written = self.written_as(sentence.text, component.name)
                if written == "part":
                    context.update(self.window(sentence.number))
                cases.append(f'Case {index}: "{surface}" -> {component.name}\n'
                             f'  {self.previous(sentence.number)}"{sentence.text}"\n'
                             f"  Evidence: written={written}")
            table = f"\nSENTENCES\n{self.table(context)}\n" if context else ""
            data = self.ask("name_judge", f"""Validate components in a document.

COMPONENTS: {', '.join(self.names)}

{TRACE_LINK_RULE}
{table}
{SURFACE_NOT_EVIDENCE}

{NAME_DEMAND}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:""", require="validations")
            approved = set()
            for item in data.get("validations", []):
                if 0 <= item.get("case", 0) - 1 < len(batch):
                    if approves(item):
                        approved.add(item["case"] - 1)
                    else:
                        approved.discard(item["case"] - 1)
            links += [Link(sentence.number, component.id, component.name, source)
                      for index, (sentence, component, _, source) in enumerate(batch)
                      if index in approved]
        return links

    def coreference_links(self):
        resolutions = []
        for batch in batches(self.sentences, COREFERENCE_BATCH):
            context = {number for sentence in batch for number in self.window(sentence.number)}
            cases = [f"--- Case {index} ---\nTARGET S{sentence.number}: {sentence.text}"
                     for index, sentence in enumerate(batch, 1)]
            data = self.ask("coreference", f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(self.names)}

SENTENCES (the document text the cases are drawn from)
{self.table(context)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

Quote the referring expression first, then name the component it points to.

{chr(10).join(cases)}

{COREFERENCE_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "candidates": ["Name", "OtherName"], "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:""", timeout=600, require_present="resolutions")
            for item in data.get("resolutions", []) if data else ():
                number = sentence_number(item.get("sentence"))
                antecedent = sentence_number(item.get("antecedent_sentence"))
                if (number in self.by_number and item.get("component") in self.names
                        and antecedent in self.by_number):
                    resolutions.append((number, item))

        latest = {(number, item["component"]): item for number, item in resolutions}
        admitted = []
        for number, item in resolutions:
            cited = latest[(number, item["component"])]
            antecedent = self.by_number[sentence_number(cited.get("antecedent_sentence"))]
            if self.written_as(antecedent.text, item["component"]) in ("exact", "alias"):
                admitted.append((number, item["component"], cited))

        component_id = {component.name: component.id for component in self.components}
        links = []
        for batch in batches(admitted, JUDGE_BATCH):
            cases = []
            for index, (number, name, cited) in enumerate(batch, 1):
                claimed = ""
                if cited.get("reference"):
                    claimed += f'  Claimed reference: "{cited.get("reference")}"\n'
                if cited.get("antecedent_text"):
                    claimed += (f'  Claimed antecedent (S{sentence_number(cited.get("antecedent_sentence"))}): '
                                f'"{cited.get("antecedent_text")}"\n')
                cases.append(f"Case {index}: pronoun/role-ref -> {name}\n{claimed}"
                             f'  {self.previous(number)}"{self.by_number[number].text}"')
            data = self.ask("coreference_judge", f"""Validate components in a document. {COREFERENCE_JUDGE_FOCUS}

COMPONENTS: {', '.join(self.names)}

{COREFERENCE_JUDGE_RULES}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then state the strongest ground there is for rejecting this case under the
rules above (or "none" if there is none), then decide: approve unless that ground is one
the rules above make decisive. An objection you could raise against most sentences is not
a ground for rejecting this one.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "objection": "<strongest ground to reject, or none>", "approve": true}}]}}
JSON only:""", require="validations")
            approved = set()
            for item in data.get("validations", []) if data else ():
                if 0 <= item.get("case", 0) - 1 < len(batch):
                    if approves(item):
                        approved.add(item["case"] - 1)
                    else:
                        approved.discard(item["case"] - 1)
            links += [Link(number, component_id[name], name, "coreference")
                      for index, (number, name, _) in enumerate(batch) if index in approved]
        return links
