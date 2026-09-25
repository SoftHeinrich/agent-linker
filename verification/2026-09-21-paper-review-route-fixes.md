# Paper review route fixes — 2026-09-21

## Scope and decisions

This verification covers the requested continuation of the paper-review fixes.
The live abstract, introduction, evaluation, results, approach, conclusion, overview
figure, and coreference figure use the stage/route terminology. The paper assigns
derivation of `written` to the judging pass and states that the name-word scan also
applies to one-word component names. The description uses the plain-language phrase
“different grammatical form,” with singular/plural as an example.
The approach design also maps its three decisions to its three stated challenges in
order: shared alias knowledge addresses vocabulary mismatch, separate routes address
different evidence forms, and evidence-based judgment addresses unsupported candidates.

The S27 observation was not changed, following the author's explicit instruction.
Stable internal LaTeX labels such as `sec:name-linker` were also retained because
they are not rendered terminology. The already-completed named-reference figure edit
was not revised in this pass.

Configuration: live `paper` submodule and the reported `s_linker126` implementation.
No LLM calls or benchmark runs were needed because these changes affect paper prose
and figure labels, not recovery behavior.

## Figure export

The checked-in PDFs were regenerated from the edited draw.io sources with
`draw.io-export` 0.3.0:

```bash
npx --yes draw.io-export@0.3.0 \
  paper/figures/drawio/approach-overview.drawio \
  -F pdf -o paper/figures/approach-overview.pdf

npx --yes draw.io-export@0.3.0 \
  paper/figures/drawio/coref-linker-illustration.drawio \
  -F split-pdf -o /tmp/coref-export-
cp /tmp/coref-export-1pdf paper/figures/coref-link-approach.pdf
```

The coreference illustration has two draw.io pages; page index 1 is the figure used
by the paper.

## Static verification

The verification checked the text extracted from both exported PDFs, the live
(non-comment) TeX prose, the approach wording, and XML parsing. It also ran the diff
whitespace gate.

```bash
git -C paper diff --check

python - <<'PY'
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from pypdf import PdfReader

def pdf_text(path):
    return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)

overview = pdf_text("paper/figures/approach-overview.pdf")
coref = pdf_text("paper/figures/coref-link-approach.pdf")
assert all(text in overview for text in (
    "Candidate\nGeneration", "Candidate\nJudging",
    "Name Candidate\nGenerator", "Coreference\nCandidate\nGenerator",
))
assert not re.search(r"Proposer|Proposers|Named Link|Coreference Link", overview)
assert all(text in coref for text in (
    "S3 ", "S4 ", "S5 ", "S6 ",
    "antecedent_sentence=S3", "antecedent_sentence=S5",
))
assert not re.search(r"antecedent_sentence=(?:23|S20)|\bS2[0-4]\b", coref)

live = []
for path in [Path("paper/main.tex"), *Path("paper/sections").glob("*.tex")]:
    for number, line in enumerate(path.read_text().splitlines(), 1):
        text = line.split("%", 1)[0]
        text = re.sub(r"\\label\{[^}]+\}", "", text)
        text = re.sub(r"\\bibliography\{[^}]+\}", "", text)
        if re.search(
            r"(?i)\\linker[nbcd]|\\(?:name|coref)Validator|"
            r"\b(?:linker|linkers|proposer|proposers)\b",
            text,
        ):
            live.append((path, number, text))
assert not live, live

approach = Path("paper/sections/approach.tex").read_text()
assert "the judging pass derives a factual \\texttt{written} value" in approach
assert "both one-word and multi-word component names" in approach
assert "such as singular instead of plural" in approach
assert "Three design decisions address the three challenges above in the same order." in approach
assert "First, alias discovery builds shared document-specific knowledge to address the vocabulary mismatch" in approach
assert "Second, \\approach separates candidate recovery into the \\routeOne and \\routeTwo" in approach
assert "Third, every remaining candidate is evaluated with the evidence for that case" in approach

for path in (
    "paper/figures/drawio/approach-overview.drawio",
    "paper/figures/drawio/coref-linker-illustration.drawio",
):
    ET.parse(path)
PY
```

Result:

```text
PASS paper diff has no whitespace errors
PASS overview PDF uses candidate-generation/candidate-judging terminology
PASS overview PDF contains no retired proposer labels
PASS coreference PDF consistently uses local S3–S6 labels
PASS live paper contains no retired linker/proposer role terminology
PASS written is attributed to the judging pass
PASS one-word and multi-word component names are both covered
PASS grammatical variants are described in plain language
PASS the three challenges map in order to three design decisions
PASS the challenge-decision mapping is combined with the stage/route structure
PASS both edited draw.io sources parse as XML
```

## Paper build

```bash
./scripts/build-paper.sh
```

Result (exit 1):

```text
latexmk is required to build the paper (install TeX Live with latexmk).
```

No alternative TeX engine is installed (`pdflatex`, `xelatex`, `lualatex`, and
`tectonic` are also absent), so a full paper build could not be run in this
environment. The exported figures were visually inspected from PNG exports; the
updated labels are visible.
