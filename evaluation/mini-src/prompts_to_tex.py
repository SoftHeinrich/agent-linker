#!/usr/bin/env python3
"""Render the prompt-template appendix verbatim from the canonical linker's source.

The five prompt builders of the reported arm are read with ``ast`` (stdlib; the linker
is not imported, so none of its dependencies are needed). Every literal piece of a
prompt -- the f-string text and the module constants it interpolates -- is copied as
written. Only the per-call data (component names, document, cases) becomes a
placeholder, shown in braces.

Before writing, each rendered template is matched against a recorded prompt of the
reported terra run: its literal text must occur in that prompt, in order, with the
placeholders matching the data in between. A template the logs do not confirm is not
written.

    python3 mini-src/prompts_to_tex.py      # writes reports/tex/prompts.tex
    python3 mini-src/sync_paper.py          # copies it to <paper>/appendix/prompts.tex
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import csv_to_tex as c2t

ROOT = Path(__file__).resolve().parents[2]
LINKER = (ROOT / "approach/src/llm_sad_sam/linkers/experimental"
          / f"s_linker{c2t.DEFAULT_ARM[1:]}.py")
LOGS = ROOT / "results/greedymerge_e2e_terra_r1_20261002/llm_logs"
OUT = c2t.TEX_OUT / "prompts.tex"

# Placeholders are written <<like this>> in the layouts below; everything else in a
# layout is literal prompt text and is checked against the logs like the rest.
COMPONENTS = "<<component names, comma-separated>>"
CASES = {
    # `_format_union_case`
    "union": ('Case <<i>>: "<<span>>" -> <<component name>>\n'
              '  <<[prev: previous sentence] >>"<<sentence>>"\n'
              '  Evidence: written=<<exact | alias | part | qualified name>>\n'
              '<<further cases>>'),
    # the case lines built in the coreference judging pass
    "coref_judge": ('Case <<i>>: pronoun/role-ref -> <<component name>>\n'
                    '  Claimed reference: "<<reference>>"\n'
                    '  Claimed antecedent (S<<n>>): "<<antecedent text>>"\n'
                    '  <<[prev: previous sentence] >>"<<sentence>>"\n'
                    '<<further cases>>'),
    # the blocks built in `_prompt_coref`
    "coref": ('--- Case <<i>> ---\n'
              'TARGET S<<n>>: <<sentence>>\n'
              '<<further cases>>'),
}

# (subsection, title, builder, the logged prompt's opening, {expression: layout}).
# Subsections follow the approach section: alias discovery and the two routes.
PROMPTS = [
    ("Alias Discovery", "Alias extraction.", "_prompt_doc_knowledge_extract",
     "Find all alternative names", {
         "', '.join(comp_names)": COMPONENTS,
         "chr(10).join(doc_lines)": "<<document sentences, one per line>>"}),
    ("Alias Discovery", "Alias judge.", "_prompt_doc_knowledge_judge",
     "JUDGE: Review these component", {
         "', '.join(comp_names)": COMPONENTS,
         "json.dumps(proposals)": "<<proposed mappings as JSON>>"}),
    ("Named-Reference Route", "Name judge.", "_prompt_union",
     "Validate components in a document.\n", {
         "', '.join(comp_names)": COMPONENTS,
         "table": "<<SENTENCES table of the nearby sentences, included when a case "
                  "writes only one word of a name>>",
         "chr(10).join(cases)": CASES["union"]}),
    ("Coreference Route", "Coreference candidate generator.", "_prompt_coref",
     "Resolve references", {
         "', '.join(comp_names)": COMPONENTS,
         "json.dumps(sentence_table)": "<<document sentences as JSON>>",
         "chr(10).join(blocks)": CASES["coref"]}),
    ("Coreference Route", "Coreference judge.", "_prompt_coref_validation",
     "Validate components in a document. Check coref", {
         "', '.join(comp_names)": COMPONENTS,
         # the coreference pass is the one caller and passes COREF_VALIDATION_FOCUS
         "f' {focus}' if focus else ''": " {COREF_VALIDATION_FOCUS}",
         "chr(10).join(cases)": CASES["coref_judge"]}),
]


def constants(tree):
    """Module-level string constants, including f-strings composed of other constants."""
    env = {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.JoinedStr):
            return "".join(value(v.value if isinstance(v, ast.FormattedValue) else v)
                           for v in node.values)
        raise KeyError(ast.unparse(node))

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = value(node.value)
            except KeyError:
                pass
    return env


def builder(tree, name):
    """The JoinedStr a prompt builder returns."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            assert len(returns) == 1 and isinstance(returns[0].value, ast.JoinedStr), name
            return returns[0].value
    raise SystemExit(f"FAIL: {LINKER.name} has no prompt builder {name}")


def segments(text):
    """Split a layout into ('lit', text) and ('ph', name) pieces."""
    out = []
    for i, piece in enumerate(re.split(r"<<(.*?)>>", text, flags=re.S)):
        if piece:
            out.append(("ph" if i % 2 else "lit", piece))
    return out


def template(fstring, env, layouts):
    out, unused = [], set(layouts)
    for part in fstring.values:
        if isinstance(part, ast.Constant):
            out.append(("lit", part.value))
            continue
        expr = ast.unparse(part.value).replace('"', "'")
        if isinstance(part.value, ast.Name) and part.value.id in env:
            out.append(("lit", env[part.value.id]))
        elif expr in layouts:
            unused.discard(expr)
            out += segments(layouts[expr].format(**env))
        else:
            raise SystemExit(f"FAIL: no placeholder for {{{expr}}}")
    assert not unused, unused
    return out


def confirm(segs, opening):
    """The template must match a recorded prompt of the reported run, end to end."""
    pattern = "".join(re.escape(t) if kind == "lit" else ".*?" for kind, t in segs)
    seen = 0
    for log in sorted(LOGS.glob("s_linker*_calls.json")):
        for call in json.loads(log.read_text()):
            prompt = call["prompt"]
            if not prompt.startswith(opening):
                continue
            seen += 1
            if re.fullmatch(pattern, prompt, re.S):
                return f"{log.name} ({seen} candidates scanned)"
    raise SystemExit(f"FAIL: no recorded prompt starting {opening!r} matches the "
                     f"template ({seen} candidates)")


LATEX = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
         "#": r"\#", "^": r"\^{}", "_": r"\_", "~": r"\textasciitilde{}", "%": r"\%",
         "<": r"\textless{}", ">": r"\textgreater{}", "—": r"\textemdash{}"}


NUMBERS = {4: "four", 5: "five", 6: "six", 7: "seven"}


def escape(text):
    # "-{}" keeps a run of hyphens from setting as a dash: the prompt writes "--".
    return re.sub(r"-(?=-)", "-{}", "".join(LATEX.get(ch, ch) for ch in text))


def latex(segs):
    """One quote block; line breaks and indentation as in the prompt."""
    text = "".join(t if kind == "lit" else "\x01" + t + "\x02" for kind, t in segs)
    lines = []
    for line in text.strip("\n").split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        body = re.sub("\x01(.*?)\x02", lambda m: r"{\normalfont\itshape\{" + m[1] + r"\}}",
                      escape(line.strip(" ")),
                      flags=re.S)
        if body.startswith("["):        # not an optional argument of the preceding \\
            body = "{}" + body
        lines.append((r"\hspace*{%.1fem}" % (indent / 2) if indent else "") + body)
    out, blank = [], False
    for line in lines:
        if not line:
            blank = True
            continue
        if out:
            out[-1] += r"\\[\medskipamount]" if blank else r"\\"
        out.append(line)
        blank = False
    return ("\\begin{quote}\\small\\ttfamily\\raggedright\n"
            + "\n".join(out) + "\n\\end{quote}\n")


def main():
    tree = ast.parse(LINKER.read_text())
    env = constants(tree)
    rel = LINKER.relative_to(ROOT)
    body = [f"% GENERATED by evaluation/mini-src/prompts_to_tex.py from {rel}.\n"
            "% Do not edit by hand: rerun prompts_to_tex.py, then sync_paper.py.\n",
            "\\section{Prompt Templates}\n\\label{app:prompts}\n\n",
            f"\\approach uses {NUMBERS[len(PROMPTS)]} \\ac{{LLM}} prompts. "
            "We give each prompt verbatim as the implementation sends it. "
            "Data that differs per call is shown in braces.\n"]
    section = None
    for sub, title, name, opening, layouts in PROMPTS:
        segs = template(builder(tree, name), env, layouts)
        print(f"OK    {name:32s} matches {confirm(segs, opening)}")
        if sub != section:
            body.append(f"\n\\subsection{{{sub}}}\n")
            section = sub
        body.append(f"\n\\paragraph{{{title}}}\n" + latex(segs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(body))
    print(f"PASS: {len(PROMPTS)} prompts rendered verbatim from {rel} to {OUT}")


if __name__ == "__main__":
    main()
