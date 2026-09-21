# Approach route terminology verification — 2026-09-21

## Scope

This check covers the terminology rewrite in:

- `paper/abbrev.tex`
- `paper/sections/approach.tex`

The paper now names the two end-to-end rows `routeOne` and `routeTwo`, the
candidate-generation components `genOne` and `genTwo`, and the corresponding
judges `judgeOne` and `judgeTwo`. Digits are spelled out because ordinary TeX
control-sequence names cannot contain digits.

Configuration: the live paper submodule on `main`, checked against the reported
`s_linker126` implementation. No LLM or API-backed evaluation was used.

## Commands

Run from `paper/`:

```bash
git diff --check -- abbrev.tex sections/approach.tex

for macro in routeOne routeTwo genOne genTwo judgeOne judgeTwo; do
  defs=$(rg -c "newcommand\\{\\\\${macro}\\}" abbrev.tex || true)
  uses=$(rg -o "\\\\${macro}" sections/approach.tex | wc -l)
  echo "${macro} definition=${defs} approach_uses=${uses}"
done

sed -E 's/\\label\{[^}]+\}//g' sections/approach.tex |
  rg -n 'TODO|XXX|\bXX\b|\\(linkerN|linkerB|linkerC|linkerD|nameValidator|corefValidator)|\b(linker|proposer|validator|qualifier|Reencoder)\b'

rg -F 'WRITTEN = ("exact", "alias", "part", "qualified name")' \
  ../approach/src/llm_sad_sam/linkers/experimental/s_linker126.py
rg -F 'ANTECEDENT_FORMS = ("exact", "alias")' \
  ../approach/src/llm_sad_sam/linkers/experimental/s_linker126.py

for figure in approach-overview named_link_approach coref-link-approach; do
  test -f "figures/${figure}.pdf"
done
for key in furnas1987vocabulary arora2017automated gemkow2018automatic \
           howard2013automatically falleri2010automatic fuchs_whos_2025; do
  rg -q "^@[A-Za-z]+\\{${key}," agent-linker.bib
done

command -v latexmk
```

## Results

```text
PASS: no whitespace errors in changed paper files
PASS: routeOne definition=1 approach_uses=4
PASS: routeTwo definition=1 approach_uses=4
PASS: genOne definition=1 approach_uses=3
PASS: genTwo definition=1 approach_uses=3
PASS: judgeOne definition=1 approach_uses=3
PASS: judgeTwo definition=1 approach_uses=5
PASS: no draft markers or retired public terms in live approach prose
PASS: paper evidence values and antecedent forms match s126
PASS: all referenced figures and citation keys exist
BLOCKED: latexmk is not installed; no PDF build was run
```

The absence check intentionally removes internal `\label{...}` values before
searching. The stable labels `sec:name-linker` and `sec:coref-linker` remain so
existing cross-references do not break; they are not rendered terminology.
