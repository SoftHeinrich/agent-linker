# Grammar-marking verification

Run on 2026-09-07 in the `paper/` VS Code workspace.

## Workspace configuration

The checked-in configuration is [`settings.json`](settings.json):

```json
{
  "ltex.enabled": ["bibtex", "latex"],
  "ltex.language": "en-US",
  "ltex.dictionary": {
    "en-US": [":ltex.dictionary.en-US.txt"]
  },
  "ltex.latex.environments": {
    "lstlisting": "ignore",
    "verbatim": "ignore"
  }
}
```

The project recommends `ltex-plus.vscode-ltex-plus` and
`james-yu.latex-workshop` in [`extensions.json`](extensions.json).

## Verification commands and results

```bash
python -m json.tool paper/.vscode/settings.json >/dev/null
python -m json.tool paper/.vscode/extensions.json >/dev/null
```

Result: `PASS` (both files are valid JSON).

```bash
codium --list-extensions --show-versions | sort | rg -i 'ltex|latex|grammarly'
```

Result:

```text
james-yu.latex-workshop@10.18.0
ltex-plus.vscode-ltex-plus@15.7.1
tecosaur.latex-utilities@0.4.14
znck.grammarly@0.24.0
```

```bash
~/.vscode-oss/extensions/ltex-plus.vscode-ltex-plus-15.7.1-universal/\
  lib/ltex-ls-plus-18.7.0/bin/ltex-cli-plus --verbose \
  paper/sections/metric.tex
```

Result: the command returned exit status `3`, which is LTeX+'s result for a
document containing diagnostics. It detected grammar/spelling issues in the
paper, including:

```text
paper/sections/metric.tex:59:42: info: 'advantegies': Possible spelling mistake found.
paper/sections/metric.tex:60:16: info: The modal verb 'must' requires the verb's base form.
paper/sections/metric.tex:62:11: info: The pronoun 'it' requires a third-person verb or a past tense.
```

Opening the paper in a fresh VSCodium window also started the bundled
`ltex-ls-plus-18.7.0` language-server process, confirming that the editor can
activate the checker for the LaTeX workspace.

An unrelated `git diff --check` still reports trailing whitespace at lines 47,
60, and 104 of the already-modified `paper/sections/metric.tex`; those edits
were left untouched.
