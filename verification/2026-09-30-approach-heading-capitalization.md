# Approach heading capitalization

Changed subsection headings 3.2 and 3.3 to title case, matching 3.1.
Verification: Python 3 source assertions from the repository root; no PDF build.

```sh
python3 - <<'PY'
from pathlib import Path
import re
p = Path('paper/sections/approach.tex')
headings = re.findall(r'\\subsection\{([^}]+)\}', p.read_text())
assert headings == ['Alias Discovery', 'Named-Reference Route', 'Coreference Route'], headings
print('PASS: 3.1 Alias Discovery; 3.2 Named-Reference Route; 3.3 Coreference Route')
a = Path('paper/abbrev.tex').read_text()
assert r'\newcommand{\routeOne}{named-reference route\xspace}' in a
assert r'\newcommand{\routeTwo}{coreference route\xspace}' in a
print('PASS: lowercase route macros retained for running prose')
PY
```

Result (exit 0):

```text
PASS: 3.1 Alias Discovery; 3.2 Named-Reference Route; 3.3 Coreference Route
PASS: lowercase route macros retained for running prose
```
