#!/usr/bin/env python3
"""Check built RQ3 table text and report TeX diagnostics."""
from pathlib import Path
import re
import sys
import pymupdf

build = Path(sys.argv[1])
doc = pymupdf.open(build / 'main.pdf')
print(f'PDF pages: {len(doc)}')
found = 0
for i, page in enumerate(doc):
    text = page.get_text()
    if 'RQ3 judging configurations' not in text:
        continue
    found += 1
    for needle in ('Configuration', 'NoName', 'NoCitation', 'NoValidator',
                   'including deterministic exclusions'):
        assert needle in text, (i+1, needle)
    print(f'PASS RQ3 labels and note on page {i+1}')
assert found == 2, found
log = (build / 'main.log').read_text()
missing = re.findall(r'LaTeX Warning: (?:Reference|Citation).*undefined', log)
assert not missing, missing
print('PASS no undefined reference or citation warnings')
print('TeX box warnings (outside the modified RQ3 tables):')
for line in log.splitlines():
    if 'Overfull' in line or 'Underfull' in line:
        print(line)
