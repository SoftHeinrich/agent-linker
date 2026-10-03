# Luna cost estimates at standard rates, 2026-10-03

OpenAI standard uncached input/output rates per million tokens are Terra US$2/12
and Luna US$0.20/1.20. Verified on 2026-10-03 against:

- https://developers.openai.com/api/docs/models/gpt-5.6-terra
- https://developers.openai.com/api/docs/models/gpt-5.6-luna

The engine uses recorded token counts, not billing amounts or service tiers.
It applies no Flex or caching discounts. Costs cover all five projects, averaged
over three runs. Input data remain the runs named in `inference_cost.py`.

The engine computes `(Total_input_k * input_rate + Total_output_k * output_rate) / 1000`.
The renderer formats those costs to two decimals. Its pricing note reads the same
PRICES constants. Both Luna costs are generated, with no handwritten result cells.

Regenerate only the cost artifacts from the repository root:

```bash
python3 evaluation/mini-src/inference_cost.py
python3 - <<'RENDER'
import sys
import shutil
from pathlib import Path
sys.path.insert(0, 'evaluation/mini-src')
import csv_to_tex as renderer
spec = next(s for s in renderer.SPECS if s['out'] == 'inference-cost.tex')
renderer.render(spec)
shutil.copyfile(renderer.TEX_OUT / spec['out'], Path('paper/table/inference-cost.tex'))
shutil.copyfile(renderer.TEX_SRC / spec['csv'], Path('paper/table/inference-cost.csv'))
RENDER
python3 verification/verify_inference_cost_compact.py
```

A scratch regeneration using `inference_cost.py --out <temporary-directory>` and
`csv_to_tex.render` matched all three usage CSVs and the LaTeX table byte for byte.
The verification below independently checks means and costs using Decimal, all
four table rows, rounding, and matching paper copies.

```text
$ python3 verification/verify_inference_cost_compact.py
PASS cost: approach/terra = US$0.478450 -> US$0.48
PASS cost: Artemis/terra = US$0.269572 -> US$0.27
PASS cost: approach/luna = US$0.053054 -> US$0.05
PASS cost: Artemis/luna = US$0.032048 -> US$0.03
PASS cost: 4 system/backend rows, 48 token values, all costs, and paper copies match
$ git diff --check -- evaluation/mini-src/inference_cost.py evaluation/mini-src/csv_to_tex.py verification/verify_inference_cost_compact.py evaluation/reports/tex/inference-cost.tex evaluation/reports/tex_src/inference_cost_by_system.csv
PASS (no whitespace errors)
$ git -C paper diff --check -- table/inference-cost.csv table/inference-cost.tex
PASS (no whitespace errors)
latexmk: not found on PATH
pdflatex: not found on PATH
tectonic: not found on PATH
```

The PDF could not be rebuilt because no LaTeX compiler is installed on PATH.
An initial whole-paper whitespace check reported an unrelated existing edit:
`sections/results.tex:212: trailing whitespace`. Checks scoped to the changed
cost artifacts pass. The unrelated text was preserved.
