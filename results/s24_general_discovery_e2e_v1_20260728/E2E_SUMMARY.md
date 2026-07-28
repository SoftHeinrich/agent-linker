# General discovery paired five-project E2E

Command configuration:

- variants: `s_linker21`, `s_linker24_role_orchestrator`
- datasets: MediaStore, TeamMates, TeaStore, BigBlueButton, JabRef
- backend/model: OpenAI / `gpt-5.6-terra`
- reasoning effort: `none`
- run: fresh paired calls, 2026-07-28

| Project | S21 TP / FP / FN | S24 TP / FP / FN | S21 F1 / F2 | S24 F1 / F2 |
| --- | --- | --- | --- | --- |
| MediaStore | 30 / 0 / 1 | 30 / 0 / 1 | 98.36 / 97.40 | 98.36 / 97.40 |
| TeamMates | 50 / 10 / 7 | 53 / 4 / 4 | 85.47 / 86.81 | 92.98 / 92.98 |
| TeaStore | 26 / 0 / 1 | 26 / 0 / 1 | 98.11 / 97.01 | 98.11 / 97.01 |
| BigBlueButton | 46 / 5 / 16 | 55 / 4 / 7 | 81.42 / 76.92 | 90.91 / 89.58 |
| JabRef | 18 / 0 / 0 | 18 / 0 / 0 | 100.00 / 100.00 | 100.00 / 100.00 |

| Variant | TP / FP / FN | Macro F1 / F2 | Pooled F1 / F2 |
| --- | --- | --- | --- |
| S21 | 170 / 15 / 25 | 92.67 / 91.63 | 89.47 / 88.08 |
| S24 | 182 / 8 / 13 | 96.07 / 95.40 | 94.55 / 93.81 |

S24 passes all four aggregate promotion gates. Its participant source adds
14 TP / 0 FP: 11 on BigBlueButton, 3 on TeamMates, and no false positives on
the other projects.

Raw output: `ablation_20260728_095746.json` and the ten variant/project link
CSVs in this directory.
