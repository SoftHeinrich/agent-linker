# Caption-length verification

Date: 2026-09-21

## Command

```sh
cd evaluation
python3 mini-src/csv_to_tex.py
python3 mini-src/sync_paper.py /mnt/hostshare/ardoco-home/agent-linker/paper --only rq
python3 /mnt/hostshare/ardoco-home/agent-linker/scripts/check-caption-length.py
python3 mini-src/check.py
python3 mini-src/gen_csv_to_temp.py
python3 mini-src/sync_paper.py /mnt/hostshare/ardoco-home/agent-linker/paper --only rq --check
```

## Result

```text
PASS: 17 active captions are one sentence with at most 14 words.
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
RESULT: all generated CSVs reproduce the committed repo copies. Repo untouched.
IN SYNC: all 24 paper file(s) match the generated output. (2 absent for this arm)
```

Configuration: evaluation arm `s126`; the RQ4 floor CSV is absent for this arm.
