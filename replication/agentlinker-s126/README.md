# AgentLinker s126 replication package

This directory contains the released `s_linker126` runtime with Python comments removed, the five document and model inputs, their SAD–SAM gold CSVs, the WordNet corpus used by the linker, and recorded outputs for three runs each on GPT-5.6 terra and luna. The parsed runtime code is unchanged. Recorded runs include the full linker and its no-knowledge setting. The `recorded/` tree holds final link CSVs, four phase snapshots per dataset, and the linker's phase and call logs. It contains no evaluation scripts or historical control linker.

Use Python 3.11 or newer. From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
OPENAI_API_KEY="$OAI_KEY" .venv/bin/python run.py --model terra --output run-output/terra-1
```

The live command processes all five datasets. Use `--model luna` for the second recorded model, `--datasets mediastore` for one dataset, or `--no-knowledge` for the corresponding released setting. Give each run a separate output directory. The environment defaults are `OPENAI_REASONING_EFFORT=none` and `OPENAI_SERVICE_TIER=flex`. A live run needs network access and an OpenAI credential. Hosted model responses may differ from the recorded runs.

To export link CSVs from a recorded final snapshot without an API call:

```bash
.venv/bin/python run.py \
  --from-cache recorded/greedymerge_e2e_terra_r1_20261002 \
  --output /tmp/agentlinker-export
```

For no-knowledge snapshots, add `--no-knowledge` and select a `greedymerge_noknow_e2e_*` directory. Snapshot files are Python pickles; load only copies from a trusted source. The live linker writes phase snapshots but does not read them to skip API calls. Offline export reads the `final.pkl` snapshot directly.

The recorded runs are the 2026-10-02 sweep documented in the parent repository's `evaluation/HOWTO-REGENERATE-RQ.md`, made with this runtime after the name judge's mention clause was replaced by a reference clause; the paper's tables are computed from these runs. The five dataset paths match the released `approach/run_ablation.py`. `BENCHMARK-LICENSE` covers the vendored benchmark inputs; the WordNet archive contains its own `LICENSE` file.

Run `sha256sum -c SHA256SUMS` from this directory to check the packaged files. `VERIFICATION.txt` records the offline package check.

The small live MediaStore rerun is in `rerun/mediastore-terra-default/`, run on the default service tier; its log is `rerun/live-default.log`. It produced 31 links, as does the recorded terra run 1 CSV; a fresh hosted-model run is not expected to match a recorded run exactly in general.
