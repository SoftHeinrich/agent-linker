# Replication package

- `agentlinker/`: AgentLinker code, benchmark inputs, and recorded runs.
- `artemis/`: the ArTEMiS baseline re-run on the same models: source, run script, and recorded runs.

Check the files with `sha256sum -c SHA256SUMS`.

## AgentLinker

Requires Python 3.11+.

```bash
cd agentlinker
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
OPENAI_API_KEY=<your-key> .venv/bin/python run.py --model terra --output run-output/terra
```

Use `--model luna` for the second model, `--datasets <name>` to run one dataset, and `--no-aliases` to disable alias discovery.
you can use flex tier for reduced model cost, but the response time can be delayed.

The recorded runs are in `recorded/<terra|luna>/<full|no-aliases>/run<1-3>/`. To replay a recorded run offline from its saved model responses:

```bash
.venv/bin/python run.py --replay recorded/terra/full/run1 --output run-output/replay
```

For a `no-aliases` run, add `--no-aliases`.

## ArTEMiS

Requires JDK 21 and Maven.

```bash
cd artemis
OPENAI_API_KEY=<your-key> ./run.sh terra
```

The recorded runs are in `recorded/<terra|luna>/run<1-3>/`, and their logs are in `logs/`. The source comes from the ArTEMiS authors' replication package (MIT licence, `taas25/LICENSE.md`), adapted to GPT-5.6.
