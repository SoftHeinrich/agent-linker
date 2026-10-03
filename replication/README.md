# Replication package

- `agentlinker/`: the approach, the five benchmark inputs, and the recorded runs the paper's tables are computed from.
- `artemis/`: the ArTEMiS baseline re-run on the same two models: its source, a run script, and the recorded outputs.

`sha256sum -c SHA256SUMS` checks every packaged file. `VERIFICATION.txt` records how the package was checked.

## AgentLinker

Python 3.11 or newer. From `agentlinker/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
OPENAI_API_KEY=<your-key> .venv/bin/python run.py --model terra --output run-output/terra-1
```

A run processes all five datasets. `--datasets mediastore` selects one, `--model luna` selects the second model, and `--no-aliases` disables alias discovery. Requests use `gpt-5.6-<model>`, reasoning effort `none`, seed 42, and the service tier in `OPENAI_SERVICE_TIER` (default `flex`; `default` selects the standard tier). For each dataset the run writes `<dataset>_links.csv` and `<dataset>_calls.json`, which holds every prompt and response. Hosted model responses can differ between runs.

- `agentlinker/inputs.py` reads the documents and the PCM models.
- `agentlinker/linker.py` holds the prompts and the workflow: alias discovery, name linking, coreference linking.
- `agentlinker/llm.py` holds the OpenAI client, the replay client, and the JSON parsing.

`recorded/<terra|luna>/<full|no-aliases>/run<1-3>/` holds the three runs per model and setting (link CSVs and call logs). These runs were made on 2026-10-02 with gpt-5.6-terra and gpt-5.6-luna.

To rerun the workflow offline, answering each prompt with its recorded response:

```bash
.venv/bin/python run.py --replay recorded/terra/full/run1 --output run-output/replay
```

Add `--no-aliases` for a `no-aliases` run. Replay stops if the code sends a prompt that was not recorded, or leaves a recorded call unused. Replaying the 60 recorded dataset runs reproduces 58 link CSVs byte for byte. Replay stops on bigbluebutton in `terra/full/run1` and `luna/full/run2`. In those runs one sentence writes two approved aliases of FSESL ("fsels" and "FreeSWITCH Event Socket Layer"). The code that made the runs picked which alias to show the name judge by iterating an unordered set; this code takes the aliases in the alias judge's order. So one case line of one name-judge prompt differs. Answering that call with its recorded response gives the recorded links.

`data/` holds the benchmark documents, models, and gold standards; `BENCHMARK-LICENSE` covers them. `nltk_data/` holds WordNet, which carries its own licence.

## ArTEMiS

`artemis/taas25/` is ArTEMiS from the TAAS25 replication package (github.com/ArDoCo/Replication-Package-TAAS25_LLM-assisted-Software-Traceability-with-Architecture-Entity-Recognition, commit `ed07208`, MIT licence in `LICENSE.md`), with these changes:

- a `GPT_5_6` model whose name comes from `OPENAI_MODEL_NAME_5_6`, next to `GPT_5_4` and `GPT_5_5` constants;
- the OpenAI organization id is optional, and the service tier comes from `OPENAI_SERVICE_TIER` (unset selects the standard tier);
- request timeout 30 minutes and 8 retries, instead of 10 minutes and the client default;
- `flatten-maven-plugin` pinned to 1.7.3, because newer versions fail on this build;
- `RawTraceLinksIT` writes the recovered doc-model and doc-code links of every project;
- `aggregator-pom.xml` builds `core` and `tlr` in one reactor.

Temperature (1.0), seed, prompts, and the pipeline (`ArtemisInTransArC`) are unchanged.

`artemis/ner/` is `named-architecture-entity-recognition` at commit `74ccb33`, the last 1.0.0-SNAPSHOT version, which TAAS25 depends on but which was never published. Its parent POM is changed from the unpublished 2.0.0-SNAPSHOT to the released 2.0.1, and token-usage logging is added; the terra runs were made before the logging was added. Prompts are unchanged.

With JDK 21, Maven, and network access:

```bash
OPENAI_API_KEY=<your-key> ./run.sh terra        # runs 1 2 3
OPENAI_API_KEY=<your-key> ./run.sh luna 2       # run 2 only
```

The script installs `ner` and `taas25` into the local Maven repository, then writes `run-output/<model>/run<i>/{doc-model,doc-code}/<project>.csv`. Each run uses its own LLM cache directory, because ArTEMiS caches every prompt and a shared cache would replay the first run.

`recorded/<terra|luna>/run<1-3>/` holds the six runs the paper reports. The terra runs were made on 2026-08-27 and the luna runs on 2026-09-24. Their model responses were not kept, so these runs cannot be replayed offline. `logs/` holds their build and run logs, with local absolute paths replaced by `<taas25>`, `<ner>`, `<workspace>` and `<home>`.
