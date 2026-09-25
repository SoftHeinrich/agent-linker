# S126 `written` label update — verification record

Date: 2026-09-17

## Change under test

The entity-judge evidence field `written` now uses the labels documented in
`paper/sections/approach.tex`:

| Label | Meaning |
| --- | --- |
| `exact` | the component's full catalog name is written as a name |
| `alias` | the document-established alternate form is written |
| `part` | one word of a multi-word component name is written |
| `qualified name` | the full catalog name occurs only inside a longer joined or dotted identifier |

The judge prompt defines these meanings explicitly.

## Fixed-input verification

Command, run from `approach/`:

```text
../.venv/bin/python -m py_compile src/llm_sad_sam/linkers/experimental/s_linker126.py
../.venv/bin/python pilot/test_s126.py
```

Result:

```text
34/34 checks passed
```

The check includes one fixed input for each of the four classifications and
checks that the prompt contains each label definition.

## Flex-tier E2E rerun

The requested runner configuration was:

```text
OPENAI_API_KEY="$OAI_KEY" STAMP=20260917credfix pilot/run_s126_e2e.sh terra 3
OPENAI_API_KEY="$OAI_KEY" STAMP=20260917credfix pilot/run_s126_e2e.sh luna 3
```

The actual shell command used a process-local fallback for the host credential
name (`${OAI_KEY:-${OAI_API_KEY:-}}`) because this environment exposed the
credential under `OAI_API_KEY`. The runner used:

```text
LLM_BACKEND=openai
OPENAI_MODEL_NAME=gpt-5.6-{terra,luna}
OPENAI_REASONING_EFFORT=none
OPENAI_SERVICE_TIER=flex
datasets=mediastore teammates teastore bigbluebutton jabref
variants=s_linker123,s_linker126 (alternating order per run)
runs=3 per backend
```

Result: blocked before the first dataset completed on both backends. Each
attempt reached the OpenAI client and failed during `phase_25_doc_extract`
with HTTP 429 `credit_balance_exhausted` (“You have no credits remaining”).
No E2E metrics were produced. Raw logs are retained in:

```text
results/greedymerge_e2e_terra_r{1,2,3}_20260917credfix.log
results/greedymerge_e2e_luna_r{1,2,3}_20260917credfix.log
```

The earlier pre-credential attempts are also retained under the corresponding
`*_20260917` directories; they failed because the documented virtualenv was
broken and the client could not import `httpx`.
