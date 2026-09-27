# CLAUDE.md

## What s126 is

`s_linker126` is the paper arm. It implements greedy whole-name span
ownership, discard of each residual multi-component surface, and an
exact-antecedent contract on coreference resolutions: a resolution whose
cited antecedent sentence does not write the component's name *as a name*
is not put to the judge. It reuses `_written_as`, already computed for
every union-judging case, so the contract adds no new computation.

`s_linker126.py` is a **standalone file** — it imports no other
`s_linkerNNN` module at runtime, only `core/`, `pcm_parser{,_v2}.py`,
`llm_client.py`, `helper_v3.py`, `linker_infra.py`.

## Active Surface

- `run_ablation.py` — ablation runner; registry holds `s_linker126`
  and `s_linker126_noknow` (RQ4 knowledge A/B, same module, `no_knowledge=True`).
  `python run_ablation.py --list-variants` prints both.
- `s_linker126.py` — the paper arm, standalone.
- `core/`, `llm_client.py`, `pcm_parser{,_v2}.py`, `helper_v3.py` — shared
  runtime.
- `linker_infra.py` — linker plumbing: `TracingLLMClient`, `ask_json`,
  checkpoint/log/metrics writers, batching and log views. No prompts,
  rule constants, or scans — those belong in the variant file.
- `pilot/` — s126 validation: `test_s126.py` (24 checks),
  `coref_exact_pilots.py` (fixture loader), `reading_pilots.py`
  (benchmark/gold-loading helpers), `score_runs.py`, `ab_stats.py`,
  and the two E2E runners `run_s126_e2e.sh` / `run_s126_e2e_noknow.sh`.

## Build & Run

```bash
pip install -e ".[openai]"
python run_ablation.py --list-variants
python run_ablation.py --variants s_linker126 --datasets mediastore
```

The host provides the OpenAI credential as **`OAI_KEY`**, not
`OPENAI_API_KEY`. Every OpenAI-backed command must map `OAI_KEY` into it
inline, in the process environment only:

```bash
OPENAI_API_KEY="$OAI_KEY" python run_ablation.py ...
```

Full five-project E2E form:

```bash
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/<run>/phase_states \
LLM_LOG_DIR=../results/<run>/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker126 \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/<run>
```

Never write either credential value to `.env`, logs, results, or tracked
files. Default benchmarking backend is set in `.env` (`LLM_BACKEND=openai`,
`gpt-5.4`). `.env` is untracked.
