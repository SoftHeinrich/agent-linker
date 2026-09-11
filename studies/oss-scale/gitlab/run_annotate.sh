#!/usr/bin/env bash
# Two-family annotation of the GitLab document; all calls cached.
set -u
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
export OPENAI_API_KEY="$OAI_KEY" OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none
$PY annotate.py --view sentence --backend openai --model gpt-5.6-terra
$PY annotate.py --view sentence --backend openai --model gpt-5.6-terra --salt r2
$PY annotate.py --view sentence --backend openai --model gpt-5.6-terra --salt r3
$PY annotate.py --view component --backend openai --model gpt-5.6-terra
echo TERRA DONE
