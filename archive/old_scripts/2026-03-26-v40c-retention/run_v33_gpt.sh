#!/bin/bash
# Run V33 on GPT-5.2
set -e
cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45
source .env
export OPENAI_API_KEY
export OPENAI_MODEL_NAME=gpt-5.2
export PHASE_CACHE_DIR=./results/phase_cache_gpt

python -c "
import sys, os
sys.path.insert(0, 'src')
os.environ['OPENAI_MODEL_NAME'] = 'gpt-5.2'

import run_ablation
from llm_sad_sam.llm_client import LLMBackend
run_ablation.BACKEND = LLMBackend.OPENAI
sys.argv = ['run_ablation.py', '--variants', 'v33']
run_ablation.main()
"
