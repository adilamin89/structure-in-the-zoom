#!/bin/bash
# Encode the generality models' hidden states, one model at a time (set PYTHON to the interpreter with torch and transformers).
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=2
run() { echo "=== $* $(date)"; $PY -X faulthandler llm_encode.py "$@"; echo "EXIT $? $*"; }
run encoder google-bert/bert-base-uncased mps bert-base-uncased
run sentemb sentence-transformers/all-MiniLM-L6-v2 mps all-MiniLM-L6-v2
run causal state-spaces/mamba-130m-hf cpu fp32 mamba-130m-hf
run causal EleutherAI/gpt-neo-1.3B mps fp16 gpt-neo-1.3B
echo CHAIN_DONE
