#!/bin/bash
# Orchestrate the full CADA-Bench cross-VLM runs across 4 GPUs (2,3,4,5).
# 5 models, 4 GPUs: the 5th job waits for its GPU slot to free.
cd /home/yvvyee/data
PYBIN=/home/yvvyee/miniconda3/envs/eva/bin/python
ENTRIES=(
  "llava-1.5-7b-hf llava15"
  "llava-v1.6-mistral-7b-hf llavanext"
  "idefics2-8b idefics2"
  "InternVL3-8B-hf internvl3"
  "SmolVLM-Instruct smolvlm"
)
GPUS=(2 3 4 5)
declare -a PID
i=0
for e in "${ENTRIES[@]}"; do
  set -- $e
  model=$1; tag=$2
  slot=$((i % 4)); gpu=${GPUS[$slot]}
  if [ -n "${PID[$slot]}" ]; then wait "${PID[$slot]}"; fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=$gpu TAG=$tag MODEL_PATH=/home/yvvyee/data/$model OBJECT=clock \
    nohup "$PYBIN" cada_multi.py > "_dl_logs/full_$tag.log" 2>&1 &
  PID[$slot]=$!
  echo "launched $tag on GPU$gpu PID ${PID[$slot]} at $(date +%H:%M:%S)"
  i=$((i + 1))
done
wait
echo "ALL FULL RUNS COMPLETE at $(date +%H:%M:%S)"
