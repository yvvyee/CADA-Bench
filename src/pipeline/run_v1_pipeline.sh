#!/bin/bash
# Autonomous CADA-Bench v1 pipeline: wait for image download to finish, run the
# 40-object plant/certify/attribute sweep across GPUs 0,2,3,4,5, then aggregate.
# Runs unattended on the server (nohup) for ~1.5 days.
cd /home/yvvyee/data
PYBIN=/home/yvvyee/miniconda3/envs/eva/bin/python

echo "[v1] $(date) waiting for download loop to finish"
while pgrep -f dl_loop.sh >/dev/null 2>&1; do sleep 120; done
echo "[v1] $(date) download done, images=$(ls oi_images/ | wc -l)"

echo "[v1] $(date) launching 40-object sweep"
MODEL_PATH=/home/yvvyee/data/Qwen2-VL-7B-Instruct N_CLEAN=5000 GPUS=0,2,3,4,5 \
  "$PYBIN" oi_sweep.py
echo "[v1] $(date) sweep complete"

echo "[v1] $(date) aggregating"
DATA_ROOT=/home/yvvyee/data "$PYBIN" aggregate_oi.py > cadabench_v1_results_stdout.txt 2>&1
echo "[v1] $(date) PIPELINE DONE"
