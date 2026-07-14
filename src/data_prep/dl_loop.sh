#!/bin/bash
# Multi-pass resume download for the CADA-Bench v1 image set. Each pass skips
# already-downloaded files, so transient S3-throttle failures converge over passes.
cd /home/yvvyee/data
PYBIN=/home/yvvyee/miniconda3/envs/eva/bin/python
for pass in 1 2 3 4 5; do
  echo "=== PASS $pass $(date +%H:%M:%S) ==="
  SPLIT=train OUT=/home/yvvyee/data/oi_images WORKERS=16 RETRY=3 \
    "$PYBIN" oi_download.py cadabench_v1_download_ids.txt
  echo "pass $pass done, count=$(ls oi_images/ | wc -l) $(date +%H:%M:%S)"
done
echo "DL_LOOP_DONE count=$(ls oi_images/ | wc -l)"
