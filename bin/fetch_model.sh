#!/bin/bash
#SBATCH --job-name=fetch-qwen38
#SBATCH --nodelist=globus3
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=fetch-%j.out
#SBATCH --error=fetch-%j.out

# Download the model straight onto globus3's LOCAL NVMe (/scratch/models).
# The inter-node link is only 1 GbE, so serving from NFS /home would add
# ~3 min of model-load time to every server start. Local disk avoids that.

set -euo pipefail

REPO="${REPO:-unsloth/Qwen3.8-27B-NVFP4}"
DEST="${DEST:-/scratch/models/$(basename "$REPO")}"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/scratch/hf-home

mkdir -p "$DEST" "$HF_HOME"

echo "[$(date)] host=$(hostname) repo=$REPO dest=$DEST"
df -h /scratch | tail -1

$HOME/serving/venv2/bin/hf download "$REPO" \
    --local-dir "$DEST" \
    --max-workers 8

echo "[$(date)] download complete"
du -sh "$DEST"
ls -la "$DEST"
