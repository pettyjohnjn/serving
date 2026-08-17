#!/bin/bash
# Mirror the venv onto globus3's local NVMe.
#
# The venv lives on NFS /home over a 1 GbE link. Importing torch + vLLM touches many
# thousands of small files, and both the API server and the engine worker do it, so a
# cold start pays that cost twice. Copying ~10 GB once (~2 min) makes every subsequent
# restart substantially faster.
#
# Usage:  sbatch bin/mirror_venv.sh     then set VENV=/scratch/venv-qwen38 for serve.sh
#SBATCH --job-name=mirror-venv
#SBATCH --nodelist=globus3
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=mirror-%j.out

set -euo pipefail
SRC=$HOME/serving/venv2
DST=/scratch/venv-qwen38

echo "[$(date)] mirroring $SRC -> $DST on $(hostname)"
mkdir -p "$DST"
rsync -a --delete "$SRC/" "$DST/"

# A venv records its own absolute path; rewrite it so the copy is self-consistent.
sed -i "s|$SRC|$DST|g" "$DST/pyvenv.cfg"
for f in "$DST"/bin/*; do
    [ -f "$f" ] && head -c2 "$f" 2>/dev/null | grep -q '#!' && sed -i "1s|$SRC|$DST|" "$f"
done

echo "[$(date)] done: $(du -sh "$DST" | cut -f1)"
"$DST/bin/python" -c "import vllm; print('mirrored vllm', vllm.__version__)"
