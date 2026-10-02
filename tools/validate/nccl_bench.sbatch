#!/bin/bash
# All-reduce latency/bandwidth between the two engine nodes over the RoCE links (needs both nodes free):
#   SERVING_ROOT=$PWD sbatch -N2 -w <nodeA>,<nodeB> -p <partition> -o logs/nccl-%j.out tools/validate/nccl_bench.sbatch
# Reference: 22-40 us per small all-reduce eager, 57-75 us inside CUDA graphs, ~21 GB/s for large messages.
#SBATCH -J nccl-bench
#SBATCH --ntasks-per-node=1 --cpus-per-task=4 --mem=16G --gres=gpu:1
#SBATCH -t 20
: "${SERVING_ROOT:?}"; . "$SERVING_ROOT/etc/site.env"
SP=$FN_SCRATCH/img/usr/local/lib/python3.12/dist-packages
export MASTER_ADDR=$ENGINE_HEAD_ADDR MASTER_PORT=29511
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$ENGINE_RDMA_IF} NCCL_IB_HCA=${NCCL_IB_HCA:-$ENGINE_RDMA_HCAS}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=INIT,NET
ulimit -l unlimited 2>/dev/null || echo "memlock stays at $(ulimit -l) KB"
srun --kill-on-bad-exit=1 bash -c "export RANK=\$SLURM_PROCID WORLD_SIZE=2 LOCAL_RANK=0
  PYTHONPATH=$SP PYTHONNOUSERSITE=1 timeout 300 python3.12 $SERVING_ROOT/tools/validate/nccl_bench.py"
