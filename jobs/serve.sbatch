#!/bin/bash
# The model engine: Qwen3.8-Flash-Next (NVFP4 experts + FP8 dense layers) on two GB10 nodes, tensor parallel over
# RoCE, vLLM. Submitted by `bin/serving start` (which passes -N2 -w $ENGINE_NODES -p $ENGINE_PARTITION -o ...);
# requeues itself at the 2-day wall limit. Needs `tools/flash-next/bin/fn build` (+ tp2 stage) on both nodes.
#
# Defaults are the validated configuration (DESIGN.md): bf16 KV 50 GiB per GPU (3.44M tokens), 32 slots, n-gram
# table staged from NVMe, FULL_DECODE_ONLY CUDA graphs with torch.compile off, MTP2, 4096-token prefill chunks,
# prefix caching. Tuning knobs (exported into the job): LAYOUT KV_DTYPE KV_BYTES SEQS CHUNK CTX PLE_MODE GRAPHS
# MTP OFFLOAD_DIR OFFLOAD_CPU_BYTES PUBLISH.
#SBATCH --job-name=qwen38-serve
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
set -uo pipefail
: "${SERVING_ROOT:?submit through bin/serving (SERVING_ROOT unset)}"
. "$SERVING_ROOT/etc/site.env"

LAYOUT=${LAYOUT:-tp2}
KV_DTYPE=${KV_DTYPE:-auto}                    # auto = bf16. fp8_e4m3 doubles the pool but loses hard retrieval
KV_BYTES=${KV_BYTES:-${ENGINE_KV_BYTES:-53687091200}}   # per GPU; earlyoom (MemAvailable < ~4.8 GiB) is the ceiling
SEQS=${SEQS:-${ENGINE_SEQS:-32}}
CHUNK=${CHUNK:-4096}                          # never 4096 together with GRAPHS=piecewise: decode collapses
CTX=${CTX:-262144}
PORT=${PORT:-8100}                            # head-node loopback only
GPU_UTIL=${GPU_UTIL:-0.80}
#   PLE_MODE: staged (n-gram table on NVMe, rows gathered before each step into a fixed GPU buffer; allows full
#             decode graphs) | mmap (rows read inside the step; needs piecewise graphs) | resident (each rank holds
#             its 23.8 GiB slice in GPU memory: ~30% faster prefill, half the KV pool)
#   GRAPHS:   nocompile (FULL_DECODE_ONLY, compile off, model runner v2) | piecewise (compile + piecewise graphs)
PLE_MODE=${PLE_MODE:-staged}; GRAPHS=${GRAPHS:-nocompile}; MTP=${MTP:-2}
# NVMe KV offload: blocked upstream for this model as of the pinned tree (DESIGN.md); kept for when it lands.
OFFLOAD_DIR=${OFFLOAD_DIR:-}; OFFLOAD_CPU_BYTES=${OFFLOAD_CPU_BYTES:-4294967296}
PUBLISH=${PUBLISH:-1}
SERVED=${SERVED_NAME:-qwen3.8-flash-next}
SP=$FN_SCRATCH/img/usr/local/lib/python3.12/dist-packages
BASE=$(ls -d "$FN_HF"/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ 2>/dev/null | grep -v -- -fp8hybrid | head -1)
SNAP=${BASE%/}-fp8hybrid
case "$LAYOUT" in
  single) TP=1 ;;
  # TP2 has its own hybrid checkpoint: the shared expert (640 wide) cannot be 128-block FP8 at 320 per rank.
  tp2)    TP=2; SNAP=${SNAP}-tp2; [ "${SLURM_JOB_NUM_NODES:-1}" = 2 ] || { echo "!! tp2 needs -N2"; exit 1; } ;;
  *) echo "!! LAYOUT must be single|tp2"; exit 1 ;;
esac
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo ">> $(date -Is) layout=$LAYOUT tp=$TP nodes=$SLURM_JOB_NODELIST head=$HEAD kv=$KV_DTYPE kv_bytes=$KV_BYTES seqs=$SEQS chunk=$CHUNK ple=$PLE_MODE graphs=$GRAPHS mtp=$MTP"

# Everything per node; srun runs it once on each node.
node_main () {
  set -uo pipefail
  local RANK=${SLURM_NODEID:-0} ME; ME=$(hostname)
  [ -f "$SNAP/.prepared" ] || { echo "!! [$ME] checkpoint not prepared: $SNAP (fn build)"; exit 1; }
  grep -q "^done. worst per-tensor" "$SNAP/fp8_convert.log" 2>/dev/null || { echo "!! [$ME] $SNAP: no completed conversion"; exit 1; }
  local IP; IP=$(ip -4 -o addr show "$ENGINE_RDMA_IF" | awk '{print $4}' | cut -d/ -f1)
  export PYTHONPATH=$SP PYTHONNOUSERSITE=1 CUDA_HOME=/usr/local/cuda
  [ -d "$SP/nvidia_cutlass_dsl/dsl_packages" ] && export PYTHONPATH="$PYTHONPATH:$SP/nvidia_cutlass_dsl/dsl_packages"
  export HF_HOME=$FN_HF HF_HUB_OFFLINE=1
  export VLLM_FP8_HYBRID=1 VLLM_USE_DEEP_GEMM=0 VLLM_QSA_EXACT_TOPK=0 VLLM_USE_FLASHINFER_SAMPLER=1
  # The deterministic top-k kernel is required: without it, concurrent cold prefills wedge the engine.
  export VLLM_QSA_DET_TOPK=1 VLLM_QSA_DET_LIB=$FN_SCRATCH/kernel-det/_C_det.so
  case "$PLE_MODE" in
    mmap)     export QWEN4EXP_PLE_MMAP=1 QWEN4EXP_PLE_MMAP_THREADS=64 QWEN4EXP_PLE_MODEL_DIR="$SNAP" ;;
    staged)   export QWEN4EXP_PLE_MMAP=1 QWEN4EXP_PLE_STAGED=1 QWEN4EXP_PLE_MMAP_THREADS=64 QWEN4EXP_PLE_MODEL_DIR="$SNAP" ;;
    resident) export QWEN4EXP_PLE_MMAP=1 QWEN4EXP_PLE_RESIDENT=1 QWEN4EXP_PLE_MMAP_THREADS=64 QWEN4EXP_PLE_MODEL_DIR="$SNAP" ;;
    *) echo "!! PLE_MODE must be staged|mmap|resident"; exit 1 ;;
  esac
  [ "$GRAPHS" = nocompile ] && export VLLM_USE_V2_MODEL_RUNNER=1
  export CUTE_DSL_ARCH=sm_121a TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a \
         FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export VLLM_CACHE_ROOT=$FN_SCRATCH/.cache/vllm-$LAYOUT-kv$KV_DTYPE-$PLE_MODE-$GRAPHS-mtp$MTP
  mkdir -p "$VLLM_CACHE_ROOT"
  if [ "$TP" = 2 ]; then
    # NCCL data over IB verbs on the RoCE HCAs; bootstrap, gloo and vLLM's control plane over TCP on the RoCE netdev.
    export NCCL_IB_HCA=${NCCL_IB_HCA:-$ENGINE_RDMA_HCAS} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$ENGINE_RDMA_IF}
    export GLOO_SOCKET_IFNAME=$ENGINE_RDMA_IF VLLM_HOST_IP=$IP
    ulimit -l unlimited 2>/dev/null || echo "[$ME] memlock stays at $(ulimit -l) KB"
  fi
  # Page-cache prewarm of the resident weight set (prevents cgroup page-cache starvation; does not speed boot).
  find -L "$SNAP" -maxdepth 1 \( -name 'layer-*.safetensors' -o -name 'model-bf16-*.safetensors' \) \
    ! -name '*.bak' -print0 | xargs -0 -P 4 -n 1 cat > /dev/null 2>/dev/null
  local SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::qwen4_exp_compute_ple_ngram_ids","vllm::qwen4_exp_ple_mmap_gather"]'
  local ARGS=(
    "$SNAP" --served-model-name "$SERVED"
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_UTIL"
    --kv-cache-memory "$KV_BYTES" --kv-cache-dtype "$KV_DTYPE" --load-format safetensors
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens "$CHUNK"
    --no-enable-flashinfer-autotune
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
    --limit-mm-per-prompt '{"image":4,"video":0}'
    --default-chat-template-kwargs '{"reasoning_effort":"medium"}'
    # disable_eagle_block_drop: without it this vLLM disables prefix reuse for the Mamba groups under MTP.
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"disable_eagle_block_drop\":true}"
  )
  if [ -n "$OFFLOAD_DIR" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF      # vLLM refuses a KV connector with expandable segments
    mkdir -p "$OFFLOAD_DIR" && chmod 700 "$OFFLOAD_DIR"
    ARGS+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$OFFLOAD_CPU_BYTES,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$OFFLOAD_DIR\"}]}}")
  fi
  case "$GRAPHS" in
    piecewise) ARGS+=(-cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="$SPLIT") ;;
    nocompile) ARGS+=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
    *) echo "!! GRAPHS must be nocompile|piecewise"; exit 1 ;;
  esac
  if [ "$TP" = 2 ]; then
    ARGS+=(--tensor-parallel-size 2 --distributed-executor-backend mp
           --nnodes 2 --node-rank "$RANK" --master-addr "$ENGINE_HEAD_ADDR" --master-port 29600)
    [ "$RANK" != 0 ] && ARGS+=(--headless)
  fi
  # The API binds the head node's loopback only; never 0.0.0.0 (vLLM leaves /tokenize etc. unauthenticated).
  [ "$RANK" = 0 ] && ARGS+=(--host 127.0.0.1 --port "$PORT")
  # Publish: reverse tunnel to the login node's 127.0.0.1:8005, where fair-proxy picks it up. The key may only
  # listen on that port (bin/serving install). Strict host key checking; retries forever.
  if [ "$RANK" = 0 ] && [ "$PUBLISH" = 1 ]; then
    ( while true; do
        ssh -N -T -i "$HOME/.ssh/id_llm_tunnel" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes \
            -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 \
            -o ServerAliveCountMax=3 -o ControlPath=none -R "127.0.0.1:8005:127.0.0.1:$PORT" "$LOGIN_HOST" 2>&1 \
          | sed -u "s/^/[publisher] /"
        echo "[publisher] $(date -Is) tunnel down; retry in 15 s"; sleep 15
      done ) &
  fi
  if [ "$RANK" = 0 ]; then
    local VPID=$$                      # exec below keeps this PID, so the helpers can watch vLLM itself
    printf '{"job_id":"%s","nodes":"%s","model":"%s","profile":"flash-next-tp2","started":"%s"}\n' \
      "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SERVED" "$(date -Is)" > "$SERVING_ROOT/etc/endpoint.json"
    # Node stats for the status page (GB10 memory is unified: /proc/meminfo is GPU memory too).
    ( while kill -0 "$VPID" 2>/dev/null; do
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        awk -v t="$(date +%s)" -v gpu="${gpu:-null}" '/MemTotal/{tot=$2} /MemAvailable/{av=$2}
          END{getline load < "/proc/loadavg"; split(load,l," ")
              printf "{\"t\": %s, \"gpu_util\": %s, \"load1\": %s, \"mem_used_gib\": %.1f, \"mem_total_gib\": %.1f}\n",
                     t, gpu, l[1], (tot-av)/1048576, tot/1048576}' /proc/meminfo \
          > "$SERVING_ROOT/logs/.node-stats.tmp" 2>/dev/null && mv "$SERVING_ROOT/logs/.node-stats.tmp" "$SERVING_ROOT/logs/node-stats.json"
        sleep 10
      done ) &
    # Liveness watchdog: an engine-core crash can leave the API server up with /health at 503 forever. After 5
    # consecutive failures kill vLLM, so the job exits nonzero and `bin/serving supervise` resubmits it.
    ( sleep 1200; fails=0
      while kill -0 "$VPID" 2>/dev/null; do
        code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
        if [ "$code" = 200 ]; then fails=0; else
          fails=$((fails + 1)); echo "[watchdog] $(date -Is) /health ${code:-000} (streak $fails)"
          [ "$fails" -ge 5 ] && { echo "[watchdog] unhealthy 5x; killing vLLM"; kill -TERM "$VPID"; sleep 30; kill -KILL "$VPID"; break; }
        fi
        sleep 30
      done ) &
  fi
  echo ">> [$ME] rank=$RANK ip=$IP launching vLLM $(date -Is)"
  exec python3.12 -m vllm.entrypoints.cli.main serve "${ARGS[@]}"
}
export -f node_main
export LAYOUT TP KV_DTYPE KV_BYTES SEQS CHUNK CTX PORT GPU_UTIL FN_SCRATCH FN_HF SP SNAP HEAD PLE_MODE GRAPHS MTP \
       OFFLOAD_DIR OFFLOAD_CPU_BYTES PUBLISH SERVED ENGINE_RDMA_IF ENGINE_RDMA_HCAS ENGINE_HEAD_ADDR LOGIN_HOST SERVING_ROOT

# Drain on scancel (TERM) or the wall limit (USR1, then requeue so the endpoint renews itself).
_drain () {
  [ -n "${STEP_PID:-}" ] || return 0
  echo "[$(date -Is)] draining"; kill -TERM "$STEP_PID" 2>/dev/null
  for _ in $(seq 120); do kill -0 "$STEP_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$STEP_PID" 2>/dev/null; STEP_PID=
}
trap '_drain; scontrol requeue "$SLURM_JOB_ID"' USR1
trap '_drain; exit 0' TERM
trap '_drain' EXIT
srun --kill-on-bad-exit=1 --cpus-per-task=20 --mem=0 --label bash -c node_main &
STEP_PID=$!
wait "$STEP_PID"; RC=$?
[ "$RC" = 143 ] && RC=0
exit "$RC"
