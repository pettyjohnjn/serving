#!/bin/bash
#SBATCH --job-name=qwen38-serve
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

# vLLM OpenAI-compatible endpoint on a DGX Spark GB10; the model comes from etc/models/.
#
# Design notes (see README.md for the full reasoning):
#   * NVFP4 weights  -> ~23 GB, native Blackwell FP4 tensor cores. Decode on this
#     box is memory-bandwidth bound (273 GB/s), so bytes-per-weight sets the speed.
#   * MTP self-speculation -> 65-82% per-position acceptance on real workloads.
#     Biggest single-user latency win available on this hardware.
#   * FP8 KV cache -> only 16 of 64 layers are full-attention (the rest are linear
#     attention with a constant-size state), so KV is cheap. FP8 halves both the
#     footprint and the per-step KV read, which directly speeds long-context decode.
#   * Model lives on globus3's LOCAL NVMe (/scratch). The inter-node link is 1 GbE,
#     so loading 23 GB over NFS would add ~3 min to every restart.

set -uo pipefail

SERVING_ROOT=${SERVING_ROOT:-$HOME/serving}

# ---- model profile ---------------------------------------------------------------
# Everything model-specific lives in etc/models/<profile>.env: the model path, the
# runtime that can load it, the tuned defaults and any extra flags/env. Sourced HERE,
# before the tunables below, so those become fallbacks rather than overrides -- which
# is what keeps the 27B path byte-identical to how it ran before profiles existed.
#   sbatch --export=ALL,MODEL_PROFILE=qwen38-flash-next bin/serve.sh
MODEL_PROFILE=${MODEL_PROFILE:-qwen38-27b}
_PROFILE=$SERVING_ROOT/etc/models/$MODEL_PROFILE.env
[ -r "$_PROFILE" ] || { echo "[$(date)] no such model profile: $_PROFILE"; exit 1; }
. "$_PROFILE" || { echo "[$(date)] model profile $MODEL_PROFILE refused to load"; exit 1; }
[ -n "${MODEL:-}" ] || { echo "[$(date)] profile $MODEL_PROFILE resolved no MODEL (weights missing?)"; exit 1; }
# Assert the profile contract. Without this a profile that forgets profile_args()
# still launches -- with every model-specific flag silently absent (no kv-cache-dtype,
# no tool-call-parser). A subtly wrong server that passes `doctor` is the worst
# outcome available here, so fail before binding anything.
[ -n "${SERVED_NAME:-}" ] || { echo "[$(date)] profile $MODEL_PROFILE sets no SERVED_NAME"; exit 1; }
declare -p VLLM_LAUNCH >/dev/null 2>&1 || { echo "[$(date)] profile $MODEL_PROFILE sets no VLLM_LAUNCH"; exit 1; }
for _f in profile_args profile_env; do
    declare -F "$_f" >/dev/null || { echo "[$(date)] profile $MODEL_PROFILE defines no $_f()"; exit 1; }
done
echo "[$(date)] profile=$MODEL_PROFILE model=$MODEL served-as=$SERVED_NAME"

MODEL=${MODEL:-/scratch/models/Qwen3.8-27B-NVFP4}
PORT=${PORT:-8000}

# ---- where the endpoint appears, independent of which node runs the job ----------
# vLLM binds loopback only. Whatever node slurm picks then publishes itself back to the
# login node over a restricted reverse SSH tunnel, so the address clients use is ALWAYS
# globus1's 127.0.0.1:8000. Two consequences:
#   * moving the job between globus2/globus3 changes nothing for users;
#   * the model port is never exposed on a public interface at all (these nodes have
#     public IPs, and an unauthenticated /tokenize on them was a live SSRF vector).
BIND_HOST=${BIND_HOST:-127.0.0.1}
PUBLISH_HOST=${PUBLISH_HOST:-globus1}
# 8005, not 8000: the reverse tunnel lands on an INTERNAL port; bin/fair-proxy.py owns
# 8000 (what clients tunnel to) and enforces per-user concurrency caps in between.
PUBLISH_PORT=${PUBLISH_PORT:-8005}
CLIENT_PORT=${CLIENT_PORT:-8000}
TUNNEL_KEY=${TUNNEL_KEY:-$HOME/.ssh/id_llm_tunnel}

# ---- tunables (override at submit time: sbatch --export=ALL,MAX_SEQS=24 ...) ----
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
# 32, not 16. Aggregate throughput keeps climbing past the old cap (203 -> 323 tok/s,
# +59%) and memory is nowhere near binding: 32 sequences at 32K context is well inside the
# 2M-token KV pool. The cost is per-request latency (16.4 -> 11.2 tok/s at full load),
# which is the right trade for batch/agentic work. Lower it to 8-16 if you ever want the
# endpoint tuned for interactive use instead.
MAX_SEQS=${MAX_SEQS:-32}
# Measured optimum. Speculation is a LARGE win at every concurrency tested -- it does not
# invert under load, because the machine stays memory-bandwidth-bound even at batch 32.
# Aggregate tok/s at MAX_SEQS=32:
#            conc:      8     16     24     32
#   SPEC_TOKENS=0:     79    141    189    227
#   SPEC_TOKENS=1:    118    203    257    309
#   SPEC_TOKENS=3:    139    230    280    323   <- chosen
# Turning speculation off costs 42-76% of aggregate throughput. SPEC_TOKENS=0 exists only
# as a diagnostic.
#
# DO NOT SET SPEC_TOKENS=4. It is not merely slower -- it crashes the engine:
#   torch.AcceleratorError: CUDA error: an illegal memory access was encountered
# reproduced on job 544, which then 500s every request. Presumably an out-of-bounds in the
# MTP/GDN decode path at 4 draft tokens on sm120. 3 is extensively exercised (several full
# benchmark sweeps, smoke suite, 78K-token needle tests) with no issue.
SPEC_TOKENS=${SPEC_TOKENS:-3}
# 2048, not 8192. This is the prefill chunk size, and prefill here is slow (~900-1300
# tok/s), so chunk size is the unit of head-of-line blocking: at 8192 one user's 130K cold
# prefill held everyone else's first token at ~23s (p50) for two minutes; at 2048 that is
# ~5s, and the big prefill itself measures no slower. Decode is unaffected (32 seqs x 4
# spec tokens = 128 per step, nowhere near the cap).
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
KV_DTYPE=${KV_DTYPE:-fp8}
# Pin the KV pool instead of deriving it from --gpu-memory-utilization. On unified
# memory, "GPU total" is the whole 119.6 GiB machine and vLLM's headroom check falls
# back to host-wide /proc/meminfo, which cannot see the slurm --mem cgroup at all.
# A fixed byte count is identical across every requeue and skips the profiling pass.
# 71 GiB = ~2.0M tokens at fp8 (37.2 KiB/token measured, job 611). Sized to leave
# ~10 GiB host headroom: weights ~22 GiB + this pool + ~16 GiB runtime on the
# 119.6 GiB unified box. Do not push further without remembering the OOM killer
# shares this memory (FlashInfer JIT rebuilds after upgrades cost several GiB).
KV_CACHE_BYTES=${KV_CACHE_BYTES:-76235669504}
# MUST still be set explicitly. kv_cache_memory_bytes ignores gpu_memory_utilization when
# *sizing* the pool, but v1/worker/utils.py:414 still gates startup on
#   free_memory >= total_memory * gpu_memory_utilization
# and this vLLM's default is 0.92 (config/cache.py:68), i.e. it demands 110 of 119.6 GiB
# free. That fails whenever anything else transiently holds memory -- e.g. restarting
# while the previous job is still releasing. 0.80 leaves ~12 GiB of slack.
GPU_UTIL=${GPU_UTIL:-0.80}
# The chat template's own default is 'xhigh', which on this box means minutes of thinking
# before any answer. Clients override per-request via
# chat_template_kwargs={"reasoning_effort":...} or {"enable_thinking":false}.
# Note the template treats 'high' and 'xhigh' as the same thing, and 'medium' injects no
# directive at all (it is the bare template).
REASONING_EFFORT=${REASONING_EFFORT:-medium}
# split  = trace in the `reasoning` field, content holds only the answer.
# inline = no --reasoning-parser; trace stays in `content` so chat UIs stream it live.
# Keep split. inline leaks reasoning prose (and a stray </think>) into `content` on tool
# turns, which tool-calling clients mis-parse -- see CLIENTS.md.
REASONING_MODE=${REASONING_MODE:-split}
# Seconds vLLM is given to finish in-flight requests on SIGTERM. vLLM's own default is
# 0, which entrypoints/launcher.py maps to "abort" -- in-flight requests are dropped.
DRAIN_SECONDS=${DRAIN_SECONDS:-240}

mkdir -p "$SERVING_ROOT/logs"
chmod 700 "$SERVING_ROOT/logs" 2>/dev/null

# ---- authorisation ----
# Default policy: SSH IS the authorisation. The endpoint binds loopback on the compute node
# and is published only to globus1's loopback, so the sole way to reach it is an SSH session
# to globus1 -- and this cluster is publickey-only (AuthenticationMethods publickey,
# AllowGroups labusers labadmins). Anyone whose key is in the trusted list can therefore use
# the model, and nobody else can. A separate API key would be a second secret to distribute
# and rotate that gates exactly the same set of people.
#
# Trade-off, deliberately accepted: without per-user keys there is no per-request attribution,
# and any process already running on globus1 can reach the port without its own SSH session.
# Both are bounded by the same trust set (people who can get a shell on globus1).
#
# Set REQUIRE_API_KEY=1 to layer keys back on (e.g. to attribute load, or to hand access to
# someone who should NOT be able to reach globus1 at all):
#   sbatch --export=ALL,REQUIRE_API_KEY=1 bin/serve.sh
REQUIRE_API_KEY=${REQUIRE_API_KEY:-0}
KEYCONF="$SERVING_ROOT/etc/vllm-keys.yaml"
KEY_ARGS=()

# SPEC_TOKENS=0 disables speculative decoding entirely -- a diagnostic only. Measurements
# above show speculation is a large win at every concurrency tested; see the table there.
REASON_ARGS=()
if [ "$REASONING_MODE" = "split" ]; then
    REASON_ARGS=(--reasoning-parser qwen3)
fi

SPEC_ARGS=()
if [ "${SPEC_TOKENS:-0}" -gt 0 ] 2>/dev/null; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"${SPEC_METHOD:-qwen3_5_mtp}\",\"num_speculative_tokens\":$SPEC_TOKENS${SPEC_EXTRA:-}}")
fi
if [ "$REQUIRE_API_KEY" = "1" ]; then
    # Via a --config file, never argv: /proc/<pid>/cmdline is world readable and this
    # partition is not exclusive, so another account could co-schedule here and read it.
    umask 077
    {
        echo "# generated by serve.sh -- do not edit; edit etc/keys.env instead"
        echo "api-key:"
        grep -E '^KEY_' "$SERVING_ROOT/etc/keys.env" | cut -d= -f2- | sed 's/^/  - /'
    } > "$KEYCONF"
    chmod 600 "$KEYCONF"
    KEY_ARGS=(--config "$KEYCONF")
    echo "[$(date)] auth: API key required ($(grep -c '^KEY_' "$SERVING_ROOT/etc/keys.env") keys loaded)"
else
    rm -f "$KEYCONF"
    echo "[$(date)] auth: SSH-only (no API key). Reachable solely via an SSH session to $PUBLISH_HOST."
fi

export HF_HOME=/scratch/hf-home
export VLLM_LOGGING_LEVEL=INFO
export OMP_NUM_THREADS=8
# Blackwell/GB10: keep allocator from fragmenting the unified memory pool.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Do not follow redirects when fetching remote media (defence in depth alongside
# --allowed-media-domains below).
export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0

# Privacy: vLLM ships anonymous usage telemetry to https://stats.vllm.ai by DEFAULT
# (envs.py: VLLM_NO_USAGE_STATS=False). It carries no prompts or completions -- it is
# hardware/config metadata (GPU and CPU model, exact kernel version, model architecture,
# quantization, context length, plus a persistent UUID) posted on every engine start.
# For a lab endpoint whose very existence may be worth not advertising, that is a needless
# outbound disclosure, so both switches are off. Prior transmissions are recorded in
# ~/.config/vllm/usage_stats.json if you want to see exactly what left.
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

# FlashInfer JIT-compiles attention kernels at startup and shells out to `ninja` and `nvcc`.
# Calling venv2/bin/vllm directly does not put venv2/bin on PATH, so ninja must be added
# explicitly or the engine dies with FileNotFoundError: 'ninja' during memory profiling.
export PATH="${VLLM_PATH_PREPEND:+$VLLM_PATH_PREPEND:}/usr/local/cuda/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# Bound JIT build parallelism. ninja otherwise launches ~nproc nvcc processes, and each
# CUTLASS FP4 GEMM unit costs a few GB of RAM. On unified memory the model weights are
# already charged to the same cgroup, so an unbounded build gets OOM-killed (exit 137).
export MAX_JOBS=${MAX_JOBS:-4}
export NVCC_THREADS=${NVCC_THREADS:-2}

# Keep JIT/compile caches on globus3's local NVMe, not on 1 GbE NFS.
export TRITON_CACHE_DIR=/scratch/jit-cache/triton
export VLLM_CACHE_ROOT=/scratch/jit-cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/scratch/jit-cache/inductor
mkdir -p /scratch/jit-cache/{triton,vllm,inductor}

NODE_IP=$(ip -4 addr show enP7s7 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
NODE_IP=${NODE_IP:-$(hostname -I | awk '{print $1}')}

# ---- preflight: can we actually publish to the login node? -------------------------
# Done BEFORE binding, because binding loopback without a working tunnel would leave the
# endpoint reachable by nobody. The forced command on the key exits 1; only ssh's own
# 255 means the connection or authentication failed.
if [ "${DRY_RUN:-0}" != 1 ] && [ "$(hostname -s)" != "$PUBLISH_HOST" ]; then
    PUBLISH_OK=0
    if [ -f "$TUNNEL_KEY" ]; then
        ssh -i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o IdentityAgent=none \
            -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
            -o ControlPath=none "$PUBLISH_HOST" true >/dev/null 2>&1
        [ $? -ne 255 ] && PUBLISH_OK=1
    fi
    if [ "$PUBLISH_OK" = "0" ]; then
        # FAIL CLOSED. The old behaviour here was to fall back to BIND_HOST=0.0.0.0 "so the
        # endpoint stays usable". That is unsafe now that REQUIRE_API_KEY defaults to 0:
        # the combination would put a completely unauthenticated vLLM -- including the
        # /tokenize SSRF surface -- on a public university IP, and because the publisher
        # retry loop then reconnects anyway, health would go green and nothing would ever
        # report it. An endpoint nobody can reach is a much cheaper failure than that.
        echo "=================================================================="
        echo " FATAL: cannot publish to $PUBLISH_HOST (ssh preflight failed)."
        echo "   $TUNNEL_KEY is missing, not authorised, or the host key changed."
        echo "   Fix with:  bash $SERVING_ROOT/bin/setup_reverse_tunnel.sh"
        echo "   Then:      serving start"
        echo
        echo " Refusing to start rather than binding a public interface."
        echo " Override deliberately (and only behind a firewall) with:"
        echo "   sbatch --export=ALL,ALLOW_PUBLIC_BIND=1,REQUIRE_API_KEY=1 bin/serve.sh"
        echo "=================================================================="
        if [ "${ALLOW_PUBLIC_BIND:-0}" = "1" ]; then
            [ "$REQUIRE_API_KEY" = "1" ] || { echo "refusing ALLOW_PUBLIC_BIND without REQUIRE_API_KEY=1"; exit 1; }
            echo "ALLOW_PUBLIC_BIND=1 set and API keys required -- binding 0.0.0.0."
            BIND_HOST=0.0.0.0
        else
            exit 1
        fi
    else
        echo "[$(date)] publish preflight OK -> $PUBLISH_HOST"
    fi
fi

echo "=================================================================="
echo " $SERVED_NAME endpoint  (profile: $MODEL_PROFILE)"
echo " job=$SLURM_JOB_ID  host=$(hostname)  ip=$NODE_IP  port=$PORT"
echo " model=$MODEL"
echo " max_len=$MAX_MODEL_LEN  max_seqs=$MAX_SEQS  kv_bytes=$KV_CACHE_BYTES util=$GPU_UTIL"
echo " kv_dtype=$KV_DTYPE  spec_tokens=$SPEC_TOKENS  effort=$REASONING_EFFORT"
echo " started=$(date)"
echo "=================================================================="

cat > "$SERVING_ROOT/etc/endpoint.json" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "compute_node": "$(hostname)",
  "bind": "$BIND_HOST:$PORT",
  "published_at": "$PUBLISH_HOST:127.0.0.1:$PUBLISH_PORT",
  "client_base_url": "http://localhost:$CLIENT_PORT/v1",
  "client_tunnel": "ssh -N -L $CLIENT_PORT:127.0.0.1:$CLIENT_PORT $PUBLISH_HOST",
  "model": "$SERVED_NAME",
  "profile": "$MODEL_PROFILE",
  "started": "$(date -Is)"
}
EOF

# ---- publish to the login node ----------------------------------------------------
# Skipped when the job already landed on the login node: then loopback IS the address.
start_publisher() {
    if [ "$(hostname -s)" = "$PUBLISH_HOST" ]; then
        echo "[$(date)] running on $PUBLISH_HOST; endpoint already at 127.0.0.1:$PORT, no tunnel needed"
        return
    fi
    if [ ! -f "$TUNNEL_KEY" ]; then
        echo "[$(date)] WARNING: $TUNNEL_KEY missing -- endpoint will NOT be reachable from $PUBLISH_HOST."
        echo "[$(date)]          run: bash $SERVING_ROOT/bin/setup_reverse_tunnel.sh"
        return
    fi
    (
        backoff=2
        while kill -0 "$VLLM_PID" 2>/dev/null; do
            # -R binds 127.0.0.1 on the login node only (GatewayPorts is 'no' anyway).
            # ExitOnForwardFailure so a stale listener surfaces instead of silently no-op'ing.
            ssh -N -T \
                -i "$TUNNEL_KEY" \
                -o IdentitiesOnly=yes -o IdentityAgent=none \
                -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
                -o ControlPath=none \
                -o ExitOnForwardFailure=yes \
                -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
                -R "127.0.0.1:$PUBLISH_PORT:$BIND_HOST:$PORT" \
                "$PUBLISH_HOST" 2>&1 | sed "s/^/[publisher] /"
            kill -0 "$VLLM_PID" 2>/dev/null || break
            echo "[$(date)] publisher tunnel dropped; retrying in ${backoff}s"
            sleep "$backoff"
            backoff=$(( backoff < 30 ? backoff * 2 : 30 ))
        done
    ) &
    PUBLISHER_PID=$!
    echo "[$(date)] publishing $BIND_HOST:$PORT -> $PUBLISH_HOST:127.0.0.1:$PUBLISH_PORT (pid $PUBLISHER_PID)"
}

# ---- shutdown handling -------------------------------------------------------
# bash's `wait` is interruptible: a trapped signal makes it return 128+signo and it does
# NOT resume. So the trap must do the draining itself, and the wait must be re-armed in a
# loop, or the batch script exits while vLLM is still serving and slurmstepd hard-kills it.
_drain() {
    kill -TERM "$VLLM_PID" 2>/dev/null
    for _ in $(seq $((DRAIN_SECONDS + 20))); do
        kill -0 "$VLLM_PID" 2>/dev/null || return 0
        sleep 1
    done
}
# Requeue LAST. `scontrol requeue` on a RUNNING job makes slurmctld begin terminating it
# immediately, so requeueing first would destroy the grace period by itself.
_on_usr1() { echo "[$(date)] USR1 (approaching time limit) -- draining then requeueing"; _drain; scontrol requeue "$SLURM_JOB_ID"; }
# TERM must NOT requeue: that is `scancel`, i.e. the operator wants it stopped.
_on_term() { echo "[$(date)] TERM -- draining and exiting"; _drain; }
trap _on_usr1 USR1
trap _on_term TERM

# NOTE: no '#' comments inside this backslash-continued command. A comment line ends the
# logical line, silently truncating every flag after it -- which once launched vLLM with NO
# arguments at all, including losing --host and exposing the port publicly.
# --served-model-name takes a list and each entry aliases the same weights; keep it to one
# name, since advertising a '-nvfp4' variant made it look like a choice between two models.
profile_args
MODEL_ARGV=("$MODEL"); [ "${MODEL_PASS:-positional}" = flag ] && MODEL_ARGV=(--model "$MODEL")

VLLM_ARGV=(
    "${VLLM_LAUNCH[@]}" "${MODEL_ARGV[@]}"
    ${KEY_ARGS[@]+"${KEY_ARGS[@]}"}
    --served-model-name "$SERVED_NAME"
    --host "$BIND_HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_UTIL"
    --enable-prefix-caching
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
    --enable-auto-tool-choice
    --enable-force-include-usage
    # usage.prompt_tokens_details.cached_tokens on every response: the only per-request
    # view of whether the prefix cache served a prompt or it was re-prefilled. Off by
    # default in vLLM; both runtimes support it.
    --enable-prompt-tokens-details
    ${REASON_ARGS[@]+"${REASON_ARGS[@]}"}
    ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"}
    --shutdown-timeout "$DRAIN_SECONDS"
    --allowed-media-domains blocked.invalid
    --disable-fastapi-docs
    --allowed-origins '[]'
)

# DRY_RUN=1 prints the exact argv and exits. Used to prove a change to the profile
# machinery leaves the 27B command line unchanged, without touching production.
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%s\n' "${VLLM_ARGV[@]}"; exit 0; fi

profile_env
"${VLLM_ARGV[@]}" &

VLLM_PID=$!
PUBLISHER_PID=""
start_publisher

# ---- node stats for the user-facing stats page -------------------------------
# The stats page runs on globus1 but GPU/host load is only visible here, so write a
# small JSON to the (NFS-shared) logs dir every 10s. nvidia-smi memory reads [N/A] on
# GB10 -- memory is unified, so host RAM *is* GPU memory; report /proc/meminfo.
(
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        awk -v t="$(date +%s)" -v gpu="${gpu:-null}" \
            '/MemTotal/{tot=$2} /MemAvailable/{av=$2}
             END{
               getline load < "/proc/loadavg"; split(load,l," ")
               printf "{\"t\": %s, \"gpu_util\": %s, \"load1\": %s, \"mem_used_gib\": %.1f, \"mem_total_gib\": %.1f}\n", \
                 t, gpu, l[1], (tot-av)/1048576, tot/1048576
             }' /proc/meminfo \
            > "$SERVING_ROOT/logs/.node-stats.tmp" 2>/dev/null \
          && mv "$SERVING_ROOT/logs/.node-stats.tmp" "$SERVING_ROOT/logs/node-stats.json"
        sleep 10
    done
) &
NODESTATS_PID=$!

# ---- liveness watchdog -------------------------------------------------------
# A mid-run engine-core crash does NOT kill the API server: vLLM 0.27.1 only flags
# engine_dead, after which /health returns 503 forever while the process keeps running.
# Waiting on the PID would never notice, so poll /health instead.
(
    sleep 900   # allow for cold start (weights + compile + JIT)
    fails=0
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
        if [ "$code" = "200" ]; then fails=0; else
            fails=$((fails + 1))
            echo "[$(date)] health check failed (http ${code:-000}), streak=$fails"
            if [ "$fails" -ge 5 ]; then
                echo "[$(date)] endpoint unhealthy 5x -- killing server so the job exits nonzero"
                kill -TERM "$VLLM_PID" 2>/dev/null; sleep 30; kill -KILL "$VLLM_PID" 2>/dev/null
                break
            fi
        fi
        sleep 30
    done
) &
WATCHDOG_PID=$!

# Re-arm the wait: only stop when vLLM has actually gone away.
rc=0
while :; do
    wait "$VLLM_PID"; rc=$?
    { [ "$rc" -gt 128 ] && kill -0 "$VLLM_PID" 2>/dev/null; } || break
done
kill "$WATCHDOG_PID" "$NODESTATS_PID" 2>/dev/null
[ -n "$PUBLISHER_PID" ] && kill "$PUBLISHER_PID" 2>/dev/null
pkill -f "id_llm_tunnel.*${PUBLISH_HOST}" 2>/dev/null

# Capture rc BEFORE any command substitution -- $(date) resets $? and would always report 0.
NOW=$(date)
echo "[$NOW] vllm exited with code $rc"
exit "$rc"
