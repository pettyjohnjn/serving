#!/bin/bash
# Acceptance tests against the RUNNING engine job, run from the login node as the service account:
#   tools/validate/validate.sh [quick|full] [label]
# quick: decode ladder 1..32, prefix-cache reuse, 6-way cold-prefill corruption probe, 16 x ~120k stress with
#        MemAvailable sampled on both nodes (earlyoom fires at ~4.8 GiB; keep the minimum above ~7 GiB).
# full:  + 64k-context decode ladder, stall test (decode pauses during a 100k cold prefill), hard long-context eval.
# The engine binds the head node's loopback, so every client runs on the head node inside the job (srun --overlap).
# Results go to logs/validate-<label>/. Reference numbers: DESIGN.md.
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
. "$ROOT/etc/site.env"
MODE=${1:-quick}; LABEL=${2:-$(date +%F-%H%M)}
V=$ROOT/tools/validate; OUT=$ROOT/logs/validate-$LABEL; mkdir -p "$OUT"
J=$(squeue -h -u "$(id -un)" -n qwen38-serve -t R -o %i | head -1); [ -n "$J" ] || { echo "no running engine job"; exit 1; }
NODES=$(squeue -h -j "$J" -o %N); HEAD=$(scontrol show hostnames "$NODES" | head -1)
U=http://127.0.0.1:8100/v1; M=${SERVED_NAME:-qwen3.8-flash-next}
on () { local n=$1; shift; srun --jobid="$J" --overlap -w "$n" -n1 --mem=6G -c2 -t 120 \
          env FN_SCRATCH="$FN_SCRATCH" "$@" 2>&1 | grep -v '^srun:'; }
mem () { for n in $(scontrol show hostnames "$NODES"); do
           echo "MEM $n $(on "$n" awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)"; done; }
echo "== validate $MODE ($LABEL): job $J on $NODES $(date -Is)" | tee "$OUT/summary.txt"
grep -hE "GPU KV cache size|Model loading took" "$ROOT/logs/serve-$J.out" | sed 's/^.*INFO/INFO/' | sort -u | tee -a "$OUT/summary.txt"
on "$HEAD" python3 "$V/bench2.py" --url "$U/chat/completions" --model "$M" --ladder 1,2,4,8,16,24,32 --label "$LABEL" \
   --out "$OUT/ladder.json" > "$OUT/ladder.txt"
python3 - "$OUT/ladder.json" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys
def walk(o):
    for k, v in o.items():
        if isinstance(v, dict) and "decode_tok_s_per_stream" in v:
            print(f"LADDER {k:>3} streams: {v['decode_tok_s_per_stream']:5.1f} tok/s per stream, {v['agg_decode_tok_s']:6.1f} aggregate")
        elif isinstance(v, dict): walk(v)
walk(json.load(open(sys.argv[1])))
EOF
on "$HEAD" python3 "$V/prefix_test.py" http://127.0.0.1:8100 "$M" 15000 | tee "$OUT/prefix.txt" | grep -E "A-repeat:" | tee -a "$OUT/summary.txt"
on "$HEAD" python3 "$V/garbage_probe.py" "$U" "$M" 6 | tee "$OUT/probe.txt" | grep RESULT | tee -a "$OUT/summary.txt"
( while sleep 5; do mem; done ) > "$OUT/stress-mem.txt" & S=$!
on "$HEAD" python3 "$V/longctx_stress.py" "$U" "$M" 16 90000 | tee "$OUT/stress.txt" | grep "^STRESS" | tee -a "$OUT/summary.txt"
kill $S 2>/dev/null
for n in $(scontrol show hostnames "$NODES"); do
  echo "MIN MemAvailable $n $(grep "^MEM $n" "$OUT/stress-mem.txt" | awk '{print $3}' | sort -n | head -1) GiB"
done | tee -a "$OUT/summary.txt"
if [ "$MODE" = full ]; then
  on "$HEAD" python3 "$V/decode_longctx.py" --ladder 1,4,8,16,32 --ctx 80000 --label "$LABEL" --out "$OUT/decode64k.json" \
     | tee "$OUT/decode64k.txt" | grep DECODE | tee -a "$OUT/summary.txt"
  on "$HEAD" python3 "$V/stall_test.py" --streams 4 --ctx 100000 --label "$LABEL" | tee -a "$OUT/summary.txt"
  on "$HEAD" python3 "$V/longctx_eval.py" --hard --label "$LABEL" --out "$OUT/hard.json" > "$OUT/hard.txt"
  grep LONGCTX "$OUT/hard.txt" | tee -a "$OUT/summary.txt"
fi
curl -s http://127.0.0.1:8005/metrics | grep -E "^vllm:num_preemptions_total" | tee -a "$OUT/summary.txt"
echo "== done $(date -Is); results in $OUT" | tee -a "$OUT/summary.txt"
