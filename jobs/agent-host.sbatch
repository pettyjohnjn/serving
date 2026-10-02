#!/bin/bash
# Agent host: one Slurm job per person holds all of that person's Pi sessions. Submitted by the dashboard as the
# person's agent account (RUNAS_MODE / SUBMIT_MODE in etc/site.env), with everything it needs in its environment:
#   WB_USER WB_CONTROL WB_TOKEN    identity, the dashboard's control API, this job's token
#   SPARK_LLM_KEY MODEL_RELAY      the person's model key, fair-proxy's keyed listener (host:port)
#   AGENT_TOOLS AGENT_CLUSTER_IF AGENT_PI_PACKAGES AGENT_PI_PROFILE SERVED_NAME
# It never reads the service account's files: the runtime and workbench/supervisor.py come from $AGENT_TOOLS.
# Renews itself at the 2-day limit (sessions resume); ends after 30 idle minutes.
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
set -uo pipefail
umask 077
: "${WB_USER:?submit from the dashboard}" "${AGENT_TOOLS:?}" "${MODEL_RELAY:?}"
. "$AGENT_TOOLS/env.sh"
export HOST_IP=$(ip -4 -o addr show "$AGENT_CLUSTER_IF" | awk '{print $4}' | cut -d/ -f1)
mkdir -p "$HOME/agents"
echo ">> $(date -Is) agent host for $WB_USER as $(id -un) on $(hostname) job $SLURM_JOB_ID (${SLURM_CPUS_PER_TASK:-?} CPUs)"

# One-time Pi setup for this account: packages, the profile's roles + config, and the model entry pointed straight
# at fair-proxy's keyed listener with the person's key (no local tunnel; the profile defaults to 127.0.0.1:8000).
MARK=$HOME/.pi/agent/.workbench-setup-v1
if [ ! -f "$MARK" ]; then
  for p in $AGENT_PI_PACKAGES; do pi install "$p" >> "$HOME/agents/setup.log" 2>&1 || echo "!! pi install $p failed"; done
  prof=$(ls -d "$HOME"/.pi/agent/git/github.com/*/pi-lab-profile 2>/dev/null | head -1)
  [ -n "$prof" ] && bash "$prof/scripts/install.sh" "$AGENT_PI_PROFILE" >> "$HOME/agents/setup.log" 2>&1
  python3 - "$HOME/.pi/agent" "$MODEL_RELAY" "${SERVED_NAME:-qwen3.8-flash-next}" <<'EOF'
import json, os, sys
d, relay, served = sys.argv[1:]
def edit(name, fn, default):
    p = os.path.join(d, name)
    obj = json.load(open(p)) if os.path.exists(p) else default
    fn(obj); tmp = p + ".tmp"; json.dump(obj, open(tmp, "w"), indent=2); os.replace(tmp, p)
def models(m):
    g = m.setdefault("providers", {}).setdefault("globus", {"api": "openai-completions", "models": []})
    g["baseUrl"] = f"http://{relay}/v1"
    g["apiKey"] = "${SPARK_LLM_KEY}"
    g["name"] = "cluster endpoint (fair-proxy, keyed)"
    if not any(x.get("id") == served for x in g["models"]):
        g["models"].append({"id": served, "name": served, "reasoning": True, "input": ["text", "image"],
                            "contextWindow": 262144, "maxTokens": 32768})
edit("models.json", models, {})
edit("settings.json", lambda s: s.update(tuiMode="fullscreen"), {})
EOF
  date -Is > "$MARK"
  echo ">> $(date -Is) pi set up for $(id -un) (log: ~/agents/setup.log)"
fi

python3 "$AGENT_TOOLS/workbench/supervisor.py" >> "$HOME/agents/supervisor.log" 2>&1 &
SUP=$!
_stop () { kill -TERM "$SUP" 2>/dev/null; wait "$SUP"; }
# USR1 = wall limit: stop cleanly and requeue; the dashboard still wants the sessions, so they resume in the new job
trap '_stop; scontrol requeue "$SLURM_JOB_ID"' USR1
trap '_stop; exit 0' TERM
wait "$SUP"
echo ">> $(date -Is) supervisor exited ($?)"
