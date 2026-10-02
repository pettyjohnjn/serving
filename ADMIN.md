# Operator guide

For whoever deploys and runs the endpoint. Everything runs as one **service account** on the login node and in
Slurm; nothing needs root at runtime. The root / Slurm-admin prerequisites are listed first.

## 1. Prerequisites (root / Slurm admin)

| what | why |
|---|---|
| a service account (no login shell needed) with a Slurm account that may use the engine nodes' partition | owns every process and file below |
| the service account can read and write the node-local trees on both engine nodes (`$FN_SCRATCH`, `$FN_HF`) | the vLLM tree and checkpoints (~150 GB per node); `fn build` creates them |
| a large shared disk path for `$WEBUI_VENV` and `$AGENT_TOOLS` (world-readable) | keep them off small root disks |
| earlyoom (or equivalent) on the compute nodes, as today | the KV sizing assumes it fires at ~4.8 GiB MemAvailable |
| **agents run as their owner**: either slurmrestd with `auth/jwt` and a way for the service to obtain a JWT for an agent account (`SUBMIT_MODE=slurmrestd`, `SLURM_TOKEN_CMD`), or a sudoers rule letting the service run exactly `sbatch ... <repo>/jobs/agent-host.sbatch` as the agent accounts (`SUBMIT_MODE=sudo`) | without one of them, `SUBMIT_MODE=sbatch` can only run agents as the service account itself (testing only) |
| agent accounts (`RUNAS_MODE`: the person's own account, a `<user>-agent` twin, or a map file) | agents act with exactly that account's permissions |
| optional: a QOS for agent host jobs (e.g. 8 CPUs / 32 GB per user) and a partition or limits that keep them off GPU work | host jobs default to 4 CPUs / 16 GB and never land on the engine nodes |
| optional: a Slurm QOS for the engine job without the 2-day MaxTime | today it requeues itself every 2 days: ~10 min outage |

## 2. Deploy

```bash
git clone <repo> ~/serving && cd ~/serving
cp etc/site.env.example etc/site.env && chmod 600 etc/site.env    # fill in every value
cd ansible && ansible-playbook -i localhost, -c local site.yml --tags build
```

The playbook (ansible/README.md) builds the x86 Open WebUI environment, the aarch64 agent runtime, the vLLM tree and
checkpoints on each engine node (first run downloads ~126 GB per node), installs the tunnel key's restricted
`authorized_keys` line and the `supervise` cron line, and starts the login-node services. Then, watching it:

```bash
serving start              # ~10 min cold boot; prints status when healthy
serving test               # 8 smoke checks (generation, reasoning, tool calls, vision, long context)
tools/validate/validate.sh quick    # ~15 min: decode ladder, prefix cache, corruption probe, 16 x 120k stress
```

Then visit Open WebUI first (http://localhost:8080 through the tunnel): the first account becomes admin. Enable web
search there if wanted (Admin Settings > Web Search; the setting lives in the database, not the environment).

## 3. Ports and processes

| where | what | binds |
|---|---|---|
| login node | fair-proxy | 127.0.0.1:8000 (users by account), `$LOGIN_CLUSTER_IP:$KEYED_PORT` (users by API key) |
| login node | identity proxy (bin/webui-idproxy.py) | 127.0.0.1:8080 (-> Open WebUI socket), 127.0.0.1:8090 (-> dashboard socket); status page at /globus-stats |
| login node | Open WebUI | unix socket `webui-data/webui.sock` (0700 dir) |
| login node | dashboard | unix socket `state/workbench/dashboard.sock`; control API `$LOGIN_CLUSTER_IP:$CONTROL_PORT` (per-job tokens) |
| login node | engine tunnel endpoint | 127.0.0.1:8005 (created by the engine job's ssh -R) |
| engine head node | vLLM | 127.0.0.1:8100 only |
| agent nodes | ttyd per session | cluster address, random port, random password |

All login-node processes are started and kept alive by `serving supervise` (cron, every 5 minutes).

## 4. Daily operation

```
serving status                      everything at a glance
serving doctor                      when something is wrong: no job / still booting / tunnel / healthy
serving restart                     drain, wait for memory, boot (~12 min total)
serving stop / start                stop means stopped: supervision pauses until the next start
serving services restart webui      (fair-proxy | webui | dashboard | all)
serving keys list|add|show|revoke <user>   keys for the keyed listener; the dashboard issues agents' keys itself
serving logs -f                     the engine job's log
```

- `supervise` resubmits a dead engine with exponential backoff (5, 10, 20... min) and gives up after 5 failures,
  writing `logs/STATUS`; `serving start` clears it.
- The engine job requeues itself at the 2-day wall limit (drain, requeue, ~10 min back). Agent host jobs do the
  same, and their sessions come back by themselves.
- A health watchdog inside the engine job kills a vLLM whose /health stays bad (engine-core crash), so the job exits
  and supervise resubmits it.

## 5. Failure modes

| symptom | cause | fix |
|---|---|---|
| job dies ~10-13 min into boot, "shutdown triggered" / KeyboardInterrupt, no CUDA OOM | earlyoom on a node | lower `ENGINE_KV_BYTES` (1-2 GiB at a time); never raise it above the validated 50 GiB without re-running validate.sh |
| `!! checkpoint not prepared` | `fn build` incomplete on a node | `FN_NODE=<node> tools/flash-next/bin/fn build && fn verify` |
| boot OK, endpoint unreachable on the login node | tunnel refused (key line, host key change) | `serving doctor`; check the `[publisher]` lines; re-run `serving install` |
| six concurrent cold prefills wedge the engine | deterministic top-k kernel not loaded | `fn verify` (check 8); never set `VLLM_QSA_DET_TOPK=0` |
| repeated prompts do not hit the prefix cache | `disable_eagle_block_drop` missing from the speculative config | keep it (jobs/serve.sbatch) |
| decode 8-15 tok/s after a config change | 4096 chunk with torch.compile on | keep `GRAPHS=nocompile` with `CHUNK=4096` |
| a person's agent page says "not reporting" | their host job is pending, starting, or cannot reach the control API | `squeue -n agent-host-<user>`; the job's log is `~/agents-host-<jobid>.out` in the agent account's home |

## 6. Secrets and state (all git-ignored, service-owned, 0600/0700)

| path | what |
|---|---|
| `etc/site.env` | site values (hosts, addresses, accounts) |
| `~/.ssh/id_llm_tunnel` | the engine's tunnel key (restricted in `authorized_keys`) |
| `state/keys.json` | API keys for the keyed listener ({key: user}); fair-proxy re-reads it on change |
| `state/workbench/users/<user>/` | each person's desired sessions, last report (incl. terminal passwords), host-job token |
| `webui-data/` | Open WebUI accounts and chats, its secret key |

## 7. Upgrading the model runtime

The vLLM tree is pinned (`tools/flash-next/etc/fn.env`: image digest, patch commit, kernel SHA). To move to a newer
nightly: update the pins, `fn build --force` on both nodes, `fn verify`, `serving restart`, then
`tools/validate/validate.sh full` and compare with DESIGN.md. Things to re-check on any new tree: prefix-cache hits
(`prefix_test`), the corruption probe, and the hard long-context eval. NVMe KV offload becomes possible once vLLM
#57145 and #58413 are in the tree (DESIGN.md).
