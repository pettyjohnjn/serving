# serving

A shared, OpenAI-compatible LLM endpoint for a small lab, plus a browser chat UI and an agent workbench, on a Slurm
cluster of DGX Sparks (GB10). One model: **Qwen3.8-Flash-Next** (125B MoE, ~6B active; NVFP4 experts, FP8 dense
layers), served by vLLM across **two Sparks with tensor parallelism** over their RoCE link.

- **Users** read [QUICKSTART.md](QUICKSTART.md) (connect in two minutes) and [CLIENTS.md](CLIENTS.md) (every
  client, from curl to coding agents).
- **Operators** read [ADMIN.md](ADMIN.md) (deploy, operate, what needs root). Why it is built this way, with the
  measurements: [DESIGN.md](DESIGN.md).

## What runs where

```
laptop --ssh -L 8000/8080/8090--> login node (service account, bin/serving)
                                   127.0.0.1:8000  fair-proxy      fair-share admission, users by account
                                   <cluster-ip>:8001  fair-proxy   same pool, users by API key (agent jobs)
                                   127.0.0.1:8080  identity proxy -> Open WebUI   (logged in as your cluster account)
                                   127.0.0.1:8090  identity proxy -> agent dashboard
                                   127.0.0.1:8005  <- reverse tunnel from the engine
engine nodes (2 x GB10):           vLLM, TP2, bf16 KV (3.4M tokens), 32 slots     jobs/serve.sbatch
other compute nodes:               one Slurm job per person holding all their Pi agent sessions
                                                                                  jobs/agent-host.sbatch
```

Nothing listens on a public interface: the engine binds its head node's loopback and publishes to the login node
over a reverse SSH tunnel whose key can do nothing else; the login-node services bind loopback (the keyed listener
binds the cluster network and refuses requests without a valid key).

## Operating it

```
serving status        engine job, reachability, live load, services
serving start|stop|restart
serving doctor        diagnose a broken or unreachable endpoint
serving keys add <user>        API key for the keyed listener (agents get theirs automatically)
serving supervise     cron, every 5 min: keeps everything up, resubmits a dead engine with backoff
tools/validate/validate.sh quick|full    acceptance tests against the running engine
```

## Layout

```
bin/serving                 the CLI (engine lifecycle, login services, keys, install, supervise)
bin/fair-proxy.py           fair-share admission (loopback + keyed listeners)
bin/webui, webui-idproxy.py Open WebUI and the identity proxy (also fronts the dashboard), status page
bin/install-webui-env.sh    x86 Open WebUI environment
bin/install-agent-tools.sh  aarch64 Node + Pi + ttyd for agent jobs
jobs/serve.sbatch           the engine (two nodes)
jobs/agent-host.sbatch      a person's agent host job
workbench/                  dashboard (login node) and supervisor (inside agent host jobs)
tools/flash-next/           builds the vLLM tree and checkpoints on each engine node (fn build / fn verify)
tools/validate/             benchmarks and acceptance tests
etc/site.env.example        every site value; copy to etc/site.env (git-ignored)
ansible/                    operator-level playbook
examples/                   client snippets
```
