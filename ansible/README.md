# Ansible deployment

Reproduces the stack on a fresh service account (or after a wipe), from a checkout at `~/serving`. Written to slot
into the cluster's `globus-admin` conventions: run it as the service account, no root.

```bash
cp etc/site.env.example etc/site.env    # fill in; it is git-ignored and names hosts, addresses and accounts
cd ansible
ansible-playbook -i localhost, -c local site.yml               # environments, tools, services, cron
ansible-playbook -i localhost, -c local site.yml --tags build  # + vLLM tree and checkpoints on the engine nodes
```

What it covers, in order:

1. `etc/site.env` is present; private `logs/ state/ webui-data/` (0700)
2. the x86 Open WebUI environment (`bin/install-webui-env.sh`, into `$WEBUI_VENV`)
3. the aarch64 agent runtime, shared and world-readable (`bin/install-agent-tools.sh`, into `$AGENT_TOOLS`)
4. with `--tags build`: `tools/flash-next/bin/fn build` + `fn verify` on each engine node (image, patches, kernel,
   checkpoint, FP8 hybrid and TP2 conversions; the first run downloads ~126 GB per node)
5. removes the single-node era's cron entries, then `bin/serving install`: the tunnel key's restricted
   `authorized_keys` line, the `supervise` cron line, and the login-node services
6. `serving` on PATH, the read-only user mirror at `/shared/llm`

What it deliberately does NOT do:

- start the engine (`serving start` is a human decision; watch the first cold start, ~10 min)
- create the Open WebUI admin: accounts come from the identity proxy; the first account to visit becomes admin, so
  the operator should visit first
- anything as root or Slurm admin: ownership of the node-local `/scratch` trees, slurmrestd/JWT or a sudoers rule
  for submitting agent jobs as their owners, the agent QOS, agent accounts, earlyoom, Tailscale. See ADMIN.md.
