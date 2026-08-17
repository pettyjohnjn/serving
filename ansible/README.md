# Ansible deployment

Reproduces the serving stack on a fresh operator account (or after a wipe). Written to
slot into the cluster's existing `globus-admin` conventions — run it as the operator
user, no root required:

```bash
cd ansible
ansible-playbook -i localhost, -c local site.yml
```

What it covers, in order:

1. directories, permissions (`logs/` `webui-data/` 0700), gitignored state
2. the two virtualenvs via `uv` (vLLM one is a long build the first time; both are
   guarded with `creates=` so reruns are no-ops)
3. the vLLM sm120 GDN kernel patch (idempotent; re-run after any vLLM upgrade)
4. the reverse-tunnel key + authorized_keys entries (delegates to
   `bin/setup_reverse_tunnel.sh`, which is already idempotent)
5. cron entries: `serving supervise` and `webui ensure`, every 5 minutes
6. the read-only user-facing mirror at `/shared/llm`
7. model weights on the compute node's local NVMe (submits `bin/fetch_model.sh`
   through slurm if the path is missing)

What it deliberately does NOT do:

- start the endpoint (`serving start` is a human decision, and first cold start
  benefits from being watched)
- create web-UI accounts (first visit auto-creates them; first-ever visitor becomes
  admin, so the operator should visit first)
- manage anything as root — sshd config, slurm, groups all belong to `globus-admin`

Variables worth overriding are at the top of `site.yml` (`serving_root`, python
version, model repo id).
