# serving

An OpenAI-compatible inference endpoint for Qwen3.8-27B on a DGX Spark (GB10)
slurm cluster, shared by a small research group. SSH is the only credential:
if you can reach the login node, you can use the model — there are no API keys,
no accounts, no passwords anywhere in the stack.

## Quick start

Open a tunnel to the login node and leave it running:

```bash
ssh -N -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 globus1
```

Then either open **http://localhost:8080** and chat in the browser (you are
logged in automatically as your cluster account), or point any OpenAI client at
the API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(r.choices[0].message.content)
```

The `api_key` is a placeholder the SDK insists on; the server ignores it.
[QUICKSTART.md](QUICKSTART.md) is this in more detail; [CLIENTS.md](CLIENTS.md)
covers tools and agents (opencode, aider, structured output, images) and the
browser UI's features.

## What to expect

| | |
|---|---|
| model | Qwen3.8-27B, NVFP4 weights, thinking model (default effort `medium`) |
| context window | 262,144 tokens per request |
| speed, solo | ~20–25 tok/s, first token in ~0.3 s |
| speed, busy | ~11 tok/s each at the 32-request cap; overflow queues, nothing is dropped |
| long prompts | cold prefill ~1,000 tok/s; prefix caching makes repeat context ~14× faster |
| fairness | work-conserving: one user may fill all 32 slots, but under contention freed slots go to whoever holds least |
| live stats | http://localhost:8080/globus-stats — tok/s, load, GPU, queue, token history, uptime |
| availability | supervised via cron; survives node reboots and the 2-day slurm limit with a ~7 min blip |

## Operating it

Everything goes through one command on the login node:

```
serving start|stop|restart      lifecycle (slurm job + drain)
serving status                  job, reachability, load, companion processes
serving test                    smoke suite against the live endpoint
serving doctor                  diagnose a broken endpoint
serving supervise               cron entry; revives anything that died
```

The browser UI has its own `bin/webui start|stop|status|ensure`. Site-specific
names (compute node, login hostname) live in `etc/site.env` — copy
`etc/site.env.example` and fill it in. `ansible/` rebuilds the whole stack on a
fresh operator account.

## More

[DESIGN.md](DESIGN.md) holds the long version: why every configuration value is
what it is, the measured performance tables, the security and privacy model, and
the operational drills the failure numbers come from.

## Model profiles

The endpoint can serve more than one model. Everything model-specific lives in
`etc/models/<profile>.env`: the weights path, the runtime that can load them, the
tuned defaults, extra flags and extra environment. Everything else — the reverse
tunnel, fair-proxy, the auth policy, draining and requeue — is shared.

    serving start                       # the declared default (qwen38-27b)
    serving restart qwen38-flash-next   # switch
    serving restart qwen38-27b          # roll back

The choice sticks. `serving supervise` resubmits from a timer with no arguments, so
without a record of it the next supervised restart would quietly bring back the other
model; it is written to `logs/.model-profile` instead. `serving status` always prints
the active profile, and when it differs from `MODEL_PROFILE` in `etc/site.env` it says
so rather than leaving the drift silent:

    model profile        qwen38-flash-next  (operator override; Ansible declares
                         qwen38-27b -- clear with: rm logs/.model-profile)

That file lives in `logs/` and not `etc/` on purpose: under configuration management
`etc/` is re-templated on every apply, and an apply must not revert an operator's
choice in the middle of an incident.

The served model name changes with the profile (`qwen3.8-27b` becomes
`qwen3.8-flash-next`), so a client that hardcodes a name breaks on a swap. The repo's
own tooling (`serving status`, `chat.py`, `smoke.py`, the status page) asks the server
instead. To keep one stable name across swaps, set `SERVED_NAME` in `etc/site.env`;
it applies to whichever profile is running, so pick something model-neutral.

The lower-level form still works and is what `serve.sh` sees:

    sbatch --export=ALL,MODEL_PROFILE=qwen38-flash-next bin/serve.sh

The profile is sourced before the tunables in `serve.sh`, so those become fallbacks.
Selecting `qwen38-27b` produces a byte-identical flag set to how the endpoint ran
before profiles existed; `DRY_RUN=1 bin/serve.sh` prints the argv and exits, which is
how that is checked without touching production.

| | qwen38-27b | qwen38-flash-next |
|---|---|---|
| runtime | `venv2/bin/vllm` (0.27.1) | extracted official image, host python3.12 |
| weights | `/scratch/models/Qwen3.8-27B-NVFP4` | `/scratch/hf/...-fp8hybrid` |
| max seqs | 32 | 8 |
| speculation | qwen3_5_mtp, 3 | mtp, 1 |
| KV | fp8, pinned pool | bf16 |
| tool parser | qwen3_xml | qwen3_coder |

Flash-Next needs its runtime built and verified first:

    tools/flash-next/bin/fn build && tools/flash-next/bin/fn verify

The profile refuses to start if the weights or the prepared hybrid checkpoint are
missing, rather than falling through to the 27B default.
