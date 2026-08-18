# Design and operations notes

Why this deployment looks the way it does: the hardware analysis, tuning measurements,
security model, and operational drills behind the endpoint. The [README](README.md)
has the short version; [CLIENTS.md](CLIENTS.md) is for users.

## Operations details

`serving doctor` is the command to reach for when something is wrong — it
distinguishes "no job", "still loading" (cold start is ~5 min), "publisher tunnel
never came up", and "healthy", and reads the real exit cause out of the last log. A
cron entry runs `serving supervise` every 5 minutes: silent when healthy, it restarts
the slurm job only when it is genuinely gone or failing health checks, revives the
companion proxies, and keeps a read-only copy of the user-facing files fresh in
`/shared/llm` (with the real login hostname substituted for the `<login-node>`
placeholder the repo carries).

**If the operator is unavailable** and the endpoint breaks in a way supervise cannot
fix, any colleague can stand up their own instance rather than needing access to this
one: clone the repo, `bin/setup_reverse_tunnel.sh` once (own tunnel key), fill in
`etc/site.env`, then `serving start`. The model weights on the compute node's local
disk are world-readable. What cannot transfer: the web UI's accounts and chat
histories, which live in the operator's home and stay there.

## The hardware, and why it dictates everything

Each node is a DGX Spark: GB10 Grace-Blackwell, **128 GB unified LPDDR5x at ~273 GB/s**,
`sm_121`, aarch64. vLLM sees 119.6 GiB as "GPU memory" because CPU and GPU share it.

Two consequences drive every decision below:

1. **Decode is memory-bandwidth bound, not compute bound.** Generating one token requires
   reading every weight once. At 273 GB/s, tokens/sec ≈ bandwidth ÷ bytes-of-weights.
   **Bytes per weight is the single number that sets single-user decode speed.**
2. **Batching is cheap for decode, but only at modest context.** The weights are read once
   per step regardless of batch size, so decode throughput scales almost linearly with
   users. But the step is not weights-only — each sequence also costs
   `~0.31 GB` of Gated-DeltaNet state traffic (read *and* written every step) plus
   `32 KiB × context` of KV reads. A usable model of a decode step:

   ```
   step_bytes ≈ 21 GB (weights)  +  B × (0.31 GB + ctx_tokens × 32 KiB)
   ```

   The per-sequence terms overtake the fixed term at roughly 120 K context at 6 users, or
   45 K at 16. Below ~16 K context, batching really is nearly free; above it, per-user
   speed degrades roughly linearly in `B × ctx`.
3. **Prefill, not decode, is the long-context bottleneck** — measured at 1,000–2,100 tok/s
   (below). Prefix caching is what makes this liveable, and it does so decisively.

Memory *capacity* is not the constraint. Bandwidth and prefill compute are.

## The model

`Qwen/Qwen3.8-27B` (released 2026-08-05, Apache-2.0), architecture `qwen3_5`. Notable:

| property | value | why it matters |
|---|---|---|
| layers | 64 | 16 full-attention, 48 **linear attention** (`full_attention_interval: 4`) |
| attention | 24 Q / 4 KV heads, head_dim 256 | only the 16 full-attn layers hold a KV cache |
| linear attn state | 48 heads × 128 × 128, fp32 | constant ~160 MB per sequence, does *not* grow with context |
| context | 262,144 | usable here — KV is unusually cheap |
| MTP | `mtp_num_hidden_layers: 1` | built-in draft head for self-speculative decoding |
| vision | 27-layer tower | multimodal; kept at bf16 |
| vocab | 248,320, untied | embed + lm_head alone are ~2.5 B params |

The hybrid attention is the pleasant surprise. Only 16 of 64 layers cache K/V, so the KV
cost is **32 KiB/token at FP8** — about a quarter of what a same-size dense-attention model
would need. Long contexts are cheap; the linear-attention layers instead carry a fixed
~160 MB recurrent state per concurrent sequence.

## Quantization: NVFP4

Weights, at 273 GB/s, translate directly into speed:

| format | weights | theoretical decode ceiling | verdict |
|---|---|---|---|
| BF16 | ~54 GB | ~5 tok/s | unusable |
| FP8 | ~27 GB | ~10 tok/s | safe but half speed |
| **NVFP4 (chosen)** | **23.4 GB** | **~12–13 tok/s** | native Blackwell FP4 |

We serve **`unsloth/Qwen3.8-27B-NVFP4`**, which is a mixed-precision recipe rather than a
blanket 4-bit cast:

- **NVFP4** (4-bit, group size 16, FP8 e4m3 scales) on MLP `gate/up/down` — the bulk of the parameters
- **FP8** channel-wise on all attention projections, the linear-attention projections, `lm_head`, *and* the last 8 layers' MLPs
- **bf16** untouched for the entire vision tower (303 ignored modules) and embeddings
- ships `model_mtp.safetensors`, so the MTP draft head survives quantization

That is why it is 23.4 GB rather than a naive ~14 GB — the precision is spent where it
matters. GB10 has native FP4 tensor cores, so this also speeds up prefill, not just decode.

## Speculative decoding (MTP) — tuned to 3 draft tokens

The model ships its own multi-token-prediction head, and vLLM 0.27.1 supports it directly
as `method: qwen3_5_mtp`. Because decode is bandwidth bound, verifying several draft tokens
in one pass is nearly free — accepted tokens are close to pure profit.

`SPEC_TOKENS` was tuned empirically. Per-position acceptance on a **real generation
workload** (not the smoke test, whose short predictable completions flatter it badly):

```
pos0 82%   pos1 65%   pos2 48%   pos3 34%
```

| SPEC_TOKENS | solo tok/s | @6 each | @6 aggregate |
|---:|---:|---:|---:|
| 2 | 22.2 | 19.7 | 105 |
| **3 (chosen)** | **25.6** | **20.0** | **105** |
| 4 | crashes | — | — |

**`SPEC_TOKENS=4` faults the engine.** `torch.AcceleratorError: CUDA error: an illegal
memory access was encountered`, after which every request returns 500. An earlier reading
showed a 3x throughput drop at this setting and it was wrongly attributed to compute
contention; re-testing (job 544) showed the real cause is a kernel fault in the MTP/GDN
decode path at 4 draft tokens on sm120. For the full concurrency picture, and why
speculation does *not* stop paying under load, see
[Tuning for agentic batch workloads](#tuning-for-agentic-batch-workloads).

A caution on acceptance figures: an early reading of 92.9% came from the smoke suite's
deterministic one-liners. Representative workloads land at 65-82% per position. Quote the
lower numbers when sizing.

## Measured facts (job 520)

| quantity | value |
|---|---|
| model weights | 21.97 GiB |
| GPU KV cache | **1,742,661 tokens** (~53 GiB at FP8) |
| attention block size | 1600 tokens (forced, to match the mamba page size) |
| NVFP4 kernel | `FlashInferCutlassNvFp4LinearKernel`, autotuned over 21 configs |
| cold start | ~9 min (170 s weights + 65 s compile + 59 s profile/JIT + capture) |
| warm restart | ~4 min (JIT and compile caches persist) |
| vision | **works** — verified with magenta/teal solids and spatial left/right questions |

The KV number is worth dwelling on: 1.74 M tokens means all six users could hold a *full*
262 K context at once (1.57 M tokens) and still fit. vLLM prints this directly as
`Maximum concurrency for 262,144 tokens per request: 6.65x`. Context is simply not a scarce
resource here — a direct consequence of only 16 of 64 layers having a KV cache.

## Measured performance

### Decode scaling, short prompts

Optimized configuration (FlashInfer sm120 GDN prefill + `SPEC_TOKENS=3`):

| concurrent | tok/s per user | tok/s aggregate | vs. original |
|---:|---:|---:|---:|
| 1 | **25.4** | 24.7 | +9% |
| 2 | 23.1 | 42.8 | +5% |
| 6 | **21.3** | **112.5** | +12% |
| 8 | 19.5 | 140.3 | +4% |

| concurrent | tok/s per user | tok/s aggregate | TTFT p50 |
|---:|---:|---:|---:|
| 1 | 23.3 | 22.8 | 0.24 s |
| 2 | 22.0 | 40.8 | 0.74 s |
| 4 | 20.5 | 63.8 | 3.55 s |
| 6 | **19.1** | **110.4** | 0.41 s |
| 8 | 18.7 | 138.0 | 0.54 s |
| 12 | 14.6 | 161.0 | 0.67 s |
| 16 | 13.6 | 203.4 | 0.90 s |

Six concurrent users keep **82 % of solo speed**; aggregate throughput scales 8.9× from 1
to 16. This is the claim in point 2 above, and it holds at short context.

### Prefill (single stream, uncached)

| prompt tokens | TTFT | prefill tok/s |
|---:|---:|---:|
| 2,237 | 1.07 s | 2,097 |
| 4,581 | 2.41 s | 1,902 |
| 9,258 | 6.01 s | 1,541 |
| 18,863 | 13.39 s | 1,408 |
| 38,839 | 31.01 s | 1,252 |
| 78,787 | 75.22 s | 1,047 |

~50–100 TFLOPS effective — well under the FP4 peak, because the Gated-DeltaNet layers run
through Triton/FLA kernels (`Using Triton/FLA GDN prefill kernel` at startup), not the
tuned FP4 GEMM path. **This is the weakest part of the deployment.**

### One user's big prefill vs everyone else

The worst realistic interference case: one user submits a huge cold context while others
hold interactive sessions. Measured with a 130K-token cold prefill (~140 s) running while
short requests probe every few seconds:

| prefill chunk (`--max-num-batched-tokens`) | others' TTFT p50 | max | big prefill rate |
|---:|---:|---:|---:|
| 8192 | 22.8 s | 31.3 s | 899 tok/s |
| **2048 (chosen)** | **5.4 s** | 7.9 s | 943 tok/s |

Each engine step processes one chunk, and at ~900 tok/s of prefill an 8192-token chunk is
a ~9 s step during which nothing else runs. Shrinking the chunk to 2048 bounds the stall
at roughly two chunks (~5 s) and costs the big prefill nothing measurable. Idle-box TTFT
is unchanged (~0.4 s).

### Prefix caching is what makes it usable

Same ~36 K context, simulating consecutive agentic turns:

| turn | prompt | TTFT | vs cold |
|---|---:|---:|---:|
| 1 — cold | 36,334 | **28.53 s** | — |
| 2 — identical | 36,334 | **2.03 s** | 14.0× |
| 3 — +500 new tokens | 36,827 | **2.91 s** | 9.8× |
| 4 — +2,000 new tokens | 38,398 | **3.79 s** | 7.5× |

The 28 s is a one-time cost per session, not per turn. Every subsequent turn is 2–4 s.
A benchmark that sends unique long prompts measures the worst case and badly understates
real agentic behaviour: a sweep of 16 concurrent unique 32 K prompts collapsed to
1.0 tok/s per user, but that was prefill contention (512 K tokens of cold prefill at
~1,000 tok/s ≈ 510 s, matching the observed TTFT p95 of 513 s), a regime prefix caching
prevents in real use.

### Fairness — work-conserving max-min scheduling

`bin/fair-proxy.py` owns the client-facing port (globus1 loopback :8000); the compute
node publishes to an internal port (:8005) behind it. Each connection is attributed to
the cluster account that opened it — the loopback peer socket is owned by that user's
sshd, and the kernel records the uid, the same mechanism the web UI login uses.

One global pool of 32 slots (`FAIR_TOTAL`, matched to `MAX_SEQS` so queueing decisions
happen here rather than FIFO inside vLLM):

- **Below capacity nothing is restricted** — a lone user can hold all 32 slots.
- **At capacity, each freed slot goes to the queued user with the fewest requests in
  flight** (ties: least recent usage by decayed slot-seconds, then arrival order). A
  newcomer's first request takes the very next slot a saturated sweep frees; the sweep
  drops to 31, and drifts toward an even split only while the newcomer keeps
  submitting. Overflow is held, never rejected.

Slots count per request (acquired at the request head, released when its response
completes), so idle keep-alive connections hold nothing. `GET /fair-stats` on the
endpoint shows the live per-user picture. Verified: solo 40-way burst reaches vLLM
peak 32; under contention the scheduler admits the light user on every release (unit
tested step-by-step).

Interactive paths — the web UI backend and the Studio repair proxy — talk to :8005
directly, outside the pool accounting: the fairness pool exists to stop *agent*
traffic from starving people. `serving supervise` keeps the proxy alive; if it dies,
port 8000 is dead until the next 5-minute cron tick.

### Boundary behaviour (verified 2026-08-17, pre-launch drills)

| case | behaviour |
|---|---|
| prompt over 262,144 tokens | clean 400 in 0.5 s, self-explanatory message, no prefill wasted |
| prompt + `max_tokens` over the limit | same 400 — a client pinning `max_tokens=32768` cannot send more than ~229 K of prompt |
| client disconnects mid-generation | slot freed 1.1 s later; no zombie sequences |
| 40 concurrent requests (cap is 32) | zero errors; 8 queue FIFO, TTFT p95 11.9 s, p50 unaffected |
| USR1 / requeue (every 2 days) | in-flight requests complete; ~7 min outage; see Restarts below |

## Tuning for agentic batch workloads

The endpoint was retuned once the dominant use case turned out to be **agentic runs** —
many parallel calls, non-interactive, so aggregate throughput matters and per-request
latency does not. That inverts the objective the first tuning pass optimized for.

Full sweep at `MAX_SEQS=32`, aggregate tok/s:

| concurrent | SPEC=0 | SPEC=1 | SPEC=3 |
|---:|---:|---:|---:|
| 8 | 79 | 118 | **139** |
| 16 | 141 | 203 | **230** |
| 24 | 189 | 257 | **280** |
| 32 | 227 | 309 | **323** |

Two results, one of which refuted the hypothesis that motivated the sweep:

1. **The concurrency cap was the real limit.** `MAX_SEQS=16` capped aggregate at 203 tok/s;
   at 32 it reaches **323 tok/s (+59%)**. Memory was never close to binding — 32 sequences
   at 32K context sits well inside the 1.7M-token KV pool. Per-request speed falls from
   16.4 to 11.2 tok/s, which is the correct trade for batch work.
2. **Speculation does *not* invert under load.** The prediction was that drafting would stop
   paying once the batch filled the machine, so `SPEC=1` or `0` would win at high
   concurrency. It never happens: SPEC=3 leads at every point, by +76% at 8 and still +42%
   at 32. The box stays memory-bandwidth-bound even at batch 32, so drafted tokens remain
   nearly free.

**`SPEC_TOKENS=4` crashes the engine.** Not a slowdown — `torch.AcceleratorError: CUDA
error: an illegal memory access was encountered`, after which every request 500s. An
earlier reading had attributed a 3× throughput drop at this setting to compute contention;
that explanation was wrong, and the real cause is a kernel fault in the MTP/GDN decode path
at 4 draft tokens on sm120. 3 is heavily exercised and clean.

**If you ever want interactive tuning instead**, set `MAX_SEQS` back to 8–16: that trades
aggregate throughput for per-request speed (25.4 tok/s solo).

## Kernel selection on GB10 (sm_121)

vLLM picks the fast FlashInfer Gated-DeltaNet prefill kernel only for compute capability
**90** (Hopper) or capability **family 100** (datacenter Blackwell). GB10 reports `(12, 1)` —
the **sm120 family** — so it matched neither test and silently fell back to Triton/FLA. Since
48 of this model's 64 layers are GDN, that path dominates time-to-first-token.

FlashInfer does ship the sm120 kernel, and dispatches on capability *major*:

```
flashinfer.gdn_prefill.chunk_gated_delta_rule_sm120   AVAILABLE
flashinfer/gdn_prefill.py:  elif _arch_major == 12: chunk_gated_delta_rule_sm120(...)
```

`bin/patch_vllm_gdn_sm120.sh` widens vLLM's gate to include the family (`--revert`,
`--status`). **Re-apply it after any vLLM upgrade** — it edits site-packages.

Measured effect, and it is smaller than the diagnosis suggested:

| prompt tokens | Triton/FLA | FlashInfer sm120 | change |
|---:|---:|---:|---:|
| 2,237 | 2,097 | 2,262 | +7.9% |
| 9,258 | 1,541 | 1,650 | +7.1% |
| 18,863 | 1,408 | 1,462 | +3.8% |
| 38,839 | 1,252 | 1,302 | +4.0% |
| 78,787 | 1,047 | 1,080 | +3.2% |

So the GDN kernel was *not* the prefill bottleneck — it is worth 3–8%. What remains is the
FP4/FP8 GEMMs plus quadratic attention across the 16 full-attention layers. Correctness was
verified before accepting it: needle-in-haystack retrieval of exact 6-digit codes at 25%, 50%
and 80% depth, the deepest inside a 78,810-token context, plus the full smoke suite.

**Constraint:** the sm120 path requires the recurrent state in float32, so this cannot be
combined with `--mamba-ssm-cache-dtype bfloat16`.

## Configuration decisions

| flag | value | reasoning |
|---|---|---|
| `--kv-cache-dtype fp8` | | halves both KV footprint *and* the per-step KV read, which directly speeds long-context decode |
| `--max-model-len 262144` | full native | affordable thanks to hybrid attention |
| `--max-num-seqs 32` | | aggregate throughput kept climbing past 16 (203 → 323 tok/s); memory never binds |
| `--kv-cache-memory` | 60 GiB pinned | on unified memory vLLM's utilization heuristic can't see the slurm cgroup; a fixed pool is identical across requeues |
| `--gpu-memory-utilization 0.80` | | still required — vLLM gates startup on free memory even with a pinned pool, and its 0.92 default fails during restarts |
| `--max-num-batched-tokens 2048` | | prefill chunk = the unit of head-of-line blocking; at 8192 a 130K cold prefill stalled others' TTFT to ~23 s, at 2048 it is ~5 s with no measured prefill cost |
| `--enable-prefix-caching` | | agentic loops resend near-identical prompts; this is a large practical win |
| `--speculative-config qwen3_5_mtp` | 3 draft tokens | see above; 4 crashes the engine |
| `--tool-call-parser qwen3_xml` | | the chat template emits `<function=`/`<parameter=` XML, not Hermes JSON |
| `--reasoning-parser qwen3` | | separates the trace into the `reasoning` field (default since 2026-08-17; required by tool-calling clients — see CLIENTS.md) |
| `--default-chat-template-kwargs` | `reasoning_effort: medium` | the template's own default is `xhigh`; see below |
| `--async-scheduling` | | compatible with MTP (vLLM treats MTP as an Eagle-class method) |
| `--enable-force-include-usage` | | every response (streams included) carries token usage, so UIs can show tok/s and context consumed |

### Reasoning effort is a first-class performance knob

Qwen3.8's chat template defaults to `reasoning_effort: "xhigh"`. At this hardware's token
rate that is minutes of thinking before a first visible answer. The server overrides the
default to `medium`; clients can adjust per request. This matters more than any other
tuning parameter for perceived speed.

The levels are not what they look like. Rendered prompts show `medium` injects no
directive at all (it is the bare template), `low` explicitly tells the model to keep
thinking brief, and `high`/`xhigh` render byte-identical prompts — the template aliases
them. So the real ladder is: off, medium, low, high/xhigh.

## Deployment notes

**Model on local NVMe.** The inter-node link is 1 GbE (the 200 GbE ConnectX ports are not
configured). Serving from NFS `/home` would add ~3 minutes to every restart, so the model
lives at `/scratch/models/` on globus3.

**Python from uv, not the system.** `/usr/bin/python3.12` has no dev headers
(`python3.12-dev` is absent and we have no passwordless sudo), and Triton JIT-compiles a
CUDA shim at startup that needs `Python.h`. The venv is built on a uv-managed CPython that
ships its own headers. Containers were the other option, but Docker needs a sudo password
here, which a batch job cannot supply.

**FlashInfer JIT and `MAX_JOBS`.** vLLM picks `FlashInferCutlassNvFp4LinearKernel` for the
NVFP4 GEMMs, and FlashInfer compiles those CUTLASS kernels on first use — it shells out to
`ninja` and `nvcc`. Two things bite here:

- `ninja` must be on `PATH`. Invoking `venv2/bin/vllm` directly does not put `venv2/bin` on
  `PATH`, so the engine died with `FileNotFoundError: 'ninja'` during memory profiling.
- ninja defaults to ~`nproc` parallel `nvcc` processes, each costing a few GB. Because
  memory is unified, the 22 GiB of model weights are charged to the *same* cgroup, so the
  unbounded build was OOM-killed (`FAILED: [code=137]`). `MAX_JOBS=4` bounds it.

The build is one-time; results cache to `~/.cache/flashinfer` (shared over NFS, so it also
covers the other nodes). Later restarts skip it entirely.

**Restarts.** The partition caps jobs at 2 days. The script sets `--requeue` and traps
`USR1` (raised 5 min before the limit) to drain and requeue itself. Measured end to end
(USR1 drill, job 588): an in-flight streaming request **completes normally** during the
drain, then vLLM exits (~30 s), slurm holds the requeued job ~2 min (`Reason=BeginTime`),
and the model reloads — **healthy again ~7 min after the signal**. Clients need no action;
their tunnels target globus1, which stays up throughout.

**Usage history.** The identity proxy samples vLLM's token counters every 30 s and folds
the deltas into hourly buckets in `logs/token-usage.json` — the counters themselves reset
on every engine restart, so they cannot be read as lifetime figures. Each probe also
records whether the endpoint answered. The stats page turns the buckets into 24 h / 7 d /
30 d / all-time token totals (decode vs prefill), a usage graph, and uptime percentages.
The buckets are whole-box aggregates: token counts and probe results only, no content and
no per-user attribution.

## Privacy

What the endpoint does and does not retain, verified against the running system rather than
assumed.

**Prompts and completions are not logged.** vLLM's `enable_log_requests` defaults to `False`
and `serve.sh` never enables it. Confirmed empirically: searching all 17 job logs for
distinctive text from the benchmark and smoke suites (`Reply with exactly`, `vault access
code`, `thread-safe LRU`) returns **zero matches**. The logs hold engine lifecycle and
metrics only. `logs/` is mode 0700.

**Nothing with request content is written to disk.** The only persisted state is compile and
JIT caches (`~/.cache/vllm/torch_compile_cache`, `/scratch/jit-cache`), model weights, and
aggregate token/uptime counters (`logs/token-usage.json` — numbers only, no content).
Conversations exist only in GPU memory for the life of the request.

**Outbound telemetry is disabled.** vLLM posts anonymous usage stats to `https://stats.vllm.ai`
**by default** (`envs.py: VLLM_NO_USAGE_STATS = False`), via
`global_http_client.post(_USAGE_STATS_SERVER, json=data)` in `usage/usage_lib.py:269`. It
carries no prompts or completions — the payload is GPU and CPU model, exact kernel version,
model architecture, quantization, context length, and a persistent UUID — but it does
advertise that this machine exists and what it runs. `serve.sh` now sets
`VLLM_NO_USAGE_STATS=1` and `VLLM_DO_NOT_TRACK=1`. **28 records were transmitted before this
was caught**; exactly what left is readable in `~/.config/vllm/usage_stats.json`.

**Data does not leave the cluster in normal use.** Remote media fetching is blocked
(`--allowed-media-domains blocked.invalid`), so an image URL cannot make the server call out;
only inline base64 `data:` images are accepted. Traffic reaches users over SSH, encrypted end
to end.

**The prefix cache is shared across all users** — the one nuance worth understanding. vLLM's
automatic prefix caching hashes token prefixes globally, with no tenant separation. This does
not let anyone *read* content they do not already possess: a cache hit requires submitting a
byte-identical prefix. But it creates a **timing side channel** — a user can infer that
someone else has already submitted a particular prompt, because a cached prefix returns its
first token in ~2 s instead of ~30 s.

For six trusted colleagues this is an accepted risk, and turning it off would forfeit the
14x speedup that makes long-context agentic loops usable. If isolation is ever needed, vLLM
0.27.1 supports per-request `cache_salt` (`v1/core/kv_cache_utils.py:579`), which folds a
caller-supplied string into the block hash: pass a distinct salt per user and their caches
stop colliding, at the cost of cross-user reuse.

**No per-user attribution.** In the default SSH-only auth mode there are no API keys, so
metrics are aggregate and requests are not attributable to individuals. That is a deliberate
trade for not distributing a second secret; `REQUIRE_API_KEY=1` restores per-user keys if you
ever want to know who is generating load.

**The web UI stores chats by design; everything else stays quiet.** Chat content (including
model-written code) persists in `webui-data/webui.db` — that is the product working, not a
leak — in a 0700 directory readable only by the operator and root. Around it: service logs
run at `GLOBAL_LOG_LEVEL=WARNING` (no per-request paths, no activity trail), the identity
proxy logs only its own startup, the repair proxy logs repair lengths and never content,
community sharing to openwebui.com is disabled, and telemetry is off. Code execution
happens in the user's browser (Pyodide) and writes nothing to the cluster.

## Security

The endpoint binds `127.0.0.1` on the compute node and is published only to globus1's
loopback over a restricted reverse tunnel, so the model port never appears on a public
interface (these nodes do have public IPs). If the publish preflight fails, `serve.sh`
exits rather than falling back to a public bind — an endpoint nobody can reach is a
cheaper failure than an open one.

Three findings from a review of the first deployment (which did bind `0.0.0.0`) are
recorded because they would be easy to reintroduce:

1. **vLLM does not authenticate every route.** The auth middleware only guards `/v1`,
   `/v2`, `/inference`, `/cohere`; `/tokenize` and `/metrics` answer without a key, and
   `/tokenize` resolves `image_url` content — a working SSRF vector from an
   unauthenticated caller (verified: it fetched a URL and failed only at image parsing).
   The mitigations stay on even behind the tunnel: `--allowed-media-domains
   blocked.invalid`, `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`, `--disable-fastapi-docs`,
   `--allowed-origins '[]'`. Base64 `data:` images short-circuit the domain check, so the
   vision flow still works.
2. **API keys must not go in argv.** `/proc/<pid>/cmdline` is world-readable and the
   partition is not exclusive, so a co-scheduled account could read them. When
   `REQUIRE_API_KEY=1`, keys go through a `--config` YAML (mode 600) instead.
3. **There is no TLS.** Fine while traffic is loopback-or-SSH only; it stops being fine
   the moment anyone re-exposes the port.
4. **The web UI's identity headers are only reachable through the identity proxy.**
   The proxy on :8080 resolves who opened each connection from kernel socket ownership
   — laptops cannot forge it. The backend that trusts those headers listens on a unix
   socket inside `webui-data/` (directory mode 0700), so only the operator account and
   root can reach it directly; no other shell user can hand it a forged identity.
   (Earlier this was a TCP loopback port, which any globus1 shell could have
   addressed — closed 2026-08-17 by launching uvicorn with `--uds` directly, since
   `open-webui serve` cannot bind a socket itself.) Residual: the operator and root
   can still impersonate any account — but they can read the database anyway.

## Files

```
serving/
├── bin/serving                  # ← the only command you need; everything else is plumbing
├── bin/serve.sh                 # the slurm job itself
├── bin/setup_reverse_tunnel.sh  # one-time: authorise the compute-node → globus1 publisher
├── bin/grant_tunnel_access.sh   # give someone LLM access without a shell
├── bin/llm-tunnel               # client-side tunnel helper (copy to a laptop)
├── bin/chat.py                  # minimal client, stdlib only (copy to a laptop)
├── bin/webui                    # Open WebUI on globus1:8080 — browser chat for non-API users
├── bin/webui-idproxy.py         # SSH-identity login + /globus-stats page (run by bin/webui)
├── bin/webui-ring.js            # context ring injected into the web UI (run by bin/webui)
├── bin/fair-proxy.py            # per-user in-flight caps on :8000 (run by supervise)
├── bin/smoke.py                 # correctness suite behind `serving test`
├── bin/repair-proxy.py          # optional: repairs tool calls mangled by buggy clients (CLIENTS.md)
├── bin/fetch_model.sh           # one-shot model download to the node's local disk
├── bin/mirror_venv.sh           # optional: copy venv to local NVMe for faster cold start
├── bin/patch_vllm_gdn_sm120.sh  # re-apply after vLLM upgrades (see kernel section)
├── examples/                    # what to hand users: ALCF-style module, API demos, opencode config
├── ansible/                     # rebuild the stack on a fresh operator account (see its README)
├── etc/keys.env                 # per-user API keys (600)
├── etc/vllm-keys.yaml           # generated from keys.env; keeps keys out of argv (600)
├── etc/endpoint.json            # written at startup: node, bind, and the client URL
├── logs/                        # 0700; rotated to the newest 15 jobs on each start
└── venv2/                       # uv-managed python 3.12 + vLLM 0.27.1
```
