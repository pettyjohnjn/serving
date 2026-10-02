# Design and operations notes

Why the endpoint is built the way it is, with the measurements behind each decision (October 2026). All numbers
are from the two engine nodes unless stated; "per stream" is decode tok/s for one request among N concurrent ones.
Reproduce any of them with `tools/validate/`.

## The hardware

DGX Spark (GB10): 128 GB of unified memory shared by CPU and GPU (~121 GiB visible), ~273 GB/s memory bandwidth,
20 Arm cores, one Blackwell GPU (sm_121). Two of them are cabled back to back with two 200 Gb ConnectX-7 RoCE
links; NCCL over IB verbs measures 22-40 us per small all-reduce (57-75 us inside CUDA graphs) and ~21 GB/s for
large messages.

Three facts drive everything:

1. **Memory is the binding resource.** The model's weights, the KV cache, the linear-attention state per sequence,
   the page cache for the n-gram table and the OS all share the same 121 GiB.
2. **earlyoom runs on the compute nodes** (`-m 4 --prefer python`): when MemAvailable drops under ~4.8 GiB it
   SIGTERMs vLLM. That is the real ceiling for the KV reservation, not CUDA. Signature: "shutdown triggered" /
   KeyboardInterrupt in the log, no CUDA OOM. Rule: keep MemAvailable above ~7 GiB under the 16 x 120k stress.
3. **Decode is bandwidth- and overhead-bound**, and batching hurts this MoE twice: more distinct experts are read
   per step, and multi-token-prediction acceptance falls with batch size.

## The model

Qwen3.8-Flash-Next: 48 layers, 12 full attention (QSA sparse attention with an indexer) and 36 Gated DeltaNet
linear-attention layers; 512 experts, top-10, ~6B active; a 47.7 GiB FP8 n-gram ("PLE") embedding table; one
MTP layer; 262,144-token context. Checkpoint: RadixArk NVFP4 (routed experts NVFP4), converted by `fn build` so the
dense layers (attention, linear-attention projections) are blockwise FP8 (128x128). The vLLM tree is the pinned
nightly (8a728663) plus tonyd2wild's PLE overlay and our patches (tools/flash-next).

## Layout: one engine across two nodes (TP2), not two independent engines

| | TP2 (deployed) | 2 independent engines |
|---|---|---|
| weights per node | 39 GiB | 76 GiB |
| KV pool, bf16 | **3.44M tokens, shared** (50 GiB/node) | 2 x 0.74M (split) |
| 1 user, 64k context | **47.1** tok/s | 35.3 |
| 8 users, 64k | 21.7 | 21.6 |
| 16 users, 64k | 15.9 | 16.0 |
| 32 users, 64k | **9.7 / 312 aggregate** | 3.0-4.4 / ~96-140 (KV exhausted) |
| 32 users, short prompts | 11.5 / 367 | 11.0 / ~354 |

With agent-sized contexts, independent engines run out of KV first (bf16 at 8 users per engine, fp8 at 16) and
then evict and re-prefill. TP2 also prefills 25-30% faster and has one prefix cache, so no session routing. What it
costs: either node down takes the endpoint down, and every 2-day requeue is a ~10 min outage.

TP2 needs its own checkpoint: the shared expert is 640 wide, 320 per rank, not a multiple of the 128x128 FP8
block, so `fn build` stage 8 writes `-fp8hybrid-tp2` with the shared expert left bf16 (0.24 GiB per rank).

## The speed recipe: full decode graphs with the n-gram table staged from disk

Published two-Spark recipes were 35-50% faster than our first TP2 build. The gain is mostly from running decode as
full CUDA graphs with torch.compile off plus 4096-token prefill chunks, not from holding the n-gram table in GPU
memory. tonyd2wild's staged gather (rows read from NVMe before each step into a fixed GPU buffer) gives the graphs
without the memory:

| TP2, bf16 KV, 16 slots | KV pool | short prompts 1/2/4/8/16 | 64k decode 1/4/8/16 |
|---|---|---|---|
| table read inside the step, piecewise + compile, MTP1, chunk 2048 | 3.65M | 31.3/27.1/21.6/17.4/13.2 | 28.5/21.3/15.7/12.0 |
| **staged, full graphs, MTP2, chunk 4096 (deployed)** | 3.57M | **44.7/36.3/29.8/22.9/16.5** | **47.1/31.9/21.7/15.9** |
| same with MTP3 | 3.50M | 35.9/37.1/28.8/20.4/15.5 | 52.4/30.9/23.1/14.6 |
| same with MTP1 | 3.65M | 38.9/30.4/28.5/22.3/16.9 | 44.9/29.9/21.3/15.0 |
| table resident in GPU memory (23.8 GiB/node), MTP3 | 1.82M | 41.4/37.5/30.0/23.0/17.4 | 49.8/31.8/23.8/17.1 |

- Resident table: ~30% faster prefill and +8-17% at 16 streams, for half the KV pool. Not worth it for long agent
  sessions; it is one knob away (`PLE_MODE=resident`). The stock loader cannot read this checkpoint's FP8 table
  (global weight_scale), hence the overlay's resident reader.
- Never combine the 4096 chunk with torch.compile on: decode collapses to 8-15 tok/s.
- The recipe barely helps a single Spark (+4-12%): there decode is bandwidth-bound; on TP2 the graphs remove launch
  and all-reduce overhead.
- Stalls: during a cold 100k-token prefill, other streams pause at most 1.6 s with the 4096 chunk.

## 32 slots

`ENGINE_SEQS=32` with a 50 GiB/node KV pin: 32 x ~88k prompts (2.8M tokens resident) complete with no preemptions
and MemAvailable never under 8.4 GiB. Up to 16 users it matches the 16-slot configuration exactly, so the extra
slots cost nothing. Per-sequence linear-attention state is ~57 MB per rank.

## KV precision: bf16

fp8 KV doubles the pool but, without calibrated scales (this checkpoint ships none; vLLM warns at boot), it costs a
little fine-grained recall. Hard long-context eval (`tools/validate/longctx_eval.py --hard`: 200 facts per document
with near-duplicate decoy keys, 10 lookups, an 8-hop chain, 32k-200k tokens, 40 documents, paired by prompt):

| run | lookups of 400 | chains of 40 | multi-value exact of 40 |
|---|---|---|---|
| bf16, piecewise build | 393 | 35 | 33 |
| bf16, deployed build | 389 | 34 | 31 |
| fp8 | 383 | 31 | 34 |

Two bf16 builds differ by 4 lookups, about the size of the fp8 gap (fp8 vs the first bf16 run p = 0.04, vs the
second p = 0.29). Best estimate: fp8 costs ~1.5-2.5 points on hard long-context lookups; every miss is a wrong entry
retrieved, never a malformed answer. The error does not accumulate in the cache over time (each entry is rounded
once), but for multi-day agents a wrong retrieval can be written into a summary and survive compactions. fp8 is
not faster on TP2, and the bf16 pool is large enough, so bf16 it is. Easy retrieval, GSM8K and MMLU-Pro showed no
difference.

## Prefix caching

Agents re-send their whole conversation every turn, so the prefix cache is what keeps prefill cheap: a repeated
24k-token prompt takes 0.7 s instead of ~11 s. On this vLLM tree, MTP's "eagle block drop" cannot identify the
drafter's KV group for this model and disables reuse for the Mamba groups; `disable_eagle_block_drop` in the
speculative config restores it (outputs cannot change: drafts are verified by the target). Only a request's final
aligned state is kept, so a prompt that shares only part of an earlier prefix does not hit.

## NVMe KV offload: blocked upstream (for now)

vLLM's OffloadingConnector with a RAM tier + filesystem tier would keep evicted sessions on local NVMe and reload
them in ~0.1-0.5 s instead of re-prefilling (and its p2p tier can move cache between engines over RDMA). In the
pinned tree it does not work for this model: (1) the QSA indexer's ring-buffer KV group trips a block-size assert
(vLLM #54743 / #57145); (2) with MTP on and align-mode Mamba state, Mamba states are dropped, so there are no hits
(#58413, open). `OFFLOAD_DIR` in jobs/serve.sbatch and `tools/validate/offload_test.py` are ready for when both land.

## Fairness

`bin/fair-proxy.py` admits 32 requests (the engine's slot count). Below capacity anyone gets every free slot; at
capacity each freed slot goes to the waiting user with the fewest requests in flight (ties: least recent usage),
so a newcomer's first request takes the next free slot and the split converges as fast as people submit. It does
not preempt running requests. Users are identified by the account that opened the loopback connection (their
sshd's uid, read from /proc/net/tcp) or, on the keyed listener used by agent jobs, by their API key. Browser chats
go to the engine directly (127.0.0.1:8005) and do not count against the pool.

## The agent workbench

- **One Slurm job per person** holds all of that person's Pi sessions (idle session ~170 MB; 8 idle sessions =
  1.34 GiB for the job). CPUs are the real limit: what the agents run.
- **Identity**: the dashboard is reachable only through the identity proxy, which sets `X-Workbench-User` from the
  connection's owner and strips any client-supplied identity header.
- **Run-as**: agents run as an agent account mapped from the person (`RUNAS_MODE`: same name, a `-agent` twin, or
  a map), so they have that account's permissions and nothing else. Submitting a job as another account needs
  `SUBMIT_MODE=slurmrestd` (JWT) or `sudo` (a sudoers rule for exactly that sbatch); `sbatch` mode only runs agents
  as the service account itself (testing).
- **No shared files**: the host job never reads the service's files (everything comes in its environment; the
  supervisor runs from the world-readable agent tools), and the dashboard never reads the agent account's home. The
  job polls its desired sessions and reports its terminals over a control API with a per-job token.
- **Terminals**: each session's ttyd binds the compute node's cluster address with a random password that only the
  dashboard holds; the dashboard proxies it (HTTP + websocket) for its owner only.
- **Model access**: Pi's model entry points at fair-proxy's keyed listener with the person's key; nothing is
  tunnelled per session.

## Privacy

- **Prompts and completions are not logged** (vLLM's request logging is off; logs hold lifecycle and metrics).
- **Telemetry is off**: vLLM posts anonymous usage stats (hardware, model and config, a persistent ID; no prompts)
  to stats.vllm.ai by default. `VLLM_NO_USAGE_STATS=1` and `VLLM_DO_NOT_TRACK=1` are set in jobs/serve.sbatch and
  the launcher lint refuses to start without them. (They were missing from the Flash-Next evaluation and first
  two-node launchers in Sept-Oct 2026; what was recorded is in the service account's `~/.config/vllm/usage_stats.json`.)
- **The prefix cache is shared** across users: no one can read another's content, but cache-hit timing reveals that
  a prefix was sent before. vLLM supports a per-request `cache_salt` if isolation is ever needed.
- **Open WebUI stores chats by design**, in a 0700 directory, at log level WARNING; community sharing and telemetry
  are off. Code execution in the UI happens in the browser (Pyodide).

## Security

- The engine binds 127.0.0.1 on its head node; the launcher lint refuses `0.0.0.0` or a missing `--host`.
- vLLM leaves `/tokenize`, `/metrics` and others unauthenticated, and `/tokenize` resolves `image_url` (a working
  SSRF probe). Kept on even behind the tunnel: `--allowed-media-domains blocked.invalid`,
  `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`, `--disable-fastapi-docs`, `--allowed-origins '[]'`. Inline `data:` images work.
- The tunnel key's `authorized_keys` line allows exactly one listen (127.0.0.1:8005), no shell, no commands, no
  forwards (`restrict,port-forwarding,permitlisten=...,permitopen="127.0.0.1:1",command="/bin/false"`).
- Open WebUI and the dashboard listen on unix sockets in 0700 directories; only the identity proxy (and the
  service account and root) can reach them, so no one can hand them a forged identity.
- The keyed listener is on the cluster network and forwards nothing without a known key. There is no TLS anywhere:
  fine while traffic is loopback, SSH, or the cluster network.
- Agent terminals: per-session random passwords; agents act with their account's permissions (they can read that
  account's `~/.ssh`; a light sandbox hiding credentials is the next hardening step).
