# Connecting to the Qwen3.8-27B endpoint

New here? [QUICKSTART.md](QUICKSTART.md) gets you from zero to a first response.

**Base URL:** `http://localhost:8000/v1` — after opening the tunnel (see below)
**Model name:** `qwen3.8-27b`
**Auth:** your SSH key — nothing else. If you can SSH to globus1, you can use the model.

It is an OpenAI-compatible API, so anything that speaks "OpenAI base URL + key" works.

---

## Reasoning effort — read this first

Qwen3.8 is a thinking model and its own chat template defaults to `reasoning_effort: "xhigh"`.
On this box that means a minute or more of thinking tokens before you see any answer.

The server default is `medium`. Override per request — `reasoning_effort` is a first-class
field on chat completions and the OpenAI SDK passes it through as a named argument:

```python
reasoning_effort="xhigh"    # hard problems
reasoning_effort="none"     # thinking off entirely — fastest
```

(`extra_body={"chat_template_kwargs": {"reasoning_effort": ...}}` is the same knob by
another route, and `{"enable_thinking": False}` ≡ `"none"`.) The template accepts exactly
`low`, `medium`, `xhigh`: `high` is an alias for `xhigh` (identical rendered prompt),
`medium` injects no directive at all, and anything else (`minimal`, `max`, …) is a 400
straight from the template. `none` never reaches the template — vLLM turns it into
thinking-disabled first. Thinking spends from `max_tokens`, so give hard problems a real
budget (16384+) or the model can exhaust it before answering. In the browser UI the same
choice is the effort pill in the prompt bar.

---

## Access — your SSH key is the credential

The cluster is publickey-only (`AuthenticationMethods publickey`, `AllowGroups labusers
labadmins`). The endpoint binds loopback on whatever node slurm picked and is published only
to globus1's loopback, so the single way to reach it is an SSH session to globus1. That means
**everyone whose key is in the trusted list can use the model, and nobody else can** — with no
API key to hand out, rotate, or leak.

```bash
ssh -N -L 8000:127.0.0.1:8000 globus1
```

Then point any OpenAI-compatible tool at **`http://localhost:8000/v1`**, model `qwen3.8-27b`.
Any value works as the API key (most clients insist on sending something) — `sk-local` is fine.

Make it permanent in `~/.ssh/config` — the same pattern you already use for the dashboard on
port 3000, so one `ssh globus1` brings up both:

```
Host globus1
    HostName            <login-node>
    User                <your-cluster-user>
    LocalForward        3000 localhost:3000      # dashboard
    LocalForward        8000 localhost:8000      # Qwen3.8-27B endpoint
    ServerAliveInterval 20
    ExitOnForwardFailure yes
```

If you use `ControlMaster`, note that ssh takes **the first obtained value** for each keyword,
so a `Host *` block placed at the top of your config silently overrides the `ControlPersist` in
a later host-specific block. Check what is actually in effect with:

```bash
ssh -G globus1 | grep -iE 'controlpath|controlpersist'
```

That also tells you where the master socket lives, which is what you delete or `ssh -O exit`
when a forwarded port stays claimed after you close the tunnel.

Or copy `bin/llm-tunnel` for a version that reconnects on drop:

```bash
# localhost:8000, auto-reconnects
./llm-tunnel

# if 8000 is taken locally
./llm-tunnel --port 9000
```

To verify, `curl http://localhost:8000/v1/models` should list `qwen3.8-27b`.

### Giving someone LLM access without giving them a shell

Useful for collaborators who should be able to use the model but have no business on the
cluster. `bin/grant_tunnel_access.sh` appends a locked-down entry to `~/.ssh/authorized_keys`:

```bash
# grant
./bin/grant_tunnel_access.sh alice ~/alice.pub

# review current grants
./bin/grant_tunnel_access.sh --list

# revoke; takes effect at once
./bin/grant_tunnel_access.sh --revoke alice
```

The entry is `restrict,port-forwarding,permitopen="127.0.0.1:8000",command="/bin/false"`.
Verified behaviour on this cluster:

| attempt | result |
|---|---|
| `ssh key@globus1 'some command'` | blocked (forced command) |
| interactive shell | `PTY allocation request failed` |
| forward to `127.0.0.1:8000` (the published endpoint) | works |
| forward to *anything else* | `administratively prohibited: open failed` |

Note `command="/bin/false"` is doing real work: `restrict` alone blocks PTY allocation but
**not** command execution, so without it the key is effectively a shell.

### Why this is the right shape

No new secret to distribute or rotate — revocation is deleting one line, and key
distribution is already solved by whatever onboards cluster users. The endpoint never
faces the internet; its only exposed surface is globus1's SSH port. API keys
(`REQUIRE_API_KEY=1`) remain available if load ever needs attributing to individuals.

---

## Using it from your own code / harness

It is a plain OpenAI-compatible server — the same shape as the ALCF inference service, only
the `base_url` differs. **Existing ALCF-shaped code usually needs no changes at all**, just a
different environment:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=sk-local
```

Most harnesses (openai SDK, litellm, aider, instructor, DSPy, langchain…) read those two
variables, so repointing is the whole migration. `OPENAI_API_KEY` can be any non-empty
string: the client library insists on sending one, the server ignores it, and SSH is the
real authentication.

### Coming from the ALCF inference service

`examples/globus_inference.py` plays the role ALCF's `inference_auth_token` does, so an
existing ALCF script is a two-line change:

```python
# ALCF
from inference_auth_token import get_access_token
client = OpenAI(api_key=get_access_token(),
                base_url="https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1")

# globus
from globus_inference import get_access_token, BASE_URL
client = OpenAI(api_key=get_access_token(), base_url=BASE_URL)
```

Everything after that — `chat.completions.create`, tools, streaming — is identical.
`examples/alcf_style.py` is a working side-by-side.

The auth models differ in a way worth knowing: ALCF issues a real bearer token over Globus
Auth because its endpoint is public. Here the endpoint is not public at all, so there is no
token to fetch, refresh, or leak — `get_access_token()` returns a placeholder purely so
ALCF-shaped code runs unmodified.

`globus_inference.get_client()` adds a preflight that separates the two failure modes the
openai client reports identically:

```
nothing is listening on localhost:8000 — the SSH tunnel is not up.
  start it:  ssh -N -L 8000:127.0.0.1:8000 globus1
```
```
the tunnel on localhost:8000 is up, but nothing is serving behind it (ConnectionResetError).
  the model may still be loading — a cold start is ~5 min
  check:     ssh globus1 serving status
```

`examples/openai_api.py` is a runnable tour: basic, streaming, reasoning control, tool
calling, JSON schema, images.

Notes that matter when you drive it hard:

- Concurrency: 32 requests are admitted at once; the rest queue, nothing is dropped.
  Aggregate throughput is ~139 tok/s at 8 concurrent, 230 at 16, 323 at 32, while
  per-request speed falls from ~24 tok/s solo to ~11 at 32.
- Prefix caching is the big lever. A cold 36K-token context costs ~28 s of prefill; the
  same prefix again costs ~2 s. Put shared content (system prompt, documents, few-shot
  examples) first and vary only the tail.
- Structured output uses the standard `response_format={"type": "json_schema", ...}`,
  not vLLM's older `guided_json`.
- Default reasoning effort is `medium`; send `enable_thinking: false` for bulk work that
  doesn't need it.

## Python (OpenAI SDK)

```python
from openai import OpenAI

# api_key can be any string — SSH is the real authentication
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")

r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Refactor this function for clarity: ..."}],
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(r.choices[0].message.content)
```

The thinking trace comes back in `message.reasoning` (and as `reasoning` in streaming
deltas); `content` holds only the final answer. Note the field name: `reasoning_content`
is what some other servers and older vLLM call it, and it is always empty here — code
reading only that name will silently conclude the model never thought. For portability:

```python
msg = r.choices[0].message
trace = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
```

(This is the server's `split` reasoning mode, the default. The alternative `inline` mode
puts the trace in `content`, which breaks tool-calling clients — don't switch it back
without reading the opencode section.)

## curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":256}'
```

## aider

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=sk-local
aider --model openai/qwen3.8-27b
```

## Browser chat (Open WebUI)

For anyone who just wants a chat window: tunnel port 8080 alongside 8000 and open
**http://localhost:8080**. You are signed in automatically as the cluster account you
SSHed in with — no signup, no password. Each person's chat history is private to their
account.

How that works: the process on globus1 that opens a tunneled connection is your own
sshd, running under your uid, and the kernel records that per socket. `bin/webui-idproxy.py`
reads the socket owner from `/proc/net/tcp` and forwards to the Open WebUI backend
(a unix socket inside the 0700 data dir) with trusted-auth headers. Browser-supplied
identity headers are stripped, so the identity cannot be claimed — only inherited from
the SSH login.

What's turned on:

- **Web search** (DuckDuckGo, keyless): toggled per message in the chat input. Fetched
  pages go straight into the model's context (`BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL`)
  rather than through an embedding pipeline — the 262K window makes that the simpler and
  better path. Searches leave the cluster, so don't paste secrets into searched chats.
- **Effort pill** (prompt bar, just left of the model selector; injected by
  `bin/webui-ring.js`): pick `off` / `low` / `med` / `xhigh` for the messages you
  send — `off` answers immediately with no thinking, `med` is the server default.
  The choice sticks per browser, across chats, and is applied by tagging outgoing
  chat requests with `reasoning_effort`; a value set in Chat Controls → Advanced
  Params still wins if you use both.
- **Context ring** (next to the effort pill): fills as the chat's context grows
  against the 262,144-token window, amber past 60%, red past 85%. Hover for the
  numbers; **click to compact** the conversation via Open WebUI's native endpoint
  (older turns become a summary, the recent 40% stays verbatim). A MutationObserver
  re-docks both widgets when the SPA rebuilds the input bar; if an upgrade renames
  the model-selector/Dictate/send anchors they fall back to floating above the bar.
  Ignore the message-info popup's token counts: they **sum every
  internal model call** behind a message (search-query generation, the response,
  titles/tags), so they read several times higher than real context usage.
- **Code execution runs in the user's browser**, not on the cluster. Both the code
  block Run button and the code interpreter use Pyodide (CPython on WebAssembly,
  loaded from a CDN): browser-sandboxed, no GPU, no cluster filesystem, no local
  files; only pure-Python/pyodide-built packages install. The code and its rendered
  output are stored only as part of the chat history in `webui-data/webui.db` on
  globus1 — nothing lands on the cluster filesystem and no process runs there. If
  cluster-side execution is ever wanted, Open WebUI supports a Jupyter engine — that
  would run as the webui account with real filesystem access, so treat it as a
  deliberate security decision, not a toggle.
- **Auto-compaction** is on server-side: any chat crossing 100,000 tokens is compacted
  automatically before the next request, so conversations don't die at the window edge.
- **Search is capped** at 3 pages × 24,000 chars so one searched message costs ≤ ~18K
  context tokens. (Uncapped, a single search dumped 100K+ of web pages into context —
  that is what an early "120K tokens on hello world" reading was.)
- vLLM runs `--enable-force-include-usage`, so token usage is present on every API
  response for any client that wants it.
- **Status page** at `/globus-stats` (served by the identity proxy, not Open WebUI),
  status-page style: an overall banner (operational / under load / offline), per-component
  health (engine, fairness gateway, web UI), and a 90-day availability strip fed by a
  30 s probe from the login node — hover a day for its uptime. Live cards refresh every
  2 s: whole-box generation and prefill tok/s, time to first token, per-stream decode
  speed, requests in flight, queue depth, active users, KV-cache use, and node GPU/
  memory/CPU. Below that, **usage history**: token totals for 24 h / 7 d / 30 d /
  all-time split decode vs prefill, request counts, and a usage graph. The history is
  aggregate whole-box counts — nothing is tracked per user or per request.
- **Branding/scale**: `WEBUI_NAME` is "Globus Cluster Inference"; `bin/webui` re-applies
  two small package patches on every start (drop the forced "(Open WebUI)" name suffix,
  18px root font) so pip upgrades cannot revert them.

**Who can read browser chats** (code the model writes included — it is all just chat
content): the author, through the UI or `GET /api/v1/chats/`; the operator and root,
who can read the whole database file directly; and an Open WebUI admin, via the admin
panel's user-chat access. Other cluster users **cannot** — chats are per-account, and
the backend that trusts identity headers listens on a unix socket inside the 0700 data
directory, so no other shell user can reach it to impersonate anyone. Sharing a chat
link makes it visible to other logged-in users; "share to openwebui.com" is disabled
(`ENABLE_COMMUNITY_SHARING=false`), so nothing can be posted off-cluster. Storage is
`webui-data/webui.db` on globus1 (directory mode 0700), plus `uploads/` for attached
files. Nothing syncs anywhere, and the service logs run at WARNING — no per-request
activity trail.

**Getting model-written code out**: use the copy button on any code block, the
per-chat export in the chat menu, or Settings → Chats → export (JSON of everything).
Code executed in-browser stays in the browser; save outputs from there. There is
deliberately no "write to my cluster home directory" path — the UI runs as the
operator account, which cannot (and should not) write into other users' homes.

It runs on globus1 under `bin/webui` (`start|stop|status|logs|ensure`), independent of
the model job — endpoint restarts don't log anyone out. A cron `ensure` entry revives
both processes after a reboot. Chats live in `webui-data/` (git-ignored, 0700).

## opencode

A terminal coding agent. It leans on tool calling far more than a chat UI does, which this
endpoint supports server-side (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`).

Install (pick one):

```bash
curl -fsSL https://opencode.ai/install | bash
brew install anomalyco/tap/opencode
npm install -g opencode-ai
```

Copy the ready-made config into place:

```bash
mkdir -p ~/.config/opencode
scp globus1:/shared/llm/examples/opencode.json ~/.config/opencode/opencode.json
```

If you already keep an `~/.config/opencode/opencode.json`, merge the `provider.globus`
block into it instead of overwriting. Then, with the tunnel up, run `opencode` in any
project directory — the config sets this model as the default, so there is no `/connect`
or login step.

Endpoint-specific notes:

- The `apiKey` is a placeholder; opencode wants the field to exist, the server ignores it.
- `limit.context` (262144) is the server's real `--max-model-len`. Don't raise it.
- `small_model` handles background chores like session titles, and each one is a real
  generation here. Point it at a faster provider if you have one configured.
- The first big prefill in a session is the slow part (~28 s at 36K tokens); prefix
  caching makes every turn after it cheap. One long-running session beats repeatedly
  starting fresh ones.

### The "Extra data" 400 on follow-up questions

`Extra data: line 1 column N (char N-1) (BadRequestError)` on a follow-up means a stored
tool call's `arguments` is valid JSON with trailing bytes — vLLM `json.loads()`es that
field when rendering the history, so every subsequent turn in that conversation fails the
same way. Retrying can't help; start a new conversation or delete the bad assistant turn.

The cause is client-side (seen with Unsloth Studio, which is beta): it appends the
streamed arguments and then the final copy again, so the field holds `{...}{...}` and the
error offset is the length of one copy. The server's own streamed arguments were verified
clean.

`bin/repair-proxy.py` works around it by truncating each `arguments` to its first
complete JSON value before forwarding. Run it on globus1 and point your tunnel at it:

```bash
./repair-proxy.py --listen 8001 --upstream 8000     # on globus1
ssh -N -L 8000:127.0.0.1:8001 globus1               # on your laptop
```

Only people who repoint their tunnel go through it. It logs repair events (lengths and a
short excerpt of the discarded tail, never prompts) and serves counters at `/proxy-stats`.
Genuinely truncated arguments pass through untouched — there is nothing safe to
reconstruct from those.

## Cline / Roo Code / Continue (VS Code)

Provider **OpenAI Compatible** → Base URL `http://localhost:8000/v1`, any API key, model `qwen3.8-27b`.
Tool calling is enabled server-side (`qwen3_xml` parser), so agentic edit/apply flows work.

## Claude Code

Claude Code speaks the Anthropic API, not OpenAI, so it needs a translating proxy
(e.g. LiteLLM in `--config` mode, or claude-code-router) pointed at this endpoint.
Given everyone already has enterprise Claude, this is usually not worth setting up.

## Images

The model is multimodal (up to 4 images per request; video disabled):

```python
messages=[{"role": "user", "content": [
    {"type": "text", "text": "What's the bug in this screenshot?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
]}]
```

---

## Etiquette and fairness

Fairness is automatic and work-conserving. If you're the only one on the box, you can
use all 32 concurrent slots — go ahead and max it out. The moment someone else
submits, their requests take priority for freed slots until things even out: your
sweep drifts down (32 → 31 → …) exactly as fast as they actually submit, and recovers
the capacity the moment they stop. Nothing is ever rejected — excess requests just
wait for one of the pool's slots.

`curl http://localhost:8000/fair-stats` shows the live picture (who holds what, who's
queued, recent usage). Browser chats don't count against the pool. Full node load —
GPU utilisation, memory, CPU — is at `http://localhost:8080/globus-stats`. A heads-up
to the group before very large overnight sweeps is still polite, but no longer
load-bearing.
