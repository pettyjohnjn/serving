# Quick start

Qwen3.8-Flash-Next on the cluster, behind an OpenAI-compatible API, plus a browser chat and an agent workbench.
If you can ssh to the login node, you already have access: there is no token, key, or signup.

## 1. Open the tunnel

```bash
ssh -N -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 -L 8090:127.0.0.1:8090 <login-node>
```

Leave it running. To make it automatic, add a block to `~/.ssh/config`:

```
Host globus1
    HostName <login-node>
    User <your-cluster-username>
    LocalForward 8000 127.0.0.1:8000
    LocalForward 8080 127.0.0.1:8080
    LocalForward 8090 127.0.0.1:8090
```

after which plain `ssh globus1` also carries the endpoint.

## 2a. Chat in the browser

Open **http://localhost:8080**. You are logged in automatically as your cluster
account — the tunnel itself is the login, so there is no signup or password. Your chat
history is yours alone.

- **Web search** (if the operator has enabled it): toggle it in the message input (the ⊕ controls) and the
  model will search and read the pages before answering.
- **Context ring** (in the toolbar under the message box, beside Dictate): fills as this chat uses up its
  262,144-token window. Hover for the numbers; click to compact the conversation
  (older turns become a summary). Chats past 100K tokens compact automatically. Ignore
  the info popup's token counts — they sum internal calls (searches, titles) and read
  several times higher than real context usage.
- **Running code**: the Run button executes Python **in your own browser** (Pyodide/
  WebAssembly), not on the cluster — it can't see the GPU, the cluster filesystem, or
  your local files. The code itself is just part of the chat history.
- **Live cluster stats** — tok/s, load, queue, cache: **http://localhost:8080/globus-stats**

## 2b. Or send a prompt from code

With curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 256}'
```

Or Python (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")

r = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(r.choices[0].message.content)
```

The `api_key` can be any non-empty string — the SDK insists on one, the server ignores
it. Your SSH key already did the authentication.

## 2c. Run coding agents on the cluster

Open **http://localhost:8090**. Start a session (a name and a working directory) and open its terminal: it is
[Pi](https://pi.dev) with the lab profile, running in your own Slurm job on the compute nodes, as your agent account,
with the cluster's model. Sessions keep running when you close the page; reopen it to see where they are, stop
them, or resume them later with the conversation intact. All your sessions share one job (default 4 CPUs, 16 GB).

## 3. Worth knowing

- The model thinks before answering (default effort `medium`). The trace comes back in
  `message.reasoning`; the answer in `message.content`. Control it per request:

  ```python
  extra_body={"chat_template_kwargs": {"reasoning_effort": "xhigh"}}   # hard problems (low | medium | xhigh)
  extra_body={"chat_template_kwargs": {"enable_thinking": False}}      # fastest
  ```

- Thinking spends from `max_tokens`, so give nontrivial questions 4096 or more.
- Context window is 262,144 tokens. Expect ~45 tok/s alone, ~23 with 8 people busy, ~11 with all 32 slots
  busy (a bit less with very long contexts); requests beyond 32 concurrent queue rather than fail, and when the
  endpoint is full the next free slot goes to whoever has the fewest requests running.
- If nothing answers, find which stage broke:

  ```bash
  # server + your account (run this first — if it lists the model, only your tunnel is wrong)
  ssh <login-node> 'curl -s -m 5 http://127.0.0.1:8000/v1/models'

  # is local port 8000 already taken? any output = yes
  lsof -nP -iTCP:8000 -sTCP:LISTEN
  ```

  A taken port: tunnel with `-L 9000:127.0.0.1:8000` and use `localhost:9000` instead.
  Add `-o ExitOnForwardFailure=yes` to the tunnel command so port clashes fail loudly.

That's all. [CLIENTS.md](CLIENTS.md) covers tools and agents (opencode, aider, images,
structured output), and `examples/` has an ALCF-style module — existing ALCF scripts run
against this endpoint with a two-line change.
