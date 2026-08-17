# Quick start

Qwen3.8-27B on the globus cluster, behind an OpenAI-compatible API. If you can
`ssh globus1`, you already have access — there is no token, key, or signup.

## 1. Open the tunnel

```bash
ssh -N -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 <login-node>
```

Leave it running. To make it automatic, add a block to `~/.ssh/config`:

```
Host globus1
    HostName <login-node>
    User <your-cluster-username>
    LocalForward 8000 127.0.0.1:8000
    LocalForward 8080 127.0.0.1:8080
```

after which plain `ssh globus1` also carries the endpoint.

## 2a. Chat in the browser

Open **http://localhost:8080**. You are logged in automatically as your cluster
account — the tunnel itself is the login, so there is no signup or password. Your chat
history is yours alone.

- **Web search**: toggle it in the message input (the ⊕ controls) and the model will
  search DuckDuckGo and read the pages before answering.
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
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 256}'
```

Or Python (`pip install openai`):

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

The `api_key` can be any non-empty string — the SDK insists on one, the server ignores
it. Your SSH key already did the authentication.

## 3. Worth knowing

- The model thinks before answering (default effort `medium`). The trace comes back in
  `message.reasoning`; the answer in `message.content`. Control it per request:

  ```python
  extra_body={"chat_template_kwargs": {"reasoning_effort": "xhigh"}}   # hard problems
  extra_body={"chat_template_kwargs": {"enable_thinking": False}}      # fastest
  ```

- Thinking spends from `max_tokens`, so give nontrivial questions 4096 or more.
- Context window is 262,144 tokens. Expect ~20 tok/s solo, ~11 tok/s when the box is
  full; requests beyond 32 concurrent queue rather than fail.
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
