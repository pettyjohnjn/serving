#!/usr/bin/env python3
"""chat.py — minimal client for the globus Qwen3.8-27B endpoint. Stdlib only.

Copy this to your laptop. No pip install, no dependencies, no API key.

    ./chat.py                        interactive chat (multi-turn, streaming)
    ./chat.py "why is the sky blue"  one-shot
    cat bug.py | ./chat.py "fix this"    pipe a file in as context

Options:
    --think            let the model reason first (slower, better on hard problems)
    --effort xhigh     low (default) | medium | xhigh
    --port 8000        local port your tunnel is on
    --raw              print the reasoning trace too
    --system "..."     set a system prompt
    --max-tokens 8192  raise the budget (thinking spends from it too)

Needs a tunnel to the cluster first:
    ssh -N -L 8000:127.0.0.1:8000 globus1
"""
import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.request

MODEL = "qwen3.8-27b"   # fallback only; main() asks the server what it is serving


def _served_model(base):
    """The endpoint serves whichever profile was selected (see etc/models/), so ask it
    rather than hardcode a name. CHAT_MODEL overrides. A server that is down falls
    back to the default; the request itself then fails with the usual message."""
    if os.environ.get("CHAT_MODEL"):
        return os.environ["CHAT_MODEL"]
    try:
        req = urllib.request.Request(f"{base}/models",
                                     headers={"Authorization": "Bearer sk-local"})
        with urllib.request.urlopen(req, timeout=5) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
        if ids:
            return ids[0]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return MODEL


class EndpointError(Exception):
    """Anything that stopped us talking to the endpoint, phrased for a human."""


def _unreachable(base, detail):
    port = base.split("//", 1)[1].split("/", 1)[0].rsplit(":", 1)[-1]
    return EndpointError(
        f"cannot reach {base}\n  {detail}\n"
        f"  is the tunnel up?  ssh -N -L {port}:127.0.0.1:8000 globus1\n"
        f"  is the server up?  ssh globus1 serving status")


def stream(messages, base, think, effort, show_reasoning, max_tokens=4096):
    """POST to /chat/completions and yield (kind, text) as tokens arrive."""
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "chat_template_kwargs": ({"reasoning_effort": effort} if think
                                 else {"enable_thinking": False}),
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 # Auth is your SSH tunnel; vLLM ignores this, but some proxies want it.
                 "Authorization": "Bearer sk-local"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        raise EndpointError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
    except urllib.error.URLError as e:
        raise _unreachable(base, e.reason) from None
    # RemoteDisconnected/ConnectionReset are OSError, NOT URLError. This is the normal
    # response when the tunnel is up but the far side has nothing listening yet (server
    # still loading): sshd accepts the connection, then closes it without a reply.
    except OSError as e:
        raise _unreachable(base, f"{type(e).__name__}: {e} (server may still be loading)") from None

    try:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                # vLLM 0.27 streams the trace as `reasoning`, but the non-streaming
                # response body calls the same thing `reasoning_content`. Accept both:
                # checking only the latter silently shows nothing while the model thinks.
                think_delta = delta.get("reasoning") or delta.get("reasoning_content")
                if show_reasoning and think_delta:
                    yield "reasoning", think_delta
                if delta.get("content"):
                    yield "content", delta["content"]
    except OSError as e:
        # Dropped mid-generation (tunnel died, job requeued). Don't lose the session.
        raise EndpointError(f"connection lost mid-reply ({type(e).__name__}: {e})") from None


def ask(messages, args, base):
    """Stream one reply and print it. Returns the text, or None if the endpoint failed."""
    out, mode = [], None
    try:
        for kind, text in stream(messages, base, args.think, args.effort, args.raw, args.max_tokens):
            if kind != mode:
                if kind == "reasoning":
                    sys.stdout.write("\033[2m[thinking] ")
                elif mode == "reasoning":
                    sys.stdout.write("\033[0m\n")
                mode = kind
            sys.stdout.write(text)
            sys.stdout.flush()
            if kind == "content":
                out.append(text)
    except EndpointError as e:
        if mode:
            sys.stdout.write("\033[0m\n")
        print(f"\n\033[31m{e}\033[0m")
        return None
    if mode == "reasoning":
        sys.stdout.write("\033[0m")
    print()
    return "".join(out)


def main():
    global MODEL
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("prompt", nargs="*")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--think", action="store_true")
    p.add_argument("--effort", default="low", choices=["low", "medium", "xhigh"])
    p.add_argument("--raw", action="store_true", help="show the reasoning trace")
    # Thinking is spent from the SAME budget as the answer, so a low cap can consume the
    # whole allowance before the model ever emits a final answer.
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--system", default=None)
    p.add_argument("-h", "--help", action="store_true")
    args = p.parse_args()
    if args.help:
        print(__doc__)
        return
    if args.effort != "low":
        args.think = True

    base = f"http://127.0.0.1:{args.port}/v1"
    MODEL = _served_model(base)
    messages = [{"role": "system", "content": args.system}] if args.system else []

    # Anything piped in becomes context for the prompt.
    #
    # Deliberately NOT `if not sys.stdin.isatty()`: stdin is also a non-tty when it is
    # /dev/null or an inherited descriptor with no writer (ssh without -t, cron, some IDE
    # terminals), and read() would then block forever with no output. Only a pipe or a
    # redirected file actually has content coming, so test for those specifically.
    piped = ""
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
        if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):
            piped = sys.stdin.read()
    except (OSError, ValueError):
        pass
    prompt = " ".join(args.prompt)

    if prompt or piped:
        content = f"{prompt}\n\n{piped}".strip() if piped else prompt
        messages.append({"role": "user", "content": content})
        sys.exit(0 if ask(messages, args, base) is not None else 1)

    print(f"{MODEL} via {base}   (Ctrl-D or 'exit' to quit, 'reset' to clear history)")
    print(f"thinking: {'on, effort=' + args.effort if args.think else 'off'}\n")
    while True:
        try:
            line = input("\033[1m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("exit", "quit"):
            return
        if line == "reset":
            messages = [m for m in messages if m["role"] == "system"]
            print("history cleared\n")
            continue
        messages.append({"role": "user", "content": line})
        try:
            reply = ask(messages, args, base)
        except KeyboardInterrupt:
            print("\n[interrupted]\n")
            messages.pop()
            continue
        if reply is None:      # endpoint problem: forget the turn, stay in the session
            messages.pop()
            print()
            continue
        messages.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
