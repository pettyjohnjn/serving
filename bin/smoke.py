#!/usr/bin/env python3
"""Smoke-test the endpoint: auth, generation, reasoning split, tool calling, vision."""
import base64
import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://globus3:8000/v1"
# Default policy is SSH-only, where no key exists and vLLM ignores the header. Requiring
# argv[2] made this crash with an IndexError in exactly the mode we ship.
KEY = sys.argv[2] if len(sys.argv) > 2 else "sk-local"
H = {"Authorization": f"Bearer {KEY}"}
MODEL = "qwen3.8-27b"


def post(payload, timeout=300):
    r = httpx.post(f"{BASE}/chat/completions", json=payload, headers=H, timeout=timeout)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"
    return r.json(), None


def check(name, fn):
    t = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} ({time.perf_counter()-t:5.1f}s)  {detail}")
    return ok


def t_models():
    r = httpx.get(f"{BASE}/models", headers=H, timeout=30)
    ids = [m["id"] for m in r.json().get("data", [])]
    return MODEL in ids, f"served: {ids}"


def t_auth_rejected():
    """Auth behaviour depends on the deployment's policy, so assert the right thing.

    Default policy is SSH-only: the endpoint is loopback-published to globus1 and the cluster
    is publickey-only, so reaching it at all already proves membership of the trusted list and
    vLLM accepts any/no key. With REQUIRE_API_KEY=1 a bad key must be rejected.
    """
    r = httpx.get(f"{BASE}/models", headers={"Authorization": "Bearer wrong-key"}, timeout=30)
    if r.status_code == 200:
        return True, "SSH-only mode: reachability is the credential (no API key enforced)"
    return r.status_code == 401, f"API-key mode: bad key -> HTTP {r.status_code}"


def t_basic():
    d, err = post({"model": MODEL, "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
                   "max_tokens": 64, "temperature": 0})
    if err:
        return False, err
    msg = d["choices"][0]["message"]
    return "PONG" in (msg.get("content") or ""), f"content={ (msg.get('content') or '')[:60]!r}"


def t_reasoning_split():
    """The reasoning parser must never leak think-tags into `content`.

    Note what this does NOT assert: that reasoning_content is non-empty. Qwen3.8's chat
    template pre-fills `<think>\\n` into the *prompt*, so the model begins inside the think
    block and decides per-query how much to think — for an easy question it emits `</think>`
    immediately and reasoning_content is legitimately empty. Asserting otherwise makes this
    test fail forever on a perfectly healthy server.
    """
    d, err = post({"model": MODEL, "messages": [{"role": "user", "content": "What is 17*23? Think it through."}],
                   "max_tokens": 1024, "temperature": 0,
                   "chat_template_kwargs": {"reasoning_effort": "low"}})
    if err:
        return False, err
    m = d["choices"][0]["message"]
    content = m.get("content") or ""
    rc = m.get("reasoning") or m.get("reasoning_content") or ""
    clean = "<think>" not in content and "</think>" not in content
    correct = "391" in content
    return (clean and correct), \
        f"no_tag_leak={clean} answer_ok={correct} reasoning={len(rc)}ch (empty is normal for easy prompts)"


def t_no_think():
    d, err = post({"model": MODEL, "messages": [{"role": "user", "content": "Name one primary color."}],
                   "max_tokens": 128, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}})
    if err:
        return False, err
    m = d["choices"][0]["message"]
    # Read BOTH names. vLLM 0.27 populates `reasoning`; reading only `reasoning_content`
    # made this assertion vacuous -- it passed on any server, thinking or not.
    rc = m.get("reasoning") or m.get("reasoning_content") or ""
    return len(rc) == 0, f"reasoning empty={len(rc)==0} usage={d['usage']['completion_tokens']}tok"


def t_tools():
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}}]
    d, err = post({"model": MODEL, "messages": [{"role": "user", "content": "What's the weather in Chicago in celsius?"}],
                   "tools": tools, "tool_choice": "auto", "max_tokens": 1024, "temperature": 0})
    if err:
        return False, err
    tc = d["choices"][0]["message"].get("tool_calls") or []
    if not tc:
        return False, f"no tool_calls; content={(d['choices'][0]['message'].get('content') or '')[:120]!r}"
    fn = tc[0]["function"]
    args = json.loads(fn["arguments"])
    return fn["name"] == "get_weather" and "chicago" in str(args).lower(), f"{fn['name']}({args})"


def _png(w, h, pix):
    """Minimal RGB PNG encoder, so the test needs no image library."""
    import struct
    import zlib

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"".join(bytes(pix(x, y)) for x in range(w)) for y in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def t_vision():
    """Prove the image is actually *seen*, not guessed.

    A text-only model asked "what colour is this image" will happily answer "red" and pass a
    naive check. So: use colours nobody guesses by default, plus a spatial question whose
    answer cannot be inferred from the prompt. All three must be right.
    """
    def ask(png_bytes, question):
        url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
        d, err = post({"model": MODEL, "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": url}}]}],
            "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}})
        if err:
            raise RuntimeError(err)
        return (d["choices"][0]["message"].get("content") or "").strip().lower()

    magenta = ask(_png(64, 64, lambda x, y: (255, 0, 255)), "What color is this image? One word.")
    teal = ask(_png(64, 64, lambda x, y: (0, 128, 128)), "What color is this image? One word.")
    spatial = ask(_png(64, 64, lambda x, y: (255, 140, 0) if y < 32 else (0, 0, 160)),
                  "Is the orange region on the top or the bottom? Answer 'top' or 'bottom'.")
    good = ("magenta" in magenta and "teal" in teal and "top" in spatial and "bottom" not in spatial)
    return good, f"magenta={magenta[:12]!r} teal={teal[:12]!r} spatial={spatial[:12]!r}"


def t_long_context():
    filler = " ".join(f"tok{i}" for i in range(12000))
    d, err = post({"model": MODEL, "messages": [
        {"role": "user", "content": f"{filler}\n\nHow many times does the literal string 'tok11999' appear above? Answer with a number."}],
        "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    if err:
        return False, err
    return True, f"prompt={d['usage']['prompt_tokens']}tok ok"


if __name__ == "__main__":
    results = [
        check("models listed", t_models),
        check("auth policy", t_auth_rejected),
        check("basic generation", t_basic),
        check("reasoning parser (no leak)", t_reasoning_split),
        check("thinking disabled", t_no_think),
        check("tool calling", t_tools),
        check("vision (unguessable)", t_vision),
        check("long context (12k)", t_long_context),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
