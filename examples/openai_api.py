#!/usr/bin/env python3
"""Using the globus Qwen3.8-27B endpoint from your own code.

It is a plain OpenAI-compatible server, so this is the same shape as the ALCF inference
service — only the base_url changes. If you already have code written against ALCF, you
usually do not need to touch it at all: just repoint the environment.

    ssh -N -L 8000:127.0.0.1:8000 globus1        # tunnel, in another terminal
    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=sk-local                # any string; SSH is the real auth
    python examples/openai_api.py

`pip install openai` is the only dependency. Every snippet below is runnable.
"""
import base64
import json
import os

from openai import OpenAI

BASE = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL = "qwen3.8-27b"

# api_key is required by the client library but ignored by the server: reaching the
# endpoint at all means you came through an authenticated SSH tunnel.
client = OpenAI(base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY", "sk-local"))


def basic():
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Name the capital of France in one word."}],
        max_tokens=32,
        temperature=0,
    )
    print("basic      :", r.choices[0].message.content.strip())
    print("            ", f"{r.usage.prompt_tokens} prompt + {r.usage.completion_tokens} completion tokens")


def streaming():
    print("streaming  : ", end="", flush=True)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 8, space separated."}],
        max_tokens=64,
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()


def reasoning():
    """Thinking is a per-request knob and the main speed/quality dial.

    The server default is reasoning_effort='low'. Raise it for hard problems; disable it
    entirely for latency-sensitive work. The trace comes back on a separate field, so it
    never contaminates `content`.
    """
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "If 3x + 7 = 22, what is x? Think it through."}],
        max_tokens=2048,
        temperature=0,
        extra_body={"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    )
    msg = r.choices[0].message
    # vLLM 0.27 calls this `reasoning`; other servers (and older vLLM) call it
    # `reasoning_content`. Read both or you will see 0 chars on a model that did think.
    trace = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
    print("reasoning  :", msg.content.strip()[:80])
    print("            ", f"{len(trace)} chars of reasoning (separate from content)")

    fast = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "If 3x + 7 = 22, what is x?"}],
        max_tokens=256,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("no-think   :", fast.choices[0].message.content.strip()[:80])


def tools():
    """Tool calling works server-side (qwen3_xml parser), so agent loops behave normally."""
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a SQL query against the experiments database",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer", "description": "max rows"},
                },
                "required": ["sql"],
            },
        },
    }]
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Get the 5 most recent runs from the runs table."}],
        tools=tool_defs,
        tool_choice="auto",
        max_tokens=1024,
        temperature=0,
    )
    calls = r.choices[0].message.tool_calls or []
    if calls:
        fn = calls[0].function
        print("tools      :", fn.name, json.loads(fn.arguments))
    else:
        print("tools      : (model answered directly)")


def structured_output():
    """Constrained decoding: guarantees parseable JSON instead of hoping for it."""
    schema = {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "bug": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["language", "bug", "severity"],
    }
    # Standard OpenAI `response_format`, not vLLM's older `guided_json` extra_body knob —
    # this is portable, and it is what vLLM 0.27 maps onto its structured-outputs backend.
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Analyse: `def f(x): return x / 0`"}],
        max_tokens=512,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "analysis", "schema": schema, "strict": True},
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("json mode  :", json.loads(r.choices[0].message.content))


def vision():
    """The model is multimodal; up to 4 images per request (video disabled)."""
    import struct
    import zlib

    def png(w, h, rgb):
        def chunk(tag, data):
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

    url = "data:image/png;base64," + base64.b64encode(png(48, 48, (255, 0, 255))).decode()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What colour is this? One word."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        max_tokens=32,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("vision     :", r.choices[0].message.content.strip())


if __name__ == "__main__":
    print(f"endpoint: {BASE}\nmodel   : {MODEL}\n")
    basic()
    streaming()
    reasoning()
    tools()
    structured_output()
    vision()
