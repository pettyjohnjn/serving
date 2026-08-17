#!/usr/bin/env python3
"""The ALCF inference example, pointed at the globus endpoint.

Diff from the ALCF original is two lines — the import and the base_url:

    - from inference_auth_token import get_access_token
    + from globus_inference import get_access_token, BASE_URL
    - base_url="https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1",
    + base_url=BASE_URL,

and the model name (`inkling-bf16` -> `qwen3.8-27b`).

Run it with the tunnel up:
    ssh -N -L 8000:127.0.0.1:8000 globus1
    python examples/alcf_style.py
"""
from openai import OpenAI

from globus_inference import BASE_URL, get_access_token

client = OpenAI(
    api_key=get_access_token(),
    base_url=BASE_URL,
)

r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Reply with just: ok"}],
)
print(r.choices[0].message.content)

r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "What files are in /tmp? Use the tool."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }],
)
print(r.choices[0].message.tool_calls)
