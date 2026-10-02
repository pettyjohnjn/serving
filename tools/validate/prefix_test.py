#!/usr/bin/env python3
"""Prefix-cache check: send one deterministic ~20k-token prompt twice (then a variant sharing its first half),
measure TTFT and read vLLM's prefix_cache_{queries,hits}_total from /metrics between calls."""
import sys, json, time, random, urllib.request, re
BASE, MODEL = sys.argv[1], sys.argv[2]               # BASE like http://127.0.0.1:8100
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 15000
VOCAB = "alpha beta gamma delta sensor reading valve pressure temperature cycle status nominal offset drift calibration log entry node cluster".split()
def text(seed, words):
    r = random.Random(seed); return " ".join(r.choice(VOCAB) + (str(r.randint(0, 999)) if r.random() < 0.2 else "") for _ in range(words))
def metrics():
    m = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    q = sum(float(x) for x in re.findall(r'^vllm:prefix_cache_queries_total\{[^}]*\} ([0-9.e+]+)', m, re.M))
    h = sum(float(x) for x in re.findall(r'^vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)', m, re.M))
    return q, h
def ask(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt + "\n\nReply with the single word: done"}],
            "max_tokens": 8, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ptoks = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
            d = json.loads(line[6:])
            if ttft is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"): ttft = time.time() - t0
            if d.get("usage"): ptoks = d["usage"].get("prompt_tokens")
    return ttft, ptoks
A = text(11, WORDS); B = text(11, WORDS // 2) + " " + text(12, WORDS // 2)   # B shares A's first half exactly
PAUSE = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
C = text(13, WORDS)   # fresh prompt for the paused sequence
seq = (("A-cold", A, 0), ("A-repeat", A, 0), ("A-repeat2", A, 0), ("B-halfshared", B, 0),
       ("C-cold", C, PAUSE), ("C-repeat-after-pause", C, PAUSE), ("C-repeat2-after-pause", C, PAUSE), ("A-again-after-pause", A, PAUSE))
for label, p, pause in seq:
    if pause: time.sleep(pause)
    q0, h0 = metrics(); ttft, ptoks = ask(p); q1, h1 = metrics()
    print(f"PREFIX {label}: prompt_tokens={ptoks} ttft={ttft:.2f}s  cache queries +{q1-q0:.0f} hits +{h1-h0:.0f}  ({(h1-h0)/max(1,q1-q0)*100:.0f}% of this request)", flush=True)
