#!/usr/bin/env python3
"""How long do decoding users freeze while someone else's long prompt is prefilled?

N streams decode (short prompts, long outputs); after --delay s a cold prompt of ~--ctx tokens arrives.
Reports, per stream, the longest gap between streamed tokens while the big prefill runs, plus the big
request's TTFT. Each prefill chunk is one engine step, so the gap grows with --max-num-batched-tokens.
  stall_test.py --url http://127.0.0.1:8100 --streams 4 --ctx 100000 --label e2
"""
import argparse, glob, json, os, random, threading, time, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--url", default="http://127.0.0.1:8100")
AP.add_argument("--model", default="qwen3.8-flash-next")
AP.add_argument("--streams", type=int, default=4)
AP.add_argument("--ctx", type=int, default=100000)
AP.add_argument("--delay", type=float, default=8.0)
AP.add_argument("--label", default="")
AP.add_argument("--src", default=os.environ.get("FN_SCRATCH", "/scratch/models/fn") + "/img/usr/local/lib/python3.12/dist-packages/vllm")
A = AP.parse_args()
FILES = sorted(f for f in glob.glob(A.src + "/**/*.py", recursive=True) if os.path.getsize(f) > 2000)

def post(body):
    r = urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=3600)

def decoder(i, out):
    body = {"model": A.model, "messages": [{"role": "user", "content": f"Stream {i}: write a very long story about a lighthouse."}],
            "max_tokens": 2500, "temperature": 0, "reasoning_effort": "none", "ignore_eos": True, "stream": True}
    ts = []
    with post(body) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line: ts.append(time.time())
    out[i] = ts

def big(out):
    rng = random.Random(f"stall-{A.ctx}-{time.time()}")   # cold on purpose: unique every run
    parts, n = [f"Unique run {time.time()}\n"], 0
    while n < A.ctx * 3.75:
        t = open(rng.choice(FILES), errors="replace").read()[:6000]; parts.append(t); n += len(t)
    parts.append("\n\nSummarize the code above in one sentence.")
    t0 = time.time()
    with post({"model": A.model, "messages": [{"role": "user", "content": "".join(parts)}], "max_tokens": 20,
               "temperature": 0, "reasoning_effort": "none", "stream": True}) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                out["ttft"] = time.time() - t0; out["start"] = t0; out["first"] = time.time(); break
        for _ in r: pass

res = {}
ds = [threading.Thread(target=decoder, args=(i, res)) for i in range(A.streams)]
[d.start() for d in ds]
time.sleep(A.delay)
bres = {}
b = threading.Thread(target=big, args=(bres,)); b.start(); b.join(); [d.join() for d in ds]
win0, win1 = bres["start"], bres["first"]
gaps = []
for i in range(A.streams):
    ts = [t for t in res[i] if win0 - 1 <= t <= win1 + 1]
    g = max((b - a for a, b in zip(ts, ts[1:])), default=float("nan"))
    gaps.append(g)
print(f"STALL {A.label} streams={A.streams} big_ctx~{A.ctx} big_ttft={bres['ttft']:.1f}s "
      f"max_gap={max(gaps):.2f}s mean_max_gap={sum(gaps)/len(gaps):.2f}s", flush=True)
