#!/usr/bin/env python3
"""Decode speed with long contexts (the production shape: ~80k-token agent conversations).

For each concurrency C: C distinct prompts of ~--ctx tokens (code haystack, deterministic) are
first prefilled into the prefix cache (max_tokens=1, outside the timing), then re-sent at the same
time with max_tokens=--gen and streamed. Per-stream decode rate = (tokens-1)/(last - first token).
  decode_longctx.py --url http://127.0.0.1:8100 --ladder 1,4,8,16 --ctx 80000 --label tp2
"""
import argparse, glob, json, os, random, statistics, threading, time, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--url", default="http://127.0.0.1:8100")
AP.add_argument("--model", default="qwen3.8-flash-next")
AP.add_argument("--ladder", default="1,4,8,16")
AP.add_argument("--ctx", type=int, default=80000)
AP.add_argument("--gen", type=int, default=256)
AP.add_argument("--label", default="")
AP.add_argument("--out", default="")
AP.add_argument("--src", default=os.environ.get("FN_SCRATCH", "/scratch/models/fn") + "/img/usr/local/lib/python3.12/dist-packages/vllm")
A = AP.parse_args()
FILES = sorted(f for f in glob.glob(A.src + "/**/*.py", recursive=True) if os.path.getsize(f) > 2000)

def prompt(i):
    rng = random.Random(f"dec-{A.ctx}-{i}")
    out, n = [f"Session {i}. Here is part of a codebase:\n"], 0
    while n < A.ctx * 3:                      # ~3 chars/token for source code
        t = open(rng.choice(FILES), errors="replace").read()[:6000]; out.append(t); n += len(t)
    out.append("\n\nWrite a detailed, long explanation of what the code above does, file by file.")
    return "".join(out)

def req(p, max_tokens, stream):
    body = {"model": A.model, "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none", "ignore_eos": True, "stream": stream}
    if stream:   # vLLM rejects stream_options on non-streaming requests
        body["stream_options"] = {"include_usage": True}
    r = urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=3600)

def run_stream(p, res, k):
    t0 = time.time(); first = last = None; n = 0; ptok = 0
    with req(p, A.gen, True) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): ptok = d["usage"].get("prompt_tokens", 0); continue
            if d["choices"] and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning")):
                now = time.time(); first = first or now; last = now; n += 1
    res[k] = dict(ttft=first - t0 if first else None, chunks=n, prompt_tokens=ptok,
                  rate=(A.gen - 1) / (last - first) if first and last > first else None)

out = {}
for C in [int(x) for x in A.ladder.split(",")]:
    ps = [prompt(i) for i in range(C)]
    t = time.time()                            # warm: prefill all C contexts into the cache
    ws = [threading.Thread(target=lambda p=p: req(p, 1, False).read()) for p in ps]
    [w.start() for w in ws]; [w.join() for w in ws]
    warm = time.time() - t
    res = {}
    ts = [threading.Thread(target=run_stream, args=(p, res, i)) for i, p in enumerate(ps)]
    t = time.time(); [x.start() for x in ts]; [x.join() for x in ts]; wall = time.time() - t
    rates = [r["rate"] for r in res.values() if r["rate"]]
    out[C] = dict(per_stream=round(statistics.mean(rates), 1) if rates else None,
                  min_stream=round(min(rates), 1) if rates else None,
                  aggregate=round(sum(rates), 1), ttft_mean=round(statistics.mean(r["ttft"] for r in res.values() if r["ttft"]), 2),
                  prompt_tokens=res[0]["prompt_tokens"], warm_s=round(warm), wall_s=round(wall))
    print(f"DECODE {A.label} ctx={res[0]['prompt_tokens']} C={C} per_stream={out[C]['per_stream']} "
          f"min={out[C]['min_stream']} agg={out[C]['aggregate']} ttft={out[C]['ttft_mean']}s warm={warm:.0f}s", flush=True)
if A.out: json.dump(out, open(A.out, "w"), indent=1)
