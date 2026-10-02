#!/usr/bin/env python3
"""Does the NVMe KV tier bring evicted sessions back fast and unchanged?

1. Prefill --n distinct ~--ctx-token conversations (more tokens than the GPU pool), --conc at a time. The first
   --probe of them also generate --gen greedy tokens, recorded as the reference.
2. Re-send those first --probe conversations one at a time. Their GPU blocks were evicted by step 1, so they come
   back from the offload tier (connector hits = vllm:external_prefix_cache_hits_total) or are recomputed.
3. Report TTFT cold vs returning, connector hit tokens, and whether the greedy output matches the reference.
  offload_test.py --url http://127.0.0.1:8100 --n 40 --ctx 100000 --probe 4 --label e4-offload
"""
import argparse, glob, json, os, random, re, threading, time, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--url", default="http://127.0.0.1:8100")
AP.add_argument("--model", default="qwen3.8-flash-next")
AP.add_argument("--n", type=int, default=40)
AP.add_argument("--ctx", type=int, default=100000)
AP.add_argument("--probe", type=int, default=4)
AP.add_argument("--gen", type=int, default=64)
AP.add_argument("--conc", type=int, default=4)
AP.add_argument("--label", default="")
AP.add_argument("--src", default=os.environ.get("FN_SCRATCH", "/scratch/models/fn") + "/img/usr/local/lib/python3.12/dist-packages/vllm")
A = AP.parse_args()
FILES = sorted(f for f in glob.glob(A.src + "/**/*.py", recursive=True) if os.path.getsize(f) > 2000)

def prompt(i):
    rng = random.Random(f"offload-{A.ctx}-{i}")
    parts, n = [f"Conversation {i}. Here is a codebase:\n"], 0
    while n < A.ctx * 3.75:
        t = open(rng.choice(FILES), errors="replace").read()[:6000]; parts.append(t); n += len(t)
    parts.append("\n\nList the five most important functions defined above and what each does.")
    return "".join(parts)

def chat(i, max_tokens):
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; usage = {}
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                piece = c["delta"].get("content")
                if piece:
                    ttft = ttft or time.time() - t0; text.append(piece)
    return dict(ttft=ttft, total=time.time() - t0, text="".join(text), usage=usage)

def metrics():
    m = urllib.request.urlopen(A.url + "/metrics", timeout=30).read().decode()
    get = lambda name: sum(float(x) for x in re.findall(rf"^{name}{{[^}}]*}} ([0-9.e+]+)$", m, re.M))
    return {k: get(f"vllm:{k}") for k in ("prefix_cache_hits_total", "prefix_cache_queries_total",
                                           "external_prefix_cache_hits_total", "external_prefix_cache_queries_total",
                                           "num_preemptions_total")}

m0 = metrics(); ref = {}; cold = {}
jobs = list(range(A.n)); lock = threading.Lock()
def worker():
    while True:
        with lock:
            if not jobs: return
            i = jobs.pop(0)
        r = chat(i, A.gen if i < A.probe else 1)
        with lock:
            cold[i] = r
            if i < A.probe: ref[i] = r["text"]
            print(f"FILL i={i} prompt_tokens={r['usage'].get('prompt_tokens')} ttft={r['ttft']:.1f}s", flush=True)
t = time.time(); ws = [threading.Thread(target=worker) for _ in range(A.conc)]; [w.start() for w in ws]; [w.join() for w in ws]
total = sum(r["usage"].get("prompt_tokens", 0) for r in cold.values())
print(f"FILLED {A.n} conversations, {total:,} prompt tokens in {time.time()-t:.0f}s", flush=True)
m1 = metrics()
for i in range(A.probe):
    before = metrics(); r = chat(i, A.gen); after = metrics()
    ext = after["external_prefix_cache_hits_total"] - before["external_prefix_cache_hits_total"]
    loc = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
    print(f"RETURN i={i} ttft={r['ttft']:.2f}s (cold {cold[i]['ttft']:.1f}s) gpu_hits={loc:,.0f} "
          f"offload_hits={ext:,.0f} of {r['usage'].get('prompt_tokens')} same_output={r['text'] == ref[i]}", flush=True)
m2 = metrics()
print(f"OFFLOAD {A.label} preemptions={m2['num_preemptions_total']-m0['num_preemptions_total']:.0f} "
      f"connector_hits_total={m2['external_prefix_cache_hits_total']-m0['external_prefix_cache_hits_total']:,.0f}", flush=True)
