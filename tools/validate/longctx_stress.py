#!/usr/bin/env python3
"""Long-context memory stress: C concurrent chat requests with ~T prompt tokens each of varied text,
max_tokens small. Reports per-request prompt tokens (from usage), TTFT, errors, and the minimum
MemAvailable / MemFree seen on this node during the run (sampled every 2 s)."""
import sys, json, time, threading, random, urllib.request
BASE, MODEL = sys.argv[1], sys.argv[2]
C = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WORDS = int(sys.argv[4]) if len(sys.argv) > 4 else 90000     # ~1.3 tok/word -> ~120k tokens
random.seed(7)
VOCAB = ("the of and to in is that for on with as by at from this be are was were it an or which " 
         "system model memory cache stream context token layer expert table node cluster request "
         "measure result policy budget schedule sample matrix vector kernel graph batch prefill decode "
         "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma omega").split()
def prompt(i):
    r = random.Random(1000 + i)
    body = " ".join(r.choice(VOCAB) + (str(r.randint(0, 999)) if r.random() < 0.15 else "") for _ in range(WORDS))
    return f"Document {i}:\n{body}\n\nIn one sentence, what is this document about?"
mem = {"min_avail": 1e12, "min_free": 1e12, "stop": False}
def sampler():
    while not mem["stop"]:
        try:
            d = dict(l.split(":")[0:1] + [l.split()[1]] for l in open("/proc/meminfo") if l.startswith(("MemFree", "MemAvailable")))
            mem["min_free"] = min(mem["min_free"], int(d["MemFree"]) / 1048576)
            mem["min_avail"] = min(mem["min_avail"], int(d["MemAvailable"]) / 1048576)
        except Exception: pass
        time.sleep(2)
res = [None] * C
def work(i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 48,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer sk-local"})
    t0 = time.time(); ttft = None; ptoks = None; ntok = 0; err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
                d = json.loads(line[6:])
                if d.get("usage"): ptoks = d["usage"].get("prompt_tokens")
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    if ttft is None: ttft = time.time() - t0
                    ntok += 1
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    res[i] = (ptoks, ttft, ntok, time.time() - t0, err)
threading.Thread(target=sampler, daemon=True).start()
th = [threading.Thread(target=work, args=(i,)) for i in range(C)]
t0 = time.time()
for t in th: t.start()
for t in th: t.join()
mem["stop"] = True
ok = 0
for i, (p, ttft, n, tot, err) in enumerate(res):
    print(f"  req{i}: prompt_tokens={p} ttft={ttft and round(ttft,1)}s out={n} total={tot:.0f}s {err or 'ok'}")
    ok += err is None and n > 0
print(f"STRESS: {ok}/{C} ok  wall={time.time()-t0:.0f}s  total_prompt_tokens={sum((r[0] or 0) for r in res):,}  "
      f"min MemAvailable={mem['min_avail']:.1f}G  min MemFree={mem['min_free']:.1f}G")
