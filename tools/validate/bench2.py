#!/usr/bin/env python3
"""Matched ladder: short unique prompts, fixed output length. Reports aggregate
throughput AND pure decode rate (excluding TTFT) so the two are never conflated."""
import json, time, threading, urllib.request, argparse, os

AP = argparse.ArgumentParser()
AP.add_argument("--url", required=True)
AP.add_argument("--model", required=True)
AP.add_argument("--key", default="")
AP.add_argument("--ladder", default="1,2,4")
AP.add_argument("--maxtok", type=int, default=200)
AP.add_argument("--label", default="")
AP.add_argument("--out", default="")
A = AP.parse_args()

TOPICS = ["ocean currents","medieval bridges","fungal networks","radio astronomy",
          "glacier retreat","paper making","desert beetles","clock escapements",
          "salt marshes","kite aerodynamics","bee navigation","volcanic glass",
          "tidal mills","seed dispersal","arctic lichen","cave formations",
          "loom weaving","bird migration","soil microbes","lightning physics",
          "coral spawning","wind erosion","peat bogs","star formation",
          "river deltas","moth camouflage","rope splicing","aurora physics",
          "sponge biology","dune movement","frost heave","spider silk"]

def stream(prompt, max_tokens):
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    hdr = {"Content-Type": "application/json"}
    if A.key:
        hdr["Authorization"] = "Bearer " + A.key
    req = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter(); ttft = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            c = raw[6:].strip()
            if c == b"[DONE]": break
            d = json.loads(c)
            if d.get("usage"): usage = d["usage"].get("completion_tokens")
            for ch in d.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    if ttft is None: ttft = time.perf_counter() - t0
                    n += 1
    tot = time.perf_counter() - t0
    return ttft or tot, tot, (usage or n)

R = {"label": A.label, "ladder": {}}
print(f"== {A.label} ==", flush=True)
for C in [int(x) for x in A.ladder.split(",")]:
    res = [None]*C
    def work(i, C=C):
        t = TOPICS[(hash((C, i, A.label)) % len(TOPICS))]
        res[i] = stream(f"Write a {A.maxtok}-token factual essay about {t}. "
                        f"Variant {C}-{i}. Do not repeat the prompt.", A.maxtok)
    th = [threading.Thread(target=work, args=(i,)) for i in range(C)]
    t0 = time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]
    wall = time.perf_counter() - t0
    ok = [r for r in res if r]
    tok = sum(r[2] for r in ok)
    agg = tok/wall
    # pure decode rate per stream, averaged (excludes prefill/TTFT)
    dec = sum(r[2]/(r[1]-r[0]) for r in ok if r[1] > r[0]) / max(len(ok),1)
    ttft = sum(r[0] for r in ok)/max(len(ok),1)
    R["ladder"][C] = {"aggregate_tok_s": round(agg,1), "decode_tok_s_per_stream": round(dec,1),
                      "agg_decode_tok_s": round(dec*C,1), "mean_ttft_s": round(ttft,2),
                      "wall_s": round(wall,1), "tokens": tok, "n_ok": len(ok)}
    print(f"  C={C:<3} agg {agg:7.1f}  decode/stream {dec:6.1f}  agg-decode {dec*C:7.1f}  ttft {ttft:5.2f}s", flush=True)

print(json.dumps(R, indent=2))
if A.out: open(A.out,"w").write(json.dumps(R, indent=2))
