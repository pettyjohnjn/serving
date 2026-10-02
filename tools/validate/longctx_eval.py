#!/usr/bin/env python3
"""Long-context retrieval eval for the KV-precision decision (fp8 vs bf16).

RULER-style tasks embedded in real source code (vLLM's own .py files from the image tree, so
the haystack is what agentic coding traffic looks like), one request per haystack, six answers
per request in JSON:
  k1..k4  single-key lookup at depths spread through the context
  mv      multi-value: all four values filed under one shared key
  vt      variable tracking: a 4-hop chain of assignments scattered through the context
Prompts are deterministic (seeded), so two engines see byte-identical inputs.

  longctx_eval.py --url http://127.0.0.1:8100 --label fp8 --lengths 32000,64000,128000,200000 --n 10
Prints one LONGCTX line per length and writes per-item results to --out.
"""
import argparse, glob, json, os, random, re, threading, time, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--url", default="http://127.0.0.1:8100")
AP.add_argument("--model", default="qwen3.8-flash-next")
AP.add_argument("--lengths", default="32000,64000,128000,200000")
AP.add_argument("--n", type=int, default=10)
AP.add_argument("--conc", type=int, default=2)
AP.add_argument("--label", default="")
AP.add_argument("--out", default="")
AP.add_argument("--hard", action="store_true")
AP.add_argument("--src", default=os.environ.get("FN_SCRATCH", "/scratch/models/fn") + "/img/usr/local/lib/python3.12/dist-packages/vllm")
A = AP.parse_args()

FILES = sorted(f for f in glob.glob(A.src + "/**/*.py", recursive=True) if os.path.getsize(f) > 2000)
WORDS = ("amber basil cobalt delta ember fjord garnet harbor indigo juniper kestrel lumen maple nickel "
         "onyx pepper quartz raven saffron tundra umber velvet willow xenon yarrow zephyr").split()

def post(path, body, timeout=1800):
    r = urllib.request.Request(A.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def ntok(text):
    return post("/tokenize", {"model": A.model, "prompt": text})["count"]

def code_chunks(rng, nchars):
    out, total = [], 0
    while total < nchars:
        f = rng.choice(FILES)
        txt = open(f, errors="replace").read()
        if len(txt) > 6000:
            s = rng.randrange(0, len(txt) - 6000); txt = txt[s:s + 6000]
        out.append(f"\n# ===== file: {f.split('/vllm/', 1)[-1]} =====\n" + txt)
        total += len(txt)
    return out

# Difficulty presets. hard: near-duplicate decoys for every asked key (digits permuted, words
# swapped), 200 registry lines, 10 lookups, an 8-hop chain with decoy branches, 8 multi-values.
P = dict(nk=4, nmv=4, hops=4, ndist=12, decoys=0) if not A.hard else dict(nk=10, nmv=8, hops=8, ndist=150, decoys=3)

def build(length, idx):
    rng = random.Random(f"{length}-{idx}" + ("-hard" if A.hard else ""))
    def key():
        return f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randrange(100, 999)}"
    val = lambda: str(rng.randrange(1_000_000, 9_999_999))
    def decoys(k):
        a, b, n = k.split("-"); out = set()
        while len(out) < P["decoys"]:
            c = rng.choice((f"{b}-{a}-{n}", f"{a}-{b}-{''.join(rng.sample(n, 3))}", f"{a}-{rng.choice(WORDS)}-{n}"))
            if c != k: out.add(c)
        return out
    singles = [(key(), val()) for _ in range(P["nk"])]
    mv_key, mv_vals = key(), [val() for _ in range(P["nmv"])]
    names = [f"VAR_{rng.choice(WORDS).upper()}_{rng.randrange(10, 99)}" for _ in range(P["hops"] + 1)]
    chain_val = val()
    facts = [(rng.uniform(0.02, 0.98), f"# registry: the access code for {k} is {v}.") for k, v in singles]
    facts += [(rng.uniform(0.02, 0.98), f"# registry: one access code filed under {mv_key} is {v}.") for v in mv_vals]
    hop_d = sorted(rng.uniform(0.02, 0.98) for _ in range(P["hops"] + 1))
    facts += [(hop_d[0], f"# assign: {names[0]} = {chain_val}")]
    facts += [(hop_d[i + 1], f"# assign: {names[i + 1]} = {names[i]}") for i in range(P["hops"])]
    if A.hard:   # decoy branches: other variables assigned from chain members, and a decoy root
        for i in range(0, P["hops"], 2):
            facts.append((rng.uniform(0.02, 0.98), f"# assign: VAR_{rng.choice(WORDS).upper()}_{rng.randrange(10, 99)} = {names[i]}"))
        facts.append((rng.uniform(0.02, 0.98), f"# assign: {names[0]}_OLD = {val()}"))
        for k, _ in singles + [(mv_key, None)]:
            for d in decoys(k):
                facts.append((rng.uniform(0.02, 0.98), f"# registry: the access code for {d} is {val()}."))
    facts += [(rng.random(), f"# registry: the access code for {key()} is {val()}.") for _ in range(P["ndist"])]
    chars = int(length * 3.0)                                # source code is ~3 chars/token here
    for _ in range(4):                                       # size to the token target with /tokenize
        chunks = code_chunks(random.Random(f"hay-{length}-{idx}"), chars)
        lines = "".join(chunks).split("\n")
        for d, line in sorted(facts, key=lambda t: -t[0]):
            lines.insert(int(d * len(lines)), line)
        doc = "\n".join(lines)
        n = ntok(doc)
        if abs(n - length) / length < 0.03: break
        chars = int(chars * length / n)
    fields = ", ".join(f'"k{i+1}": access code for {k}' for i, (k, _) in enumerate(singles))
    q = (f"\n\n----- end of codebase -----\n\nThe codebase above contains comment lines starting with '# registry:' "
         f"and '# assign:'. Answer from those lines only; keys must match exactly. Reply with ONLY a JSON object, no prose:\n"
         f'{{{fields}, "mv": list of ALL access codes filed under {mv_key}, '
         f'"vt": the numeric value of {names[-1]} after following the assign lines}}\n'
         f"Use strings for codes.")
    gold = {f"k{i+1}": v for i, (_, v) in enumerate(singles)}
    gold.update(mv=sorted(mv_vals), vt=chain_val)
    return doc + q, gold, n

def score(text, gold):
    m = re.search(r"\{.*\}", text or "", re.S)
    try: ans = json.loads(m.group(0)) if m else {}
    except Exception: ans = {}
    s = {}
    for k in [k for k in gold if k not in ("mv",)]:
        s[k] = int(str(ans.get(k, "")).strip() == gold[k])
    got = ans.get("mv", [])
    got = sorted(str(x).strip() for x in got) if isinstance(got, list) else []
    s["mv"] = len(set(got) & set(gold["mv"])) / len(gold["mv"])   # partial credit per value
    s["mv_exact"] = int(got == gold["mv"])
    return s, ans

results, lock = [], threading.Lock()
def work(jobs):
    while True:
        with lock:
            if not jobs: return
            length, idx = jobs.pop(0)
        prompt, gold, n = build(length, idx)
        t = time.time()
        try:
            r = post("/v1/chat/completions", {"model": A.model, "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": 600, "temperature": 0, "reasoning_effort": "none"})
            text = r["choices"][0]["message"].get("content") or ""
            err = ""
        except Exception as e:
            text, err = "", repr(e)[:200]
        s, ans = score(text, gold)
        rec = dict(length=length, idx=idx, tokens=n, secs=round(time.time() - t, 1), score=s, gold=gold, answer=ans,
                   raw=text[:400], err=err)
        with lock:
            results.append(rec)
            print(f"ITEM len={length} idx={idx} tok={n} {time.time()-t:.0f}s "
                  + " ".join(f"{k}={v}" for k, v in s.items()) + (f" ERR {err}" if err else ""), flush=True)

lengths = [int(x) for x in A.lengths.split(",")]
jobs = [(L, i) for L in lengths for i in range(A.n)]
ts = [threading.Thread(target=work, args=(jobs,)) for _ in range(A.conc)]
[t.start() for t in ts]; [t.join() for t in ts]
for L in lengths:
    rs = [r for r in results if r["length"] == L]
    if not rs: continue
    avg = lambda k: sum(r["score"][k] for r in rs) / len(rs)
    ks = [k for k in rs[0]["score"] if re.fullmatch(r"k\d+", k)]
    single = sum(avg(k) for k in ks) / len(ks)
    print(f"LONGCTX {A.label} len={L} n={len(rs)} single={single:.3f} mv={avg('mv'):.3f} mv_exact={avg('mv_exact'):.3f} "
          f"vt={avg('vt'):.3f} errors={sum(1 for r in rs if r['err'])}", flush=True)
if A.out:
    json.dump(sorted(results, key=lambda r: (r["length"], r["idx"])), open(A.out, "w"), indent=1)
