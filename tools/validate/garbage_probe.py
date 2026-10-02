"""Concurrent-prefill corruption probe for vLLM PR #55375 (fused PLE conv state-index
stride). Fires N cold ~6k-token prompts at once so several prefills batch in one step
with MTP on, then flags any reply that degenerates into a repeated-token loop."""
import json, sys, threading, urllib.request, collections, re
BASE, MODEL, N = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6
H = {"Authorization": "Bearer sk-local", "Content-Type": "application/json"}
def prompt(i):
    body = "\n".join(f"entry {i}-{j:04d}: sensor reading {j*7 % 97}, status nominal, note {j%5}." for j in range(520))
    return f"Log file #{i}:\n{body}\n\nIn three sentences, what does this log describe?"
out = [None] * N
def go(i):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 160, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(b).encode(), headers=H), timeout=600))
        out[i] = r["choices"][0]["message"]["content"] or ""
    except Exception as e:
        out[i] = f"<<error {e}>>"
ts = [threading.Thread(target=go, args=(i,)) for i in range(N)]
[t.start() for t in ts]; [t.join() for t in ts]
bad = 0
for i, txt in enumerate(out):
    words = re.findall(r"\w+", txt.lower()); uniq = len(set(words)) / max(len(words), 1)
    runs = max((len(m.group(0)) for m in re.finditer(r"(\b\w+\b)(\W+\1){4,}", txt)), default=0)
    err = txt.startswith("<<error"); loop = (uniq < 0.35 and len(words) > 30) or runs > 0
    bad += loop or err
    print(f"  req{i}: {len(words):3d} words  unique-ratio {uniq:.2f}  {'LOOP/GARBAGE' if loop else ('ERROR' if err else 'ok')}  | {txt[:70]!r}")
errs = sum(1 for t in out if t.startswith("<<error")); loops = bad - errs
print(f"RESULT: {N-bad}/{N} clean" + (f"   <-- {loops} garbage/loop" if loops else "") + (f"   <-- {errs} error/timeout (engine hung or died, not corruption)" if errs else ""))
