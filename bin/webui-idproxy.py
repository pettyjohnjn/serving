#!/usr/bin/env python3
"""webui-idproxy — log people into Open WebUI as the cluster account they SSHed in with.

How the identity is known: a user reaches the UI through `ssh -L 8080:...`, and the
process on this host that opens the forwarded connection is their own sshd, running
under their uid. The kernel records that uid per socket, so the owner of the peer
socket in /proc/net/tcp IS the tunneling user — not guessable, not spoofable from the
laptop side. This proxy resolves it per connection and forwards every request to the
Open WebUI backend with the trusted-auth headers it expects
(WEBUI_AUTH_TRUSTED_EMAIL_HEADER / _NAME_HEADER). Accounts are auto-created on first
visit as <unix-user>@globus.local; there is no signup or password.

Incoming X-Globus-* headers are always stripped, so a browser cannot claim an identity.
Requests and responses stream through unmodified otherwise; websocket upgrades switch
the connection to a blind pipe after the handshake request.

Run by bin/webui; not meant to be started by hand.
"""
import asyncio
import json
import os
import pwd
import socket
import sys
import time
import urllib.request

LISTEN = ("127.0.0.1", 8080)
# The backend listens on a unix socket inside the 0700 data dir — see bin/webui.
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
UPSTREAM_SOCKET = ROOT + "/webui-data/webui.sock"
# Engine metrics come from the published port directly: probing through the
# fairness gateway would bill every sampler fetch to the operator's usage ledger.
VLLM_METRICS = "http://127.0.0.1:8005/metrics"
FAIR_STATS = "http://127.0.0.1:8000/fair-stats"
EMAIL_HEADER = "X-Globus-Email"
NAME_HEADER = "X-Globus-Name"
MAX_HEADER = 256 * 1024

# ---- live stats page (served by the proxy itself, not Open WebUI) -----------------
# GET /globus-stats       tiny self-contained page, polls the JSON below
# GET /globus-stats.json  snapshot of the vLLM counters that matter to users
# GET /globus-usage.json  accumulated token/uptime history (see next section)
# tok/s is computed client-side from successive counter deltas, so the page shows the
# LIVE cluster rate, not a lifetime average.

STATS_KEYS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prefix_cache_queries_total": "prefix_queries",
    "vllm:prefix_cache_hits_total": "prefix_hits",
    "vllm:request_success_total": "requests",
    "vllm:time_to_first_token_seconds_sum": "ttft_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:inter_token_latency_seconds_sum": "itl_sum",
    "vllm:inter_token_latency_seconds_count": "itl_count",
}


NODE_STATS = ROOT + "/logs/node-stats.json"

# ---- token usage + availability accounting ----------------------------------------
# vLLM's prompt/generation counters reset to zero on every engine restart, and the
# slurm job requeues at least every 2 days, so lifetime/window figures cannot be read
# off /metrics directly. A background task samples the counters every 30 s,
# accumulates the deltas into hourly buckets, and persists them across restarts of
# both vLLM and this proxy. A counter that went backwards means the engine restarted:
# the post-restart value IS the delta (whatever ran between the last sample and the
# restart is lost — bounded by one sample interval). Each probe also records whether
# the endpoint answered, which is what the uptime figures are made of — availability
# as seen from the login node, while this proxy is running to observe it.
#
# GET /globus-usage.json serves the derived view; the stats page draws it.
USAGE_FILE = ROOT + "/logs/token-usage.json"
USAGE_SAMPLE_S = 30
USAGE_KEEP_H = 92 * 24                 # hourly buckets kept; the strip shows 90 days
USAGE_VIEW = {}                        # latest derived view, replaced atomically


def usage_load():
    try:
        with open(USAGE_FILE) as f:
            s = json.load(f)
        if isinstance(s.get("hours"), dict) and "lifetime" in s:
            for v in s["hours"].values():           # older, shorter layouts
                while len(v) < 5:
                    v.append(0)
            s["lifetime"].setdefault("up", 0)
            s["lifetime"].setdefault("total", 0)
            s["lifetime"].setdefault("reqs", 0)
            return s
    except (OSError, ValueError):
        pass
    return {"since": time.time(),
            "lifetime": {"prompt": 0.0, "gen": 0.0, "up": 0, "total": 0, "reqs": 0},
            "last": None, "hours": {}}


def usage_update(state, snap, now):
    """Fold one stats_snapshot into the hourly buckets: [prompt, gen, up, probes, reqs]."""
    hour = state["hours"].setdefault(str(int(now // 3600)), [0.0, 0.0, 0, 0, 0])
    hour[3] += 1
    state["lifetime"]["total"] += 1
    if snap.get("up"):
        hour[2] += 1
        state["lifetime"]["up"] += 1
    if snap.get("up") and "prompt_tokens" in snap and "generation_tokens" in snap:
        cur = [snap["prompt_tokens"], snap["generation_tokens"], snap.get("requests", 0.0)]
        last = state.get("last")
        if last is not None:
            while len(last) < 3:
                last.append(0.0)
            d = [c - l for c, l in zip(cur, last)]
            if min(d) < 0:                     # engine restarted, counters reset
                d = cur
            hour[0] += d[0]
            hour[1] += d[1]
            hour[4] += d[2]
            state["lifetime"]["prompt"] += d[0]
            state["lifetime"]["gen"] += d[1]
            state["lifetime"]["reqs"] += d[2]
        state["last"] = cur
    cutoff = int(now // 3600) - USAGE_KEEP_H
    for k in [k for k in state["hours"] if int(k) < cutoff]:
        del state["hours"][k]


def usage_derive(state, now):
    cur = int(now // 3600)

    def window(nh):
        p = g = up = total = r = 0
        for k, v in state["hours"].items():
            if int(k) > cur - nh:
                p += v[0]
                g += v[1]
                up += v[2]
                total += v[3]
                r += v[4]
        return {"prompt": p, "gen": g, "up": up, "total": total, "reqs": r}

    # hourly rows feed both the usage chart (prompt, gen) and the 90-day
    # availability strip (up, probes)
    hourly = sorted((int(k), round(v[0]), round(v[1]), v[2], v[3])
                    for k, v in state["hours"].items())
    return {"since": state["since"], "lifetime": dict(state["lifetime"]),
            "day": window(24), "week": window(24 * 7), "month": window(24 * 30),
            "hourly": hourly}


def usage_save(state):
    tmp = USAGE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, USAGE_FILE)


async def usage_sampler():
    global USAGE_VIEW
    loop = asyncio.get_running_loop()
    state = usage_load()
    USAGE_VIEW = usage_derive(state, time.time())
    while True:
        try:
            snap = await loop.run_in_executor(None, stats_snapshot)
            now = time.time()
            usage_update(state, snap, now)
            await loop.run_in_executor(None, usage_save, state)
            USAGE_VIEW = usage_derive(state, now)
        except Exception:
            pass
        await asyncio.sleep(USAGE_SAMPLE_S)


def fetch_webui_ok():
    """One-line health check against the backend's unix socket."""
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.settimeout(2)
        c.connect(UPSTREAM_SOCKET)
        c.sendall(b"GET /health HTTP/1.1\r\nHost: webui\r\nConnection: close\r\n\r\n")
        head = c.recv(1024)
        c.close()
        return b" 200" in head.split(b"\r\n", 1)[0]
    except OSError:
        return False


def stats_snapshot():
    out = {"t": time.time(), "up": False, "max_context": 262144}
    # GPU/host figures are written by the slurm job on the compute node every 10 s
    # (NFS-shared logs dir); stale entries mean the job is down or reloading.
    try:
        with open(NODE_STATS) as f:
            node = json.load(f)
        if time.time() - node.get("t", 0) < 60:
            out["node"] = node
    except (OSError, ValueError):
        pass
    out["webui_ok"] = fetch_webui_ok()
    try:
        d = json.loads(urllib.request.urlopen(FAIR_STATS, timeout=3).read())
        out["fair"] = {"ok": True, "users": len(d.get("in_flight", {}))}
    except (OSError, ValueError):
        out["fair"] = {"ok": False}
    try:
        text = urllib.request.urlopen(VLLM_METRICS, timeout=5).read().decode()
    except OSError:
        out["engine"] = False
        return out
    out["engine"] = True
    # "up" — what the uptime history records — means a user request would succeed:
    # the engine is answering AND the gateway in front of it is alive.
    out["up"] = bool(out["fair"]["ok"])
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0]
        key = STATS_KEYS.get(name)
        if key:
            try:
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


STATS_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Globus Cluster Inference — status</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#F5F6F8;--card:#FFFFFF;--ink:#1A2028;--muted:#6A7484;--border:#E3E7ED;
 --ok:#1F9D6C;--warn:#C08A1E;--down:#C24B42;--nodata:#D9DEE5;
 --okbg:#EAF6F0;--warnbg:#FAF3E3;--downbg:#F9ECEA;
 --chart1:#0E7C66;--chart2:#A9C6BD}
@media (prefers-color-scheme: dark){:root{--bg:#101418;--card:#191F26;--ink:#E6EAEF;
 --muted:#8D98A6;--border:#2A323C;--ok:#2FBF83;--warn:#D4A43C;--down:#D8685F;
 --nodata:#2A323C;--okbg:#16241E;--warnbg:#262114;--downbg:#271817;
 --chart1:#3FB99C;--chart2:#39564D}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:2.5rem 1rem 3rem}
main{max-width:46rem;margin:0 auto}
header{display:flex;align-items:baseline;gap:1rem;margin-bottom:1rem}
h1{font-size:1.3rem;margin:0;letter-spacing:-.01em}
.pill{margin-left:auto;display:flex;align-items:center;gap:.45em;font-size:.85rem;color:var(--muted);white-space:nowrap}
.dot{width:.6em;height:.6em;border-radius:50%;background:var(--nodata);flex:none}
.dot.ok{background:var(--ok)}.dot.warn{background:var(--warn)}.dot.down{background:var(--down)}
.banner{border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.75rem;background:var(--card)}
.banner.ok{background:var(--okbg);border-color:var(--ok)}
.banner.warn{background:var(--warnbg);border-color:var(--warn)}
.banner.down{background:var(--downbg);border-color:var(--down)}
.banner .state{font-weight:650;font-size:1.05rem}
.banner .bsub{color:var(--muted);font-size:.85rem;margin-top:.15rem}
.eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:1.75rem 0 .6rem;display:flex;align-items:baseline}
.eyebrow .r{margin-left:auto;font-weight:500;letter-spacing:0;text-transform:none;font-size:.8rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem}
.strip{display:flex;gap:2px;height:34px;align-items:stretch}
.strip i{flex:1 1 0;max-width:7px;border-radius:2px;background:var(--nodata);cursor:default}
.strip i.ok{background:var(--ok)}.strip i.warn{background:var(--warn)}.strip i.down{background:var(--down)}
.axis{display:flex;justify-content:space-between;color:var(--muted);font-size:.72rem;margin-top:.4rem}
.comp{padding:0}
.comp .row{display:flex;align-items:center;gap:.6rem;padding:.7rem 1rem;border-top:1px solid var(--border);font-size:.9rem}
.comp .row:first-child{border-top:0}
.comp .st{margin-left:auto;display:flex;align-items:center;gap:.45em;font-size:.8rem;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.2rem,1fr));gap:.7rem}
.m .v{font:600 1.35rem/1.25 ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.m .l{color:var(--muted);font-size:.78rem;margin-top:.15rem}
.chart{margin-top:.7rem}
.chead{display:flex;align-items:center;margin-bottom:.5rem;font-size:.8rem;color:var(--muted)}
.legend i{display:inline-block;width:.65em;height:.65em;border-radius:2px;margin:0 .35em 0 .9em;background:var(--chart1)}
.legend i.pf{background:var(--chart2)}
.rng{margin-left:auto;display:flex;gap:.25rem}
.rng button{font:inherit;color:var(--muted);background:none;border:1px solid transparent;border-radius:6px;padding:.15rem .5rem;cursor:pointer}
.rng button.on{color:var(--ink);border-color:var(--border)}
canvas{width:100%;height:190px;display:block}
footer{color:var(--muted);font-size:.78rem;margin-top:2rem;line-height:1.7}
#tip{position:fixed;z-index:10;background:var(--ink);color:var(--bg);font-size:.75rem;padding:.3rem .55rem;border-radius:6px;pointer-events:none;display:none;white-space:nowrap}
</style></head><body><main>
<header><h1>Globus Cluster Inference</h1>
<span class="pill"><span class="dot" id="pdot"></span><span id="ptxt">connecting…</span></span></header>
<div class="banner" id="banner"><div class="state" id="bstate">Connecting to the cluster…</div>
<div class="bsub" id="bsub">this page auto-refreshes every 2 s</div></div>

<div class="eyebrow">Availability<span class="r" id="upcts"></span></div>
<div class="card"><div class="strip" id="strip"></div>
<div class="axis"><span id="ax0">90 days ago</span><span>today</span></div></div>

<div class="eyebrow">Components</div>
<div class="card comp">
<div class="row">Inference engine <span class="sm" style="color:var(--muted);font-size:.78rem">vLLM on globus3</span><span class="st"><span class="dot" id="c-eng"></span><span id="t-eng">–</span></span></div>
<div class="row">Fairness gateway <span style="color:var(--muted);font-size:.78rem">API :8000</span><span class="st"><span class="dot" id="c-gw"></span><span id="t-gw">–</span></span></div>
<div class="row">Web UI <span style="color:var(--muted);font-size:.78rem">chat :8080</span><span class="st"><span class="dot" id="c-web"></span><span id="t-web">–</span></span></div>
</div>

<div class="eyebrow">Serving right now</div>
<div class="grid">
<div class="card m"><div class="v" id="gen">–</div><div class="l">generation tok/s, whole box</div></div>
<div class="card m"><div class="v" id="pre">–</div><div class="l">prefill tok/s, whole box</div></div>
<div class="card m"><div class="v" id="ttft">–</div><div class="l">time to first token, recent</div></div>
<div class="card m"><div class="v" id="stream">–</div><div class="l">per-stream decode tok/s, recent</div></div>
<div class="card m"><div class="v" id="run">–</div><div class="l">requests in flight (32 slots)</div></div>
<div class="card m"><div class="v" id="wait">–</div><div class="l">queued past the cap</div></div>
<div class="card m"><div class="v" id="users">–</div><div class="l">users active now</div></div>
<div class="card m"><div class="v" id="kv">–</div><div class="l">KV cache used (2M-token pool)</div></div>
</div>

<div class="eyebrow">Node — globus3</div>
<div class="grid">
<div class="card m"><div class="v" id="gpu">–</div><div class="l">GPU utilisation</div></div>
<div class="card m"><div class="v" id="mem">–</div><div class="l">unified memory (this IS GPU memory)</div></div>
<div class="card m"><div class="v" id="cpu">–</div><div class="l">CPU load, 1 min (20 cores)</div></div>
</div>

<div class="eyebrow">Usage</div>
<div class="grid">
<div class="card m"><div class="v" id="u24">–</div><div class="l">tokens, 24 h · <span id="u24s"></span></div></div>
<div class="card m"><div class="v" id="u7">–</div><div class="l">tokens, 7 d · <span id="u7s"></span></div></div>
<div class="card m"><div class="v" id="u30">–</div><div class="l">tokens, 30 d · <span id="u30s"></span></div></div>
<div class="card m"><div class="v" id="ul">–</div><div class="l"><span id="uls">tokens, all time</span></div></div>
<div class="card m"><div class="v" id="ureq">–</div><div class="l">requests, 24 h · <span id="ureqs"></span> all time</div></div>
</div>
<div class="card chart">
<div class="chead"><span class="legend"><b>history</b><i></i>decode<i class="pf"></i>prefill</span>
<span class="rng"><button data-r="24h" class="on">24 h</button><button data-r="7d">7 d</button><button data-r="30d">30 d</button></span></div>
<canvas id="chart"></canvas>
</div>

<footer>qwen3.8-27b · NVFP4 · 262,144-token context per request · default reasoning effort medium<br>
availability probed every 30 s from the login node · history since <span id="since">–</span> ·
docs: <code>/shared/llm/QUICKSTART.md</code> on the cluster<br><span id="age"></span></footer>
</main><div id="tip"></div><script>
const g=id=>document.getElementById(id);
const tip=g('tip');
function fmt(n){n=Math.round(n);return n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':''+n}
function pct(up,total,dp){return total?(100*up/total).toFixed(dp)+'%':'–'}
function setDot(el,cls){el.className='dot '+cls}
let prev=null;
async function tick(){
 try{
  const s=await (await fetch('/globus-stats.json',{cache:'no-store'})).json();
  const eng=!!s.engine,gw=s.fair&&s.fair.ok,web=!!s.webui_ok;
  setDot(g('c-eng'),eng?'ok':'down');g('t-eng').textContent=eng?'operational':'down';
  setDot(g('c-gw'),gw?'ok':'down');g('t-gw').textContent=gw?'operational':'down';
  setDot(g('c-web'),web?'ok':'down');g('t-web').textContent=web?'operational':'down';
  const banner=g('banner');
  if(!eng||!gw){
   banner.className='banner down';setDot(g('pdot'),'down');g('ptxt').textContent='offline';
   g('bstate').textContent='Model server offline';
   g('bsub').textContent='auto-restart usually brings it back within ~7 minutes — this page recovers by itself';
   prev=null;return;
  }
  const waiting=Math.round(s.waiting??0);
  if(waiting>0||!web){
   banner.className='banner warn';setDot(g('pdot'),'warn');g('ptxt').textContent='degraded';
   g('bstate').textContent=waiting>0?'Operational — under load':'API operational — web UI down';
   g('bsub').textContent=waiting>0?waiting+' request(s) queued past the 32-slot cap · new requests will wait briefly':'the OpenAI API works; the browser chat backend is not answering';
  }else{
   banner.className='banner ok';setDot(g('pdot'),'ok');g('ptxt').textContent='operational';
   g('bstate').textContent='All systems operational';
   g('bsub').textContent='updated '+new Date().toLocaleTimeString()+' · auto-refreshes every 2 s';
  }
  g('run').textContent=Math.round(s.running??0);
  g('wait').textContent=waiting;
  g('kv').textContent=((s.kv_usage??0)*100).toFixed(1)+'%';
  if(s.fair&&s.fair.users!=null)g('users').textContent=s.fair.users;
  if(s.node){
   if(s.node.gpu_util!=null)g('gpu').textContent=s.node.gpu_util+'%';
   if(s.node.mem_used_gib!=null)g('mem').textContent=s.node.mem_used_gib.toFixed(0)+' / '+s.node.mem_total_gib.toFixed(0)+' GiB';
   if(s.node.load1!=null)g('cpu').textContent=s.node.load1;
  }
  if(prev&&s.t>prev.t){
   const dt=s.t-prev.t;
   g('gen').textContent=Math.max(0,(s.generation_tokens-prev.generation_tokens)/dt).toFixed(1);
   g('pre').textContent=Math.max(0,(s.prompt_tokens-prev.prompt_tokens)/dt).toFixed(0);
   const dttc=(s.ttft_count??0)-(prev.ttft_count??0),dtts=(s.ttft_sum??0)-(prev.ttft_sum??0);
   if(dttc>0){const v=dtts/dttc;g('ttft').textContent=v<1?(v*1000).toFixed(0)+' ms':v.toFixed(1)+' s'}
   const dic=(s.itl_count??0)-(prev.itl_count??0),dis=(s.itl_sum??0)-(prev.itl_sum??0);
   if(dic>0&&dis>0)g('stream').textContent=(dic/dis).toFixed(1);
  }
  prev=s;
  g('age').textContent='';
 }catch(e){g('age').textContent='stats fetch failed: '+e;setDot(g('pdot'),'warn');g('ptxt').textContent='unreachable'}
}
// ---- usage history + availability strip (fetched every minute) ----
let usage=null,range='24h';
function showTip(e,html){tip.innerHTML=html;tip.style.display='block';
 tip.style.left=Math.min(window.innerWidth-tip.offsetWidth-8,Math.max(8,e.clientX-tip.offsetWidth/2))+'px';
 tip.style.top=(e.clientY-tip.offsetHeight-12)+'px'}
function hideTip(){tip.style.display='none'}
function strip(){
 const el=g('strip');el.innerHTML='';
 const byDay=new Map();
 for(const [h,,,up,total] of usage.hourly){
  const d=new Date(h*36e5);const k=d.getFullYear()+'-'+d.getMonth()+'-'+d.getDate();
  const e=byDay.get(k)||[0,0];e[0]+=up;e[1]+=total;byDay.set(k,e);
 }
 let up90=0,t90=0;
 for(let i=89;i>=0;i--){
  const d=new Date(Date.now()-i*864e5);
  const k=d.getFullYear()+'-'+d.getMonth()+'-'+d.getDate();
  const e=byDay.get(k);
  const bar=document.createElement('i');
  let txt;
  if(!e||!e[1]){txt='no data'}
  else{
   up90+=e[0];t90+=e[1];
   const p=100*e[0]/e[1];
   bar.className=p>=99.9?'ok':p>=95?'warn':'down';
   txt=p.toFixed(p===100?0:2)+'% up';
  }
  const lbl=d.toLocaleDateString([],{month:'short',day:'numeric'});
  bar.onmousemove=ev=>showTip(ev,'<b>'+lbl+'</b> — '+txt);
  bar.onmouseleave=hideTip;
  el.appendChild(bar);
 }
 g('upcts').textContent='24 h '+pct(usage.day.up,usage.day.total,1)+' · 30 d '+pct(usage.month.up,usage.month.total,2)+' · 90 d '+pct(up90,t90,2);
}
function cards(){
 const s=(id,w)=>{g(id).textContent=fmt(w.prompt+w.gen);g(id+'s').textContent=fmt(w.gen)+' decode / '+fmt(w.prompt)+' prefill'};
 s('u24',usage.day);s('u7',usage.week);s('u30',usage.month);
 const L=usage.lifetime;
 g('ul').textContent=fmt(L.prompt+L.gen);
 g('uls').textContent='tokens since '+new Date(usage.since*1000).toLocaleDateString()+' · '+fmt(L.gen)+' decode / '+fmt(L.prompt)+' prefill';
 g('ureq').textContent=fmt(usage.day.reqs??0);
 g('ureqs').textContent=fmt(L.reqs??0);
 g('since').textContent=new Date(usage.since*1000).toLocaleDateString();
}
function draw(){
 if(!usage)return;
 const cv=g('chart'),ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1;
 const W=cv.clientWidth||600,H=190;
 cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 ctx.clearRect(0,0,W,H);
 const cs=getComputedStyle(document.documentElement);
 const ACC=cs.getPropertyValue('--chart1').trim(),PF=cs.getPropertyValue('--chart2').trim(),
       MUT=cs.getPropertyValue('--muted').trim(),BRD=cs.getPropertyValue('--border').trim();
 const cfg={'24h':[1,24],'7d':[6,28],'30d':[24,30]}[range],bh=cfg[0],n=cfg[1];
 const hours=new Map(usage.hourly.map(r=>[r[0],r]));
 const curH=Math.floor(Date.now()/36e5);
 const B=[];
 for(let i=n-1;i>=0;i--){
  let p=0,d=0;const end=curH-i*bh;
  for(let h=end-bh+1;h<=end;h++){const r=hours.get(h);if(r){p+=r[1];d+=r[2]}}
  B.push({t:end*36e5,p,d});
 }
 const max=Math.max(1,...B.map(b=>b.p+b.d));
 const padL=8,padR=8,top=18,bot=20,cw=(W-padL-padR)/n,ph=H-top-bot;
 ctx.strokeStyle=BRD;ctx.lineWidth=1;
 [0.5,1].forEach(f=>{const y=top+ph*(1-f)+.5;ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke()});
 ctx.fillStyle=MUT;ctx.font='11px system-ui';ctx.textAlign='left';
 ctx.fillText(fmt(max)+' tok',padL,top-6);
 B.forEach((b,i)=>{
  const x=padL+i*cw+1,w=Math.max(1,cw-2);
  const hd=ph*b.d/max,hp=ph*b.p/max;
  let y=H-bot;
  ctx.fillStyle=ACC;ctx.fillRect(x,y-hd,w,hd);y-=hd;
  ctx.fillStyle=PF;ctx.fillRect(x,y-hp,w,hp);
 });
 ctx.fillStyle=MUT;ctx.textAlign='center';
 const step=range==='24h'?6:range==='7d'?4:5;
 B.forEach((b,i)=>{
  if((n-1-i)%step)return;
  const d=new Date(b.t);
  const lab=range==='24h'?d.getHours()+':00':(d.getMonth()+1)+'/'+d.getDate();
  ctx.fillText(lab,padL+i*cw+cw/2,H-6);
 });
}
async function utick(){try{
 usage=await (await fetch('/globus-usage.json',{cache:'no-store'})).json();
 if(usage.lifetime){cards();strip()}
 draw();
}catch(e){}}
document.querySelectorAll('.rng button').forEach(b=>b.onclick=()=>{
 range=b.dataset.r;
 document.querySelectorAll('.rng button').forEach(x=>x.classList.toggle('on',x===b));
 draw();
});
window.addEventListener('resize',()=>{draw()});
tick();setInterval(tick,2000);
utick();setInterval(utick,60000);
</script></body></html>
"""


def local_response(path):
    """Return response bytes for proxy-served paths, or None to proxy through."""
    if path == "/globus-stats.json":
        body = json.dumps(stats_snapshot()).encode()
        ctype = b"application/json"
    elif path == "/globus-usage.json":
        body = json.dumps(USAGE_VIEW).encode()
        ctype = b"application/json"
    elif path in ("/globus-stats", "/globus-stats/"):
        body = STATS_PAGE.encode()
        ctype = b"text/html; charset=utf-8"
    else:
        return None
    return (b"HTTP/1.1 200 OK\r\nContent-Type: " + ctype +
            b"\r\nContent-Length: " + str(len(body)).encode() +
            b"\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" + body)


def uid_of_peer(port):
    """uid owning the loopback socket (127.0.0.1:port -> LISTEN). Kernel truth."""
    want_local = f"0100007F:{port:04X}"
    want_rem = f"0100007F:{LISTEN[1]:04X}"
    with open("/proc/net/tcp") as f:
        next(f)
        for line in f:
            p = line.split()
            if p[1] == want_local and p[2] == want_rem:
                return int(p[7])
    return None


def identity(port):
    uid = uid_of_peer(port)
    if uid is None:
        return None
    try:
        pw = pwd.getpwuid(uid)
    except KeyError:
        return None
    name = (pw.pw_gecos.split(",")[0] or pw.pw_name).strip() or pw.pw_name
    return pw.pw_name, name


async def pipe(reader, writer):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def copy_body(headers_lc, reader, writer):
    """Forward one request body according to its framing. Returns False on EOF."""
    if "content-length" in headers_lc:
        n = int(headers_lc["content-length"])
        while n > 0:
            chunk = await reader.read(min(n, 65536))
            if not chunk:
                return False
            writer.write(chunk)
            await writer.drain()
            n -= len(chunk)
    elif headers_lc.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            writer.write(size_line)
            try:
                size = int(size_line.strip().split(b";")[0], 16)
            except ValueError:
                return False
            data = await reader.readexactly(size + 2)      # chunk + CRLF
            writer.write(data)
            await writer.drain()
            if size == 0:
                break
    return True


async def client_to_upstream(reader, writer, email, name, client_w):
    """Parse each request head, strip identity headers, inject ours, stream the body."""
    while True:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                ConnectionResetError):
            return
        lines = head.split(b"\r\n")
        request_line = lines[0]

        # Paths the proxy answers itself (stats). Browsers do not pipeline, so any
        # earlier proxied response on this connection has already fully passed through.
        parts = request_line.split(b" ")
        if len(parts) >= 2 and parts[0] == b"GET":
            path = parts[1].split(b"?", 1)[0].decode(errors="replace")
            resp = await asyncio.get_running_loop().run_in_executor(
                None, local_response, path)
            if resp is not None:
                client_w.write(resp)
                await client_w.drain()
                return                                      # Connection: close
        headers_lc, out = {}, [request_line]
        upgrade = False
        for line in lines[1:]:
            if not line:
                continue
            key = line.split(b":", 1)[0].strip().lower()
            if key in (EMAIL_HEADER.lower().encode(), NAME_HEADER.lower().encode()):
                continue                                    # no self-claimed identities
            if b":" in line:
                k, v = line.split(b":", 1)
                headers_lc[k.strip().lower().decode()] = v.strip().decode(errors="replace")
            out.append(line)
        if headers_lc.get("upgrade", "").lower() == "websocket":
            upgrade = True
        out.append(f"{EMAIL_HEADER}: {email}".encode())
        out.append(f"{NAME_HEADER}: {name}".encode())
        writer.write(b"\r\n".join(out) + b"\r\n\r\n")
        await writer.drain()
        if not await copy_body(headers_lc, reader, writer):
            return
        if upgrade:                                         # websocket: blind pipe now
            await pipe(reader, writer)
            return


async def handle(client_r, client_w):
    peer = client_w.get_extra_info("peername")
    ident = identity(peer[1]) if peer else None
    if ident is None:
        client_w.write(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n"
                       b"Connection: close\r\n\r\n"
                       b"Could not resolve your cluster account from this connection.\r\n")
        await client_w.drain()
        client_w.close()
        return
    user, name = ident
    try:
        up_r, up_w = await asyncio.open_unix_connection(UPSTREAM_SOCKET)
    except OSError:
        client_w.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\n"
                       b"Connection: close\r\n\r\n"
                       b"The web UI backend is not running. Try: ssh globus1 serving/bin/webui status\r\n")
        await client_w.drain()
        client_w.close()
        return
    await asyncio.gather(
        client_to_upstream(client_r, up_w, f"{user}@globus.local", name, client_w),
        pipe(up_r, client_w),
        return_exceptions=True,
    )
    for w in (client_w, up_w):
        try:
            w.close()
        except Exception:
            pass


async def main():
    server = await asyncio.start_server(handle, *LISTEN, limit=MAX_HEADER)
    sampler = asyncio.create_task(usage_sampler())
    print(f"idproxy: {LISTEN[0]}:{LISTEN[1]} -> {UPSTREAM_SOCKET}", flush=True)
    try:
        async with server:
            await server.serve_forever()
    finally:
        sampler.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
