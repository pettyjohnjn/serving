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
import sys
import time
import urllib.request

LISTEN = ("127.0.0.1", 8080)
# The backend listens on a unix socket inside the 0700 data dir — see bin/webui.
ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
UPSTREAM_SOCKET = ROOT + "/webui-data/webui.sock"
VLLM_METRICS = "http://127.0.0.1:8000/metrics"
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
USAGE_KEEP_H = 31 * 24                 # hourly buckets kept; windows need 30 days
USAGE_VIEW = {}                        # latest derived view, replaced atomically


def usage_load():
    try:
        with open(USAGE_FILE) as f:
            s = json.load(f)
        if isinstance(s.get("hours"), dict) and "lifetime" in s:
            for v in s["hours"].values():           # older layout: [prompt, gen]
                while len(v) < 4:
                    v.append(0)
            s["lifetime"].setdefault("up", 0)
            s["lifetime"].setdefault("total", 0)
            return s
    except (OSError, ValueError):
        pass
    return {"since": time.time(),
            "lifetime": {"prompt": 0.0, "gen": 0.0, "up": 0, "total": 0},
            "last": None, "hours": {}}


def usage_update(state, snap, now):
    """Fold one stats_snapshot into the hourly buckets: [prompt, gen, up, probes]."""
    hour = state["hours"].setdefault(str(int(now // 3600)), [0.0, 0.0, 0, 0])
    hour[3] += 1
    state["lifetime"]["total"] += 1
    if snap.get("up"):
        hour[2] += 1
        state["lifetime"]["up"] += 1
    if snap.get("up") and "prompt_tokens" in snap and "generation_tokens" in snap:
        cur = [snap["prompt_tokens"], snap["generation_tokens"]]
        last = state.get("last")
        if last is not None:
            dp, dg = cur[0] - last[0], cur[1] - last[1]
            if dp < 0 or dg < 0:                   # engine restarted, counters reset
                dp, dg = cur[0], cur[1]
            hour[0] += dp
            hour[1] += dg
            state["lifetime"]["prompt"] += dp
            state["lifetime"]["gen"] += dg
        state["last"] = cur
    cutoff = int(now // 3600) - USAGE_KEEP_H
    for k in [k for k in state["hours"] if int(k) < cutoff]:
        del state["hours"][k]


def usage_derive(state, now):
    cur = int(now // 3600)

    def window(nh):
        p = g = up = total = 0
        for k, v in state["hours"].items():
            if int(k) > cur - nh:
                p += v[0]
                g += v[1]
                up += v[2]
                total += v[3]
        return {"prompt": p, "gen": g, "up": up, "total": total}

    hourly = sorted((int(k), round(v[0]), round(v[1]))
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
    try:
        text = urllib.request.urlopen(VLLM_METRICS, timeout=5).read().decode()
    except OSError:
        return out
    out["up"] = True
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
<title>Globus Cluster Inference — stats</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#F7F8FA;--ink:#1B1F26;--muted:#5A6472;--accent:#0E7C66;--soft:#A8C3BB;--border:#DCE1E7;--card:#FFFFFF}
@media (prefers-color-scheme: dark){:root{--bg:#14171B;--ink:#E4E8EC;--muted:#98A2AE;--accent:#3FB99C;--soft:#33443F;--border:#2A3038;--card:#1C2127}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;padding:2.5rem 1rem}
main{max-width:40rem;margin:0 auto}
h1{font-size:1.25rem;margin:0 0 .25rem}
h2{font-size:.95rem;margin:1.5rem 0 .6rem}
.sub{color:var(--muted);margin:0 0 1.5rem;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));gap:.75rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.9rem 1rem}
.card .v{font:600 1.5rem/1.2 ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.card .l{color:var(--muted);font-size:.8rem;margin-top:.2rem}
.down{color:#B4423C;font-weight:600}
#age{color:var(--muted);font-size:.8rem;margin-top:1.25rem}
.chart{margin-top:.75rem}
.chead{display:flex;align-items:center;margin-bottom:.5rem;font-size:.8rem;color:var(--muted)}
.legend i{display:inline-block;width:.65em;height:.65em;border-radius:2px;margin:0 .35em 0 .9em;background:var(--accent)}
.legend i.pf{background:var(--soft)}
.rng{margin-left:auto;display:flex;gap:.25rem}
.rng button{font:inherit;color:var(--muted);background:none;border:1px solid transparent;border-radius:6px;padding:.15rem .5rem;cursor:pointer}
.rng button.on{color:var(--ink);border-color:var(--border)}
canvas{width:100%;height:190px;display:block}
</style></head><body><main>
<h1>Globus Cluster Inference</h1>
<p class="sub">qwen3.8-27b on globus3 &middot; live, refreshes every 2 s</p>
<div class="grid">
<div class="card"><div class="v" id="gen">–</div><div class="l">generation tok/s (whole box)</div></div>
<div class="card"><div class="v" id="pre">–</div><div class="l">prefill tok/s (whole box)</div></div>
<div class="card"><div class="v" id="run">–</div><div class="l">requests in flight</div></div>
<div class="card"><div class="v" id="wait">–</div><div class="l">queued (past the cap of 32)</div></div>
<div class="card"><div class="v" id="kv">–</div><div class="l">KV cache used (1.7M-token pool)</div></div>
<div class="card"><div class="v" id="hit">–</div><div class="l">prefix cache hit rate (lifetime)</div></div>
<div class="card"><div class="v" id="gpu">–</div><div class="l">GPU utilisation (globus3)</div></div>
<div class="card"><div class="v" id="mem">–</div><div class="l">node memory (unified — this IS GPU memory)</div></div>
<div class="card"><div class="v" id="cpu">–</div><div class="l">CPU load, 1 min (20 cores)</div></div>
</div>
<h2>Tokens served &amp; reliability</h2>
<div class="grid">
<div class="card"><div class="v" id="u24">–</div><div class="l">tokens, last 24 h &middot; <span id="u24s"></span></div></div>
<div class="card"><div class="v" id="u7">–</div><div class="l">tokens, last 7 days &middot; <span id="u7s"></span></div></div>
<div class="card"><div class="v" id="u30">–</div><div class="l">tokens, last 30 days &middot; <span id="u30s"></span></div></div>
<div class="card"><div class="v" id="ul">–</div><div class="l"><span id="uls">tokens, all time</span></div></div>
<div class="card"><div class="v" id="up30">–</div><div class="l">uptime, last 30 days &middot; <span id="upx"></span></div></div>
</div>
<div class="card chart">
<div class="chead"><span class="legend"><b>usage</b><i></i>decode<i class="pf"></i>prefill</span>
<span class="rng"><button data-r="24h" class="on">24 h</button><button data-r="7d">7 d</button><button data-r="30d">30 d</button></span></div>
<canvas id="chart"></canvas>
</div>
<p id="age"></p>
</main><script>
const g=id=>document.getElementById(id);
let prev=null;
async function tick(){
  try{
    const s=await (await fetch('/globus-stats.json',{cache:'no-store'})).json();
    if(!s.up){g('age').innerHTML='<span class="down">model server unreachable</span> — it may be reloading (~5 min); this page will recover by itself';prev=null;return}
    g('run').textContent=Math.round(s.running??0);
    g('wait').textContent=Math.round(s.waiting??0);
    g('kv').textContent=((s.kv_usage??0)*100).toFixed(1)+'%';
    if(s.prefix_queries>0)g('hit').textContent=(100*s.prefix_hits/s.prefix_queries).toFixed(0)+'%';
    if(s.node){
      if(s.node.gpu_util!=null)g('gpu').textContent=s.node.gpu_util+'%';
      if(s.node.mem_used_gib!=null)g('mem').textContent=s.node.mem_used_gib.toFixed(0)+' / '+s.node.mem_total_gib.toFixed(0)+' GiB';
      if(s.node.load1!=null)g('cpu').textContent=s.node.load1;
    }
    if(prev&&s.t>prev.t){
      const dt=s.t-prev.t;
      g('gen').textContent=Math.max(0,(s.generation_tokens-prev.generation_tokens)/dt).toFixed(1);
      g('pre').textContent=Math.max(0,(s.prompt_tokens-prev.prompt_tokens)/dt).toFixed(0);
    }
    prev=s;
    g('age').textContent='context window 262,144 tokens per request \\u00b7 updated '+new Date().toLocaleTimeString();
  }catch(e){g('age').textContent='stats fetch failed: '+e}
}
// ---- usage history (sampled server-side every 30 s, fetched every minute) ----
let usage=null,range='24h';
function fmt(n){n=Math.round(n);return n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':''+n}
function pct(w,dp){return w.total?(100*w.up/w.total).toFixed(dp)+'%':'–'}
function cards(){
  const s=(id,w)=>{g(id).textContent=fmt(w.prompt+w.gen);g(id+'s').textContent=fmt(w.gen)+' decode / '+fmt(w.prompt)+' prefill'};
  s('u24',usage.day);s('u7',usage.week);s('u30',usage.month);
  const L=usage.lifetime;
  g('ul').textContent=fmt(L.prompt+L.gen);
  g('uls').textContent='tokens since '+new Date(usage.since*1000).toLocaleDateString()+' \\u00b7 '+fmt(L.gen)+' decode / '+fmt(L.prompt)+' prefill';
  g('up30').textContent=pct(usage.month,2);
  g('upx').textContent='24 h: '+pct(usage.day,1)+' \\u00b7 all: '+pct(L,2)+' (probed every 30 s)';
}
function draw(){
  if(!usage)return;
  const cv=g('chart'),ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1;
  const W=cv.clientWidth||600,H=190;
  cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  const cs=getComputedStyle(document.documentElement);
  const ACC=cs.getPropertyValue('--accent').trim(),PF=cs.getPropertyValue('--soft').trim(),
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
  if(usage.lifetime)cards();draw();
}catch(e){}}
document.querySelectorAll('.rng button').forEach(b=>b.onclick=()=>{
  range=b.dataset.r;
  document.querySelectorAll('.rng button').forEach(x=>x.classList.toggle('on',x===b));
  draw();
});
window.addEventListener('resize',draw);
tick();setInterval(tick,2000);
utick();setInterval(utick,60000);
</script></body></html>"""


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
