#!/usr/bin/env python3
"""Agent dashboard: each person's Pi sessions, run as that person's agent account in one Slurm job per person.

Runs as the service account on the login node (started by `bin/serving`). Two faces:

  UI       unix socket state/workbench/dashboard.sock, reachable only through bin/webui-idproxy.py on 127.0.0.1:8090,
           which sets X-Workbench-User to the cluster account that opened the connection. No other login.
             /                     your sessions, new-session form
             /agents/<name>/...    a session's web terminal (proxied to its ttyd, credentials added here)
             /api/new  /api/stop/<name>  /api/resume/<name>  /api/log
  control  $LOGIN_CLUSTER_IP:$CONTROL_PORT, for the agent host jobs (jobs/agent-host.sbatch). A host job polls its
           person's desired sessions and reports what runs (ttyd address + password per session). Authenticated by
           the per-job token the dashboard generated at submit time. The host job runs as the agent account and
           never needs this service's files; this service never needs the agent account's files.

State (service-owned, 0700): state/workbench/users/<user>/{desired.json, report.json, host.json}
Run-as accounts and the submit method come from etc/site.env (RUNAS_MODE, SUBMIT_MODE).
"""
import asyncio, hmac, html, json, os, pwd, re, secrets, shlex, subprocess, time, urllib.request
from aiohttp import web, ClientSession, BasicAuth, WSMsgType, ClientTimeout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def site_env():
    out = subprocess.run(["bash", "-c", f"set -a; . '{ROOT}/etc/site.env'; env -0"], capture_output=True)
    return dict(kv.split("=", 1) for kv in out.stdout.decode().split("\0") if "=" in kv)


S = site_env()
ME = pwd.getpwuid(os.getuid()).pw_name
STATE = os.path.join(ROOT, "state", "workbench")
USERS = os.path.join(STATE, "users")
SOCK = os.path.join(STATE, "dashboard.sock")
KEYS = os.path.join(ROOT, "state", "keys.json")
CONTROL = (S["LOGIN_CLUSTER_IP"], int(S.get("CONTROL_PORT", "8092")))
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
USER_RE = re.compile(r"^[a-z_][a-z0-9_.-]{0,31}$")
PER_SESSION_MB = 200          # measured: 8 idle sessions = 1342 MiB for the whole job (shared libraries)
STALE_S = 60                  # a host job that has not reported for this long is shown as not reporting


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


# ---------------------------------------------------------------- state
def udir(user):
    d = os.path.join(USERS, user); os.makedirs(d, mode=0o700, exist_ok=True); return d


def load(user, name, default):
    try:
        return json.load(open(os.path.join(udir(user), name)))
    except Exception:
        return default


def save(user, name, obj):
    p = os.path.join(udir(user), name); tmp = p + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def runas(user):
    mode = S.get("RUNAS_MODE", "same")
    if mode == "same":
        return user
    if mode == "suffix":
        return user + S.get("RUNAS_SUFFIX", "-agent")
    if mode == "map":
        for line in open(S["RUNAS_MAP"]):
            f = line.split()
            if len(f) == 2 and f[0] == user:
                return f[1]
        return None
    raise SystemExit(f"bad RUNAS_MODE {mode}")


def api_key(user):
    """The person's model key for fair-proxy's keyed listener (created on first use)."""
    keys = json.load(open(KEYS)) if os.path.exists(KEYS) else {}
    for k, u in keys.items():
        if u == user:
            return k
    k = "sk-spark-" + secrets.token_urlsafe(24); keys[k] = user
    tmp = KEYS + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(keys, f, indent=1)
    os.replace(tmp, KEYS)
    return k


# ---------------------------------------------------------------- slurm
def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"}


def host_job(user, account):
    r = subprocess.run(["squeue", "-h", "-u", account, "-n", f"agent-host-{user}", "-o", "%i|%T|%N|%M|%l|%C|%m"],
                       capture_output=True, text=True, env=clean_env(), timeout=30)
    rows = [l.split("|") for l in r.stdout.split()]
    if not rows:
        return None
    rows.sort(key=lambda x: x[1] != "RUNNING")
    jid, st, node, el, lim, cpus, mem = rows[0]
    return dict(job=jid, state=st, node=node, elapsed=el, limit=lim, cpus=cpus, mem=mem)


def submit(user, account, cpus, mem, tlim):
    token = secrets.token_urlsafe(24)
    env = {"WB_USER": user, "WB_CONTROL": f"http://{CONTROL[0]}:{CONTROL[1]}", "WB_TOKEN": token,
           "SPARK_LLM_KEY": api_key(user), "MODEL_RELAY": f"{S['LOGIN_CLUSTER_IP']}:{S.get('KEYED_PORT', '8001')}",
           # the agent account cannot read this service's files: everything the job needs comes in its environment
           "AGENT_TOOLS": S["AGENT_TOOLS"], "AGENT_CLUSTER_IF": S.get("AGENT_CLUSTER_IF", "eth0"),
           "AGENT_PI_PACKAGES": S.get("AGENT_PI_PACKAGES", ""), "AGENT_PI_PROFILE": S.get("AGENT_PI_PROFILE", "lab"),
           "SERVED_NAME": S.get("SERVED_NAME", "qwen3.8-flash-next")}
    script = os.path.join(ROOT, "jobs", "agent-host.sbatch")
    args = ["-J", f"agent-host-{user}", "-p", S.get("AGENT_PARTITION", "main"), "-c", cpus, f"--mem={mem}",
            "-t", tlim, f"--exclude={S.get('AGENT_NODES_EXCLUDE', '')}", "-o", f"agents-host-%j.out"]
    mode = S.get("SUBMIT_MODE", "sbatch")
    if mode in ("sbatch", "sudo"):
        bad = [k for k, v in env.items() if "," in v]
        if bad:
            raise RuntimeError(f"comma in {bad}: cannot pass through sbatch --export (use spaces in AGENT_PI_PACKAGES)")
        exp = "--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items())
        # -o is relative to the submit dir: the agent account's home (the job runs there)
        home = pwd.getpwnam(account).pw_dir
        cmd = ["sbatch", "--parsable", f"--chdir={home}", exp, *args, script]
        if mode == "sbatch":
            if account != ME:
                raise RuntimeError(f"SUBMIT_MODE=sbatch can only run agents as {ME}, not {account}")
        else:
            cmd = ["sudo", "-n", "-u", account, *cmd]
        r = subprocess.run(cmd, capture_output=True, text=True, env=clean_env(), timeout=60)
        if r.returncode:
            raise RuntimeError("submit failed: " + r.stderr.strip())
        jid = r.stdout.strip()
    elif mode == "slurmrestd":
        tok = subprocess.run(shlex.split(S["SLURM_TOKEN_CMD"].format(user=account)), capture_output=True, text=True,
                             timeout=30).stdout.strip().split("=")[-1]
        home = pwd.getpwnam(account).pw_dir
        body = {"script": open(script).read(), "job": {
            "name": f"agent-host-{user}", "partition": S.get("AGENT_PARTITION", "main"), "cpus_per_task": int(cpus),
            "memory_per_node": {"set": True, "number": int(mem[:-1]) * (1024 if mem.endswith("G") else 1)},
            "time_limit": {"set": True, "number": slurm_minutes(tlim)}, "current_working_directory": home,
            "standard_output": f"{home}/agents-host-%j.out", "requeue": True,
            "excluded_nodes": S.get("AGENT_NODES_EXCLUDE", "").split(","),
            "environment": [f"{k}={v}" for k, v in env.items()] + [f"HOME={home}", f"USER={account}",
                                                                    "PATH=/usr/local/bin:/usr/bin:/bin"]}}
        url = f"{S['SLURMRESTD_URL']}/slurm/{S.get('SLURMRESTD_API', 'v0.0.41')}/job/submit"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "X-SLURM-USER-NAME": account, "X-SLURM-USER-TOKEN": tok})
        out = json.load(urllib.request.urlopen(req, timeout=60))
        if out.get("errors"):
            raise RuntimeError(f"slurmrestd: {out['errors']}")
        jid = str(out.get("job_id"))
    else:
        raise RuntimeError(f"bad SUBMIT_MODE {mode}")
    save(user, "host.json", dict(job=jid, token=token, account=account, submitted=time.time()))
    log(f"submitted agent host for {user} as {account}: job {jid} ({cpus} CPUs, {mem})")
    return jid


def slurm_minutes(t):
    d, _, hms = t.rpartition("-"); p = [int(x) for x in hms.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return int(d or 0) * 1440 + p[0] * 60 + p[1] + (1 if p[2] else 0)


def ensure_host(user, cpus=None, mem=None, tlim=None):
    account = runas(user)
    if account is None:
        raise RuntimeError(f"no agent account mapped for {user}")
    try:
        pwd.getpwnam(account)
    except KeyError:
        raise RuntimeError(f"agent account {account} does not exist")
    h = host_job(user, account)
    if h:
        return f"host job {h['job']} ({h['state'].lower()})"
    jid = submit(user, account, cpus or S.get("AGENT_DEFAULT_CPUS", "4"), mem or S.get("AGENT_DEFAULT_MEM", "16G"),
                 tlim or "2-00:00:00")
    return f"submitted host job {jid} as {account}"


# ---------------------------------------------------------------- UI (behind the identity proxy)
@web.middleware
async def identity(request, handler):
    user = request.headers.get("X-Workbench-User", "")
    if not USER_RE.match(user):
        raise web.HTTPForbidden(text="no identity (reach this through the login node's port 8090)")
    request["user"] = user
    return await handler(request)


def sessions_view(user):
    desired = load(user, "desired.json", {})
    rep = load(user, "report.json", {})
    fresh = time.time() - rep.get("t", 0) < STALE_S
    return desired, (rep.get("sessions", {}) if fresh else {}), rep, fresh


async def index(request):
    user = request["user"]; account = runas(user)
    h = host_job(user, account) if account else None
    desired, live, rep, fresh = sessions_view(user)
    rows = []
    for n in sorted(desired):
        d, l = desired[n], live.get(n, {})
        if d.get("state") == "run":
            st = "running" if l.get("attach") else (l.get("error") or ("starting" if h else "waiting for host"))
        else:
            st = "stopped"
        open_ = f'<a href="/agents/{n}/" target="_blank">open</a>' if l.get("attach") else ""
        btn = (f'<button onclick="act(\'stop\',\'{n}\')">stop</button>' if d.get("state") == "run" else
               f'<button onclick="act(\'resume\',\'{n}\')">resume</button>')
        rows.append(f"<tr><td><b>{html.escape(n)}</b></td><td class={st.split()[0]}>{html.escape(st)}</td>"
                    f"<td>{html.escape(d.get('workdir', ''))}</td><td>{html.escape(d.get('created', ''))}</td>"
                    f"<td>{open_} {btn}</td></tr>")
    nrun = sum(1 for d in desired.values() if d.get("state") == "run")
    if h:
        mem_mb = int(re.sub(r"\D", "", h["mem"]) or 0) * (1024 if h["mem"].upper().endswith("G") else 1)
        hosttxt = (f"your host job <b>{h['job']}</b> {h['state'].lower()} on {h['node'] or '-'} as <b>{account}</b> · "
                   f"{h['cpus']} CPUs · {h['mem']} · up {h['elapsed']} of {h['limit']} · {nrun} sessions "
                   f"(idle capacity ~{max(1, mem_mb // PER_SESSION_MB)} by memory; plan 1-2 CPUs per busy agent)"
                   + ("" if fresh else " · <i>not reporting yet</i>") + " · <a href='/api/log' target=_blank>host log</a>")
    else:
        hosttxt = f"no host job running: starting a session submits one (runs as <b>{account}</b>)"
    body = f"""
    <h2>Pi agents for {html.escape(user)} <small>(lab profile · model {html.escape(S.get('SERVED_NAME', 'qwen3.8-flash-next'))})</small></h2>
    <p>{hosttxt}</p>
    <table><tr><th>session</th><th>state</th><th>workdir</th><th>created</th><th></th></tr>
    {''.join(rows) or '<tr><td colspan=5><i>no sessions yet</i></td></tr>'}</table>
    <h3>New session</h3>
    <form id=f onsubmit="return newSession()">
      name <input name=name required pattern="[a-z0-9][a-z0-9-]{{0,30}}" size=14>
      workdir <input name=workdir size=36 placeholder="~/agents/&lt;name&gt;/work">
      <button>start</button><br><small>host job size (used only when no host job is running):</small>
      cpus <input name=cpus value={S.get('AGENT_DEFAULT_CPUS', '4')} size=2>
      mem <input name=mem value={S.get('AGENT_DEFAULT_MEM', '16G')} size=4>
      time <input name=time value=2-00:00:00 size=10>
    </form><p id=msg></p>
    <p class=note>All your sessions share one Slurm job on the compute nodes and run as {html.escape(account or '?')}.
    They keep running when you close this page. Quitting pi in its terminal stops that session; "resume" brings back
    the same conversation. The host job renews itself at the 2-day limit and ends after 30 idle minutes.</p>
    <script>
    async function post(u, data) {{ const r = await fetch(u, {{method:'POST', body: data}}); return [r.ok, await r.text()]; }}
    async function newSession() {{ const [ok, t] = await post('/api/new', new FormData(document.getElementById('f')));
      document.getElementById('msg').textContent = t; if (ok) setTimeout(() => location.reload(), 3000); return false; }}
    async function act(a, n) {{ if (a == 'stop' && !confirm('stop ' + n + '?')) return;
      const [ok, t] = await post('/api/' + a + '/' + n); document.getElementById('msg').textContent = t;
      setTimeout(() => location.reload(), 2500); }}
    setTimeout(() => location.reload(), 30000);
    </script>"""
    return web.Response(content_type="text/html", text=PAGE.format(body=body))


def size_ok(cpus, mem, tlim):
    return (cpus.isdigit() and 1 <= int(cpus) <= 16 and re.fullmatch(r"\d+[MG]", mem)
            and re.fullmatch(r"(\d+-)?\d{1,2}(:\d{2}){0,2}", tlim))


async def new_session(request):
    user = request["user"]; f = await request.post()
    name = f.get("name", "").strip()
    if not NAME_RE.match(name):
        return web.Response(status=400, text="name: lowercase letters, digits and '-', up to 31 chars")
    desired = load(user, "desired.json", {})
    if desired.get(name, {}).get("state") == "run":
        return web.Response(status=409, text=f"{name} is already running")
    workdir = f.get("workdir", "").strip() or desired.get(name, {}).get("workdir") or f"~/agents/{name}/work"
    if not (workdir.startswith("~/") or workdir == "~") or ".." in workdir.split("/"):
        return web.Response(status=400, text="workdir must be inside the agent account's home (~/...)")
    cpus, mem, tlim = f.get("cpus") or "4", f.get("mem") or "16G", f.get("time") or "2-00:00:00"
    if not size_ok(cpus, mem, tlim):
        return web.Response(status=400, text="bad cpus/mem/time")
    desired[name] = dict(state="run", workdir=workdir, created=desired.get(name, {}).get("created") or time.strftime("%F %H:%M"))
    save(user, "desired.json", desired)
    try:
        hosttxt = await asyncio.get_running_loop().run_in_executor(None, ensure_host, user, cpus, mem, tlim)
    except RuntimeError as e:
        return web.Response(status=500, text=str(e))
    return web.Response(text=f"{name}: requested; {hosttxt}. The terminal appears once it is running.")


async def set_state(request, state):
    user, name = request["user"], request.match_info["name"]
    desired = load(user, "desired.json", {})
    if name not in desired:
        raise web.HTTPNotFound()
    desired[name]["state"] = state; save(user, "desired.json", desired)
    if state == "run":
        try:
            hosttxt = await asyncio.get_running_loop().run_in_executor(None, ensure_host, user)
        except RuntimeError as e:
            return web.Response(status=500, text=str(e))
        return web.Response(text=f"resuming {name}; {hosttxt}")
    return web.Response(text=f"stopping {name}; its conversation is kept and can be resumed")


async def stop_session(request):
    return await set_state(request, "stop")


async def resume_session(request):
    return await set_state(request, "run")


async def host_log(request):
    rep = load(request["user"], "report.json", {})
    return web.Response(text=rep.get("log") or "(no report from a host job yet)")


async def term(request):
    user, name = request["user"], request.match_info["name"]
    _, live, _, _ = sessions_view(user)
    a = live.get(name, {}).get("attach") if NAME_RE.match(name) else None
    if not a:
        return web.Response(status=503, text="session not running (or still starting)")
    url = f"http://{a['host']}:{a['port']}{request.path_qs}"
    auth_ = BasicAuth(a["user"], a["password"])
    if request.headers.get("Upgrade", "").lower() == "websocket":
        ws_client = web.WebSocketResponse(protocols=("tty",))
        await ws_client.prepare(request)
        async with ClientSession(timeout=ClientTimeout(total=None)) as s:
            async with s.ws_connect(url.replace("http://", "ws://"), protocols=("tty",), auth=auth_) as ws_up:
                async def up():
                    async for m in ws_client:
                        if m.type == WSMsgType.BINARY: await ws_up.send_bytes(m.data)
                        elif m.type == WSMsgType.TEXT: await ws_up.send_str(m.data)
                        else: break
                async def down():
                    async for m in ws_up:
                        if m.type == WSMsgType.BINARY: await ws_client.send_bytes(m.data)
                        elif m.type == WSMsgType.TEXT: await ws_client.send_str(m.data)
                        else: break
                await asyncio.wait([asyncio.create_task(up()), asyncio.create_task(down())],
                                   return_when=asyncio.FIRST_COMPLETED)
        return ws_client
    async with ClientSession() as s:
        async with s.request(request.method, url, auth=auth_, data=await request.read(),
                             headers={k: v for k, v in request.headers.items()
                                      if k.lower() in ("accept", "content-type", "accept-encoding")}) as r:
            hdrs = {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "content-encoding")}
            return web.Response(status=r.status, body=await r.read(), headers=hdrs)


# ---------------------------------------------------------------- control API (agent host jobs)
def control_auth(request):
    user = request.match_info["user"]
    if not USER_RE.match(user):
        raise web.HTTPNotFound()
    h = load(user, "host.json", {})
    if not (h.get("token") and hmac.compare_digest(request.headers.get("X-Workbench-Token", ""), h["token"])):
        raise web.HTTPUnauthorized()
    return user


async def control_desired(request):
    user = control_auth(request)
    return web.json_response({"sessions": load(user, "desired.json", {})})


async def control_report(request):
    user = control_auth(request)
    rep = await request.json()
    rep["t"] = time.time()
    desired = load(user, "desired.json", {})
    changed = False
    for n, s in rep.get("sessions", {}).items():
        if s.get("exited") and desired.get(n, {}).get("state") == "run":
            desired[n]["state"] = "stop"; changed = True           # the person quit pi in the terminal
    if changed:
        save(user, "desired.json", desired)
    if "log" not in rep:                                           # the log tail comes every ~15 s
        rep["log"] = load(user, "report.json", {}).get("log", "")
    save(user, "report.json", rep)
    return web.json_response({"ok": True})


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Agents</title><style>
body{{font:14px system-ui,sans-serif;margin:2em;max-width:1100px}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}} .running{{color:#0a7a2f}} .starting{{color:#a66b00}}
.stopped{{color:#777}} .note{{color:#555}} input{{padding:3px}} button{{padding:3px 10px}}</style></head>
<body>{body}</body></html>"""


def sync_runtime():
    """Agent jobs run as other accounts and cannot read this service's tree: publish the job-side code into the
    world-readable agent tools directory (owned by this service)."""
    dst = os.path.join(S["AGENT_TOOLS"], "workbench"); os.makedirs(dst, exist_ok=True); os.chmod(dst, 0o755)
    for f in ("supervisor.py",):
        src = os.path.join(ROOT, "workbench", f); tmp = os.path.join(dst, f + ".tmp")
        with open(src, "rb") as a, open(tmp, "wb") as b:
            b.write(a.read())
        os.chmod(tmp, 0o644); os.replace(tmp, os.path.join(dst, f))


async def main():
    os.makedirs(USERS, mode=0o700, exist_ok=True); os.chmod(STATE, 0o700)
    sync_runtime()
    ui = web.Application(middlewares=[identity])
    ui.router.add_get("/", index)
    ui.router.add_post("/api/new", new_session)
    ui.router.add_post("/api/stop/{name}", stop_session)
    ui.router.add_post("/api/resume/{name}", resume_session)
    ui.router.add_get("/api/log", host_log)
    ui.router.add_route("*", "/agents/{name}/{tail:.*}", term)
    ctl = web.Application(client_max_size=1 << 20)
    ctl.router.add_get("/control/{user}/desired", control_desired)
    ctl.router.add_post("/control/{user}/report", control_report)
    r1, r2 = web.AppRunner(ui, access_log=None), web.AppRunner(ctl, access_log=None)
    await r1.setup(); await r2.setup()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    await web.UnixSite(r1, SOCK).start(); os.chmod(SOCK, 0o600)
    await web.TCPSite(r2, *CONTROL).start()
    log(f"dashboard: ui {SOCK} (via identity proxy), control {CONTROL[0]}:{CONTROL[1]}, "
        f"runas={S.get('RUNAS_MODE', 'same')}, submit={S.get('SUBMIT_MODE', 'sbatch')}")
    await asyncio.Event().wait()


asyncio.run(main())
