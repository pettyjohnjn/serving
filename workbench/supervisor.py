#!/usr/bin/env python3
"""Agent host supervisor: runs inside jobs/agent-host.sbatch as the agent account and keeps that person's Pi
sessions in the state the dashboard asks for.

Every 2 s: GET $WB_CONTROL/control/<user>/desired, start what should run and does not (Pi in its own tmux session
plus a password-protected ttyd on the cluster network), stop what should not run, restart a dead ttyd, then POST the
live picture (ttyd address + password per session, sessions whose Pi exited, a tail of this log) to
$WB_CONTROL/control/<user>/report. Authenticated with the per-job token in $WB_TOKEN.

Files (agent account's home): ~/agents/tmux.sock, ~/agents/<name>/pi (Pi's session store; --session-id <name>
resumes it after a stop or a requeue), ~/agents/<name>/exited, ~/agents/supervisor.log.
Exits (ending the job) after IDLE_MIN minutes with nothing to run.
"""
import json, os, secrets, signal, socket, subprocess, time, urllib.request

USER = os.environ["WB_USER"]                 # the person (identity); this process runs as their agent account
CONTROL = os.environ["WB_CONTROL"].rstrip("/") + f"/control/{USER}"
TOKEN = os.environ["WB_TOKEN"]
IP = os.environ["HOST_IP"]
HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "agents")
SOCK = os.path.join(BASE, "tmux.sock")
LOG = os.path.join(BASE, "supervisor.log")
JOB = os.environ.get("SLURM_JOB_ID", "?")
IDLE_MIN = float(os.environ.get("IDLE_MIN", "30"))
NAME_OK = set("abcdefghijklmnopqrstuvwxyz0123456789-")
ttyds, attach, errors = {}, {}, {}
stopping = False


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def call(path, body=None):
    req = urllib.request.Request(CONTROL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"X-Workbench-Token": TOKEN, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=15))


def tmux(*args, check=False):
    return subprocess.run(["tmux", "-S", SOCK, *args], capture_output=True, text=True, check=check)


def live_sessions():
    return {s[2:] for s in tmux("list-sessions", "-F", "#{session_name}").stdout.split() if s.startswith("s-")}


def resolve_workdir(spec, name):
    w = spec.get("workdir") or f"~/agents/{name}/work"
    p = os.path.realpath(os.path.expanduser(w))
    if not (p == HOME or p.startswith(HOME + os.sep)):
        raise ValueError(f"workdir outside {HOME}")
    return p


def start(name, spec):
    d = os.path.join(BASE, name)
    workdir = resolve_workdir(spec, name)
    os.makedirs(workdir, exist_ok=True); os.makedirs(os.path.join(d, "pi"), exist_ok=True)
    marker = os.path.join(d, "exited")
    if os.path.exists(marker):
        os.remove(marker)
    cmd = f"pi --session-dir '{d}/pi' --session-id '{name}' --name '{name}'; date -Is > '{marker}'"
    tmux("new-session", "-d", "-s", f"s-{name}", "-x", "220", "-y", "50", "-c", workdir, cmd, check=True)
    log("started", name, "in", workdir)


def ensure_ttyd(name):
    p = ttyds.get(name)
    if p and p.poll() is None:
        return
    s = socket.socket(); s.bind((IP, 0)); port = s.getsockname()[1]; s.close()
    pw = secrets.token_urlsafe(18)
    ttyds[name] = subprocess.Popen(
        ["ttyd", "-i", IP, "-p", str(port), "-b", f"/agents/{name}", "-c", f"pi:{pw}", "-W",
         "-t", f"titleFixed=agent {name}", "-t", "fontSize=14", "tmux", "-S", SOCK, "attach", "-t", f"s-{name}"],
        stdout=open(os.path.join(BASE, name, "ttyd.log"), "a"), stderr=subprocess.STDOUT)
    attach[name] = dict(host=IP, port=port, user="pi", password=pw)


def stop(name, reason):
    tmux("kill-session", "-t", f"s-{name}")
    p = ttyds.pop(name, None)
    if p and p.poll() is None:
        p.terminate()
    attach.pop(name, None)
    log("stopped", name, f"({reason})")


def log_tail(n=6000):
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - n)); return f.read().decode(errors="replace")
    except OSError:
        return ""


def reconcile(desired):
    live = live_sessions(); exited = set()
    for name, spec in desired.items():
        if not set(name) <= NAME_OK or len(name) > 31:
            continue
        want = spec.get("state") == "run"
        if name not in live and name in ttyds and os.path.exists(os.path.join(BASE, name, "exited")):
            stop(name, "pi exited"); exited.add(name); continue      # the person quit pi in the terminal
        try:
            if want and name not in live:
                start(name, spec)
            if want:
                ensure_ttyd(name); errors.pop(name, None)
            elif name in live or name in ttyds:
                stop(name, "requested")
        except (ValueError, subprocess.CalledProcessError, OSError) as e:
            errors[name] = f"error: {e}"; log("start failed", name, e)
    for name in list(ttyds):                                          # removed from desired entirely
        if name not in desired:
            stop(name, "removed")
    return {n: {"attach": attach.get(n), "exited": n in exited, "error": errors.get(n)}
            for n in set(desired) | exited}, any(s.get("state") == "run" for s in desired.values())


def shutdown(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, shutdown)
os.makedirs(BASE, mode=0o700, exist_ok=True)
tmux("new-session", "-d", "-s", "_host", "sleep infinity")             # keeps the tmux server alive
for opt in (("-g", "mouse", "on"), ("-g", "history-limit", "50000"), ("-s", "extended-keys", "on")):
    tmux("set-option", *opt)
log(f"host up: {USER} as {os.environ.get('USER')} job {JOB} on {socket.gethostname()}")
idle_since, last_log = None, 0.0
while not stopping:
    try:
        desired = call("/desired")["sessions"]
        sessions, busy = reconcile(desired)
        body = {"host": {"job": JOB, "node": socket.gethostname(), "account": os.environ.get("USER")},
                "sessions": sessions}
        if time.time() - last_log > 15:
            body["log"] = log_tail(); last_log = time.time()
        call("/report", body)
    except Exception as e:                                             # dashboard restarting: keep sessions running
        log("control unreachable:", e); busy = bool(ttyds)
    if busy:
        idle_since = None
    else:
        idle_since = idle_since or time.time()
        if time.time() - idle_since > IDLE_MIN * 60:
            log(f"idle for {IDLE_MIN:g} min; ending host job"); break
    time.sleep(2)
for n in list(ttyds):
    p = ttyds.pop(n)
    if p.poll() is None:
        p.terminate()
tmux("kill-server")
log("host down")
