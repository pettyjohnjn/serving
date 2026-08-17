#!/usr/bin/env python3
"""fair-proxy — work-conserving max-min fairness for the model endpoint.

Sits on globus1:8000 (what clients tunnel to) in front of the reverse-tunnel
listener on 8005 (where the compute node publishes vLLM). Each connection is
attributed to the cluster account that opened it — the peer socket on loopback is
owned by that user's sshd, and the kernel records the uid.

Scheduling: a single global pool of FAIR_TOTAL slots (default 32, matching vLLM's
MAX_SEQS so queueing decisions happen HERE, not FIFO inside vLLM).

  * Below capacity, everyone is admitted immediately — a lone user can hold all 32
    slots. Nothing idles for fairness's sake.
  * At capacity, requests queue, and each freed slot goes to the queued user with
    the FEWEST requests in flight (ties: least recent usage, then arrival order).
    So when a second user shows up against a saturated sweep, their first request
    takes the very next freed slot; the sweep drifts down to 31, and toward an
    even split only if the newcomer keeps submitting.

Usage metric for tie-breaks: decayed slot-seconds (how long your requests have held
slots lately, 10-minute half-life). It tracks compute share without parsing bodies;
aggregate-token accounting would need response parsing and buys little over this.

Slots are per REQUEST (acquired at the request head, released when its response
completes — Content-Length counted, chunked until the 0-chunk, which also covers
SSE). Idle keep-alive connections hold nothing. Overflow is held, never rejected.

GET /fair-stats (from any user) returns the live per-user picture as JSON.

Env: FAIR_LISTEN (8000), FAIR_UPSTREAM (8005), FAIR_TOTAL (32),
     FAIR_PER_USER (optional hard per-user ceiling on top; unset = none).
Run by `serving supervise`; logs are one line per state change, never content.
"""
import asyncio
import collections
import itertools
import json
import os
import pwd
import sys
import time

LISTEN = ("127.0.0.1", int(os.environ.get("FAIR_LISTEN", "8000")))
UPSTREAM = ("127.0.0.1", int(os.environ.get("FAIR_UPSTREAM", "8005")))
TOTAL = int(os.environ.get("FAIR_TOTAL", "32"))
PER_USER = int(os.environ.get("FAIR_PER_USER", "0"))    # 0 = no per-user ceiling
USAGE_HALFLIFE = 600.0
MAX_HEADER = 256 * 1024


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FairScheduler:
    """Global slot pool with max-min fair admission under contention."""

    def __init__(self, total):
        self.total = total
        self.inflight = collections.Counter()
        self._usage = {}                       # user -> (decayed slot-seconds, ts)
        self._waiters = []                     # [(seq, user, Future)]
        self._seq = itertools.count()

    def usage(self, user):
        v, ts = self._usage.get(user, (0.0, time.time()))
        return v * (0.5 ** ((time.time() - ts) / USAGE_HALFLIFE))

    def _charge(self, user, held):
        self._usage[user] = (self.usage(user) + held, time.time())

    def _capped(self, user):
        return PER_USER and self.inflight[user] >= PER_USER

    async def acquire(self, user):
        if sum(self.inflight.values()) < self.total and not self._capped(user):
            self.inflight[user] += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((next(self._seq), user, fut))
        if len(self._waiters) == 1 or self.inflight[user] == 0:
            log(f"{user}: queued (pool full: {dict(self.inflight)})")
        await fut                              # resolved by release(); inflight
                                               # is incremented by the releaser.

    def release(self, user, held):
        self.inflight[user] -= 1
        if self.inflight[user] <= 0:
            del self.inflight[user]
        self._charge(user, held)
        # hand the freed slot to the most deserving waiter
        while self._waiters:
            live = [w for w in self._waiters if not w[2].cancelled()]
            if not live:
                self._waiters.clear()
                return
            live = [w for w in live if not self._capped(w[1])]
            if not live:
                return                          # everyone waiting is at their ceiling
            seq, wuser, fut = min(
                live, key=lambda w: (self.inflight[w[1]], self.usage(w[1]), w[0]))
            self._waiters.remove((seq, wuser, fut))
            self.inflight[wuser] += 1
            fut.set_result(None)
            return

    def snapshot(self):
        return {
            "total_slots": self.total,
            "in_flight": dict(self.inflight),
            "queued": collections.Counter(u for _, u, f in self._waiters
                                          if not f.cancelled()),
            "recent_usage_slot_seconds": {u: round(self.usage(u), 1)
                                          for u in set(self._usage) | set(self.inflight)},
        }


SCHED = FairScheduler(TOTAL)


def uid_of_peer(port):
    want_local = f"0100007F:{port:04X}"
    want_rem = f"0100007F:{LISTEN[1]:04X}"
    with open("/proc/net/tcp") as f:
        next(f)
        for line in f:
            p = line.split()
            if p[1] == want_local and p[2] == want_rem:
                return int(p[7])
    return None


def user_of_peer(port):
    uid = uid_of_peer(port)
    if uid is None:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid{uid}"


async def read_head(reader):
    raw = await reader.readuntil(b"\r\n\r\n")
    headers = {}
    for line in raw.split(b"\r\n")[1:]:
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.strip().lower().decode()] = v.strip().decode(errors="replace")
    return raw, headers


async def copy_body(headers, reader, writer):
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            writer.write(size_line)
            size = int(size_line.strip().split(b";")[0], 16)
            writer.write(await reader.readexactly(size + 2))
            await writer.drain()
            if size == 0:
                return
    n = int(headers.get("content-length", 0))
    while n > 0:
        chunk = await reader.read(min(n, 65536))
        if not chunk:
            raise ConnectionResetError("EOF mid-body")
        writer.write(chunk)
        await writer.drain()
        n -= len(chunk)


def stats_response(snapshot):
    counts = snapshot["queued"]
    snapshot["queued"] = dict(counts)
    body = json.dumps(snapshot, indent=1).encode()
    return (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Cache-Control: no-store\r\nContent-Length: " + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n" + body)


async def handle(client_r, client_w):
    peer = client_w.get_extra_info("peername")
    user = user_of_peer(peer[1]) if peer else None
    if user is None:
        client_w.close()
        return
    up_r = up_w = None
    try:
        while True:
            try:
                req_raw, req_h = await read_head(client_r)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            first_line = req_raw.split(b"\r\n", 1)[0]
            method = first_line.split(b" ", 1)[0]

            if method == b"GET" and first_line.split(b" ")[1].split(b"?")[0] == b"/fair-stats":
                client_w.write(stats_response(SCHED.snapshot()))
                await client_w.drain()
                return

            if up_w is None:
                try:
                    up_r, up_w = await asyncio.open_connection(*UPSTREAM)
                except OSError:
                    client_w.write(
                        b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\n"
                        b"Connection: close\r\n\r\n"
                        b"model endpoint not published (job down or reloading)\r\n")
                    await client_w.drain()
                    return

            await SCHED.acquire(user)
            t0 = time.time()
            try:
                up_w.write(req_raw)
                await up_w.drain()
                await copy_body(req_h, client_r, up_w)

                resp_raw, resp_h = await read_head(up_r)
                client_w.write(resp_raw)
                await client_w.drain()
                status = int(resp_raw.split(b" ", 2)[1])
                while status < 200:            # 1xx: another head follows
                    resp_raw, resp_h = await read_head(up_r)
                    client_w.write(resp_raw)
                    await client_w.drain()
                    status = int(resp_raw.split(b" ", 2)[1])
                bodyless = (method == b"HEAD" or status in (204, 304))
                if not bodyless:
                    if ("content-length" not in resp_h
                            and resp_h.get("transfer-encoding", "").lower() != "chunked"):
                        while True:            # read-to-close framing
                            chunk = await up_r.read(65536)
                            if not chunk:
                                return
                            client_w.write(chunk)
                            await client_w.drain()
                    await copy_body(resp_h, up_r, client_w)
            finally:
                SCHED.release(user, time.time() - t0)

            if (req_h.get("connection", "").lower() == "close"
                    or resp_h.get("connection", "").lower() == "close"):
                return
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError,
            asyncio.LimitOverrunError, ValueError):
        pass
    finally:
        for w in (client_w, up_w):
            try:
                if w:
                    w.close()
            except Exception:
                pass


async def main():
    server = await asyncio.start_server(handle, *LISTEN, limit=MAX_HEADER)
    log(f"fair-proxy: {LISTEN[0]}:{LISTEN[1]} -> {UPSTREAM[0]}:{UPSTREAM[1]}, "
        f"pool {TOTAL}, max-min fair under contention"
        + (f", per-user ceiling {PER_USER}" if PER_USER else ""))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
