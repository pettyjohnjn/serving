#!/usr/bin/env python3
"""repair-proxy — front the endpoint and repair malformed tool-call arguments.

Some clients mis-accumulate streamed tool-call deltas and store "valid JSON + trailing
junk" in tool_calls[].function.arguments (seen with Unsloth Studio: it appends the
arguments twice). vLLM json.loads()es that field when rendering the chat template, so
every follow-up in the conversation then 400s with "Extra data: line 1 column N" and
retrying cannot recover.

This proxy truncates each arguments string to its first complete JSON value
(json.JSONDecoder().raw_decode) and passes everything else through untouched, streaming
included. It logs only repair events -- lengths and a short excerpt of the discarded
tail, never whole prompts. Counters at /proxy-stats.

Run on globus1:    ./repair-proxy.py --listen 8001 --upstream 8000
Laptop tunnel:     ssh -N -L 8000:127.0.0.1:8001 globus1
"""
import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DECODER = json.JSONDecoder()
STATS = {"requests": 0, "repaired_calls": 0, "repaired_requests": 0}
LOCK = threading.Lock()


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def repair_arguments(raw):
    """Return (fixed, discarded) if `raw` is JSON followed by junk, else (raw, None)."""
    if not isinstance(raw, str) or not raw.strip():
        return raw, None
    try:
        json.loads(raw)
        return raw, None                      # already clean
    except json.JSONDecodeError:
        pass
    try:
        _, end = DECODER.raw_decode(raw.lstrip())
    except json.JSONDecodeError:
        return raw, None                      # truncated or garbage: not ours to fix
    offset = len(raw) - len(raw.lstrip()) + end
    discarded = raw[offset:]
    if not discarded.strip():
        return raw, None
    return raw[:offset], discarded


def repair_body(body):
    """Repair tool-call arguments in a chat-completions body. Returns (new_body, n)."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body, 0
    if not isinstance(payload, dict):
        return body, 0
    n = 0
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            fixed, discarded = repair_arguments(fn.get("arguments"))
            if discarded is None:
                continue
            n += 1
            # Lengths and shape only — never content. This log is on a shared box.
            dup = discarded.strip().startswith(fixed[:40]) if len(fixed) >= 40 else False
            log(f"REPAIR {fn.get('name')!r}: kept {len(fixed)} ch, discarded {len(discarded)} ch"
                f"{' [duplicate of the kept JSON]' if dup else ''}")
            fn["arguments"] = fixed
    return (json.dumps(payload).encode() if n else body), n


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "repair-proxy"

    def log_message(self, *a):                # quiet; we do our own logging
        pass

    def _relay(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        n = 0
        if body and "chat/completions" in self.path:
            body, n = repair_body(body)
            with LOCK:
                STATS["requests"] += 1
                STATS["repaired_calls"] += n
                STATS["repaired_requests"] += 1 if n else 0

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
        req = urllib.request.Request(f"{self.server.upstream}{self.path}",
                                     data=body or None, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=self.server.timeout_s)
        except urllib.error.HTTPError as e:   # relay upstream errors verbatim
            payload = e.read()
            if n:
                log(f"upstream still {e.code} after repairing {n} call(s): {payload[:160]!r}")
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as e:
            msg = json.dumps({"error": {"message": f"proxy: {type(e).__name__}: {e}",
                                        "type": "ProxyError", "code": 502}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:                        # stream SSE through without buffering
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                               # client hung up mid-stream

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        if self.path == "/proxy-stats":
            with LOCK:
                body = json.dumps(STATS, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._relay("GET")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen", type=int, default=8001)
    # 8005 = the published endpoint itself. Studio traffic is interactive, so it goes
    # around the fairness proxy rather than competing with bulk agent traffic.
    p.add_argument("--upstream", type=int, default=8005)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=900.0)
    a = p.parse_args()

    srv = ThreadingHTTPServer((a.host, a.listen), Handler)
    srv.upstream = f"http://127.0.0.1:{a.upstream}"
    srv.timeout_s = a.timeout
    srv.daemon_threads = True
    log(f"listening on {a.host}:{a.listen} -> {srv.upstream}")
    log("repairs are logged here; GET /proxy-stats for counters")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log(f"stopping. {STATS}")
        sys.exit(0)


if __name__ == "__main__":
    main()
