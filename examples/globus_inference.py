"""globus_inference — the analogue of ALCF's `inference_auth_token`, for the globus endpoint.

ALCF:
    from openai import OpenAI
    from inference_auth_token import get_access_token
    client = OpenAI(api_key=get_access_token(),
                    base_url="https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1")

globus:
    from openai import OpenAI
    from globus_inference import get_access_token, BASE_URL
    client = OpenAI(api_key=get_access_token(), base_url=BASE_URL)

...or just `client = globus_inference.get_client()`, which additionally checks the tunnel and
tells you what to do when it is down.

The auth models differ in an important way. ALCF issues you a bearer token over Globus Auth,
so the endpoint is public and the token is the credential. Here the endpoint is not public at
all: it listens on globus1's loopback only, so possessing an SSH session to globus1 *is* the
credential. There is no token to fetch, refresh, or leak — `get_access_token()` exists purely
so your ALCF-shaped code keeps working unchanged.

Drop this file next to your script, or add this directory to PYTHONPATH.
"""
from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request

__all__ = ["get_access_token", "get_client", "BASE_URL", "MODEL", "EndpointDown"]

MODEL = "qwen3.8-27b"
DEFAULT_PORT = int(os.environ.get("GLOBUS_LLM_PORT", "8000"))
BASE_URL = os.environ.get("OPENAI_BASE_URL", f"http://localhost:{DEFAULT_PORT}/v1")


class EndpointDown(RuntimeError):
    """The endpoint could not be reached, with the actual next step to take."""


def get_access_token() -> str:
    """Return an API key.

    Unlike ALCF there is no token to obtain: the server runs with no API key because it is
    only reachable through an authenticated SSH tunnel. The openai client library still
    requires a non-empty string, so return a placeholder. Override with OPENAI_API_KEY if
    the deployment is ever switched to REQUIRE_API_KEY=1.
    """
    return os.environ.get("OPENAI_API_KEY") or "sk-local"


def _port_from(base_url: str) -> int:
    try:
        return int(base_url.split("//", 1)[1].split("/", 1)[0].rsplit(":", 1)[-1])
    except (IndexError, ValueError):
        return DEFAULT_PORT


def check(base_url: str = BASE_URL, timeout: float = 5.0) -> None:
    """Raise EndpointDown with actionable advice, instead of a bare connection error.

    Distinguishes the two failure modes that look identical to the openai client:
    nothing listening locally (no tunnel) versus a tunnel whose far side is empty
    (server not started, or still loading — a cold start takes about 5 minutes).
    """
    port = _port_from(base_url)
    tunnel_cmd = f"ssh -N -L {port}:127.0.0.1:8000 globus1"

    with socket.socket() as s:
        s.settimeout(timeout)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            raise EndpointDown(
                f"nothing is listening on localhost:{port} — the SSH tunnel is not up.\n"
                f"  start it:  {tunnel_cmd}")

    health = base_url.rsplit("/v1", 1)[0] + "/health"
    try:
        urllib.request.urlopen(health, timeout=timeout)
    except urllib.error.HTTPError:
        pass  # answered, just not 2xx — good enough to prove something is serving
    except (urllib.error.URLError, OSError) as e:
        raise EndpointDown(
            f"the tunnel on localhost:{port} is up, but nothing is serving behind it "
            f"({type(e).__name__}).\n"
            f"  the model may still be loading — a cold start is ~5 min\n"
            f"  check:     ssh globus1 serving status\n"
            f"  start it:  ssh globus1 serving start") from None


def get_client(base_url: str = BASE_URL, verify: bool = True, **kwargs):
    """An OpenAI client pointed at the endpoint, pre-flighted so failures are legible."""
    from openai import OpenAI

    if verify:
        check(base_url)
    kwargs.setdefault("timeout", 900.0)
    return OpenAI(api_key=get_access_token(), base_url=base_url, **kwargs)
