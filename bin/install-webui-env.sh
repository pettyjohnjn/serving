#!/bin/bash
# Build the Open WebUI environment for the login node (x86_64). Idempotent; run as the service account:
#   bin/install-webui-env.sh
# Installs into $WEBUI_VENV (etc/site.env; keep it on a large shared disk, not a small root disk). CPU-only torch:
# Open WebUI only needs it for its embedding model. bin/webui patches the installed package on every start.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
. "$ROOT/etc/site.env"
OPEN_WEBUI_VERSION=${OPEN_WEBUI_VERSION:-0.11.0}     # the version bin/webui's patches and webui-ring.js target
[ "$(uname -m)" = x86_64 ] || { echo "run on the login node (x86_64)"; exit 1; }
export UV_CACHE_DIR=${UV_CACHE_DIR:-$(dirname "$WEBUI_VENV")/uv-cache}
mkdir -p "$(dirname "$WEBUI_VENV")"
[ -x "$WEBUI_VENV/bin/python" ] || nice -n 10 uv venv -q "$WEBUI_VENV" --python python3.12
nice -n 10 uv pip install -q --python "$WEBUI_VENV/bin/python" "open-webui==$OPEN_WEBUI_VERSION" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$WEBUI_VENV/bin/python" -c "import open_webui, torch, aiohttp; print('open-webui env ok:', torch.__version__)"
