#!/bin/bash
# Install the agent runtime for the compute nodes (aarch64): Node.js, Pi, ttyd, into $AGENT_TOOLS (world-readable,
# shared by every agent account). Run as the service account ON THE LOGIN NODE:
#   bin/install-agent-tools.sh
# The build runs on a compute node (Pi's npm dependencies must be built for aarch64) in node-local /tmp, and comes
# back as ONE tarball that is unpacked here, where the shared disk is local: writing thousands of small files over
# NFS from a compute node is ~3 files/s. Idempotent: skips when the installed versions already match.
# Per-account Pi configuration (packages, the lab profile, model entry) is done by jobs/agent-host.sbatch on an
# account's first session, not here.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
. "$ROOT/etc/site.env"
NODE_VERSION=${NODE_VERSION:-v24.21.0}
PI_PACKAGE=${PI_PACKAGE:-@earendil-works/pi-coding-agent@1.0.0}
TTYD_VERSION=${TTYD_VERSION:-1.7.7}
WANT="node=$NODE_VERSION pi=$PI_PACKAGE ttyd=$TTYD_VERSION"
T=$AGENT_TOOLS

if [ "${1:-}" = --build ]; then          # runs on the compute node
  OUT=$2; B=$(mktemp -d /tmp/agent-tools.XXXX); trap 'rm -rf "$B"' EXIT
  umask 022; cd "$B"; mkdir -p bin
  f=node-$NODE_VERSION-linux-arm64.tar.xz
  curl -sfLO "https://nodejs.org/dist/$NODE_VERSION/$f"
  curl -sfL "https://nodejs.org/dist/$NODE_VERSION/SHASUMS256.txt" | grep " $f\$" | sha256sum -c -
  tar -xJf "$f" && rm -f "$f" && ln -s "node-$NODE_VERSION-linux-arm64" node
  PATH="$B/node/bin:$PATH" NPM_CONFIG_PREFIX="$B/npm" npm install -g --no-fund --no-audit "$PI_PACKAGE" >/dev/null
  curl -sfL -o bin/ttyd "https://github.com/tsl0922/ttyd/releases/download/$TTYD_VERSION/ttyd.aarch64"
  curl -sfL "https://github.com/tsl0922/ttyd/releases/download/$TTYD_VERSION/SHA256SUMS" | grep "ttyd.aarch64\$" \
    | awk '{print $1"  bin/ttyd"}' | sha256sum -c -
  chmod 755 bin/ttyd
  PATH="$B/node/bin:$B/npm/bin:$B/bin:$PATH"; echo "built: node $(node --version), pi $(pi --version), $(ttyd --version)"
  echo "$WANT" > VERSION
  tar -czf "$OUT.tmp" . && mv "$OUT.tmp" "$OUT"
  exit 0
fi

[ "$(uname -m)" = x86_64 ] || { echo "run on the login node; the build step goes to a compute node by itself"; exit 1; }
if [ -f "$T/VERSION" ] && [ "$(cat "$T/VERSION")" = "$WANT" ]; then echo "agent tools current ($WANT)"; exit 0; fi
mkdir -p "$(dirname "$T")"
TARBALL=$(dirname "$T")/.agent-tools-build.tgz
srun -p "$AGENT_PARTITION" --exclude="$AGENT_NODES_EXCLUDE" -n1 -c4 --mem=8G -t 30 \
  bash "$0" --build "$TARBALL" 2>&1 | grep -v '^srun'
NEW=$T.new; rm -rf "$NEW"; mkdir -p "$NEW"; (umask 022; tar -xzf "$TARBALL" -C "$NEW"); rm -f "$TARBALL"
# The npm bin links point into the tree with relative paths, so the tree is relocatable.
cat > "$NEW/env.sh" <<EOF
# Agent runtime (aarch64). Sourced by jobs/agent-host.sbatch.
export PATH="$T/node/bin:$T/npm/bin:$T/bin:\$PATH"
EOF
chmod -R a+rX,go-w "$NEW"     # every agent account executes this tree: nobody else may write it
rm -rf "$T.old"; [ -d "$T" ] && mv "$T" "$T.old"; mv "$NEW" "$T"; rm -rf "$T.old"
echo "installed into $T: $(cat "$T/VERSION")"
