#!/bin/bash
# grant_tunnel_access.sh — give someone LLM access WITHOUT giving them a shell.
#
#   ./grant_tunnel_access.sh alice ~/keys/alice.pub
#
# Appends a locked-down entry to ~/.ssh/authorized_keys that can do exactly one thing:
# forward a TCP connection to 127.0.0.1:8000 on globus1 (where the serving node publishes
# itself over its own reverse tunnel). No shell, no pty, no agent forwarding, no
# SFTP, no other destinations. The holder can use the OpenAI endpoint and nothing else.
#
# This is the natural way to "ride" the cluster's existing SSH-key authentication: the
# key IS the credential, it is revoked by deleting one line, and the machinery for
# distributing and trusting keys already exists.
#
# NOTE: this grants access under YOUR account. Anyone you add can reach the endpoint as
# a network destination, so give them their own vLLM API key from etc/keys.env too if
# you want per-person attribution in the logs.

set -euo pipefail

NAME=${1:-}
PUBKEY=${2:-}
AUTH=${HOME}/.ssh/authorized_keys
TARGET=${TARGET:-127.0.0.1:8000}

if [ -z "$NAME" ] || [ -z "$PUBKEY" ]; then
    echo "usage: $0 <label> <path-to-public-key>" >&2
    echo "       $0 --list        show current tunnel-only grants" >&2
    echo "       $0 --revoke NAME remove a grant" >&2
    exit 2
fi

if [ "$NAME" = "--list" ]; then
    grep -n 'permitopen=' "$AUTH" 2>/dev/null | sed 's/ssh-[a-z0-9-]* [A-Za-z0-9+/=]*/<key>/' || echo "(none)"
    exit 0
fi
if [ "$NAME" = "--revoke" ]; then
    LABEL=$PUBKEY
    cp "$AUTH" "$AUTH.bak.$(date +%s)"
    grep -v "llm-tunnel:${LABEL}\$" "$AUTH" > "$AUTH.tmp" && mv "$AUTH.tmp" "$AUTH"
    chmod 600 "$AUTH"; echo "revoked llm-tunnel:${LABEL}"
    exit 0
fi

[ -r "$PUBKEY" ] || { echo "cannot read $PUBKEY" >&2; exit 1; }
ssh-keygen -lf "$PUBKEY" >/dev/null || { echo "$PUBKEY is not a valid public key" >&2; exit 1; }

KEY=$(awk '{print $1" "$2}' "$PUBKEY")
mkdir -p "$(dirname "$AUTH")"; touch "$AUTH"; chmod 600 "$AUTH"

if grep -qF "$(awk '{print $2}' "$PUBKEY")" "$AUTH"; then
    echo "that key is already present in $AUTH -- refusing to add twice" >&2
    exit 1
fi

# `restrict` turns everything off, then port-forwarding is selectively re-enabled and
# pinned to one destination with permitopen. Order matters: restrict must come first.
#
# command="/bin/false" is NOT redundant. `restrict` disables PTY allocation, but it does
# NOT disable command execution -- without the forced command, `ssh key@host 'cmd'` still
# runs cmd and the holder effectively has a shell. Verified on this cluster:
#   command execution  -> blocked        interactive shell   -> "PTY allocation request failed"
#   forward 127.0.0.1:8000 -> works (200)   forward anywhere else -> "administratively prohibited"
# `-N` (what llm-tunnel uses) never requests a command, so forwarding is unaffected.
printf 'restrict,port-forwarding,permitopen="%s",command="/bin/false" %s llm-tunnel:%s\n' \
    "$TARGET" "$KEY" "$NAME" >> "$AUTH"

echo "granted tunnel-only access to '$NAME' (destination $TARGET)"
echo
echo "Tell them to run:"
echo "  ssh -N -L 8000:${TARGET} ${USER}@<login-node>"
echo "then use base_url http://localhost:8000/v1"
