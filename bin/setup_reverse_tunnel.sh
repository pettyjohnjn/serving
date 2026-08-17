#!/bin/bash
# setup_reverse_tunnel.sh — run ONCE on globus1. Authorises the compute-node -> globus1
# reverse tunnel that makes the endpoint's address independent of which node runs the job.
#
#   bash ~/serving/bin/setup_reverse_tunnel.sh
#
# What it does: appends ONE locked-down line to ~/.ssh/authorized_keys for the dedicated
# key ~/.ssh/id_llm_tunnel (already generated). That key can do exactly one thing —
# bind 127.0.0.1:8000 on globus1 as a reverse forward. It cannot open a shell, cannot
# run a command, cannot allocate a pty, and cannot forward anywhere else.
#
#   restrict          all restrictions on
#   port-forwarding   ...except forwarding
#   permitlisten      ...and only a remote listen on 127.0.0.1:8000
#   permitopen 127.0.0.1:1  effectively blocks -L forwards with this key
#   command=/bin/false  `restrict` does NOT block command execution on its own
#
# Undo with:  bash setup_reverse_tunnel.sh --revoke

set -euo pipefail

AUTH="$HOME/.ssh/authorized_keys"
KEY="$HOME/.ssh/id_llm_tunnel"
LABEL="llm-reverse-tunnel"
LISTEN="127.0.0.1:8000"

if [ "${1:-}" = "--revoke" ]; then
    [ -f "$AUTH" ] || { echo "no authorized_keys"; exit 0; }
    cp "$AUTH" "$AUTH.bak.$(date +%s)"
    grep -v " ${LABEL}\$" "$AUTH" > "$AUTH.tmp" && mv "$AUTH.tmp" "$AUTH"
    chmod 600 "$AUTH"
    echo "revoked '$LABEL'. Remaining keys:"; ssh-keygen -lf "$AUTH"
    exit 0
fi

[ -f "$KEY.pub" ] || { echo "missing $KEY.pub -- generate with:"; echo "  ssh-keygen -t ed25519 -N '' -f $KEY -C '$LABEL'"; exit 1; }

mkdir -p "$HOME/.ssh"; touch "$AUTH"; chmod 600 "$AUTH"

if grep -q " ${LABEL}\$" "$AUTH"; then
    echo "'$LABEL' is already authorised; nothing to do."
else
    cp "$AUTH" "$AUTH.bak.$(date +%s)"
    printf 'restrict,port-forwarding,permitlisten="%s",permitopen="127.0.0.1:1",command="/bin/false" %s %s\n' \
        "$LISTEN" "$(awk '{print $1" "$2}' "$KEY.pub")" "$LABEL" >> "$AUTH"
    chmod 600 "$AUTH"
    echo "authorised '$LABEL' (may listen on $LISTEN only)"
fi

echo
echo "authorized_keys now contains:"
ssh-keygen -lf "$AUTH"

echo
echo "Verifying the compute node can use it (via globus1 itself)..."
if ssh -i "$KEY" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes \
       -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
       -N -R "$LISTEN:127.0.0.1:22" globus1 -f -o ControlPath=none 2>/dev/null; then
    sleep 1; pkill -f "id_llm_tunnel.*globus1" 2>/dev/null || true
    echo "  reverse forward accepted -- setup complete."
else
    echo "  NOTE: could not verify automatically (port may be busy). This is usually fine;"
    echo "  the serving job retries the tunnel on a loop."
fi
