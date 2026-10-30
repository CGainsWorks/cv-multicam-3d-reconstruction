#!/bin/sh
# Give the loopback interface one address per simulated camera.
#
#   scripts/genicam/loopback_addresses.sh 3      # 127.0.0.1, .2 and .3
#
# Every GigE Vision device answers GVCP on UDP port 3956 of its own address, so
# three simulated cameras need three addresses.  127.0.0.1 is already there;
# this adds 127.0.0.2 onwards to `lo`.  It needs CAP_NET_ADMIN: sudo on a CI
# runner, `--cap-add NET_ADMIN` in a container, where it only touches the
# container's own network namespace.  Nothing leaves the machine.

set -eu

n=${1:-3}
i=2
while [ "$i" -le "$n" ]; do
    addr="127.0.0.$i"
    if ! ip -4 addr show dev lo | grep -q "inet $addr/"; then
        if [ "$(id -u)" -eq 0 ]; then
            ip addr add "$addr/8" dev lo
        else
            sudo ip addr add "$addr/8" dev lo
        fi
    fi
    i=$((i + 1))
done
ip -4 addr show dev lo | grep inet
