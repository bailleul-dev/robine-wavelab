#!/bin/sh
# Convert every LiveSPICE test schematic the converter supports.
# Usage: scripts/convert-livespice.sh <LiveSPICE checkout> [circuits dir]
set -u
tests="${1:?path to a LiveSPICE checkout}/Tests"
out="${2:-../robine-spice/circuits}"
lab="$(dirname "$0")/../target/release/spice-lab"
mkdir -p "$out/basics"

convert() {
    for f in "$tests/$1"/*.schx; do
        slug=$(basename "$f" .schx | tr 'A-Z' 'a-z' | sed 's/+/-plus-/g; s/[^a-z0-9]\{1,\}/-/g; s/^-//; s/-$//')
        if msg=$("$lab" convert "$f" -o "$2/$slug.cir" 2>&1); then
            echo "ok    $2/$slug.cir"
        else
            rm -f "$2/$slug.cir"
            echo "skip  $(basename "$f"): ${msg#error: }"
        fi
    done
}

convert Examples "$out"
convert Unstable "$out"
convert Circuits "$out/basics"
