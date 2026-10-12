#!/bin/sh
# Interleaved A/B benchmark of two nam-lab binaries (e.g. two builds of the
# library). Runs ABBA rounds so that neither binary systematically goes first,
# keeps each binary's best ns/sample, prints the ratio B/A per model size.
#
#   scripts/ab.sh <nam-lab-A> <nam-lab-B> [model.nam] [rounds] [block]
#
# Expect about ±1 % noise on a quiet machine; a loaded laptop shows ±5 %.
set -eu
A=$1; B=$2
MODEL=${3:-../robine-nam/crates/robine-nam/tests/fixtures/A2.nam}
ROUNDS=${4:-4}
BLOCK=${5:-64}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

run() { # tag bin
  "$2" bench -m "$MODEL" --blocks "$BLOCK" --seconds 2 --iters 3 --impl robine --brief \
    | awk -v tag="$1" '/small|large/ {print tag, $2, $6}' >> "$TMP/all"
}
i=0
while [ "$i" -lt "$ROUNDS" ]; do
  run A "$A"; run B "$B"; run B "$B"; run A "$A"
  i=$((i + 1))
done
for size in small large; do
  a=$(awk -v s="$size" '$1=="A" && $2==s {print $3}' "$TMP/all" | sort -n | head -1)
  b=$(awk -v s="$size" '$1=="B" && $2==s {print $3}' "$TMP/all" | sort -n | head -1)
  [ -n "$a" ] && [ -n "$b" ] && awk -v s="$size" -v a="$a" -v b="$b" 'BEGIN {printf "%-6s A %7.1f ns  B %7.1f ns  B/A %.3f\n", s, a, b, b / a}'
done
