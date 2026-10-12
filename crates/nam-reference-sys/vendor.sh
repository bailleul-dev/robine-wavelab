#!/bin/sh
# Fetch the pinned upstream NeuralAmpModelerCore (with its Eigen and
# nlohmann/json submodules) into vendor/. Idempotent: a vendor/ already at
# the pinned commit is left alone, anything else is replaced.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/UPSTREAM"
DEST="$HERE/vendor/NeuralAmpModelerCore"

if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null)" = "$UPSTREAM_COMMIT" ] \
   && [ -f "$DEST/Dependencies/eigen/Eigen/Dense" ]; then
  echo "vendor: NeuralAmpModelerCore already at $UPSTREAM_COMMIT"
  exit 0
fi

rm -rf "$DEST"
mkdir -p "$HERE/vendor"
echo "vendor: fetching NeuralAmpModelerCore @ $UPSTREAM_COMMIT ($UPSTREAM_DATE)"
git init -q "$DEST"
git -C "$DEST" remote add origin "$UPSTREAM_URL"
git -C "$DEST" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
git -C "$DEST" submodule update -q --init --recursive --depth 1
echo "vendor: done ($(git -C "$DEST" rev-parse --short HEAD))"
