#!/bin/sh
# Build GstPEAQ (https://github.com/HSU-ANT/gstpeaq, LGPL), an open-source
# implementation of PEAQ (ITU-R BS.1387), for `wavelab null --peaq`.
#
# Needs GStreamer and pkg-config:
#   macOS:  brew install gstreamer pkgconf
#   Debian: apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good
#
# The sources are fetched at a pinned commit into vendor/ and compiled into
# build/, both next to this script and both git-ignored. Nothing is installed
# outside this directory.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
commit=7397e49f5ede2f2429fb78f6663784e548401fec
packages="gstreamer-1.0 gstreamer-base-1.0 gstreamer-fft-1.0"
nolfs="-c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false"

if ! pkg-config --exists $packages; then
    echo "GStreamer development files not found (see the top of $0)" >&2
    exit 1
fi
if [ ! -d "$here/vendor/gstpeaq/.git" ]; then
    mkdir -p "$here/vendor"
    # The repository keeps its documentation in Git LFS; the sources are
    # plain files, so the LFS filter is switched off rather than required.
    git $nolfs clone -q https://github.com/HSU-ANT/gstpeaq.git "$here/vendor/gstpeaq"
fi
git $nolfs -C "$here/vendor/gstpeaq" checkout -q "$commit"

src="$here/vendor/gstpeaq/src"
out="$here/build"
mkdir -p "$out"
# What its configure script would have written.
cat > "$out/config.h" <<'CONFIG'
#define PACKAGE "gstpeaq"
#define PACKAGE_NAME "GstPEAQ"
#define PACKAGE_VERSION "0.6.1"
#define PACKAGE_STRING "GstPEAQ 0.6.1"
#define PACKAGE_BUGREPORT "martin.holters@hsu-hh.de"
#define PACKAGE_URL "http://ant.hsu-hh.de/gstpeaq"
#define VERSION "0.6.1"
#define GST_API_VERSION "1.0"
CONFIG
case "$(uname)" in
    Darwin) lib=libgstpeaq.dylib; shared="-dynamiclib" ;;
    *) lib=libgstpeaq.so; shared="-shared -fPIC" ;;
esac
cflags="-O2 -w -DHAVE_CONFIG_H -I$out -I$src $(pkg-config --cflags $packages)"
libs="$(pkg-config --libs $packages) -lm"
# shellcheck disable=SC2086
cc $cflags $shared -o "$out/$lib" \
    "$src/gstpeaq.c" "$src/gstpeaqplugin.c" "$src/earmodel.c" "$src/leveladapter.c" \
    "$src/modpatt.c" "$src/fftearmodel.c" "$src/fbearmodel.c" "$src/movaccum.c" \
    "$src/movs.c" "$src/nn.c" $libs
# shellcheck disable=SC2086
cc $cflags -o "$out/peaq" "$src/peaq.c" $libs
echo "built $out/peaq and $out/$lib"
