#!/bin/sh
# Build the pinned Aravis release and the scene camera into a prefix.
#
#   scripts/genicam/setup_aravis.sh apt      # system packages (Ubuntu 24.04)
#   scripts/genicam/setup_aravis.sh build    # Aravis and arv-scene-gv-camera
#   scripts/genicam/setup_aravis.sh env      # print the environment to export
#
# Aravis is the open-source GenICam and GigE Vision library that machine-vision
# software on Linux is commonly built on.  It is built from the release tarball
# rather than taken from apt for two reasons: the version is then pinned by a
# checksum and not by whatever the distribution ships this month; Ubuntu's
# aravis-tools-cli package does not include `arv-fake-gv-camera`, the GigE
# Vision camera simulator the CI check runs against.
#
# CI caches the prefix on the hash of this file and of the scene camera's
# source, so the build only runs when one of them changes.

set -eu

ARAVIS_VERSION=0.8.36
ARAVIS_SHA256=246deaa0042a387ff1bc00332d0fb80537ce14abde2c28d1a54c91f17adc51bf
ARAVIS_URL="https://github.com/AravisProject/aravis/releases/download/${ARAVIS_VERSION}/aravis-${ARAVIS_VERSION}.tar.xz"

PREFIX=${ARAVIS_PREFIX:-/opt/aravis}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SCENE_SRC="$ROOT/capture/fakecam/arv-scene-gv-camera.c"

# Build tools, the libraries Aravis links against, GObject introspection (so
# Python can load Aravis through PyGObject) and the headers PyGObject 3.48 and
# pycairo compile against.
APT_PACKAGES="build-essential ca-certificates curl gobject-introspection \
iproute2 libcairo2-dev libgirepository1.0-dev libglib2.0-dev libxml2-dev \
meson ninja-build pkg-config xz-utils zlib1g-dev"

as_root() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

do_apt() {
    as_root apt-get update -qq
    # shellcheck disable=SC2086
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        --no-install-recommends $APT_PACKAGES
}

# Root only when the prefix cannot be written as the current user: CI builds
# into the home directory so the result can be cached without sudo.
as_installer() {
    parent=$(dirname "$PREFIX")
    if { [ -d "$PREFIX" ] && [ -w "$PREFIX" ]; } || { [ ! -e "$PREFIX" ] && [ -w "$parent" ]; }; then
        "$@"
    else
        as_root "$@"
    fi
}

libdir() {
    # Meson is told --libdir=lib, so this does not depend on the multiarch
    # triplet of the machine that built it.
    echo "$PREFIX/lib"
}

do_build() {
    if [ -x "$PREFIX/bin/arv-fake-gv-camera-0.8" ] && [ -x "$PREFIX/bin/arv-scene-gv-camera" ]; then
        echo "aravis ${ARAVIS_VERSION} already built in $PREFIX"
        return 0
    fi
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    curl -fsSL --retry 3 -o "$work/aravis.tar.xz" "$ARAVIS_URL"
    echo "${ARAVIS_SHA256}  $work/aravis.tar.xz" | sha256sum -c -
    tar -xJf "$work/aravis.tar.xz" -C "$work"
    src="$work/aravis-${ARAVIS_VERSION}"

    # Only the library, its command-line tools (which include the fake GigE
    # Vision camera) and the introspection data.  No viewer, no GStreamer
    # plugin, no USB3 Vision: nothing here needs them.
    meson setup "$src/build" "$src" \
        --prefix="$PREFIX" --libdir=lib --buildtype=release \
        -Dviewer=disabled -Dgst-plugin=disabled -Dusb=disabled \
        -Dpacket-socket=disabled -Dtests=false -Ddocumentation=disabled \
        -Dintrospection=enabled
    ninja -C "$src/build"
    as_installer ninja -C "$src/build" install

    # The scene camera needs one private header from this exact release (see
    # the comment at the top of its source), so it is compiled here, against
    # the tree that was just built and never against any other version.
    PKG_CONFIG_PATH="$(libdir)/pkgconfig" cc -O2 -Wall -Wextra -Werror \
        -I"$src/src" -I"$src/build/src" \
        "$SCENE_SRC" -o "$work/arv-scene-gv-camera" \
        $(PKG_CONFIG_PATH="$(libdir)/pkgconfig" pkg-config --cflags --libs aravis-0.8) \
        -Wl,-rpath,"$(libdir)"
    as_installer install -m 0755 "$work/arv-scene-gv-camera" "$PREFIX/bin/arv-scene-gv-camera"
    echo "aravis ${ARAVIS_VERSION} installed in $PREFIX"
}

do_env() {
    # KEY=value lines for $GITHUB_ENV or `export`.  $PREFIX/bin goes on PATH
    # separately (CI writes it to $GITHUB_PATH).
    echo "ARAVIS_PREFIX=$PREFIX"
    echo "LD_LIBRARY_PATH=$(libdir)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "GI_TYPELIB_PATH=$(libdir)/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
}

case "${1:-}" in
    apt) do_apt ;;
    build) do_build ;;
    env) do_env ;;
    *) echo "usage: $0 apt|build|env" >&2; exit 2 ;;
esac
