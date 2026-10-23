#!/bin/sh
# Linux release builds inside the manylinux_2_28 image (glibc 2.28), so the
# CLI archive and the wheel run on every glibc Linux since 2018 (RHEL 8,
# Debian 10, Ubuntu 18.10). release.yml runs it as
#
#   docker run --rm -v "$PWD:/io" -w /io IMAGE sh scripts/release_container.sh cli|wheel TARGET OUTPUT
#
# It installs the pinned Rust (and, for a wheel, maturin) inside the image and
# runs release_build.py; a wheel is also installed and run there with the
# oldest Python its abi3 tag allows (3.10). HOST_UID/HOST_GID, when set, own
# the outputs afterwards.
set -eu

kind=$1
target=$2
output=$3
toolchain=${RUSTUP_TOOLCHAIN:-1.99.1}
rustup_version=1.29.2
case "$(uname -m)" in
    x86_64) rustup_sha256=5be191e8c70a5007074254cb74d78dbfdd25cd5a61c27de7b8e5410122ec24f4 ;;
    aarch64) rustup_sha256=a9c168dbea81a84671c6b7a5c1795cb0eb699020110d107ddfca427bd797e697 ;;
    *) echo "release_container.sh: unsupported machine $(uname -m)" >&2; exit 1 ;;
esac
case "$target" in
    "$(uname -m)-unknown-linux-gnu") ;;
    *) echo "release_container.sh: $target is not this image's target" >&2; exit 1 ;;
esac

export HOME=/tmp/release-home CARGO_HOME=/tmp/release-home/cargo RUSTUP_HOME=/tmp/release-home/rustup
mkdir -p "$HOME"
# The checkout belongs to the runner's user; the build reads its commit.
git config --global --add safe.directory "$PWD"
curl --proto '=https' --tlsv1.2 -fsS --retry 5 -o "$HOME/rustup-init" \
    "https://static.rust-lang.org/rustup/archive/$rustup_version/$target/rustup-init"
echo "$rustup_sha256  $HOME/rustup-init" | sha256sum -c -
chmod +x "$HOME/rustup-init"
"$HOME/rustup-init" -y --no-modify-path --profile minimal --default-toolchain "$toolchain"
export PATH="$CARGO_HOME/bin:$PATH"
rustc -vV

python=/opt/python/cp313-cp313/bin/python
"$python" -m venv /tmp/release-venv
case "$kind" in
    cli)
        /tmp/release-venv/bin/python scripts/release_build.py cli --target "$target" --output "$output"
        ;;
    wheel)
        /tmp/release-venv/bin/python -m pip install --disable-pip-version-check "maturin==1.15.0"
        PATH="/tmp/release-venv/bin:$PATH" /tmp/release-venv/bin/python scripts/release_build.py wheel \
            --target "$target" --output "$output"
        for wheel in "$output"/*.whl; do
            /tmp/release-venv/bin/python scripts/release_build.py smoke-wheel --no-tests \
                --python /opt/python/cp310-cp310/bin/python "$wheel"
        done
        ;;
    *) echo "release_container.sh: cli or wheel, not $kind" >&2; exit 1 ;;
esac
if [ -n "${HOST_UID:-}" ]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$output" target
fi
