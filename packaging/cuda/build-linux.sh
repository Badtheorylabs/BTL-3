#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
target=${1:-}
output=${2:-}
case "$target" in
    linux-x86_64)
        platform=linux/amd64
        architectures='89-real;120-real'
        ;;
    linux-arm64)
        platform=linux/arm64
        architectures='121-real'
        ;;
    *)
        echo "usage: $0 {linux-x86_64|linux-arm64} OUTPUT" >&2
        exit 2
        ;;
esac
if [ -z "$output" ]; then
    echo "usage: $0 $target OUTPUT" >&2
    exit 2
fi
case "$output" in
    /*) ;;
    *) output="$PWD/$output" ;;
esac
if [ -e "$output" ]; then
    echo "output already exists: $output" >&2
    exit 2
fi
command -v docker >/dev/null 2>&1 || {
    echo "docker with buildx is required" >&2
    exit 2
}
docker buildx version >/dev/null 2>&1 || {
    echo "docker buildx plugin is required" >&2
    exit 2
}

stage=$(mktemp -d "${TMPDIR:-/tmp}/btl3-cuda.XXXXXX")
trap 'rm -rf "$stage"' EXIT HUP INT TERM
docker buildx build \
    --platform "$platform" \
    --build-arg "CUDA_ARCHITECTURES=$architectures" \
    --file "$root/packaging/cuda/Dockerfile" \
    --target export \
    --output "type=local,dest=$stage" \
    "$root/native/llama.cpp"
python3 "$root/tools/build_cuda_bundle.py" \
    --target "$target" --source "$stage" --output "$output"
