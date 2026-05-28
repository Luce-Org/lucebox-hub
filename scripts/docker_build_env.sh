#!/usr/bin/env bash
# Build and use the mounted-worktree CUDA build environment.
#
# This exercises the same CUDA/toolchain layer as the production Dockerfile
# builder, but bind-mounts the checkout at /workspace instead of COPYing source
# into the image. That keeps repeated integration builds fast while preserving
# the production CMake build shape.
#
# Usage:
#   scripts/docker_build_env.sh
#   DFLASH_CUDA_ARCHES=120 scripts/docker_build_env.sh
#   DFLASH27B_FA_ALL_QUANTS=ON scripts/docker_build_env.sh  # production-like slow path
#   TARGETS="test_dflash dflash_server test_server_unit" scripts/docker_build_env.sh
#   scripts/docker_build_env.sh --configure-only
#   scripts/docker_build_env.sh --shell
#   scripts/docker_build_env.sh -- cmake --version

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/.." && pwd))"

IMAGE="${IMAGE:-lucebox-hub:build-env}"
BUILD_DIR="${BUILD_DIR:-.docker-build/server}"
# Local integration builds default to one architecture to avoid the 5-6x CUDA
# template-instantiation cost. Production images keep the full matrix via
# docker-bake.hcl / scripts/build_image.sh.
DFLASH_CUDA_ARCHES="${DFLASH_CUDA_ARCHES:-86}"
# Fast local default for repeated integration builds. Set ON to match the
# production Dockerfile's full asymmetric-KV kernel matrix.
DFLASH27B_FA_ALL_QUANTS="${DFLASH27B_FA_ALL_QUANTS:-OFF}"
DFLASH27B_ENABLE_BSA="${DFLASH27B_ENABLE_BSA:-ON}"
EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}"
TARGETS="${TARGETS:-test_dflash dflash_server}"
DO_BUILD=1

if [ "${1:-}" = "--configure-only" ]; then
    DO_BUILD=0
    shift
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '1,28p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# Keep image tagging/version logic aligned with scripts/build_image.sh.
raw=$(git describe --tags --match 'lucebox-v*' --always --dirty 2>/dev/null || true)
VERSION="${raw#lucebox-v}"
export VERSION
export REGISTRY="${REGISTRY:-}"
export DFLASH_CUDA_ARCHES

echo "[docker-build-env] building mounted-worktree image: ${IMAGE}"
docker buildx bake build-env-local --load >/dev/null

mkdir -p "$BUILD_DIR"

run_args=(
    --rm
    --workdir /workspace
    --user "$(id -u):$(id -g)"
    --volume "$PWD:/workspace"
    --volume "lucebox-uv-cache:/tmp/uv-cache"
    --volume "lucebox-cmake-downloads:/tmp/cmake-downloads"
    --env HOME=/tmp
    --env UV_CACHE_DIR=/tmp/uv-cache
    --env CMAKE_DOWNLOAD_DIRECTORY=/tmp/cmake-downloads
    --env DFLASH_CUDA_ARCHES="$DFLASH_CUDA_ARCHES"
    --env DFLASH27B_FA_ALL_QUANTS="$DFLASH27B_FA_ALL_QUANTS"
    --env DFLASH27B_ENABLE_BSA="$DFLASH27B_ENABLE_BSA"
    --env EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS"
    --env DO_BUILD="$DO_BUILD"
)

if [ -t 0 ]; then
    run_args+=(--interactive --tty)
fi

if [ "${1:-}" = "--shell" ]; then
    shift
    exec docker run "${run_args[@]}" "$IMAGE" bash "$@"
fi

if [ "${1:-}" = "--" ]; then
    shift
    exec docker run "${run_args[@]}" "$IMAGE" "$@"
fi

# Same core CMake configure/build steps as the production Dockerfile builder;
# only source/build locations differ because /workspace is bind-mounted.
exec docker run "${run_args[@]}" "$IMAGE" bash -lc "
set -euo pipefail
cmake -S /workspace/server -B /workspace/${BUILD_DIR} \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DDFLASH27B_USER_CUDA_ARCHITECTURES=\"\${DFLASH_CUDA_ARCHES}\" \
    -DCMAKE_CUDA_ARCHITECTURES=\"\${DFLASH_CUDA_ARCHES}\" \
    -DDFLASH27B_FA_ALL_QUANTS=\"\${DFLASH27B_FA_ALL_QUANTS}\" \
    -DDFLASH27B_ENABLE_BSA=\"\${DFLASH27B_ENABLE_BSA}\" \
    \${EXTRA_CMAKE_ARGS}
if [ "\${DO_BUILD}" = 1 ]; then
    cmake --build /workspace/${BUILD_DIR} --target ${TARGETS} --parallel
fi
" 
