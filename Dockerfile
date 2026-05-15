# syntax=docker/dockerfile:1.7

# ─── Stage 1: builder ───────────────────────────────────────────────────────
# Two-variant build. CUDA_VERSION / UBUNTU_VERSION / DFLASH_CUDA_ARCHES are
# build args so a single Dockerfile produces:
#   • lucebox-hub:cuda12  — CUDA 12.9.1, sm_75;80;86;89;90;120;121
#   • lucebox-hub:cuda13  — CUDA 13.0.1, sm_75;80;86;89;90;110;120;121
# Each image carries every dflash-supported NVIDIA arch its toolkit can
# target. The only image-level discriminator is Jetson AGX Thor sm_110,
# which requires nvcc 13+. Pick based on which CUDA driver branch your
# host is pinned to, not on which GPU you have.
# See docker-bake.hcl for the canonical invocation.
ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Fat-binary CUDA arch list, semicolon-separated. Defaults cover the cuda13
# variant; the bake file overrides for cuda12 to drop Thor. dflash-supported
# arches:
#   75  Turing      RTX 2080 Ti
#   80  Ampere      A100
#   86  Ampere      RTX 3090, A40, A10
#   89  Ada         RTX 4090, L40
#   90  Hopper      H100
#   110 Thor        Jetson AGX Thor (CUDA 13+ only)
#   120 Blackwell   RTX 5090, RTX 5090 Laptop
#   121 GB10        DGX Spark
# Pre-Turing arches (sm_60/61/70/72) are intentionally excluded — dflash's
# BF16/WMMA paths have no fallback below sm_75. Each arch adds ~50-200 MB
# of fat-binary kernel code and ~3-5 min of nvcc time per .cu translation
# unit.
ARG DFLASH_CUDA_ARCHES="75;80;86;89;90;110;120;121"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        ninja-build \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# CUDA driver stub. nvidia/cuda:*-devel images ship the driver stub at
# /usr/local/cuda/lib64/stubs/libcuda.so but not as libcuda.so.1. ld follows
# the NEEDED reference inside libggml-cuda.so by SONAME (libcuda.so.1) when
# linking executables, so without this symlink + ld.so.conf entry the
# test_dflash link step fails with `undefined reference to cuMem*`.
# At runtime the host driver provides the real libcuda.so.1 via
# --gpus all; the stub is only for build-time symbol resolution.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

WORKDIR /src
COPY . /src

# Submodules (`dflash/deps/llama.cpp`, `dflash/deps/Block-Sparse-Attention`)
# must be populated on the host before `docker build` — `.git/` is excluded
# by .dockerignore so we cannot re-fetch them inside the image. ggml's own
# CMakeLists also asserts this and errors with the right command if missing,
# but failing here gives a clearer message before nvcc spins up.
RUN test -f /src/dflash/deps/llama.cpp/ggml/CMakeLists.txt \
    || (echo "ERROR: dflash/deps/llama.cpp submodule not initialised. Run on host:" >&2 \
        && echo "       git submodule update --init --recursive" >&2 \
        && exit 1)

# Configure + build. `DFLASH27B_USER_CUDA_ARCHITECTURES` pins the arch list
# through dflash's own logic (skips its auto-extend rules that depend on
# nvcc version inspection); `CMAKE_CUDA_ARCHITECTURES` also gets set so the
# vendored ggml-cuda subproject picks up the same list.
# CMAKE_BUILD_WITH_INSTALL_RPATH=ON embeds CMakeLists.txt's $ORIGIN-relative
# CMAKE_INSTALL_RPATH (`$ORIGIN/deps/llama.cpp/ggml/src`, etc.) into the
# binary at link time, instead of the default absolute build-tree paths.
# Without this the binary loses its ggml shared libs after COPY to the
# runtime stage (`libggml.so.0: cannot open shared object file`).
RUN cmake -S /src/dflash -B /src/dflash/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DDFLASH27B_USER_CUDA_ARCHITECTURES="${DFLASH_CUDA_ARCHES}" \
        -DCMAKE_CUDA_ARCHITECTURES="${DFLASH_CUDA_ARCHES}" \
    && cmake --build /src/dflash/build --target test_dflash --parallel

# Prune the build tree to only what the runtime stage needs: the test_dflash
# binary and the ggml shared libs its embedded rpath ($ORIGIN/deps/...) looks
# up. Drops ~1 GB per image of CMakeFiles/, libdflash27b.a (statically linked
# into the binary), ninja state, compile_commands.json, and the
# template-instance .o tree from ggml-cuda.
RUN cd /src/dflash/build \
    && find . -mindepth 1 -maxdepth 1 \
            ! -name test_dflash ! -name deps -exec rm -rf {} + \
    && find deps -mindepth 1 -type f ! -name 'lib*.so*' -delete \
    && find deps -depth -type d -empty -delete

# ─── Stage 2: runtime ───────────────────────────────────────────────────────
# Runtime image: ships nvidia driver libs but no nvcc / dev headers. Matches
# the builder's CUDA version so the test_dflash binary's libcudart SONAME
# resolves at runtime against the same major.minor.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        pciutils \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python 3.11 (required by the workspace) and resolves the
# lucebox-dflash + pflash members declared in pyproject.toml.
RUN curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh

WORKDIR /opt/lucebox-hub

# Workspace files for uv sync (root pyproject + lock + README + workspace
# member manifests). Each is a leaf file or small dir so layers stay tiny.
COPY --from=builder /src/start_server.sh /opt/lucebox-hub/start_server.sh
COPY --from=builder /src/pyproject.toml /src/uv.lock /src/README.md /opt/lucebox-hub/
COPY --from=builder /src/pflash /opt/lucebox-hub/pflash
COPY --from=builder /src/megakernel/pyproject.toml /src/megakernel/README.md \
                   /opt/lucebox-hub/megakernel/

# dflash: ship the Python orchestration (scripts/), the pyproject + README
# that uv resolves against, and the pruned build tree (binary + .so files
# from the prune step in the builder stage). Source code, headers, tests,
# and submodule sources stay in the builder.
COPY --from=builder /src/dflash/scripts /opt/lucebox-hub/dflash/scripts
COPY --from=builder /src/dflash/pyproject.toml /src/dflash/README.md \
                   /opt/lucebox-hub/dflash/
COPY --from=builder /src/dflash/build /opt/lucebox-hub/dflash/build

RUN test -x /opt/lucebox-hub/dflash/build/test_dflash \
    && chmod +x /opt/lucebox-hub/start_server.sh

# Register the ggml lib dir with ld.so so libggml-cpu.so (loaded transitively
# by libggml.so) resolves. CMakeLists.txt sets a `$ORIGIN/deps/...` RUNPATH
# uniformly across all linked artefacts — correct for test_dflash in
# dflash/build/, broken for the .so files in deps/llama.cpp/ggml/src/ which
# would need a plain `$ORIGIN`. ld.so.conf side-steps the RPATH bug without
# patching every shared lib.
RUN printf '%s\n%s\n' \
        /opt/lucebox-hub/dflash/build/deps/llama.cpp/ggml/src \
        /opt/lucebox-hub/dflash/build/deps/llama.cpp/ggml/src/ggml-cuda \
        > /etc/ld.so.conf.d/lucebox-ggml.conf \
    && ldconfig

# Resolve Python deps for the dflash server (server.py + bench harness).
# Megakernel is an optional extra and is intentionally skipped — its CUDA
# extension would require nvcc + matching torch headers in this stage.
# `--no-cache` keeps wheels from being persisted in the layer; hardlink mode
# means the venv files live alongside the cache during the install but the
# cache is gone by the time the layer commits, so we don't double-pay.
ENV UV_LINK_MODE=hardlink \
    UV_NO_CACHE=1
RUN uv sync --no-dev --frozen 2>/dev/null \
    || uv sync --no-dev

# Models live in dflash/models/ — bind-mount or volume them in.
# Example:
#   docker run --rm --gpus all -p 8080:8080 \
#       -v "$PWD/dflash/models:/opt/lucebox-hub/dflash/models" \
#       lucebox-hub
# The VOLUME declaration keeps the path out of the image layer cache; the
# bind mount above replaces it with the host directory at run time.
VOLUME ["/opt/lucebox-hub/dflash/models"]

ENV DFLASH_HOST=0.0.0.0 \
    DFLASH_PORT=8080 \
    DFLASH_BIN=/opt/lucebox-hub/dflash/build/test_dflash

EXPOSE 8080

ENTRYPOINT ["/opt/lucebox-hub/start_server.sh"]
