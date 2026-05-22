# docker-bake.hcl — Lucebox hub prebuild matrix.
#
# Single CUDA 12 image from one Dockerfile. Additional CUDA stacks are
# intentionally omitted.
#
#   docker buildx bake cuda12-local   # CUDA 12.8.1, tagged lucebox-hub:cuda12
#   docker buildx bake cuda12         # CI target; tags come from metadata-action
#                                 # Arches: sm_75;80;86;89;90;120
#
# Pre-Turing arches (Pascal sm_60/61, Volta sm_70) are intentionally
# excluded — dflash's kernels assume sm_75+ with no fallback below
# (dflash/CMakeLists.txt:276).
#
# The CI `cuda12` target takes tags from docker/metadata-action. The local
# `cuda12-local` target keeps a simple lucebox-hub:cuda12 tag for manual builds.
#
# Override the registry/tag via env: `TAG=v0.1 REGISTRY=ghcr.io/luce-org/ \
#   docker buildx bake cuda12-local`.

variable "REGISTRY" { default = "" }
variable "TAG"      { default = "" }

# Image name prefix. With the default empty REGISTRY/TAG you get
# `lucebox-hub:cuda12`.
function "image_tag" {
    params = [variant]
    result = TAG == "" ? "${REGISTRY}lucebox-hub:${variant}" : "${REGISTRY}lucebox-hub:${TAG}-${variant}"
}

group "default" {
    targets = ["cuda12-local"]
}

# CI integration. docker/metadata-action in .github/workflows/docker.yml
# emits a bake-file that defines a `docker-metadata-action` target carrying
# tags + labels derived from the ref. Both build targets inherit from it.
# Local `docker buildx bake` invocations do not provide the metadata-action
# file, so this empty target keeps inheritance valid.
target "docker-metadata-action" {}

# ── CUDA 12.8 ───────────────────────────────────────────────────────────────
# CUDA 12.8 matches the uv-managed PyTorch cu128 stack and carries current-gen
# consumer Blackwell sm_120 coverage. Thor/GB10 variants stay out of this
# build matrix.
target "_cuda12-base" {
    context    = "."
    dockerfile = "Dockerfile"
    args = {
        CUDA_VERSION        = "12.8.1"
        UBUNTU_VERSION      = "22.04"
        DFLASH_CUDA_ARCHES  = "75;80;86;89;90;120"
    }
}

target "cuda12" {
    inherits = ["_cuda12-base", "docker-metadata-action"]
}

target "cuda12-local" {
    inherits = ["_cuda12-base"]
    tags = [image_tag("cuda12")]
}
