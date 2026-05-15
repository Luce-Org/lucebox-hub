# docker-bake.hcl — Lucebox hub prebuild matrix.
#
# Two variants from one Dockerfile, split by CUDA toolkit / driver-branch
# pin rather than hardware generation. Each image carries every dflash-
# supported NVIDIA arch its toolkit can target, so users pick on the
# driver-pin axis (cluster IT policy, kernel module version) without
# having to also reason about hardware coverage:
#
#   docker buildx bake cuda12   # CUDA 12.9.1 (R535-series driver branch)
#                                 # Arches: sm_75;80;86;89;90;120;121
#                                 # Drops Thor sm_110 (needs nvcc 13+).
#
#   docker buildx bake cuda13   # CUDA 13.0.1 (current driver branch)
#                                 # Arches: sm_75;80;86;89;90;110;120;121
#                                 # Full dflash-supported matrix.
#
#   docker buildx bake          # builds both (group `default`)
#
# Both targets resolve every Turing-and-newer GPU dflash supports; the
# only image-level discriminator is Jetson AGX Thor (cuda13 only).
# Pre-Turing arches (Pascal sm_60/61, Volta sm_70) are intentionally
# excluded — dflash's kernels assume sm_75+ with no fallback below
# (dflash/CMakeLists.txt:276).
#
# Each target produces:
#   • lucebox-hub:cuda12  /  lucebox-hub:<tag>-cuda12
#   • lucebox-hub:cuda13  /  lucebox-hub:<tag>-cuda13
#
# Override the registry/tag via env: `TAG=v0.1 REGISTRY=ghcr.io/luce-org/ \
#   docker buildx bake cuda13`.

variable "REGISTRY" { default = "" }
variable "TAG"      { default = "" }

# Image name prefix. With the default empty REGISTRY/TAG you get
# `lucebox-hub:cuda12` / `lucebox-hub:cuda13`.
function "image_tag" {
    params = [variant]
    result = TAG == "" ? "${REGISTRY}lucebox-hub:${variant}" : "${REGISTRY}lucebox-hub:${TAG}-${variant}"
}

group "default" {
    targets = ["cuda12", "cuda13"]
}

# CI integration. docker/metadata-action in .github/workflows/docker.yml
# emits a bake-file that defines a `docker-metadata-action` target carrying
# tags + labels derived from the ref. Both build targets inherit from it.
# For local `docker buildx bake` invocations the metadata-action file is
# absent and this placeholder is a no-op (empty tags + labels do not
# override the explicit `tags = [image_tag(...)]` below).
target "docker-metadata-action" {}

# ── CUDA 12.9 · R535-series driver branch ──────────────────────────────────
# Full dflash-supported arch matrix except Thor sm_110 (Thor needs nvcc 13+).
# 12.9 is the latest 12.x point release; it adds sm_121 codegen (DGX Spark /
# GB10) on top of 12.8's sm_120 (consumer Blackwell) support, so a 12.x-
# pinned host gets current-gen consumer Blackwell coverage too.
target "cuda12" {
    inherits   = ["docker-metadata-action"]
    context    = "."
    dockerfile = "Dockerfile"
    args = {
        CUDA_VERSION        = "12.9.1"
        UBUNTU_VERSION      = "22.04"
        DFLASH_CUDA_ARCHES  = "75;80;86;89;90;120;121"
    }
    tags = [image_tag("cuda12")]
}

# ── CUDA 13.0 · current driver branch ──────────────────────────────────────
# Full dflash-supported arch matrix including Jetson AGX Thor (sm_110).
target "cuda13" {
    inherits   = ["docker-metadata-action"]
    context    = "."
    dockerfile = "Dockerfile"
    args = {
        CUDA_VERSION        = "13.0.1"
        UBUNTU_VERSION      = "24.04"
        DFLASH_CUDA_ARCHES  = "75;80;86;89;90;110;120;121"
    }
    tags = [image_tag("cuda13")]
}
