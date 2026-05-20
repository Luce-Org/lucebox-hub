// Device placement configuration for model backends.
//
// Describes which GPU(s) to use for a model. Supports:
//   - Single-GPU: just gpu field
//   - Multi-GPU layer-split: layer_split_gpus + optional weights
//   - Peer access between GPUs

#pragma once

#include "placement_backend.h"

#include <vector>

namespace dflash27b {

struct DevicePlacement {
    PlacementBackend backend = PlacementBackend::Auto;
    int gpu = 0;                              // primary GPU (single-GPU mode)

    // Multi-GPU layer-split. Empty = single GPU mode.
    std::vector<int>    layer_split_gpus;     // GPU IDs for each shard
    std::vector<double> layer_split_weights;  // proportional layer distribution (optional)

    bool peer_access = false;                 // enable CUDA/HIP peer access between GPUs
    int  max_ctx     = 8192;                  // max KV cache context length

    bool is_layer_split() const { return layer_split_gpus.size() > 1; }

    int primary_gpu() const {
        return layer_split_gpus.empty() ? gpu : layer_split_gpus[0];
    }
};

}  // namespace dflash27b
