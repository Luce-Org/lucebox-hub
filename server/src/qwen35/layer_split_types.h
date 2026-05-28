// layer_split_types.h — qwen35 layer-split shard types.
//
// Shared layer-split metadata lives in common/layer_split_utils.h. This header
// keeps only the qwen35-specific shard payload: TargetWeights, TargetCache, and
// the qwen35 step graph.

#pragma once

#include "internal.h"
#include "step_graph.h"
#include "common/layer_split_utils.h"
#include "dflash_layer_split_runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>

namespace dflash::common {

// ── Per-GPU qwen35 shard for layer-split target ─────────────────────

struct Qwen35LayerSplitShard : LayerSplitShardMeta {
    TargetWeights weights;
    TargetCache cache;
    StepGraph layer_graph;
};

} // namespace dflash::common
