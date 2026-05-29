// Common MoE hybrid swap manager — promotes/demotes experts at request boundaries.

#pragma once

#include "moe_hybrid_placement.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats;

struct MoeHybridSwapAction {
    int layer_idx = -1;
    int evict_expert = -1;
    int promote_expert = -1;
    uint64_t evict_count = 0;
    uint64_t promote_count = 0;
};

struct MoeHybridSwapPlan {
    MoeHybridPlacement next_placement;
    std::vector<MoeHybridSwapAction> actions;
};

struct MoeHybridSwapPolicy {
    int max_swaps_total = 0;          // 0 = no swaps
    uint64_t min_promote_gain = 1;    // promoted expert count must exceed evicted by this amount
};

bool build_moe_hybrid_swap_plan(const MoeHybridPlacement & current,
                                const MoeHybridRoutingStats & stats,
                                const MoeHybridSwapPolicy & policy,
                                MoeHybridSwapPlan & out,
                                std::string * err = nullptr);

}  // namespace dflash::common
