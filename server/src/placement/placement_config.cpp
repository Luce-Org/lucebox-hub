#include "placement_config.h"

#include <cmath>
#include <set>

namespace dflash::common {

std::string validate_device_placement(
    const DevicePlacement & dp,
    int device_count)
{
    const bool validate_device_count = device_count >= 0;
    if (validate_device_count && device_count == 0) {
        return "no GPU devices available";
    }

    if (dp.gpu < 0 ||
        (validate_device_count && dp.gpu >= device_count)) {
        return "primary gpu " + std::to_string(dp.gpu) + " out of range" +
               (validate_device_count
                    ? " [0, " + std::to_string(device_count) + ")"
                    : "");
    }

    if (!dp.layer_split_gpus.empty()) {
        if (dp.layer_split_gpus.size() < 2) {
            return "layer_split_gpus must have at least 2 entries";
        }

        std::set<int> seen;
        for (int g : dp.layer_split_gpus) {
            if (g < 0 ||
                (validate_device_count && g >= device_count)) {
                return "layer_split gpu " + std::to_string(g) +
                       " out of range" +
                       (validate_device_count
                            ? " [0, " + std::to_string(device_count) + ")"
                            : "");
            }
            if (!seen.insert(g).second) {
                return "duplicate gpu " + std::to_string(g) + " in layer_split_gpus";
            }
        }

        if (!dp.layer_split_weights.empty() &&
            dp.layer_split_weights.size() != dp.layer_split_gpus.size()) {
            return "layer_split_weights size (" +
                   std::to_string(dp.layer_split_weights.size()) +
                   ") != gpu count (" +
                   std::to_string(dp.layer_split_gpus.size()) + ")";
        }

        for (double w : dp.layer_split_weights) {
            if (w <= 0.0 || !std::isfinite(w)) {
                return "layer_split_weights must be positive finite values";
            }
        }
    }

    if (dp.max_ctx <= 0) return "max_ctx must be positive";

    return {};
}

}  // namespace dflash::common
