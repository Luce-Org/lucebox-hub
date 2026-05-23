#include "layer_split_utils.h"

#include "common/peer_access.h"
#include "common/snapshot_backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace dflash::common {

std::vector<LayerSplitRange> compute_layer_ranges(
    int n_layer,
    int n_gpus,
    const std::vector<double> & weights)
{
    std::vector<LayerSplitRange> ranges;
    if (n_layer <= 0 || n_gpus <= 0 || n_gpus > n_layer) return ranges;

    std::vector<double> w = weights;
    if (w.empty()) w.assign((size_t)n_gpus, 1.0);
    if ((int)w.size() != n_gpus) return ranges;

    double sum = 0.0;
    for (double v : w) sum += v;
    if (sum <= 0.0) return ranges;

    ranges.reserve((size_t)n_gpus);
    int begin = 0;
    double accum = 0.0;
    for (int i = 0; i < n_gpus; i++) {
        accum += w[i];
        int end = (i == n_gpus - 1)
            ? n_layer
            : (int)std::llround((accum / sum) * n_layer);
        const int min_end = begin + 1;
        const int max_end = n_layer - (n_gpus - i - 1);
        end = std::max(min_end, std::min(max_end, end));
        ranges.push_back({begin, end});
        begin = end;
    }
    return ranges;
}

bool init_layer_split_shard_metas(
        std::vector<LayerSplitShardMeta *> shards,
        const std::vector<int> & gpus,
        const std::vector<LayerSplitRange> & ranges,
        const char * log_prefix) {
    if (shards.size() != gpus.size() || shards.size() != ranges.size()) return false;
    const char * prefix = log_prefix ? log_prefix : "target-split";
    for (size_t i = 0; i < shards.size(); ++i) {
        auto * shard = shards[i];
        if (!shard) return false;
        shard->gpu = gpus[i];
        shard->layer_begin = ranges[i].begin;
        shard->layer_end = ranges[i].end;
        shard->backend = ggml_backend_cuda_init(shard->gpu);
        if (!shard->backend) {
            std::fprintf(stderr, "[%s] backend init failed gpu=%d\n",
                         prefix, shard->gpu);
            return false;
        }
    }
    return true;
}

bool enable_layer_split_peer_access(
        const std::vector<int> & gpus,
        bool peer_access) {
    if (!peer_access) return true;
    for (size_t i = 0; i < gpus.size(); ++i) {
        for (size_t j = i + 1; j < gpus.size(); ++j) {
            (void)enable_peer_access_pair(gpus[i], gpus[j]);
        }
    }
    return true;
}

bool init_layer_split_snapshot_backends(
        const std::vector<LayerSplitShardMeta *> & shards,
        std::vector<ggml_backend_t> & snapshot_backends,
        const char * log_prefix) {
    const char * prefix = log_prefix ? log_prefix : "target-split";
    snapshot_backends.assign(shards.size(), nullptr);
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto * shard = shards[i];
        if (!shard || !shard->backend) return false;
        snapshot_backends[i] = create_snapshot_backend(shard->backend);
        if (!snapshot_backends[i]) {
            std::fprintf(stderr,
                "[%s] snapshot backend init failed gpu=%d\n",
                prefix, shard->gpu);
            return false;
        }
    }
    return true;
}

void free_layer_split_snapshot_backends(
        const std::vector<LayerSplitShardMeta *> & shards,
        std::vector<ggml_backend_t> & snapshot_backends) {
    const size_t n = std::min(shards.size(), snapshot_backends.size());
    for (size_t i = 0; i < n; ++i) {
        if (!shards[i]) continue;
        free_snapshot_backend(snapshot_backends[i], shards[i]->backend);
    }
    snapshot_backends.clear();
}

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

    return {};  // ok
}

}  // namespace dflash::common
