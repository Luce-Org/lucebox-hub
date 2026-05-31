// Remote target-shard execution configuration for mixed-backend layer split.

#pragma once

#include <string>

namespace dflash::common {

struct RemoteTargetShardConfig {
    std::string ipc_bin;
    std::string work_dir;

    bool enabled() const { return !ipc_bin.empty(); }
    bool has_aux_options() const { return !work_dir.empty(); }
};

}  // namespace dflash::common
