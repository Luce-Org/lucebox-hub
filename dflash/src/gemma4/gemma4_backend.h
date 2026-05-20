// Gemma4Backend — ModelBackend for Gemma4 iSWA+MoE models.
//
// Architecture: iSWA hybrid attention, MoE with shared+routed experts,
// per-layer embeddings, KV sharing, logit softcapping.

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "gemma4_internal.h"
#include "common/sampler.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <random>
#include <string>
#include <vector>

namespace dflash27b {

struct Gemma4BackendConfig {
    const char *    model_path = nullptr;
    DevicePlacement device;
    int             stream_fd  = -1;
    int             chunk      = 512;
};

class Gemma4Backend : public ModelBackend {
public:
    explicit Gemma4Backend(const Gemma4BackendConfig & cfg);
    ~Gemma4Backend() override;

    Gemma4Backend(const Gemma4Backend &) = delete;
    Gemma4Backend & operator=(const Gemma4Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate(const GenerateRequest & req,
                            const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate(int slot,
                                        const GenerateRequest & req,
                                        const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    bool try_handle_command(const std::string & line,
                            const DaemonIO & io) override;

    void shutdown() override;

private:
    Gemma4BackendConfig   cfg_;
    ggml_backend_t        backend_ = nullptr;
    ggml_backend_t        snap_backend_ = nullptr;
    Gemma4Weights         w_;
    Gemma4Cache           cache_;
    bool                  parked_ = false;

    // Sampler
    SamplerCfg            sampler_;
    std::mt19937_64       sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    Gemma4Snapshot        snapshots_[PREFIX_SLOTS];

    // Prefill prompt tokens in chunks, return committed position.
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io);

    // Autoregressive decode loop.
    bool do_decode(int committed, int n_gen,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io);
};

}  // namespace dflash27b
