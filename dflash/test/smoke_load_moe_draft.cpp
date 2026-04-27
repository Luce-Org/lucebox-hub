// Smoke test: load 35B-A3B DFlash draft model from safetensors.
// Validates that DraftHparams are read correctly from config.json,
// tensor shapes match, and all 8 layers are loaded.
//
// Usage: smoke_load_moe_draft <path/to/model.safetensors>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    DraftWeights w;
    if (!load_draft_safetensors(argv[1], backend, w)) {
        std::fprintf(stderr, "load_draft_safetensors: %s\n", dflash27b_last_error());
        return 1;
    }

    bool ok = true;
    auto & hp = w.hparams;

    std::printf("[hparams] n_layer=%d hidden=%d n_head=%d n_kv_head=%d head_dim=%d "
                "intermediate=%d n_target_layers=%d block_size=%d\n",
        hp.n_layer, hp.hidden, hp.n_head, hp.n_kv_head, hp.head_dim,
        hp.intermediate, hp.n_target_layers, hp.block_size);
    std::printf("[rope] theta=%.0f factor=%.1f beta_fast=%.1f beta_slow=%.1f orig_ctx=%d\n",
        hp.rope_theta, hp.rope_factor, hp.rope_beta_fast, hp.rope_beta_slow, hp.rope_orig_ctx);

    if (hp.n_layer != 8)       { std::fprintf(stderr, "FAIL: n_layer=%d expected 8\n", hp.n_layer); ok = false; }
    if (hp.hidden != 2048)     { std::fprintf(stderr, "FAIL: hidden=%d expected 2048\n", hp.hidden); ok = false; }
    if (hp.n_head != 32)       { std::fprintf(stderr, "FAIL: n_head=%d expected 32\n", hp.n_head); ok = false; }
    if (hp.n_kv_head != 4)     { std::fprintf(stderr, "FAIL: n_kv_head=%d expected 4\n", hp.n_kv_head); ok = false; }
    if (hp.head_dim != 128)    { std::fprintf(stderr, "FAIL: head_dim=%d expected 128\n", hp.head_dim); ok = false; }
    if (hp.intermediate != 6144){ std::fprintf(stderr, "FAIL: intermediate=%d expected 6144\n", hp.intermediate); ok = false; }
    if (hp.n_target_layers != 5){ std::fprintf(stderr, "FAIL: n_target_layers=%d expected 5\n", hp.n_target_layers); ok = false; }
    if (hp.block_size != 16)   { std::fprintf(stderr, "FAIL: block_size=%d expected 16\n", hp.block_size); ok = false; }
    if (hp.rope_factor != 64.0f){ std::fprintf(stderr, "FAIL: rope_factor=%.1f expected 64.0\n", hp.rope_factor); ok = false; }
    if (hp.rope_beta_fast != 32.0f){ std::fprintf(stderr, "FAIL: rope_beta_fast=%.1f expected 32.0\n", hp.rope_beta_fast); ok = false; }
    if (hp.rope_beta_slow != 1.0f) { std::fprintf(stderr, "FAIL: rope_beta_slow=%.1f expected 1.0\n", hp.rope_beta_slow); ok = false; }
    if (hp.rope_orig_ctx != 4096)  { std::fprintf(stderr, "FAIL: rope_orig_ctx=%d expected 4096\n", hp.rope_orig_ctx); ok = false; }

    if ((int)w.layers.size() != hp.n_layer) {
        std::fprintf(stderr, "FAIL: layers.size=%zu expected %d\n", w.layers.size(), hp.n_layer);
        ok = false;
    }

    std::printf("[fc] ne=[%lld,%lld]\n", (long long)w.fc->ne[0], (long long)w.fc->ne[1]);
    const int64_t fc_in = hp.n_target_layers * hp.hidden;
    if (w.fc->ne[1] != hp.hidden || w.fc->ne[0] != fc_in) {
        std::fprintf(stderr, "FAIL: fc shape [%lld,%lld] expected [%lld,%lld]\n",
            (long long)w.fc->ne[0], (long long)w.fc->ne[1], (long long)fc_in, (long long)hp.hidden);
        ok = false;
    }

    if (!w.layers.empty()) {
        auto & L0 = w.layers[0];
        const int64_t q_dim = hp.n_head * hp.head_dim;
        const int64_t kv_dim = hp.n_kv_head * hp.head_dim;
        std::printf("[layer0] wq=[%lld,%lld] wk=[%lld,%lld] w_gate=[%lld,%lld]\n",
            (long long)L0.wq->ne[0], (long long)L0.wq->ne[1],
            (long long)L0.wk->ne[0], (long long)L0.wk->ne[1],
            (long long)L0.w_gate->ne[0], (long long)L0.w_gate->ne[1]);
        if (L0.wq->ne[1] != q_dim || L0.wq->ne[0] != hp.hidden) {
            std::fprintf(stderr, "FAIL: wq shape mismatch\n"); ok = false;
        }
        if (L0.wk->ne[1] != kv_dim || L0.wk->ne[0] != hp.hidden) {
            std::fprintf(stderr, "FAIL: wk shape mismatch\n"); ok = false;
        }
        if (L0.w_gate->ne[1] != hp.intermediate || L0.w_gate->ne[0] != hp.hidden) {
            std::fprintf(stderr, "FAIL: w_gate shape mismatch\n"); ok = false;
        }
    }

    free_draft_weights(w);
    ggml_backend_free(backend);

    if (ok) {
        std::printf("OK\n");
        return 0;
    }
    return 1;
}
