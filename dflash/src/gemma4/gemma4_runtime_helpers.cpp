// Gemma4 runtime helpers — chunked-prefill graph builder and token embedders.
//
// Ported from feature/gemma4-support dflash/test/gemma4/test_gemma4_dflash.cpp
// (ranges 213-275, 342-425, 525-612 per the AR-port source digest).
// Namespace adjusted to dflash27b; local sampler types and driver-only
// scaffolding stripped.

#include "gemma4_runtime_helpers.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace dflash27b {

// ─── step_graph_free ────────────────────────────────────────────────────────

void step_graph_free(StepGraph & sg) {
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    sg.gf        = nullptr;
    sg.inp_embed = nullptr;
    sg.positions = nullptr;
    sg.attn_mask = nullptr;
    sg.swa_mask  = nullptr;
    sg.logits    = nullptr;
}

// ─── build_causal_mask ──────────────────────────────────────────────────────

void build_causal_mask(std::vector<uint16_t> & out,
                       int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
}

// ─── build_swa_causal_mask ──────────────────────────────────────────────────
//
// Non-monotonic ring mask.  The K view is always the full ring (ring_size slots,
// ring_win_start==0).  Slot k_view maps to absolute position via:
//   latest_slot = (kv_end - 1) % ring_size
//   offset_back = (latest_slot - k_view + ring_size) % ring_size
//   abs_k       = (kv_end - 1) - offset_back
//
// mask[q_idx][k_view_idx] = 0 (attend) iff:
//   abs_k >= (abs_q - swa_window + 1) AND abs_k <= abs_q AND abs_k >= 0
// else -inf.

void build_swa_causal_mask(std::vector<uint16_t> & out,
                            int kv_start,
                            int n_tokens,
                            int swa_window,
                            int ring_size,
                            int kv_end) {
    const int kv_pad = align_up(ring_size, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int latest_slot = ((kv_end - 1) % ring_size + ring_size) % ring_size;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int q_lo  = std::max(0, abs_q - swa_window + 1);
        for (int k_view = 0; k_view < ring_size; k_view++) {
            const int offset_back = (latest_slot - k_view + ring_size) % ring_size;
            const int abs_k       = (kv_end - 1) - offset_back;
            const bool valid = (abs_k >= q_lo && abs_k <= abs_q && abs_k >= 0);
            if (valid) {
                out[(size_t)q * kv_pad + k_view] = F16_ZERO;
            }
        }
    }
}

// ─── build_gemma4_step ──────────────────────────────────────────────────────

bool build_gemma4_step(StepGraph & sg,
                       const GemmaTargetWeights & w,
                       GemmaTargetCache & cache,
                       ggml_backend_t backend,
                       int kv_start,
                       int n_tokens,
                       bool with_mask,
                       bool capture,
                       bool use_pflash,
                       float pflash_alpha,
                       int fa_window,
                       bool last_token_logits_only) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int kv_len = kv_start + n_tokens;
        const int kv_pad = align_up(kv_len, g_kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);

        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
        ggml_set_output(sg.attn_mask);  // force gallocr to allocate even if no op references it

        // SWA mask is required for every SWA dispatch — including single-token
        // decode (n_tokens==1). When swa_mask is null, gemma4_target_graph falls
        // back to attn_mask, which is sized for kv_len rather than the SWA window;
        // the resulting dimension mismatch lets FA read past the populated cache
        // region and corrupts attention. Catastrophic with TQ3_0 KV (it amplifies
        // uninitialized-cache noise into a fixed-point repetition loop), benign
        // but technically wrong with Q8_0 KV.
        const SwaView swa_view = compute_swa_view(kv_start, n_tokens,
                                                   w.swa_window, cache.swa_ctx_alloc);
        const int swa_kv_pad = align_up(swa_view.effective_win_len, g_kq_stride_pad);
        sg.swa_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, swa_kv_pad, q_pad);
        ggml_set_name(sg.swa_mask, "swa_mask");
        ggml_set_input(sg.swa_mask);
        ggml_set_output(sg.swa_mask);  // force gallocr to allocate even if no op references it
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    GemmaGraphInputs gi{};
    gi.inp_embed              = sg.inp_embed;
    gi.positions              = sg.positions;
    gi.attn_mask              = sg.attn_mask;
    gi.swa_mask               = sg.swa_mask;
    gi.n_tokens               = n_tokens;
    gi.kv_start               = kv_start;
    gi.capture_layers         = capture;
    gi.fa_window              = fa_window;
    gi.use_pflash             = use_pflash;
    gi.pflash_alpha           = pflash_alpha;
    gi.last_token_logits_only = last_token_logits_only;

    GemmaGraphOutputs go = build_gemma4_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── embed_token ────────────────────────────────────────────────────────────

bool embed_token(const GemmaTargetWeights & w,
                 int32_t tok,
                 ggml_tensor * inp_embed,
                 ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb((size_t)hidden);
    if (!w.embedder.embed(&tok, 1, emb.data())) {
        std::fprintf(stderr, "[embed_token] failed for tok=%d\n", tok);
        return false;
    }
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden);
    (void)backend;
    return true;
}

// ─── embed_tokens_batch ─────────────────────────────────────────────────────

bool embed_tokens_batch(const GemmaTargetWeights & w,
                        const int32_t * ids,
                        int n,
                        ggml_tensor * inp_embed,
                        ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb((size_t)hidden * n);
    if (!w.embedder.embed(ids, n, emb.data())) {
        std::fprintf(stderr, "[embed_batch] failed for %d tokens\n", n);
        return false;
    }
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden * n);
    (void)backend;
    return true;
}

}  // namespace dflash27b
