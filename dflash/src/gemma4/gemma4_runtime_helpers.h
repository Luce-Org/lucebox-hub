// Helpers for Gemma4Backend chunked prefill and autoregressive decode.
//
// Ported from feature/gemma4-support dflash/test/gemma4/test_gemma4_dflash.cpp
// (ranges 213-275, 342-425, 525-612 per the AR-port source digest).

#pragma once

#include "../internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash27b {

// ─── KQ mask padding constants ───────────────────────────────────────────────
//
// ggml_flash_attn_ext requires kv_len aligned to g_kq_stride_pad (32 for Q8/F16
// KV, 256 for TQ3_0 KV). KQ_MASK_PAD aligns the q dimension. These values match
// the runtime that build_gemma4_graph expects — the mask tensor dimensions must
// match the tensors the graph allocates for attn_mask / swa_mask.
static constexpr int      KQ_MASK_PAD    = 32;
static constexpr int      g_kq_stride_pad = 32;
static constexpr uint16_t F16_NEG_INF    = 0xFC00u;
static constexpr uint16_t F16_ZERO       = 0x0000u;

static inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// ─── Per-step graph state ────────────────────────────────────────────────────
//
// Rebuilt each forward pass because kv_len varies chunk to chunk.
// Freed and reallocated via build_gemma4_step / step_graph_free.

struct StepGraph {
    ggml_context   * ctx        = nullptr;
    ggml_cgraph    * gf         = nullptr;
    ggml_gallocr_t   alloc      = nullptr;
    ggml_tensor    * inp_embed  = nullptr;
    ggml_tensor    * positions  = nullptr;
    ggml_tensor    * attn_mask  = nullptr;
    ggml_tensor    * swa_mask   = nullptr;
    ggml_tensor    * logits     = nullptr;
};

void step_graph_free(StepGraph & sg);

// ─── Attention mask builders ─────────────────────────────────────────────────

void build_causal_mask(std::vector<uint16_t> & out,
                       int kv_len, int n_tokens, int kv_start);

// Non-monotonic ring mask for SWA layers.
// ring_size = swa_view.effective_win_len = swa_ctx_alloc.
// kv_end    = kv_start + n_tokens.
void build_swa_causal_mask(std::vector<uint16_t> & out,
                            int kv_start,
                            int n_tokens,
                            int swa_window,
                            int ring_size,
                            int kv_end);

// ─── Graph builder ───────────────────────────────────────────────────────────

bool build_gemma4_step(StepGraph & sg,
                       const GemmaTargetWeights & w,
                       GemmaTargetCache & cache,
                       ggml_backend_t backend,
                       int kv_start,
                       int n_tokens,
                       bool with_mask,
                       bool capture,
                       bool use_pflash            = false,
                       float pflash_alpha         = 0.12f,
                       int fa_window              = 0,
                       bool last_token_logits_only = false);

// ─── Token embedding ─────────────────────────────────────────────────────────

bool embed_token(const GemmaTargetWeights & w,
                 int32_t tok,
                 ggml_tensor * inp_embed,
                 ggml_backend_t backend);

bool embed_tokens_batch(const GemmaTargetWeights & w,
                        const int32_t * ids,
                        int n,
                        ggml_tensor * inp_embed,
                        ggml_backend_t backend);

}  // namespace dflash27b
