// Shared inputs/outputs for the DFlash draft graph builder.
#pragma once

#include <cstdint>
#include <vector>

#include "ggml.h"

namespace dflash27b {

struct DraftWeights; // fwd

struct DraftGraphInputs {
    int           ctx_len;          // length of target_hidden_cat along ne[1]
    ggml_tensor * noise_embed;      // [hidden, q_len=16, 1] f32
    ggml_tensor * target_hidden_cat;// [5*hidden, ctx_len, 1] f32
    ggml_tensor * positions_q;      // [q_len] i32   values [ctx_len..ctx_len+q_len-1]
    ggml_tensor * positions_k;      // [ctx_len+q_len] i32   values [0..ctx_len+q_len-1]
    // Optional SWA mask for long-context sliding-attention layers.
    // Shape [kv_len, q_len] or padded [kv_pad, q_pad], type F16, values
    // 0 for visible positions and -inf for masked positions.
    ggml_tensor * attn_mask = nullptr;
    // Optional: if non-null, the graph projects final hidden states through
    // this LM head (shape [hidden, vocab]) and returns logits instead of
    // hidden states. Used for DFlash integration where the draft shares the
    // target's lm_head.
    ggml_tensor * lm_head = nullptr;
};

struct DraftGraphOutputs {
    ggml_tensor * hidden_states;    // [hidden, q_len, 1]  (always set)
    ggml_tensor * logits;           // [vocab, q_len, 1]   (non-null iff lm_head was provided)
};

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in);

bool draft_graph_needs_swa_mask(const DraftWeights & w, int ctx_len);
void build_draft_swa_mask(std::vector<uint16_t> & out,
                          int ctx_len,
                          int q_len,
                          int swa_window);

} // namespace dflash27b
