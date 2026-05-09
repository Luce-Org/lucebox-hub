// Single-step MTP (Multi-Token Prediction) graph builder for Gemma4.
//
// Builds a ggml compute graph that, given one token id and the target's last
// full-attention hidden state h_prev, produces:
//   - out_logits  : F32 [n_vocab, 1]  full vocabulary row
//   - out_h_post  : F32 [n_embd_backbone, 1]  next h_prev for the γ chain
//   - out_argmax  : I32 [1]  greedy draft token (4-byte host pull per step)
//
// Architecture (mirrors atomicbot's gemma4-assistant.cpp lines 28-256):
//   1. Token embedding from target.tok_embd, scaled by sqrt(n_embd_backbone).
//   2. Concat [tok_emb, h_prev] → pre_projection → [n_embd, 1].
//   3. 4 transformer blocks (cross-attention into target KV):
//      RMSNorm → Q proj → Q-norm → RoPE → cross-attn (reads donor K/V) →
//      wo → post_attn_norm → residual → ffn_norm → GELU FFN → post_ffn_norm →
//      residual → optional out_scale.
//   4. output_norm → post_projection → h_post  [n_embd_backbone, 1].
//   5. LM head: dense (tied tok_embd) or centroid-routed for ordered embeddings.
//   6. In-graph argmax.
//
// Cross-attention contract:
//   - Each MTP layer reads K/V from w.layers[il].donor_target_layer in the
//     target KV cache (resolved at load time as the LAST target layer whose
//     SWA type matches this MTP layer).
//   - V is ALWAYS read from the cache (use_k_as_v=false): per HF Gemma4 the
//     V slot stores rms-normed non-rotated vectors, distinct from post-RoPE K.
//   - The K/V view covers [0, attn_pos) = all committed target positions.
//     attn_pos is passed in via the in_pos tensor (caller sets it to
//     cache.cur_pos before each step).
//   - KV mask is not needed: all committed positions ≤ attn_pos are uniformly
//     admitted (step position > attn_pos, so every cell is in the causal cone).
//     We pass nullptr to ggml_flash_attn_ext for the mask argument.
//
// Centroid LM head (use_ordered_embeddings=true, always active for Dense 31B):
//   cent_logits = mul_mat(mtp_centroids, h_inner)
//   top_k_ids   = ggml_top_k(cent_logits, centroid_top_k)
//   sel_ids     = get_rows(token_ordering_view, top_k_ids)
//   sel_logits  = mul_mat(get_rows(tok_embd, flat_sel_ids), h_inner)
//   full_row    = scatter sel_logits into [-1e30 fill] via ggml_set_rows
//
// When use_ordered_embeddings is false (fallback, unlikely for 31B assistant):
//   out_logits = mul_mat(tok_embd, h_inner)  — dense tied head.

#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace dflash27b {

static constexpr float MTP_RMS_EPS = GEMMA4_RMS_EPS;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static ggml_tensor * mtp_rms_norm_mul(ggml_context * ctx,
                                       ggml_tensor  * x,
                                       ggml_tensor  * weight) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, MTP_RMS_EPS);
    return ggml_mul(ctx, n, weight);
}

// GELU FFN with SwiGLU-like gate: w_down @ (gelu(w_gate @ x) * (w_up @ x))
static ggml_tensor * mtp_gelu_ffn(ggml_context        * ctx,
                                   ggml_tensor         * cur,
                                   const MtpLayerWeights & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up,   cur);
    ggml_tensor * gu   = ggml_geglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.ffn_down, gu);
}

// ─── Public graph builder ─────────────────────────────────────────────────────

bool build_mtp_step_graph(const MtpDrafterWeights  & w,
                          const GemmaTargetCache   & target_cache,
                          const GemmaTargetWeights & target,
                          MtpStepGraph             & out,
                          int                        attn_pos) {
    // ── Validate prerequisites ────────────────────────────────────────────────
    if (!w.pre_projection || !w.post_projection || !w.output_norm) {
        set_last_error("build_mtp_step_graph: MtpDrafterWeights missing pre/post projection or output_norm");
        return false;
    }
    if ((int)w.layers.size() == 0) {
        set_last_error("build_mtp_step_graph: no MTP layers");
        return false;
    }
    if (!target.tok_embd) {
        set_last_error("build_mtp_step_graph: target.tok_embd is null");
        return false;
    }
    if (w.n_embd == 0 || w.n_embd_backbone == 0) {
        set_last_error("build_mtp_step_graph: n_embd or n_embd_backbone is 0");
        return false;
    }

    const int n_embd_backbone = w.n_embd_backbone;
    const int n_layer         = (int)w.layers.size();
    const int n_vocab         = (int)target.tok_embd->ne[1];

    // Validate layer 0 donor KV slot (each layer validates its own in the loop).
    {
        const int32_t donor_il_0 = w.layers[0].donor_target_layer;
        if (donor_il_0 < 0 || donor_il_0 >= (int)target_cache.layer_to_kv_idx.size()) {
            set_last_error("build_mtp_step_graph: invalid donor_target_layer for MTP layer 0");
            return false;
        }
        const int kv_slot_0 = target_cache.layer_to_kv_idx[donor_il_0];
        const int kv_read_slot_0 = (kv_slot_0 >= 0) ? kv_slot_0
            : ((donor_il_0 < (int)target_cache.layer_to_donor_kv.size())
                ? target_cache.layer_to_donor_kv[donor_il_0] : -1);
        if (kv_read_slot_0 < 0 || kv_read_slot_0 >= (int)target_cache.attn_k.size()) {
            set_last_error("build_mtp_step_graph: donor KV slot unresolvable for MTP layer 0");
            return false;
        }
    }

    // ── Allocate ggml context ─────────────────────────────────────────────────
    // Conservative tensor overhead: 3 inputs + ~50 ops per layer + outputs.
    const size_t n_tensors_est = (size_t)(3 + n_layer * 60 + 20);
    ggml_init_params ip{};
    ip.mem_size   = n_tensors_est * ggml_tensor_overhead() + 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        set_last_error("build_mtp_step_graph: ggml_init failed");
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    // ── Input tensors ─────────────────────────────────────────────────────────
    ggml_tensor * in_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_tok);
    ggml_set_name(in_tok, "mtp_in_tok");

    ggml_tensor * in_h_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_backbone, 1);
    ggml_set_input(in_h_prev);
    ggml_set_name(in_h_prev, "mtp_in_h_prev");

    // in_pos: absolute target position for this draft step's RoPE.
    // Caller sets this to (cache.cur_pos + step_offset).
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_pos);
    ggml_set_name(in_pos, "mtp_in_pos");

    // ── 1. Token embedding from target (shared weight) ────────────────────────
    // get_rows selects row in_tok from target.tok_embd (shape [n_vocab, n_embd_backbone])
    // Result: [n_embd_backbone, 1]
    ggml_tensor * tok_e = ggml_get_rows(ctx, target.tok_embd, in_tok);
    ggml_set_name(tok_e, "mtp_tok_embd");

    // Gemma4 scales token embeddings by sqrt(n_embd_backbone) at input pipeline
    const float tok_scale = std::sqrt((float)n_embd_backbone);
    tok_e = ggml_scale(ctx, tok_e, tok_scale);
    ggml_set_name(tok_e, "mtp_tok_embd_scaled");

    // ── 2. Concat [tok_e, h_prev] and project to n_embd ──────────────────────
    // Both are [n_embd_backbone, 1]; concat on axis 0 → [2*n_embd_backbone, 1]
    ggml_tensor * inp_cat = ggml_concat(ctx, tok_e, in_h_prev, 0);
    ggml_set_name(inp_cat, "mtp_concat");

    // pre_projection: [2*n_embd_backbone, n_embd] (ggml ne[0]=2*n_bb, ne[1]=n_embd)
    // mul_mat(A, x): A->ne[0] must == x->ne[0]; output ne[0]=A->ne[1]
    ggml_tensor * inpL = ggml_mul_mat(ctx, w.pre_projection, inp_cat);
    ggml_set_name(inpL, "mtp_pre_proj_out");

    // ── 3. Transformer blocks ─────────────────────────────────────────────────
    for (int il = 0; il < n_layer; ++il) {
        const MtpLayerWeights & L = w.layers[il];
        const bool is_swa = L.is_swa;

        // Resolve donor KV slot
        const int32_t donor_il = L.donor_target_layer;
        if (donor_il < 0 || donor_il >= (int)target_cache.layer_to_kv_idx.size()) {
            set_last_error("build_mtp_step_graph: invalid donor_target_layer");
            ggml_free(ctx);
            return false;
        }
        const int kv_slot = target_cache.layer_to_kv_idx[donor_il];
        const int kv_read_slot = (kv_slot >= 0) ? kv_slot
            : ((donor_il < (int)target_cache.layer_to_donor_kv.size())
                ? target_cache.layer_to_donor_kv[donor_il] : -1);
        if (kv_read_slot < 0 || kv_read_slot >= (int)target_cache.attn_k.size()) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "build_mtp_step_graph: donor KV slot unresolvable for MTP layer %d", il);
            set_last_error(buf);
            ggml_free(ctx);
            return false;
        }
        ggml_tensor * cache_k = target_cache.attn_k[kv_read_slot];
        ggml_tensor * cache_v = target_cache.attn_v[kv_read_slot];
        if (!cache_k || !cache_v) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "build_mtp_step_graph: null KV cache for MTP layer %d donor slot %d", il, kv_read_slot);
            set_last_error(buf);
            ggml_free(ctx);
            return false;
        }

        // KV cache layout: [head_dim_kv, max_ctx, n_head_kv]
        const int64_t head_dim_kv = cache_k->ne[0];
        const int64_t n_head_kv   = cache_k->ne[2];

        // Q dimensions: derive per-layer from wq and attn_q_norm shapes.
        // wq: [n_embd, n_head_q * head_dim_q]  → q_out_dim = n_head_q * head_dim_q
        // attn_q_norm: [head_dim_q]            → head_dim_q (per-head RMS norm weight)
        const int64_t q_out_dim  = L.wq->ne[1];
        const int64_t head_dim_q = L.attn_q_norm->ne[0];  // per-head Q dimension
        const int64_t n_head_q   = q_out_dim / head_dim_q;

        // a) RMSNorm
        ggml_tensor * cur = mtp_rms_norm_mul(ctx, inpL, L.attn_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_norm_%d", il);
            ggml_set_name(cur, name);
        }

        // b) Q projection: [n_embd, 1] → [n_head*head_dim, 1]
        ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
        // Reshape to [head_dim, n_head, 1] for per-head ops
        Qcur = ggml_reshape_3d(ctx, Qcur, head_dim_q, n_head_q, 1);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qcur_%d", il);
            ggml_set_name(Qcur, name);
        }

        // c) Q-norm (per-head RMSNorm, attn_q_norm shape: [head_dim_q])
        Qcur = mtp_rms_norm_mul(ctx, Qcur, L.attn_q_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qcur_normed_%d", il);
            ggml_set_name(Qcur, name);
        }

        // d) RoPE on Q
        // Use the target's rope_theta (SWA layers) or the full-attn layer's rope_freqs.
        // For MTP cross-attention: SWA layers use rope_theta_swa, full layers use rope_theta
        // (with per-layer freq_factors from the donor layer).
        // We use the target's SWA/full rope parameters mirroring atomicbot.
        ggml_tensor * rope_freq_factors = nullptr;
        float rope_theta_val = target.rope_theta_swa;
        if (!is_swa) {
            rope_theta_val = target.rope_theta;
            // For full-attention MTP layers, use the donor target layer's rope_freqs
            if (donor_il >= 0 && donor_il < (int)target.layers.size()) {
                rope_freq_factors = target.layers[donor_il].rope_freqs;
            }
        }
        Qcur = ggml_rope_ext(ctx, Qcur, in_pos,
                              rope_freq_factors,
                              (int)head_dim_q, GGML_ROPE_TYPE_NEOX,
                              /*n_ctx_orig=*/0,
                              rope_theta_val, /*freq_scale=*/1.0f,
                              /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                              /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qcur_pos_%d", il);
            ggml_set_name(Qcur, name);
        }

        // e) Cross-attention
        // Q: [head_dim, n_head, 1] — permute to [head_dim, 1, n_head] for FA
        ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
        Qfa = ggml_cont(ctx, Qfa);

        // K/V view: view [0, attn_pos) from the target KV cache.
        // cache_k: [head_dim_kv, max_ctx, n_head_kv]
        // We view attn_pos slots starting at offset 0.
        // For SWA layers, attn_pos may exceed the ring buffer (swa_ctx_alloc).
        // Clip to actual cache size — only committed positions exist.
        const int64_t kv_seq_len = std::min((int64_t)attn_pos, cache_k->ne[1]);
        // Pad to 1 minimum to avoid zero-size tensors when attn_pos==0.
        const int64_t kv_view_len = std::max(kv_seq_len, (int64_t)1);

        ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
            head_dim_kv, kv_view_len, n_head_kv,
            cache_k->nb[1], cache_k->nb[2],
            /*offset=*/0);
        ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
            head_dim_kv, kv_view_len, n_head_kv,
            cache_v->nb[1], cache_v->nb[2],
            /*offset=*/0);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Kfa_%d", il);
            ggml_set_name(Kfa, name);
            std::snprintf(name, sizeof(name), "mtp_Vfa_%d", il);
            ggml_set_name(Vfa, name);
        }

        // Flash attention — no causal mask needed: all KV positions ≤ attn_pos
        // are uniformly admitted since step position > attn_pos.
        // Gemma4 attn_scale = 1.0 (matches target graph).
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa,
                                                  /*mask=*/nullptr,
                                                  /*scale=*/target.attn_scale,
                                                  /*max_bias=*/0.0f,
                                                  /*logit_softcap=*/0.0f);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_out_%d", il);
            ggml_set_name(attn, name);
        }

        // Reshape: [head_dim*n_head, 1] then output projection
        attn = ggml_reshape_2d(ctx, attn, head_dim_q * n_head_q, 1);
        cur = ggml_mul_mat(ctx, L.wo, attn);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_proj_%d", il);
            ggml_set_name(cur, name);
        }

        // f) Post-attention norm
        cur = mtp_rms_norm_mul(ctx, cur, L.attn_post_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_post_norm_%d", il);
            ggml_set_name(cur, name);
        }

        // g) Attention residual
        ggml_tensor * attn_residual = ggml_add(ctx, cur, inpL);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_residual_%d", il);
            ggml_set_name(attn_residual, name);
        }

        // h) FFN norm
        ggml_tensor * ffn_in = mtp_rms_norm_mul(ctx, attn_residual, L.ffn_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_ffn_norm_%d", il);
            ggml_set_name(ffn_in, name);
        }

        // i) GELU FFN
        ggml_tensor * ffn_out = mtp_gelu_ffn(ctx, ffn_in, L);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_ffn_out_%d", il);
            ggml_set_name(ffn_out, name);
        }

        // j) Post-FFN norm
        ffn_out = mtp_rms_norm_mul(ctx, ffn_out, L.ffn_post_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_ffn_post_norm_%d", il);
            ggml_set_name(ffn_out, name);
        }

        // k) FFN residual
        cur = ggml_add(ctx, ffn_out, attn_residual);

        // l) Optional per-layer output scale
        if (L.out_scale) {
            cur = ggml_mul(ctx, cur, L.out_scale);
            {
                char name[64]; std::snprintf(name, sizeof(name), "mtp_out_scaled_%d", il);
                ggml_set_name(cur, name);
            }
        }

        inpL = cur;
    }

    // ── 4. Output norm ────────────────────────────────────────────────────────
    ggml_tensor * h_inner = mtp_rms_norm_mul(ctx, inpL, w.output_norm);
    ggml_set_name(h_inner, "mtp_result_norm");

    // ── 5. Post-projection → h_post (next h_prev) ─────────────────────────────
    // post_projection: [n_embd, n_embd_backbone] (ggml ne[0]=n_embd, ne[1]=n_embd_backbone)
    ggml_tensor * h_post = ggml_mul_mat(ctx, w.post_projection, h_inner);
    ggml_set_name(h_post, "mtp_post_proj_out");

    // ── 6. LM head ────────────────────────────────────────────────────────────
    ggml_tensor * logits = nullptr;

    if (w.use_ordered_embeddings && w.centroids && w.n_centroids > 0) {
        // Centroid-routed LM head (matches atomicbot lines 190-235).
        // All mul_mat ops use h_inner [n_embd, 1] (MTP's own hidden space, n_embd=1024).
        // The embedding source is the MTP model's own tok_embd [n_embd, n_vocab] (w.tok_embd),
        // NOT the target's tok_embd (which is in backbone space and used only in step 1).
        if (!w.tok_embd) {
            set_last_error("build_mtp_step_graph: use_ordered_embeddings=true but w.tok_embd is null (token_embd.weight missing from GGUF)");
            ggml_free(ctx);
            return false;
        }

        const int64_t n_c   = (int64_t)w.n_centroids;
        const int64_t top_k = (int64_t)w.centroid_top_k;
        // vsc: tokens per centroid slot
        const int64_t vsc   = (int64_t)n_vocab / n_c;

        // centroid_logits = mul_mat(centroids, h_inner) → [n_centroids, 1]
        // centroids: [n_embd, n_centroids] (ne[0]=n_embd, ne[1]=n_centroids)
        ggml_tensor * centroid_logits = ggml_mul_mat(ctx, w.centroids, h_inner);
        ggml_set_name(centroid_logits, "mtp_centroid_logits");

        // top-k centroid indices
        ggml_tensor * topk_idx = ggml_top_k(ctx, centroid_logits, (int)top_k);
        ggml_set_name(topk_idx, "mtp_centroid_topk_idx");

        // View token_ordering as [vsc, n_centroids] (I32)
        const size_t ordering_row_bytes = ggml_row_size(GGML_TYPE_I32, vsc);
        ggml_tensor * ordering = ggml_view_2d(ctx, w.token_ordering,
            vsc, n_c, ordering_row_bytes, /*offset=*/0);
        ggml_set_name(ordering, "mtp_token_ordering_view");

        // Gather candidate token ids for top-k centroids: [vsc, top_k, 1]
        ggml_tensor * sel_ids = ggml_get_rows(ctx, ordering, topk_idx);
        ggml_set_name(sel_ids, "mtp_selected_token_ids");

        // Flatten to 1D for embedding lookup
        const int64_t n_sel = top_k * vsc;
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx, sel_ids, n_sel);
        ggml_set_name(flat_ids, "mtp_selected_token_ids_flat");

        // Gather embeddings for selected tokens from MTP's own tok_embd [n_embd, n_vocab].
        // get_rows selects n_sel rows → [n_embd, n_sel]
        ggml_tensor * sel_emb = ggml_get_rows(ctx, w.tok_embd, flat_ids);
        ggml_set_name(sel_emb, "mtp_selected_embd");

        // Sparse logits: mul_mat(sel_emb, h_inner):
        // sel_emb [n_embd, n_sel], h_inner [n_embd, 1] → [n_sel, 1]
        ggml_tensor * sel_logits = ggml_mul_mat(ctx, sel_emb, h_inner);
        ggml_set_name(sel_logits, "mtp_selected_logits");
        ggml_tensor * sel_logits_f32 = ggml_cast(ctx, sel_logits, GGML_TYPE_F32);
        ggml_set_name(sel_logits_f32, "mtp_selected_logits_f32");

        // Build full vocab row pre-filled with -1e30
        ggml_tensor * logits_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, 1);
        logits_full = ggml_fill_inplace(ctx, logits_full, -1e30f);
        ggml_set_name(logits_full, "mtp_logits_masked_base");

        // Scatter selected logits into full row
        ggml_tensor * scatter_dst = ggml_cont_2d(ctx, logits_full, 1, (int64_t)n_vocab);
        ggml_tensor * scatter_src = ggml_cont_2d(ctx, sel_logits_f32, 1, n_sel);
        logits = ggml_set_rows(ctx, scatter_dst, scatter_src, flat_ids);
        logits = ggml_reshape_2d(ctx, logits, n_vocab, 1);
        ggml_set_name(logits, "mtp_logits_full");
    } else {
        // Dense tied LM head: mul_mat(tok_embd, h_post) → [n_vocab, 1]
        // For non-ordered-embeddings models (n_embd == n_embd_backbone), use h_post
        // (post-projected to n_embd_backbone) so dimensions match target.tok_embd.
        // Prefer w.tok_embd (MTP's own, in n_embd space) if available, else
        // fall back to target.tok_embd (in n_embd_backbone space) with h_post.
        if (w.tok_embd) {
            // MTP has its own tied LM head in n_embd space
            logits = ggml_mul_mat(ctx, w.tok_embd, h_inner);
        } else {
            // Fallback: use target's tok_embd against the backbone-projected hidden
            logits = ggml_mul_mat(ctx, target.tok_embd, h_post);
        }
        ggml_set_name(logits, "mtp_logits_dense");
    }

    // Optional logit softcapping (matches target's softcap=30)
    if (target.logit_softcap > 0.0f) {
        logits = ggml_scale(ctx, logits, 1.0f / target.logit_softcap);
        logits = ggml_tanh(ctx, logits);
        logits = ggml_scale(ctx, logits, target.logit_softcap);
        ggml_set_name(logits, "mtp_logits_softcapped");
    }

    // ── 7. In-graph argmax ─────────────────────────────────────────────────────
    ggml_tensor * argmax = ggml_argmax(ctx, logits);
    ggml_set_name(argmax, "mtp_argmax");

    // Expand all outputs into the graph
    ggml_build_forward_expand(gf, argmax);
    ggml_build_forward_expand(gf, h_post);
    // Note: logits is already in argmax's DAG, but mark it as output for diagnostic reads.
    ggml_set_output(logits);
    ggml_set_output(h_post);
    ggml_set_output(argmax);

    // ── Populate output struct ────────────────────────────────────────────────
    out.ctx        = ctx;
    out.gf         = gf;
    out.in_tok     = in_tok;
    out.in_h_prev  = in_h_prev;
    out.in_pos     = in_pos;
    out.out_logits = logits;
    out.out_h_post = h_post;
    out.out_argmax = argmax;

    return true;
}

void free_mtp_step_graph(MtpStepGraph & g) {
    if (g.ctx) {
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }
    g.gf         = nullptr;
    g.in_tok     = nullptr;
    g.in_h_prev  = nullptr;
    g.in_pos     = nullptr;
    g.out_logits = nullptr;
    g.out_h_post = nullptr;
    g.out_argmax = nullptr;
}

} // namespace dflash27b
