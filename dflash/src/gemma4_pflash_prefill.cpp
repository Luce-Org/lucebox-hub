// Layer-by-layer prefill for Gemma4 using pFlash (flash_prefill) for full-
// attention layers and ggml flash_attn_ext for SWA layers.
//
// Full-attention layers: Graph A (Q/K/V proj + RoPE) → flash_prefill_forward
//                        → Graph B (output proj + FFN + residuals).
// SWA layers: single ggml graph per chunk (attn_norm → FA → FFN → residual).
//
// Fused graph optimization: Graph B for full-attn layer N is fused with
// SWA layer N+1 and Graph A for full-attn layer N+2 into a single ggml graph,
// reducing graph build+alloc+compute cycles by ~3x.
//
// All state is written into GemmaTargetCache (KV cache, target_feat).
// On return: cache.cur_pos = n_prompt, cache.last_tok = argmax of last token.

#include "internal.h"
#include "flashprefill.h"

#if DFLASH27B_MIN_SM >= 80
#include <cuda_runtime.h>
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dflash27b {

static constexpr float PFLASH_EPS = GEMMA4_RMS_EPS;

// ─── PersBuf: GPU tensor with its own ggml_context + backend buffer ──────────

struct PersBuf {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

static bool make_pers(ggml_backend_t backend, ggml_type type, int n_dim,
                      const int64_t * dims, PersBuf & out) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4 + 1024;
    ip.no_alloc   = true;
    ip.mem_buffer = nullptr;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    if      (n_dim == 1) out.t = ggml_new_tensor_1d(out.ctx, type, dims[0]);
    else if (n_dim == 2) out.t = ggml_new_tensor_2d(out.ctx, type, dims[0], dims[1]);
    else if (n_dim == 3) out.t = ggml_new_tensor_3d(out.ctx, type, dims[0], dims[1], dims[2]);
    else return false;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    return out.buf != nullptr;
}

static void free_pers(PersBuf & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.t = nullptr;
}

// ─── Local helpers ────────────────────────────────────────────────────────────

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

// GeGLU FFN matching the Gemma4 graph implementation exactly.
// Uses ggml_geglu_split (not separate gelu + mul).
static ggml_tensor * build_geglu_ffn(ggml_context * ctx,
                                     ggml_tensor * cur,
                                     const GemmaTargetLayer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.w_up,   cur);
    ggml_tensor * gu   = ggml_geglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);
}

// MoE FFN — copied from gemma4_target_graph.cpp (static there; duplicated here).
static ggml_tensor * build_moe_ffn(ggml_context * ctx,
                                   ggml_cgraph *  gf,
                                   const GemmaTargetWeights & w,
                                   const GemmaTargetLayer & L,
                                   ggml_tensor * cur_shared_ffn,
                                   ggml_tensor * cur_moe_ffn,
                                   ggml_tensor * cur_for_router,
                                   int n_tokens) {
    const int n_embd        = w.n_embd;
    const int n_expert_used = w.n_expert_used;
    const int n_expert      = w.n_expert;
    const int n_ff_exp      = w.n_ff_exp;

    ggml_tensor * shared_out = nullptr;
    if (L.w_gate && L.w_up && L.w_down) {
        ggml_tensor * sg  = ggml_mul_mat(ctx, L.w_gate, cur_shared_ffn);
        ggml_tensor * su  = ggml_mul_mat(ctx, L.w_up,   cur_shared_ffn);
        ggml_tensor * sgu = ggml_geglu_split(ctx, sg, su);
        shared_out = ggml_mul_mat(ctx, L.w_down, sgu);
        if (L.ffn_post_norm_1) {
            shared_out = rms_norm_mul(ctx, shared_out, L.ffn_post_norm_1, PFLASH_EPS);
        }
    }

    ggml_tensor * router_in = ggml_rms_norm(ctx, cur_for_router, PFLASH_EPS);
    router_in = ggml_scale(ctx, router_in, 1.0f / std::sqrt((float)n_embd));
    if (L.ffn_gate_inp_s) {
        router_in = ggml_mul(ctx, router_in, L.ffn_gate_inp_s);
    }
    ggml_tensor * router_logits = ggml_mul_mat(ctx, L.ffn_gate_inp, router_in);
    ggml_tensor * probs = ggml_soft_max(ctx, router_logits);
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx, probs, n_expert_used);

    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected_experts);
    {
        ggml_tensor * w2d  = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        ggml_tensor * wsum = ggml_sum_rows(ctx, w2d);
        wsum  = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        w2d   = ggml_div(ctx, w2d, wsum);
        weights = ggml_reshape_3d(ctx, w2d, 1, n_expert_used, n_tokens);
    }

    ggml_tensor * expert_out = nullptr;
    if (L.ffn_gate_up_exps && L.ffn_down_exps) {
        ggml_tensor * x = ggml_reshape_3d(ctx, cur_moe_ffn, n_embd, 1, n_tokens);
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, L.ffn_gate_up_exps,
                                                x, selected_experts);

        const size_t elt = ggml_element_size(gate_up);
        ggml_tensor * g_half = ggml_view_3d(ctx, gate_up,
            n_ff_exp, n_expert_used, n_tokens,
            (size_t)n_ff_exp * 2 * elt,
            (size_t)n_ff_exp * 2 * n_expert_used * elt,
            0);
        ggml_tensor * u_half = ggml_view_3d(ctx, gate_up,
            n_ff_exp, n_expert_used, n_tokens,
            (size_t)n_ff_exp * 2 * elt,
            (size_t)n_ff_exp * 2 * n_expert_used * elt,
            (size_t)n_ff_exp * elt);

        g_half = ggml_cont(ctx, g_half);
        u_half = ggml_cont(ctx, u_half);
        ggml_tensor * activated = ggml_mul(ctx, ggml_gelu(ctx, g_half), u_half);
        activated = ggml_mul(ctx, activated, weights);

        ggml_tensor * down_out = ggml_mul_mat_id(ctx, L.ffn_down_exps,
                                                  activated, selected_experts);

        if (L.ffn_down_exps_s) {
            down_out = ggml_mul(ctx, down_out, L.ffn_down_exps_s);
        }

        ggml_build_forward_expand(gf, down_out);
        expert_out = ggml_view_2d(ctx, down_out,
                                   n_embd, n_tokens,
                                   down_out->nb[2],
                                   0);
        ggml_build_forward_expand(gf, expert_out);
        for (int ei = 1; ei < n_expert_used; ++ei) {
            ggml_tensor * slice = ggml_view_2d(ctx, down_out,
                                               n_embd, n_tokens,
                                               down_out->nb[2],
                                               (size_t)ei * down_out->nb[1]);
            ggml_build_forward_expand(gf, slice);
            expert_out = ggml_add(ctx, expert_out, slice);
            ggml_build_forward_expand(gf, expert_out);
        }

        if (L.ffn_post_norm_2) {
            expert_out = rms_norm_mul(ctx, expert_out, L.ffn_post_norm_2, PFLASH_EPS);
        }
    }

    if (shared_out && expert_out) return ggml_add(ctx, shared_out, expert_out);
    if (shared_out)               return shared_out;
    if (expert_out)               return expert_out;
    return cur_shared_ffn;
}

// ─── Capture target features into cache.target_feat (ring buffer) ────────────

static void capture_target_feat(ggml_context * ctx, ggml_cgraph * gf,
                                 const GemmaTargetWeights & w,
                                 GemmaTargetCache & cache,
                                 ggml_tensor * cur,
                                 int il, int kv_start, int cs, int cl) {
    if (!cache.target_feat) return;
    for (int k = 0; k < w.n_capture_layers; k++) {
        if (w.capture_layer_ids[k] != il) continue;
        const size_t elt = ggml_element_size(cache.target_feat);
        const size_t col_stride = (size_t)w.n_capture_layers * w.n_embd * elt;
        const int slot_start = (kv_start + cs) % cache.target_feat_cap;
        const int pre_n  = std::min(cl, cache.target_feat_cap - slot_start);
        const int post_n = cl - pre_n;

        ggml_tensor * dst1 = ggml_view_2d(ctx, cache.target_feat,
            w.n_embd, pre_n, col_stride,
            (size_t)slot_start * col_stride + (size_t)k * w.n_embd * elt);
        ggml_tensor * src1 = ggml_view_2d(ctx, cur,
            w.n_embd, pre_n, cur->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src1, dst1));

        if (post_n > 0) {
            ggml_tensor * dst2 = ggml_view_2d(ctx, cache.target_feat,
                w.n_embd, post_n, col_stride,
                (size_t)k * w.n_embd * elt);
            ggml_tensor * src2 = ggml_view_2d(ctx, cur,
                w.n_embd, post_n, cur->nb[1],
                (size_t)pre_n * cur->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src2, dst2));
        }
        break;
    }
}

// ─── Graph-fragment helpers (all share caller's ctx + gf) ────────────────────

// Struct to hold all chunk-level context needed by graph fragment builders.
struct ChunkCtx {
    int S;
    int cs;   // chunk start token index
    int cl;   // chunk length (tokens)
    int kv_start;
    ggml_tensor * h_view;       // view into hidden_buf for this chunk [n_embd, cl]
    ggml_tensor * pos_chunk;    // view into pos_buf for this chunk [cl]
};

// Build Graph A ops: attn_norm → Q/K/V proj + RoPE → write to Q/K/V bufs + KV cache.
// Returns nothing (writes are the outputs via ggml_cpy expand).
static void build_graph_A_ops(ggml_context * ctx, ggml_cgraph * gf,
                               const GemmaTargetWeights & w,
                               const GemmaTargetLayer & L,
                               GemmaTargetCache & cache,
                               ggml_tensor * cache_k, ggml_tensor * cache_v,
                               PersBuf & Q_buf, PersBuf & K_buf, PersBuf & V_buf,
                               int il, int n_kv_layer,
                               const ChunkCtx & cc) {
    const int n_embd = w.n_embd;
    const int n_head = w.n_head;
    const int D      = w.head_dim;

    ggml_tensor * h_norm = rms_norm_mul(ctx, cc.h_view, L.attn_norm, PFLASH_EPS);

    // Q: [n_embd, cl] → [D, n_head, cl]
    ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, h_norm);
    Q = ggml_reshape_3d(ctx, Q, D, n_head, cc.cl);
    Q = rms_norm_mul(ctx, Q, L.q_norm, PFLASH_EPS);
    Q = ggml_rope_ext(ctx, Q, cc.pos_chunk, L.rope_freqs,
        D, GGML_ROPE_TYPE_NEOX, 0,
        w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // K: [n_embd, cl] → [D, n_kv_layer, cl]
    ggml_tensor * K = ggml_mul_mat(ctx, L.wk, h_norm);
    K = ggml_reshape_3d(ctx, K, D, n_kv_layer, cc.cl);
    if (L.k_norm)
        K = rms_norm_mul(ctx, K, L.k_norm, PFLASH_EPS);
    else
        K = ggml_rms_norm(ctx, K, PFLASH_EPS);
    K = ggml_rope_ext(ctx, K, cc.pos_chunk, L.rope_freqs,
        D, GGML_ROPE_TYPE_NEOX, 0,
        w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // V: [n_embd, cl] → [D, n_kv_layer, cl]
    ggml_tensor * V = ggml_mul_mat(ctx, L.wv, h_norm);
    V = ggml_reshape_3d(ctx, V, D, n_kv_layer, cc.cl);
    V = ggml_rms_norm(ctx, V, PFLASH_EPS);

    // Write Q/K/V to persistent BF16 buffers for pFlash
    const size_t q_esz  = ggml_element_size(Q_buf.t);
    const size_t kv_esz = ggml_element_size(K_buf.t);

    ggml_tensor * Q_dst = ggml_view_3d(ctx, Q_buf.t, D, n_head, cc.cl,
        q_esz * D, q_esz * D * n_head,
        (size_t)cc.cs * q_esz * D * n_head);
    ggml_tensor * K_dst = ggml_view_3d(ctx, K_buf.t, D, n_kv_layer, cc.cl,
        kv_esz * D, kv_esz * D * n_kv_layer,
        (size_t)cc.cs * kv_esz * D * n_kv_layer);
    ggml_tensor * V_dst = ggml_view_3d(ctx, V_buf.t, D, n_kv_layer, cc.cl,
        kv_esz * D, kv_esz * D * n_kv_layer,
        (size_t)cc.cs * kv_esz * D * n_kv_layer);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, Q, Q_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, K, K_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, V, V_dst));

    // Also write quantized K/V into KV cache for decode reuse
    if (cache_k && cache_v) {
        ggml_tensor * Kcur_T = ggml_permute(ctx, K, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(ctx, V, 0, 2, 1, 3);
        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            D, cc.cl, n_kv_layer,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * (cc.kv_start + cc.cs));
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            D, cc.cl, n_kv_layer,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * (cc.kv_start + cc.cs));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }
    (void)il;
}

// Build Graph B ops: output proj → post_attn_norm → residual → FFN → residual.
// Takes h_in (the hidden state entering this full-attn layer, same as h_view in
// the original Graph B), and the attn_out chunk from attn_out_buf.
// Returns the output hidden state tensor (not yet written back to hidden_buf).
static ggml_tensor * build_graph_B_ops(ggml_context * ctx, ggml_cgraph * gf,
                                        const GemmaTargetWeights & w,
                                        const GemmaTargetLayer & L,
                                        GemmaTargetCache & cache,
                                        PersBuf & attn_out_buf,
                                        int il, int n_kv_layer,
                                        const ChunkCtx & cc) {
    const int n_head = w.n_head;
    const int D      = w.head_dim;
    (void)n_kv_layer;

    const size_t a_esz  = ggml_element_size(attn_out_buf.t);
    ggml_tensor * attn_chunk = ggml_view_2d(ctx, attn_out_buf.t,
        D * n_head, cc.cl, a_esz * D * n_head,
        (size_t)cc.cs * a_esz * D * n_head);

    ggml_tensor * attn_proj = ggml_mul_mat(ctx, L.wo, attn_chunk);
    if (L.attn_post_norm)
        attn_proj = rms_norm_mul(ctx, attn_proj, L.attn_post_norm, PFLASH_EPS);

    ggml_tensor * h_after = ggml_add(ctx, attn_proj, cc.h_view);

    ggml_tensor * ffn_in  = rms_norm_mul(ctx, h_after, L.ffn_norm, PFLASH_EPS);
    ggml_tensor * ffn_out = nullptr;
    if (L.ffn_gate_inp) {
        ggml_tensor * moe_in = L.ffn_pre_norm_2
            ? rms_norm_mul(ctx, h_after, L.ffn_pre_norm_2, PFLASH_EPS)
            : ffn_in;
        ffn_out = build_moe_ffn(ctx, gf, w, L, ffn_in, moe_in, h_after, cc.cl);
    } else {
        ffn_out = build_geglu_ffn(ctx, ffn_in, L);
    }
    if (L.ffn_post_norm)
        ffn_out = rms_norm_mul(ctx, ffn_out, L.ffn_post_norm, PFLASH_EPS);

    ggml_tensor * cur = ggml_add(ctx, ffn_out, h_after);

    if (L.out_scale) cur = ggml_mul(ctx, cur, L.out_scale);

    capture_target_feat(ctx, gf, w, cache, cur, il, cc.kv_start, cc.cs, cc.cl);

    return cur;
}

// Build SWA layer ops: attn_norm → Q/K/V → FA → output proj → FFN → residual.
// Takes h_in as the input hidden state (may be the output of Graph B, i.e. cur_b).
// The h_view_orig is used for the residual add (same as h_in in the original code).
// Returns the output hidden state tensor.
static ggml_tensor * build_swa_ops(ggml_context * ctx, ggml_cgraph * gf,
                                    const GemmaTargetWeights & w,
                                    const GemmaTargetLayer & L,
                                    GemmaTargetCache & cache,
                                    ggml_tensor * cache_k, ggml_tensor * cache_v,
                                    int il, int n_kv_layer,
                                    ggml_tensor * h_in,
                                    ggml_tensor ** out_attn_mask,
                                    const ChunkCtx & cc) {
    const int n_head  = w.n_head;
    const int D_swa   = w.head_dim_swa;

    ggml_tensor * cur = rms_norm_mul(ctx, h_in, L.attn_norm, PFLASH_EPS);

    // Q
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
    Qcur = ggml_reshape_3d(ctx, Qcur, D_swa, n_head, cc.cl);
    Qcur = rms_norm_mul(ctx, Qcur, L.q_norm, PFLASH_EPS);
    Qcur = ggml_rope_ext(ctx, Qcur, cc.pos_chunk, nullptr,
        D_swa, GGML_ROPE_TYPE_NEOX, 0,
        w.rope_theta_swa, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // K/V + cache write
    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    const bool write_kv_swa = (cache_k && cache_v);
    if (write_kv_swa) {
        Kcur = ggml_mul_mat(ctx, L.wk, cur);
        Kcur = ggml_reshape_3d(ctx, Kcur, D_swa, n_kv_layer, cc.cl);
        if (L.k_norm)
            Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, PFLASH_EPS);
        else
            Kcur = ggml_rms_norm(ctx, Kcur, PFLASH_EPS);
        Kcur = ggml_rope_ext(ctx, Kcur, cc.pos_chunk, nullptr,
            D_swa, GGML_ROPE_TYPE_NEOX, 0,
            w.rope_theta_swa, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Vcur = ggml_mul_mat(ctx, L.wv, cur);
        Vcur = ggml_reshape_3d(ctx, Vcur, D_swa, n_kv_layer, cc.cl);
        Vcur = ggml_rms_norm(ctx, Vcur, PFLASH_EPS);

        ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);
        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            D_swa, cc.cl, n_kv_layer,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * (cc.kv_start + cc.cs));
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            D_swa, cc.cl, n_kv_layer,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * (cc.kv_start + cc.cs));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }

    // SWA window
    int win_start = 0;
    if (w.swa_window > 0 && (cc.kv_start + cc.cs) > w.swa_window)
        win_start = (cc.kv_start + cc.cs) - w.swa_window;
    int win_len = (cc.kv_start + cc.cs + cc.cl) - win_start;

    if (cache_k && (cache.kv_k_type == GGML_TYPE_TQ3_0 || D_swa >= 512)) {
        const int pad = 256 / (int)ggml_type_size(cache.kv_k_type);
        if (pad > 0) {
            win_start = (win_start / pad) * pad;
            win_len   = (cc.kv_start + cc.cs + cc.cl) - win_start;
        }
    }

    // Build SWA causal mask (F16 required by ggml_flash_attn_ext)
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, win_len, cc.cl);
    if (out_attn_mask) *out_attn_mask = attn_mask;

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = nullptr;
    ggml_tensor * Vfa = nullptr;
    if (cache_k && cache_v) {
        Kfa = ggml_view_3d(ctx, cache_k,
            D_swa, win_len, n_kv_layer,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * win_start);
        Vfa = ggml_view_3d(ctx, cache_v,
            D_swa, win_len, n_kv_layer,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * win_start);
    }

    ggml_tensor * attn_out = nullptr;
    if (Kfa && Vfa) {
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa,
            attn_mask, 1.0f, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx, attn, D_swa * n_head, cc.cl);
    } else {
        // No KV cache available: zero output
        attn_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_swa * n_head, cc.cl);
    }

    // Output projection
    ggml_tensor * attn_proj = ggml_mul_mat(ctx, L.wo, attn_out);
    if (L.attn_post_norm)
        attn_proj = rms_norm_mul(ctx, attn_proj, L.attn_post_norm, PFLASH_EPS);

    ggml_tensor * h_after = ggml_add(ctx, attn_proj, h_in);

    // FFN
    ggml_tensor * ffn_in  = rms_norm_mul(ctx, h_after, L.ffn_norm, PFLASH_EPS);
    ggml_tensor * ffn_out = nullptr;
    if (L.ffn_gate_inp) {
        ggml_tensor * moe_in = L.ffn_pre_norm_2
            ? rms_norm_mul(ctx, h_after, L.ffn_pre_norm_2, PFLASH_EPS)
            : ffn_in;
        ffn_out = build_moe_ffn(ctx, gf, w, L, ffn_in, moe_in, h_after, cc.cl);
    } else {
        ffn_out = build_geglu_ffn(ctx, ffn_in, L);
    }
    if (L.ffn_post_norm)
        ffn_out = rms_norm_mul(ctx, ffn_out, L.ffn_post_norm, PFLASH_EPS);

    ggml_tensor * result = ggml_add(ctx, ffn_out, h_after);

    if (L.out_scale) result = ggml_mul(ctx, result, L.out_scale);

    capture_target_feat(ctx, gf, w, cache, result, il, cc.kv_start, cc.cs, cc.cl);

    return result;
}

// Helper: fill and upload the SWA causal mask to GPU.
static void fill_swa_mask(ggml_tensor * attn_mask, int win_start, int win_len,
                           int cs, int cl, int kv_start, int swa_window) {
    constexpr uint16_t F16_ZERO    = 0x0000;
    constexpr uint16_t F16_NEG_INF = 0xFC00;
    std::vector<uint16_t> mask_data((size_t)win_len * cl);
    for (int qi = 0; qi < cl; qi++) {
        const int abs_q = kv_start + cs + qi;
        for (int ki = 0; ki < win_len; ki++) {
            const int abs_k   = win_start + ki;
            const bool causal = abs_k <= abs_q;
            const bool in_win = (swa_window <= 0) ||
                                (abs_q - abs_k < swa_window);
            mask_data[qi * win_len + ki] = (causal && in_win) ? F16_ZERO : F16_NEG_INF;
        }
    }
    ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
        mask_data.size() * sizeof(uint16_t));
}

// Compute SWA window bounds for a chunk (accounts for quantization padding).
// cache_k_present must match whether cache_k was non-null in the graph build,
// so the mask dimensions align with the attn_mask tensor created in build_swa_ops.
static void swa_window_bounds(const GemmaTargetWeights & w,
                               const GemmaTargetCache & cache,
                               int cs, int cl, int kv_start,
                               bool cache_k_present,
                               int & win_start_out, int & win_len_out) {
    int win_start = 0;
    if (w.swa_window > 0 && (kv_start + cs) > w.swa_window)
        win_start = (kv_start + cs) - w.swa_window;
    int win_len = (kv_start + cs + cl) - win_start;

    // Mirror the padding condition in build_swa_ops exactly.
    if (cache_k_present && (cache.kv_k_type == GGML_TYPE_TQ3_0 || w.head_dim_swa >= 512)) {
        const int pad = 256 / (int)ggml_type_size(cache.kv_k_type);
        if (pad > 0) {
            win_start = (win_start / pad) * pad;
            win_len   = (kv_start + cs + cl) - win_start;
        }
    }
    win_start_out = win_start;
    win_len_out   = win_len;
}

// ─── pFlash invocation helper ─────────────────────────────────────────────────

static int run_pflash(const GemmaTargetWeights & w,
                      GemmaTargetCache & cache,
                      ggml_backend_t backend,
                      PersBuf & Q_buf, PersBuf & K_buf, PersBuf & V_buf,
                      PersBuf & attn_out_buf,
                      int il, int S, int n_head, int n_kv_layer, int D,
                      const flashprefill::FlashPrefillConfig & fp_cfg) {
    (void)cache;
#if DFLASH27B_MIN_SM >= 80
    {
        int rc = flashprefill::flash_prefill_forward_bf16(
            Q_buf.t->data, K_buf.t->data, V_buf.t->data, attn_out_buf.t->data,
            1, S, n_head, n_kv_layer, D,
            1.0f, fp_cfg);
        if (rc != 0) return rc;
        cudaDeviceSynchronize();
    }
#else
    {
        int rc = flashprefill::flash_prefill_forward_q8(
            backend,
            Q_buf.t->data, K_buf.t->data, V_buf.t->data, attn_out_buf.t->data,
            1, S, n_head, n_kv_layer, D,
            1.0f, (int)ggml_element_size(Q_buf.t), fp_cfg);
        if (rc != 0) return rc;
    }
#endif
    std::fprintf(stderr, "[pflash] layer %d/%d done\n", il + 1, w.n_layer);
    return 0;
}

// ─── Public entry point ───────────────────────────────────────────────────────

int gemma4_pflash_prefill(const GemmaTargetWeights & w,
                          GemmaTargetCache & cache,
                          ggml_backend_t backend,
                          const int32_t * prompt_ids, int n_prompt,
                          float pflash_alpha) {
    const int S         = n_prompt;
    const int n_embd    = w.n_embd;
    const int n_layer   = w.n_layer;
    const int n_head    = w.n_head;
    const int D         = w.head_dim;
    const int D_swa     = w.head_dim_swa;
    const int n_head_kv = w.n_head_kv;

    // All chunked operations (both Graph A and SWA) use SWA_CHUNK since fused
    // graphs include SWA attention which constrains the chunk size.
    const int SWA_CHUNK = std::min(32768, std::max(1024, w.swa_window));
    const ggml_type half_type = GGML_TYPE_BF16;

    // ── Persistent GPU buffers ────────────────────────────────────────────────
    PersBuf hidden_buf, pos_buf, Q_buf, K_buf, V_buf, attn_out_buf;

    {
        int64_t dims[2] = {n_embd, S};
        if (!make_pers(backend, GGML_TYPE_F32, 2, dims, hidden_buf)) {
            set_last_error("pflash: failed to alloc hidden_buf"); return -1;
        }
    }
    {
        int64_t dims[1] = {S};
        if (!make_pers(backend, GGML_TYPE_I32, 1, dims, pos_buf)) {
            set_last_error("pflash: failed to alloc pos_buf");
            free_pers(hidden_buf); return -1;
        }
    }

    const int D_max = std::max(D, D_swa);
    int max_n_kv = n_head_kv;
    for (int kv : w.head_kv_per_layer) max_n_kv = std::max(max_n_kv, kv);

    {
        int64_t dims[3] = {D_max, n_head, S};
        if (!make_pers(backend, half_type, 3, dims, Q_buf)) {
            set_last_error("pflash: failed to alloc Q_buf");
            free_pers(hidden_buf); free_pers(pos_buf); return -1;
        }
    }
    {
        int64_t dims[3] = {D, max_n_kv, S};
        if (!make_pers(backend, half_type, 3, dims, K_buf)) {
            set_last_error("pflash: failed to alloc K_buf");
            free_pers(hidden_buf); free_pers(pos_buf); free_pers(Q_buf); return -1;
        }
    }
    {
        int64_t dims[3] = {D, max_n_kv, S};
        if (!make_pers(backend, half_type, 3, dims, V_buf)) {
            set_last_error("pflash: failed to alloc V_buf");
            free_pers(hidden_buf); free_pers(pos_buf); free_pers(Q_buf); free_pers(K_buf);
            return -1;
        }
    }
    {
        int64_t dims[2] = {(int64_t)D * n_head, S};
        if (!make_pers(backend, half_type, 2, dims, attn_out_buf)) {
            set_last_error("pflash: failed to alloc attn_out_buf");
            free_pers(hidden_buf); free_pers(pos_buf); free_pers(Q_buf);
            free_pers(K_buf); free_pers(V_buf); return -1;
        }
    }

    auto cleanup = [&]() {
        free_pers(hidden_buf); free_pers(pos_buf);
        free_pers(Q_buf); free_pers(K_buf); free_pers(V_buf);
        free_pers(attn_out_buf);
    };

    // ── Fill position buffer [0..S-1] ─────────────────────────────────────────
    {
        std::vector<int32_t> pos(S);
        for (int i = 0; i < S; i++) pos[i] = i;
        ggml_backend_tensor_set(pos_buf.t, pos.data(), 0, S * sizeof(int32_t));
    }

    // ── Embed tokens → hidden_buf (scaled by √n_embd, matching Gemma4 embedding) ──
    {
        std::vector<float> emb((size_t)n_embd * S);
        if (!w.embedder.embed(prompt_ids, S, emb.data())) {
            cleanup(); set_last_error("pflash: embed failed"); return -1;
        }
        const float scale = std::sqrt((float)n_embd);
        for (int i = 0; i < n_embd * S; i++) emb[i] *= scale;
        ggml_backend_tensor_set(hidden_buf.t, emb.data(), 0, (size_t)n_embd * S * sizeof(float));
    }

    // ── pFlash config ─────────────────────────────────────────────────────────
    flashprefill::FlashPrefillConfig fp_cfg;
    fp_cfg.alpha = pflash_alpha;
    if (const char * a = std::getenv("DFLASH_FP_ALPHA")) {
        float v = (float)std::atof(a);
        if (v > 0.0f && v < 1.0f) fp_cfg.alpha = v;
    }

    // ── ggml graph allocator (reused across all graphs) ───────────────────────
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) { cleanup(); set_last_error("pflash: gallocr failed"); return -1; }

    // ── Pre-reserve gallocr for largest expected graph (fused B + SWA + A) ───
    // This eliminates gallocr reallocation overhead across all ~40 graph builds.
    {
        ggml_init_params ip_reserve{};
        ip_reserve.mem_size = ggml_tensor_overhead() * 512
                            + ggml_graph_overhead_custom(4096, false)
                            + 512 * 1024;
        ip_reserve.no_alloc = true;
        ggml_context * rctx = ggml_init(ip_reserve);
        ggml_cgraph * rgf = ggml_new_graph_custom(rctx, 4096, false);

        // Build a dummy graph sized for the largest expected tensors:
        // A fused [B + SWA + A] graph with SWA_CHUNK tokens.
        // Largest matmuls: FFN (n_ff × n_embd × rc) and attention (n_embd × n_embd × rc).
        const int rc = SWA_CHUNK;

        int64_t n_ff_eff = w.n_ff;
        if (w.n_ff_exp > 0) n_ff_eff = std::max(n_ff_eff, (int64_t)w.n_ff_exp);

        ggml_tensor * dummy_h    = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, n_embd, rc);
        ggml_tensor * dummy_norm = ggml_new_tensor_1d(rctx, GGML_TYPE_F32, n_embd);
        ggml_tensor * dummy_w1   = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, n_embd, n_ff_eff);
        ggml_tensor * dummy_w2   = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, n_ff_eff, n_embd);

        // Chain ops: two FFN passes (representing fused B + SWA) with shared intermediates
        ggml_tensor * t = ggml_rms_norm(rctx, dummy_h, 1e-6f);
        t = ggml_mul(rctx, t, dummy_norm);
        ggml_tensor * g = ggml_mul_mat(rctx, dummy_w1, t);
        ggml_tensor * u = ggml_mul_mat(rctx, dummy_w1, t);
        ggml_tensor * gu = ggml_mul(rctx, g, u);
        t = ggml_mul_mat(rctx, dummy_w2, gu);
        t = ggml_add(rctx, t, dummy_h);

        // Second FFN pass (SWA layer)
        ggml_tensor * t2 = ggml_rms_norm(rctx, t, 1e-6f);
        t2 = ggml_mul(rctx, t2, dummy_norm);
        ggml_tensor * g2 = ggml_mul_mat(rctx, dummy_w1, t2);
        ggml_tensor * u2 = ggml_mul_mat(rctx, dummy_w1, t2);
        ggml_tensor * gu2 = ggml_mul(rctx, g2, u2);
        t2 = ggml_mul_mat(rctx, dummy_w2, gu2);
        t2 = ggml_add(rctx, t2, t);
        ggml_build_forward_expand(rgf, t2);

        ggml_gallocr_reserve(galloc, rgf);
        ggml_free(rctx);
    }

    auto t_start = std::chrono::steady_clock::now();

    // ── Build layer pair list for fused graph execution ───────────────────────
    // Collect indices of full-attn and SWA layers in order.
    // Pairs: (full_il, swa_il) where swa_il immediately follows full_il.
    // Any layers that don't fit this pattern are handled as standalone.
    struct LayerPair { int full_il; int swa_il; };
    std::vector<LayerPair> pairs;
    std::vector<bool> layer_handled(n_layer, false);

    for (int il = 0; il < n_layer - 1; il++) {
        const bool is_swa_il   = (il < (int)w.swa_layers.size()) ? w.swa_layers[il]   : true;
        const bool is_swa_next = ((il+1) < (int)w.swa_layers.size()) ? w.swa_layers[il+1] : true;
        if (!is_swa_il && is_swa_next) {
            pairs.push_back({il, il + 1});
            layer_handled[il]     = true;
            layer_handled[il + 1] = true;
            il++;  // skip the SWA layer since it's paired
        }
    }
    // Any remaining unhandled layers will be processed as standalone below.

    // Helper lambda: get layer KV info
    auto get_layer_kv = [&](int il, ggml_tensor *& out_cache_k, ggml_tensor *& out_cache_v,
                             int & out_n_kv_layer, int & out_kv_idx, bool & out_write_kv) {
        out_n_kv_layer = (!w.head_kv_per_layer.empty() && il < (int)w.head_kv_per_layer.size())
                       ? w.head_kv_per_layer[il] : n_head_kv;
        out_kv_idx   = cache.layer_to_kv_idx[il];
        out_write_kv = (out_kv_idx >= 0);
        const int read_kv_idx = out_write_kv ? out_kv_idx : cache.layer_to_donor_kv[il];
        out_cache_k = (read_kv_idx >= 0) ? cache.attn_k[read_kv_idx] : nullptr;
        out_cache_v = (read_kv_idx >= 0) ? cache.attn_v[read_kv_idx] : nullptr;
    };

    constexpr int kv_start = 0;  // prefill always starts at position 0

    // ── Process each pair (and standalone layers) in order ───────────────────
    // We iterate pairs in order, but also need to handle layers that weren't
    // paired (e.g. two consecutive full-attn layers, trailing full-attn, etc.).
    // Strategy: walk il = 0..n_layer-1, skip layers that were handled in pairs
    // but process pairs when we reach the full_il.

    // Build an ordered processing list: either a pair index or a standalone layer.
    struct ProcessItem {
        bool is_pair;
        int  pair_idx;    // if is_pair
        int  standalone;  // if !is_pair
    };
    std::vector<ProcessItem> process_list;
    {
        int pair_cursor = 0;
        for (int il = 0; il < n_layer; ) {
            if (pair_cursor < (int)pairs.size() && pairs[pair_cursor].full_il == il) {
                process_list.push_back({true, pair_cursor, -1});
                il = pairs[pair_cursor].swa_il + 1;
                pair_cursor++;
            } else if (!layer_handled[il]) {
                process_list.push_back({false, -1, il});
                il++;
            } else {
                il++;  // skip (handled in pair already)
            }
        }
    }

    // ── Main processing loop ──────────────────────────────────────────────────
    for (const auto & item : process_list) {
        if (!item.is_pair) {
            // ── Standalone layer ──────────────────────────────────────────────
            const int il = item.standalone;
            const auto & L = w.layers[il];
            const bool is_swa = (il < (int)w.swa_layers.size()) ? w.swa_layers[il] : true;

            ggml_tensor * cache_k = nullptr;
            ggml_tensor * cache_v = nullptr;
            int n_kv_layer = 0, kv_idx = -1;
            bool write_kv = false;
            get_layer_kv(il, cache_k, cache_v, n_kv_layer, kv_idx, write_kv);
            (void)kv_idx;

            if (is_swa) {
                // ── Standalone SWA layer ──────────────────────────────────────
                for (int cs = 0; cs < S; cs += SWA_CHUNK) {
                    const int cl = std::min(SWA_CHUNK, S - cs);

                    ggml_init_params ip{};
                    ip.mem_size = ggml_tensor_overhead() * 512
                                + ggml_graph_overhead_custom(8192, false)
                                + 512 * 1024;
                    ip.no_alloc = true;
                    ggml_context * gctx = ggml_init(ip);
                    ggml_cgraph  * gf   = ggml_new_graph_custom(gctx, 8192, false);

                    const size_t h_esz  = ggml_element_size(hidden_buf.t);
                    ggml_tensor * h_view = ggml_view_2d(gctx, hidden_buf.t,
                        n_embd, cl, n_embd * h_esz,
                        (size_t)cs * n_embd * h_esz);
                    ggml_tensor * pos_chunk = ggml_view_1d(gctx, pos_buf.t, cl,
                        (size_t)cs * sizeof(int32_t));

                    ChunkCtx cc{S, cs, cl, kv_start, h_view, pos_chunk};
                    ggml_tensor * attn_mask = nullptr;
                    ggml_tensor * cur = build_swa_ops(gctx, gf, w, L, cache,
                        cache_k, cache_v, il, n_kv_layer, h_view, &attn_mask, cc);

                    // Write back residual
                    ggml_build_forward_expand(gf, ggml_cpy(gctx, cur, h_view));

                    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                        cleanup(); ggml_gallocr_free(galloc); ggml_free(gctx);
                        set_last_error("pflash: SWA gallocr failed"); return -1;
                    }

                    // Fill and upload mask
                    {
                        int win_start = 0, win_len = 0;
                        swa_window_bounds(w, cache, cs, cl, kv_start,
                                          cache_k != nullptr, win_start, win_len);
                        fill_swa_mask(attn_mask, win_start, win_len, cs, cl,
                                      kv_start, w.swa_window);
                    }

                    ggml_backend_graph_compute(backend, gf);
                    ggml_free(gctx);
                }
            } else {
                // ── Standalone full-attn layer ────────────────────────────────
                if (!write_kv) {
                    std::fprintf(stderr, "[pflash] layer %d: shared KV (no write), skipping\n", il);
                    continue;
                }

                // Graph A
                for (int cs = 0; cs < S; cs += SWA_CHUNK) {
                    const int cl = std::min(SWA_CHUNK, S - cs);

                    ggml_init_params ipA{};
                    ipA.mem_size = ggml_tensor_overhead() * 256
                                 + ggml_graph_overhead_custom(4096, false)
                                 + 256 * 1024;
                    ipA.no_alloc = true;
                    ggml_context * gA  = ggml_init(ipA);
                    ggml_cgraph  * gfA = ggml_new_graph_custom(gA, 4096, false);

                    const size_t h_esz  = ggml_element_size(hidden_buf.t);
                    ggml_tensor * h_view = ggml_view_2d(gA, hidden_buf.t,
                        n_embd, cl, n_embd * h_esz,
                        (size_t)cs * n_embd * h_esz);
                    ggml_tensor * pos_chunk = ggml_view_1d(gA, pos_buf.t, cl,
                        (size_t)cs * sizeof(int32_t));

                    ChunkCtx cc{S, cs, cl, kv_start, h_view, pos_chunk};
                    build_graph_A_ops(gA, gfA, w, L, cache, cache_k, cache_v,
                                      Q_buf, K_buf, V_buf, il, n_kv_layer, cc);

                    if (!ggml_gallocr_alloc_graph(galloc, gfA)) {
                        cleanup(); ggml_gallocr_free(galloc); ggml_free(gA);
                        set_last_error("pflash: Graph A gallocr failed"); return -1;
                    }
                    ggml_backend_graph_compute(backend, gfA);
                    ggml_free(gA);
                }

                // pFlash
                {
                    int rc = run_pflash(w, cache, backend, Q_buf, K_buf, V_buf, attn_out_buf,
                                        il, S, n_head, n_kv_layer, D, fp_cfg);
                    if (rc != 0) {
                        cleanup(); ggml_gallocr_free(galloc);
                        set_last_error("pflash: flash_prefill failed layer " + std::to_string(il));
                        return -1;
                    }
                }

                // Graph B
                for (int cs = 0; cs < S; cs += SWA_CHUNK) {
                    const int cl = std::min(SWA_CHUNK, S - cs);

                    ggml_init_params ipB{};
                    ipB.mem_size = ggml_tensor_overhead() * 512
                                 + ggml_graph_overhead_custom(8192, false)
                                 + 512 * 1024;
                    ipB.no_alloc = true;
                    ggml_context * gB  = ggml_init(ipB);
                    ggml_cgraph  * gfB = ggml_new_graph_custom(gB, 8192, false);

                    const size_t h_esz  = ggml_element_size(hidden_buf.t);
                    ggml_tensor * h_view = ggml_view_2d(gB, hidden_buf.t,
                        n_embd, cl, n_embd * h_esz,
                        (size_t)cs * n_embd * h_esz);
                    ggml_tensor * pos_chunk = ggml_view_1d(gB, pos_buf.t, cl,
                        (size_t)cs * sizeof(int32_t));

                    ChunkCtx cc{S, cs, cl, kv_start, h_view, pos_chunk};
                    ggml_tensor * cur = build_graph_B_ops(gB, gfB, w, L, cache,
                        attn_out_buf, il, n_kv_layer, cc);

                    ggml_build_forward_expand(gfB, ggml_cpy(gB, cur, h_view));

                    if (!ggml_gallocr_alloc_graph(galloc, gfB)) {
                        cleanup(); ggml_gallocr_free(galloc); ggml_free(gB);
                        set_last_error("pflash: Graph B gallocr failed"); return -1;
                    }
                    ggml_backend_graph_compute(backend, gfB);
                    ggml_free(gB);
                }
            }

            if (il == 0 || il == n_layer - 1 || (il % 10 == 0))
                std::fprintf(stderr, "[pflash] layer %d/%d done\n", il + 1, n_layer);

        } else {
            // ── Fused pair: Graph A(full) → pFlash → fused [B(full) + SWA] ───
            const int full_il = pairs[item.pair_idx].full_il;
            const int swa_il  = pairs[item.pair_idx].swa_il;

            const auto & L_full = w.layers[full_il];
            const auto & L_swa  = w.layers[swa_il];

            ggml_tensor * full_cache_k = nullptr, * full_cache_v = nullptr;
            int full_n_kv = 0, full_kv_idx = -1; bool full_write_kv = false;
            get_layer_kv(full_il, full_cache_k, full_cache_v, full_n_kv, full_kv_idx, full_write_kv);
            (void)full_kv_idx;

            ggml_tensor * swa_cache_k = nullptr, * swa_cache_v = nullptr;
            int swa_n_kv = 0, swa_kv_idx = -1; bool swa_write_kv = false;
            get_layer_kv(swa_il, swa_cache_k, swa_cache_v, swa_n_kv, swa_kv_idx, swa_write_kv);
            (void)swa_kv_idx; (void)swa_write_kv;

            if (!full_write_kv) {
                // Fallback: skip full-attn, process SWA standalone
                std::fprintf(stderr, "[pflash] layer %d: shared KV (no write), skipping\n", full_il);
                // Process SWA standalone
                for (int cs = 0; cs < S; cs += SWA_CHUNK) {
                    const int cl = std::min(SWA_CHUNK, S - cs);

                    ggml_init_params ip{};
                    ip.mem_size = ggml_tensor_overhead() * 512
                                + ggml_graph_overhead_custom(8192, false)
                                + 512 * 1024;
                    ip.no_alloc = true;
                    ggml_context * gctx = ggml_init(ip);
                    ggml_cgraph  * gf   = ggml_new_graph_custom(gctx, 8192, false);

                    const size_t h_esz  = ggml_element_size(hidden_buf.t);
                    ggml_tensor * h_view = ggml_view_2d(gctx, hidden_buf.t,
                        n_embd, cl, n_embd * h_esz,
                        (size_t)cs * n_embd * h_esz);
                    ggml_tensor * pos_chunk = ggml_view_1d(gctx, pos_buf.t, cl,
                        (size_t)cs * sizeof(int32_t));

                    ChunkCtx cc{S, cs, cl, kv_start, h_view, pos_chunk};
                    ggml_tensor * attn_mask = nullptr;
                    ggml_tensor * cur = build_swa_ops(gctx, gf, w, L_swa, cache,
                        swa_cache_k, swa_cache_v, swa_il, swa_n_kv, h_view, &attn_mask, cc);
                    ggml_build_forward_expand(gf, ggml_cpy(gctx, cur, h_view));

                    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                        cleanup(); ggml_gallocr_free(galloc); ggml_free(gctx);
                        set_last_error("pflash: SWA gallocr failed"); return -1;
                    }
                    {
                        int win_start = 0, win_len = 0;
                        swa_window_bounds(w, cache, cs, cl, kv_start,
                                          swa_cache_k != nullptr, win_start, win_len);
                        fill_swa_mask(attn_mask, win_start, win_len, cs, cl,
                                      kv_start, w.swa_window);
                    }
                    ggml_backend_graph_compute(backend, gf);
                    ggml_free(gctx);
                }
                continue;
            }

            // ── Graph A for full_il (chunked, standalone) ─────────────────────
            for (int cs = 0; cs < S; cs += SWA_CHUNK) {
                const int cl = std::min(SWA_CHUNK, S - cs);

                ggml_init_params ipA{};
                ipA.mem_size = ggml_tensor_overhead() * 256
                             + ggml_graph_overhead_custom(4096, false)
                             + 256 * 1024;
                ipA.no_alloc = true;
                ggml_context * gA  = ggml_init(ipA);
                ggml_cgraph  * gfA = ggml_new_graph_custom(gA, 4096, false);

                const size_t h_esz  = ggml_element_size(hidden_buf.t);
                ggml_tensor * h_view = ggml_view_2d(gA, hidden_buf.t,
                    n_embd, cl, n_embd * h_esz,
                    (size_t)cs * n_embd * h_esz);
                ggml_tensor * pos_chunk = ggml_view_1d(gA, pos_buf.t, cl,
                    (size_t)cs * sizeof(int32_t));

                ChunkCtx cc{S, cs, cl, kv_start, h_view, pos_chunk};
                build_graph_A_ops(gA, gfA, w, L_full, cache, full_cache_k, full_cache_v,
                                  Q_buf, K_buf, V_buf, full_il, full_n_kv, cc);

                if (!ggml_gallocr_alloc_graph(galloc, gfA)) {
                    cleanup(); ggml_gallocr_free(galloc); ggml_free(gA);
                    set_last_error("pflash: Graph A gallocr failed"); return -1;
                }
                ggml_backend_graph_compute(backend, gfA);
                ggml_free(gA);
            }

            // ── pFlash for full_il ─────────────────────────────────────────────
            {
                int rc = run_pflash(w, cache, backend, Q_buf, K_buf, V_buf, attn_out_buf,
                                    full_il, S, n_head, full_n_kv, D, fp_cfg);
                if (rc != 0) {
                    cleanup(); ggml_gallocr_free(galloc);
                    set_last_error("pflash: flash_prefill failed layer " + std::to_string(full_il));
                    return -1;
                }
            }

            // ── Fused Graph [B(full_il) + SWA(swa_il)] ───────────────────────
            // The hidden state flows directly from B's output into SWA's input —
            // no write-to-hidden_buf + read-from-hidden_buf between them.
            // Only after SWA is done do we write back to hidden_buf.
            for (int cs = 0; cs < S; cs += SWA_CHUNK) {
                const int cl = std::min(SWA_CHUNK, S - cs);

                // Fused graph is ~2x larger: use 12288 nodes and more context memory.
                ggml_init_params ip{};
                ip.mem_size = ggml_tensor_overhead() * 1024
                            + ggml_graph_overhead_custom(12288, false)
                            + 1024 * 1024;
                ip.no_alloc = true;
                ggml_context * gctx = ggml_init(ip);
                ggml_cgraph  * gf   = ggml_new_graph_custom(gctx, 12288, false);

                const size_t h_esz  = ggml_element_size(hidden_buf.t);
                ggml_tensor * h_view = ggml_view_2d(gctx, hidden_buf.t,
                    n_embd, cl, n_embd * h_esz,
                    (size_t)cs * n_embd * h_esz);
                ggml_tensor * pos_chunk = ggml_view_1d(gctx, pos_buf.t, cl,
                    (size_t)cs * sizeof(int32_t));

                ChunkCtx cc_full{S, cs, cl, kv_start, h_view, pos_chunk};

                // Graph B for full_il: h_view → cur_b
                ggml_tensor * cur_b = build_graph_B_ops(gctx, gf, w, L_full, cache,
                    attn_out_buf, full_il, full_n_kv, cc_full);

                // SWA for swa_il: cur_b → cur_swa (no write to hidden_buf in between)
                ChunkCtx cc_swa{S, cs, cl, kv_start, cur_b, pos_chunk};
                ggml_tensor * attn_mask_swa = nullptr;
                ggml_tensor * cur_swa = build_swa_ops(gctx, gf, w, L_swa, cache,
                    swa_cache_k, swa_cache_v, swa_il, swa_n_kv, cur_b, &attn_mask_swa, cc_swa);

                // Write fused result back to hidden_buf
                ggml_build_forward_expand(gf, ggml_cpy(gctx, cur_swa, h_view));

                if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                    cleanup(); ggml_gallocr_free(galloc); ggml_free(gctx);
                    set_last_error("pflash: fused B+SWA gallocr failed"); return -1;
                }

                // Fill SWA mask (must be done after alloc, before compute)
                {
                    int win_start = 0, win_len = 0;
                    swa_window_bounds(w, cache, cs, cl, kv_start,
                                      swa_cache_k != nullptr, win_start, win_len);
                    fill_swa_mask(attn_mask_swa, win_start, win_len, cs, cl,
                                  kv_start, w.swa_window);
                }

                ggml_backend_graph_compute(backend, gf);
                ggml_free(gctx);
            }

            if (full_il == 0 || swa_il == n_layer - 1 || (swa_il % 10 == 0))
                std::fprintf(stderr, "[pflash] layer %d-%d/%d done\n",
                    full_il + 1, swa_il + 1, n_layer);
        }
    }

    // ── Final: norm + lm_head on last token → argmax ─────────────────────────
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 64
                    + ggml_graph_overhead_custom(512, false)
                    + 64 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);
        ggml_cgraph  * gf   = ggml_new_graph_custom(gctx, 512, false);

        const size_t h_esz  = ggml_element_size(hidden_buf.t);
        ggml_tensor * last_h = ggml_view_2d(gctx, hidden_buf.t,
            n_embd, 1, n_embd * h_esz,
            (size_t)(S - 1) * n_embd * h_esz);

        ggml_tensor * normed = rms_norm_mul(gctx, last_h, w.out_norm, PFLASH_EPS);
        ggml_tensor * logits = ggml_mul_mat(gctx, w.output, normed);

        if (w.logit_softcap > 0.0f) {
            logits = ggml_scale(gctx, logits, 1.0f / w.logit_softcap);
            logits = ggml_tanh(gctx, logits);
            logits = ggml_scale(gctx, logits, w.logit_softcap);
        }

        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            cleanup(); ggml_gallocr_free(galloc); ggml_free(gctx);
            set_last_error("pflash: final gallocr failed"); return -1;
        }
        ggml_backend_graph_compute(backend, gf);

        std::vector<float> logits_cpu(w.n_vocab);
        ggml_backend_tensor_get(logits, logits_cpu.data(), 0, w.n_vocab * sizeof(float));

        int best = 0;
        float best_val = logits_cpu[0];
        for (int i = 1; i < w.n_vocab; i++) {
            if (logits_cpu[i] > best_val) { best_val = logits_cpu[i]; best = i; }
        }

        cache.cur_pos  = S;
        cache.last_tok = best;

        ggml_free(gctx);
    }

    auto t_end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::fprintf(stderr, "[pflash] prefill %d tokens in %.1f ms (%.1f tok/s)\n",
        S, ms, S / (ms / 1000.0));

    ggml_gallocr_free(galloc);
    cleanup();
    return 0;
}

} // namespace dflash27b
