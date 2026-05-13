// Gemma4Backend implementation. See gemma4_backend.h for the contract.
//
// This PR ships the AR-greedy path end-to-end and stubs the DFlash / MTP
// decode loops with TODOs pointing to where the source logic lives. The
// follow-up PR ports those loops verbatim from feature/gemma4-support's
// test_gemma4_dflash.cpp.

#include "gemma4_backend.h"
#include "gemma4_runtime_helpers.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash27b {

// ── ctor / dtor ─────────────────────────────────────────────────────────

Gemma4Backend::Gemma4Backend(const Gemma4BackendArgs & args)
    : args_(args) {}

Gemma4Backend::~Gemma4Backend() {
    shutdown();
}

// ── init ────────────────────────────────────────────────────────────────

bool Gemma4Backend::init() {
    backend_ = ggml_backend_cuda_init(0);
    if (!backend_) {
        std::fprintf(stderr, "[gemma4] ggml_backend_cuda_init failed\n");
        return false;
    }

    if (!load_gemma4_target_gguf(args_.target_path, backend_, target_w_)) {
        std::fprintf(stderr, "[gemma4] load_gemma4_target_gguf: %s\n",
                     dflash27b_last_error());
        return false;
    }

    std::vector<int> extra_q8;
    if (args_.draft_method == Gemma4DraftMethod::kMtp && !args_.mtp_path.empty()) {
        std::vector<bool> mtp_swa;
        if (get_mtp_swa_pattern(args_.mtp_path, mtp_swa) && !mtp_swa.empty()) {
            // Donor layers needed by MTP cross-attention must avoid TQ3/FWHT.
            // resolve_mtp_donor_layers fills this once both target + mtp are loaded.
            // For now we pre-load mtp_w_ to resolve the donor set.
            if (!load_gemma4_mtp_assistant(args_.mtp_path, backend_, mtp_w_)) {
                std::fprintf(stderr, "[gemma4] load_gemma4_mtp_assistant: %s\n",
                             dflash27b_last_error());
                return false;
            }
            mtp_loaded_ = true;
            resolve_mtp_donor_layers(mtp_w_, target_w_.swa_layers);
            for (const auto & L : mtp_w_.layers) {
                if (L.donor_target_layer >= 0) extra_q8.push_back(L.donor_target_layer);
            }
        }
    }

    if (!create_gemma4_cache(target_w_, args_.max_ctx, backend_, cache_,
                             extra_q8, /*target_feat_cap_hint=*/0,
                             /*enable_dflash_capture_overrides=*/args_.draft_enable_capture_overrides)) {
        std::fprintf(stderr, "[gemma4] create_gemma4_cache: %s\n",
                     dflash27b_last_error());
        return false;
    }

    if (args_.draft_method == Gemma4DraftMethod::kMtp) {
        // Enable h_prev capture; allocate the destination tensor in cache_.
        // Allocation lives in the draft/MTP runtime port (follow-up PR).
        cache_.mtp_h_prev_enabled  = true;
        cache_.mtp_last_full_layer = -1;
        for (int il = (int)target_w_.swa_layers.size() - 1; il >= 0; --il) {
            if (!target_w_.swa_layers[il]) { cache_.mtp_last_full_layer = il; break; }
        }
    }

    if (args_.draft_method == Gemma4DraftMethod::kDFlash && !args_.draft_path.empty()) {
        if (!load_gemma4_draft_safetensors(args_.draft_path, backend_, draft_w_) &&
            !load_gemma4_draft_gguf(args_.draft_path, backend_, draft_w_)) {
            std::fprintf(stderr, "[gemma4] load_gemma4_draft_*: %s\n",
                         dflash27b_last_error());
            return false;
        }
        draft_loaded_ = true;
        if (!create_draft_kv_cache(draft_w_, backend_, cache_,
                                   args_.draft_kv_cap_override)) {
            std::fprintf(stderr, "[gemma4] create_draft_kv_cache: %s\n",
                         dflash27b_last_error());
            return false;
        }
    }

    return true;
}

// ── banner ──────────────────────────────────────────────────────────────

void Gemma4Backend::print_ready_banner() const {
    const char * method = "none";
    switch (args_.draft_method) {
        case Gemma4DraftMethod::kDFlash: method = "dflash"; break;
        case Gemma4DraftMethod::kMtp:    method = "mtp";    break;
        case Gemma4DraftMethod::kNone:   method = "none";   break;
    }
    std::printf("[gemma4-daemon] ready n_layer=%d n_embd=%d vocab=%d max_ctx=%d "
                "draft=%s pflash=%d\n",
                target_w_.n_layer, target_w_.n_embd, target_w_.n_vocab,
                args_.max_ctx, method, (int)args_.use_pflash);
    std::fflush(stdout);
}

// ── park / unpark (target only; draft/mtp parking lands with the runtime PR) ──

bool Gemma4Backend::park(const std::string & what) {
    if (what.empty() || what == "all" || what == "target") {
        // Cache + weights remain resident; we just flag the state so the loop
        // refuses to generate. Real park frees GPU buffers — follow-up.
        target_parked_ = true;
        return true;
    }
    std::fprintf(stderr, "[gemma4] park: '%s' not yet implemented\n", what.c_str());
    return false;
}

bool Gemma4Backend::unpark(const std::string & what) {
    if (what.empty() || what == "all" || what == "target") {
        target_parked_ = false;
        return true;
    }
    return false;
}

// ── snapshots (stubbed at the table level; real save/restore in follow-up) ──

bool Gemma4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return false;
    slots_[slot].used    = true;
    slots_[slot].cur_pos = cache_.cur_pos;
    return true;
}

void Gemma4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return;
    slots_[slot].used    = false;
    slots_[slot].cur_pos = -1;
}

bool Gemma4Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= kMaxSlots) return false;
    return slots_[slot].used;
}

int Gemma4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= kMaxSlots) return -1;
    return slots_[slot].cur_pos;
}

GenerateResult Gemma4Backend::restore_and_generate(int /*slot*/,
                                                   const GenerateRequest & /*req*/,
                                                   const DaemonIO & /*io*/) {
    GenerateResult r;
    r.ok    = false;
    r.error = "restore_not_implemented";
    return r;
}

// ── handle_compress / free_drafter (pflash compress lands with the runtime PR) ──

bool Gemma4Backend::handle_compress(const std::string & /*line*/,
                                    const DaemonIO & /*io*/) {
    std::fprintf(stderr, "[gemma4] compress not yet implemented\n");
    return false;
}

void Gemma4Backend::free_drafter() {
    if (draft_loaded_) {
        free_draft_kv_cache(cache_);
        free_gemma4_draft_weights(draft_w_);
        draft_loaded_ = false;
    }
}

// ── generate ────────────────────────────────────────────────────────────

GenerateResult Gemma4Backend::generate(const GenerateRequest & req,
                                       const DaemonIO & io) {
    GenerateResult result;
    const int N = (int)req.prompt.size();
    if (N == 0 || req.n_gen <= 0 || N + req.n_gen > args_.max_ctx) {
        result.error = "overflow";
        return result;
    }

    reset_gemma4_cache(cache_);

    std::vector<float> last_logits;
    if (!prefill(req.prompt, last_logits, result.prefill_s)) {
        result.error = "prefill";
        return result;
    }

    // Inline snapshot if requested
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= N) {
        if (snapshot_save(req.snap_slot)) {
            slots_[req.snap_slot].cur_pos = req.snap_pos;
        }
    }

    std::vector<int32_t> tokens;
    bool ok = false;
    switch (args_.draft_method) {
        case Gemma4DraftMethod::kDFlash:
            ok = decode_dflash(req.n_gen, last_logits, req, io, tokens, result.decode_s);
            break;
        case Gemma4DraftMethod::kMtp:
            ok = decode_mtp(req.n_gen, last_logits, req, io, tokens, result.decode_s);
            break;
        case Gemma4DraftMethod::kNone:
        default:
            ok = decode_autoregressive(req.n_gen, last_logits, req, io, tokens, result.decode_s);
            break;
    }

    if (!ok) {
        result.error = "decode";
        return result;
    }
    result.ok     = true;
    result.tokens = std::move(tokens);
    return result;
}

// ── shutdown ────────────────────────────────────────────────────────────

void Gemma4Backend::shutdown() {
    free_drafter();
    if (mtp_loaded_) {
        free_gemma4_mtp_assistant(mtp_w_);
        mtp_loaded_ = false;
    }
    free_gemma4_cache(cache_);
    free_gemma4_target_weights(target_w_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

// ── internal: prefill ────────────────────────────────────────────────────
//
// Runs chunked prefill via build_gemma4_graph; returns last-token logits.
// Real implementation involves ggml_init / ggml_new_graph / build_gemma4_graph
// / ggml_backend_graph_compute. Stubbed here with a clear marker; the AR
// decode loop below depends on this and will be exercised once the body
// lands in the runtime follow-up PR.

bool Gemma4Backend::prefill(const std::vector<int32_t> & prompt,
                            std::vector<float> & out_last_logits,
                            double & out_prefill_s) {
    const int n_prompt   = (int)prompt.size();
    const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;
    const int chunk_size = std::min(n_prompt, swa_window);

    StepGraph sg{};
    bool ok = true;

    auto t0 = std::chrono::steady_clock::now();

    for (int cs = 0; cs < n_prompt; cs += chunk_size) {
        const int chunk_n  = std::min(chunk_size, n_prompt - cs);
        const bool is_last = (cs + chunk_n == n_prompt);
        const bool need_mask = (cs + chunk_n > 1);

        if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                               cs, chunk_n, need_mask,
                               /*capture=*/true,
                               args_.use_pflash, args_.pflash_alpha,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/true)) {
            std::fprintf(stderr, "[gemma4] prefill build failed at %d\n", cs);
            ok = false;
            break;
        }

        if (!embed_tokens_batch(target_w_, prompt.data() + cs, chunk_n,
                                sg.inp_embed, backend_)) {
            std::fprintf(stderr, "[gemma4] embed_tokens_batch failed\n");
            ok = false;
            break;
        }

        {
            std::vector<int32_t> pos(chunk_n);
            for (int i = 0; i < chunk_n; i++) pos[i] = cs + i;
            ggml_backend_tensor_set(sg.positions, pos.data(), 0,
                                    sizeof(int32_t) * chunk_n);
        }

        if (sg.attn_mask && sg.attn_mask->buffer) {
            const int kv_len = cs + chunk_n;
            std::vector<uint16_t> mask_buf;
            build_causal_mask(mask_buf, kv_len, chunk_n, cs);
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        if (sg.swa_mask && sg.swa_mask->buffer) {
            const SwaView swa_view = compute_swa_view(cs, chunk_n,
                                                       swa_window, cache_.swa_ctx_alloc);
            std::vector<uint16_t> swa_buf;
            build_swa_causal_mask(swa_buf,
                                  /*kv_start*/ cs,
                                  /*n_tokens*/ chunk_n,
                                  /*swa_window*/ swa_window,
                                  /*ring_size*/ swa_view.effective_win_len,
                                  /*kv_end*/ cs + chunk_n);
            ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                    sizeof(uint16_t) * swa_buf.size());
        }

        auto st = ggml_backend_graph_compute(backend_, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[gemma4] prefill compute failed at %d\n", cs);
            ok = false;
            break;
        }

        cache_.cur_pos = cs + chunk_n;

        if (is_last) {
            const int vocab = target_w_.n_vocab;
            out_last_logits.resize(vocab);
            ggml_backend_tensor_get(sg.logits, out_last_logits.data(),
                                    0, sizeof(float) * vocab);
        }

        step_graph_free(sg);
    }

    auto t1 = std::chrono::steady_clock::now();
    out_prefill_s = std::chrono::duration<double>(t1 - t0).count();

    step_graph_free(sg);
    return ok && !out_last_logits.empty();
}

// ── internal: AR decode ─────────────────────────────────────────────────

bool Gemma4Backend::decode_autoregressive(int n_gen,
                                          std::vector<float> & last_logits_io,
                                          const GenerateRequest & req,
                                          const DaemonIO & io,
                                          std::vector<int32_t> & out_tokens,
                                          double & out_decode_s) {
    out_tokens.clear();
    out_tokens.reserve(n_gen);

    // history for repetition penalty — start with the full prompt
    std::vector<int32_t> history(req.prompt.begin(), req.prompt.end());

    const bool do_sample = req.sampler.temp > 0.0f;

    auto argmax = [](const std::vector<float> & ll) {
        int best = 0; float bv = ll[0];
        for (size_t i = 1; i < ll.size(); ++i)
            if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
        return best;
    };

    auto pick = [&](const std::vector<float> & ll) -> int {
        return do_sample
            ? sample_logits(ll.data(), (int)ll.size(), req.sampler, history, sampler_rng_)
            : argmax(ll);
    };

    // First token sampled from prefill logits
    int32_t cur_tok = (int32_t)pick(last_logits_io);

    StepGraph sg{};
    bool ok = true;

    auto t0 = std::chrono::steady_clock::now();

    for (int s = 0; s < n_gen; ++s) {
        // EOS check
        if ((target_w_.eos_id      >= 0 && cur_tok == target_w_.eos_id) ||
            (target_w_.eos_chat_id >= 0 && cur_tok == target_w_.eos_chat_id)) {
            break;
        }

        out_tokens.push_back(cur_tok);
        history.push_back(cur_tok);

        if (req.stream) {
            io.emit(cur_tok);
        }

        const int committed = cache_.cur_pos;

        if (committed >= args_.max_ctx - 2) {
            break;
        }

        if (!build_gemma4_step(sg, target_w_, cache_, backend_,
                               committed, /*n_tokens=*/1,
                               /*with_mask=*/true,
                               /*capture=*/false,
                               /*use_pflash=*/false, args_.pflash_alpha,
                               /*fa_window=*/0)) {
            std::fprintf(stderr, "[gemma4] AR build failed at step %d\n", s);
            ok = false;
            break;
        }

        if (sg.attn_mask && sg.attn_mask->buffer) {
            const int kv_len = committed + 1;
            std::vector<uint16_t> mask_buf;
            build_causal_mask(mask_buf, kv_len, 1, committed);
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        if (sg.swa_mask && sg.swa_mask->buffer) {
            const int swa_window = target_w_.swa_window > 0 ? target_w_.swa_window : 1024;
            const SwaView swa_view = compute_swa_view(committed, 1,
                                                       swa_window, cache_.swa_ctx_alloc);
            std::vector<uint16_t> swa_buf;
            build_swa_causal_mask(swa_buf,
                                  /*kv_start*/ committed,
                                  /*n_tokens*/ 1,
                                  /*swa_window*/ swa_window,
                                  /*ring_size*/ swa_view.effective_win_len,
                                  /*kv_end*/ committed + 1);
            ggml_backend_tensor_set(sg.swa_mask, swa_buf.data(), 0,
                                    sizeof(uint16_t) * swa_buf.size());
        }

        if (!embed_token(target_w_, cur_tok, sg.inp_embed, backend_)) {
            std::fprintf(stderr, "[gemma4] embed_token failed for tok=%d\n", cur_tok);
            ok = false;
            break;
        }

        {
            int32_t pos_val = committed;
            ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));
        }

        {
            auto st = ggml_backend_graph_compute(backend_, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[gemma4] AR compute failed at step %d\n", s);
                ok = false;
                break;
            }
        }

        cache_.cur_pos = committed + 1;

        const int vocab = target_w_.n_vocab;
        std::vector<float> logits_cpu(vocab);
        ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                sizeof(float) * vocab);
        cur_tok = (int32_t)pick(logits_cpu);

        step_graph_free(sg);
    }

    auto t1 = std::chrono::steady_clock::now();
    out_decode_s = std::chrono::duration<double>(t1 - t0).count();

    step_graph_free(sg);

    if (req.stream) {
        io.emit(-1);
    }

    return ok;
}

// ── internal: DFlash decode (TODO in runtime follow-up PR) ──────────────

bool Gemma4Backend::decode_dflash(int /*n_gen*/,
                                  std::vector<float> & /*last_logits_io*/,
                                  const GenerateRequest & /*req*/,
                                  const DaemonIO & /*io*/,
                                  std::vector<int32_t> & /*out_tokens*/,
                                  double & out_decode_s) {
    out_decode_s = 0.0;
    std::fprintf(stderr,
                 "[gemma4] DFlash decode TODO — port "
                 "build_draft_kv_prefill_graph + build_gemma4_draft_graph + "
                 "DDTree verify from feature/gemma4-support "
                 "test_gemma4_dflash.cpp:~1400-2400\n");
    return false;
}

// ── internal: MTP decode (TODO in runtime follow-up PR) ─────────────────

bool Gemma4Backend::decode_mtp(int /*n_gen*/,
                               std::vector<float> & /*last_logits_io*/,
                               const GenerateRequest & /*req*/,
                               const DaemonIO & /*io*/,
                               std::vector<int32_t> & /*out_tokens*/,
                               double & out_decode_s) {
    out_decode_s = 0.0;
    std::fprintf(stderr,
                 "[gemma4] MTP decode TODO — port build_mtp_step_graph + "
                 "γ=1 verify path from feature/gemma4-support "
                 "test_gemma4_dflash.cpp:~2400-2800\n");
    return false;
}

}  // namespace dflash27b
