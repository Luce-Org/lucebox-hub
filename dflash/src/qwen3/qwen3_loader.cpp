// GGUF loader for Qwen3-0.6B drafter. Reads weights from a BF16 GGUF file
// produced by `convert_hf_to_gguf.py Qwen/Qwen3-0.6B`. Sets up ggml tensors
// on the requested backend.
//
// Tensor layout (verified via gguf reader):
//   token_embd.weight                 BF16 [hidden=1024, vocab=151936]
//   output_norm.weight                F32  [hidden]
//   output.weight                     BF16 [hidden, vocab] (lm_head)
//
//   blk.<i>.attn_norm.weight          F32  [hidden]
//   blk.<i>.attn_q.weight             BF16 [hidden, q_dim=2048]
//   blk.<i>.attn_k.weight             BF16 [hidden, kv_dim=1024]
//   blk.<i>.attn_v.weight             BF16 [hidden, kv_dim]
//   blk.<i>.attn_output.weight        BF16 [q_dim, hidden]
//   blk.<i>.attn_q_norm.weight        F32  [head_dim=128]
//   blk.<i>.attn_k_norm.weight        F32  [head_dim]
//   blk.<i>.ffn_norm.weight           F32  [hidden]
//   blk.<i>.ffn_gate.weight           BF16 [hidden, ffn=3072]
//   blk.<i>.ffn_up.weight             BF16 [hidden, ffn]
//   blk.<i>.ffn_down.weight           BF16 [ffn, hidden]
//
// We mmap the GGUF file and copy each tensor's bytes to the backend buffer
// (mirrors the dflash gguf_target_loader pattern).

#include "qwen3_drafter_model.h"
#include "slim_drafter.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

bool copy_tensor_from_file(gguf_context * gctx, const char * name,
                           const void * mmap_base, size_t data_offset,
                           ggml_tensor * dst) {
    int idx = gguf_find_tensor(gctx, name);
    if (idx < 0) {
        std::fprintf(stderr, "[qwen3-0.6b] missing tensor: %s\n", name);
        return false;
    }
    const size_t off = gguf_get_tensor_offset(gctx, idx);
    const size_t bytes = ggml_nbytes(dst);
    const uint8_t * src = (const uint8_t *)mmap_base + data_offset + off;
    ggml_backend_tensor_set(dst, src, 0, bytes);
    return true;
}

uint32_t get_u32(gguf_context * g, const char * key, uint32_t def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_u32(g, k);
}

float get_f32(gguf_context * g, const char * key, float def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_f32(g, k);
}

} // namespace

// Detect arch prefix from general.architecture; returns true if arch has QK-norm.
static bool arch_has_qk_norm(gguf_context * gctx, std::string & out_prefix) {
    int k = gguf_find_key(gctx, "general.architecture");
    std::string arch = (k >= 0) ? gguf_get_val_str(gctx, k) : "qwen3";
    if (arch == "llama" || arch == "qwen2") {
        out_prefix = arch + ".";
        return false;
    }
    out_prefix = "qwen3.";
    return true;
}

bool load_qwen3_drafter_model(const std::string & path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out) {
    out.backend = backend;

    gguf_init_params iparams{ /*no_alloc=*/ false, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Detect architecture; set key prefix and QK-norm presence.
    std::string pfx;
    const bool has_qk_norm = arch_has_qk_norm(gctx, pfx);
    std::fprintf(stderr, "[drafter-loader] arch prefix=%s qk_norm=%s\n",
                 pfx.c_str(), has_qk_norm ? "yes" : "no");
    std::fflush(stderr);

    // Per-arch fallback defaults (qwen3 defaults for backward compat).
    const bool is_llama_family = (pfx == "llama." || pfx == "qwen2.");
    const uint32_t def_n_embd    = is_llama_family ?  576 : 1024;
    const uint32_t def_n_ff      = is_llama_family ? 1536 : 3072;
    const uint32_t def_n_head    = is_llama_family ?    9 :   16;
    const uint32_t def_n_head_kv = is_llama_family ?    3 :    8;
    const uint32_t def_n_layer   = is_llama_family ?   30 :   28;
    const uint32_t def_n_ctx     = is_llama_family ? 8192 : 40960;
    const uint32_t def_head_dim  = is_llama_family ?   64 :  128;
    const uint32_t def_n_vocab   = is_llama_family ? 49152 : 151936;
    const float    def_rope      = is_llama_family ? 100000.0f : 1000000.0f;

    out.n_embd     = (int)get_u32(gctx, (pfx + "embedding_length").c_str(),        def_n_embd);
    out.n_ff       = (int)get_u32(gctx, (pfx + "feed_forward_length").c_str(),      def_n_ff);
    out.n_head     = (int)get_u32(gctx, (pfx + "attention.head_count").c_str(),     def_n_head);
    out.n_head_kv  = (int)get_u32(gctx, (pfx + "attention.head_count_kv").c_str(),  def_n_head_kv);
    out.n_layer    = (int)get_u32(gctx, (pfx + "block_count").c_str(),              def_n_layer);
    out.n_ctx_max  = (int)get_u32(gctx, (pfx + "context_length").c_str(),           def_n_ctx);
    out.head_dim   = (int)get_u32(gctx, (pfx + "attention.key_length").c_str(),     def_head_dim);
    out.n_vocab    = (int)get_u32(gctx, (pfx + "vocab_size").c_str(),               def_n_vocab);
    out.rope_theta = get_f32(gctx,      (pfx + "rope.freq_base").c_str(),            def_rope);

    // Detect weight quant type from blk.0.attn_q.weight; support BF16 and Q8_0.
    ggml_type wtype = GGML_TYPE_BF16;
    {
        int tidx = gguf_find_tensor(gctx, "blk.0.attn_q.weight");
        if (tidx >= 0) {
            // gguf_context created with no_alloc=false builds a ggml_context
            // internally; use gguf_get_tensor_type to read the stored type.
            // Fallback: check the tensor size to distinguish Q8_0 from BF16.
            // BF16 blk.0.attn_q is [1024, 2048] = 2097152 elements * 2 bytes = 4194304 B
            // Q8_0 blk.0.attn_q is 2097152 * 1.0625 bytes = 2228224 B (32 bytes per 32 elems + 2 byte scale)
            size_t tsz = gguf_get_tensor_size(gctx, tidx);
            if (tsz == 2228224) {
                wtype = GGML_TYPE_Q8_0;
            }
        }
    }
    std::fprintf(stderr, "[qwen3-0.6b] detected weight type: %s\n",
                 wtype == GGML_TYPE_Q8_0 ? "Q8_0" : "BF16");
    std::fflush(stderr);

    // PFLASH_DRAFTER_SLIM=1: skip allocating tensors that are unreachable in
    // pflash+ee7 mode. Dead tensors: output.weight (lm_head, ~310 MB) and all
    // layers >= DFLASH_DRAFTER_EARLY_EXIT_N (~570 MB for 21 layers). Total
    // savings ~880 MB vs full load. When slim is off, behavior is identical.
    const bool slim_mode = []() -> bool {
        const char * v = std::getenv("PFLASH_DRAFTER_SLIM");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    // Resolve forward-pass layer limit (mirrors qwen3_graph.cpp logic exactly).
    // We read it once at load so slim knows how many layers to allocate.
    const int fwd_limit = [&]() -> int {
        const char * e = std::getenv("DFLASH_DRAFTER_EARLY_EXIT_N");
        if (e) {
            int v = std::atoi(e);
            if (v > 0 && v < out.n_layer) return v;
        }
        return out.n_layer;
    }();
    // Number of layers to actually allocate on GPU.
    const int n_alloc_layers = slim_mode ? fwd_limit : out.n_layer;

    if (slim_mode) {
        const int64_t dead_layers_mb = (int64_t)(out.n_layer - n_alloc_layers) * 29; // ~29 MB/layer
        const int64_t output_mb = (int64_t)out.n_embd * out.n_vocab * 2 / (1024 * 1024);
        std::fprintf(stderr,
            "[drafter-loader] PFLASH_DRAFTER_SLIM=1: allocating %d/%d layers, "
            "skipping output.weight. Saves ~%ld MB vs full load.\n",
            n_alloc_layers, out.n_layer,
            (long)(dead_layers_mb + output_mb));
        std::fflush(stderr);
    }

    // Compute total tensor metadata size for context allocation.
    // Qwen3 has 11 tensors/layer (includes q_norm + k_norm); llama-family has 9.
    const int n_layer = out.n_layer;
    const int n_tensors_per_layer = has_qk_norm ? 11 : 9;
    // Slim: only n_alloc_layers layers; no output.weight (2 top tensors instead of 3).
    const int n_top_tensors = slim_mode ? 2 : 3;
    const int total_tensors = n_top_tensors + n_alloc_layers * n_tensors_per_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * total_tensors + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);

    const int n_embd = out.n_embd;
    const int n_ff   = out.n_ff;
    const int n_head = out.n_head;
    const int n_head_kv = out.n_head_kv;
    const int head_dim  = out.head_dim;
    const int n_vocab   = out.n_vocab;
    const int q_dim     = n_head * head_dim;
    const int kv_dim    = n_head_kv * head_dim;

    // Top-level tensors. tok_embd/output use wtype; norms stay F32.
    // Slim: tok_embd and out_norm always allocated; output.weight only when !slim.
    out.tok_embd = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_vocab);
    out.out_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_name(out.tok_embd, "token_embd.weight");
    ggml_set_name(out.out_norm, "output_norm.weight");
    if (!slim_mode) {
        out.output = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_vocab);
        ggml_set_name(out.output, "output.weight");
    }
    // out.output stays nullptr in slim mode; forward_qwen3_drafter_model
    // never dereferences it (it doesn't call lm_head in scoring mode).

    // Allocate tensor metadata for active layers [0, n_alloc_layers).
    // Layers [n_alloc_layers, n_layer) keep all-nullptr DraftLayer fields
    // (safe: fwd_layer_limit in qwen3_graph.cpp never exceeds n_alloc_layers
    //  when DFLASH_DRAFTER_EARLY_EXIT_N matches).
    out.layers.resize(n_layer);
    for (int il = 0; il < n_alloc_layers; ++il) {
        auto & L = out.layers[il];
        L.attn_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.wq        = ggml_new_tensor_2d(out.ctx, wtype, n_embd, q_dim);
        L.wk        = ggml_new_tensor_2d(out.ctx, wtype, n_embd, kv_dim);
        L.wv        = ggml_new_tensor_2d(out.ctx, wtype, n_embd, kv_dim);
        L.wo        = ggml_new_tensor_2d(out.ctx, wtype, q_dim, n_embd);
        if (has_qk_norm) {
            L.q_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
            L.k_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        }
        // else q_norm/k_norm stay nullptr (llama-family has no per-head QK-norm)
        L.ffn_norm  = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.ffn_gate  = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_ff);
        L.ffn_up    = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_ff);
        L.ffn_down  = ggml_new_tensor_2d(out.ctx, wtype, n_ff, n_embd);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for Qwen3-0.6B drafter");
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // mmap the GGUF data section.
    const size_t data_off = gguf_get_data_offset(gctx);
#if defined(_WIN32)
    std::wstring wpath;
    {
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) {
            set_last_error("MultiByteToWideChar failed for " + path);
            gguf_free(gctx);
            return false;
        }
        wpath.resize(wlen - 1);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    }
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        set_last_error("CreateFileW failed for " + path);
        gguf_free(gctx);
        return false;
    }
    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMapping) {
        set_last_error("CreateFileMappingA failed for " + path);
        gguf_free(gctx);
        return false;
    }
    void * mm = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!mm) {
        set_last_error("MapViewOfFile failed for " + path);
        gguf_free(gctx);
        return false;
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st; ::fstat(fd, &st);
    void * mm = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        set_last_error("mmap failed for " + path);
        gguf_free(gctx);
        return false;
    }
#endif

    bool ok = true;
    ok &= copy_tensor_from_file(gctx, "token_embd.weight", mm, data_off, out.tok_embd);
    ok &= copy_tensor_from_file(gctx, "output_norm.weight", mm, data_off, out.out_norm);
    // output.weight (lm_head): skipped in slim mode (never used by the drafter scorer).
    if (!slim_mode) {
        if (gguf_find_tensor(gctx, "output.weight") >= 0) {
            ok &= copy_tensor_from_file(gctx, "output.weight", mm, data_off, out.output);
        } else {
            // Tied weights: copy tok_embd data into output tensor
            std::vector<uint8_t> tmp(ggml_nbytes(out.tok_embd));
            ggml_backend_tensor_get(out.tok_embd, tmp.data(), 0, tmp.size());
            ggml_backend_tensor_set(out.output, tmp.data(), 0, tmp.size());
        }
    }
    // Load layer weights. Slim: only active layers [0, n_alloc_layers).
    char nm[128];
    for (int il = 0; il < n_alloc_layers; ++il) {
        const auto & L = out.layers[il];
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight",   il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wq);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wk);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wv);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wo);
        if (has_qk_norm) {
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.q_norm);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.k_norm);
        }
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_gate);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_up);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_down);
    }
#if defined(_WIN32)
    UnmapViewOfFile(mm);
#else
    ::munmap(mm, st.st_size);
#endif
    gguf_free(gctx);

    if (!ok) {
        set_last_error("one or more Qwen3-0.6B tensors failed to load");
        ggml_backend_buffer_free(out.buf);
        ggml_free(out.ctx);
        out.buf = nullptr;
        out.ctx = nullptr;
        return false;
    }
    return true;
}

void free_qwen3_drafter_model(Qwen3DrafterWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
    w.tok_embd = w.out_norm = w.output = nullptr;
    w.backend = nullptr;
}

} // namespace dflash::common
