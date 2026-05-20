// dflash_server — native C++ HTTP server for dflash27b.
//
// Replaces the Python server.py for production use. Owns the ModelBackend
// directly (no subprocess, no pipe protocol), enabling:
//   - Immediate client-disconnect cancellation (via send() failure)
//   - Lower latency (no IPC overhead)
//   - Single binary deployment
//
// Usage:
//   dflash_server <model.gguf> [--draft <draft.gguf>] [--port 8080]
//                              [--host 0.0.0.0] [--max-ctx 131072]
//                              [--max-tokens 4096] [--gpu 0]

#include "http_server.h"
#include "common/backend_factory.h"
#include "common/gguf_inspect.h"
#include "common/peer_access.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace dflash27b;

static bool parse_int_list(const char * value, std::vector<int> & out) {
    out.clear();
    if (!value || !*value) return false;
    const char * p = value;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) return false;
        out.push_back((int)v);
        if (*end == '\0') return true;
        if (*end != ',') return false;
        p = end + 1;
        if (!*p) return false;
    }
    return !out.empty();
}

static bool parse_double_list(const char * value, std::vector<double> & out) {
    out.clear();
    if (!value || !*value) return false;
    const char * p = value;
    while (*p) {
        char * end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) return false;
        out.push_back(v);
        if (*end == '\0') return true;
        if (*end != ',') return false;
        p = end + 1;
        if (!*p) return false;
    }
    return !out.empty();
}

static bool parse_backend_arg(const char * value, PlacementBackend & out) {
    return value && parse_placement_backend(value, out);
}

static bool validate_server_placement(const BackendArgs & bargs) {
    const PlacementBackend compiled = compiled_placement_backend();
    if (!placement_backend_supported(bargs.device.backend)) {
        std::fprintf(stderr,
            "[server] --target-backend=%s is unsupported in this binary "
            "(compiled backend: %s)\n",
            placement_backend_name(bargs.device.backend),
            placement_backend_name(compiled));
        return false;
    }
    if (!placement_backend_supported(bargs.draft_backend)) {
        std::fprintf(stderr,
            "[server] --draft-backend=%s is unsupported in this binary "
            "(compiled backend: %s)\n",
            placement_backend_name(bargs.draft_backend),
            placement_backend_name(compiled));
        return false;
    }
    const PlacementBackend target = bargs.device.backend == PlacementBackend::Auto
        ? compiled : bargs.device.backend;
    const PlacementBackend draft = bargs.draft_backend == PlacementBackend::Auto
        ? target : bargs.draft_backend;
    if (target != draft) {
        std::fprintf(stderr,
            "[server] mixed target/draft backends are not implemented in the "
            "native server yet (target=%s draft=%s)\n",
            placement_backend_name(target), placement_backend_name(draft));
        return false;
    }
    if (!bargs.device.layer_split_gpus.empty()) {
        std::fprintf(stderr,
            "[server] target layer split is not implemented in the native "
            "server yet (--target-gpus was provided)\n");
        return false;
    }
    if (!bargs.device.layer_split_weights.empty()) {
        std::fprintf(stderr,
            "[server] --target-layer-split requires native target layer split "
            "support, which is not implemented yet\n");
        return false;
    }
    return true;
}

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --draft <path>       Draft model for speculative decode (qwen35 only)\n"
        "  --port <N>           Listen port (default: 8080)\n"
        "  --host <addr>        Bind address (default: 0.0.0.0)\n"
        "  --max-ctx <N>        Max context length (default: 131072)\n"
        "  --max-tokens <N>     Default max output tokens (default: 4096)\n"
        "  --gpu <N>            Target GPU device (default: 0)\n"
        "  --draft-gpu <N>      Draft GPU device (default: 0)\n"
        "  --target-backend <auto|cuda|hip>  Target backend (default: auto)\n"
        "  --draft-backend <auto|cuda|hip>   Draft backend (default: auto)\n"
        "  --target-gpus <ids>  Reserved target layer-split GPU list (comma-separated)\n"
        "  --target-layer-split <weights>  Reserved layer-split weights\n"
        "  --peer-access        Enable peer access for multi-GPU placement\n"
        "  --chunk <N>          Chunked-prefill chunk size (default: 512)\n"
        "  --fa-window <N>     Flash-attention sliding window (default: 0=full)\n"
        "  --model-name <name>  Model name for /v1/models (default: dflash)\n"
        "  --ddtree             Enable DDTree speculative decode\n"
        "  --ddtree-budget <N>  DDTree budget (default: 64)\n"
        "  --no-cors            Disable CORS headers\n"
        "\n"
        "KV cache:\n"
        "  --cache-type-k <type>  KV cache K type (f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0,tq3_0)\n"
        "  --cache-type-v <type>  KV cache V type (same choices as above)\n"
        "                         Default: tq3_0 when max_ctx>6144, else q4_0\n"
        "\n"
        "PFlash (speculative prefill compression):\n"
        "  --prefill-compression off|auto|always  (default: off)\n"
        "  --prefill-threshold <N>     Token threshold for auto mode (default: 32000)\n"
        "  --prefill-keep-ratio <F>    Fraction of tokens to keep (default: 0.05)\n"
        "  --prefill-drafter <path>    Drafter GGUF for compression (Qwen3-0.6B)\n"
        "  --prefill-skip-park         Skip park/unpark (for >=32GB GPUs)\n"
        "\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '-') {
        print_usage(argv[0]);
        return 2;
    }

    // Parse arguments.
    BackendArgs bargs;
    ServerConfig sconfig;
    bargs.model_path = argv[1];
    std::string cache_type_k;  // explicit --cache-type-k override
    std::string cache_type_v;  // explicit --cache-type-v override

    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            bargs.draft_path = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            sconfig.port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            sconfig.host = argv[++i];
        } else if (std::strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            sconfig.max_ctx = v;
            bargs.device.max_ctx = v;
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            sconfig.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            bargs.device.gpu = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0 && i + 1 < argc) {
            bargs.draft_gpu = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--target-backend") == 0 && i + 1 < argc) {
            if (!parse_backend_arg(argv[++i], bargs.device.backend)) {
                std::fprintf(stderr, "[server] bad --target-backend value\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--draft-backend") == 0 && i + 1 < argc) {
            if (!parse_backend_arg(argv[++i], bargs.draft_backend)) {
                std::fprintf(stderr, "[server] bad --draft-backend value\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--target-gpus") == 0 && i + 1 < argc) {
            if (!parse_int_list(argv[++i], bargs.device.layer_split_gpus)) {
                std::fprintf(stderr, "[server] bad --target-gpus value\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--target-layer-split") == 0 && i + 1 < argc) {
            if (!parse_double_list(argv[++i], bargs.device.layer_split_weights)) {
                std::fprintf(stderr, "[server] bad --target-layer-split value\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--peer-access") == 0) {
            bargs.device.peer_access = true;
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            bargs.chunk = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--fa-window") == 0 && i + 1 < argc) {
            bargs.fa_window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--model-name") == 0 && i + 1 < argc) {
            sconfig.model_name = argv[++i];
        } else if (std::strcmp(argv[i], "--ddtree") == 0) {
            bargs.ddtree_mode = true;
            bargs.fast_rollback = true;
        } else if (std::strcmp(argv[i], "--ddtree-budget") == 0 && i + 1 < argc) {
            bargs.ddtree_budget = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-cors") == 0) {
            sconfig.enable_cors = false;
        } else if (std::strcmp(argv[i], "--prefill-compression") == 0 && i + 1 < argc) {
            const char * mode = argv[++i];
            if (std::strcmp(mode, "auto") == 0)
                sconfig.pflash_mode = ServerConfig::PflashMode::AUTO;
            else if (std::strcmp(mode, "always") == 0)
                sconfig.pflash_mode = ServerConfig::PflashMode::ALWAYS;
            else {
                std::fprintf(stderr, "[server] unknown --prefill-compression mode: '%s' (expected: auto, always, off)\n", mode);
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--prefill-threshold") == 0 && i + 1 < argc) {
            sconfig.pflash_threshold = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prefill-keep-ratio") == 0 && i + 1 < argc) {
            sconfig.pflash_keep_ratio = (float)std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--prefill-drafter") == 0 && i + 1 < argc) {
            sconfig.pflash_drafter_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prefill-skip-park") == 0) {
            sconfig.pflash_skip_park = true;
        } else if (std::strcmp(argv[i], "--cache-type-k") == 0 && i + 1 < argc) {
            cache_type_k = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-type-v") == 0 && i + 1 < argc) {
            cache_type_v = argv[++i];
        } else {
            std::fprintf(stderr, "[server] unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!validate_server_placement(bargs)) return 2;

    // Sync max_ctx: if --max-ctx was not provided, use the backend's default.
    // This prevents the HTTP server from accepting prompts larger than the
    // KV cache the backend actually allocates.
    if (sconfig.max_ctx <= 0) {
        sconfig.max_ctx = bargs.device.max_ctx;
    }

    // ── Apply environment defaults (mirrors server.py logic) ────────────
    // Explicit --cache-type-k/v override via env vars.
    if (!cache_type_k.empty()) {
        setenv("DFLASH27B_KV_K", cache_type_k.c_str(), 1);
    }
    if (!cache_type_v.empty()) {
        setenv("DFLASH27B_KV_V", cache_type_v.c_str(), 1);
    }

    // Auto-select TQ3_0 KV cache for large contexts (saves ~40% VRAM).
    // Q4_0 remains default for short contexts where quality matters more.
    if (sconfig.max_ctx > 6144 && cache_type_k.empty() && cache_type_v.empty()) {
        setenv("DFLASH27B_KV_TQ3", "1", 0);  // don't overwrite user env
    }

    // PFlash performance defaults: BSA kernel + sparse alpha + full attention window.
    bool pflash_enabled = (sconfig.pflash_mode != ServerConfig::PflashMode::OFF);
    if (pflash_enabled) {
        setenv("DFLASH_FP_USE_BSA", "1", 0);
        setenv("DFLASH_FP_ALPHA", "0.85", 0);
        setenv("DFLASH27B_FA_WINDOW", "0", 0);
    }

    // Load tokenizer.
    std::fprintf(stderr, "[server] loading tokenizer from %s\n", bargs.model_path);
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(bargs.model_path)) {
        std::fprintf(stderr, "[server] tokenizer load failed\n");
        return 1;
    }

    // Load pflash drafter tokenizer (if pflash enabled).
    Tokenizer drafter_tokenizer;
    if (pflash_enabled) {
        if (sconfig.pflash_drafter_path.empty()) {
            std::fprintf(stderr, "[server] --prefill-compression requires --prefill-drafter\n");
            return 1;
        }
        std::fprintf(stderr, "[server] loading pflash drafter tokenizer from %s\n",
                     sconfig.pflash_drafter_path.c_str());
        if (!drafter_tokenizer.load_from_gguf(sconfig.pflash_drafter_path.c_str())) {
            std::fprintf(stderr, "[server] drafter tokenizer load failed\n");
            return 1;
        }
        std::fprintf(stderr, "[server] pflash: mode=%s threshold=%d keep=%.3f skip_park=%d\n",
                     sconfig.pflash_mode == ServerConfig::PflashMode::AUTO ? "auto" : "always",
                     sconfig.pflash_threshold, sconfig.pflash_keep_ratio,
                     (int)sconfig.pflash_skip_park);
    }

    // Create backend.
    g_peer_access_opt_in = bargs.device.peer_access;
    std::fprintf(stderr, "[server] creating backend...\n");
    auto backend = create_backend(bargs);
    if (!backend) {
        std::fprintf(stderr, "[server] backend creation failed\n");
        return 1;
    }

    // Start HTTP server.
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "[server] ╭─── Configuration ───────────────────────────────────╮\n");
    std::fprintf(stderr, "[server] │  host            = %s\n", sconfig.host.c_str());
    std::fprintf(stderr, "[server] │  port            = %d\n", sconfig.port);
    std::fprintf(stderr, "[server] │  model           = %s\n", bargs.model_path);
    std::fprintf(stderr, "[server] │  draft           = %s\n", bargs.draft_path ? bargs.draft_path : "(none)");
    std::fprintf(stderr, "[server] │  model_name      = %s\n", sconfig.model_name.c_str());
    std::fprintf(stderr, "[server] │  max_ctx         = %d\n", sconfig.max_ctx);
    std::fprintf(stderr, "[server] │  max_tokens      = %d\n", sconfig.max_tokens);
    std::fprintf(stderr, "[server] │  target_backend = %s\n",
                 placement_backend_name(bargs.device.backend));
    std::fprintf(stderr, "[server] │  gpu             = %d\n", bargs.device.gpu);
    std::fprintf(stderr, "[server] │  draft_backend  = %s\n",
                 placement_backend_name(bargs.draft_backend));
    std::fprintf(stderr, "[server] │  draft_gpu       = %d\n", bargs.draft_gpu);
    std::fprintf(stderr, "[server] │  peer_access     = %s\n",
                 bargs.device.peer_access ? "ON" : "off");
    std::fprintf(stderr, "[server] │  chunk           = %d\n", bargs.chunk);
    std::fprintf(stderr, "[server] │  fa_window       = %d\n", bargs.fa_window);
    std::fprintf(stderr, "[server] │  ddtree          = %s\n", bargs.ddtree_mode ? "ON" : "off");
    std::fprintf(stderr, "[server] │  ddtree_budget   = %d\n", bargs.ddtree_budget);
    std::fprintf(stderr, "[server] │  cors            = %s\n", sconfig.enable_cors ? "ON" : "off");
    std::fprintf(stderr, "[server] │  cache_type_k    = %s\n",
        cache_type_k.empty() ? (sconfig.max_ctx > 6144 ? "tq3_0 (auto)" : "q4_0 (default)") : cache_type_k.c_str());
    std::fprintf(stderr, "[server] │  cache_type_v    = %s\n",
        cache_type_v.empty() ? (sconfig.max_ctx > 6144 ? "tq3_0 (auto)" : "q4_0 (default)") : cache_type_v.c_str());
    std::fprintf(stderr, "[server] │  pflash          = %s\n",
        sconfig.pflash_mode == ServerConfig::PflashMode::AUTO ? "auto" :
        sconfig.pflash_mode == ServerConfig::PflashMode::ALWAYS ? "always" : "off");
    if (pflash_enabled) {
    std::fprintf(stderr, "[server] │  pflash_threshold= %d\n", sconfig.pflash_threshold);
    std::fprintf(stderr, "[server] │  pflash_keep     = %.3f\n", sconfig.pflash_keep_ratio);
    std::fprintf(stderr, "[server] │  pflash_drafter  = %s\n", sconfig.pflash_drafter_path.c_str());
    std::fprintf(stderr, "[server] │  pflash_skip_park= %s\n", sconfig.pflash_skip_park ? "ON" : "off");
    std::fprintf(stderr, "[server] │  fp_use_bsa      = %s\n", getenv("DFLASH_FP_USE_BSA") ? "ON" : "off");
    std::fprintf(stderr, "[server] │  fp_alpha        = %s\n", getenv("DFLASH_FP_ALPHA") ? getenv("DFLASH_FP_ALPHA") : "0.12 (default)");
    }
    std::fprintf(stderr, "[server] ╰─────────────────────────────────────────────────────╯\n\n");

    HttpServer server(*backend, tokenizer, sconfig);
    if (pflash_enabled) {
        server.set_drafter_tokenizer(&drafter_tokenizer);
    }
    int ret = server.run();

    // Cleanup.
    backend->shutdown();
    return ret;
}
