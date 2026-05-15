// Thin Gemma4 DFlash benchmark driver backed by Gemma4Backend.
//
// This target intentionally exercises src/gemma4/gemma4_backend.cpp rather than
// carrying a second inline speculative decode loop.

#include "gemma4_backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash27b;

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <target.gguf> --draft <draft_dir_or.gguf> [options]\n"
        "  --prompt <text>       prompt text (byte-fallback tokenized)\n"
        "  --tokens <ids>        comma/space-separated token ids\n"
        "  --n-predict <N>       generated token count (default: 128)\n"
        "  --ctx-size <N>        context size (default: 4096)\n"
        "  --kv-k <type>         K cache type env value (default: q8_0)\n"
        "  --kv-v <type>         V cache type env value (default: q8_0)\n"
        "  --budget <N>          DDTree budget (default: 22)\n"
        "  --draft-max <N>       DFlash draft block cap (default: 16)\n"
        "  --pflash              enable sparse-FA prefill path\n",
        argv0);
}

static std::vector<int32_t> tokenize_byte_fallback(const std::string & text) {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back((int32_t)c);
    return ids;
}

static std::vector<int32_t> parse_token_ids(const std::string & s) {
    std::vector<int32_t> ids;
    const char * p = s.c_str();
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) {
            ++p;
            continue;
        }
        ids.push_back((int32_t)v);
        p = end;
    }
    return ids;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string draft_path;
    std::string prompt_text = "Hello, world!";
    std::string token_ids_str;
    std::string kv_k = "q8_0";
    std::string kv_v = "q8_0";
    int n_predict = 128;
    int ctx_size = 4096;
    int budget = 22;
    int draft_max = 16;
    bool use_sparse_fa = false;
    bool ignore_eos = false;
    SamplerCfg sampler;

    for (int i = 1; i < argc; i++) {
        auto require_next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };

        if      (std::strcmp(argv[i], "--model")     == 0) model_path = require_next("--model");
        else if (std::strcmp(argv[i], "--draft")     == 0) draft_path = require_next("--draft");
        else if (std::strcmp(argv[i], "--prompt")    == 0) prompt_text = require_next("--prompt");
        else if (std::strcmp(argv[i], "--tokens")    == 0) token_ids_str = require_next("--tokens");
        else if (std::strcmp(argv[i], "--n-predict") == 0) n_predict = std::atoi(require_next("--n-predict"));
        else if (std::strcmp(argv[i], "--ctx-size")  == 0) ctx_size = std::atoi(require_next("--ctx-size"));
        else if (std::strcmp(argv[i], "--max-ctx")   == 0) ctx_size = std::atoi(require_next("--max-ctx"));
        else if (std::strcmp(argv[i], "--kv-k")      == 0) kv_k = require_next("--kv-k");
        else if (std::strcmp(argv[i], "--kv-v")      == 0) kv_v = require_next("--kv-v");
        else if (std::strcmp(argv[i], "--budget")    == 0) budget = std::atoi(require_next("--budget"));
        else if (std::strcmp(argv[i], "--draft-max") == 0) draft_max = std::atoi(require_next("--draft-max"));
        else if (std::strcmp(argv[i], "--temp")      == 0) sampler.temp = (float)std::atof(require_next("--temp"));
        else if (std::strcmp(argv[i], "--seed")      == 0) sampler.seed = (uint64_t)std::atoll(require_next("--seed"));
        else if (std::strcmp(argv[i], "--pflash")    == 0) use_sparse_fa = true;
        else if (std::strcmp(argv[i], "--ignore-eos") == 0) ignore_eos = true;
        else if (std::strcmp(argv[i], "--bench") == 0) { /* no-op */ }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "warning: unknown argument: %s\n", argv[i]);
        }
    }

    if (model_path.empty() || draft_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    if (ctx_size <= 0 || n_predict <= 0) {
        std::fprintf(stderr, "error: --ctx-size and --n-predict must be positive\n");
        return 2;
    }

    setenv("DFLASH27B_KV_K", kv_k.c_str(), 1);
    setenv("DFLASH27B_KV_V", kv_v.c_str(), 1);

    Gemma4BackendConfig cfg;
    cfg.target_path = model_path;
    cfg.draft_path = draft_path;
    cfg.draft_method = Gemma4DraftMethod::kDFlash;
    cfg.max_ctx = ctx_size;
    cfg.use_sparse_fa = use_sparse_fa;
    cfg.draft_max_block = draft_max > 0 ? draft_max : 16;
    cfg.draft_enable_capture_overrides = true;
    cfg.ddtree_budget = std::max(0, budget);
    cfg.ddtree_temp = 1.0f;
    cfg.ddtree_chain_seed = true;
    cfg.ignore_eos = ignore_eos;

    std::printf("[cfg] model=%s draft=%s ctx=%d n_predict=%d kv_k=%s kv_v=%s "
                "budget=%d draft_max=%d pflash=%d temp=%.2f\n",
                model_path.c_str(), draft_path.c_str(), ctx_size, n_predict,
                kv_k.c_str(), kv_v.c_str(), cfg.ddtree_budget,
                cfg.draft_max_block, (int)use_sparse_fa, sampler.temp);

    Gemma4Backend backend(cfg);
    if (!backend.init()) {
        std::fprintf(stderr, "error: Gemma4Backend init failed\n");
        return 1;
    }

    std::vector<int32_t> prompt_ids = token_ids_str.empty()
        ? tokenize_byte_fallback(prompt_text)
        : parse_token_ids(token_ids_str);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "error: prompt produced no tokens\n");
        backend.shutdown();
        return 2;
    }
    constexpr int32_t kGemmaBos = 2;
    if (prompt_ids.front() != kGemmaBos) prompt_ids.insert(prompt_ids.begin(), kGemmaBos);

    GenerateRequest req;
    req.prompt = std::move(prompt_ids);
    req.n_gen = n_predict;
    req.sampler = sampler;
    req.stream = false;

    DaemonIO io;
    GenerateResult result = backend.generate(req, io);
    if (!result.ok) {
        std::fprintf(stderr, "error: generate failed at %s\n", result.error.c_str());
        backend.shutdown();
        return 1;
    }

    for (int32_t tok : result.tokens) std::printf("%d ", tok);
    const double decode_ms = result.decode_s * 1000.0;
    const double tok_s = result.decode_s > 0.0
        ? (double)result.tokens.size() / result.decode_s
        : 0.0;
    std::printf("\n[stats] generated=%zu  decode_ms=%.1f  tok/s=%.2f  first_tok_ms=0.00\n",
                result.tokens.size(), decode_ms, tok_s);
    std::printf("[stats] prefill=%zu tokens  context_used=%zu/%d\n",
                req.prompt.size(), req.prompt.size() + result.tokens.size(), ctx_size);

    backend.shutdown();
    return 0;
}
