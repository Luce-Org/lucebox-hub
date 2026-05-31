// Standalone backend IPC daemon entry point.

#include "backend_ipc.h"
#include "dflash_draft_ipc.h"
#include "pflash_drafter_ipc.h"
#include "qwen35_target_shard_ipc.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

bool parse_int_list(const char * raw, std::vector<int> & out) {
    out.clear();
    if (!raw || !*raw) return false;
    std::string text(raw);
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty()) return false;
        char * parse_end = nullptr;
        const long value = std::strtol(item.c_str(), &parse_end, 10);
        if (parse_end == item.c_str() || *parse_end != '\0') return false;
        out.push_back((int)value);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return !out.empty();
}

}  // namespace

int main(int argc, char ** argv) {
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    const char * payload_path = nullptr;
    int arg_begin = 0;

    if (argc >= 3 && std::strncmp(argv[1], "--backend-ipc-mode=", 19) == 0) {
        std::string mode_name(argv[1] + 19);
        if (!parse_backend_ipc_mode(mode_name, mode)) {
            std::fprintf(stderr, "[backend-ipc-daemon] unknown mode: %s\n",
                         mode_name.c_str());
            return 2;
        }
        payload_path = argv[2];
        arg_begin = 3;
    } else if (argc >= 3 && std::strcmp(argv[1], "--backend-ipc-mode") == 0) {
        if (!parse_backend_ipc_mode(argv[2], mode) || argc < 4) {
            std::fprintf(stderr, "[backend-ipc-daemon] bad --backend-ipc-mode\n");
            return 2;
        }
        payload_path = argv[3];
        arg_begin = 4;
    } else {
        std::fprintf(stderr,
            "usage: %s --backend-ipc-mode=dflash-draft <draft.safetensors|draft.gguf> "
            "--ring-cap=N --stream-fd=FD [--payload-fd=FD] "
            "[--shared-payload-fd=FD --shared-payload-bytes=N] [--draft-gpu=N]\n"
            "   or: %s --backend-ipc-mode=pflash-compress <drafter.gguf> "
            "--stream-fd=FD [--draft-gpu=N]\n"
            "   or: %s --backend-ipc-mode=qwen35-target-shard <target.gguf> "
            "--stream-fd=FD --target-gpu=N --layer-begin=N --layer-end=N "
            "--max-ctx=N\n",
            argv[0],
            argv[0],
            argv[0]);
        return 2;
    }

    int ring_cap = 4096;
    int draft_gpu = 0;
    int target_gpu = 0;
    std::vector<int> target_gpus;
    std::vector<int> layer_begins;
    std::vector<int> layer_ends;
    int layer_begin = -1;
    int layer_end = -1;
    int max_ctx = 8192;
    int max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    int kq_stride_pad = 32;
    int fa_window = 0;
    int payload_fd = -1;
    int stream_fd = -1;
    int shared_payload_fd = -1;
    size_t shared_payload_bytes = 0;
    bool enable_dflash = false;
    for (int i = arg_begin; i < argc; i++) {
        if (std::strncmp(argv[i], "--ring-cap=", 11) == 0) {
            ring_cap = std::atoi(argv[i] + 11);
        } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
            if (i + 1 < argc) ring_cap = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            draft_gpu = std::max(0, std::atoi(argv[i] + 12));
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            if (i + 1 < argc) draft_gpu = std::max(0, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--target-gpu=", 13) == 0) {
            target_gpu = std::max(0, std::atoi(argv[i] + 13));
        } else if (std::strcmp(argv[i], "--target-gpu") == 0) {
            if (i + 1 < argc) target_gpu = std::max(0, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--target-gpus=", 14) == 0) {
            if (!parse_int_list(argv[i] + 14, target_gpus)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --target-gpus\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--target-gpus") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], target_gpus)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --target-gpus\n");
                return 2;
            }
        } else if (std::strncmp(argv[i], "--layer-begin=", 14) == 0) {
            layer_begin = std::atoi(argv[i] + 14);
        } else if (std::strcmp(argv[i], "--layer-begin") == 0) {
            if (i + 1 < argc) layer_begin = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--layer-begins=", 15) == 0) {
            if (!parse_int_list(argv[i] + 15, layer_begins)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-begins\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--layer-begins") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], layer_begins)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-begins\n");
                return 2;
            }
        } else if (std::strncmp(argv[i], "--layer-end=", 12) == 0) {
            layer_end = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--layer-end") == 0) {
            if (i + 1 < argc) layer_end = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--layer-ends=", 13) == 0) {
            if (!parse_int_list(argv[i] + 13, layer_ends)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-ends\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--layer-ends") == 0) {
            if (i + 1 >= argc || !parse_int_list(argv[++i], layer_ends)) {
                std::fprintf(stderr, "[backend-ipc-daemon] bad --layer-ends\n");
                return 2;
            }
        } else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) {
            max_ctx = std::atoi(argv[i] + 10);
        } else if (std::strcmp(argv[i], "--max-ctx") == 0) {
            if (i + 1 < argc) max_ctx = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--max-verify-tokens=", 20) == 0) {
            max_verify_tokens = std::atoi(argv[i] + 20);
        } else if (std::strcmp(argv[i], "--max-verify-tokens") == 0) {
            if (i + 1 < argc) max_verify_tokens = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--kq-stride-pad=", 16) == 0) {
            kq_stride_pad = std::atoi(argv[i] + 16);
        } else if (std::strcmp(argv[i], "--kq-stride-pad") == 0) {
            if (i + 1 < argc) kq_stride_pad = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--fa-window=", 12) == 0) {
            fa_window = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--fa-window") == 0) {
            if (i + 1 < argc) fa_window = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--stream-fd") == 0) {
            if (i + 1 < argc) stream_fd = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--payload-fd=", 13) == 0) {
            payload_fd = std::atoi(argv[i] + 13);
        } else if (std::strcmp(argv[i], "--payload-fd") == 0) {
            if (i + 1 < argc) payload_fd = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--shared-payload-fd=", 20) == 0) {
            shared_payload_fd = std::atoi(argv[i] + 20);
        } else if (std::strcmp(argv[i], "--shared-payload-fd") == 0) {
            if (i + 1 < argc) shared_payload_fd = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--shared-payload-bytes=", 23) == 0) {
            shared_payload_bytes = (size_t)std::strtoull(argv[i] + 23, nullptr, 10);
        } else if (std::strcmp(argv[i], "--shared-payload-bytes") == 0) {
            if (i + 1 < argc) {
                shared_payload_bytes = (size_t)std::strtoull(argv[++i], nullptr, 10);
            }
        } else if (std::strcmp(argv[i], "--enable-dflash") == 0) {
            enable_dflash = true;
        } else {
            std::fprintf(stderr, "[backend-ipc-daemon] unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    switch (mode) {
        case BackendIpcMode::DFlashDraft:
            return run_dflash_draft_ipc_daemon(payload_path, ring_cap, draft_gpu,
                                               stream_fd, payload_fd,
                                               shared_payload_fd,
                                               shared_payload_bytes);
        case BackendIpcMode::PFlashCompress:
            return run_pflash_drafter_ipc_daemon(payload_path, draft_gpu, stream_fd);
        case BackendIpcMode::Qwen35TargetShard:
            if (target_gpus.empty()) target_gpus.push_back(target_gpu);
            if (layer_begins.empty()) layer_begins.push_back(layer_begin);
            if (layer_ends.empty()) layer_ends.push_back(layer_end);
            return run_qwen35_target_shard_ipc_daemon(
                payload_path, target_gpus, layer_begins, layer_ends, max_ctx,
                max_verify_tokens, kq_stride_pad, fa_window, stream_fd,
                payload_fd, shared_payload_fd, shared_payload_bytes,
                enable_dflash);
    }
    std::fprintf(stderr, "[backend-ipc-daemon] unsupported mode\n");
    return 2;
}
