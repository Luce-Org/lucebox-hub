// Standalone backend IPC daemon entry point.

#include "backend_ipc.h"
#include "dflash_draft_ipc.h"
#include "pflash_drafter_ipc.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

using namespace dflash::common;

namespace {

bool parse_nonnegative_int(const char * text, int & out) {
    if (!text || !*text) return false;
    const char * p = text;
    while (std::isspace((unsigned char)*p)) ++p;
    if (*p == '-') return false;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (errno == ERANGE || end == p || *end != '\0' ||
        value < 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    out = (int)value;
    return true;
}

bool parse_size_arg(const char * text, size_t & out) {
    if (!text || !*text) return false;
    const char * p = text;
    while (std::isspace((unsigned char)*p)) ++p;
    if (*p == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(p, &end, 10);
    if (errno == ERANGE || end == p || *end != '\0' ||
        value > (unsigned long long)std::numeric_limits<size_t>::max()) {
        return false;
    }
    out = (size_t)value;
    return true;
}

bool require_value(int & i, int argc, char ** argv, const char * opt, const char *& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "[backend-ipc-daemon] missing value for %s\n", opt);
        return false;
    }
    out = argv[++i];
    return true;
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
            "--stream-fd=FD [--draft-gpu=N]\n",
            argv[0],
            argv[0]);
        return 2;
    }

    int ring_cap = 4096;
    int draft_gpu = 0;
    int payload_fd = -1;
    int stream_fd = -1;
    int shared_payload_fd = -1;
    size_t shared_payload_bytes = 0;
    for (int i = arg_begin; i < argc; i++) {
        if (std::strncmp(argv[i], "--ring-cap=", 11) == 0) {
            if (!parse_nonnegative_int(argv[i] + 11, ring_cap)) return 2;
        } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--ring-cap", value) ||
                !parse_nonnegative_int(value, ring_cap)) return 2;
        } else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            if (!parse_nonnegative_int(argv[i] + 12, draft_gpu)) return 2;
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--draft-gpu", value) ||
                !parse_nonnegative_int(value, draft_gpu)) return 2;
        } else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            if (!parse_nonnegative_int(argv[i] + 12, stream_fd)) return 2;
        } else if (std::strcmp(argv[i], "--stream-fd") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--stream-fd", value) ||
                !parse_nonnegative_int(value, stream_fd)) return 2;
        } else if (std::strncmp(argv[i], "--payload-fd=", 13) == 0) {
            if (!parse_nonnegative_int(argv[i] + 13, payload_fd)) return 2;
        } else if (std::strcmp(argv[i], "--payload-fd") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--payload-fd", value) ||
                !parse_nonnegative_int(value, payload_fd)) return 2;
        } else if (std::strncmp(argv[i], "--shared-payload-fd=", 20) == 0) {
            if (!parse_nonnegative_int(argv[i] + 20, shared_payload_fd)) return 2;
        } else if (std::strcmp(argv[i], "--shared-payload-fd") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--shared-payload-fd", value) ||
                !parse_nonnegative_int(value, shared_payload_fd)) return 2;
        } else if (std::strncmp(argv[i], "--shared-payload-bytes=", 23) == 0) {
            if (!parse_size_arg(argv[i] + 23, shared_payload_bytes)) return 2;
        } else if (std::strcmp(argv[i], "--shared-payload-bytes") == 0) {
            const char * value = nullptr;
            if (!require_value(i, argc, argv, "--shared-payload-bytes", value) ||
                !parse_size_arg(value, shared_payload_bytes)) return 2;
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
    }
    std::fprintf(stderr, "[backend-ipc-daemon] unsupported mode\n");
    return 2;
}
