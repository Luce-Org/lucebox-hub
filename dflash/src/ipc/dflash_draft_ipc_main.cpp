// Standalone DFlash draft IPC daemon entry point.

#include "dflash_draft_ipc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 3 || std::strcmp(argv[1], "--draft-ipc-daemon") != 0) {
        std::fprintf(stderr,
            "usage: %s --draft-ipc-daemon <draft.safetensors|draft.gguf> "
            "--ring-cap=N --stream-fd=FD [--draft-gpu=N]\n",
            argv[0]);
        return 2;
    }

    const char * draft_path = argv[2];
    int ring_cap = 4096;
    int draft_gpu = 0;
    int stream_fd = -1;
    for (int i = 3; i < argc; i++) {
        if (std::strncmp(argv[i], "--ring-cap=", 11) == 0) {
            ring_cap = std::atoi(argv[i] + 11);
        } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
            if (i + 1 < argc) ring_cap = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            draft_gpu = std::max(0, std::atoi(argv[i] + 12));
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            if (i + 1 < argc) draft_gpu = std::max(0, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--stream-fd") == 0) {
            if (i + 1 < argc) stream_fd = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "[draft-ipc-daemon] unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    return run_dflash_draft_ipc_daemon(draft_path, ring_cap, draft_gpu, stream_fd);
}
