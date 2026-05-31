// Server status tracking for the /status introspection page.
//
// Thread-safe status tracker: worker thread writes, HTTP client threads read.
// Designed for minimal overhead on the inference hot path.

#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace dflash::common {

using json = nlohmann::json;

// Performance record for one completed request.
struct PerfRecord {
    double prefill_tok_s    = 0.0;
    double decode_tok_s     = 0.0;
    float  accept_rate      = 0.0f;
    int    prompt_tokens    = 0;
    int    completion_tokens = 0;
    bool   cache_hit        = false;
    bool   pflash           = false;
    bool   spec_decode      = false;
    std::chrono::steady_clock::time_point timestamp;
};

// Live inference phase.
enum class InferencePhase {
    IDLE,
    PREFILL,
    DECODE,
};

static inline const char * phase_name(InferencePhase p) {
    switch (p) {
    case InferencePhase::IDLE:    return "idle";
    case InferencePhase::PREFILL: return "prefill";
    case InferencePhase::DECODE:  return "decode";
    default:                      return "unknown";
    }
}

class ServerStatus {
public:
    static constexpr int kMaxHistory = 50;

    // Called by worker thread to update live state.
    void set_running(const std::string & prompt_excerpt, int prompt_tokens,
                     bool is_stream) {
        std::lock_guard<std::mutex> lk(mu_);
        phase_ = InferencePhase::PREFILL;
        prompt_excerpt_ = prompt_excerpt;
        prompt_tokens_ = prompt_tokens;
        completion_tokens_ = 0;
        is_stream_ = is_stream;
        draft_tokens_.clear();
        started_at_ = std::chrono::steady_clock::now();
    }

    void set_decode() {
        std::lock_guard<std::mutex> lk(mu_);
        phase_ = InferencePhase::DECODE;
    }

    void update_completion_tokens(int n) {
        std::lock_guard<std::mutex> lk(mu_);
        completion_tokens_ = n;
    }

    void set_draft_tokens(const std::vector<std::string> & tokens) {
        std::lock_guard<std::mutex> lk(mu_);
        draft_tokens_ = tokens;
    }

    void set_idle() {
        std::lock_guard<std::mutex> lk(mu_);
        phase_ = InferencePhase::IDLE;
        prompt_excerpt_.clear();
        draft_tokens_.clear();
    }

    void record_perf(const PerfRecord & rec) {
        std::lock_guard<std::mutex> lk(mu_);
        if ((int)perf_history_.size() >= kMaxHistory) {
            perf_history_.erase(perf_history_.begin());
        }
        perf_history_.push_back(rec);
        total_requests_++;
    }

    // Snapshot current state as JSON (thread-safe).
    json to_json() const {
        InferencePhase phase;
        std::string prompt_excerpt;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        bool is_stream = false;
        std::vector<std::string> draft_tokens;
        std::vector<PerfRecord> history;
        int total_requests = 0;
        double elapsed_s = 0.0;

        {
            std::lock_guard<std::mutex> lk(mu_);
            phase = phase_;
            prompt_excerpt = prompt_excerpt_;
            prompt_tokens = prompt_tokens_;
            completion_tokens = completion_tokens_;
            is_stream = is_stream_;
            draft_tokens = draft_tokens_;
            history = perf_history_;
            total_requests = total_requests_;
            if (phase != InferencePhase::IDLE) {
                elapsed_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started_at_).count();
            }
        }

        json j;
        j["phase"] = phase_name(phase);
        j["total_requests"] = total_requests;

        if (phase != InferencePhase::IDLE) {
            j["current"] = {
                {"prompt_excerpt", prompt_excerpt},
                {"prompt_tokens", prompt_tokens},
                {"completion_tokens", completion_tokens},
                {"stream", is_stream},
                {"elapsed_s", elapsed_s},
                {"draft_tokens", draft_tokens},
            };
        } else {
            j["current"] = nullptr;
        }

        json perf = json::array();
        for (const auto & r : history) {
            perf.push_back({
                {"prefill_tok_s", r.prefill_tok_s},
                {"decode_tok_s", r.decode_tok_s},
                {"accept_rate", r.accept_rate},
                {"prompt_tokens", r.prompt_tokens},
                {"completion_tokens", r.completion_tokens},
                {"cache_hit", r.cache_hit},
                {"pflash", r.pflash},
                {"spec_decode", r.spec_decode},
            });
        }
        j["perf_history"] = perf;

        return j;
    }

    // Format as SSE event string: "event: status\ndata: {json}\n\n"
    std::string to_sse_event() const {
        std::string data = to_json().dump();
        return "event: status\ndata: " + data + "\n\n";
    }

private:
    mutable std::mutex mu_;

    // Live state.
    InferencePhase phase_ = InferencePhase::IDLE;
    std::string prompt_excerpt_;
    int prompt_tokens_ = 0;
    int completion_tokens_ = 0;
    bool is_stream_ = false;
    std::vector<std::string> draft_tokens_;
    std::chrono::steady_clock::time_point started_at_;

    // History.
    std::vector<PerfRecord> perf_history_;
    int total_requests_ = 0;
};

// RAII guard that resets status to idle on scope exit.
struct StatusGuard {
    ServerStatus & status;
    ~StatusGuard() { status.set_idle(); }
};

}  // namespace dflash::common
