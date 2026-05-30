// Pure, correct-by-construction adaptive compression-regime router.
// No IO, no globals, no GPU, no ggml/llama deps — header-only.
//
// Decides whether the transitive anchor cascade should run at full expansion
// (FullCascade, recall-preserving default) or be throttled
// (Throttle, fires ONLY when expansion_ratio >= policy threshold).
//
// Build (standalone):
//   g++-11 -std=gnu++17 -O2 -I. -o test_regime_router test/test_regime_router.cpp
// CMake:  cmake --build build --target test_regime_router -j
//         ctest -R regime_router --output-on-failure
#pragma once

#include <cmath>
#include <limits>

namespace dflash::common {

// ─── Input ───────────────────────────────────────────────────────────────────

// All inputs are cheap lexical counts already available in the cascade path.
struct CascadeStats {
    int n_chunks;
    int forced_anchor_only;    // chunks forced by BASE anchors, pre-cascade
    int forced_after_cascade;  // chunks forced AFTER transitive cascade
    int prompt_tokens;         // S
    int keep_floor_chunks;     // ceil(keep_ratio * n_chunks) budget (informational)
};

// ─── Policy ──────────────────────────────────────────────────────────────────

struct RouterPolicy {
    int    threshold_tokens         = 32000;           // below this: passthrough
    double expansion_throttle_ratio = INFINITY;        // DEFAULT disabled
    int    min_anchor_chunks        = 1;               // don't throttle if too few anchors
};

// ─── Output ──────────────────────────────────────────────────────────────────

enum class Regime { FullCascade, Throttle };

struct RegimeDecision {
    Regime      regime;
    double      expansion_ratio;
    const char* reason;
};

// ─── Core function ───────────────────────────────────────────────────────────

// decide_regime — pure, no IO, no globals.
//
// Expansion ratio R = forced_after_cascade / forced_anchor_only
//   (defined as 1.0 when forced_anchor_only == 0 to avoid division by zero).
//
// Transition to Throttle ONLY on the last branch; every other path returns
// FullCascade so the default deployment posture is recall-preserving.
inline RegimeDecision decide_regime(const CascadeStats& s, const RouterPolicy& p) {
    // Compute R first (needed for degenerate guard + return value).
    const double R = (s.forced_anchor_only > 0)
        ? static_cast<double>(s.forced_after_cascade) / s.forced_anchor_only
        : 1.0;

    // Guard: degenerate inputs — return FullCascade, no further processing.
    if (s.n_chunks <= 0 || s.forced_anchor_only < 0 || s.forced_after_cascade < 0)
        return { Regime::FullCascade, R, "degenerate" };

    // Passthrough: prompt too short to compress meaningfully.
    if (s.prompt_tokens < p.threshold_tokens)
        return { Regime::FullCascade, R, "below_threshold" };

    // Guard: too few base anchors — throttle would be meaningless.
    if (s.forced_anchor_only < p.min_anchor_chunks)
        return { Regime::FullCascade, R, "too_few_anchors" };

    // Only transition: cascade expanded beyond the policy limit.
    if (R >= p.expansion_throttle_ratio)
        return { Regime::Throttle, R, "cascade_over_expansion" };

    return { Regime::FullCascade, R, "default_safe" };
}

// ─── V2 Router ───────────────────────────────────────────────────────────────
//
// Adaptive compression router v2.
// Splits on prompt TYPE (agentic vs retrieval) rather than cascade expansion
// ratio R (which was refuted as a keep predictor, Spearman ρ=-0.27).
//
// Additional guards:
//   sparse_prompt_guard — skip compression when new_content_tokens is tiny
//     (plumbing turns: recent orchestration continuity must not be dropped)
//   recency_floor_turns — always keep the last K turns whole in the agentic path
//
// Sentinel for "keep all turns" recency in SAFE decisions:
static constexpr int kRecencyKeepAll = (1 << 20);

// Size-adaptive recency floor sentinel.
// When recency_floor_tokens == kRecencyFloorAuto the compress path computes
//   R = min(1024, ceil(0.04 * prompt_tokens))
// instead of using a fixed token count.  0 = off (no-op default).
static constexpr int kRecencyFloorAuto = -1;

struct RequestFeatures {
    bool is_agentic;           // tool schemas / tool_use|tool_result blocks present
    int  prompt_tokens;        // total S
    int  new_content_tokens;   // newest turn content size (sparse-plumbing detector)
};

struct RouterPolicyV2 {
    bool   enabled                    = false;   // DEFAULT DISABLED → exact no-op
    int    threshold_tokens           = 32000;   // below → passthrough
    double agentic_keep_target        = 0.25;    // conservative floor, closes empty-failure tail
    double full_keep_target           = 1.0;     // retrieval/QA & safe fallbacks
    int    recency_floor_turns        = 2;       // keep last K turns whole (continuity)
    int    sparse_new_content_tokens  = 256;     // below this → sparse_prompt_guard fires
};

// recency_floor_for — pure helper, no IO.
//
// Returns the concrete token floor for a given prompt size and policy:
//   recency_floor_tokens == 0           → 0  (off, no-op)
//   recency_floor_tokens == kRecencyFloorAuto (-1)
//                                       → min(1024, ceil(0.04 * prompt_tokens))
//   recency_floor_tokens  > 0           → recency_floor_tokens  (explicit override)
//
// "one turn equivalent" lower-bound: the agentic throttle path in decide_v2
// already reserves recency_floor_turns whole turns; this helper computes the
// token-count floor passed to the compress path for the token-budget guard.
inline int recency_floor_for(int prompt_tokens, int recency_floor_tokens) {
    if (recency_floor_tokens == 0)
        return 0;
    if (recency_floor_tokens == kRecencyFloorAuto) {
        // min(1024, ceil(0.04 * S)) — scales with context, caps at 1024
        const int adaptive = static_cast<int>(
            std::ceil(0.04 * static_cast<double>(prompt_tokens < 0 ? 0 : prompt_tokens)));
        return (adaptive < 1024) ? adaptive : 1024;
    }
    // Explicit positive override.
    return (recency_floor_tokens > 0) ? recency_floor_tokens : 0;
}

struct RouterDecisionV2 {
    double      keep_target;
    int         recency_floor_turns;
    bool        cascade;
    const char* reason;
};

// decide_v2 — pure, no IO, no globals.
//
// SAFE path: keep_target=full_keep_target, recency=kRecencyKeepAll, cascade=true.
// Returns SAFE when:
//   - p.enabled == false                        (deploy no-op, correct-by-construction)
//   - f.prompt_tokens <= 0 || f.new_content_tokens < 0  (degenerate)
//   - f.prompt_tokens < p.threshold_tokens      (below threshold)
//   - f.new_content_tokens < p.sparse_new_content_tokens (sparse_prompt_guard)
// Throttling path (only when all guards pass):
//   - is_agentic → {agentic_keep_target, recency_floor_turns, cascade=false}
//   - else       → {full_keep_target,    recency_floor_turns, cascade=true}
inline RouterDecisionV2 decide_v2(const RequestFeatures& f,
                                   const RouterPolicyV2&   p) {
    // Helper: SAFE return (keep everything, cascade on, recency = keep-all).
    const RouterDecisionV2 SAFE_disabled        = { p.full_keep_target, kRecencyKeepAll, true, "disabled_noop"       };
    const RouterDecisionV2 SAFE_degenerate      = { p.full_keep_target, kRecencyKeepAll, true, "degenerate"          };
    const RouterDecisionV2 SAFE_below_threshold = { p.full_keep_target, kRecencyKeepAll, true, "below_threshold"     };
    const RouterDecisionV2 SAFE_sparse          = { p.full_keep_target, kRecencyKeepAll, true, "sparse_prompt_guard" };

    // 1. Deploy no-op: disabled router is an exact no-op (correct-by-construction).
    if (!p.enabled)
        return SAFE_disabled;

    // 2. Degenerate inputs: prompt_tokens <= 0 or new_content_tokens < 0.
    if (f.prompt_tokens <= 0 || f.new_content_tokens < 0)
        return SAFE_degenerate;

    // 3. Below threshold: prompt too short to compress meaningfully.
    if (f.prompt_tokens < p.threshold_tokens)
        return SAFE_below_threshold;

    // 4. Sparse-prompt guard: tiny new-content turn (plumbing class).
    //    Compression would drop recent orchestration continuity with no anchor signal.
    if (f.new_content_tokens < p.sparse_new_content_tokens)
        return SAFE_sparse;

    // 5. Throttling paths — all guards passed.
    if (f.is_agentic)
        return { p.agentic_keep_target, p.recency_floor_turns, false, "agentic_throttle" };

    return { p.full_keep_target, p.recency_floor_turns, true, "retrieval_full" };
}

}  // namespace dflash::common
