// Unit tests for dflash::common::decide_regime() — pure function, no GPU.
//
// Build (standalone, from repo root):
//   g++-11 -std=gnu++17 -O2 -I server/src/common
//     -o /tmp/test_regime_router server/test/test_regime_router.cpp
// CMake:
//   cmake --build build --target test_regime_router -j
//   ctest -R regime_router --output-on-failure

#include "regime_router.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

using namespace dflash::common;

// ─── Minimal test framework (mirrors test_adaptive_keep_ratio.cpp) ───────────

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define TEST_ASSERT_MSG(expr, msg) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s -- %s\n", \
                     __FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

static inline bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build a policy with expansion_throttle_ratio disabled (default safe).
static RouterPolicy default_policy() { return {}; }

// Build a policy that throttles at ratio >= r.
static RouterPolicy throttle_policy(double r,
                                    int threshold = 32000,
                                    int min_anchor = 1) {
    RouterPolicy p;
    p.threshold_tokens         = threshold;
    p.expansion_throttle_ratio = r;
    p.min_anchor_chunks        = min_anchor;
    return p;
}

static CascadeStats make_stats(int n_chunks,
                                int anchor_only,
                                int after_cascade,
                                int prompt_tokens,
                                int keep_floor = 0) {
    return { n_chunks, anchor_only, after_cascade, prompt_tokens, keep_floor };
}

// ─── T1: DEPLOY-NO-OP ────────────────────────────────────────────────────────
// With the DEFAULT RouterPolicy (ratio=INFINITY), decide_regime must return
// FullCascade for ANY stats, including pathologically large expansion.

static void t1_deploy_noop() {
    RouterPolicy p = default_policy();

    // Normal case
    {
        auto d = decide_regime(make_stats(100, 10, 20, 50000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T1a: default policy must always give FullCascade");
    }
    // Huge expansion: forced_anchor_only=10, forced_after_cascade=1000, prompt=100K
    {
        auto d = decide_regime(make_stats(500, 10, 1000, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T1b: huge expansion with default policy must be FullCascade");
    }
    // Prompt below threshold
    {
        auto d = decide_regime(make_stats(50, 5, 500, 1000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T1c: short prompt with default policy must be FullCascade");
    }
    // Zero anchors
    {
        auto d = decide_regime(make_stats(100, 0, 0, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T1d: zero anchors with default policy must be FullCascade");
    }
    // Sweep: 50 random-ish stat combinations
    for (int i = 1; i <= 50; ++i) {
        CascadeStats s = make_stats(i * 10,
                                    i,
                                    i * 100,          // R = 100, very high
                                    i * 5000);
        auto d = decide_regime(s, p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T1-sweep: default policy must be FullCascade for all stats");
    }
}

// ─── T2: DEGENERATE ──────────────────────────────────────────────────────────
// Degenerate inputs must not crash or div-by-zero, and must return FullCascade.

static void t2_degenerate() {
    RouterPolicy p = throttle_policy(2.0);   // would throttle if R >= 2

    // n_chunks == 0
    {
        auto d = decide_regime(make_stats(0, 5, 10, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T2a: n_chunks=0 must return FullCascade");
        TEST_ASSERT_MSG(std::isfinite(d.expansion_ratio),
                        "T2a: expansion_ratio must be finite when n_chunks=0");
    }
    // forced_anchor_only == 0 (no anchors before cascade) → R defaults to 1.0
    {
        auto d = decide_regime(make_stats(100, 0, 50, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T2b: forced_anchor_only=0 must return FullCascade");
        TEST_ASSERT_MSG(approx_eq(d.expansion_ratio, 1.0),
                        "T2b: expansion_ratio must be 1.0 when forced_anchor_only=0");
    }
    // Negative forced_anchor_only
    {
        auto d = decide_regime(make_stats(100, -1, 50, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T2c: negative forced_anchor_only must return FullCascade");
        TEST_ASSERT_MSG(std::isfinite(d.expansion_ratio),
                        "T2c: expansion_ratio must be finite for negative anchor count");
    }
    // Negative forced_after_cascade
    {
        auto d = decide_regime(make_stats(100, 5, -1, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T2d: negative forced_after_cascade must return FullCascade");
    }
    // Both negative
    {
        auto d = decide_regime(make_stats(100, -3, -7, 100000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T2e: both counts negative must return FullCascade");
    }
}

// ─── T3: BELOW-THRESHOLD ─────────────────────────────────────────────────────
// prompt_tokens < threshold → FullCascade regardless of R and finite ratio.

static void t3_below_threshold() {
    RouterPolicy p = throttle_policy(1.5, /*threshold=*/32000, /*min_anchor=*/1);

    // prompt = threshold - 1 (just below)
    {
        auto d = decide_regime(make_stats(100, 10, 1000, 31999), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T3a: prompt just below threshold must be FullCascade");
        TEST_ASSERT_MSG(std::string(d.reason) == "below_threshold",
                        "T3a: reason must be 'below_threshold'");
    }
    // prompt = 0
    {
        auto d = decide_regime(make_stats(100, 10, 9999, 0), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T3b: prompt=0 must be FullCascade");
    }
    // Even with R = 1000 and finite ratio = 2.0, still FullCascade below threshold
    {
        auto d = decide_regime(make_stats(200, 5, 5000, 100), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T3c: tiny prompt, huge R, finite ratio -> FullCascade");
    }
}

// ─── T4: TOO-FEW-ANCHORS ─────────────────────────────────────────────────────
// forced_anchor_only < min_anchor_chunks → FullCascade.

static void t4_too_few_anchors() {
    RouterPolicy p = throttle_policy(2.0, /*threshold=*/32000, /*min_anchor=*/3);
    // forced_anchor_only = 2 < min_anchor = 3
    {
        auto d = decide_regime(make_stats(100, 2, 1000, 50000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T4a: anchors below min must be FullCascade");
    }
    // forced_anchor_only = 0 < min_anchor = 3
    {
        auto d = decide_regime(make_stats(100, 0, 500, 50000), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T4b: zero anchors below min must be FullCascade");
    }
    // forced_anchor_only = 3 == min_anchor = 3: NOT too few → may throttle
    {
        auto d = decide_regime(make_stats(100, 3, 300, 50000), p);
        // R = 300/3 = 100 >= 2.0 → should be Throttle
        TEST_ASSERT_MSG(d.regime == Regime::Throttle,
                        "T4c: anchors == min AND R >= ratio must throttle");
    }
}

// ─── T5: MONOTONE ────────────────────────────────────────────────────────────
// With a finite ratio policy, once Throttle triggers at R it must stay Throttle
// for all larger R.

static void t5_monotone() {
    // Policy: ratio=3.0, threshold=32000, min_anchor=1, prompt_tokens=50000
    RouterPolicy p = throttle_policy(3.0, 32000, 1);
    const int prompt = 50000;
    const int anchor = 10;  // fixed; vary after_cascade to control R

    // R = 2.9 → FullCascade
    {
        // after = anchor * R = 10 * 2.9 = 29
        auto d = decide_regime(make_stats(100, anchor, 29, prompt), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T5a: R=2.9 < 3.0 must be FullCascade");
    }
    // R = 3.0 → Throttle (boundary: >= triggers)
    {
        // after = 10 * 3 = 30
        auto d = decide_regime(make_stats(100, anchor, 30, prompt), p);
        TEST_ASSERT_MSG(d.regime == Regime::Throttle,
                        "T5b: R=3.0 == ratio must be Throttle");
    }
    // R = 10.0 → Throttle
    {
        auto d = decide_regime(make_stats(100, anchor, 100, prompt), p);
        TEST_ASSERT_MSG(d.regime == Regime::Throttle,
                        "T5c: R=10.0 >> ratio must be Throttle");
    }
    // Monotone sweep: for all integer R from 1 to 100, once Throttle appears
    // it must not flip back to FullCascade.
    bool seen_throttle = false;
    bool monotone      = true;
    for (int r_int = 1; r_int <= 100; ++r_int) {
        // after = anchor * r_int → exact integer R
        auto d = decide_regime(make_stats(200, anchor, anchor * r_int, prompt), p);
        if (d.regime == Regime::Throttle) {
            seen_throttle = true;
        } else if (seen_throttle) {
            // Flipped back to FullCascade after Throttle was seen: not monotone
            monotone = false;
            std::fprintf(stderr,
                         "  MONOTONE VIOLATION at R=%d: Throttle then FullCascade\n",
                         r_int);
            break;
        }
    }
    TEST_ASSERT_MSG(seen_throttle, "T5d: sweep must trigger Throttle at some R");
    TEST_ASSERT_MSG(monotone,      "T5e: regime must be monotone (no FullCascade after Throttle)");
}

// ─── T6: BOUNDARY ────────────────────────────────────────────────────────────
// R exactly == ratio → Throttle; R = ratio - epsilon → FullCascade.

static void t6_boundary() {
    const double ratio   = 5.0;
    RouterPolicy p       = throttle_policy(ratio, 32000, 1);
    const int    anchor  = 1000;   // use large anchor to get precise integer ratios
    const int    prompt  = 50000;

    // R exactly == ratio: after = anchor * ratio = 5000
    {
        auto d = decide_regime(make_stats(500, anchor, anchor * (int)ratio, prompt), p);
        TEST_ASSERT_MSG(d.regime == Regime::Throttle,
                        "T6a: R exactly == ratio must be Throttle");
    }

    // R = ratio - epsilon where epsilon = 0.5/anchor (one less chunk → R < ratio)
    {
        // after = anchor * ratio - 1 = 4999 → R = 4.999 < 5.0
        auto d = decide_regime(make_stats(500, anchor, anchor * (int)ratio - 1, prompt), p);
        TEST_ASSERT_MSG(d.regime == Regime::FullCascade,
                        "T6b: R just below ratio must be FullCascade");
    }
}

// ─── T7: RATIO-VALUE ─────────────────────────────────────────────────────────
// Check that expansion_ratio is computed correctly.

static void t7_ratio_value() {
    RouterPolicy p = default_policy();  // regime doesn't matter; check ratio value

    // forced_anchor_only=10, forced_after_cascade=85 → R = 8.5
    {
        auto d = decide_regime(make_stats(100, 10, 85, 50000), p);
        TEST_ASSERT_MSG(approx_eq(d.expansion_ratio, 8.5),
                        "T7a: R must be 85/10 = 8.5");
    }
    // forced_anchor_only=0 → R must be 1.0 (no div-by-zero)
    {
        auto d = decide_regime(make_stats(100, 0, 50, 50000), p);
        TEST_ASSERT_MSG(approx_eq(d.expansion_ratio, 1.0),
                        "T7b: forced_anchor_only=0 must give expansion_ratio=1.0");
    }
    // forced_anchor_only=5, forced_after_cascade=5 → R = 1.0
    {
        auto d = decide_regime(make_stats(100, 5, 5, 50000), p);
        TEST_ASSERT_MSG(approx_eq(d.expansion_ratio, 1.0),
                        "T7c: equal anchors before/after must give R=1.0");
    }
    // forced_anchor_only=7, forced_after_cascade=7 → R = 1.0 (no expansion)
    {
        auto d = decide_regime(make_stats(100, 7, 7, 50000), p);
        TEST_ASSERT_MSG(approx_eq(d.expansion_ratio, 1.0),
                        "T7d: no cascade expansion must give R=1.0");
    }
    // Verify ratio when throttle policy triggers: ratio value should still be correct
    {
        RouterPolicy tp = throttle_policy(3.0);
        auto d = decide_regime(make_stats(100, 4, 20, 50000), tp);
        // R = 20/4 = 5.0 → Throttle, ratio = 5.0
        TEST_ASSERT_MSG(d.regime == Regime::Throttle,
                        "T7e: R=5.0 >= 3.0 must throttle");
        TEST_ASSERT_MSG(approx_eq(d.expansion_ratio, 5.0),
                        "T7e: expansion_ratio must be 5.0");
    }
}

// ─── V2 helpers ──────────────────────────────────────────────────────────────

// Default v2 policy: disabled (deploy no-op).
static RouterPolicyV2 default_v2_policy() { return {}; }

// Enabled v2 policy with default field values.
static RouterPolicyV2 enabled_v2_policy() {
    RouterPolicyV2 p;
    p.enabled = true;
    return p;
}

static RequestFeatures make_features(bool is_agentic,
                                      int  prompt_tokens,
                                      int  new_content_tokens) {
    return { is_agentic, prompt_tokens, new_content_tokens };
}

// ─── T8: DEPLOY-NO-OP (v2) ───────────────────────────────────────────────────
// enabled=false → SAFE for every input, including is_agentic=true and huge prompts.
// Correct-by-construction: disabled router must be an exact no-op.

static void t8_v2_deploy_noop() {
    RouterPolicyV2 p = default_v2_policy();   // enabled=false

    // Baseline: normal agentic prompt, well above threshold.
    {
        auto d = decide_v2(make_features(true, 100000, 10000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T8a: disabled->keep_target must be full_keep_target");
        TEST_ASSERT_MSG(d.cascade,
                        "T8a: disabled->cascade must be true");
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T8a: disabled->recency must be keep-all sentinel");
    }
    // Sweep: all combinations of is_agentic, varying prompt and new_content sizes.
    for (int i = 0; i < 4; ++i) {
        bool agentic   = (i & 1) != 0;
        int  prompt    = (i & 2) ? 100000 : 500;
        int  new_toks  = (i & 2) ? 10000 : 10;
        auto d = decide_v2(make_features(agentic, prompt, new_toks), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T8-sweep: disabled->keep_target must be full_keep_target");
        TEST_ASSERT_MSG(d.cascade,
                        "T8-sweep: disabled->cascade must be true");
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T8-sweep: disabled->recency must be keep-all sentinel");
    }
    // Explicitly: is_agentic=true, large prompt, large new_content — must be SAFE.
    {
        auto d = decide_v2(make_features(true, 200000, 50000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T8b: disabled, agentic, huge prompt -> SAFE");
        TEST_ASSERT_MSG(d.cascade, "T8b: disabled -> cascade=true");
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T8b: disabled -> recency keep-all");
    }
}

// ─── T9: SPARSE-PROMPT GUARD (failure-class fix) ─────────────────────────────
// is_agentic=true, prompt above threshold, BUT new_content < sparse threshold.
// This is the LONG_A-t11/LONG_B-t10 plumbing class: a tiny tool_result riding
// on long history. Compression must NOT throttle here (would drop continuity).

static void t9_sparse_prompt_guard() {
    RouterPolicyV2 p = enabled_v2_policy();

    // Canonical failure case: 3-word tool_result on 43K history.
    {
        auto d = decide_v2(make_features(true, 43000, 8), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T9a: sparse agentic turn must be SAFE (full keep), not throttled");
        TEST_ASSERT_MSG(d.cascade,
                        "T9a: sparse_prompt_guard must cascade=true (SAFE)");
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T9a: sparse_prompt_guard -> recency keep-all");
        TEST_ASSERT_MSG(std::string(d.reason) == "sparse_prompt_guard",
                        "T9a: reason must be 'sparse_prompt_guard'");
    }
    // new_content = sparse_new_content_tokens - 1 (just below the guard).
    {
        auto d = decide_v2(make_features(true, 50000, p.sparse_new_content_tokens - 1), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T9b: new_content just below sparse threshold -> SAFE");
        TEST_ASSERT_MSG(std::string(d.reason) == "sparse_prompt_guard",
                        "T9b: reason must be 'sparse_prompt_guard'");
    }
    // new_content = 0 (degenerate new turn, still sparse guard NOT degenerate path).
    // Note: 0 < sparse_new_content_tokens (256) so sparse guard fires first.
    {
        auto d = decide_v2(make_features(true, 40000, 0), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T9c: new_content=0 -> SAFE (sparse guard or degenerate, both SAFE)");
    }
    // Confirm: new_content = sparse_new_content_tokens (AT the boundary → NOT sparse).
    // is_agentic=true above threshold with enough new content → throttle kicks in.
    {
        auto d = decide_v2(make_features(true, 50000, p.sparse_new_content_tokens), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.agentic_keep_target),
                        "T9d: new_content==sparse threshold -> agentic throttle applies");
        TEST_ASSERT_MSG(!d.cascade,
                        "T9d: agentic throttle -> cascade=false");
    }
}

// ─── T10: AGENTIC-THROTTLE ───────────────────────────────────────────────────
// enabled, is_agentic=true, prompt > threshold, new_content > sparse threshold
// → keep_target=agentic_keep_target, cascade=false, recency >= 1.

static void t10_agentic_throttle() {
    RouterPolicyV2 p = enabled_v2_policy();

    {
        auto d = decide_v2(make_features(true, 40000, 3000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.agentic_keep_target),
                        "T10a: agentic throttle -> keep_target=agentic_keep_target");
        TEST_ASSERT_MSG(!d.cascade,
                        "T10a: agentic throttle -> cascade=false");
        TEST_ASSERT_MSG(d.recency_floor_turns == p.recency_floor_turns,
                        "T10a: agentic throttle -> recency matches policy");
        TEST_ASSERT_MSG(d.recency_floor_turns >= 1,
                        "T10a: recency_floor_turns must be >= 1 (continuity guaranteed)");
        TEST_ASSERT_MSG(std::string(d.reason) == "agentic_throttle",
                        "T10a: reason must be 'agentic_throttle'");
    }
    // Custom policy: verify fields propagate.
    {
        RouterPolicyV2 p2  = p;
        p2.agentic_keep_target  = 0.30;
        p2.recency_floor_turns  = 5;
        auto d = decide_v2(make_features(true, 60000, 1000), p2);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, 0.30),
                        "T10b: custom agentic_keep_target propagated");
        TEST_ASSERT_MSG(d.recency_floor_turns == 5,
                        "T10b: custom recency_floor_turns propagated");
    }
}

// ─── T11: RETRIEVAL-FULL ─────────────────────────────────────────────────────
// enabled, is_agentic=false, prompt > threshold, new_content > sparse threshold
// → cascade=true, keep_target=full_keep_target.

static void t11_retrieval_full() {
    RouterPolicyV2 p = enabled_v2_policy();

    {
        auto d = decide_v2(make_features(false, 40000, 3000), p);
        TEST_ASSERT_MSG(d.cascade,
                        "T11a: retrieval -> cascade=true");
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T11a: retrieval -> keep_target=full_keep_target");
        TEST_ASSERT_MSG(std::string(d.reason) == "retrieval_full",
                        "T11a: reason must be 'retrieval_full'");
    }
    // Custom full_keep_target.
    {
        RouterPolicyV2 p2       = p;
        p2.full_keep_target     = 0.80;
        auto d = decide_v2(make_features(false, 50000, 5000), p2);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, 0.80),
                        "T11b: custom full_keep_target propagated");
        TEST_ASSERT_MSG(d.cascade, "T11b: retrieval -> cascade=true");
    }
}

// ─── T12: BELOW-THRESHOLD (v2) ───────────────────────────────────────────────
// prompt_tokens < threshold_tokens → SAFE regardless of is_agentic and new_content.

static void t12_v2_below_threshold() {
    RouterPolicyV2 p = enabled_v2_policy();

    // Agentic, just below threshold, plenty of new content.
    {
        auto d = decide_v2(make_features(true, p.threshold_tokens - 1, 5000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T12a: agentic, below threshold -> SAFE");
        TEST_ASSERT_MSG(d.cascade,
                        "T12a: below threshold -> cascade=true");
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T12a: below threshold -> recency keep-all");
        TEST_ASSERT_MSG(std::string(d.reason) == "below_threshold",
                        "T12a: reason must be 'below_threshold'");
    }
    // Non-agentic, at threshold boundary - 1.
    {
        auto d = decide_v2(make_features(false, p.threshold_tokens - 1, 5000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T12b: non-agentic, below threshold -> SAFE");
    }
    // Custom threshold.
    {
        RouterPolicyV2 p2     = p;
        p2.threshold_tokens   = 10000;
        auto d = decide_v2(make_features(true, 9999, 2000), p2);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p2.full_keep_target),
                        "T12c: custom threshold, below it -> SAFE");
        TEST_ASSERT_MSG(std::string(d.reason) == "below_threshold",
                        "T12c: reason must be 'below_threshold'");
    }
}

// ─── T13: RECENCY-FLOOR INVARIANT ────────────────────────────────────────────
// In every throttling decision (non-SAFE), recency_floor_turns >= 1.
// In every SAFE decision, recency_floor_turns >= kRecencyKeepAll.

static void t13_recency_floor_invariant() {
    RouterPolicyV2 p = enabled_v2_policy();

    // Throttle path (agentic): recency >= 1.
    {
        auto d = decide_v2(make_features(true, 50000, 1000), p);
        TEST_ASSERT_MSG(!approx_eq(d.keep_target, p.full_keep_target) ||
                         d.recency_floor_turns >= 1,
                        "T13a: throttled decision must have recency >= 1");
        TEST_ASSERT_MSG(d.recency_floor_turns >= 1,
                        "T13a: agentic throttle recency_floor_turns >= 1 (continuity)");
    }
    // SAFE paths: recency must be keep-all.
    // disabled
    {
        RouterPolicyV2 pd; pd.enabled = false;
        auto d = decide_v2(make_features(true, 50000, 1000), pd);
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T13b: disabled SAFE recency must be keep-all");
    }
    // sparse_prompt_guard
    {
        auto d = decide_v2(make_features(true, 50000, 5), p);
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T13c: sparse_prompt_guard SAFE recency must be keep-all");
    }
    // below_threshold
    {
        auto d = decide_v2(make_features(true, 1000, 500), p);
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T13d: below_threshold SAFE recency must be keep-all");
    }
    // retrieval_full path: recency = policy value (not keep-all, it's a throttle-adjacent path)
    {
        auto d = decide_v2(make_features(false, 50000, 1000), p);
        TEST_ASSERT_MSG(d.recency_floor_turns >= 1,
                        "T13e: retrieval_full recency >= 1");
    }
    // Custom recency_floor_turns: verify agentic propagates it.
    for (int k = 1; k <= 10; ++k) {
        RouterPolicyV2 pk          = p;
        pk.recency_floor_turns     = k;
        auto d = decide_v2(make_features(true, 50000, 1000), pk);
        TEST_ASSERT_MSG(d.recency_floor_turns == k,
                        "T13f: agentic throttle recency must equal policy recency_floor_turns");
    }
}

// ─── T14: DEGENERATE (v2) ────────────────────────────────────────────────────
// prompt_tokens <= 0 or new_content_tokens < 0 → SAFE (no crash, no garbage).

static void t14_v2_degenerate() {
    RouterPolicyV2 p = enabled_v2_policy();

    // prompt_tokens = 0
    {
        auto d = decide_v2(make_features(true, 0, 500), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14a: prompt_tokens=0 -> SAFE");
        TEST_ASSERT_MSG(d.cascade, "T14a: degenerate -> cascade=true");
        TEST_ASSERT_MSG(d.recency_floor_turns >= kRecencyKeepAll,
                        "T14a: degenerate -> recency keep-all");
        TEST_ASSERT_MSG(std::string(d.reason) == "degenerate",
                        "T14a: reason must be 'degenerate'");
    }
    // prompt_tokens < 0
    {
        auto d = decide_v2(make_features(false, -1, 100), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14b: negative prompt_tokens -> SAFE");
        TEST_ASSERT_MSG(std::string(d.reason) == "degenerate",
                        "T14b: reason must be 'degenerate'");
    }
    // new_content_tokens < 0
    {
        auto d = decide_v2(make_features(true, 50000, -1), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14c: negative new_content_tokens -> SAFE");
        TEST_ASSERT_MSG(std::string(d.reason) == "degenerate",
                        "T14c: reason must be 'degenerate'");
    }
    // Both degenerate
    {
        auto d = decide_v2(make_features(true, -5, -10), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14d: both degenerate -> SAFE");
    }
}

// ─── T15: RECENCY_FLOOR_FOR — off ────────────────────────────────────────────
// recency_floor_tokens == 0 → always 0 regardless of prompt size.

static void t15_recency_floor_off() {
    // 0 → off
    TEST_ASSERT_MSG(recency_floor_for(0,      0) == 0, "T15a: S=0 R=0 -> 0");
    TEST_ASSERT_MSG(recency_floor_for(1000,   0) == 0, "T15b: S=1000 R=0 -> 0");
    TEST_ASSERT_MSG(recency_floor_for(100000, 0) == 0, "T15c: S=100K R=0 -> 0");
    // Negative R (shouldn't happen but must be safe)
    TEST_ASSERT_MSG(recency_floor_for(10000, -2) == 0, "T15d: negative R (not sentinel) -> 0");
}

// ─── T16: RECENCY_FLOOR_FOR — auto ───────────────────────────────────────────
// kRecencyFloorAuto (-1) → min(1024, ceil(0.04 * S)).

static void t16_recency_floor_auto() {
    const int A = kRecencyFloorAuto;

    // S=0: ceil(0.04*0)=0
    TEST_ASSERT_MSG(recency_floor_for(0, A) == 0,   "T16a: S=0 auto -> 0");
    // S=1000: ceil(0.04*1000)=40
    TEST_ASSERT_MSG(recency_floor_for(1000,  A) == 40,  "T16b: S=1000 auto -> 40");
    // S=10000: ceil(0.04*10000)=400
    TEST_ASSERT_MSG(recency_floor_for(10000, A) == 400, "T16c: S=10K auto -> 400");
    // S=25000: ceil(0.04*25000)=1000
    TEST_ASSERT_MSG(recency_floor_for(25000, A) == 1000, "T16d: S=25K auto -> 1000");
    // S=25001: ceil(0.04*25001)=1001 but capped at 1024
    // actually 0.04*25001=1000.04 → ceil=1001 < 1024 → 1001
    TEST_ASSERT_MSG(recency_floor_for(25001, A) == 1001, "T16e: S=25001 auto -> 1001");
    // S=25600: 0.04*25600=1024.0 → ceil=1024
    TEST_ASSERT_MSG(recency_floor_for(25600, A) == 1024, "T16f: S=25600 auto -> 1024");
    // S=26000: 0.04*26000=1040 → ceil=1040 but capped at 1024
    TEST_ASSERT_MSG(recency_floor_for(26000, A) == 1024, "T16g: S=26000 auto -> cap 1024");
    // S=100000: 0.04*100000=4000 → capped at 1024
    TEST_ASSERT_MSG(recency_floor_for(100000, A) == 1024, "T16h: S=100K auto -> cap 1024");
    // S=-1: negative prompt treated as 0 → 0
    TEST_ASSERT_MSG(recency_floor_for(-1, A) == 0,  "T16i: S=-1 auto -> 0");
}

// ─── T17: RECENCY_FLOOR_FOR — explicit ───────────────────────────────────────
// Any explicit positive value is returned unchanged (no prompt-size influence).

static void t17_recency_floor_explicit() {
    // Explicit override ignores prompt size
    TEST_ASSERT_MSG(recency_floor_for(1000,  512)  == 512,  "T17a: explicit 512");
    TEST_ASSERT_MSG(recency_floor_for(100000, 512) == 512,  "T17b: explicit 512, large S");
    TEST_ASSERT_MSG(recency_floor_for(1000, 1024)  == 1024, "T17c: explicit 1024");
    TEST_ASSERT_MSG(recency_floor_for(1000, 2048)  == 2048, "T17d: explicit 2048 > cap");
    TEST_ASSERT_MSG(recency_floor_for(0,    256)   == 256,  "T17e: explicit 256, S=0");
    // Monotone: explicit > auto at short prompts
    const int A = kRecencyFloorAuto;
    TEST_ASSERT_MSG(recency_floor_for(1000, 512) > recency_floor_for(1000, A),
                    "T17f: explicit 512 > auto(1000)=40");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== test_regime_router ===\n");

    RUN_TEST(t1_deploy_noop);
    RUN_TEST(t2_degenerate);
    RUN_TEST(t3_below_threshold);
    RUN_TEST(t4_too_few_anchors);
    RUN_TEST(t5_monotone);
    RUN_TEST(t6_boundary);
    RUN_TEST(t7_ratio_value);

    std::fprintf(stderr, "--- v2 ---\n");
    RUN_TEST(t8_v2_deploy_noop);
    RUN_TEST(t9_sparse_prompt_guard);
    RUN_TEST(t10_agentic_throttle);
    RUN_TEST(t11_retrieval_full);
    RUN_TEST(t12_v2_below_threshold);
    RUN_TEST(t13_recency_floor_invariant);
    RUN_TEST(t14_v2_degenerate);

    std::fprintf(stderr, "--- recency_floor_for ---\n");
    RUN_TEST(t15_recency_floor_off);
    RUN_TEST(t16_recency_floor_auto);
    RUN_TEST(t17_recency_floor_explicit);

    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
