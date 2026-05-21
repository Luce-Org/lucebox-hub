#!/usr/bin/env bash
# Day 4: A/B/C bandit vs fixed-keep validation.
# Each condition gets its own flock, starts a fresh server, runs one request, tears down.
set -euo pipefail

WORKTREE="/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto"
RESULTS_DIR="$WORKTREE/dflash/bench/results/2026-05-21_mvp_day4_v2"
SERVER_BIN="$WORKTREE/dflash/build/dflash_server"
TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf"
PFLASH_DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
CLAUDE_BIN="${CLAUDE_BIN:-/home/peppi/.local/bin/claude}"
PROMPT_FILE="$WORKTREE/harness/clients/prompts/decode_check.txt"
MARKER="OK_DONE"
CLAUDE_TIMEOUT=600

HOST=127.0.0.1
PORT=18080
MODEL_ID="luce-dflash"
API_KEY="sk-lucebox"
BASE_URL="http://$HOST:$PORT"

mkdir -p "$RESULTS_DIR"
echo "=== Day 4 A/B/C start $(date -Is) ===" | tee "$RESULTS_DIR/run.log"

# ─── run_condition ──────────────────────────────────────────────────────────
# Args: LABEL KEEP_RATIO SESSION_ID(or empty)
run_condition() {
    local label="$1"
    local keep="$2"
    local sid="$3"
    local cdir="$RESULTS_DIR/$label"
    mkdir -p "$cdir"

    local slog="$cdir/server.log"
    local cout="$cdir/client.out"
    local mfile="$cdir/metrics.txt"

    echo "--- [$label] keep=$keep sid='$sid' $(date -Is) ---" | tee -a "$RESULTS_DIR/run.log"
    local t0; t0=$(date +%s)

    flock /tmp/dflash_gpu.lock bash <<INNER
set -euo pipefail
export DFLASH27B_KV_K=tq3_0
export DFLASH27B_KV_V=tq3_0
export GGML_CUDA_NO_VMM=1

# Start server
"$SERVER_BIN" "$TARGET" \
    --draft "$DRAFT" \
    --prefill-drafter "$PFLASH_DRAFTER" \
    --host $HOST --port $PORT \
    --max-ctx 98304 --max-tokens 512 \
    --model-name "$MODEL_ID" \
    --ddtree --ddtree-budget 16 \
    --prefill-compression always \
    --prefill-keep-ratio $keep \
    > "$slog" 2>&1 &
SPID=\$!

# Wait for health
for i in \$(seq 1 120); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    sleep 1
    if ! kill -0 "\$SPID" 2>/dev/null; then
        echo "server died" >&2; tail -n 40 "$slog" >&2; exit 1
    fi
    if [[ \$i -eq 120 ]]; then echo "server timeout" >&2; exit 1; fi
done
echo "server up (pid=\$SPID)"

PROMPT="\$(<"$PROMPT_FILE")"

if [[ -n "$sid" ]]; then
    # Bandit path: inject session_id via extra_body
    PAYLOAD=\$(jq -n --arg p "\$PROMPT" --arg sid "$sid" \
        '{model:"luce-dflash",max_tokens:512,messages:[{role:"user",content:\$p}],extra_body:{session_id:\$sid}}')
    curl -s -X POST "$BASE_URL/v1/messages" \
        -H "Content-Type: application/json" \
        -H "x-api-key: $API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -d "\$PAYLOAD" > "$cout" 2>&1 || true
else
    # Fixed path: use claude CLI
    mkdir -p "$cdir/claude_home"
    HOME="$cdir/claude_home" \
    ANTHROPIC_API_KEY="$API_KEY" \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    CLAUDE_CODE_API_BASE_URL="$BASE_URL" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    CLAUDE_CODE_DISABLE_TELEMETRY=1 \
    CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
    timeout ${CLAUDE_TIMEOUT}s "$CLAUDE_BIN" \
        --print --output-format json \
        --model "$MODEL_ID" --tools none \
        --permission-mode dontAsk --no-session-persistence \
        "\$PROMPT" </dev/null > "$cout" 2>&1 || true
fi

kill "\$SPID" 2>/dev/null || true
wait "\$SPID" 2>/dev/null || true
INNER

    local t1; t1=$(date +%s)
    local wall=$((t1 - t0))

    # OK_DONE marker
    local ok_done="NO"
    if grep -q "$MARKER" "$cout" 2>/dev/null; then ok_done="YES"; fi

    # accept_rate from JSON response
    local ar
    ar=$(python3 -c "
import json, sys
try:
    d=json.load(open('$cout'))
    ar=d.get('usage',{}).get('accept_rate','N/A')
except:
    ar='N/A'
print(ar)" 2>/dev/null || echo "N/A")

    # bandit log lines
    local bandit; bandit=$(grep '\[pflash-bandit\]' "$slog" 2>/dev/null || echo "none")

    # drafter_fwd timing (ms)
    local dfwd; dfwd=$(grep -oP '\[drafter\] forward\+score \K[0-9.]+' "$slog" 2>/dev/null | \
        awk '{s+=$1;n++}END{if(n)printf "%.1f (n=%d)",s/n,n;else print "N/A"}' || echo "N/A")

    {
        echo "label=$label"
        echo "keep_ratio=$keep"
        echo "session_id=$sid"
        echo "wall_s=$wall"
        echo "ok_done=$ok_done"
        echo "accept_rate=$ar"
        echo "mean_drafter_fwd_ms=$dfwd"
        echo "bandit_log:"
        echo "$bandit"
    } | tee "$mfile" | tee -a "$RESULTS_DIR/run.log"

    echo "[$label] wall=${wall}s ok=$ok_done ar=$ar" | tee -a "$RESULTS_DIR/run.log"
}

# ─── Run the three conditions ────────────────────────────────────────────────
run_condition "A_fixed_low"  "0.05" ""
run_condition "B_fixed_high" "0.20" ""
run_condition "C_bandit"     "0.10" "claude_code_s1"

echo "=== Day 4 done $(date -Is) ===" | tee -a "$RESULTS_DIR/run.log"

# ─── Print summary table ─────────────────────────────────────────────────────
echo ""
echo "=== SUMMARY ==="
printf "%-18s %10s %8s %12s %8s  %s\n" "Condition" "wall_s" "ok_done" "accept_rate" "keep" "bandit"
for cond in A_fixed_low B_fixed_high C_bandit; do
    mf="$RESULTS_DIR/$cond/metrics.txt"
    if [[ -f "$mf" ]]; then
        wall=$(grep "^wall_s=" "$mf" | cut -d= -f2)
        ok=$(grep "^ok_done=" "$mf" | cut -d= -f2)
        ar=$(grep "^accept_rate=" "$mf" | cut -d= -f2)
        keep=$(grep "^keep_ratio=" "$mf" | cut -d= -f2)
        sid=$(grep "^session_id=" "$mf" | cut -d= -f2)
        bandit_note=""
        if [[ -n "$sid" ]]; then bandit_note="yes"; else bandit_note="-"; fi
        printf "%-18s %10s %8s %12s %8s  %s\n" "$cond" "$wall" "$ok" "$ar" "$keep" "$bandit_note"
    fi
done
