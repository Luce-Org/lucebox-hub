#!/usr/bin/env bash
# Multi-client baseline vs slim prefill bench.
# Locked stack: ee7 + Q3_K_S target + adaptive bandit + skip-park.
# Slim adds PFLASH_DRAFTER_SLIM=1.

set -euo pipefail

TARGET=/home/peppi/models/qwen3.6-27b-q3ks/Qwen3.6-27B-Q3_K_S.gguf
DRAFT=/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf
DRAFTER=/home/peppi/models/Qwen3-0.6B-BF16.gguf
SERVER=/home/peppi/Dev/lucebox-hub/dflash/build/dflash_server
HARNESS=/home/peppi/Dev/lucebox-hub/.claude/worktrees/harness-adapters
OUTDIR=/home/peppi/Dev/lucebox-hub/bench/2026-05-23_multiclient_slim_prefill

CLIENTS=(claude_code hermes codex pi opencode)
CONDITIONS=(baseline slim)

export DFLASH_DRAFTER_EARLY_EXIT_N=7
export DFLASH_DRAFTER_SCORE_LAYERS=7
export DFLASH_SERVER_BIN=$SERVER

for condition in "${CONDITIONS[@]}"; do
    if [[ "$condition" == "slim" ]]; then
        export PFLASH_DRAFTER_SLIM=1
    else
        unset PFLASH_DRAFTER_SLIM
    fi

    outdir="$OUTDIR/$condition"
    mkdir -p "$outdir"

    for client in "${CLIENTS[@]}"; do
        echo "=== Running $client / $condition ==="
        csv="$outdir/${client}.csv"
        server_log="$outdir/${client}_server.log"
        run_log="$outdir/${client}_run.log"

        cd "$HARNESS"
        python3 -m harness.client_test_runner bandit-session \
            --client "$client" \
            --turns 3 \
            --target "$TARGET" \
            --draft "$DRAFT" \
            --bin "$SERVER" \
            --pflash-drafter "$DRAFTER" \
            --output "$csv" \
            2>&1 | tee "$run_log" || true

        # Copy server log from the results dir
        latest_results=$(ls -td dflash/bench/results/*_adaptive_evidence 2>/dev/null | head -1)
        if [[ -n "$latest_results" && -f "$latest_results/server.log" ]]; then
            cp "$latest_results/server.log" "$server_log"
        fi

        echo "  Done. Run log: $run_log"
        echo "  Server log: $server_log"
        sleep 5
    done
done

echo "=== All clients done. Parsing prefill metrics ==="
python3 /home/peppi/Dev/lucebox-hub/bench/2026-05-23_multiclient_slim_prefill/parse_prefill.py
