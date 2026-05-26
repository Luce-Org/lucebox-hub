#!/usr/bin/env bash
# lucebox.sh — host-side wrapper for the lucebox-hub container.
#
# Two jobs:
#
#   1) Probe the host (driver, docker, NVIDIA Container Toolkit, VRAM, RAM,
#      systemd), select the CUDA 12 image, and
#      dispatch into the in-container Python CLI via `docker run`. The
#      Python CLI lives at /opt/lucebox-hub/lucebox/ inside the image and is
#      the single source of truth for orchestration logic — TOML config,
#      autotune rules, server bring-up, benchmark sweeps, smoke tests, model
#      downloads.
#
#   2) Manage a user-level systemd unit (~/.config/systemd/user/lucebox.service)
#      so the server can run as a long-lived service without keeping a shell
#      open. install/uninstall/start/stop/enable/disable/status/logs all
#      delegate to systemctl --user / journalctl --user.
#
# Install:
#   curl -fsSL https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/lucebox.sh \
#        -o ~/.local/bin/lucebox.sh && chmod +x ~/.local/bin/lucebox.sh
#
# Then: lucebox.sh check && lucebox.sh install && lucebox.sh start
#
# No root is ever taken automatically. Anything that needs sudo (package
# install, loginctl enable-linger) is printed for the user to run.

set -euo pipefail

VERSION="0.2.0"
SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

# ── tunables / env overrides ───────────────────────────────────────────────
IMAGE_BASE="${LUCEBOX_IMAGE:-ghcr.io/luce-org/lucebox-hub}"
CONTAINER_NAME="${LUCEBOX_CONTAINER:-lucebox}"
DEFAULT_PORT="${LUCEBOX_PORT:-8080}"
UNIT_NAME="lucebox.service"
UNIT_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT_NAME"

# CUDA driver floor for the prebuilt CUDA 12 image.
# shellcheck disable=SC2034
MIN_DRIVER_CUDA12=525

# ── output helpers ────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_INFO='\033[1;34m'; C_OK='\033[1;32m'; C_WARN='\033[1;33m'
    C_ERR='\033[1;31m'; C_DIM='\033[2m'; C_RST='\033[0m'
else
    C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''; C_DIM=''; C_RST=''
fi

info()  { printf '%b[INFO]%b  %s\n' "$C_INFO" "$C_RST" "$*"; }
ok()    { printf '%b[OK]%b    %s\n' "$C_OK"   "$C_RST" "$*"; }
warn()  { printf '%b[WARN]%b  %s\n' "$C_WARN" "$C_RST" "$*"; }
err()   { printf '%b[ERROR]%b %s\n' "$C_ERR"  "$C_RST" "$*" >&2; }
hint()  { printf '       %b%s%b\n'  "$C_DIM"  "$*"     "$C_RST"; }
die()   { err "$*"; exit 1; }

# ── host probing ──────────────────────────────────────────────────────────
# Sets the LUCEBOX_HOST_* variables consumed by the in-container Python CLI
# (passed through with -e). The Python side trusts these and doesn't reprobe
# — it can't see the host's /proc anyway, only the container's.

probe_host() {
    LUCEBOX_HOST_NPROC=$(nproc 2>/dev/null || echo 1)
    LUCEBOX_HOST_RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    LUCEBOX_HOST_GPU_VENDOR="none"
    LUCEBOX_HOST_GPU_NAME=""
    LUCEBOX_HOST_GPU_COUNT=0
    LUCEBOX_HOST_VRAM_GB=0
    LUCEBOX_HOST_GPU_SM=""
    LUCEBOX_HOST_DRIVER_VERSION=""
    LUCEBOX_HOST_DRIVER_MAJOR=0

    if command -v nvidia-smi &>/dev/null; then
        local q
        if q=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
                          --format=csv,noheader,nounits 2>/dev/null) && [ -n "$q" ]; then
            LUCEBOX_HOST_GPU_VENDOR="nvidia"
            LUCEBOX_HOST_GPU_NAME=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $1}')
            local mem_mib
            mem_mib=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $2}')
            LUCEBOX_HOST_VRAM_GB=$((mem_mib / 1024))
            LUCEBOX_HOST_DRIVER_VERSION=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $3}')
            LUCEBOX_HOST_DRIVER_MAJOR=${LUCEBOX_HOST_DRIVER_VERSION%%.*}
            local cc
            cc=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $4}')
            LUCEBOX_HOST_GPU_SM="${cc//./}"
            LUCEBOX_HOST_GPU_COUNT=$(printf '%s\n' "$q" | wc -l)
        fi
    fi

    LUCEBOX_HOST_HAS_SYSTEMD=0
    if command -v systemctl &>/dev/null && systemctl --user show-environment &>/dev/null; then
        LUCEBOX_HOST_HAS_SYSTEMD=1
    fi

    LUCEBOX_HOST_IS_WSL=0
    if grep -qi microsoft /proc/version 2>/dev/null \
       || [ -e /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        LUCEBOX_HOST_IS_WSL=1
    fi

    LUCEBOX_HOST_HAS_DOCKER=0
    LUCEBOX_HOST_DOCKER_VERSION=""
    if command -v docker &>/dev/null && docker ps &>/dev/null; then
        LUCEBOX_HOST_HAS_DOCKER=1
        LUCEBOX_HOST_DOCKER_VERSION=$(timeout 5 docker version --format '{{.Server.Version}}' 2>/dev/null || echo "")
    fi

    LUCEBOX_HOST_HAS_CTK="none"
    if [ "$LUCEBOX_HOST_HAS_DOCKER" = "1" ]; then
        if command -v nvidia-container-runtime &>/dev/null; then
            LUCEBOX_HOST_HAS_CTK="runtime"
        elif command -v nvidia-ctk &>/dev/null \
             && nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu'; then
            LUCEBOX_HOST_HAS_CTK="cdi"
        elif command -v nvidia-ctk &>/dev/null; then
            LUCEBOX_HOST_HAS_CTK="installed-unwired"
        fi
    fi

    export LUCEBOX_HOST_NPROC LUCEBOX_HOST_RAM_GB LUCEBOX_HOST_GPU_VENDOR
    export LUCEBOX_HOST_GPU_NAME LUCEBOX_HOST_GPU_COUNT LUCEBOX_HOST_VRAM_GB
    export LUCEBOX_HOST_GPU_SM LUCEBOX_HOST_DRIVER_VERSION LUCEBOX_HOST_DRIVER_MAJOR
    export LUCEBOX_HOST_HAS_SYSTEMD LUCEBOX_HOST_IS_WSL
    export LUCEBOX_HOST_HAS_DOCKER LUCEBOX_HOST_DOCKER_VERSION
    export LUCEBOX_HOST_HAS_CTK
}

pick_variant() {
    # CUDA 12.8 is the supported image for this branch.
    echo "${LUCEBOX_VARIANT:-cuda12}"
}

# ── prereq checks (host-only) ─────────────────────────────────────────────
# Print-and-exit on anything that needs root to install. The Python CLI does
# the richer reporting; this is the bare minimum to make `docker run` viable.

require_host_prereqs() {
    local missing=0
    if ! command -v docker &>/dev/null; then
        err "docker is not installed"
        hint "Install: https://docs.docker.com/engine/install/"
        missing=1
    elif ! docker ps &>/dev/null; then
        err "docker daemon not reachable"
        hint "sudo systemctl start docker   (or: add your user to the 'docker' group, then re-login)"
        missing=1
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        err "nvidia-smi not found — no NVIDIA driver detected"
        hint "Install the NVIDIA driver: https://www.nvidia.com/Download/index.aspx"
        missing=1
    elif ! nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
        err "nvidia-smi present but NVML calls fail — likely a driver/library mismatch"
        hint "Reboot, or reinstall the matching NVIDIA driver package"
        missing=1
    fi

    [ "$missing" = "0" ] || exit 1
}

require_ctk() {
    case "$LUCEBOX_HOST_HAS_CTK" in
        runtime|cdi) return 0 ;;
        installed-unwired)
            err "NVIDIA Container Toolkit installed but not wired into docker"
            hint "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            hint "  or generate a CDI spec: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
            exit 1 ;;
        none|*)
            err "NVIDIA Container Toolkit not installed"
            hint "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
            hint "Then register with docker:"
            hint "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            exit 1 ;;
    esac
}

require_systemd() {
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" != "1" ]; then
        err "user systemd is not available — required for $1"
        hint "On WSL: set 'systemd=true' under [boot] in /etc/wsl.conf, then 'wsl --shutdown'."
        hint "Otherwise: install systemd, or run 'lucebox.sh serve' to run in the foreground without systemd."
        exit 1
    fi
}

# ── docker run construction ───────────────────────────────────────────────
# All the Python-CLI subcommands share the same docker run incantation:
# mount the host docker socket (so the in-container CLI can spawn server /
# bench containers on the host daemon), mount $HOME at the same path (so
# paths look identical in and out), and pass host facts via env. When an
# NVIDIA GPU is detected we also pass --gpus all so the orchestrator can
# call nvidia-smi during profile snapshot export; without it nvidia_smi_csv (and
# any downstream power/utilization fields) come back empty.

DOCKER_SOCK_PATH="${DOCKER_HOST:-/var/run/docker.sock}"
DOCKER_SOCK_PATH="${DOCKER_SOCK_PATH#unix://}"

build_orchestrator_argv() {
    local variant="$1"; shift
    local tty=()
    if [ -t 0 ] && [ -t 1 ]; then
        tty=(-it)
    else
        tty=(-i)
    fi
    local argv=(docker run --rm "${tty[@]}")
    if [ "${LUCEBOX_HOST_GPU_VENDOR:-none}" = "nvidia" ]; then
        argv+=(--gpus all)
    fi
    argv+=(--name "${CONTAINER_NAME}-cli-$$")
    argv+=(--user "$(id -u):$(id -g)")
    if [ -S "$DOCKER_SOCK_PATH" ]; then
        argv+=(--group-add "$(stat -c '%g' "$DOCKER_SOCK_PATH")")
    fi
    argv+=(-v "$DOCKER_SOCK_PATH:/var/run/docker.sock")
    argv+=(-v "$HOME:$HOME")
    argv+=(-w "$PWD")
    argv+=(-e "HOME=$HOME")
    # Host facts — Python side reads these instead of reprobing.
    local var
    for var in $(compgen -e | grep '^LUCEBOX_HOST_' || true); do
        argv+=(-e "$var=${!var}")
    done
    # User overrides for image/port/container name propagate too.
    argv+=(-e "LUCEBOX_IMAGE=$IMAGE_BASE")
    argv+=(-e "LUCEBOX_VARIANT=$variant")
    argv+=(-e "LUCEBOX_PORT=$DEFAULT_PORT")
    argv+=(-e "LUCEBOX_CONTAINER=$CONTAINER_NAME")
    [ -n "${LUCEBOX_MODELS:-}" ] && argv+=(-e "LUCEBOX_MODELS=$LUCEBOX_MODELS")
    [ -n "${HF_TOKEN:-}" ] && argv+=(-e "HF_TOKEN=$HF_TOKEN")

    argv+=("${IMAGE_BASE}:${variant}")
    # `lucebox` is the entrypoint subcommand handled by server/scripts/entrypoint.sh
    # — it execs `python -m lucebox` with whatever args we pass on.
    argv+=(lucebox "$@")
    printf '%s\n' "${argv[@]}"
}

# ── subcommand implementations ────────────────────────────────────────────

cmd_serve() {
    # Long-running foreground server. Also what systemd's ExecStart= calls.
    #
    # Two-stage so config.toml takes effect:
    #   1. Run an ephemeral orchestrator container that emits the canonical
    #      server docker-run argv from .lucebox/config.toml (one arg per
    #      line on stdout).
    #   2. Exec that argv.
    #
    # If stage 1 fails (image not pulled yet, no config), fall back to a
    # conservative docker run — the container's own VRAM-tiered autotune
    # picks reasonable defaults from there.
    require_host_prereqs
    probe_host
    require_ctk
    local variant
    variant=$(pick_variant)

    local orch_argv server_argv
    mapfile -t orch_argv < <(build_orchestrator_argv "$variant" print-serve-argv)

    if mapfile -t server_argv < <("${orch_argv[@]}" 2>/dev/null) \
       && [ "${#server_argv[@]}" -gt 0 ] \
       && [ "${server_argv[0]}" = "docker" ]; then
        info "Starting lucebox server (variant=$variant, from config.toml)"
        exec "${server_argv[@]}"
    fi

    warn "Couldn't fetch server argv from container (image not pulled?) — using fallback"
    info "Starting lucebox server (variant=$variant, port=$DEFAULT_PORT, defaults only)"
    local fallback_models="${LUCEBOX_MODELS:-$HOME/models}"
    exec docker run --rm \
        --name "$CONTAINER_NAME" \
        --gpus all \
        -p "$DEFAULT_PORT:8080" \
        -v "$HOME:$HOME" \
        -v "$fallback_models:/opt/lucebox-hub/server/models" \
        "${IMAGE_BASE}:${variant}"
}

cmd_systemd_install() {
    require_host_prereqs
    probe_host
    require_systemd "service install"
    local docker_bin
    docker_bin=$(command -v docker)

    mkdir -p "$(dirname "$UNIT_PATH")"
    cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Lucebox hub LLM inference server
Documentation=https://github.com/Luce-Org/lucebox-hub
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=exec
Restart=on-failure
RestartSec=10
ExecStart=$SCRIPT_PATH serve
ExecStop=$docker_bin stop -t 30 $CONTAINER_NAME
TimeoutStopSec=45

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    ok "Installed $UNIT_PATH"

    # Check for linger — without it, the unit dies when the user logs out.
    local linger
    linger=$(loginctl show-user "$USER" 2>/dev/null | awk -F= '/^Linger=/{print $2}')
    if [ "$linger" != "yes" ]; then
        warn "Linger is off for $USER — the service will stop when you log out"
        hint "To enable (requires sudo): sudo loginctl enable-linger \"$USER\""
    fi

    printf '\nNext:\n'
    hint "  $SCRIPT_NAME start            # start now"
    hint "  $SCRIPT_NAME enable           # start at every login"
    hint "  $SCRIPT_NAME logs             # follow the journal"
}

cmd_systemd_uninstall() {
    require_systemd "service uninstall"
    if systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
        info "Stopping $UNIT_NAME"
        systemctl --user stop "$UNIT_NAME" || true
    fi
    if systemctl --user is-enabled --quiet "$UNIT_NAME" 2>/dev/null; then
        info "Disabling $UNIT_NAME"
        systemctl --user disable "$UNIT_NAME" || true
    fi
    if [ -f "$UNIT_PATH" ]; then
        rm -f "$UNIT_PATH"
        ok "Removed $UNIT_PATH"
    else
        info "No unit at $UNIT_PATH — nothing to remove"
    fi
    systemctl --user daemon-reload
    hint "Config and models are left in place. Remove them by hand if you want."
}

cmd_systemctl_passthrough() {
    local action="$1"
    require_systemd "$action"
    if [ ! -f "$UNIT_PATH" ]; then
        err "$UNIT_NAME is not installed — run '$SCRIPT_NAME install' first"
        exit 1
    fi
    case "$action" in
        start|stop|restart|enable|disable)
            exec systemctl --user "$action" "$UNIT_NAME" ;;
        status)
            exec systemctl --user status "$UNIT_NAME" --no-pager ;;
        *)
            die "unknown systemctl passthrough: $action" ;;
    esac
}

cmd_logs() {
    require_systemd "logs"
    # Pure passthrough: any flags the user wants (-f, -n, --since, ...) go
    # straight to journalctl. Default is follow.
    if [ $# -eq 0 ]; then
        exec journalctl --user -u "$UNIT_NAME" -f
    fi
    exec journalctl --user -u "$UNIT_NAME" "$@"
}

cmd_pull() {
    # Pull has to run on the host. Delegating this into the container creates a
    # stale-image trap: docker may start an old local tag before the fresh tag
    # has been pulled.
    require_host_prereqs
    local variant
    variant=$(pick_variant)
    info "Pulling ${IMAGE_BASE}:${variant}"
    exec docker pull "${IMAGE_BASE}:${variant}"
}

cmd_in_container() {
    # Generic dispatcher: anything that isn't a systemd action goes here.
    # Runs the in-container Python CLI with the supplied argv.
    require_host_prereqs
    probe_host
    # CTK isn't strictly required for every subcommand (e.g. `configure`
    # only writes a TOML file), but the server-spawning subcommands need it.
    # Letting docker error its own way is fine for the no-CTK case.
    local variant
    variant=$(pick_variant)
    local argv
    mapfile -t argv < <(build_orchestrator_argv "$variant" "$@")
    exec "${argv[@]}"
}

usage() {
    cat <<EOF
lucebox.sh $VERSION — host-side wrapper for the lucebox-hub container

Service management (via user systemd):
  install               install user systemd unit
  uninstall             stop, disable, remove the unit (keeps config + models)
  start | stop          systemctl --user start|stop lucebox
  enable | disable      systemctl --user enable|disable lucebox
  status                systemctl --user status lucebox
  logs [args]           journalctl --user -u lucebox  (default: -f)

Direct server invocation (foreground, no systemd):
  serve                 docker run the server in the foreground

Provisioning + workloads (delegated to the in-container Python CLI):
  check                 host + docker readiness report
  configure             write .lucebox/config.toml (heuristic)
  pull                  docker pull the cuda12 image
  download-models       fetch target GGUF + DFlash draft (via the container)
  smoke                 hit /v1/chat/completions on a running server
  benchmark             sweep DFLASH_* knobs inside the container, write tuned config
  profile               update profile evidence; use --export-snapshot to export
  print-run             print the docker-run command for the server

Misc:
  help, --help, -h      this message
  version, --version    print version

Environment overrides:
  LUCEBOX_IMAGE         image name without tag (default: ghcr.io/luce-org/lucebox-hub)
  LUCEBOX_VARIANT       image tag to pull/run (default: cuda12)
  LUCEBOX_PORT          host port for the server (default: 8080)
  LUCEBOX_CONTAINER     server container name (default: lucebox)
  LUCEBOX_MODELS        host model directory (default: ~/models)
  HF_TOKEN              propagated to download-models for gated HF repos
EOF
}

# ── dispatch ──────────────────────────────────────────────────────────────

main() {
    local cmd="${1:-help}"
    [ $# -gt 0 ] && shift
    case "$cmd" in
        # Systemd surface
        install)          cmd_systemd_install "$@" ;;
        uninstall)        cmd_systemd_uninstall "$@" ;;
        start|stop|restart|enable|disable|status)
                          cmd_systemctl_passthrough "$cmd" "$@" ;;
        logs)             cmd_logs "$@" ;;

        # Direct server
        serve)            cmd_serve "$@" ;;
        pull)             cmd_pull "$@" ;;

        # Help / version
        help|--help|-h)   usage ;;
        version|--version) printf '%s\n' "$VERSION" ;;

        # Everything else → in-container Python CLI
        *)                cmd_in_container "$cmd" "$@" ;;
    esac
}

main "$@"
