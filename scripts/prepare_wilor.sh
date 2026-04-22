#!/usr/bin/env bash
# =============================================================================
# prepare_wilor.sh
# Set up WiLoR submodule and download AnyHand WiLoR checkpoint.
# Run from the repo root:  bash scripts/prepare_wilor.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' not found. Please install it first."
}

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        info "Already exists, skipping: $dest"
        return 0
    fi
    info "Downloading: $dest"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "$url" -O "$dest"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar "$url" -o "$dest"
    else
        die "Neither wget nor curl found. Install one of them."
    fi
}

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
require_cmd git
require_cmd python

# ---------------------------------------------------------------------------
# 1. Initialise WiLoR submodule
# ---------------------------------------------------------------------------
info "=== [1/4] Initialising WiLoR submodule ==="

if [ ! -f "WiLoR/.git" ] && [ ! -d "WiLoR/.git" ]; then
    # Not yet initialised — try submodule update first, fall back to add
    if git submodule status WiLoR 2>/dev/null | grep -q '^-'; then
        info "Running: git submodule update --init WiLoR"
        git submodule update --init WiLoR
    elif [ ! -d "WiLoR" ] || [ -z "$(ls -A WiLoR 2>/dev/null)" ]; then
        info "WiLoR submodule not configured yet — adding it now."
        git submodule add https://github.com/rolpotamias/WiLoR.git WiLoR
        git submodule update --init WiLoR
    else
        warn "WiLoR directory exists but is not a submodule. Skipping submodule init."
    fi
else
    info "WiLoR submodule already initialised."
fi

# ---------------------------------------------------------------------------
# 2. Install WiLoR Required Dependencies
# ---------------------------------------------------------------------------
info "=== [2/4] Installing WiLoR Required Dependencies ==="

pip install -q -r WiLoR/requirements.txt
info "WiLoR dependencies installed."


# ---------------------------------------------------------------------------
# 3. Download WiLoR hand detector
# ---------------------------------------------------------------------------
info "=== [3/4] Downloading WiLoR hand detector ==="

WILOR_HF="https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models"
download "${WILOR_HF}/detector.pt" "pretrained_models/detector.pt"

# ---------------------------------------------------------------------------
# 4. Download AnyHand WiLoR checkpoint + config
# ---------------------------------------------------------------------------
info "=== [4/4] Downloading AnyHand WiLoR checkpoint ==="

mkdir -p pretrained_models

HF_USER="chen-si-02"
HF_REPO="AnyHand-Models"
HF_BASE="https://huggingface.co/${HF_USER}/${HF_REPO}/resolve/main"

if [ "$HF_USER" = "<YOUR_HF_USERNAME>" ]; then
    warn "HF_USER is still a placeholder. Edit this script and set your HuggingFace username."
    warn "Skipping checkpoint download."
else
    download "${HF_BASE}/anyhand_wilor.ckpt"          "pretrained_models/anyhand_wilor.ckpt"
    download "${HF_BASE}/model_config_wilor.yaml"     "pretrained_models/model_config_wilor.yaml"
fi


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  WiLoR setup complete."
echo ""
echo "  pretrained_models/"
echo "  ├── anyhand_wilor.ckpt       ← AnyHand fine-tuned WiLoR"
echo "  ├── model_config_wilor.yaml  ← matching config"
echo "  └── detector.pt              ← hand detector"
echo ""
echo "  ACTION REQUIRED:"
echo "  Place MANO_RIGHT.pkl at:  mano_data/MANO_RIGHT.pkl"
echo "  Download from: https://mano.is.tue.mpg.de/"
echo "============================================================"
