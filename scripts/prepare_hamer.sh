#!/usr/bin/env bash
# =============================================================================
# prepare_hamer.sh
# Set up HaMeR (under third_party/hamer) and download AnyHand HaMeR checkpoint.
# Run from the repo root:  bash scripts/prepare_hamer.sh
#
# NOTE: HaMeR requires ViTPose to be installed. This script handles that.
#       If your system does not have CUDA, ViTPose may need extra configuration.
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
# 1. Initialise HaMeR submodule (under third_party/)
# ---------------------------------------------------------------------------
info "=== [1/5] Initialising HaMeR submodule ==="

mkdir -p third_party

HAMER_DIR="third_party/hamer"

if [ -d "${HAMER_DIR}/.git" ] || [ -f "${HAMER_DIR}/.git" ]; then
    info "HaMeR submodule already initialised."
elif git submodule status "$HAMER_DIR" 2>/dev/null | grep -q '^-'; then
    info "Running: git submodule update --init ${HAMER_DIR}"
    git submodule update --init "$HAMER_DIR"
elif [ ! -d "$HAMER_DIR" ] || [ -z "$(ls -A "$HAMER_DIR" 2>/dev/null)" ]; then
    info "HaMeR submodule not configured — adding it now."
    git submodule add https://github.com/geopavlakos/hamer.git "$HAMER_DIR"
    git submodule update --init "$HAMER_DIR"
else
    warn "${HAMER_DIR} exists but is not a git submodule. Attempting to use as-is."
fi

# Verify that the clone is non-empty
if [ ! -f "${HAMER_DIR}/setup.cfg" ] && [ ! -f "${HAMER_DIR}/pyproject.toml" ]; then
    die "HaMeR directory appears empty. Check submodule setup."
fi

# ---------------------------------------------------------------------------
# 2. Install HaMeR Python package (core, no detectron2 needed)
# ---------------------------------------------------------------------------
info "=== [2/5] Installing HaMeR Python package ==="

# Install core hamer package. We skip detectron2/body-detector extras since
# AnyHand uses WiLoR's YOLO hand detector instead.
pip install -q -e "${HAMER_DIR}/"
info "HaMeR core installed."

# ---------------------------------------------------------------------------
# 3. Install ViTPose (required for HaMeR's ViT backbone)
# ---------------------------------------------------------------------------
info "=== [3/5] Installing ViTPose backbone ==="

VITPOSE_DIR="${HAMER_DIR}/third-party/ViTPose"

if [ ! -d "$VITPOSE_DIR" ]; then
    # ViTPose may be a nested submodule inside HaMeR
    info "Initialising ViTPose nested submodule inside HaMeR..."
    git -C "$HAMER_DIR" submodule update --init --recursive
fi

if [ -d "$VITPOSE_DIR" ] && { [ -f "${VITPOSE_DIR}/setup.py" ] || [ -f "${VITPOSE_DIR}/setup.cfg" ]; }; then
    pip install -q -v -e "$VITPOSE_DIR"
    info "ViTPose installed."
else
    warn "ViTPose directory not found at ${VITPOSE_DIR}."
    warn "You may need to install it manually:"
    warn "  pip install -v -e ${HAMER_DIR}/third-party/ViTPose"
fi

# ---------------------------------------------------------------------------
# 4. Download AnyHand HaMeR checkpoint + config
# ---------------------------------------------------------------------------
info "=== [4/5] Downloading AnyHand HaMeR checkpoint ==="

# HaMeR expects its config file to live next to the checkpoint.
# We mirror its internal layout: pretrained_models/hamer_ckpts/checkpoints/
CKPT_DIR="pretrained_models/hamer_ckpts/checkpoints"
mkdir -p "$CKPT_DIR"

HF_USER="chen-si-02"
HF_REPO="AnyHand-Models"
HF_BASE="https://huggingface.co/${HF_USER}/${HF_REPO}/resolve/main"

if [ "$HF_USER" = "<YOUR_HF_USERNAME>" ]; then
    warn "HF_USER is still a placeholder. Edit this script and set your HuggingFace username."
    warn "Skipping checkpoint download."
else
    download "${HF_BASE}/anyhand_hamer.ckpt"          "${CKPT_DIR}/anyhand_hamer.ckpt"
    # HaMeR's load_hamer() looks for model_config.yaml in the SAME dir as the ckpt.
    download "${HF_BASE}/model_config_hamer.yaml"     "${CKPT_DIR}/model_config.yaml"
fi

# ---------------------------------------------------------------------------
# 5. Download mano_mean_params.npz (required by HaMeR)
# ---------------------------------------------------------------------------
info "=== [5/5] Downloading HaMeR auxiliary data ==="

HAMER_DATA_DIR="pretrained_models/hamer_ckpts/data"
mkdir -p "$HAMER_DATA_DIR"

HAMER_HF="https://huggingface.co/spaces/geopavlakos/hamer/resolve/main"
download "${HAMER_HF}/_DATA/data/mano_mean_params.npz" "${HAMER_DATA_DIR}/mano_mean_params.npz" || \
    warn "Could not download mano_mean_params.npz — you may need to fetch it manually."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  HaMeR setup complete."
echo ""
echo "  third_party/hamer/           ← HaMeR submodule"
echo "  pretrained_models/"
echo "  └── hamer_ckpts/"
echo "      ├── checkpoints/"
echo "      │   ├── anyhand_hamer.ckpt    ← AnyHand fine-tuned HaMeR"
echo "      │   └── model_config.yaml     ← matching config"
echo "      └── data/"
echo "          └── mano_mean_params.npz"
echo ""
echo "  ACTION REQUIRED (if not already done for WiLoR):"
echo "  Place MANO_RIGHT.pkl at:  mano_data/MANO_RIGHT.pkl"
echo "  Download from: https://mano.is.tue.mpg.de/"
echo "============================================================"
