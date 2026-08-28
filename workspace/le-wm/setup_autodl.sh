#!/bin/bash
# AutoDL instance bootstrap for v5.2 Plan E (AutoDL subset).
# Run ONCE on fresh AutoDL 4090-48G instance.
#
# Steps:
#   1. Configure tsinghua pip mirror BEFORE any pip install
#   2. Create .venv with Python 3.10 via uv (matches v3 setup)
#   3. Install torch + LeWM deps
#   4. Download Push-T data
#   5. Verify v3 Stage-1 ckpt is present
#   6. (Optional) Prep PointMaze data for B9
#
# Usage:
#   bash setup_autodl.sh           # full bootstrap
#   bash setup_autodl.sh --skip-data  # skip data download (already on disk)
set -e

SKIP_DATA=false
if [ "$1" = "--skip-data" ]; then SKIP_DATA=true; fi

cd /workspace/le-wm
echo "==== AutoDL bootstrap @ $(date) ===="

# ---------------------------------------------------------------------------
# 1. Pip mirror (CRITICAL — do this BEFORE any pip install)
# ---------------------------------------------------------------------------
echo "==== [1/6] Configure pip mirror ===="
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# ---------------------------------------------------------------------------
# 2. uv + Python 3.10
# ---------------------------------------------------------------------------
echo "==== [2/6] Install uv and Python 3.10 ===="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.10 || true

if [ ! -d /workspace/le-wm/.venv ]; then
    uv venv /workspace/le-wm/.venv --python 3.10
fi

# ---------------------------------------------------------------------------
# 3. Install torch (CUDA 12.x) + LeWM deps
# ---------------------------------------------------------------------------
echo "==== [3/6] Install PyTorch + LeWM deps ===="
source /workspace/le-wm/.venv/bin/activate
# Install torch with CUDA 12.4 (matches v3 setup)
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision torchaudio 2>&1 | tail -3

# Then install the rest from tsinghua
pip install \
    hydra-core omegaconf einops \
    transformers \
    h5py hdf5plugin \
    pytorch-lightning \
    scipy scikit-learn \
    matplotlib \
    pyyaml \
    paramiko 2>&1 | tail -3

# LeWM-specific packages (stable-pretraining, stable-worldmodel)
pip install stable-pretraining stable-worldmodel 2>&1 | tail -3 || \
    echo "WARN: stable-pretraining/stable-worldmodel install failed — verify"

echo "==== [3/6] Verify torch + CUDA ===="
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# ---------------------------------------------------------------------------
# 4. Download Push-T data
# ---------------------------------------------------------------------------
echo "==== [4/6] Push-T data setup ===="
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/stablewm_home}
mkdir -p "$STABLEWM_HOME"

if [ "$SKIP_DATA" = "true" ]; then
    echo "==> Skipping data download (--skip-data)"
elif [ -f "$STABLEWM_HOME/pusht_expert_train.h5" ]; then
    echo "==> Push-T h5 already exists: $(ls -lh $STABLEWM_HOME/pusht_expert_train.h5)"
else
    echo "==> Downloading Push-T data"
    python download_data.py --dataset pusht --out "$STABLEWM_HOME/pusht_expert_train.h5" || {
        echo "FATAL: Push-T download failed — try hf-mirror fallback or scp from local"
        exit 1
    }
fi

# ---------------------------------------------------------------------------
# 5. Verify v3 Stage-1 ckpt
# ---------------------------------------------------------------------------
echo "==== [5/6] Verify v3 Stage-1 ckpt ===="
CKPT=/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt
if [ ! -f "$CKPT" ]; then
    echo "FATAL: v3 Stage-1 ckpt not at $CKPT"
    echo "       scp it from your local machine first."
    exit 1
fi
echo "==> v3 Stage-1 ckpt: $(ls -lh $CKPT)"

# ---------------------------------------------------------------------------
# 6. (Optional) PointMaze prep for B9
# ---------------------------------------------------------------------------
echo "==== [6/6] PointMaze data (B9) prep ===="
if [ -f "$STABLEWM_HOME/pointmaze_train.h5" ]; then
    echo "==> PointMaze h5 already exists"
elif [ -f config/train/data/pointmaze.yaml ]; then
    echo "==> Attempting PointMaze download (LeWM scripts may not have this — manual fallback to TwoRoom is OK)"
    python download_data.py --dataset pointmaze --out "$STABLEWM_HOME/pointmaze_train.h5" || \
        echo "WARN: PointMaze download failed — for B9 use TwoRoom fallback (data=tworoom)"
else
    echo "==> No pointmaze.yaml config found. For B9, use TwoRoom fallback."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "==== Bootstrap complete @ $(date) ===="
echo "Next:"
echo "  bash run_pipeline_v5.sh autodl-subset 2>&1 | tee logs/autodl_subset.log"
echo "Or one phase at a time:"
echo "  bash run_pipeline_v5.sh sanity-quick"
echo "  bash run_pipeline_v5.sh B4prime_baseline"
echo "  bash run_pipeline_v5.sh B4prime_uvlowr"
echo "  bash run_pipeline_v5.sh B9"
