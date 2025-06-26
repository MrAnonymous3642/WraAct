#!/bin/bash

# === setup_wraact_env.sh ===
# One-click script to set up the Conda environment for the WraAct project

set -e  # Exit immediately on error
ENV_NAME="wraact"

# === Detect shell type and initialize conda ===
echo "[INFO] Detecting shell and initializing Conda..."
if [[ "$SHELL" == */zsh ]]; then
    eval "$(conda shell.zsh hook)"
elif [[ "$SHELL" == */bash ]]; then
    eval "$(conda shell.bash hook)"
else
    echo "[ERROR] Unsupported shell: $SHELL. Please use bash or zsh."
    exit 1
fi

# === Check for existing environment ===
echo "[INFO] Checking if environment '$ENV_NAME' already exists..."
if conda info --envs | grep -qE "^$ENV_NAME[[:space:]]"; then
    echo "[ERROR] Conda environment '$ENV_NAME' already exists. Please remove it manually or use a different name."
    exit 1
fi

# === Create environment ===
echo "[INFO] Creating Conda environment '$ENV_NAME' with Python 3.12..."
conda create -n $ENV_NAME python=3.12 --yes

# === Activate environment ===
echo "[INFO] Activating environment '$ENV_NAME'..."
conda activate $ENV_NAME

# === Install PyTorch with/without CUDA ===
echo "[INFO] Detecting CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    TORCH_BACKEND="CUDA"
else
    echo "[INFO] No CUDA detected. Installing PyTorch with CPU support..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    TORCH_BACKEND="CPU"
fi

# === Install additional packages ===
echo "[INFO] Installing additional Python packages..."
pip install \
    gurobipy==11.0.3 \
    scipy==1.15.3 \
    onnx==1.18.0 \
    onnxruntime==1.22.0 \
    numba==0.61.2 \
    pycddlib==2.1.7 \
    matplotlib

echo ""
echo "[SUCCESS] Environment '$ENV_NAME' setup complete with PyTorch ($TORCH_BACKEND)."
echo "[INFO] To activate this environment later, run: conda activate $ENV_NAME"
