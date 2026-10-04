#!/usr/bin/env bash
# Fresh pod -> working Jupyter, in one command. Must be safe to re-run.
#
#   bash runpod/bootstrap.sh
#
# Assumes a CUDA 12 pod with the network volume mounted at /workspace.
set -euo pipefail

REPO_URL="https://github.com/0elena0/bachelor-thesis-RL-in-locomotion.git"
WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/bachelor-thesis-RL-in-locomotion"
EXPERIMENTS_DIR="${WORKSPACE}/experiments"

echo "==> Checking the persistent volume"
if [ ! -d "${WORKSPACE}" ]; then
    echo "ERROR: ${WORKSPACE} does not exist. Attach a network volume to this pod." >&2
    exit 1
fi

echo "==> Fetching the repo"
if [ -d "${REPO_DIR}/.git" ]; then
    git -C "${REPO_DIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

echo "==> Installing dependencies"
pip install --upgrade pip
pip install --ignore-installed blinker
pip install -r requirements-gpu.txt
pip install -e ".[dev]"

echo "==> Wiring experiments/ to the persistent volume"
mkdir -p "${EXPERIMENTS_DIR}"
rm -rf "${REPO_DIR}/experiments"
ln -s "${EXPERIMENTS_DIR}" "${REPO_DIR}/experiments"

echo "==> Environment"
if [ ! -f .env ]; then
    cp .env.example .env
    sed -i "s|^RL_EXPERIMENTS_DIR=.*|RL_EXPERIMENTS_DIR=${EXPERIMENTS_DIR}|" .env
    sed -i "s|^MUJOCO_GL=.*|MUJOCO_GL=egl|" .env
    echo "    Wrote .env — add your WANDB_API_KEY before training."
fi

echo "==> Fetching robot assets (MuJoCo Menagerie, ~2GB, first run only)"
python -c "from rl_locomotion.envs.registry import ensure_menagerie; ensure_menagerie()"

echo "==> Verifying the GPU is actually visible to JAX"
nvidia-smi
python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo
echo "Done. Start Jupyter with:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "Then open notebooks/remote/01_setup_check.ipynb"
