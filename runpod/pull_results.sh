#!/usr/bin/env bash
# Bring one run's artifacts back to the laptop for figure-making.

set -euo pipefail

POD="${POD:?set POD=user@host}"
POD_PORT="${POD_PORT:-22}"
RUN="${1:?usage: pull_results.sh <experiment-name>}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/experiments/runs}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/experiments/runs"

mkdir -p "${LOCAL_DIR}"


EXCLUDE=(--exclude 'checkpoints/')
if [ "${WITH_CHECKPOINTS:-0}" = "1" ]; then
    EXCLUDE=()
fi

rsync -avz --progress \
    -e "ssh -p ${POD_PORT}" \
    "${EXCLUDE[@]}" \
    "${POD}:${REMOTE_RUNS}/${RUN}/" "${LOCAL_DIR}/${RUN}/"

echo "Pulled ${RUN} -> ${LOCAL_DIR}/${RUN}"
echo "(set WITH_CHECKPOINTS=1 to include checkpoint files)"
