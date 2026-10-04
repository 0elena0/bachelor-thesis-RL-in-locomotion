#!/usr/bin/env bash

#   v1  erfi_study          erfi_study_rough
#   v2  erfi_study_v2       erfi_study_v2_rough
#   v3  erfi_study_v3       erfi_study_v3_rough
#   bh  erfi_study_bh_rough erfi_study_bh        

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

if [ -z "${RL_EXPERIMENTS_DIR:-}" ]; then
    if [ -d /workspace ]; then
        RL_EXPERIMENTS_DIR=/workspace/experiments/redo
    else
        RL_EXPERIMENTS_DIR="${REPO}/experiments/redo"
    fi
fi
export RL_EXPERIMENTS_DIR
ROOT="${RL_EXPERIMENTS_DIR}"
LOGS="${ROOT}/logs"
STATUS="${ROOT}/status.txt"
mkdir -p "${LOGS}"

STUDIES="${STUDIES:-erfi_study erfi_study_rough erfi_study_v2 erfi_study_v2_rough erfi_study_v3 erfi_study_v3_rough erfi_study_bh_rough erfi_study_bh}"
SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke"
fi
TRAIN_FLAGS=""
if [ -n "${NUM_TIMESTEPS:-}" ]; then
    TRAIN_FLAGS="${TRAIN_FLAGS} --num-timesteps ${NUM_TIMESTEPS}"
fi
if [ -n "${NUM_EVALS:-}" ]; then
    TRAIN_FLAGS="${TRAIN_FLAGS} --num-evals ${NUM_EVALS}"
fi

# Provenance: which code produced this tree.
{
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "host    $(hostname)"
    echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "studies ${STUDIES}"
    echo "train flags ${TRAIN_FLAGS:-<config defaults>}"
    echo "dirty files:"
    git status --short 2>/dev/null
} > "${ROOT}/provenance.txt"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }

T_ALL=$(date +%s)
note "==== run_all_studies start -> ${ROOT}"

for study in ${STUDIES}; do
    cfg="configs/experiment/${study}.yaml"
    log="${LOGS}/${study}.log"
    if [ ! -f "${cfg}" ]; then
        note "SKIP ${study}: no such config ${cfg}"
        continue
    fi

    t0=$(date +%s)
    note "---- ${study}: train"
    python scripts/train.py --config "${cfg}" ${TRAIN_FLAGS} ${SMOKE_FLAG} >> "${log}" 2>&1
    rc_train=$?
    if [ ${rc_train} -ne 0 ]; then
        note "FAIL ${study}: train exited ${rc_train} (see ${log})"
        continue
    fi

    note "---- ${study}: eval"
    if [ -n "${SMOKE_FLAG}" ]; then
        # smoke runs land in <out>_smoke; evaluate those with a tiny protocol
        out="$(python -c "import yaml;print(yaml.safe_load(open('${cfg}'))['out'])")"
        python scripts/eval.py --config "${cfg}" --runs "${ROOT}/${out}_smoke" --n-episodes 2 --plot >> "${log}" 2>&1
    else
        python scripts/eval.py --config "${cfg}" --plot >> "${log}" 2>&1
    fi
    rc_eval=$?
    dt=$(( ($(date +%s) - t0) / 60 ))
    if [ ${rc_eval} -ne 0 ]; then
        note "FAIL ${study}: eval exited ${rc_eval} after ${dt} min (see ${log})"
    else
        note "DONE ${study} in ${dt} min"
    fi
done

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_all_studies finished in ${dt_all} min"
note "tree size: $(du -sh "${ROOT}" 2>/dev/null | cut -f1)"
echo
echo "Pull it back to the laptop with:"
echo "  rsync -avz --progress -e 'ssh -p <port>' root@<ip>:${ROOT}/ experiments/redo/"
echo "(add --exclude 'params_0*' to leave the intermediate checkpoints behind)"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
