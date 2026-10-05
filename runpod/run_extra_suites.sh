#!/usr/bin/env bash


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

# The five Go1 studies. The Berkeley Humanoid ones are deliberately absent.
STUDIES="${STUDIES:-erfi_study_l2.5 erfi_study_rough_l2.5 erfi_study_v3_l2.5 erfi_study_v3_rough_l2.5 erfi_study_curr_v3_l2.5}"
N_EPISODES="${N_EPISODES:-50}"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
count_rows() { wc -l < "$1" 2>/dev/null | tr -d ' ' || echo 0; }

# shellcheck disable=SC2086
run_stage() {   # run_stage <label> <logfile> <suite...>
    local label="$1" log="$2"; shift 2
    local t0; t0=$(date +%s)
    note "---- ${label}: suites $* on ${STUDIES}"
    python scripts/eval_terrain.py --studies ${STUDIES} --suites "$@" \
        --n-episodes "${N_EPISODES}" >> "${LOGS}/${log}" 2>&1
    local rc=$? dt
    dt=$(( ($(date +%s) - t0) / 60 ))
    if [ ${rc} -ne 0 ]; then
        note "FAIL ${label} exited ${rc} after ${dt} min (see ${LOGS}/${log})"
    else
        note "DONE ${label} in ${dt} min"
    fi
}

T_ALL=$(date +%s)
note "==== run_extra_suites start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
for s in ${STUDIES}; do
    note "     ${s}: $(ls "${ROOT}/${s}"/*/seed*/params_final 2>/dev/null | wc -l | tr -d ' ')/18 policies"
done

if [ "${SMOKE:-0}" = "1" ]; then
    note "---- smoke: 2 episodes, one study, all four new suites"
    python scripts/eval_terrain.py --studies erfi_study_v3_rough_l2.5 \
        --suites bowl_slope_fine rough_bowl_slope_fine combined_relief combined_bowl \
        --n-episodes 2 --force > "${LOGS}/smoke_extra.log" 2>&1 \
        && note "     smoke ok" || note "FAIL smoke (see ${LOGS}/smoke_extra.log)"
    # the 2-episode CSVs would otherwise be mistaken for real results
    rm -f "${ROOT}"/erfi_study_v3_rough_l2.5/*_bowl_slope_fine.csv \
          "${ROOT}"/erfi_study_v3_rough_l2.5/*_rough_bowl_slope_fine.csv \
          "${ROOT}"/erfi_study_v3_rough_l2.5/*_combined_relief.csv \
          "${ROOT}"/erfi_study_v3_rough_l2.5/*_combined_bowl.csv
    note "==== smoke done"
    exit 0
fi

[ "${SKIP_SLOPE:-0}" != "1" ] && run_stage "1 fine slope" eval_slope_fine.log bowl_slope_fine rough_bowl_slope_fine
[ "${SKIP_COMBINED:-0}" != "1" ] && run_stage "2 combined" eval_combined.log combined_relief combined_bowl

for s in ${STUDIES}; do
    for suite in bowl_slope_fine rough_bowl_slope_fine combined_relief combined_bowl; do
        f="${ROOT}/${s}/results_${suite}.csv"
        [ -f "${f}" ] && note "     ${s}/${suite}: $(( $(count_rows "${f}") - 1 )) rows"
    done
done

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_extra_suites finished in ${dt_all} min"
echo
echo "Pull back with:"
echo "  rsync -avz --progress -e 'ssh -p <port> -i ~/.ssh/id_ed25519' \\"
echo "    --include '*/' --include 'results_*.csv' --include 'summary_*.csv' --exclude '*' \\"
echo "    root@<ip>:${ROOT}/ experiments/redo/"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
