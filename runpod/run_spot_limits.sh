#!/usr/bin/env bash
 
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

if [ -z "${RL_EXPERIMENTS_DIR:-}" ]; then
    if [ -d /workspace ]; then RL_EXPERIMENTS_DIR=/workspace/experiments/redo; else RL_EXPERIMENTS_DIR="${REPO}/experiments/redo"; fi
fi
export RL_EXPERIMENTS_DIR
ROOT="${RL_EXPERIMENTS_DIR}"; LOGS="${ROOT}/logs"; STATUS="${ROOT}/status_spot_limits.txt"
mkdir -p "${LOGS}"

CFG_FLAT=configs/experiment/erfi_study_spot_v3.yaml
CFG_ROUGH=configs/experiment/erfi_study_spot_v3_rough.yaml
CFG_CURR=configs/experiment/erfi_study_curr_spot_v3.yaml
OUT_FLAT="${ROOT}/erfi_study_spot_v3"
FRACTION="${FRACTION:-0.5}"
SEEDS="${SEEDS:-0 1 2}"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
run() { local log="$1"; shift; "$@" >> "${LOGS}/${log}.log" 2>&1; }

{
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "commit  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "fraction ${FRACTION}  timing ${TIMING:-1}  sweep ${SWEEP:-1}  seeds ${SEEDS}"
    echo "dirty files:"; git status --short 2>/dev/null
} > "${ROOT}/provenance_spot_limits.txt"

T_ALL=$(date +%s)
note "==== run_spot_limits start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"

# ---------------------------------------------------------------- 0 smoke
if [ "${SMOKE:-0}" = "1" ]; then
    note "---- stage 0: smoke (2 M steps, flat study, every condition)"
    STUDIES="erfi_study_spot_v3" SMOKE=1 bash runpod/run_all_studies.sh > "${LOGS}/smoke_spot.log" 2>&1 \
        && note "DONE smoke" || note "FAIL smoke (see ${LOGS}/smoke_spot.log)"
    exit 0
fi

# ---------------------------------------------------------------- 1 timing
if [ "${TIMING:-1}" = "1" ]; then
    note "---- stage 1: timing (none seed 0, 20 M steps, 2 evals)"
    t0=$(date +%s)
    run timing_spot python scripts/train.py --config "${CFG_FLAT}" --conditions none --seeds 0 \
        --num-timesteps 20000000 --num-evals 2 --out erfi_study_spot_v3_timing
    rc=$?; dt=$(( $(date +%s) - t0 ))
    if [ ${rc} -ne 0 ]; then
        note "FAIL stage 1: train.py exited ${rc} after ${dt} s (see ${LOGS}/timing_spot.log). Not installed? Run runpod/bootstrap.sh first."
        exit 1
    fi
    note "timing: 20 M in $(( dt / 60 )).$(( (dt % 60) / 6 )) min -> 300 M about $(( dt * 15 / 60 )) min per run (incl. JIT once)"
fi

# ---------------------------------------------------------------- 2 none
note "---- stage 2: none seeds ${SEEDS} on flat ground, the study's budget"
t0=$(date +%s)
# shellcheck disable=SC2086
run none_spot python scripts/train.py --config "${CFG_FLAT}" --conditions none --seeds ${SEEDS}
[ $? -ne 0 ] && { note "FAIL stage 2 (see ${LOGS}/none_spot.log); stopping"; exit 1; }
note "DONE stage 2 in $(( ($(date +%s) - t0) / 60 )) min"

# ---------------------------------------------------------------- 3 measure
if grep -q "rfi_lim:.*measured" "${CFG_FLAT}"; then
    note "skip stage 3: ${CFG_FLAT} already carries a measured vector"
else
    note "---- stage 3: nominal eval of none, pick the best walker, measure stance torques"
    run measure_spot python scripts/eval.py --config "${CFG_FLAT}" --params payload_kg --force
    best=$(python - "${OUT_FLAT}/results.csv" <<'PYEOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
nom = df[(df.condition == "none") & (df.param == "payload_kg") & (df.level == 0.0)]
row = nom.sort_values("progress_m", ascending=False).iloc[0]
print(row["run"], f"{row['progress_m']:.2f}")
PYEOF
)
    best_run=$(echo "${best}" | cut -d' ' -f1); best_prog=$(echo "${best}" | cut -d' ' -f2)
    note "best none policy: ${best_run} (${best_prog} m at nominal)"
    if python -c "import sys; sys.exit(0 if float('${best_prog}') >= 2.5 else 1)"; then
        run measure_spot python scripts/measure_stance_torque.py "${OUT_FLAT}/${best_run}" \
            --fraction "${FRACTION}" --write "${CFG_FLAT}" "${CFG_ROUGH}" "${CFG_CURR}"
        [ $? -ne 0 ] && { note "FAIL stage 3: measurement failed (see ${LOGS}/measure_spot.log); stopping"; exit 1; }
        note "DONE stage 3: limits written to the three spot configs"
        grep -h "rfi_lim:" "${CFG_FLAT}" | tee -a "${STATUS}"
        cat "${OUT_FLAT}/${best_run}/stance_torque.json" | python -c "import json,sys; d=json.load(sys.stdin); print('  RMS per joint (Nm):', [round(x,2) for x in d['rms_per_joint']])" | tee -a "${STATUS}"
    else
        note "FAIL stage 3: no none seed walks 2.5 m at nominal (best ${best_prog} m). Provisional limits kept; stopping."
        note "      -> the Spot task/reward needs attention before any ERFI condition is trained."
        exit 1
    fi
    rm -f "${OUT_FLAT}/results.csv" "${OUT_FLAT}/summary_success_rate.csv"
fi

# ---------------------------------------------------------------- 4 sweep
if [ "${SWEEP:-1}" = "1" ]; then
    note "---- stage 4: limit sweep 0.25 / 0.5 / 1.0 x vector (erfi_50, rao; seed 0; 50 M)"
    for f in 0.25 0.5 1.0; do
        LIM=$(python -c "import yaml,sys; f=float(sys.argv[1]); v=yaml.safe_load(open('${CFG_FLAT}'))['train']['rfi_lim']; v=v if isinstance(v,list) else [v]; print(' '.join(f'{f*x:.3f}' for x in v))" "${f}")
        # shellcheck disable=SC2086
        run sweep_spot python scripts/train.py --config "${CFG_FLAT}" --conditions erfi_50 rao --seeds 0 \
            --num-timesteps 50000000 --num-evals 5 --rfi-lim ${LIM} --rao-lim ${LIM} --out "erfi_limits_spot_f${f}"
        [ $? -ne 0 ] && note "FAIL sweep f=${f} train (see ${LOGS}/sweep_spot.log)"
        run sweep_spot python scripts/eval.py --config "${CFG_FLAT}" --runs "${ROOT}/erfi_limits_spot_f${f}" --params payload_kg gravity
    done
    note "sweep summary (final eval reward, nominal progress):"
    python - "${ROOT}" <<'PYEOF' | tee -a "${STATUS}"
import json, sys, glob, os
import pandas as pd
root = sys.argv[1]
for f in ("0.25", "0.5", "1.0"):
    d = f"{root}/erfi_limits_spot_f{f}"
    res = pd.read_csv(f"{d}/results.csv") if os.path.exists(f"{d}/results.csv") else None
    for run in sorted(glob.glob(f"{d}/*/seed0")):
        cond = os.path.basename(os.path.dirname(run))
        s = json.load(open(f"{run}/summary.json")) if os.path.exists(f"{run}/summary.json") else {}
        prog = "n/a"
        if res is not None:
            nom = res[(res.condition == cond) & (res.param == "payload_kg") & (res.level == 0.0)]
            if len(nom):
                prog = f"{float(nom.progress_m.iloc[0]):.2f} m"
        print(f"  f={f:5s} {cond:8s} reward {s.get('final_reward', float('nan')):7.2f}  nominal progress {prog}")
PYEOF
    note "decision rule (design 4.1): largest fraction at which erfi_50 and rao leave the standing plateau and erfi_50 walks 2.5 m."
    note "if it is not ${FRACTION}: rerun stage 3 with FRACTION=<f> after removing the 'measured' lines, or edit the three configs by hand."
fi

note "==== run_spot_limits finished in $(( ($(date +%s) - T_ALL) / 60 )) min"
echo
echo "Next: commit and push the measured limits so the other pods can pull them:"
echo "  git add configs/experiment/erfi_study_spot_v3.yaml configs/experiment/erfi_study_spot_v3_rough.yaml configs/experiment/erfi_study_curr_spot_v3.yaml"
echo "  git commit -m 'spot: measured ERFI torque limits' && git push"
echo "then on each pod:  git pull && STUDY=<flat|rough|curr> STOP_POD=1 bash runpod/run_spot_everything.sh"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then note "stopping pod ${RUNPOD_POD_ID}"; runpodctl stop pod "${RUNPOD_POD_ID}"
    else note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"; fi
fi
