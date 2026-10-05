# Comparison of Methods for Robust Transfer of Reinforcement Learning Policies in Locomotion

Code for the bachelor's thesis **"Comparison of Methods for Robust Transfer of
Reinforcement Learning Policies in Locomotion"**.

The thesis compares methods for narrowing the sim-to-real gap: domain randomization
and the family of methods based on random torque injection at the joints (RFI, RAO,
ERFI-C, ERFI-50; Campanaro et al. 2022). The methods are compared on **four robots**
and with **three kinds of training**. Everything is built on
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(MJX / MuJoCo Warp) and Brax PPO.

## Explanation of experiments

In total, the experiments were done on: **6 training conditions × 3 seeds × 3 kinds of training × 4 robots**.

### Training conditions

| Condition | What is added during training |
|---|---|
| `none` | nothing, Playground's task as it is |
| `dr` | Playground's domain randomization (the same randomizer for Go1, A1 and Spot) |
| `rfi` | random torque `τ_r ~ U(−lim, lim)` at every joint |
| `rao` | random torque offset `τ_o ~ U(−lim, lim)`, one per episode |
| `erfi_c` | RFI and RAO together, in every episode |
| `erfi_50` | each episode is either RFI or RAO, with probability 1/2 |

Main part of the implementation is here: [`src/rl_locomotion/envs/erfi.py`](src/rl_locomotion/envs/erfi.py).

### Robots

| Robot | Mass | Note | Torque limit (RFI/RAO) |
|---|---|---|---|
| **Unitree Go1** | 12.74 kg | Playground's joystick task with tuned hyperparameters | 2.5 Nm on every joint |
| **Unitree A1** | 12.45 kg | the robot of the original paper; task modelled on Go1's ([`envs/a1/`](src/rl_locomotion/envs/a1/)) | 2.5 Nm on every joint |
| **Boston Dynamics Spot** | 50.3 kg | Playground's Spot task taken over whole ([`envs/spot/`](src/rl_locomotion/envs/spot/)) | vector: 0.5 × RMS torque per joint |
| **Berkeley Humanoid** | 16.06 kg | biped with its own task ([`envs/bh/`](src/rl_locomotion/envs/bh/)) | vector: 0.5 × RMS torque per joint |

The Spot and Berkeley vectors were measured with
[`scripts/measure_stance_torque.py`](scripts/measure_stance_torque.py) on the best
`none` policy and written into their configs.

### Three kinds of training

| Training | Terrain during training |
|---|---|
| **flat** | `flat_terrain`: flat ground, friction 0.6 |
| **rough** | `rough_terrain`: Playground's 20 × 20 m heightfield, 0.05 m relief, friction 1.0 |
| **curriculum** | `rough_terrain` in 4 stages of increasing relief 0.0 → 0.015 → 0.03 → 0.05 m; each stage starts from the previous stage's final parameters, the optimizer state is not carried over |

### Studies → configs → results

Each study is one file in [`configs/experiment/`](configs/experiment/). Results are
written to `$RL_EXPERIMENTS_DIR/<output folder>`. For the thesis this was
`experiments/redo/`.

| Robot | Training | Config | Output folder | Budget per policy |
|---|---|---|---|---|
| Go1 | flat | `erfi_study_v3.yaml` | `erfi_study_v3_l2.5` | 300 M ¹ |
| Go1 | rough | `erfi_study_v3_rough.yaml` | `erfi_study_v3_rough_l2.5` | 300 M ¹ |
| Go1 | curriculum | `erfi_study_curr_v3.yaml` | `erfi_study_curr_v3_l2.5` | 4 × 75 M |
| A1 | flat | `erfi_study_a1_v3.yaml` | `erfi_study_a1_v3_l2.5` | 300 M ¹ |
| A1 | rough | `erfi_study_a1_v3_rough.yaml` | `erfi_study_a1_v3_rough_l2.5` | 300 M ¹ |
| A1 | curriculum | `erfi_study_curr_a1_v3.yaml` | `erfi_study_curr_a1_v3_l2.5` | 4 × 75 M |
| Spot | flat | `erfi_study_spot_v3.yaml` | `erfi_study_spot_v3` | 300 M |
| Spot | rough | `erfi_study_spot_v3_rough.yaml` | `erfi_study_spot_v3_rough` | 300 M |
| Spot | curriculum | `erfi_study_curr_spot_v3.yaml` | `erfi_study_curr_spot_v3` | 4 × 75 M |
| Berkeley | flat | `erfi_study_bh.yaml` | `erfi_study_bh` | 200 M |
| Berkeley | rough | `erfi_study_bh_rough.yaml` | `erfi_study_bh_rough` | 200 M |
| Berkeley | curriculum | `erfi_study_curr_bh.yaml` | `erfi_study_curr_bh` | 4 × 50 M |

The thesis studies were run with
`--num-timesteps 300000000 --num-evals 15`.

In addition, `erfi_study_spot_v3_lim20.yaml` is Spot on flat ground with a uniform
20 Nm limit (`rao` and `erfi_c` only). It is the larger-limit experiment mentioned
in the thesis conclusion.

## Training

- **Task:** Playground's joystick task (tracking a commanded velocity), extended
  with the history of the tracking error `q* − q` over the last 7 steps
  (`history_target_error: true`) and with torque injection.
- **Policy input:** 192 dimensions on the quadrupeds, as for the A1 in the original
  paper. Berkeley has 196, because it keeps its four-element gait clock. The policy
  is blind in both cases, with no terrain perception.
- **Critic:** sees Playground's privileged state (123 dimensions on the quadrupeds,
  114 on the biped). On Berkeley the critic also sees the RAO offset
  (`erfi.critic_sees_offset`), giving 127 dimensions.
- **Algorithm:** Brax PPO with Playground's tuned parameters for the task
  ([`training/ppo.py`](src/rl_locomotion/training/ppo.py)).
- **Networks:** policy [512, 512] as in the original paper, critic [512, 256, 128].
- **Simulator:** Every policy gets its
  own folder with `params_final`, `curve.json`, `env_config.json`, `ppo_config.json`,
  `spec.json` and `summary.json`, so evaluation is always built on the same robot
  and terrain.

## Evaluation

**Common settings:** command 0.5 m/s forward, 8 s episodes, 50 episodes per level of
each parameter. An attempt succeeds if the robot does not fall and covers at least
2.5 m along its initial heading. 

**Base protocol**: each level changes exactly one
parameter. Implemented in [`eval/perturb.py`](src/rl_locomotion/eval/perturb.py).

| Parameter | Levels (Go1) |
|---|---|
| `payload_kg` | 0 … 6 kg |
| `push_N` | 0 … 40 N, for 3 s from the first second, random horizontal direction |
| `friction` | flat ground 0.2 … 0.8; rough 0.3 … 1.3 |
| `gravity` | −2 … −18 m/s² |
| `kp_scale` | 0.33 … 1.5 |

On Spot, payload and push are set as the same fractions of mass and weight as on Go1.
On Berkeley, payload and push are fractions of its own mass and weight, the push acts
along two axes (`push_N_sagittal`, `push_N_lateral`), and a lowered base counts as a
fall.

**Evaluation terrains** are in [`scripts/eval_terrain.py`](scripts/eval_terrain.py).
All of them run on the `rough_terrain` scene, whatever terrain the policy was trained
on.

| In the thesis | Suite in the code | Description |
|---|---|---|
| rough_relief (T1) | `rough_relief` | rough field, relief 0.05 … 0.10 m (Berkeley 0.05 … 0.15 m) |
| rough_bowl_slope (T2) | **`rough_bowl_slope_fine`** | rough "bowl", slope 10 … 26° in 2° steps, 16 s episodes |
| rough_bowl_protocol (T3) | `rough_bowl_protocol` | the base protocol on a 10° rough bowl |
| combined_relief | `combined_relief` | payload × friction × push at the same time (27 points), rough field |
| combined_bowl | `combined_bowl` | the same, on a 10° rough bowl |


**Metrics** (one row per policy × parameter × level in `results*.csv`):
`success_rate`, `fall_rate`, `low_base_rate` (Berkeley only), `progress_m`
(displacement projected on the initial heading), `tracking_rmse` and
`tracking_rmse_alive`.


**Main finding:** ERFI-C and DR give the best results. ERFI-C is best for Go1, and for
Berkeley when trained with the curriculum. DR is best for Spot, and for Berkeley
trained on flat ground. The differences between the methods are smaller than in the
original paper, and ERFI-50 does not stand out.

## Reproduction

### Installation

The project runs on two kinds of machine:

| Task | Laptop (CPU) | GPU machine |
|---|---|---|
| Editing code and configs, `pytest` | ✓ | ✓ |
| Browsing environments, short rollouts, videos | ✓ | ✓  |
| Interactive MuJoCo viewer | ✓ | ✗ (no display) |
| Plots and tables from finished `results*.csv` | ✓ | ✓ |
| **Training** | ✗ | ✓ |
| Full evaluation (`eval.py`, `eval_terrain.py`) | slow | ✓ |

**Requirements:** Python 3.10+ (3.11 is used), conda on the laptop, git, and about 3 GB of disk for the robot models. The GPU
machine needs an NVIDIA GPU with CUDA 12. The thesis used RunPod pods (RTX 4090,
RTX 6000 Ada, A100).

#### Laptop (Windows, macOS, Linux)

```bash
git clone https://github.com/nesovicelena/RL-for-Locomotion.git
cd RL-for-Locomotion

conda env create -f environment.yml      
conda activate rl-locomotion
pip install -r requirements-cpu.txt      
pip install -e ".[dev]"                  # this package + MuJoCo, Playground, Brax, ... + pytest, ruff, jupyterlab
```



#### GPU machine (RunPod or any CUDA 12 Linux box)

1. **Create the pod** from a CUDA 12 template and
   **attach a network volume mounted at `/workspace`**. 

2. **Connect and run the bootstrap script:**

   ```bash
   ssh -p <port> root@<ip>
   cd /workspace && git clone https://github.com/nesovicelena/RL-for-Locomotion.git
   cd RL-for-Locomotion && bash runpod/bootstrap.sh
   ```

   [`runpod/bootstrap.sh`](runpod/bootstrap.sh) is safe to re-run on every new pod. It:
   - pulls the latest code,
   - installs the pinned GPU packages (`requirements-gpu.txt`: JAX with CUDA 12, Brax,
     Playground, MuJoCo, Warp) and this package,
   - points `experiments/` at `/workspace/experiments`,
   - downloads the robot models,
   - checks that JAX sees the GPU.

   Without RunPod, the same steps by hand:

   ```bash
   pip install -r requirements-gpu.txt
   pip install -e ".[dev]"
   python -c "from rl_locomotion.envs.registry import ensure_menagerie; ensure_menagerie()"
   ```

3. **Set the environment variables** in every new shell:

   ```bash
   export MUJOCO_GL=egl
   export RL_EXPERIMENTS_DIR=/workspace/experiments/redo
   ```

5. **Run long jobs inside `tmux`**: `tmux new -s train`, start the job and come back later.

6. **Copy the results to the laptop** after every finished study, not only at the end:

   ```bash
   # on the laptop; skips the large intermediate checkpoints
   PODS="root@<ip>:<port>" bash runpod/pull_loop.sh
   ```

7. **Stop the pod** when it is idle.




### Training (GPU)

```bash
export RL_EXPERIMENTS_DIR=/workspace/experiments/redo

# flat and rough (Go1, A1, Spot; for Berkeley leave out --num-timesteps/--num-evals)
python scripts/train.py --config configs/experiment/erfi_study_v3.yaml \
    --num-timesteps 300000000 --num-evals 15
python scripts/train.py --config configs/experiment/erfi_study_v3_rough.yaml \
    --num-timesteps 300000000 --num-evals 15

# curriculum (stages and budgets are in the config)
python scripts/train_curriculum.py --config configs/experiment/erfi_study_curr_v3.yaml
```

Finished policies and stages are skipped, so the same command resumes after an
interruption.


### Evaluation (GPU)

```bash
# base protocol on the training terrain -> results.csv
python scripts/eval.py --config configs/experiment/erfi_study_curr_v3.yaml --plot

# the thesis terrains -> results_<suite>.csv
python scripts/eval_terrain.py --studies erfi_study_v3_l2.5 erfi_study_curr_v3_l2.5 \
    --suites rough_relief rough_bowl_slope_fine rough_bowl_protocol combined_relief combined_bowl
```

### Plots and tables (laptop, no GPU)

Every plot is a function of the `results*.csv` files alone:

```bash
python scripts/replot.py --lang sr --no-legend --out-suffix _sr_nolegend   # Cyrillic labels, as in the thesis
python scripts/replot.py                                                    # English labels
```


## Repository layout

```
configs/experiment/   one YAML file per study (conditions, seeds, training, evaluation)
src/rl_locomotion/
  envs/erfi.py        ERFI mixin over the Playground tasks, layout per robot
  envs/a1, bh, spot/  tasks and scenes for A1, Berkeley Humanoid and Spot
  envs/terrain.py     heightfield, bowl and rough bowl
  training/ppo.py     Brax PPO; one self-describing folder per policy
  eval/perturb.py     base protocol and metrics
scripts/              train.py, train_curriculum.py, eval.py, eval_terrain.py, replot.py, ...
runpod/               scripts for the GPU pod (bootstrap, study programmes, fetching results)
notebooks/            local/ = environment catalogue and robot images; 
tests/                CPU tests (ERFI, A1, Berkeley, Spot)
```