# PASDP Implementation

This directory contains the algorithm implementation, simulation configuration, training and evaluation tools, real-robot utilities, and vendored dependencies associated with the [PASDP project page](https://huang-1-yi.github.io/pasdp/).

The original source snapshot is preserved. See [SOURCE_IMPORT.md](SOURCE_IMPORT.md) for its commit and [README_UPSTREAM.md](README_UPSTREAM.md) for the original detailed documentation. The snapshot includes newer development work and additional baselines; its default configurations are not a substitute for the exact configurations and checkpoints used for each paper result.

## Working directory

Clone the combined repository and run algorithm commands from `code/`:

```bash
git clone https://github.com/Huang-1-Yi/pasdp.git
cd pasdp/code
```

Preserve the directory structure. Configuration paths, dataset paths, and local dependency paths are relative to this directory. The website remains at the parent repository root.

## Environment

Training and simulation target Linux with NVIDIA CUDA. The bundled current simulation environment is [conda_environment_sim_torch27_v8.yaml](conda_environment_sim_torch27_v8.yaml), using Python 3.10, PyTorch 2.7.1, and CUDA 12.8 wheels.

Install the system prerequisites for the selected environments first: CMake and the GL/EGL/GLEW development libraries; legacy MetaWorld/Adroit additionally requires the MuJoCo 2.1 installation used by `mujoco-py`. Adapt the YAML's user-specific `LD_LIBRARY_PATH` before creating an environment on another machine. Then, from this directory:

```bash
mamba env create -f conda_environment_sim_torch27_v8.yaml
conda activate sim_torch27_v8
python -m pip check
```

The YAML installs packages from `third_party/`, including editable dependencies and the bundled PyTorch3D wheel. Keep that directory in place after installation. If reusing an environment installed from another checkout, rebind its editable packages to this checkout; see [README_UPSTREAM.md](README_UPSTREAM.md).

Older environment files are preserved for reference. `conda_environment_sim.yaml` also contains historical shell commands and is not the current environment-creation entry point.

## Data

Datasets and trained weights are not bundled. For Robomimic/MimicGen, the expected absolute-action dataset layout is:

```text
data/robomimic/datasets/<task_name>/<task_name>_abs.hdf5
```

For example, provide `data/robomimic/datasets/coffee_d0/coffee_d0_abs.hdf5`. To link an existing centralized dataset directory:

```bash
DP_DATA_ROOT=/absolute/path/to/datasets bash tests/运行命令/链接数据集.sh
```

The supplied helper also supports MetaWorld and Adroit datasets. Generated data caches and training outputs under `data/` are ignored by Git.

## Simulation training

The native PASDP experiment entry point is [pasdp_a_native_robomimic.yaml](diffusion_policy/config/experiment/pasdp_a_native_robomimic.yaml):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1 \
python -u train_sim.py --config-name=train \
  experiment=pasdp_a_native_robomimic \
  task_name=coffee_d0 n_demo=200 training.seed=42 \
  training.resume=false
```

Select the physical GPU in both GPU environment variables. Select another task with `task_name`. Baseline and ablation configurations are under [diffusion_policy/config/experiment/](diffusion_policy/config/experiment/); inspect the chosen configuration before running a comparison.

New native PASDP runs default to `data/outputs/pasdp_experiment_a/<date>/<run>/`. Each run holds its configuration, logs, checkpoints, and rollout outputs. To resume a particular run, use its existing Hydra output directory:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1 \
python -u train_sim.py --config-name=train \
  experiment=pasdp_a_native_robomimic \
  task_name=coffee_d0 n_demo=200 training.seed=42 \
  training.resume=true hydra.run.dir=/absolute/path/to/existing/run
```

Use the original run's complete configuration and overrides when resuming. `training.resume=true` alone does not search other run directories.

## Standalone evaluation

The bundled [eval_sim_rollout.py](eval_sim_rollout.py) loads a checkpoint and runs simulation evaluation:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
python eval_sim_rollout.py \
  --checkpoint /absolute/path/to/run/checkpoints/latest.ckpt \
  --output_dir data/evaluations/coffee_seed42 \
  --device cuda:0 \
  --n-envs 20 --n-train 10 --n-test 30 \
  --test-start-seed 100000
```

These explicit evaluation counts match the snapshot's `env_runner/robomimic/standard.yaml`. The evaluator's CLI defaults differ, so specify the protocol used by the experiment being reported. Keep success-rate evaluation and inference-latency measurement traceable to their own records.

## Real robots and historical scripts

Real-robot utilities and their device-specific setup are documented in [README_UPSTREAM.md](README_UPSTREAM.md) and [diffusion_policy/envs/realword/](diffusion_policy/envs/realword/). Historical queue scripts under `运行命令及文档/` may contain `/home/hy/dp_pro6000`, device addresses, and fixed Conda paths. Adapt them to this checkout before use; their internal paths were preserved during import.

## Dependencies and licenses

Keep upstream license files in `third_party/` and the notices alongside other vendored components. This import does not assign a new license to the original project or override dependency licenses. Robot meshes, textures, SDKs, wheels, and upstream test fixtures are intentional source assets.

This integration was checked for source-file integrity and static paths. It was not trained or evaluated on this Windows machine.
