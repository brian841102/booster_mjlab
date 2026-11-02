<img src="./assets/banner.webp" alt="Project banner">

# booster_mjlab

booster_mjlab is an [mjlab](https://github.com/mujocolab/mjlab) integration for the Booster K1.
It provides the K1 robot model, velocity and motion-tracking tasks, and an [Adversarial Motion Priors](https://arxiv.org/abs/2104.02180) (AMP) training pipeline that learns natural-looking gaits from motion capture data.

Real-robot and simulation demos are on the [project page](https://intelligentroboticslab.github.io/booster_mjlab/),
which also runs the velocity policy live in the browser (MuJoCo WebAssembly + the exported network).

## Getting Started

booster_mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Install from source:**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/IntelligentRoboticsLab/booster-mjlab.git && cd booster-mjlab
uv run list_envs
```

The last command syncs the environment and prints every registered task.

## Training Examples

### 1. Velocity Tracking

Train a Booster K1 to follow velocity commands on flat terrain. The reward has been tuned to work together with AMP reference motions.
The default [motion dataset](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1) contains a few locomotion clips retargeted from [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
It is a small curated subset, not the full LAFAN1 collection: 7 clips (~4 min total) cut from two source
trials, `run2_subject1` and `walk3_subject2`, each split into several segments. Both trials include natural
left and right turns, not just straight-line walking/running.

| Clip | Source trial | Frames | Duration (s) |
|---|---|---|---|
| `run2_subject1_05` | `run2_subject1` | 1068 | 35.6 |
| `run2_subject1_06` | `run2_subject1` | 670 | 22.3 |
| `run2_subject1_07` | `run2_subject1` | 1255 | 41.8 |
| `walk3_subject2_01` | `walk3_subject2` | 470 | 15.7 |
| `walk3_subject2_02` | `walk3_subject2` | 2332 | 77.7 |
| `walk3_subject2_03` | `walk3_subject2` | 316 | 10.5 |
| `walk3_subject2_04` | `walk3_subject2` | 998 | 33.3 |

```bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --env.scene.num-envs 4096
```

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --wandb-run-path your-org/mjlab/run-id
```

To train on your own motions, pass a Hugging Face dataset (`namespace/repo`) or a local motion file or directory
with `--agent.dataset-root`.

The `-DA-` tasks train with left/right symmetry data augmentation. Drop `DA` from the task ID to train without it.

The `-Muon-` tasks use the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) (Jordan et al., 2024) for the
actor and critic weight matrices. Drop `Muon` from the task ID for the Adam variant with otherwise identical settings.

### 2. Motion Imitation

Train a Booster K1 to track motion capture clips on flat terrain. Motions are managed as WandB artifacts;
see [Preparing Motions for Tracking](docs/motion_preparation.md) for converting and uploading GMR (`.pkl`)
or BeyondMimic (`.csv`) clips.

Once a motion is uploaded, train and evaluate with:

```bash
uv run train Mjlab-Tracking-Flat-Booster-K1 --registry-name your-org/motion_upload/motion-tracking:v0 --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Booster-K1 --wandb-run-path your-org/mjlab/run-id
```

For a motion prepared with `csv_to_npz`, pass `--registry-name your-org/motions/motion_name` instead.

### 3. SuperMimic Motion Tracking (Nemov3)

Train the Nemov3 wheel-legged humanoid to track dance clips retargeted from videos with
[SuperMimic](docs/supermimic.md): the policy learns by backpropagating a differentiable tracking reward through a
learned world model, instead of PPO. The clips are made by the `submodules/retarget` pipeline
(video -> GVHMR -> GMR -> Nemov3) and stay local; see
[Preparing Motions for Tracking](docs/motion_preparation.md#supermimic-clips-for-mjlab-supermimic-) for making them.
Every clip in the motion directory (default `submodules/retarget/data/supermimic_motions/nemov3_dance3`, three dances)
is trained together.

```bash
git submodule update --init submodules/retarget
uv run train Mjlab-SuperMimic-Flat-Booster-Nemov3 --agent.run-name exp1
```

Evaluate every checkpoint while training (normalized return and failed clips, written to the run's `eval.csv`; the
rollouts are saved as `model_<it>_traj.npz`), then look at the policies in the browser, either replaying the recorded
rollouts offline (checkpoint dropdown, robot vs. reference ghost, no GPU needed) or running a policy live:

```bash
uv run evaluate-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --run-dir logs/rsl_rl/nemov3_supermimic/latest --watch True
uv run replay-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --run-dir logs/rsl_rl/nemov3_supermimic/latest
uv run play Mjlab-SuperMimic-Flat-Booster-Nemov3 --checkpoint-file logs/rsl_rl/nemov3_supermimic/<run>/model_12000.pt --viewer viser
```

On three dance clips, 12,000 iterations (~1.2 h on one RTX 4090) reach a normalized return of 0.91-0.93 with no
failed clip.

### 4. Browse Motion Data

Inspect, play back and edit motion clips in the browser before training on them:

```bash
uv run visualize-motions --help
```

Clips from the [lafan_locomotion_k1](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)
dataset, retargeted onto the parallel-ankle K1:

<picture>
    <source
    srcset="./assets/motion_grid_dark.webp"
    media="(prefers-color-scheme: dark)">
    <source
    srcset="./assets/motion_grid_light.webp"
    media="(prefers-color-scheme: light)">
    <img
    src="./assets/motion_grid_light.webp"
    alt="Retargeted lafan_locomotion_k1 clips playing on the K1"
    width="720">
</picture>

## Tasks

Velocity tasks come in `Flat` and `Rough` terrain variants, optionally with `-Amp-` (motion prior) and
`-DA-` (symmetry data augmentation). Tasks suffixed `-Parallel` use the parallel-linkage ankle model of the K1.
Tracking tasks (`Mjlab-Tracking-*` with PPO, `Mjlab-SuperMimic-*` with SuperMimic) run on flat terrain. List them all
with:

```bash
uv run list_envs
```

## Deployment

Example code for deploying trained policies on the real Booster K1 is coming soon.

## Hungry for more?

booster_mjlab is a whIRLwind project. Our mission is to push the boundaries of robotics and AI in robot football. We are always looking for interested students or collaboration partners to join us. Learn more at [whirlwind.team](https://whirlwind.team).

<picture>
    <source
    srcset="./assets/whirlwind_logo_light.png"
    media="(prefers-color-scheme: dark)"
    height="125"
    />
    <img
    src="./assets/whirlwind_logo_dark.png"
    alt="whIRLwind logo"
    height="125"
    />
</picture>
