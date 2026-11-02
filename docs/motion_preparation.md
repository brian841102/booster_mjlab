# Preparing Motions for Tracking

booster_mjlab has two motion-tracking pipelines, and each takes its clips in a different form:

| Tasks | Algorithm | Clips | Where they live |
|---|---|---|---|
| `Mjlab-Tracking-*` (K1) | BeyondMimic-style PPO tracking | GMR `.pkl` or BeyondMimic `.csv`, converted to `motion.npz` | WandB artifacts ([below](#wandb-motions-for-mjlab-tracking-)) |
| `Mjlab-SuperMimic-*` (Nemov3) | [SuperMimic](supermimic.md) (world model + BPTT) | SuperMimic clip `.csv` from the retarget submodule | a local directory ([below](#supermimic-clips-for-mjlab-supermimic-)) |

Both can start from a video: the [retarget submodule](#retargeting-videos-submodulesretarget) turns it into GMR `.pkl`
files and SuperMimic clips for Nemov3.

## WandB motions (for `Mjlab-Tracking-*`)

Clips can come from [GMR](https://github.com/YanjieZe/GMR/) (`.pkl`, the format produced when retargeting
[GVHMR](https://github.com/zju3dv/GVHMR) video reconstructions) or from
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) (`.csv`).

Set your WandB entity first:

```bash
export WANDB_ENTITY=your-organization-name
```

### GMR motions

`upload-motion` converts a `.pkl` clip to the tracking format and uploads it to WandB in one step:

```bash
uv run upload-motion --input-file /path/to/motion.pkl --entity your-org
```

The motion is uploaded as an artifact named `<file-stem>-tracking` to the `motion_upload` project
(override with `--project` and `--artifact-name`). The command prints the artifact's full name, e.g.
`your-org/motion_upload/motion-tracking:v0`, which you pass as `--registry-name` when training.

### BeyondMimic motions

First create a registry collection in your WandB workspace named `motions`, then run:

```bash
MUJOCO_GL=egl uv run python -m booster_mjlab.scripts.csv_to_npz \
    --input-file /path/to/motion.csv \
    --output-name motion_name \
    --input-fps 30 \
    --output-fps 50 \
    --render
```

This uploads the motion to the `motions` registry as `your-org/motions/motion_name`, which you pass as
`--registry-name` when training.

## Retargeting videos (`submodules/retarget`)

The retarget submodule ([nemov3_retarget](https://github.com/brian841102/nemov3_retarget)) reconstructs the person
in a video with GVHMR (SMPL-X), recovers the world trajectory from a moving camera, and retargets it onto Nemov3
with GMR (wheel post-processing, single-wheel balance). Its own GVHMR / GMR environments are set up as described in
`submodules/retarget/README.md`:

```bash
git submodule update --init --recursive submodules/retarget   # GVHMR and GMR are nested submodules
cd submodules/retarget
scripts/run_video_to_nemov3.sh data/videos/<name>.mp4 <gpu>   # -> data/retarget_out/<name>_{full,hw}.{csv,pkl,json}
```

`<name>_hw.pkl` is a booster_mjlab `MotionFile` for the current hardware build (head, waist and wrist pitch fixed),
usable as an AMP reference clip; `<name>_full.pkl` keeps those joints. Videos and everything under `data/` stay out
of git (copyright, size).

## SuperMimic clips (for `Mjlab-SuperMimic-*`)

A SuperMimic clip is the retargeted motion framed by the robot's HOME pose: 1 s still at HOME, a 1 s blend into the
motion, the motion, a 1 s blend back and 1 s at HOME. It is a header-less 30 fps CSV, one row per frame: root
position (3), root quaternion xyzw (4), the 20 joint angles in `NEMOV3_CSV_JOINT_NAMES` order
(`tasks/supermimic/config/nemov3/env_cfgs.py`). From the retarget output:

```bash
cd submodules/retarget
python scripts/make_supermimic_clip.py --src data/retarget_out/<name>_hw.csv --end_s <s> \
    --out data/supermimic_motions/<name>/<name>.csv
# --start_s trims the start (e.g. while the person is not fully in frame), --blend_s sets the blend length,
# --smooth lets the root follow inertia across the blends
```

Training uses every `*.csv` in one directory, so collect the clips to train together in one directory:

```bash
mkdir -p data/supermimic_motions/nemov3_dance3
cp data/supermimic_motions/{roller32,roller_house,three_turn}/*.csv data/supermimic_motions/nemov3_dance3/
```

`submodules/retarget/data/supermimic_motions/nemov3_dance3` is the default; pick another directory with
`--env.supermimic.motion-dir` (relative paths are taken from the repository root).

## Next steps

See the [Motion Imitation](../README.md#2-motion-imitation) and
[SuperMimic Motion Tracking](../README.md#3-supermimic-motion-tracking-nemov3) sections of the README for training and
evaluation commands, and use `uv run visualize-motions` to inspect `MotionFile` clips before training on them.
