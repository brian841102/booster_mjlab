# SuperMimic：世界模型动作追踪（Nemov3）

SuperMimic 用一个学到的世界模型 s_{t+1} = f(s_t, a_t, xi)，把可微的追踪奖励沿想象轨迹反向传播来训练策略（BPTT），
critic 只负责截断后的价值。与 `tasks/tracking`（BeyondMimic：PPO + mjlab 的奖励 / 观测管理器）是两套不同的 MDP 和算法。
本文件说明仓库里的位置、数据准备、训练 / 评估指令，以及从 supermimic_vibed（Isaac Lab 3 / Newton）移植过来时对齐的细节。
**约定：每做一次 SuperMimic 实验，在文末「实验记录」追加一节。**

## 目录

```
src/booster_mjlab/
├── supermimic/                    # 算法（与 amp/ 同级）
│   ├── config.py                  # SuperMimicRunnerCfg：论文 Table A.3 超参数、checkpoint 频率
│   ├── state.py                   # 状态 S = (p, R, v, w, q, qd) 与朝向不变特征
│   ├── modules/                   # world_model.py、policy.py（Policy + Critic）、common.py（MLP、RunningNorm、bf16）
│   ├── storage/replay_buffer.py   # (时间, env) 排布的 GPU replay buffer
│   ├── runners/supermimic_runner.py  # 收集 -> 世界模型 -> BPTT 策略 + critic；save / load
│   └── utils/rotations.py
├── tasks/supermimic/              # 任务：追踪什么、怎么打分
│   ├── supermimic_env_cfg.py      # SuperMimicEnvCfg = mjlab 场景 / 物理 / 启动随机化 + SuperMimicCfg（奖励、终止、xi 范围）
│   ├── supermimic_env.py          # SuperMimicEnv：设关节目标、步进物理、按参考状态重置、按 bin 回报抽样起点
│   ├── mdp/                       # kinematics.py（torch FK）、motion.py（clip CSV -> 50 Hz）、tracking.py（可微观测 / 奖励 / 终止）
│   └── config/nemov3/             # Mjlab-SuperMimic-Flat-Booster-Nemov3 的注册、env 与 runner 配置
├── scripts/evaluate_supermimic.py # uv run evaluate-supermimic
└── scripts/replay_supermimic.py   # uv run replay-supermimic（离线轨迹回放）
submodules/retarget/               # 视频 -> GVHMR -> GMR -> nemov3 参考动作（独立仓库，git submodule）
```

奖励、终止、观测都是 `mdp/tracking.py` 里对状态的可微函数：仿真用它标注转移，策略也通过世界模型对它求梯度，
所以 env 的 reward / termination / observation 管理器是空的，runner 不调用 `env.step`，而是由 `SuperMimicEnv`
直接驱动 mjlab 的 `sim` / `scene`（与 `ManagerBasedRlEnv.step` 的 decimation 循环相同）。

## 数据

clip 由 retarget 子模块生成（`scripts/make_supermimic_clip.py`，30 fps CSV：根位置 3、根四元数 xyzw 4、20 个关节，
列顺序见 `NEMOV3_CSV_JOINT_NAMES`），放在子模块的 `data/` 下（不提交）：

```bash
git submodule update --init submodules/retarget          # 不需要 GVHMR / GMR 时不必递归
mkdir -p submodules/retarget/data/supermimic_motions/nemov3_dance3
# 复制 roller32.csv、roller_house.csv、three_turn.csv 到上面的目录
```

默认训练目录 `submodules/retarget/data/supermimic_motions/nemov3_dance3`（三段舞蹈，大多数实验的基准）；
换一组 clip 用 `--env.supermimic.motion-dir <目录>`（相对路径以仓库根目录为准），目录里有几段就一起训练几段。

## 训练与评估

```bash
# 训练（日志：logs/rsl_rl/nemov3_supermimic/<时间>_<run_name>/，latest 指向最新一次）
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-SuperMimic-Flat-Booster-Nemov3 --agent.run-name exp1_xxx

# 另开一个终端，评估每个 checkpoint（eval.csv、model_<it>_eval.json、model_<it>_traj.npz）
CUDA_VISIBLE_DEVICES=0 uv run evaluate-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 \
    --run-dir logs/rsl_rl/nemov3_supermimic/latest --watch True

# 单个 checkpoint
uv run evaluate-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --checkpoint <run>/model_11000.pt

# 离线轨迹回放（评估录下的 model_<it>_traj.npz；checkpoint 下拉、实体 = 策略、蓝色半透明 = 参考，不占 GPU）
uv run replay-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --run-dir logs/rsl_rl/nemov3_supermimic/latest

# 实时推理（viser，网页里可切换 checkpoint、暂停、重置）
CUDA_VISIBLE_DEVICES=0 uv run play Mjlab-SuperMimic-Flat-Booster-Nemov3 --checkpoint-file <run>/model_10000.pt --viewer viser
```

`play`：每个 env 播一段 clip（env 数 = clip 数），从第一帧开始，播完或跌倒（训练的终止条件）就从头重播；
评估条件（摩擦 1.0、无随机化、无噪声）。参考帧、观测历史由 `SuperMimicPlayPolicy` 维护，动作经 env 的
`joint_pos` / `wheel_vel` 动作项下发，与训练时 `SuperMimicEnv.step` 设的目标相同；实时推理回报与评估一致
（exp1 第 12000 轮：roller32 0.9080 / 0.9080、roller_house 0.9267 / 0.9267）。

- 12000 轮；checkpoint 3000 轮前每 1000 轮、之后每 500 轮（`save_interval`、`save_interval_late`、`save_late_from`）。
- `model_<it>.pt` 存网络和配置；`optim.pt` 只存最新一次的优化器状态，供 `--agent.resume True` 续训（replay buffer 不存，续训时先重新收集）。
- 训练曲线：`train_log.csv`，同时写 TensorBoard（`--agent.logger tensorboard`，默认）。
- 评估：每段 clip 从第一帧播放一次，摩擦 1.0、无随机化、无噪声；`norm_return` = 不打折回报 / 满分，触发训练终止条件即算失败。

## 与 supermimic_vibed（exp5）对齐的地方

参考训练：supermimic_vibed `logs/nemov3_multi/2026-10-26_14-21-06_exp5_roller_house_slim`（3 段舞蹈，12000 轮，
最佳 11000 轮 0.9257、0/3 失败；sim2real 随机化之前的最后一版）。移植时保持一致的：

| 项目 | 设定 |
|---|---|
| 机器人 | booster `nemov3.xml`，焊死头 / 腰 / 腕 pitch（20 关节），实机 PD（`NEMOV3_DEPLOY_PD_GAINS`）；任务内另改：手肘范围 (−1.57, 0.1)、手臂力矩上限按 URDF（5.6 / 1.27 Nm，XML 是 20）、关闭自碰撞（只和地面碰） |
| 物理 | 200 Hz（timestep 0.005，decimation 4），implicitfast，pyramidal，Newton 求解器 100 / 50 次迭代，tolerance 1e-6 |
| 动作 | 位置关节 target = default + xi 偏置 + 0.5·a；轮子速度 target = 10·a |
| xi | 机器人摩擦 U(0.3, 1.2)（地面摩擦 0，MuJoCo 取两者较大值）、躯干 CoM ±(1.25, 2.5, 2.5) cm（启动时抽一次）、关节偏置 ±0.01 rad（每个 episode） |
| 奖励 / 观测 / 终止 / 网络 / 超参数 | 逐行移植，数值不变 |

已核对：torch FK 与 MuJoCo 刚体位姿一致（位置误差 1e-6 m、速度 2e-6）；前 550 轮训练曲线与 exp5 几乎重合
（第 500 轮 sim_r 0.372 vs 0.376、ep_len 27.6 vs 27.3；第 550 轮 imag_r 0.207 vs 0.210）；速度 6.6 it/s（同 exp5）。

物理引擎从 Newton（MJCF → USD → MuJoCo-Warp）换成 mjlab（MJCF 直接进 MuJoCo-Warp），轮子碰撞体直接用网格的凸包，
不再需要 `--hull_vertices`。

## 实验记录

### exp1：移植验证（2026-11-02）

`logs/rsl_rl/nemov3_supermimic/2026-11-02_14-59-18_exp1_dance3_port`，三段舞蹈，默认配置（= exp5），12000 轮，1.24 h（exp5 1.21 h）。

| 轮数 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 | 11000 | 12000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp1（mjlab） | 0.6530 | 0.8349 | 0.8823 | 0.8930 | 0.9065 | 0.9148 | 0.9138 | 0.9098 | 0.9187 | **0.9320** | 0.9151 | 0.9101 |
| exp5（Isaac Lab / Newton） | 0.6619 | 0.8236 | 0.8781 | 0.8793 | 0.9155 | 0.8904 | 0.9014 | 0.9213 | 0.9252 | 0.9008 | **0.9257** | 0.9175 |

- 所有评估都是 0/3 失败。最佳 10000 轮 0.9320（roller32 0.9176、roller_house 0.9501、three_turn 0.9283），高于 exp5 最佳 0.9257（11000 轮）和 exp3 最佳 0.9276（8000 轮）。
- 8000–12000 轮的平均：exp1 0.9171、exp5 0.9181，持平；曲线在 0.90–0.93 之间波动，与 exp5 一样。
- 结论：移植后训练成效保持（最佳回报略高），可以在这个基础上继续做实验。
