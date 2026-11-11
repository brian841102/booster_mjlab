# Pino v1 AMP 滑冰（不抬脚）方案

> **用户决定（2026-11-03）**：CSV 裁掉 0~1.2 s；用这段（抬脚的）轨迹 + 约束训练不抬脚版本；资产放进 `src/booster_mjlab/robots/pinov1/`；CSV 转 pkl 的工具仿照 nemov3 复制来的工具改写，可以直接覆盖 `assets/amp_sources/pinov1_skating/data/` 下的 `csv_to_npz.py`、`replay_npz_mujoco.py`。**先写方案，批准后才改代码。**
>
> **用户回答（2026-11-03，第二轮）**：(1) 速度控制、PD 增益与其他细节参考 `/home/lyh/Downloads/xdream.py`（XDream X1 的 IsaacLab 配置）；(2) 默认姿态先用 **B**（用户给的姿态），有问题再改 A；(3) 先只训练**直线前进**；(4) 后轮阻尼照 xdream.py 改，该档没有的再问。
>
> 实验记录：`tools/pinov1_finetune.md`（每个实验在启动、仿真评估、实机三个时间点各写一次）。

## 0. 已知事实

| 项目 | 内容 |
|---|---|
| 模型 | `xdream_assets/pino/pino_v1/mjcf/pino_joint23_wheel.xml`，14.9 kg；21 个关节 + 每只脚前后两个轮子（直排，像溜冰鞋）。后轮有电机（36 N·m），前轮从动 |
| XML 执行器 | 23 个 `<motor>`（纯力矩，**没有 PD 增益**）；nemov3 的 XML 自带 `<position>` / `<velocity>` 执行器和 kp / kd |
| 轮子 | 后轮半径约 5 cm、前轮约 3.75 cm（网格尺寸）；后轮关节阻尼 1.0、前轮 0 |
| 默认姿态（用户） | 髋 pitch −0.1、膝 0.2、踝 −0.13、肩 pitch 0.15、肩 roll ±0.12、肘 1.2，其余 0 |
| 参考（裁掉 0~1.2 s 后） | 4.07 s、30 fps；速度 0.54~1.37 m/s（平均 1.0）；根高 0.34~0.40 m；单腿蹬一次约 1 s、一个左右周期约 2 s（约 0.5 Hz，片段里只有 2 个周期）；轮子 34~56% 的帧离地 |
| 参考的平均腿姿 | 髋 pitch −0.54、膝 0.82、踝 −0.09，**比用户的默认姿态蹲得多**（默认膝 0.2） |

## 1. 机器人资产：`src/booster_mjlab/robots/pinov1/`

照 `robots/nemov3/` 的结构：

```
src/booster_mjlab/robots/pinov1/
├── __init__.py
├── pinov1_constants.py      # 关节顺序、默认姿态、执行器、动作缩放、镜像表、get_pinov1_robot_cfg()
└── xmls/
    ├── pinov1.xml           # 由 pino_joint23_wheel.xml 改写
    └── meshes/              # 32 个 STL（19 MB；nemov3 的 meshes 85 MB 已在仓库里，同样不用 LFS）
```

`pinov1.xml` 相对上游 XML 只做这些改动（文件开头写明来源 commit 与改动清单）：
1. `meshdir` 改为 `meshes/`。
2. 执行器：21 个位置关节改成 `<position name="<joint>_act" kp kd forcerange>`，2 个后轮改成 `<velocity name="<joint>_vel" kv forcerange>`，命名与 nemov3 一致（`_act` / `_vel`）；前轮不加执行器（从动）。
3. 删掉地板、灯光等场景元素（mjlab 的 scene 自己加地面），保留 IMU site；碰撞几何体放到 group 3（同 nemov3 `get_spec`）。
4. 在每个轮子下方加接地 site（供"轮子离地高度"奖励 / 评估用，同 nemov3 的 `NEMOV3_WHEEL_GROUND_SITES`）。

`pinov1_constants.py`：
- `PINOV1_JOINT_ORDER`（XML 声明顺序，含 4 个轮子关节）；`HOME_KEYFRAME` 用用户给的默认姿态，根高度用正运动学算到"后轮刚好贴地"。
- `XmlActuatorCfg` 每个关节一组（同 nemov3 的理由：域随机化按执行器 id 索引）。
- 动作缩放：0.25 × forcerange / kp（同 nemov3）→ 腿 / 踝 / 躯干 0.26 rad、手臂 0.20 rad；后轮速度缩放按 1.3 m/s ÷ 5 cm ≈ 26 rad/s 留余量（速度上限 34.77）。
- 镜像表 `pinov1_mirror_maps()`：`left_* ↔ right_*`，roll / yaw 变号、躯干 yaw 变号。**用上游给的 `skating_op_mirror_fps30.csv` 核对**：镜像表作用在原片段上，应与镜像 CSV 逐帧一致。

**执行器参数（来自 `xdream.py`，IsaacLab 的 ImplicitActuator）**：

| 组 | 关节 | kp | kd | 力矩上限 | 速度上限 | armature |
|---|---|---|---|---|---|---|
| 腿 + 躯干 | 髋 pitch / roll / yaw、膝、躯干 | 40 | 3.0 | 42 N·m | 37.59 rad/s | 0.0019363 |
| 踝 | 踝 pitch | 40 | 3.0 | 42 N·m | 37.59 rad/s | 0.0019363 |
| 手臂 | 肩 pitch / roll / yaw、肘、腕 | 15 | 2.0 | 12 N·m | 23.04 rad/s | 0.00552096 |
| 后轮 | `*_rear_wheel_joint` | 0（速度控制） | **4.0**（即 kv） | 36 N·m | 34.77 rad/s | 0.0015028 |
| 前轮 | `*_front_wheel_joint` | 不加执行器，完全自由转动 | | | | |

所有关节 friction 0.001；`soft_joint_pos_limit_factor` 0.95；初始根高度 0.4 m。

**关节阻尼（原第 4 个问题，按 xdream.py 解决）**：上游 XML 的默认类给所有关节加了被动阻尼 1.0、armature 0.01（后轮也是阻尼 1.0）。xdream.py 里阻尼只来自执行器（上表的 kd / kv），前轮完全自由。所以 `pinov1.xml` 把关节被动阻尼设为 0、armature 换成上表的值、frictionloss 0.001，阻尼全部由执行器提供；前轮阻尼 0、armature 0。这样后轮的阻力就是速度控制的 kv 4.0（nemov3 是 0.5，pinov1 的后轮控制明显"更硬"）。


## 2. 数据工具（覆盖复制来的两个文件 + 新增 pkl 转换）

### 2.1 `assets/amp_sources/pinov1_skating/data/csv_to_npz.py`（改写）
- 输入：无表头 CSV（7 列根位姿 xyz + 四元数 xyzw，21 列关节，关节顺序 = `pino_joint21_nowheel_retarget.xml`）、`--fps 30`、`--start_s`（裁剪起点，本次 1.2）。
- 用 `pinov1.xml` 做正运动学，输出与 nemov3 npz 相同的键：`fps`、`joint_names`、`joint_pos`、`body_names`、`body_pos_w`、`body_quat_w`（wxyz）。轮子关节填 0。
- 保留 nemov3 版本里有用的检查，删掉 nemov3 专属的部分（Fast06 端点修正、`bind_relative_inverse` 坐标重映射、表头解析）：
  - 四元数归一化检查；
  - 关节限位检查（超出只警告，不截断）；
  - 输出时打印速度范围、轮子离地比例、首尾衔接（即我之前 `inspect_csv.py` 的内容）。
- 用法：`python data/csv_to_npz.py data/csv/skating_op_fps30.csv --start_s 1.2 --output_dir data/npz`。

### 2.2 `assets/amp_sources/pinov1_skating/data/replay_npz_mujoco.py`（改写）
- 回放 npz（或直接回放 CSV），交互窗口或输出视频（`--video`），左前视角、标注时间和四个轮子高度（即我之前临时的 `replay_csv.py`）。

### 2.3 `src/booster_mjlab/scripts/pinov1_skating_to_pkl.py`（新增，仿照 `nemov3_skating_to_pkl.py`）
- npz → `assets/amp/pinov1_skating/skating_op.pkl`（GMR 格式：`fps`、`root_pos`、`root_rot` xyzw、`dof_pos` 按 `PINOV1_JOINT_ORDER`、`local_body_pos`、`link_body_list`）。
- 只转原片段；左右平衡用训练时的镜像增强（同 nemov3），镜像 CSV 只用来核对镜像表。
- 用法：`uv run python -m booster_mjlab.scripts.pinov1_skating_to_pkl`。

## 3. 任务：`src/booster_mjlab/tasks/velocity/config/pinov1/`

照 nemov3 的结构（`__init__.py` 注册、`env_cfgs.py`、`rl_cfg.py`），但**只放第一个实验需要的东西**，不搬 nemov3 的历史分支。直接从 nemov3 已验证有效的配方起步（nemov3 Exp 33 / 41 那一套）：

| 设定 | 取值 | 来源 |
|---|---|---|
| 判别器特征 | 腿 + 手臂 + 躯干的关节角（不含轮子），不含机身速度 / 关节速度 | nemov3 Exp 11~19 |
| 判别器窗口 K / 梯度惩罚 / 风格权重 | 30 帧 / 1.0 / 0.7 | nemov3 rl_cfg |
| 相位时钟（只作观测） | **0.5 Hz**（按参考的左右周期 ~2 s；nemov3 是 0.74 Hz） | nemov3 Exp 18~19 |
| 观测 | 去掉轮子绝对角度 | nemov3 Exp 19 |
| 增强 | 镜像 + 速度 ±10% + 镜像损失（SymLoss，系数 10） | nemov3 Exp 30~33 |
| 指令 | 第一个实验只训**直行 0~1.3 m/s**（nemov3 的顺序：先直行、后转弯 Exp 41） | — |
| episode / 环境 / 轮数 | 10 s / 4096 / 先 2000 轮，每 400 轮看一次 | — |
| 执行器 | 第 1 节的 PD；部署延迟、域随机化等 sim2real 项目先不加（等上机前再加，nemov3 Exp 81~90 的顺序） | — |

### 不抬脚的约束（本方案的关键）
参考片段约一半的帧在抬脚，判别器只看关节角时仍会被"抬脚的腿型"吸引。用两项约束挡住：
1. **轮子贴地**：任一轮子的接地 site 离地 > 1 cm 时，按超出量的平方惩罚（4 个轮子都算；新奖励，写进 `mdp/`）。这是直接针对目标的约束。
2. **髋 roll 上限**（nemov3 的 `narrow_stance`，> 0.14 rad 时惩罚）：nemov3 不抬脚就是靠它挡住 Fast06 的抬脚风格。pinov1 的参考髋 roll 0~0.64 rad，上限值按参考的分布重新定（先算不抬脚帧的 95 分位）。

预期的冲突与应对：参考里蹬地和抬脚连在一起，加约束后风格奖励可能偏低、步态变小。判断时先看"有没有左右交替蹬地"，再看幅度。如果交替出不来，第二步才考虑只用"轮子都着地"的那些帧当参考（会让片段更短，先不做）。

### 默认姿态：先用用户给的姿态（方案 B）
nemov3 的教训（`nemov3_constants.py` 的 HOME_KEYFRAME 注释）：默认姿态太直立时，判别器光看"静态偏差"就能分出真假，腿学不动。pinov1 参考的平均腿姿（髋 pitch −0.54、膝 0.82）比用户默认（−0.1 / 0.2）蹲 0.4~0.6 rad，而上面的动作缩放只有 0.26 rad/单位。**用户决定先用 B**：照用户姿态，髋 pitch / 膝的动作缩放放大（同 nemov3 的逐关节放大），让探索够得到参考的蹲姿。**观察点**：训练初期风格奖励是否一直很低、腿是否一直站直不动；出现就改 A（默认姿态取参考平均蹲姿）。

## 4. 评估与判断标准（第一个实验）

工具：先把 nemov3 里会用到的评估改成可指定机器人（或在 `tools/pinov1_tools/` 复制精简版），至少要有：
- 影片（480p、左前视角，同 nemov3 `record_video.py`）；
- 四个轮子离地比例与最大离地高度；
- 速度跟踪 0.3 / 0.6 / 1.0 / 1.3 m/s；
- 左右交替（左右髋 pitch 相关 < −0.7 的比例）、摆腿幅度；
- 站立 30 s 摔倒数。

判断标准：
1. **不抬脚**：轮子离地 > 1 cm 的时间 < 5%；
2. **有滑冰步态**：0.6 m/s 以上左右交替蹬地、摆腿幅度随速度增加；
3. 速度跟踪误差 < 0.15 m/s，站立不摔；
4. 用户目视。

## 5. 执行顺序（每步做完都写进 `tools/pinov1_finetune.md`）

1. 资产与常量：`robots/pinov1/`，检查模型能编译、默认姿态落地、执行器数量 23、镜像表与镜像 CSV 一致。
2. 数据工具：改写 `csv_to_npz.py` / `replay_npz_mujoco.py`，新增 `pinov1_skating_to_pkl.py`，生成 pkl，回放影片给用户确认（裁剪后的动作）。
3. 任务注册 + 不抬脚约束 + 评估工具，冒烟训练（128 环境、3 轮）。
4. Exp 1 正式训练，按第 4 节评估。

## 6. 已决定（2026-11-03）
1. PD / 速度控制 / armature：照 `xdream.py`（第 1 节表格）。
2. 默认姿态：B，有问题再改 A。
3. 第一个实验只训直线前进 0~1.3 m/s。
4. 关节被动阻尼：照 `xdream.py`，全部由执行器提供（第 1 节）。

xdream.py 没有、仍沿用 nemov3 做法的：控制频率（策略 50 Hz）、延迟 / 域随机化（第一个实验先不加）。
