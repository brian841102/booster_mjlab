nemov3 实验评估脚本（在仓库根目录运行，`.venv/bin/python3 tools/nemov3_tools/<script>.py ...`）。实验记录见 `tools/nemov3_finetune.md`。

## 常用（当前阶段：对称性、摆腿 / 站立、速度）
直接载入 checkpoint（`.pt`，用 play 的方式建策略），多环境批量跑，逐 episode 统计。

- `sym_batch_eval.py <task> <ckpt.pt> [envs=32] [cmds] [steps=750] [tag]` — **对称性评估（以它为准）**：用 play 的方式加载 checkpoint，每个速度同时跑 `envs` 个 episode；逐 episode 的节奏比（另报 0.45~0.55 带内占比）/ 幅度比 / 静态偏差，`SB-signed` 行给有符号均值（区分系统性偏向与镜像对）。门槛附近用 64 个 episode 复核（32 个有 ±0.03 的幅度比噪声）。
- `posture_batch_eval.py <task> <ckpt.pt> [envs=64] [cmds=0.3,0.6,1.0,1.3] [steps=600] [tag]` — 批量姿态评估（PyTorch 策略，逐 episode 统计再平均）：两轮前后摆动 std（前后摆腿幅度）、重心左右转移 std、base 相对两轮中点的偏移、躯干俯仰、前进速度、摔倒数，以及站立 / 行进时的转向角速度晃动 std（"原地乱扭"）和关节角速度 RMS。
- `turn_eval.py <task> <ckpt.pt> [envs=32] [vx=0,0.6,1.0,1.3] [wz=-1.3,-0.6,0,0.6,1.3] [steps=600] [tag]` — 转弯评估（Turn 任务）：在"前进速度 × 转向速度"网格上看转向 / 前进速度跟踪、摔倒、**转弯时的摆腿与直行相比**（左右腿分别列出，左转 +wz 时右腿是外侧腿）、是否仍左右交替、左转与右转是否互为镜像（左右关节偏差正负相反）。原地（vx = 0）旋转允许腿不摆。
- `stand_fall_time.py <task> <ckpt.pt> <tag>` — 指令 0（站立）下 256 个环境站 30 s：摔倒数、摔倒时间分布、摔倒者的初始速度。
- `fall_diag.py <task> <ckpt.pt> <tag> [envs=256]` — 站立摔倒诊断：摔倒方向（前 / 后 / 侧）、倒下前 1 / 3 / 6 s 的躯干俯仰 / 横滚晃动、轮速抖动，以及没倒的环境的站姿（躯干后仰、髋 pitch / 小腿平均、左右髋 pitch / yaw 差）。
- `speed_track.py <episodes> <steps> <cmds,逗号分隔> tag=onnx [tag=onnx ...]` — 同一个环境、同一套指令下比较多个检查点的前向速度跟踪（均值 / bias / RMSE，丢掉前 100 步起步段）。环境变量 `TASK` 指定任务。输入是 ONNX（训练时每次保存会在 run 目录导出 `<run>.onnx`，对应最近一次保存的 checkpoint，先 `cp` 出来）。
- `symmetry_check.py [n]` — 验证 `mdp/nemov3_symmetry.py` 的镜像规则：mirror(obs(s)) 是否等于 obs(镜像状态)（注意 `compute(update_history=True)`，不带参数会返回缓存）。**改了观测项（例如加入转弯）后必须重跑。** 用 `TASK=<task>` 指定任务；焊死关节（`fixed_joints`）的任务也适用。
- `disc_by_speed.py <task> <ckpt.pt> [envs=32] [cmds] [steps=600]` — 判别器诊断：按指令速度给策略样本打分（D 均值 / p10 / p90、风格奖励）、左右镜像后的得分，以及判别器对各 AMP 关节的敏感度（它在用哪些特征拒绝策略）。

## 抬脚滑冰（Exp 46+）
- `lift_eval.py <task> <ckpt.pt> [envs=32] [vx=0.6,1.0,1.3] [wz=0] [steps=600] [tag]` — 抬脚统计：离地 > 2 / 5 cm 的时间、每只轮子每秒抬几次 / 峰高 / 每次多久、单轮支撑占比、左右交替、抬的是不是后脚、微弹跳，**步态周期**（右脚离地到下一次右脚离地；Fast06 1.33 s，用户接受 ±10%），以及单轮支撑时怎么平衡的 BAL 行。
- `lift_pose.py <task> <ckpt.pt> [envs=32] [vx=1.0,1.3] [steps=600] [tag] [--rows]` / `lift_pose.py ref` — **用户认定的姿态标准（CRIT 行）**：以每次抬脚为时间轴，抬起的是后脚（在支撑脚后方的时间 ≥ 80%）、后腿往后伸（抬脚前 60% 平均在后方 ≥ 8 cm）、抬起那侧的手整段在前（≥ 90%，否则就是"同手同脚"）；`--rows` 打印抬脚 5 个阶段的位置 / 手 / 膝 / 髋。`ref` = Fast06 的数值。
- `eval_turn.sh <task> <run dir> <tag> <iteration...>` — 转弯 checkpoint 的整套评估（Exp 41 / 78 / 79 用的那一套；等 checkpoint 存好才跑）：`turn_eval` 转弯网格、`sym_batch_eval` 直行对称、`posture_batch_eval` 姿态 / 站立晃动、`stand_fall_time` 站立 30 s。
- `eval_tempo.sh <task> <run dir> <tag> <iteration...>` — 一个 checkpoint 的整套评估（等 checkpoint 存好才跑，可与训练同时进行）：`lift_eval`（含 TEMPO：周期是否在 Fast06 ±10%）、`lift_pose` CRIT、`posture_batch_eval`。
- `record_video.py <task> <ckpt.pt> <out.mp4> [segments=forward] [w=854] [h=480] [title]` — 离屏录影（`MUJOCO_GL=egl`），左前方跟拍、由慢到快，画面标注指令 / 实际速度与两轮离地高度。分段可以写预设名 `forward`（慢到快直行，默认）/ `forward-turn`（直行 + 前进时转弯极限）/ `stand`（站 30 s）/ `forward-backward-turn`（站立 → 前进 → 后退到 −0.5 → 后退时 ±1.3 转弯 → 原地 ±1.3 → 站立），或自己写；分段也可以写 `速度:转弯角速度:秒`（例如 `1.0:1.3:5`），画面另外标注转弯指令 / 实际角速度。
- `backward_stance.py <task> <ckpt.pt> <tag> [vx=-0.3] [seconds=12] [envs=64] [start=pool|default]` — 后退时两轮会不会越劈越开（Exp 88 实机问题、Exp 90 评估）：静止 3 s 后给 vx，每秒打印两轮横向间距（机体系，cm）、两腿髋 roll、实际 vx、累计摔倒数。通过标准：间距 ≤ 25 cm、髋 roll ≤ 0.14 rad、无摔倒。
- `asym_probe.py <task> <ckpt.pt> [envs=32] [cases=all|name,name] [stand_s=20] [drive_s=12] [drive_vx=1.0]` — 找出哪一种左右不对称能复现实机症状（Exp 90 站立歪倒 / 走起来不对称）：每个工况单独建环境（关掉所有启动随机化），加一种不对称（左轮 kd / 摩擦、左腿关节零位偏差、躯干质心、IMU 倾斜），量静止与直行的偏航漂移、侧漂、躯干 roll、髋 roll、轮速蠕动。工况在脚本里的 `CASES`。
- `record_reference.py <out.mp4> [clip=.../skatingfast06.pkl] [start_s=2] [seconds=12] [w] [h] [title]` — 参考动捕片段按同样机位回放（无物理），用来和策略影片并排对照。
- `train_curve.py <train.log> [window=100] [metric substrings]` — 从训练 log 按区间平均 task / style 奖励与各项指标（判断"奖励不再增长就停"）。

## archive/（早期调试 / 一次性排查，保留是因为实验记录里引用了它们）
大多输入 ONNX、单环境（AMP 重置姿态池只有 ~5 个初始姿态，统计有偏，见实验记录「对称性评估方法的两次修正」），环境变量 `TASK` 指定任务。

步态是否出现（Exp 13~24 阶段）：
- `archive/gait_episodic.py <onnx> <cmd> [episodes] [tag]` — 当时的首选评估：每 200 步（=4 s 训练 episode）重置一次，统计每个 episode 的带通 rms / 峰值频率 / 右左相位（圆周均值 + 跨 episode 一致性）/ 相干性。**以后从零训练新步态（例如转弯）时，判断"步态有没有出现"仍可以用它。**
- `archive/gait_band.py <onnx> <cmd> [steps] [tag]` — 0.3~3 Hz 带通步态指标（rms / 峰值频率 / 右左相位差 / 相干性），同时打印三条参考。
- `archive/leg_series.py <onnx> <cmd> [steps] [out.npy]` — 打印髋 pitch/小腿的时间序列 + 慢/快分量 std，肉眼看步态。
- `archive/policy_stats.py <onnx> <cmd> <tag>` — 旧版腿部统计（摆动 std、左右相关、主频；主频容易被慢漂移主导）。
- `archive/tb_cmp.py name=<run_dir> ... <iters,逗号分隔>` — 从 tfevents 里对比判别器 / 奖励曲线。

判别器排查（排查 2 / 3、Exp 14）：
- `archive/disc_permutation.py <onnx> <ckpt.pt> cur <tag> <task> <K>` — 判别器特征置换重要性 + 帧序打乱检验（`AMP_BASE=0` 表示 AMP 观测里没有机身线速度/重力；关节速度去掉后还需要再改一下 D 的布局）。
- `archive/feat_stats.py`（同参数）— 策略 vs 参考的逐特征均值/标准差。
- `archive/chatter.py <onnx> <ckpt.pt> <noise 0/1>` — 确定性 vs 带探索噪声下的关节速度谱（验证"噪声骗速度特征"）。
- `archive/wheelzero.py <onnx> <0|1> <steps>` — 长 rollout 并可选择把观测里的轮子绝对角度置 0，验证"轮子角度观测导致长时间步态消失"（Exp 18；仅适用于旧观测 32 维 joint_pos 的 checkpoint，需 `NEMOV3_OBSERVE_WHEEL_ANGLE = True`）。

抬脚（Exp 25~28，已暂停）：
- `archive/measure_foot_lift.py` — 从参考动作档案（npz 世界坐标）量每条 clip 的脚抬起高度（轮最低点 = 轮心 z - 0.075；同时给出固定点定义作对照）、抬脚次数 / 持续时间 / 速度。
- `archive/foot_lift_eval.py <onnx> <cmd> [episodes] [tag]` — 策略 rollout 里的轮子抬脚高度（峰值 / 持续时间 / 占比），对照 Fast06 参考。
- `archive/lift_replay_check.py` — 把机器人直接摆成 Fast06 的抬脚峰值姿态并零动作运行几步，检查 `wheel_lift` 奖励的机制（高度 / 门控 / 腾空计时）以及轮子多快掉回地面。
- `archive/twist_eval.py <onnx> <cmd> [episodes] [tag]` — 骨盆偏航 / 髋 roll / 髋 yaw 在策略 rollout 里的摆动幅度（总标准差 + 0.4~3 Hz 带通 rms），并打印 Fast06 参考的标准差。
- `archive/ref_playback_feasibility.py [start_frame] [wheel_sign]` — 把 Fast06 参考关节轨迹作为 PD 目标开环回放（轮子按参考速度滚动），看仿真里轮子是否像参考那样抬起、机器人是否保持直立（用来判断抬脚的动力学可行性）。

站姿 / 对称性（Exp 29~32，已被上面的批量版取代）：
- `archive/stance_width.py` — 轮距与髋 roll 的分布（参考动作与策略），髋 roll 上限 0.14 rad 就是据此定的（Exp 29）。
- `archive/asymmetry_eval.py <onnx> [episodes] [cmds] [tag]` — 旧的对称性评估（单环境；已被 `sym_batch_eval.py` 取代）。
- `archive/posture_eval.py <onnx> [episodes] [cmds] [tag]` — 单环境姿态评估（已被 `posture_batch_eval.py` 取代）。
- `archive/base_midpoint.py [onnx] [episodes] [cmds]` — base 相对两轮中点的偏移（Exp 31，该奖励已从代码移除）。
- `archive/mirror_ratio_check.py <onnx> <ckpt.pt> [steps]` — PPO 对称数据增强里镜像样本的重要性比率（Exp 32：解释为什么续训时数据增强会失败）。
