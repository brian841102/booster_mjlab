nemov3 实验评估脚本（在仓库根目录运行，`.venv/bin/python3 tools/nemov3_tools/<script>.py ...`）。实验记录见 `tools/nemov3_finetune.md`。

## 常用（当前阶段：对称性、摆腿 / 站立、速度）
直接载入 checkpoint（`.pt`，用 play 的方式建策略），多环境批量跑，逐 episode 统计。

- `sym_batch_eval.py <task> <ckpt.pt> [envs=32] [cmds] [steps=750] [tag]` — **对称性评估（以它为准）**：用 play 的方式加载 checkpoint，每个速度同时跑 `envs` 个 episode；逐 episode 的节奏比（另报 0.45~0.55 带内占比）/ 幅度比 / 静态偏差，`SB-signed` 行给有符号均值（区分系统性偏向与镜像对）。门槛附近用 64 个 episode 复核（32 个有 ±0.03 的幅度比噪声）。
- `posture_batch_eval.py <task> <ckpt.pt> [envs=64] [cmds=0.3,0.6,1.0,1.3] [steps=600] [tag]` — 批量姿态评估（PyTorch 策略，逐 episode 统计再平均）：两轮前后摆动 std（前后摆腿幅度）、重心左右转移 std、base 相对两轮中点的偏移、躯干俯仰、前进速度、摔倒数，以及站立 / 行进时的转向角速度晃动 std（"原地乱扭"）和关节角速度 RMS。
- `turn_eval.py <task> <ckpt.pt> [envs=32] [vx=0,0.6,1.0,1.3] [wz=-1.3,-0.6,0,0.6,1.3] [steps=600] [tag]` — 转弯评估（Turn 任务）：在"前进速度 × 转向速度"网格上看转向 / 前进速度跟踪、摔倒、**转弯时的摆腿与直行相比**（左右腿分别列出，左转 +wz 时右腿是外侧腿）、是否仍左右交替、左转与右转是否互为镜像（左右关节偏差正负相反）。原地（vx = 0）旋转允许腿不摆。
- `stand_fall_time.py <task> <ckpt.pt> <tag>` — 指令 0（站立）下 256 个环境站 30 s：摔倒数、摔倒时间分布、摔倒者的初始速度。
- `fall_diag.py <task> <ckpt.pt> <tag> [envs=256]` — 站立摔倒诊断：摔倒方向（前 / 后 / 侧）、倒下前 1 / 3 / 6 s 的躯干俯仰 / 横滚晃动、轮速抖动，以及没倒的环境的站姿（躯干后仰、髋 pitch / 小腿平均、左右髋 pitch / yaw 差）。
- `speed_track.py <episodes> <steps> <cmds,逗号分隔> tag=onnx [tag=onnx ...]` — 同一个环境、同一套指令下比较多个检查点的前向速度跟踪（均值 / bias / RMSE，丢掉前 100 步起步段）。环境变量 `TASK` 指定任务。输入是 ONNX（训练时每次保存会在 run 目录导出 `<run>.onnx`，对应最近一次保存的 checkpoint，先 `cp` 出来）。
- `symmetry_check.py [n]` — 验证 `mdp/nemov3_symmetry.py` 的镜像规则：mirror(obs(s)) 是否等于 obs(镜像状态)（注意 `compute(update_history=True)`，不带参数会返回缓存）。**改了观测项（例如加入转弯）后必须重跑。**
- `disc_by_speed.py <task> <ckpt.pt> [envs=32] [cmds] [steps=600]` — 判别器诊断：按指令速度给策略样本打分（D 均值 / p10 / p90、风格奖励）、左右镜像后的得分，以及判别器对各 AMP 关节的敏感度（它在用哪些特征拒绝策略）。

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
