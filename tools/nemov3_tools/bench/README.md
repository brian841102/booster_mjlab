# 轮子台架测量（Exp 90 实机站立 5.8 Hz 抖动的排查）

**目的**：实机原地站立时，轮速目标以约 5.8 Hz 来回摆（±2.3 rad/s），仿真里同一个策略几乎不动；固定延迟 0~150 ms、每步随机命令滞后（最多 30~70 ms）都复现不了。剩下的候选是仿真轮子（理想速度执行器，kv 0.5、±6 N·m）与实机 RS01 的差别。这个测试**不经过策略**，直接给轮子一组已知命令，测出命令 → 速度的真实响应：死区、摩擦、滞后、一阶时间常数、增益、频率响应，再用这些去修仿真轮子。

文件（本目录）：
- `wheel_bench_profile.py` — 测试序列（刺激与分析共用，不依赖 ROS）。
- `wheel_bench_probe.py` — 机器人上的测试脚本。照 `tools/amp_left_yaw_probe.py` 的写法，**只通过共享电机驱动器的 DDS** 驱动一个轮子，不用策略链路。默认只读，`--stage hold` / `--stage full` 才会动电机。
- `wheel_bench_analyze.py` — 读 probe 的 `.jsonl`（或 rosbag），算死区、滞后、增益、频率响应，写 `*.result.json`。

> 为什么不走 `lhframework_hw` 的 `/wheel_commands`：硬件节点只在预热完成、`/internal_policy/start`（CH7）为真、且 `/leg_commands` 与 `/wheel_commands` 都新鲜时才放行轮速命令（`hw_node_control.cpp`），没有 policy runner 就走不通。直接用电机驱动器的 DDS，和已有的单关节诊断同一条路径。

## 测试序列（每个轮子 112 s，一次只动一个轮子，另一个命令 0）

| 时间 | 内容 | 看什么 |
|---|---|---|
| 0~5 s | 0 rad/s | 停转时的速度噪声、蠕动、静态力矩 |
| 5~25 s | 0 → +1.5 → 0 三角波 | 起转的命令阈值（死区）、回程停转阈值（迟滞 / 摩擦） |
| 25~45 s | 0 → −1.5 → 0 | 反方向 |
| 45~75 s | +1, −1, +2, −2, +4, −4 rad/s，各 3 s + 间隔 2 s | 死区时间、10~90% 上升时间、超调、稳态增益、正负方向差异 |
| 80~112 s | 振幅 1.5 rad/s、0.5 → 12 Hz 指数扫频 | 各频率的滞后（ms）与增益；特别看 6 Hz 附近 |

命令绝对值不超过 4 rad/s（策略站立时的轮速目标标准差约 2.3 rad/s）。

## 安全（必须全部满足）

1. **机器人吊起，两个轮子离地、能自由转**，周围无人手和线缆。
2. **策略链路全部停掉**（`lhframework_hw`、`lhframework_runner`、`sbus_to_cmdvel`）；只留共享电机驱动器（`start-amp-motor-driver.sh --domain-id 99`）。脚本发现另一个命令发布者就中止。
3. 操作员在急停旁；Ctrl-C 会先发约 100 ms 零速命令再 `disable`。
4. 脚本自动中止条件：反馈超过 100 ms 没更新、电机离线或有故障码、通道 / ID 不符、轮速超过 12 rad/s、命令循环卡顿超过 50 ms。命令速度不超过 4 rad/s。
5. 只动被选中的那一个轮子，所有 RPC 都只带它的名字（没有 all、没有 enable / 清错 / 标零）。

## 流程（交互进行；每一步都等上一步结果）

1. 吊起机器人，确认两个轮子能自由转（手转一下）。
2. 把 `bench/` 目录复制到机器人 `module/service/motion_control_sim2real_service/tools/wheel_bench/`（新目录，不覆盖任何现有文件）。
3. 起共享电机驱动器：`bash "$tools/start-amp-motor-driver.sh" --domain-id 99`，确认 `docker ps` 里只有 motor_driver。
4. **只读检查**：`probe-` 同样的 `pr ros2:exec` 方式运行 `wheel_bench_probe.py --wheel left --output <目录>/left.jsonl`，看打印的电机状态（位置、速度、mode、故障、在线）。
5. **hold 阶段**：`--stage hold`，输入 `YES`；设模式后 3 s 零速命令，看 `after set_mode` 打印的 mode / motor_state、手转轮子时是否有阻尼感（kd 0.5）。
6. **full 阶段**：`--stage full`，约 112 s（输出文件要换新名，脚本不覆盖已有文件）。
7. 右轮重复 4~6（`--wheel right`）。
8. 把 `.jsonl` 拷回开发机分析：`uv run --with numpy python tools/nemov3_tools/bench/wheel_bench_analyze.py left.jsonl`。先用 `--selftest` 确认分析本身没问题。

## 怎么解读

| 现象 | 说明 | 下一步 |
|---|---|---|
| 起转阈值 ≥ 0.3 rad/s，且回程停转阈值明显更小 | 死区 + 静摩擦：零速附近会形成极限环 | 仿真轮子加死区与库仑摩擦（`geom/joint frictionloss`），重训或先测 Exp 90 能否承受 |
| 6 Hz 处滞后 ≥ 60 ms 或增益 < 0.4 | 轮子在 6 Hz 时与命令几乎反相，闭环在该频率增益 ≥ 1 时就会振荡 | 仿真里把轮子的滞后、带宽改成实测值 |
| 滞后随幅值变化（1 rad/s 的死区时间明显大于 4 rad/s） | 非线性（死区 / 摩擦） | 同第一行 |
| 正、负方向明显不同 | 左右 / 方向不对称 | 单独建模或排查装配 |
| 全部指标都接近仿真（滞后 ~35 ms、无死区、增益 ~0.9） | 轮子本身不是原因 | 转向观测噪声形状（陀螺噪声集中在 1~1.4 Hz）、总线 / 驱动器抖动 |

## 局限

- 轮子悬空、无负载：没有地面摩擦和车身惯量，只测电机 + 驱动器 + 总线。
- 命令是直接发给电机驱动器的（目标 50 Hz 更新、帧 200 Hz 重复，和硬件节点一致），测到的是**驱动器 + 总线 + 电机**，不含 `lhframework_hw` 的软件层（它的软件层抖动已在 bag 里量过：很小）；分不开"电机本身"与"驱动器内部"，要分开需要 CAN 抓包（`capture_can_feedback.py --interfaces can2`）。
- 速度反馈取自电机驱动器的状态话题，时间用接收时刻。
- ⚠ 未验证：`set_mode` 之后 RS01 是否需要另外的 enable 才会响应 MIT 命令（HT 电机的诊断说"MIT 命令即可激活"）。hold 阶段就是用来确认这一点的。
