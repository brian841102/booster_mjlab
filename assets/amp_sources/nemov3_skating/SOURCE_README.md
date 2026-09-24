

# 旧机器人
conda activate isaaclab
cd /home/zhuning/zn/nemo/amp
python data/replay_npz_mujoco.py \
  --robot nemov3_old \
  --motion_file data/npz/nemov3_skating2.npz

# 新机器人
conda activate isaaclab
cd /home/zhuning/zn/nemo/amp
python data/replay_npz_mujoco.py \
  --robot nemov3 \
  --motion_file data/npz/nemov3_skating2.npz


/home/gez/github/flamingo_rl_train/external/amp/data


# 新动作示例
cd /home/gez/github/flamingo_rl_train/external/amp
python data/replay_npz_mujoco.py \
  --robot nemov3 \
  --motion_file data/npz/Robot_Model0530_V2_Skating02_IK_control.npz
