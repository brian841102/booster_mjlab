import pickle, numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
d = pickle.load(open('assets/amp/nemov3_skating_straight/skatingfast06.pkl', 'rb'))
npz = np.load('assets/amp_sources/nemov3_skating/data/npz/Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz', allow_pickle=True); bn = [str(x) for x in npz['body_names']]
hR = npz['body_pos_w'][:, bn.index('right_wheel_link'), 2] - 0.075; hL = npz['body_pos_w'][:, bn.index('left_wheel_link'), 2] - 0.075
cfg = load_env_cfg('Mjlab-Velocity-Flat-Amp-Clock-Arms-FootLiftShaped-Booster-Nemov3'); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (1.2, 1.2); tw.rel_standing_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; lift = env.reward_manager.get_term_cfg('wheel_lift'); sens = env.scene['feet_ground_contact']
ids = lift.params['asset_cfg'].body_ids
for name, fr in (('right-foot lift peak', int(np.argmax(hR))), ('left-foot lift peak', int(np.argmax(hL)))):
    env.reset()
    rp = torch.tensor(d['root_pos'][fr], dtype=torch.float32, device='cuda:0')[None]; rq = d['root_rot'][fr]; rq_wxyz = torch.tensor([rq[3], rq[0], rq[1], rq[2]], dtype=torch.float32, device='cuda:0')[None]
    rp[:, 0] = 0; rp[:, 1] = 0; rp[:, 2] += 0.0
    robot.write_root_link_pose_to_sim(torch.cat([rp, rq_wxyz], dim=-1)); robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device='cuda:0'))
    robot.write_joint_state_to_sim(torch.tensor(d['dof_pos'][fr], dtype=torch.float32, device='cuda:0')[None], torch.zeros(1, 32, device='cuda:0'))
    print(f'RC {name}: frame {fr}; ref heights R {hR[fr]*100:.1f} L {hL[fr]*100:.1f} cm')
    for t in range(6):
        env.step(torch.zeros(1, 32, device='cuda:0'))
        h = (robot.data.body_link_pos_w[0, ids, 2] - 0.075).cpu().numpy() * 100; air = sens.data.current_air_time[0].cpu().numpy(); found = sens.data.found[0].cpu().numpy()
        v = lift.func(env, **lift.params)[0].item()
        print(f'   step {t}: h_R,h_L cm {h[0]:.1f},{h[1]:.1f} | air_time R,L {air[0]:.2f},{air[1]:.2f} | contact R,L {found[0]},{found[1]} | wheel_lift raw {v:.3f}')
env.close()
