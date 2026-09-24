import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs = dict(a.split('=', 1) for a in sys.argv[1:-1]); its = [int(x) for x in sys.argv[-1].split(',')]
tags = ['Discriminator/accuracy_policy','Discriminator/accuracy_expert','Discriminator/policy_pred','Discriminator/expert_pred','Discriminator/loss','Train/mean_reward','Train/mean_episode_length','Policy/mean_std','Metrics/style_reward','Episode_Reward/track_linear_velocity','Episode_Reward/upright','Episode_Reward/action_rate_l2','Episode_Reward/angular_momentum','Episode_Reward/self_collisions']
for name, d in runs.items():
    f = glob.glob(f'{d}/events.out.tfevents*')[0]; ea = EventAccumulator(f, size_guidance={'scalars': 0}); ea.Reload(); avail = set(ea.Tags()['scalars'])
    print('==', name)
    for tg in tags:
        cand = [t for t in avail if t.endswith(tg.split('/', 1)[1]) or t == tg]
        if not cand: continue
        t = tg if tg in avail else sorted(cand, key=len)[0]
        ev = ea.Scalars(t); d2 = {e.step: e.value for e in ev}
        vals = [f'{d2[min(d2, key=lambda s: abs(s-i))]:.3f}' for i in its]
        print(f'  {t:48s}', ' '.join(vals))
