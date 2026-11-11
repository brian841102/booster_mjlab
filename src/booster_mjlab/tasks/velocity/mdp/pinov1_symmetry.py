"""Left/right symmetry data augmentation for the Pino v1 velocity tasks: ``nemov3_symmetry`` with pinov1's joint
mirror table (pinov1_mirror_maps, derived from pinov1.xml's joint axes). Same observation / command / clock rules."""

from functools import partial

from booster_mjlab.robots.pinov1.pinov1_constants import pinov1_mirror_maps
from booster_mjlab.tasks.velocity.mdp import nemov3_symmetry

mirror_obs = partial(nemov3_symmetry.mirror_obs, maps_fn=pinov1_mirror_maps)
mirror_actions = partial(nemov3_symmetry.mirror_actions, maps_fn=pinov1_mirror_maps)
augment_symmetries = partial(nemov3_symmetry.augment_symmetries, maps_fn=pinov1_mirror_maps)
