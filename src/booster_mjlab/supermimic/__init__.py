"""SuperMimic: motion tracking by backpropagating through a learned world model.

Algorithm package (like ``booster_mjlab.amp``): world model, policy and critic (``modules``), the replay buffer
(``storage``) and the training loop (``runners``). The task -- reference clips, the differentiable reward /
termination / observation and the simulator loop -- lives in ``booster_mjlab.tasks.supermimic``.
"""

from .config import SuperMimicRunnerCfg

__all__ = ["SuperMimicRunnerCfg"]
