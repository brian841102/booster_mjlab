"""Viser-only switches for interactive `play` sessions.

The Viser viewer only lets *command terms* add GUI controls, so the switches
live on a command term that emits no command at all. Events that should be
toggleable at runtime read the switch state through `push_if_enabled`-style
wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    import viser
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class PlayControlsCommand(CommandTerm):
    """Holds runtime switches shown in the Viser "Play controls" folder."""

    cfg: PlayControlsCommandCfg

    def __init__(self, cfg: PlayControlsCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.push_enabled = cfg.push_enabled
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        with server.gui.add_folder("Play controls"):
            push = server.gui.add_checkbox(
                "Push disturbance", initial_value=self.push_enabled
            )

            @push.on_update
            def _(_) -> None:
                self.push_enabled = push.value


@dataclass(kw_only=True)
class PlayControlsCommandCfg(CommandTermCfg):
    push_enabled: bool = False
    """Initial state of the "Push disturbance" switch."""

    def build(self, env: ManagerBasedRlEnv) -> PlayControlsCommand:
        return PlayControlsCommand(self, env)


def push_if_enabled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "play_controls",
) -> None:
    """`push_by_setting_velocity`, gated by the Viser "Push disturbance" switch."""
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PlayControlsCommand)
    if term.push_enabled:
        push_by_setting_velocity(env, env_ids, velocity_range)
