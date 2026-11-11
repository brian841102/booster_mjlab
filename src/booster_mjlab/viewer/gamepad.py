"""Native MuJoCo play viewer whose velocity command comes from a gamepad.

Reads the Linux joystick device (``/dev/input/js*``, the kernel's joystick API: 8-byte events of time / value / type /
number) on a background thread, so no extra package is needed. Every control step it writes the twist command of all
environments from the sticks:

* forward speed  = left stick up / down, scaled to the task's ``lin_vel_x`` range (up -> max, down -> min, e.g. -0.5);
* yaw rate       = right stick left / right, scaled to the task's ``ang_vel_z`` range (left -> turn left, +wz);
* both sticks released -> 0 / 0, i.e. commanded to stand.

``heading_hold=K`` (play ``--heading-hold [K]``, off by default so the policy's own yaw drift stays visible) adds an
outer heading loop like mjlab's ``heading_command``: while driving with the turn stick released the yaw-rate command is
wz = -K * wrap(heading - target), clipped to the ``ang_vel_z`` range; the target is the heading at the moment the turn
stick is released (or the robot starts moving / the episode resets), so steering never fights it.

Axis numbers default to the Zikway HID gamepad layout read on 2026-11-05 (axes 0 / 1 left stick, 2 / 3 right stick,
4 / 5 triggers, 6 / 7 d-pad); override them with ``--gamepad-axes VX,WZ``. Pressed buttons are printed with their
number; ``--gamepad-reset-button N`` resets the environment on that button (Enter in the MuJoCo window always does).
"""

from __future__ import annotations

import os
import struct
import threading
import time

import torch
from mjlab.viewer import NativeMujocoViewer

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
AXIS_MAX = 32767.0


class Gamepad:
    """Latest axis / button state of a Linux joystick device, updated by a daemon thread."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.axes: dict[int, float] = {}
        self.buttons: dict[int, int] = {}
        self.pressed: list[int] = []  # buttons pressed since the last ``pop_pressed``
        self._lock = threading.Lock()
        self._fd = os.open(path, os.O_RDONLY)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                event = os.read(self._fd, 8)
            except OSError:
                print(f"[gamepad] {self.path} disconnected; command stays at 0")
                with self._lock:
                    self.axes.clear()
                return
            _, value, kind, number = struct.unpack("IhBB", event)
            with self._lock:
                if kind & JS_EVENT_AXIS:
                    self.axes[number] = max(-1.0, value / AXIS_MAX)
                elif kind & JS_EVENT_BUTTON:
                    self.buttons[number] = value
                    if value and not kind & JS_EVENT_INIT:
                        self.pressed.append(number)

    def axis(self, number: int) -> float:
        with self._lock:
            return self.axes.get(number, 0.0)

    def pop_pressed(self) -> list[int]:
        with self._lock:
            pressed, self.pressed = self.pressed, []
        return pressed


def _deadzone(x: float, dz: float) -> float:
    """Zero inside +-dz, rescaled to reach +-1 at the edge."""
    if abs(x) <= dz:
        return 0.0
    return (abs(x) - dz) / (1.0 - dz) * (1.0 if x > 0 else -1.0)


class GamepadNativeViewer(NativeMujocoViewer):
    """mjlab's native viewer with the "twist" command driven by a gamepad."""

    def __init__(
        self,
        env,
        policy,
        *args,
        device_path: str = "/dev/input/js0",
        vx_axis: int = 1,
        wz_axis: int = 2,
        reset_button: int | None = None,
        deadzone: float = 0.08,
        command_name: str = "twist",
        heading_hold: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(env, policy, *args, **kwargs)
        self.gamepad = Gamepad(device_path)
        self.vx_axis, self.wz_axis = vx_axis, wz_axis
        self.reset_button = reset_button
        self.deadzone = deadzone
        self.term = env.unwrapped.command_manager.get_term(command_name)
        ranges = self.term.cfg.ranges
        self.vx_range, self.wz_range = tuple(ranges.lin_vel_x), tuple(ranges.ang_vel_z)
        self._last_print = 0.0
        self.heading_hold = heading_hold
        self._target: torch.Tensor | None = None  # per-env target heading of the heading hold
        self._robot = env.unwrapped.scene[self.term.cfg.entity_name]
        self._episode_length = env.unwrapped.episode_length_buf
        print(
            f"[gamepad] {device_path}: left stick (axis {vx_axis}) -> vx {self.vx_range} m/s, right stick (axis "
            f"{wz_axis}) -> wz {self.wz_range} rad/s; release both to stand. Reset: Enter"
            + (f" or button {reset_button}" if reset_button is not None else "")
            + (f"; heading hold K = {heading_hold}" if heading_hold else "; heading hold off")
        )

    def _command(self) -> tuple[float, float]:
        # Stick up / left read negative on this layout, so both are flipped.
        fwd = _deadzone(-self.gamepad.axis(self.vx_axis), self.deadzone)
        turn = _deadzone(-self.gamepad.axis(self.wz_axis), self.deadzone)
        vx = fwd * self.vx_range[1] if fwd >= 0 else -fwd * self.vx_range[0]
        wz = turn * self.wz_range[1] if turn >= 0 else -turn * self.wz_range[0]
        return vx, wz

    def _execute_step(self) -> bool:
        for button in self.gamepad.pop_pressed():
            if button == self.reset_button:
                self.request_reset()
            else:
                print(f"[gamepad] button {button}")
        vx, wz = self._command()
        hold = None
        if self.heading_hold:
            heading = self._robot.data.heading_w
            fresh = self._episode_length <= 1
            if self._target is None or wz != 0.0 or vx == 0.0:
                self._target = heading.clone()  # steering / standing: follow the robot, hold from here on
            elif fresh.any():
                self._target[fresh] = heading[fresh]
            if wz == 0.0 and vx != 0.0:
                err = heading - self._target
                hold = (-self.heading_hold * torch.atan2(torch.sin(err), torch.cos(err))).clamp(*self.wz_range)
        term = self.term
        # Hold the command: no resampling, no standing / heading / world-frame envs overriding it.
        term.time_left.fill_(1.0e9)
        for flag in ("is_standing_env", "is_heading_env", "is_world_env"):
            if hasattr(term, flag):
                getattr(term, flag).fill_(False)
        term.vel_command_b[:] = torch.tensor((vx, 0.0, wz), device=term.vel_command_b.device)
        if hold is not None:
            term.vel_command_b[:, 2] = hold
        now = time.monotonic()
        if now - self._last_print > 1.0:
            extra = f"  heading-hold wz {hold[0].item():+.3f}" if hold is not None else ""
            print(f"[gamepad] command vx {vx:+.2f} m/s  wz {wz:+.2f} rad/s{extra}   ", end="\r", flush=True)
            self._last_print = now
        return super()._execute_step()
