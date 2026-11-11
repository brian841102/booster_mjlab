"""Thin wrapper around mjlab's play script.

Swaps the viser viewer for :class:`RecordingViserPlayViewer`, which
adds a record / stop-recording button that captures the browser view to an mp4
under ``--record-dir``. Frames are timed by the sim clock, so the video plays
back in real time even when the machine can't step the policy at full speed.

With ``--gamepad /dev/input/js0`` it opens the native MuJoCo viewer instead and drives the velocity command from a
gamepad (:class:`GamepadNativeViewer`: left stick forward / backward, right stick turn; ``--gamepad-axes VX,WZ`` and
``--gamepad-reset-button N`` adjust the mapping). ``--heading-hold [K]`` (gamepad only, default off; K defaults to 1.0)
adds a heading-hold outer loop while the turn stick is released (see :mod:`booster_mjlab.viewer.gamepad`).
"""

import functools
import sys

import mjlab.scripts.play as mjlab_play

from booster_mjlab.viewer import DEFAULT_VIDEO_DIR, GamepadNativeViewer, RecordingViserPlayViewer


def _pop_flag(argv: list[str], flag: str) -> str | None:
    """Remove ``--flag VALUE`` or ``--flag=VALUE`` from argv; return VALUE."""
    for i, tok in enumerate(argv):
        if tok == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value")
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if tok.startswith(flag + "="):
            value = tok.split("=", 1)[1]
            del argv[i]
            return value
    return None


def _pop_optional_float(argv: list[str], flag: str, default: float) -> float | None:
    """Remove ``--flag``, ``--flag VALUE`` or ``--flag=VALUE`` from argv; return VALUE, ``default`` without one, or None."""
    for i, tok in enumerate(argv):
        if tok.startswith(flag + "="):
            del argv[i]
            return float(tok.split("=", 1)[1])
        if tok == flag:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            try:
                value = float(nxt) if nxt is not None else None
            except ValueError:
                value = None
            del argv[i : i + (2 if value is not None else 1)]
            return default if value is None else value
    return None


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        # mjlab's help won't list --record-dir (we pop it before mjlab parses),
        # so surface it here.
        print(
            "booster_mjlab play adds:\n"
            "  --record-dir STR  (viser viewer: where recorded mp4s are written, "
            f"default {DEFAULT_VIDEO_DIR})\n"
            "  --gamepad PATH    (e.g. /dev/input/js0: native MuJoCo viewer, velocity command from the gamepad)\n"
            "  --gamepad-axes VX,WZ          (axis numbers, default 1,2: left stick up/down, right stick left/right)\n"
            "  --gamepad-reset-button N      (button that resets the env; default none, Enter always works)\n"
            "  --heading-hold [K]            (gamepad: hold the heading while the turn stick is released, wz = -K * error;\n"
            "                                 off by default, K = 1.0 if given without a value)\n",
            file=sys.stderr,
        )
    record_dir = _pop_flag(argv, "--record-dir")
    gamepad = _pop_flag(argv, "--gamepad")
    gamepad_axes = _pop_flag(argv, "--gamepad-axes")
    gamepad_reset = _pop_flag(argv, "--gamepad-reset-button")
    heading_hold = _pop_optional_float(argv, "--heading-hold", 1.0)
    if heading_hold is not None and gamepad is None:
        raise SystemExit("--heading-hold needs --gamepad")
    if gamepad is not None:
        vx_axis, wz_axis = (int(a) for a in (gamepad_axes or "1,2").split(","))
        mjlab_play.NativeMujocoViewer = functools.partial(  # type: ignore[assignment]
            GamepadNativeViewer,
            device_path=gamepad,
            vx_axis=vx_axis,
            wz_axis=wz_axis,
            reset_button=int(gamepad_reset) if gamepad_reset is not None else None,
            heading_hold=heading_hold,
        )
        if _pop_flag(argv, "--viewer") not in (None, "native", "auto"):
            raise SystemExit("--gamepad uses the native MuJoCo viewer")
        argv += ["--viewer", "native"]
    sys.argv = [sys.argv[0], *argv]

    # Viser viewer with a record button.
    mjlab_play.ViserPlayViewer = functools.partial(  # type: ignore[assignment]
        RecordingViserPlayViewer,
        video_dir=record_dir or DEFAULT_VIDEO_DIR,
    )

    mjlab_play.main()


if __name__ == "__main__":
    main()
