"""Wheel bench test profile, shared by the stimulus publisher and the analysis (pure Python, no ROS).

One wheel is driven at a time (robot hung, wheels off the ground); the other wheel is commanded 0. Times are seconds
since the first /wheel_commands message of the test. Velocities are the robot-axis wheel velocity targets (rad/s), the
same quantity the policy publishes on /wheel_commands (its output * 20).

  hold   0 .. 5 s       0 rad/s           noise / creep / quantization of the stopped wheel
  ramp+  5 .. 25 s      0 -> +1.5 -> 0    triangular, 10 s up + 10 s down: first-motion threshold (deadband), hysteresis
  ramp-  25 .. 45 s     0 -> -1.5 -> 0
  steps  45 .. 75 s     +1, -1, +2, -2, +4, -4 rad/s, 3 s on + 2 s off each: delay, rise time, overshoot, gain,
                        and the difference between the two directions
  chirp  80 .. 112 s    1.5 rad/s amplitude, 0.5 -> 12 Hz exponential sweep over 30 s, 1 s taper at both ends:
                        gain and phase (delay) versus frequency; the hardware's standing jitter is at ~5.8 Hz
"""

import math

HOLD = (0.0, 5.0)
RAMP_POS = (5.0, 25.0)
RAMP_NEG = (25.0, 45.0)
STEPS_START = 45.0
STEP_LEVELS = (1.0, -1.0, 2.0, -2.0, 4.0, -4.0)
STEP_ON, STEP_OFF = 3.0, 2.0
CHIRP = (80.0, 112.0)
CHIRP_SWEEP = 30.0
CHIRP_F0, CHIRP_F1, CHIRP_AMP = 0.5, 12.0, 1.5
RAMP_PEAK = 1.5
DURATION = CHIRP[1]
MAX_ABS = 4.0  # nothing above this is ever commanded


def step_windows():
    """[(level, t_on_start, t_on_end)] of the step section."""
    out, t = [], STEPS_START
    for level in STEP_LEVELS:
        out.append((level, t, t + STEP_ON))
        t += STEP_ON + STEP_OFF
    return out


def chirp_phase(t):
    """Phase (rad) of the exponential sweep at time t since the chirp start."""
    k = math.log(CHIRP_F1 / CHIRP_F0) / CHIRP_SWEEP
    return 2 * math.pi * CHIRP_F0 * (math.exp(k * t) - 1) / k


def chirp_freq(t):
    return CHIRP_F0 * math.exp(math.log(CHIRP_F1 / CHIRP_F0) / CHIRP_SWEEP * t)


def command(t):
    """Wheel velocity target (rad/s) at time t (s) since the start of the test; 0 outside the profile."""
    if t < 0 or t >= DURATION:
        return 0.0
    if RAMP_POS[0] <= t < RAMP_POS[1]:
        u = (t - RAMP_POS[0]) / (RAMP_POS[1] - RAMP_POS[0])
        return RAMP_PEAK * (2 * u if u < 0.5 else 2 * (1 - u))
    if RAMP_NEG[0] <= t < RAMP_NEG[1]:
        u = (t - RAMP_NEG[0]) / (RAMP_NEG[1] - RAMP_NEG[0])
        return -RAMP_PEAK * (2 * u if u < 0.5 else 2 * (1 - u))
    for level, t_on, t_off in step_windows():
        if t_on <= t < t_off:
            return level
    if CHIRP[0] <= t < CHIRP[1]:
        tc = t - CHIRP[0]
        taper = min(1.0, tc, CHIRP[1] - CHIRP[0] - tc)  # 1 s linear taper at both ends
        return CHIRP_AMP * taper * math.sin(chirp_phase(min(tc, CHIRP_SWEEP)))
    return 0.0


assert max(abs(l) for l in STEP_LEVELS) <= MAX_ABS and CHIRP_AMP <= MAX_ABS and RAMP_PEAK <= MAX_ABS
