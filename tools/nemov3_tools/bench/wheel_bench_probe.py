#!/usr/bin/env python3
"""Wheel bench test: drive ONE wheel through the public motor-driver DDS with the profile of wheel_bench_profile.py.

Modelled on the existing single-joint diagnostic tools/amp_left_yaw_probe.py (same bootstrap, same motor-driver RPCs, same
guard style). It does NOT use the policy chain: only the shared motor driver (domain 99) must be running; lhframework_hw,
the runner and sbus_to_cmdvel must be stopped. The robot must be HUNG with the wheels free.

Command sent to the wheel, like lhframework_hw does with mode_type1 = 0 and no torque assist (the config's wheel_mit_* are
all 0): MIT frame, position 0, velocity = target, effort 0, kp 0, kd 0.5. The target is updated at 50 Hz (the policy's rate)
and the frame is repeated at 200 Hz (the hardware node's rate).

    python3 wheel_bench_probe.py --wheel left --output <dir>/left.jsonl                  # read only
    python3 wheel_bench_probe.py --wheel left --output <dir>/left.jsonl --stage hold     # arm, 3 s of zero velocity
    python3 wheel_bench_probe.py --wheel left --output <dir>/left.jsonl --stage full     # the whole profile

Stages: (no --stage) listens to the feedback only, no RPC and no command; hold = disable, set mode, enable (this wheel
only), 3 s of zero velocity commands, disable (checks that the wheel is controllable); full = hold, then the profile (112 s).
Output: JSON lines with every feedback frame (source timestamp, receive time, position, velocity, effort, current,
temperature, mode, fault) and every command; read it with wheel_bench_analyze.py.
"""
import argparse
import asyncio
import json
import math
import signal
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # tools/: calibrate_upper_zero (bootstrap, protobuf loader)
import calibrate_upper_zero as calibration  # noqa: E402
import wheel_bench_profile as profile  # noqa: E402

WHEELS = {"left": ("left_wheel_joint", 2, 10), "right": ("right_wheel_joint", 2, 5)}  # name, can channel, can id
MODE_TYPE1 = 0  # lhframework_hw.yaml: mode_type1
KP, KD = 0.0, 0.5  # motor_full_body.toml: wheels kp_in_use 0, kd_in_use 0.5
TARGET_HZ, FRAME_HZ = 50.0, 200.0
ABORT_SPEED = 12.0  # rad/s measured; the commanded speed never exceeds profile.MAX_ABS
STALE_S = 0.1
HOLD_SECONDS = 3.0


class Probe:
    def __init__(self, runtime, pb, output, wheel):
        self.runtime, self.pb, self.output = runtime, pb, output
        self.name, self.chan, self.canid = WHEELS[wheel]
        self.snapshot = None
        self.received = 0.0
        self.foreign_command = False
        self.own_stamps = deque(maxlen=4096)
        self.stop = False
        self.sequence = 0
        self.last_frame = 0.0

    def log(self, kind, **data):
        self.output.write(json.dumps(dict(kind=kind, monotonic_ns=time.monotonic_ns(), **data)) + "\n")

    async def attach(self):
        ep = self.runtime.subscriber("/motor_driver/state_cyclonedds")
        await ep.bus.subscribe_bytes(ep.channel, self.receive)
        self.command = self.runtime.publisher("/motor_driver/command_cyclonedds")
        await self.command.bus.subscribe_bytes(self.command.channel, self.receive_command)

    async def receive(self, payload):
        state = self.pb.MotorStateArray.FromString(bytes(payload))
        if self.snapshot is not None and state.sequence == self.snapshot.sequence:
            return
        self.snapshot, self.received = state, time.monotonic()
        for m in state.motors:
            if m.name == self.name:
                self.log("feedback", sequence=state.sequence, source_us=state.timestamp_us, position=m.position,
                         velocity=m.velocity, effort=m.effort, current=m.current, temperature=m.temperature,
                         mode=m.mode, motor_state=m.motor_state, fault=m.fault_code, online=m.online)

    async def receive_command(self, payload):
        cmd = self.pb.MotorCommandArray.FromString(bytes(payload))
        if cmd.monotonic_timestamp_us not in self.own_stamps:
            self.foreign_command = True
            self.log("foreign_command", names=[m.name for m in cmd.motors])

    def read(self):
        if self.foreign_command:
            raise RuntimeError("another motor command publisher is active; stop the competing controller")
        if self.stop:
            raise RuntimeError("operator interrupted the test")
        if self.snapshot is None or time.monotonic() - self.received > STALE_S:
            raise RuntimeError("motor feedback missing or stale (> 100 ms)")
        for m in self.snapshot.motors:
            if m.name == self.name:
                if (m.chan, m.canid) != (self.chan, self.canid):
                    raise RuntimeError(f"{self.name}: can channel/id {(m.chan, m.canid)} is not {(self.chan, self.canid)}")
                if not m.online or m.fault_code:
                    raise RuntimeError(f"{self.name}: offline or fault {m.fault_code}")
                if not all(math.isfinite(x) for x in (m.position, m.velocity)):
                    raise RuntimeError(f"{self.name}: nonfinite feedback")
                if abs(m.velocity) > ABORT_SPEED:
                    raise RuntimeError(f"{self.name}: speed {m.velocity:+.2f} rad/s over {ABORT_SPEED}")
                return m
        raise RuntimeError(f"{self.name} not in the motor feedback")

    async def rpc(self, operation):
        mode = operation == "set_mode"
        cls = self.pb.SetMotorModeRequest if mode else self.pb.MotorTargetRequest
        req = cls(names=[self.name], all=False)
        if mode:
            req.mode = MODE_TYPE1
        ep = self.runtime.rpc_client(f"motor_driver_{operation}_cyclonedds")
        payload = await ep.bus.request_bytes(ep.channel, req.SerializeToString(), timeout_sec=1.0)
        reply = (self.pb.SetMotorModeResponse if mode else self.pb.CommonResponse).FromString(bytes(payload))
        self.log("rpc", operation=operation, success=reply.success, message=reply.message)
        if not reply.success or reply.failed_names:
            raise RuntimeError(f"{operation} failed: {reply.message}")

    async def send(self, velocity):
        velocity = max(-profile.MAX_ABS, min(profile.MAX_ABS, velocity))
        self.sequence += 1
        stamp = time.monotonic_ns() // 1000
        self.own_stamps.append(stamp)
        cmd = self.pb.MotorCommandArray(timestamp_us=time.time_ns() // 1000, monotonic_timestamp_us=stamp,
                                       sequence=self.sequence)
        cmd.motors.add(name=self.name, command_mask=31, position=0.0, velocity=velocity, effort=0.0, kp=KP, kd=KD)
        self.log("command", velocity=velocity, sequence=self.sequence)
        await self.command.bus.publish_bytes(self.command.channel, cmd.SerializeToString())

    async def drive(self, seconds, target_fn, label):
        """Frames at FRAME_HZ, target refreshed at TARGET_HZ; guards every frame."""
        start = last = time.monotonic()
        next_frame, target, next_status = start, 0.0, 0.0
        step = 1.0 / FRAME_HZ
        while True:
            now = time.monotonic()
            if now - last > 0.05:
                raise RuntimeError("command loop stalled > 50 ms")
            last = now
            t = now - start
            if t >= seconds:
                return
            m = self.read()
            if int(t * TARGET_HZ) != int((t - step) * TARGET_HZ) or t < step:
                target = target_fn(t)
            await self.send(target)
            if t >= next_status:
                print(f"{label} t={t:6.1f}s cmd={target:+.2f} vel={m.velocity:+.3f} effort={m.effort:+.3f}", flush=True)
                next_status += 2.0
            next_frame += step
            await asyncio.sleep(max(0.0, next_frame - time.monotonic()))

    async def run(self, stage):
        await self.attach()
        await asyncio.sleep(2.0)  # collect feedback and discover competing publishers before anything is sent
        m = self.read()
        print(f"{self.name} can{self.chan} id{self.canid}: pos={m.position:+.4f} vel={m.velocity:+.4f} effort={m.effort:+.4f} "
              f"mode={m.mode} motor_state={m.motor_state} fault={m.fault_code} online={m.online} temp={m.temperature:.1f}", flush=True)
        if abs(m.velocity) > 0.3:
            raise RuntimeError("the wheel is not stationary (< 0.3 rad/s required)")
        self.log("plan", stage=stage, wheel=self.name, duration=profile.DURATION, kp=KP, kd=KD, mode=MODE_TYPE1)
        if stage is None:
            print("READ ONLY: no RPC and no command sent.", flush=True)
            return
        try:
            await self.rpc("disable")
            await self.rpc("set_mode")
            await asyncio.sleep(0.1)
            # lhframework_hw does setModeAllType1 then enableAll; without the enable the RS01 ignores the MIT frames
            # (Exp 90 bench, first attempt: wheel limp, effort 0 at a 1.35 rad/s command). Only this wheel is enabled.
            await self.rpc("enable")
            await self.send(0.0)
            await asyncio.sleep(0.1)
            m = self.read()
            print(f"after set_mode + enable: mode={m.mode} motor_state={m.motor_state} fault={m.fault_code} "
                  f"temp={m.temperature:.1f}", flush=True)
            await self.drive(HOLD_SECONDS, lambda t: 0.0, "hold")
            if stage == "full":
                self.log("marker", name="test_start")
                await self.drive(profile.DURATION, profile.command, "test")
            await self.drive(1.0, lambda t: 0.0, "stop")
        finally:
            try:
                for _ in range(20):  # zero velocity for 100 ms even after an abort
                    await self.send(0.0)
                    await asyncio.sleep(1.0 / FRAME_HZ)
            finally:
                await self.rpc("disable")
                print(f"{self.name} disable acknowledged.", flush=True)


async def run(args):
    bootstrap = calibration.communication_bootstrap()
    pb = calibration.load_protocol()
    runtime = await bootstrap(HERE / "wheel_bench_comm.yaml")
    try:
        with args.output.open("x") as output:
            probe = Probe(runtime, pb, output, args.wheel)
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, setattr, probe, "stop", True)
            try:
                await probe.run(args.stage)
            except BaseException as exc:
                probe.log("stopped", reason=str(exc))
                raise
            finally:
                output.flush()
    finally:
        await runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", choices=tuple(WHEELS), required=True)
    parser.add_argument("--stage", choices=("hold", "full"), help="default: read only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yes", action="store_true", help="the caller (run_wheel_probe.sh) already asked for the confirmation")
    args = parser.parse_args()
    if args.stage and not args.yes:
        print("The robot must be HUNG with BOTH wheels free, the policy chain stopped, the operator at the e-stop.")
        print("Type YES to continue: ", end="", flush=True)
        if sys.stdin.readline().strip() != "YES":
            print("not confirmed.")
            return 1
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(f"STOP: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
