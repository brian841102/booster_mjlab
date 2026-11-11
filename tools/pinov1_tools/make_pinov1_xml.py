"""Generate src/booster_mjlab/robots/pinov1/xmls/pinov1.xml (+ meshes) from the xdream_assets repository's
pino_joint23_wheel.xml. Actuator / joint parameters from the XDream X1 IsaacLab config (xdream.py, user 2026-11-03).

usage: python tools/pinov1_tools/make_pinov1_xml.py <xdream_assets checkout>
       (git clone ssh://git@code.xdreamdev.com:2222/somalab/xdream_assets.git; pinov1.xml was made from commit 2fdadd5)
"""
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path(sys.argv[1]) / "pino" / "pino_v1"
DST = Path(__file__).resolve().parents[2] / "src" / "booster_mjlab" / "robots" / "pinov1" / "xmls"

# (joint regex, kp, kd, force limit, armature, frictionloss) -- xdream.py ImplicitActuatorCfg groups.
POSITION = (
    (r".*_hip_(pitch|roll|yaw)_joint|.*_knee_joint|torso_joint|.*_ankle_pitch_joint", 40.0, 3.0, 42.0, 0.0019363),
    (r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist)_joint", 15.0, 2.0, 12.0, 0.00552096),
)
REAR_WHEEL = (r".*_rear_wheel_joint", 4.0, 36.0, 0.0015028)  # velocity control: kv, force limit, armature
FRICTIONLOSS = 0.001

root = ET.parse(SRC / "mjcf/pino_joint23_wheel.xml").getroot()
root.set("model", "pinov1")
root.find("compiler").set("meshdir", "meshes/")
root.insert(0, ET.Element("option", {"integrator": "implicitfast"}))

# Scene elements: mjlab adds its own terrain / lights.
asset = root.find("asset")
for el in list(asset):
    if el.tag in ("texture", "material"):
        asset.remove(el)
world = root.find("worldbody")
for el in list(world):
    if el.tag in ("geom", "light"):
        world.remove(el)

# Joint defaults: the upstream default class adds damping 1 / armature 0.01 to every joint; xdream.py has no passive
# damping (all of it comes from the actuators) and per-group armature, and the front wheels spin free.
# mjlab enables multi-CCD for mesh collisions, which needs zero geom margin (the upstream default class sets 0.001).
root.find("default/default/geom").attrib.pop("margin", None)
default_joint = root.find("default/default/joint")
for k in ("damping", "armature", "frictionloss"):
    default_joint.attrib.pop(k, None)
names = []
for j in root.iter("joint"):
    name = j.get("name")
    if name is None or j.get("type") == "free":
        continue
    names.append(name)
    for k in ("damping", "armature", "frictionloss", "actuatorfrcrange"):
        j.attrib.pop(k, None)
    j.set("damping", "0")
    if "wheel" in name:
        j.attrib.pop("range", None)
        j.set("limited", "false")
    if re.fullmatch(REAR_WHEEL[0], name):
        j.set("armature", str(REAR_WHEEL[3]))
        j.set("frictionloss", str(FRICTIONLOSS))
    elif "front_wheel" in name:
        j.set("armature", "0")
        j.set("frictionloss", "0")
    else:
        arm = next(g[4] for g in POSITION if re.fullmatch(g[0], name))
        j.set("armature", str(arm))
        j.set("frictionloss", str(FRICTIONLOSS))

# Wheels: name the collision meshes (friction randomization) and add a ground-contact site at the bottom of each wheel
# (the AMP pose-pool reset rejects poses with a site below the ground; nemov3.xml has the same sites).
RADII = {"rear": 0.0499, "front": 0.0375}
for body in root.iter("body"):
    name = body.get("name", "")
    if not name.endswith("_wheel_link"):
        continue
    for g in body.findall("geom"):
        if g.get("mesh", "").endswith("_C"):
            g.set("name", name.replace("_link", "_collision"))
    radius = RADII["rear" if "rear" in name else "front"]
    body.insert(1, ET.Element("site", {"name": name.replace("_link", "_ground"), "pos": f"0 0 -{radius}", "size": "0.005"}))

# Actuators: <position> (kp / kv = kd) for the 21 joints, <velocity> for the 2 rear wheels, none for the front wheels.
act = root.find("actuator")
for el in list(act):
    act.remove(el)
for name in names:
    if "front_wheel" in name:
        continue
    if re.fullmatch(REAR_WHEEL[0], name):
        kv, f = REAR_WHEEL[1], REAR_WHEEL[2]
        act.append(ET.Element("velocity", {"name": f"{name}_vel", "joint": name, "kv": str(kv),
                                           "forcelimited": "true", "forcerange": f"-{f} {f}"}))
    else:
        _, kp, kd, f, _ = next(g for g in POSITION if re.fullmatch(g[0], name))
        act.append(ET.Element("position", {"name": f"{name}_act", "joint": name, "kp": str(kp), "kv": str(kd),
                                           "forcelimited": "true", "forcerange": f"-{f} {f}"}))

# Sensors: same names as nemov3.xml (the velocity task reads robot/imu_ang_vel, robot/imu_lin_vel).
sensor = root.find("sensor")
for el in list(sensor):
    sensor.remove(el)
for tag, attrs in (
    ("framequat", {"name": "orientation", "objtype": "site", "objname": "imu"}),
    ("velocimeter", {"name": "imu_lin_vel", "site": "imu"}),
    ("gyro", {"name": "imu_ang_vel", "site": "imu"}),
    ("accelerometer", {"name": "imu_lin_acc", "site": "imu"}),
    ("subtreeangmom", {"name": "root_angmom", "body": "pelvis"}),
):
    sensor.append(ET.Element(tag, attrs))

ET.indent(root, space="  ")
header = (
    "<!-- Pino v1 (XDream X1, wheeled humanoid: 21 joints + 2 driven rear wheels + 2 free front wheels).\n"
    "     Generated from xdream_assets (ssh://git@code.xdreamdev.com:2222/somalab/xdream_assets.git, commit 2fdadd5)\n"
    "     pino/pino_v1/mjcf/pino_joint23_wheel.xml with these changes:\n"
    "       - meshdir meshes/; floor, light and their texture / material removed (mjlab adds the scene);\n"
    "       - joint parameters from the XDream X1 IsaacLab config (xdream.py): no passive damping (the upstream default\n"
    "         class had damping 1 / armature 0.01 on every joint), armature 0.0019363 legs / torso / ankles,\n"
    "         0.00552096 arms, 0.0015028 rear wheels, 0 front wheels; frictionloss 0.001 (front wheels 0);\n"
    "       - actuators: torque <motor>s replaced by <position> kp 40 / kv 3 (42 Nm) for legs, torso, ankles,\n"
    "         kp 15 / kv 2 (12 Nm) for arms, <velocity> kv 4 (36 Nm) for the rear wheels, none for the front wheels;\n"
    "       - sensors renamed to the nemov3.xml set (imu_lin_vel, imu_ang_vel, ...);\n"
    "       - default geom margin 0.001 removed (mjlab's multi-CCD collision needs margin 0);\n"
    "       - wheel collision meshes named <wheel>_collision, a <wheel>_ground site at the bottom of each wheel. -->\n"
)
DST.mkdir(parents=True, exist_ok=True)
(DST / "pinov1.xml").write_text(header + ET.tostring(root, encoding="unicode") + "\n")
shutil.copytree(SRC / "meshes", DST / "meshes", dirs_exist_ok=True)
print("wrote", DST / "pinov1.xml", "joints", len(names))
