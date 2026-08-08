"""Compute VisTacFusion GT pose from raw pose json files.

Given a data root (sim or real), for each object:
  1. Read session_000/session.json -> rz0 = base_rotation[2], half = _target_size_mm/2000
  2. For each sample's *_pose.json:
     - delta_rz = rotation_euler[2] - rz0
     - cos_rz, sin_rz = cos(delta_rz), sin(delta_rz)
     - x_norm = (cos_rz * sample_x - sin_rz * sample_y) / half
     - y_norm = (sin_rz * sample_x + cos_rz * sample_y) / half
     - GT pose = [cos_rz, sin_rz, x_norm, y_norm]

This is identical to SimVisuoTactileDataset._load_pose in
  VisTacFusion/vistacfusion/data/dataset.py
"""
import json, math, os, argparse


def get_rz0_half(root, obj_name):
    s0 = os.path.join(root, obj_name, "session_000", "session.json")
    with open(s0) as f:
        d = json.load(f)
    rz0 = d["base_rotation"][2]
    half = d.get("_target_size_mm", 82.0) / 2.0 / 1000.0
    return rz0, half


def load_pose(pose_path, rz0, half):
    with open(pose_path) as f:
        data = json.load(f)
    delta_rz = data["rotation_euler"][2] - rz0
    cos_rz = math.cos(delta_rz)
    sin_rz = math.sin(delta_rz)
    sx, sy = data["sample_x"], data["sample_y"]
    x_norm = (cos_rz * sx - sin_rz * sy) / max(half, 1e-8)
    y_norm = (sin_rz * sx + cos_rz * sy) / max(half, 1e-8)
    return [cos_rz, sin_rz, x_norm, y_norm]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--session", default=None, help="e.g. session_000 (default: all)")
    parser.add_argument("--sample", type=int, default=None, help="e.g. 33 (default: all)")
    args = parser.parse_args()

    rz0, half = get_rz0_half(args.root, args.obj)
    print(f"obj={args.obj}  rz0={rz0:.6f}  half={half:.6f}")

    obj_dir = os.path.join(args.root, args.obj)
    sessions = sorted(s for s in os.listdir(obj_dir) if s.startswith("session_"))
    if args.session:
        sessions = [args.session]

    for sess in sessions:
        raw = os.path.join(obj_dir, sess, "sensor_0000", "raw_data")
        if not os.path.isdir(raw):
            continue
        files = sorted(f for f in os.listdir(raw) if f.endswith("_pose.json") and "_gt" not in f)
        for f in files:
            idx = int(f.replace("_pose.json", ""))
            if args.sample is not None and idx != args.sample:
                continue
            pose = load_pose(os.path.join(raw, f), rz0, half)
            print(f"  {sess}/{idx:04d}  GT=[{pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}, {pose[3]:.6f}]")
