import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pathlib import Path

MESH_DIR = Path("meshes")  # obj mesh files
RENDER_DIR = Path("renders")  # session directories


def load_object_info(obj_name):
    """One-time per object: load mesh, get session_000 info."""
    mesh = trimesh.load(str(MESH_DIR / f"{obj_name}.obj"), force="mesh")

    with open(RENDER_DIR / obj_name / "session_000" / "session.json") as f:
        d0 = json.load(f)

    fixed_scale = d0["fixed_scale"]
    target_size = d0.get("_target_size_mm", 82.0)
    half = target_size / 2 / 1000
    rz0 = d0["base_rotation"][2]

    return mesh, fixed_scale, half, rz0, d0["base_rotation"]


def get_session_center(mesh, fixed_scale, base_rotation):
    """Projected bbox center for a specific session."""
    R_3d = Rotation.from_euler("xyz", base_rotation)
    v = R_3d.apply(mesh.vertices) / fixed_scale
    cx = (v[:, 0].min() + v[:, 0].max()) / 2
    cy = (v[:, 1].min() + v[:, 1].max()) / 2
    return np.array([cx, cy])


def compute_pose(cx, cy, session_center, delta_rz, half):
    """
    Compute normalized pose label for one sample.

    Args:
        cx, cy:          raw values from session.json valid_cells
        session_center:  projected bbox center for this session [2]
        delta_rz:        session base_rotation[2] - session_000 base_rotation[2]
        half:            target_size_mm / 2 / 1000

    Returns:
        [x_norm, y_norm, sin(theta), cos(theta)]   all in [-1, 1]
    """
    R = np.array([[np.cos(delta_rz), -np.sin(delta_rz)],
                  [np.sin(delta_rz),  np.cos(delta_rz)]])

    offset = np.array([cx - session_center[0], cy - session_center[1]])
    pt_obj = R.T @ offset           # rotate to object frame
    x_norm = -pt_obj[0] / half      # negate x, normalize
    y_norm =  pt_obj[1] / half      # normalize
    return [x_norm, y_norm, np.sin(delta_rz), np.cos(delta_rz)]


# === Usage ===
# obj_name = "peg3"
# mesh, fixed_scale, half, rz0, rot0 = load_object_info(obj_name)
#
# for session in sessions:
#     with open(session / "session.json") as f:
#         data = json.load(f)
#     delta_rz = data["base_rotation"][2] - rz0
#     sc = get_session_center(mesh, fixed_scale, data["base_rotation"])
#
#     for cell in data["valid_cells"]:
#         label = compute_pose(cell["cx"], cell["cy"], sc, delta_rz, half)
