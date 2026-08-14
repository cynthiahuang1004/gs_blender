"""
extend_sessions.py
==================
Extend existing renders_v3 sessions from 200 to 800 valid_cells (append 600
new cells, indices 0200-0799). Existing cells / rendered samples untouched.

Sampling logic is copied from prepare_session.py (ridge-based random sampling,
same constants as the v3 sessions: random-200). Session-specific parameters
(base_rotation, fixed_scale, depth range, sensor length) are read from each
session.json, so appended cells are consistent with the original ones.

Original session.json is backed up as session.json.bak200 (never overwritten
if it already exists).

Usage:
    python extend_sessions.py            # extend the 12x3 target sessions
    python extend_sessions.py --dry-run  # report only, no writes
"""
import os, json, math, random, argparse, hashlib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MESH_DIR = os.path.join(ROOT_DIR, 'meshes')
RENDER_DIR = os.path.join(ROOT_DIR, 'renders_v3')

N_NEW = 600           # cells to append per session
TARGET_TOTAL = 800

# ── Same constants as prepare_session.py (v3 generation) ──
AUTO_CLIP_BAND_MM = 7.0
RIDGE_JITTER_MM = 1.0
USE_SURFACE_SAMPLES = True
N_SURFACE_SAMPLES = 200000
MIN_VERTS_IN_CELL = 1
MIN_CONTACT_FRAC = 0.05
FEATURE_Z_RANGE_MIN_MM = 1.0
MIN_RIDGE_FRAC_IN_CELL = 0.15

TARGETS = {
    'pattern_01_2_lines_angle_1_2': ['session_000', 'session_001', 'session_011'],
    'pattern_01_2_lines_angle_2':   ['session_002', 'session_003', 'session_004'],
    'pattern_01_2_lines_angle_3':   ['session_000', 'session_001', 'session_011'],
    'pattern_04_3_lines_angle_1':   ['session_000', 'session_001', 'session_011'],
    'pattern_04_3_lines_angle_2':   ['session_008', 'session_009', 'session_010'],
    'pattern_06_5_lines_angle_1':   ['session_000', 'session_001', 'session_011'],
    'pattern_31_rod':               ['session_002', 'session_003', 'session_004'],
    'pattern_32':                   ['session_002', 'session_003', 'session_004'],
    'pattern_33':                   ['session_002', 'session_003', 'session_004'],
    'pattern_35':                   ['session_002', 'session_003', 'session_004'],
    'pattern_36':                   ['session_002', 'session_003', 'session_004'],
    'pattern_37':                   ['session_002', 'session_003', 'session_004'],
}


def gen_cells(mesh, base_rotation, fixed_scale, length_mean,
              depth_min, depth_max, n_cells, start_idx, seed):
    """Ridge-based random cell sampling (prepare_session.py logic)."""
    rng = random.Random(seed)
    np.random.seed(seed & 0x7fffffff)

    sensor_width = 2 * length_mean
    R = Rotation.from_euler('xyz', base_rotation)

    if USE_SURFACE_SAMPLES:
        surf_raw, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_SAMPLES)
        sample_pts = R.apply(surf_raw) / fixed_scale
    else:
        sample_pts = R.apply(mesh.vertices) / fixed_scale

    # AUTO_CLIP: feature region near lowest z
    z_min = sample_pts[:, 2].min()
    feat_mask = sample_pts[:, 2] < z_min + AUTO_CLIP_BAND_MM / 1000.0
    if feat_mask.sum() < 5:
        raise RuntimeError('too few feature points')
    fv = sample_pts[feat_mask]
    ridge_xy = fv[:, :2]

    proj_xy = sample_pts[:, :2]
    ridge_zone_thresh = float(z_min) + AUTO_CLIP_BAND_MM / 1000.0
    jitter = RIDGE_JITTER_MM / 1000.0

    def make_cell(cx, cy, idx):
        in_cell = (
            (proj_xy[:, 0] >= cx - sensor_width/2) &
            (proj_xy[:, 0] <= cx + sensor_width/2) &
            (proj_xy[:, 1] >= cy - sensor_width/2) &
            (proj_xy[:, 1] <= cy + sensor_width/2)
        )
        n_in = int(in_cell.sum())
        if n_in < MIN_VERTS_IN_CELL:
            return None
        z_in = sample_pts[in_cell, 2]
        if float(z_in.max() - z_in.min()) < FEATURE_Z_RANGE_MIN_MM / 1000.0:
            return None
        ridge_frac = int((z_in < ridge_zone_thresh).sum()) / n_in
        if ridge_frac < MIN_RIDGE_FRAC_IN_CELL:
            return None
        lowest = float(z_in.min())
        contact_frac = int(((z_in - lowest) < depth_max).sum()) / n_in
        if contact_frac < MIN_CONTACT_FRAC:
            return None
        return {
            'gx': idx, 'gy': 0,
            'cx': round(cx, 6), 'cy': round(cy, 6),
            'rx': round(float(base_rotation[0]), 6),
            'ry': round(float(base_rotation[1]), 6),
            'rz': round(float(base_rotation[2]), 6),
            'depth_min': depth_min, 'depth_max': depth_max,
            'contact_frac': round(contact_frac, 3),
        }

    cells = []
    attempts = 0
    max_attempts = n_cells * 40
    while len(cells) < n_cells and attempts < max_attempts:
        attempts += 1
        rv = ridge_xy[rng.randrange(len(ridge_xy))]
        cx = float(rv[0]) + rng.uniform(-jitter, jitter)
        cy = float(rv[1]) + rng.uniform(-jitter, jitter)
        cell = make_cell(cx, cy, start_idx + len(cells))
        if cell is not None:
            cells.append(cell)
    return cells, attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    mesh_cache = {}
    for obj, sessions in TARGETS.items():
        if obj not in mesh_cache:
            mesh_cache[obj] = trimesh.load(
                os.path.join(MESH_DIR, f'{obj}.obj'), force='mesh')
        mesh = mesh_cache[obj]

        for sess in sessions:
            spath = os.path.join(RENDER_DIR, obj, sess, 'session.json')
            with open(spath) as f:
                session = json.load(f)

            cur = len(session['valid_cells'])
            if cur >= TARGET_TOTAL:
                print(f'{obj}/{sess}: already {cur} cells, skip')
                continue
            n_new = TARGET_TOTAL - cur

            length_mean = session['_sensor_length_mm'] / 1000.0
            seed = int(hashlib.md5(f'{obj}/{sess}/extend'.encode()).hexdigest()[:8], 16)

            cells, attempts = gen_cells(
                mesh, session['base_rotation'], session['fixed_scale'],
                length_mean, session['OBJ_DEPTH_MIN'], session['OBJ_DEPTH_MAX'],
                n_new, start_idx=cur, seed=seed)

            status = 'OK' if len(cells) == n_new else f'ONLY {len(cells)}/{n_new}'
            print(f'{obj}/{sess}: {cur} -> {cur + len(cells)} cells '
                  f'({attempts} attempts) [{status}]')

            if args.dry_run:
                continue

            bak = spath + '.bak200'
            if not os.path.exists(bak):
                with open(bak, 'w') as f:
                    json.dump(session, f, indent=2)

            session['valid_cells'].extend(cells)
            session['NUM_OBJ_SAMPLES'] = len(session['valid_cells'])
            session['_grid'] = f"random-{len(session['valid_cells'])} (ext+{len(cells)})"
            with open(spath, 'w') as f:
                json.dump(session, f, indent=2)

    print('\nDone.' + (' (dry run, nothing written)' if args.dry_run else ''))


if __name__ == '__main__':
    main()
