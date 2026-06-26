"""
batch_prepare_session.py

Non-interactive batch driver for prepare_session.py.
Generates sessions for a fixed list of objects, each with:
  - target size      = 82 mm
  - sensor half-len   = 5 mm   (LENGTH_MEAN)
  - overlap ratio     = 40%    (TARGET_OVERLAP)
  - base rotation     = +Z up (original) = (0, 0, 0)
  - Z rotation step   = 30°  → 12 sessions per object (0°,30°,...,330°)
  - NUM_RANDOM_SAMPLES per session (set below)

Output structure: renders/<obj_name>/session_xxx/
Reuses prepare_session.py's functions, so the produced session.json is
identical to what the interactive tool would write.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')  # headless: no display needed

import math
import numpy as np
from scipy.spatial.transform import Rotation

import prepare_session as ps

# ── config for this batch ───────────────────────────────────────
NUM_RANDOM_SAMPLES = 200
Z_STEP_DEG         = 30.0
BASE_NAME          = '+X up'
BASE_EULER         = (0.0, math.pi / 2, 0.0)   # +X up = (0°, 90°, 0°)

# 每個物體各自的 target size（mm）
TARGET_SIZE_MM = {
    'button':    82.0,
    'edge':      150.0,
    'hex_key':   150.0,
    'marble':    120.0,
    'ping_pong': 50.0,
}
OBJECTS = list(TARGET_SIZE_MM.keys())
# ────────────────────────────────────────────────────────────────

# push batch config into the reused module's globals
ps.NUM_RANDOM_SAMPLES = NUM_RANDOM_SAMPLES

n_rot = max(1, int(round(360.0 / Z_STEP_DEG)))
z_angles_deg = [i * Z_STEP_DEG for i in range(n_rot)]

summary = []  # (obj, n_sessions, [valid_cell_counts])

for obj_name in OBJECTS:
    mesh_path = os.path.join(ps.MESH_DIR, obj_name + '.obj')
    if not os.path.exists(mesh_path):
        print(f'⚠️  SKIP {obj_name}: mesh 不存在 ({mesh_path})')
        continue

    target_size = TARGET_SIZE_MM[obj_name] / 1000.0

    print('\n' + '=' * 60)
    print(f'OBJECT: {obj_name}  (target {TARGET_SIZE_MM[obj_name]:.0f}mm, rotation {BASE_NAME})')
    print('=' * 60)

    mesh = ps.trimesh.load(mesh_path, force='mesh')

    # fixed_scale: shared across sessions, from top-ranked rotation's max dim
    # (replicates prepare_session.py interactive scale logic)
    results_init = ps.try_candidate_rotations(mesh)
    R_top = Rotation.from_euler('xyz', results_init[0][2])
    verts_rotated = R_top.apply(mesh.vertices)
    max_dim = (verts_rotated.max(axis=0) - verts_rotated.min(axis=0)).max()
    fixed_scale = max_dim / target_size
    print(f'  max_dim={max_dim*1000:.1f}mm  fixed_scale={fixed_scale:.4f}')

    # session index continues from any existing sessions for this object
    obj_render_dir = os.path.join(ps.RENDER_DIR, obj_name)
    os.makedirs(obj_render_dir, exist_ok=True)
    existing = [d for d in os.listdir(obj_render_dir) if d.startswith('session_')]
    session_idx = len(existing)
    if existing:
        print(f'  已存在 {len(existing)} 個 session，從 session_{session_idx:03d} 接續')

    counts = []
    for z_deg in z_angles_deg:
        chosen_euler = (BASE_EULER[0], BASE_EULER[1],
                        BASE_EULER[2] + np.radians(z_deg))
        chosen_name = BASE_NAME + (f' + Z{z_deg:+.0f}°' if z_deg != 0 else '')

        params = ps.compute_sampling_params(
            mesh, chosen_euler, fixed_scale,
            target_overlap=ps.TARGET_OVERLAP,
            length_mean=ps.LENGTH_MEAN,
        )
        ps.write_session(params, chosen_euler, chosen_name, obj_name,
                         fixed_scale, target_size, session_idx)
        counts.append(params['NUM_OBJ_SAMPLES'])
        session_idx += 1

    summary.append((obj_name, len(counts), counts))

# ── final summary ───────────────────────────────────────────────
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
grand_total = 0
short_objs = []
for obj_name, n_sess, counts in summary:
    total = sum(counts)
    grand_total += total
    cmin, cmax = (min(counts), max(counts)) if counts else (0, 0)
    flag = '' if cmin >= NUM_RANDOM_SAMPLES else '  ⚠️ 有 session 湊不滿'
    if cmin < NUM_RANDOM_SAMPLES:
        short_objs.append((obj_name, cmin, cmax))
    print(f'  {obj_name:<32} {n_sess:>2} sessions  '
          f'cells/session [{cmin}-{cmax}]  total={total}{flag}')

print('-' * 60)
print(f'物體數: {len(summary)}   總 sample 數(= 之後 NUM_SENSORS=1 的張數): {grand_total}')
if short_objs:
    print('\n⚠️  以下物體有 session 湊不滿 '
          f'{NUM_RANDOM_SAMPLES}（特徵不足，屬正常，可日後補）:')
    for obj_name, cmin, cmax in short_objs:
        print(f'    {obj_name}: 最少 {cmin} / session')
print('\n下一步: python run_blender.py')
