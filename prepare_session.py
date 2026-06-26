"""
prepare_session.py

Grid-based sampling strategy with multi-session support.
Each session uses a different base rotation (object orientation),
allowing coverage of different faces of the object.
"""

import os, json, math, random
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR   = os.path.join(SCRIPT_DIR, 'meshes')
RENDER_DIR = os.path.join(SCRIPT_DIR, 'renders')

OBJ_SIZE_MIN   = 0.082
OBJ_SIZE_MAX   = 0.082
TARGET_OVERLAP = 0.40
LENGTH_MEAN    = 0.005
OBJ_DEPTH_MIN  = 0.0010
OBJ_DEPTH_MAX  = 0.0010
USE_NORMAL_ROTATION = False

CLIP_X_MM = 45.0    # 手動 clip；AUTO_CLIP=True 時忽略
CLIP_Y_MM = 20.0
AUTO_CLIP = True            # True: 從 mesh 自動偵測 feature 區
AUTO_CLIP_BAND_MM = 7.0     # 距最低 z 多少 mm 內視為「特徵」
AUTO_CLIP_MARGIN_MM = 1.0   # feature bbox 外再加多少 margin
SAMPLE_ON_RIDGE = True      # True: 隨機抽 cx,cy 時直接從 ridge vertex 取
                            #       (才不會浪費在 ridge 之間的空白)
RIDGE_JITTER_MM = 1.0       # ridge vertex 周圍多少 mm 的擾動

# 用 mesh 表面均勻撒點代替 vertex (解決 tube 主體 tessellation 太稀的問題)
USE_SURFACE_SAMPLES = True
N_SURFACE_SAMPLES = 200000  # 在 mesh 表面均勻撒幾個點

SAMPLING_METHOD = 'random'        # 'grid' or 'random'
NUM_RANDOM_SAMPLES = 100
MIN_VERTS_IN_CELL = 1
MIN_CONTACT_FRAC = 0.05           # tactile contact region must cover ≥ 10% of FOV
FEATURE_Z_RANGE_MIN_MM = 1.0     # cell 內 z 變化 < 此值 (= 平面) 視為背景,丟棄
MIN_RIDGE_FRAC_IN_CELL  = 0.15   # cell 內「位於全域 ridge zone 的點」占比下限


# ══════════════════════════════════════════════════════════════
# Rotation selection
# ══════════════════════════════════════════════════════════════

def compute_best_rotation(mesh):
    verts = mesh.vertices
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)
    thin_axis = eigenvectors[:, idx[0]]
    z_target = np.array([0, 0, 1.0])
    dot = np.dot(thin_axis, z_target)
    if abs(dot) > 0.999:
        R = Rotation.from_euler('x', np.pi) if dot < 0 else Rotation.identity()
    else:
        axis = np.cross(thin_axis, z_target)
        axis /= np.linalg.norm(axis)
        R = Rotation.from_rotvec(axis * np.arccos(np.clip(dot, -1, 1)))
    return tuple(R.as_euler('xyz')), eigenvalues[idx]


def estimate_contact_area(mesh, rotation_euler):
    R = Rotation.from_euler('xyz', rotation_euler)
    verts_rotated = R.apply(mesh.vertices)
    xy = verts_rotated[:, :2]
    try:
        return ConvexHull(xy).volume
    except Exception:
        return 0.0


def try_candidate_rotations(mesh):
    candidates = [('PCA auto', compute_best_rotation(mesh)[0])]
    for name, euler in [
        ('+Z up (original)', (0, 0, 0)),
        ('-Z up (flipped)',   (math.pi, 0, 0)),
        ('+X up',             (0, math.pi/2, 0)),
        ('-X up',             (0, -math.pi/2, 0)),
        ('+Y up',             (-math.pi/2, 0, 0)),
        ('-Y up',             (math.pi/2, 0, 0)),
    ]:
        candidates.append((name, euler))
    results = [(estimate_contact_area(mesh, e), n, e) for n, e in candidates]
    results.sort(reverse=True)
    return results


def get_rotated_xy_bounds(mesh, rotation_euler, scale):
    R = Rotation.from_euler('xyz', rotation_euler)
    verts = R.apply(mesh.vertices) / scale
    return verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()


# ══════════════════════════════════════════════════════════════
# Per-cell normal rotation
# ══════════════════════════════════════════════════════════════

def compute_cell_rotation(mesh, rotation_euler, fixed_scale, cx, cy,
                           sensor_width, nearby_mask, max_tilt_deg=15.0):
    R_base = Rotation.from_euler('xyz', rotation_euler)
    normals_rot = R_base.apply(mesh.vertex_normals)

    if nearby_mask.sum() == 0:
        return rotation_euler

    nearby_normals = normals_rot[nearby_mask]
    weights = np.clip(nearby_normals[:, 2], 0, None)
    if weights.sum() < 1e-8:
        weights = np.ones(len(nearby_normals))
    avg_normal = (nearby_normals * weights[:, None]).sum(axis=0)
    avg_normal /= np.linalg.norm(avg_normal)

    # 如果平均法向量朝下（Z < 0），直接用 base rotation
    if avg_normal[2] < 0:
        return rotation_euler

    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(avg_normal, z_axis))
    if abs(dot) > 0.9999:
        return rotation_euler

    axis = np.cross(avg_normal, z_axis)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    angle = min(angle, np.radians(max_tilt_deg))

    extra_R = Rotation.from_rotvec(axis * angle)
    final_R = extra_R * R_base
    
    # 確認最終旋轉不會讓物體翻轉太多
    final_euler = final_R.as_euler('xyz')
    if abs(np.degrees(final_euler[0])) > 45 or abs(np.degrees(final_euler[1])) > 45:
        return rotation_euler  # 退回 base rotation
        
    return tuple(final_euler)

# ══════════════════════════════════════════════════════════════
# Per-cell depth
# ══════════════════════════════════════════════════════════════

def compute_cell_depth_range(verts_proj, cx, cy, sensor_width,
                              global_depth_min, global_depth_max):
    in_cell = (
        (verts_proj[:, 0] >= cx - sensor_width / 2) &
        (verts_proj[:, 0] <= cx + sensor_width / 2) &
        (verts_proj[:, 1] >= cy - sensor_width / 2) &
        (verts_proj[:, 1] <= cy + sensor_width / 2)
    )
    if in_cell.sum() == 0:
        return global_depth_min, global_depth_max

    z_max = verts_proj[in_cell, 2].max()
    depth_min = max(global_depth_min, z_max * 1.05)
    depth_max = min(global_depth_max, z_max * 1.50)

    if depth_min >= depth_max:
        depth_min = global_depth_max * 0.8
        depth_max = global_depth_max

    return round(float(depth_min), 7), round(float(depth_max), 7)


# ══════════════════════════════════════════════════════════════
# Grid sampling
# ══════════════════════════════════════════════════════════════

def compute_sampling_params(mesh, rotation_euler, fixed_scale,
                             target_overlap=0.40, length_mean=0.010):
    sensor_width = 2 * length_mean
    step = sensor_width * (1 - target_overlap)

    R = Rotation.from_euler('xyz', rotation_euler)
    verts_proj = R.apply(mesh.vertices) / fixed_scale

    # 用 mesh 表面均勻撒點當作 sampling 用的「特徵點池」
    # 解決原本只用 vertex 時 tube 主體稀疏的問題
    if USE_SURFACE_SAMPLES:
        surf_raw, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_SAMPLES)
        sample_pts = R.apply(surf_raw) / fixed_scale
        print(f'  [SURFACE_SAMPLE] 在 mesh 表面均勻撒 {len(sample_pts)} 個點')
    else:
        sample_pts = verts_proj

    obj_xmin, obj_xmax = sample_pts[:, 0].min(), sample_pts[:, 0].max()
    obj_ymin, obj_ymax = sample_pts[:, 1].min(), sample_pts[:, 1].max()
    obj_cx = (obj_xmin + obj_xmax) / 2
    obj_cy = (obj_ymin + obj_ymax) / 2

    # 自動偵測 feature 區（最低 z 附近的 surface point 的 XY 包圍盒 + ridge 池）
    clip_x_mm_use = CLIP_X_MM
    clip_y_mm_use = CLIP_Y_MM
    ridge_xy = None
    if AUTO_CLIP:
        z_min = sample_pts[:, 2].min()
        feat_mask = sample_pts[:, 2] < z_min + AUTO_CLIP_BAND_MM / 1000.0
        if feat_mask.sum() >= 5:
            fv = sample_pts[feat_mask]
            ridge_xy = fv[:, :2]  # 留下來供 sample 用
            fx_min, fx_max = fv[:, 0].min(), fv[:, 0].max()
            fy_min, fy_max = fv[:, 1].min(), fv[:, 1].max()
            obj_cx = (fx_min + fx_max) / 2
            obj_cy = (fy_min + fy_max) / 2
            clip_x_mm_use = (fx_max - fx_min) * 1000 + 2 * AUTO_CLIP_MARGIN_MM
            clip_y_mm_use = (fy_max - fy_min) * 1000 + 2 * AUTO_CLIP_MARGIN_MM
            print(f'  [AUTO_CLIP] feature center=({obj_cx*1000:.1f}, {obj_cy*1000:.1f})mm, '
                  f'clip={clip_x_mm_use:.1f} × {clip_y_mm_use:.1f}mm '
                  f'({feat_mask.sum()} feature points in {AUTO_CLIP_BAND_MM}mm band)')

    if clip_x_mm_use is not None:
        half_clip_x = clip_x_mm_use / 1000.0 / 2
        obj_xmin = max(obj_xmin, obj_cx - half_clip_x)
        obj_xmax = min(obj_xmax, obj_cx + half_clip_x)
    if clip_y_mm_use is not None:
        half_clip_y = clip_y_mm_use / 1000.0 / 2
        obj_ymin = max(obj_ymin, obj_cy - half_clip_y)
        obj_ymax = min(obj_ymax, obj_cy + half_clip_y)

    x_width = obj_xmax - obj_xmin
    y_width = obj_ymax - obj_ymin

    grid_nx = max(2, math.ceil(x_width / step) + 1)
    grid_ny = max(2, math.ceil(y_width / step) + 1)
    half_x = grid_nx * step / 2
    half_y = grid_ny * step / 2
    margin = length_mean * 0.3

    x_min = obj_cx - half_x - margin
    x_max = obj_cx + half_x + margin
    y_min = obj_cy - half_y - margin
    y_max = obj_cy + half_y + margin

    proj_xy = sample_pts[:, :2]

    # 全域 ridge zone 上限 = z_min_global + AUTO_CLIP_BAND_MM
    # 用來判斷 cell 是否真的「壓在 ridge 上」
    z_min_global = float(sample_pts[:, 2].min())
    ridge_zone_thresh = z_min_global + AUTO_CLIP_BAND_MM / 1000.0

    clip_x_half = (clip_x_mm_use / 1000.0 / 2) if clip_x_mm_use is not None else float('inf')
    clip_y_half = (clip_y_mm_use / 1000.0 / 2) if clip_y_mm_use is not None else float('inf')

    def make_cell(cx, cy, idx_x, idx_y):
        in_cell = (
            (proj_xy[:, 0] >= cx - sensor_width/2) &
            (proj_xy[:, 0] <= cx + sensor_width/2) &
            (proj_xy[:, 1] >= cy - sensor_width/2) &
            (proj_xy[:, 1] <= cy + sensor_width/2)
        )
        n_in_cell = int(in_cell.sum())
        if n_in_cell < MIN_VERTS_IN_CELL:
            return None

        z_in_cell = sample_pts[in_cell, 2]
        lowest_z = float(z_in_cell.min())

        # 過濾「平面 cell」: cell 內 z 變化太小 = plate 平面, 不是 feature
        z_range = float(z_in_cell.max() - z_in_cell.min())
        if z_range < FEATURE_Z_RANGE_MIN_MM / 1000.0:
            return None

        # 過濾「離 ridge 太遠的 cell」: cell 內必須有夠多點在全域 ridge zone 內
        ridge_count = int((z_in_cell < ridge_zone_thresh).sum())
        ridge_frac = ridge_count / n_in_cell
        if ridge_frac < MIN_RIDGE_FRAC_IN_CELL:
            return None

        press_depth = OBJ_DEPTH_MAX
        contact_count = int(((z_in_cell - lowest_z) < press_depth).sum())
        contact_frac = contact_count / n_in_cell
        if contact_frac < MIN_CONTACT_FRAC:
            return None

        if USE_NORMAL_ROTATION:
            cell_euler = compute_cell_rotation(
                mesh, rotation_euler, fixed_scale,
                cx, cy, sensor_width, in_cell)
        else:
            cell_euler = rotation_euler
        d_min, d_max = OBJ_DEPTH_MIN, OBJ_DEPTH_MAX
        return {
            'gx': idx_x,
            'gy': idx_y,
            'cx': round(cx, 6),
            'cy': round(cy, 6),
            'rx': round(float(cell_euler[0]), 6),
            'ry': round(float(cell_euler[1]), 6),
            'rz': round(float(cell_euler[2]), 6),
            'depth_min': d_min,
            'depth_max': d_max,
            'contact_frac': round(contact_frac, 3),
        }

    valid_cells = []
    if SAMPLING_METHOD == 'random':
        use_ridge = SAMPLE_ON_RIDGE and ridge_xy is not None and len(ridge_xy) > 0
        if use_ridge:
            jitter = RIDGE_JITTER_MM / 1000.0
            print(f'  [SAMPLE_ON_RIDGE] 從 {len(ridge_xy)} 個 ridge 點抽樣 '
                  f'(±{RIDGE_JITTER_MM}mm jitter)')
        rand_xmin = obj_cx - clip_x_half
        rand_xmax = obj_cx + clip_x_half
        rand_ymin = obj_cy - clip_y_half
        rand_ymax = obj_cy + clip_y_half
        max_attempts = NUM_RANDOM_SAMPLES * 20
        attempts = 0
        while len(valid_cells) < NUM_RANDOM_SAMPLES and attempts < max_attempts:
            attempts += 1
            if use_ridge:
                # 從 ridge vertex 隨機抽,加小擾動
                rv = ridge_xy[random.randrange(len(ridge_xy))]
                cx = rv[0] + random.uniform(-jitter, jitter)
                cy = rv[1] + random.uniform(-jitter, jitter)
            else:
                cx = random.uniform(rand_xmin, rand_xmax)
                cy = random.uniform(rand_ymin, rand_ymax)
            cell = make_cell(cx, cy, len(valid_cells), 0)
            if cell is not None:
                valid_cells.append(cell)
        if len(valid_cells) < NUM_RANDOM_SAMPLES:
            print(f'⚠️  只取得 {len(valid_cells)}/{NUM_RANDOM_SAMPLES} '
                  f'個有效 sample (試了 {attempts} 次)')
    else:
        for gy in range(grid_ny):
            for gx in range(grid_nx):
                cx = x_min + step/2 + gx * step
                cy = y_min + step/2 + gy * step
                if abs(cx - obj_cx) > clip_x_half or abs(cy - obj_cy) > clip_y_half:
                    continue
                cell = make_cell(cx, cy, gx, gy)
                if cell is not None:
                    valid_cells.append(cell)

    actual_overlap = 1 - step / sensor_width

    if SAMPLING_METHOD == 'random':
        x_min, x_max = obj_cx - clip_x_half, obj_cx + clip_x_half
        y_min, y_max = obj_cy - clip_y_half, obj_cy + clip_y_half

    return {
        'X_MIN': round(x_min, 5),
        'X_MAX': round(x_max, 5),
        'Y_MIN': round(y_min, 5),
        'Y_MAX': round(y_max, 5),
        'NUM_OBJ_SAMPLES': len(valid_cells),
        'valid_cells': valid_cells,
        'grid_nx': grid_nx,
        'grid_ny': grid_ny,
        'step_mm': round(step * 1000, 2),
        'step': step,
        'sensor_width_mm': round(sensor_width * 1000, 2),
        'sensor_width': sensor_width,
        'actual_overlap_pct': round(actual_overlap * 100, 1),
        'obj_xy_mm': (round(x_width*1000, 1), round(y_width*1000, 1)),
        'verts_proj': verts_proj,
        'z_anchor': float(sample_pts[:, 2].min()),
    }


# ══════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════

def visualize_sampling(mesh, base_rotation, fixed_scale, params,
                       length_mean, obj_name, session_idx):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{obj_name} — Session {session_idx:03d} Sampling Preview', fontsize=13)

    R = Rotation.from_euler('xyz', base_rotation)
    verts_rot = R.apply(mesh.vertices) / fixed_scale
    px = verts_rot[:, 0] * 1000
    py = verts_rot[:, 1] * 1000
    pz = verts_rot[:, 2] * 1000
    n_pts = len(px)
    step_idx = max(1, n_pts // 5000)

    sensor_half = params['sensor_width'] / 2 * 1000
    step = params['step']
    valid_set = {(c['gx'], c['gy']) for c in params['valid_cells']}

    # Left: top view
    ax = axes[0]
    ax.set_title('Top view + sensor grid', fontsize=10)
    ax.set_aspect('equal')
    ax.scatter(px[::step_idx], py[::step_idx], s=0.5, c='steelblue', alpha=0.3,
               label='Object surface')

    if SAMPLING_METHOD == 'random':
        for cell in params['valid_cells']:
            cx_mm = cell['cx'] * 1000
            cy_mm = cell['cy'] * 1000
            rect = patches.Rectangle(
                (cx_mm - sensor_half, cy_mm - sensor_half),
                sensor_half*2, sensor_half*2,
                linewidth=0.4, edgecolor='tomato', facecolor='tomato', alpha=0.05
            )
            ax.add_patch(rect)
            ax.plot(cx_mm, cy_mm, '.', color='tomato', markersize=2)
    else:
        for gy in range(params['grid_ny']):
            for gx in range(params['grid_nx']):
                cx = (params['X_MIN'] + step/2 + gx*step) * 1000
                cy = (params['Y_MIN'] + step/2 + gy*step) * 1000
                is_valid = (gx, gy) in valid_set
                color = 'tomato' if is_valid else 'gray'
                alpha = 0.15 if is_valid else 0.03
                rect = patches.Rectangle(
                    (cx - sensor_half, cy - sensor_half),
                    sensor_half*2, sensor_half*2,
                    linewidth=0.6, edgecolor=color, facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)
                if is_valid:
                    ax.plot(cx, cy, '+', color='tomato', markersize=4, markeredgewidth=0.8)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: side view
    ax2 = axes[1]
    ax2.set_title('Side view (XZ) — depth ranges', fontsize=10)
    ax2.set_aspect('equal')
    ax2.scatter(px[::step_idx], pz[::step_idx], s=0.5, c='steelblue', alpha=0.3)

    n_bins = 80
    bin_edges = np.linspace(px.min(), px.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    z_max_per_bin, z_min_per_bin = [], []
    for i in range(n_bins):
        in_bin = (px >= bin_edges[i]) & (px < bin_edges[i+1])
        if in_bin.sum() > 0:
            z_max_per_bin.append(pz[in_bin].max())
            z_min_per_bin.append(pz[in_bin].min())
        else:
            z_max_per_bin.append(np.nan)
            z_min_per_bin.append(np.nan)
    ax2.plot(bin_centers, z_max_per_bin, color='tomato', linewidth=1.5, label='Z max')
    ax2.plot(bin_centers, z_min_per_bin, color='navy', linewidth=1.5, label='Z min')

    for cell in params['valid_cells']:
        cx_mm = cell['cx'] * 1000
        ax2.plot([cx_mm - sensor_half, cx_mm + sensor_half],
                 [cell['depth_min']*1000]*2,
                 color='tomato', lw=0.8, alpha=0.4, linestyle='--')
        ax2.plot([cx_mm - sensor_half, cx_mm + sensor_half],
                 [cell['depth_max']*1000]*2,
                 color='darkorange', lw=0.8, alpha=0.4, linestyle='--')

    ax2.axhline(y=OBJ_DEPTH_MIN*1000, color='tomato', linestyle=':', lw=1, alpha=0.4,
                label=f'depth min {OBJ_DEPTH_MIN*1000:.1f}mm')
    ax2.axhline(y=OBJ_DEPTH_MAX*1000, color='darkorange', linestyle=':', lw=1, alpha=0.4,
                label=f'depth max {OBJ_DEPTH_MAX*1000:.1f}mm')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# ══════════════════════════════════════════════════════════════
# Write session
# ══════════════════════════════════════════════════════════════

def write_session(params, chosen_euler, chosen_name, obj_name,
                  fixed_scale, target_size, session_idx):
    session_dir = os.path.join(RENDER_DIR, f'session_{session_idx:03d}')
    os.makedirs(session_dir, exist_ok=True)

    session = {
        'obj':               obj_name,
        'base_rotation':     list(chosen_euler),
        'rotation_step':     0.0,
        'fixed_scale':       fixed_scale,
        'X_MIN':             params['X_MIN'],
        'X_MAX':             params['X_MAX'],
        'Y_MIN':             params['Y_MIN'],
        'Y_MAX':             params['Y_MAX'],
        'NUM_OBJ_SAMPLES':   params['NUM_OBJ_SAMPLES'],
        'OBJ_DEPTH_MIN':     OBJ_DEPTH_MIN,
        'OBJ_DEPTH_MAX':     OBJ_DEPTH_MAX,
        'z_anchor':          params['z_anchor'],
        'valid_cells':       params['valid_cells'],
        '_rotation_name':        chosen_name,
        '_rotation_degrees':     [round(np.degrees(e), 1) for e in chosen_euler],
        '_sensor_length_mm':     LENGTH_MEAN * 1000,
        '_sensor_step':          params['step'],
        '_target_size_mm':       target_size * 1000,
        '_grid':                 (f'random-{NUM_RANDOM_SAMPLES}' if SAMPLING_METHOD == 'random'
                                  else f'{params["grid_nx"]}x{params["grid_ny"]}'),
        '_overlap_pct':          params['actual_overlap_pct'],
        '_sampling_method':      SAMPLING_METHOD,
        '_use_normal_rotation':  USE_NORMAL_ROTATION,
    }

    session_path = os.path.join(session_dir, 'session.json')
    with open(session_path, 'w') as f:
        json.dump(session, f, indent=2)

    print(f'\n✅ Written to session_{session_idx:03d}/session.json')
    print(f'   Rotation: {chosen_name}  {[round(np.degrees(e),1) for e in chosen_euler]}°')
    print(f'   Valid cells: {params["NUM_OBJ_SAMPLES"]}')


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

obj_files = sorted([f.replace('.obj','') for f in os.listdir(MESH_DIR) if f.endswith('.obj')])
print('='*55)
print('Available objects:')
for i, name in enumerate(obj_files):
    print(f'  [{i:2d}] {name}')
print('='*55)

while True:
    try:
        idx = int(input('Select object index: ').strip())
        if 0 <= idx < len(obj_files):
            obj_name = obj_files[idx]
            break
        print(f'Please enter 0 to {len(obj_files)-1}')
    except ValueError:
        print('Please enter a number')

print(f'\nSelected: {obj_name}')
mesh_path = os.path.join(MESH_DIR, obj_name + '.obj')
mesh = trimesh.load(mesh_path, force='mesh')
verts = mesh.vertices
extents = verts.max(axis=0) - verts.min(axis=0)
print(f'Vertices: {len(verts)}')
print(f'Size: X={extents[0]*1000:.1f}mm  Y={extents[1]*1000:.1f}mm  Z={extents[2]*1000:.1f}mm')

# Scale (shared across all sessions)
results_init = try_candidate_rotations(mesh)
R_pca = Rotation.from_euler('xyz', results_init[0][2])
verts_rotated = R_pca.apply(mesh.vertices)
max_dim = (verts_rotated.max(axis=0) - verts_rotated.min(axis=0)).max()

print(f'\nMax dimension after PCA rotation: {max_dim*1000:.1f}mm')
size_input = input(
    f'Target size mm (Enter for random {OBJ_SIZE_MIN*1000:.0f}~{OBJ_SIZE_MAX*1000:.0f}mm): '
).strip()
target_size = float(size_input)/1000.0 if size_input else random.uniform(OBJ_SIZE_MIN, OBJ_SIZE_MAX)
fixed_scale = max_dim / target_size
print(f'fixed_scale={fixed_scale:.4f}, scaled max dim={target_size*1000:.1f}mm')

# Sensor params (shared across all sessions)
length_input = input(
    f'\nSensor half-length (current {LENGTH_MEAN*1000:.1f}mm, Enter for default): '
).strip()
if length_input:
    LENGTH_MEAN = float(length_input) / 1000.0

overlap_input = input(
    f'Overlap ratio (current {TARGET_OVERLAP*100:.0f}%, Enter for default): '
).strip()
if overlap_input:
    TARGET_OVERLAP = float(overlap_input) / 100.0

# Find existing session index
os.makedirs(RENDER_DIR, exist_ok=True)
existing = [d for d in os.listdir(RENDER_DIR) if d.startswith('session_')]
session_idx = len(existing)

# Multi-session loop
while True:
    print(f'\n{"="*55}')
    print(f'Session {session_idx:03d}')
    print('='*55)

    # Select rotation
    results = try_candidate_rotations(mesh)
    print(f'\n  {"Rank":<4} {"Name":<22} {"Area":>10}   {"rx":>7} {"ry":>7} {"rz":>7}')
    print('  ' + '-'*65)
    for rank, (area, name, euler) in enumerate(results):
        marker = ' ← recommended' if rank == 0 else ''
        print(f'  [{rank}]  {name:<22} {area*1e6:>8.1f}mm²  '
              f'{np.degrees(euler[0]):>6.1f}° {np.degrees(euler[1]):>6.1f}°'
              f' {np.degrees(euler[2]):>6.1f}°{marker}')

    choice = input('\nSelect rotation (Enter = recommended): ').strip()
    chosen_rank = int(choice) if choice.isdigit() else 0
    _, base_name, base_euler = results[chosen_rank]
    print(f'Using: {base_name}')

    # Z 軸旋轉步進角度：輸入 30 → 自動產生 12 個 session (0°, 30°, 60°, ..., 330°)
    z_step_input = input(
        'Z rotation step in degrees (Enter = 0, single session; e.g. 30 → 12 sessions): '
    ).strip()
    batch_mode = False
    z_angles_deg = [0.0]
    if z_step_input:
        try:
            z_step = float(z_step_input)
            if z_step > 0:
                n_rot = max(1, int(round(360.0 / z_step)))
                z_angles_deg = [i * z_step for i in range(n_rot)]
                batch_mode = True
                print(f'→ Batch mode: {n_rot} sessions at Z = '
                      f'{[round(a, 1) for a in z_angles_deg]}°')
        except ValueError:
            print('Invalid input, using single session at Z=0°')

    for z_deg in z_angles_deg:
        chosen_euler = (base_euler[0], base_euler[1],
                        base_euler[2] + np.radians(z_deg))
        chosen_name = base_name + (f' + Z{z_deg:+.0f}°' if z_deg != 0 else '')

        # Sampling loop（batch 模式只在 z=0 那次互動，後續用相同參數）
        first_in_batch = (z_deg == z_angles_deg[0])
        while True:
            params = compute_sampling_params(
                mesh, chosen_euler, fixed_scale,
                target_overlap=TARGET_OVERLAP,
                length_mean=LENGTH_MEAN
            )
            n = params['NUM_OBJ_SAMPLES']

            print(f'\n── Session {session_idx:03d}  ({chosen_name}) ──')
            print(f'  Object XY:      {params["obj_xy_mm"][0]}mm × {params["obj_xy_mm"][1]}mm')
            print(f'  Sensor FOV:     {params["sensor_width_mm"]}mm diameter')
            print(f'  Valid cells:    {n}')

            # batch 模式下，非第一個 session 直接通過
            if batch_mode and not first_in_batch:
                break

            visualize_sampling(mesh, chosen_euler, fixed_scale, params,
                               LENGTH_MEAN, obj_name, session_idx)
            prompt = ('\nConfirm? [y = batch generate all / n = adjust params]: '
                      if batch_mode else
                      '\nConfirm? [y = yes / n = adjust params]: ')
            confirm = input(prompt).strip().lower()
            if confirm == 'y' or confirm == '':
                plt.close('all')
                break
            plt.close('all')

            length_input = input(
                f'Sensor half-length (current {LENGTH_MEAN*1000:.1f}mm, Enter to keep): '
            ).strip()
            if length_input:
                LENGTH_MEAN = float(length_input) / 1000.0
            overlap_input = input(
                f'Overlap ratio (current {TARGET_OVERLAP*100:.0f}%, Enter to keep): '
            ).strip()
            if overlap_input:
                TARGET_OVERLAP = float(overlap_input) / 100.0

        write_session(params, chosen_euler, chosen_name, obj_name,
                      fixed_scale, target_size, session_idx)
        session_idx += 1

    if batch_mode:
        break

    more = input('\nAdd another rotation/session? [y = yes / n = done]: ').strip().lower()
    if more != 'y':
        break

print(f'\n✅ Total {session_idx} session(s) created.')
print('Next: python run_blender.py')