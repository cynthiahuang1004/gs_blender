"""Verify GT 3D reconstruction by comparing against original mesh."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import open3d as o3d
import trimesh
import os, json
import matplotlib.cm as cm
CMAP_ERR = cm.get_cmap('viridis_r')  # 0=黃(min err), 1=深藍紫(max err)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(SCRIPT_DIR, 'meshes', 'pattern_01_2_lines_angle_1_2.obj')
RECON_PATH = os.path.join(SCRIPT_DIR, 'renders', 'gt_reconstruction.ply')

session_dirs = sorted([
    d for d in os.listdir(os.path.join(SCRIPT_DIR, 'renders'))
    if d.startswith('session_')
])
session_path = os.path.join(SCRIPT_DIR, 'renders', session_dirs[0], 'session.json')
with open(session_path) as f:
    sess = json.load(f)

fixed_scale = sess['fixed_scale']
base_rotation = sess.get('base_rotation', [0, 0, 0])
print(f'fixed_scale = {fixed_scale}')
print(f'base_rotation = {base_rotation}')
print(f'z_anchor = {sess.get("z_anchor", "N/A")}')

# Load original mesh, apply same scale + rotation as prepare_session 做的
mesh_orig = trimesh.load(MESH_PATH, force='mesh')
from scipy.spatial.transform import Rotation
R = Rotation.from_euler('xyz', base_rotation)
orig_verts = R.apply(mesh_orig.vertices) / fixed_scale  # = verts_proj

# ── 載入「所有 session 的原始 GT 點」(沿用 gt3d.py 同樣的反投影邏輯) ──
def load_all_raw_points():
    pts = []
    render_dir = os.path.join(SCRIPT_DIR, 'renders')
    sess_dirs = sorted([os.path.join(render_dir, d) for d in os.listdir(render_dir)
                        if d.startswith('session_') and
                        os.path.isdir(os.path.join(render_dir, d))])
    print(f'\n載入 {len(sess_dirs)} 個 session 的 raw GT 點...')

    for sp in sess_dirs:
        with open(os.path.join(sp, 'session.json')) as f:
            sess_local = json.load(f)
        br = sess_local.get('base_rotation', [0.0, 0.0, 0.0])
        cx_, cy_, cz_ = np.cos(br[0]), np.cos(br[1]), np.cos(br[2])
        sx_, sy_, sz_ = np.sin(br[0]), np.sin(br[1]), np.sin(br[2])
        Rx = np.array([[1,0,0],[0,cx_,-sx_],[0,sx_,cx_]])
        Ry = np.array([[cy_,0,sy_],[0,1,0],[-sy_,0,cy_]])
        Rz = np.array([[cz_,-sz_,0],[sz_,cz_,0],[0,0,1]])
        R_inv = (Rz @ Ry @ Rx).T

        session_pts = []
        for sd in sorted(os.listdir(sp)):
            if not sd.startswith('sensor_'):
                continue
            raw_dir = os.path.join(sp, sd, 'raw_data')
            if not os.path.exists(raw_dir):
                continue
            for fn in sorted(os.listdir(raw_dir)):
                if not (fn.endswith('_gt.npy') and len(fn) == 11):
                    continue
                pp = os.path.join(raw_dir, fn.replace('_gt.npy', '_pose.json'))
                if not os.path.exists(pp):
                    continue
                dmap = np.load(os.path.join(raw_dir, fn))
                with open(pp) as f:
                    pose = json.load(f)
                mask = dmap > 1e-6
                if mask.sum() == 0:
                    continue
                H, W = dmap.shape
                length = pose['camera_length']
                ox, oy, oz = pose['location']
                cam_h = abs(pose['camera_location'][2])
                xs = np.linspace(length, -length, W)
                ys = np.linspace(length, -length, H)
                xv, yv = np.meshgrid(xs, ys)
                wz = -dmap[mask]
                persp = (wz + cam_h) / cam_h
                X = xv[mask] * persp - ox
                Y = yv[mask] * persp - oy
                Z = wz - oz
                session_pts.append(np.column_stack([X, Y, Z]))
        if session_pts:
            sess_arr = np.vstack(session_pts)
            if any(abs(r) > 1e-6 for r in br):
                sess_arr = (R_inv @ sess_arr.T).T
            pts.append(sess_arr)
            print(f'  {os.path.basename(sp)}: {len(sess_arr)} points')
    return np.vstack(pts) if pts else np.empty((0, 3))

raw_pts = load_all_raw_points()
print(f'總原始 GT 點數: {len(raw_pts)}')

# Load reconstructed Poisson mesh (post-processed)
recon = o3d.io.read_triangle_mesh(RECON_PATH)
recon_verts = np.asarray(recon.vertices)
print(f'Poisson mesh vertex 數: {len(recon_verts)}')

# 用所有 session 合併的 raw GT 點來做評估
recon_verts = raw_pts

print('\n=== Bounding box comparison (mm) ===')
print(f'{"":18}{"X range":>20}{"Y range":>20}{"Z range":>20}')
for name, v in [('Original', orig_verts),
                ('Raw GT (all sess)', recon_verts)]:
    print(f'{name:18}'
          f'  [{v[:,0].min()*1000:7.2f},{v[:,0].max()*1000:7.2f}]'
          f'  [{v[:,1].min()*1000:7.2f},{v[:,1].max()*1000:7.2f}]'
          f'  [{v[:,2].min()*1000:7.2f},{v[:,2].max()*1000:7.2f}]')

print('\n=== Dimensions (mm) ===')
for name, v in [('Original', orig_verts),
                ('Raw GT (all sess)', recon_verts)]:
    dx = (v[:,0].max() - v[:,0].min()) * 1000
    dy = (v[:,1].max() - v[:,1].min()) * 1000
    dz = (v[:,2].max() - v[:,2].min()) * 1000
    print(f'{name:18}  X={dx:.2f}  Y={dy:.2f}  Z={dz:.2f}')

print('\n=== Point-to-mesh-surface distance ===')

mesh_scaled = trimesh.Trimesh(vertices=orig_verts, faces=mesh_orig.faces)
surface_pts, _ = trimesh.sample.sample_surface(mesh_scaled, 200000)

orig_pcd = o3d.geometry.PointCloud()
orig_pcd.points = o3d.utility.Vector3dVector(surface_pts)
kd = o3d.geometry.KDTreeFlann(orig_pcd)

sample_idx = np.random.choice(len(recon_verts),
                              min(20000, len(recon_verts)),
                              replace=False)
sampled = recon_verts[sample_idx]
nearest_pts = np.empty_like(sampled)
distances = np.empty(len(sampled))
for i, p in enumerate(sampled):
    _, idx, _ = kd.search_knn_vector_3d(p, 1)
    nearest_pts[i] = surface_pts[idx[0]]
    distances[i] = np.linalg.norm(p - nearest_pts[i])

print(f'  mean   = {distances.mean()*1000:.3f} mm')
print(f'  median = {np.median(distances)*1000:.3f} mm')
print(f'  p90    = {np.percentile(distances, 90)*1000:.3f} mm')
print(f'  p99    = {np.percentile(distances, 99)*1000:.3f} mm')
print(f'  max    = {distances.max()*1000:.3f} mm')

# ── Hausdorff distance（最大誤差，最壞情況）──
print('\n=== Hausdorff distance ===')
print(f'  Hausdorff (recon → orig) = {distances.max()*1000:.3f} mm')

# ── Per-axis error（|dx|, |dy|, |dz| 個別分量誤差）──
print('\n=== Per-axis absolute error (|dx|, |dy|, |dz|) ===')
axis_err = np.abs(sampled - nearest_pts)  # (N, 3)
print(f'{"":12}{"X (mm)":>14}{"Y (mm)":>14}{"Z (mm)":>14}')
for name, fn in [('mean',   lambda x: x.mean()),
                  ('median', lambda x: np.median(x)),
                  ('p90',    lambda x: np.percentile(x, 90)),
                  ('max',    lambda x: x.max())]:
    print(f'  {name:<10}'
          f'{fn(axis_err[:,0])*1000:>14.3f}'
          f'{fn(axis_err[:,1])*1000:>14.3f}'
          f'{fn(axis_err[:,2])*1000:>14.3f}')

# ── 視覺化 1：灰 mesh + 紅點雲 ──
print('\nViz 1: gray = original, red = reconstruction')
orig_o3d = o3d.geometry.TriangleMesh()
orig_o3d.vertices = o3d.utility.Vector3dVector(orig_verts)
orig_o3d.triangles = o3d.utility.Vector3iVector(mesh_orig.faces)
orig_o3d.paint_uniform_color([0.7, 0.7, 0.7])
orig_o3d.compute_vertex_normals()

recon_pcd = o3d.geometry.PointCloud()
recon_pcd.points = o3d.utility.Vector3dVector(recon_verts)
recon_pcd.paint_uniform_color([1.0, 0.2, 0.2])

o3d.visualization.draw_geometries([orig_o3d, recon_pcd],
                                  window_name='Original (gray) vs Reconstruction (red)',
                                  mesh_show_back_face=True)

# ── 視覺化 2：誤差 heatmap（綠=準確，紅=偏差大）──
print('Viz 2: error heatmap (green = accurate, red = high error)')
HEATMAP_MAX_MM = 0.8  # 超過這個誤差就全紅
err_norm = np.clip(distances / (HEATMAP_MAX_MM / 1000.0), 0.0, 1.0)
colors = CMAP_ERR(err_norm)[:, :3]   # viridis_r: 0=黃, 1=深藍紫

heatmap_pcd = o3d.geometry.PointCloud()
heatmap_pcd.points = o3d.utility.Vector3dVector(sampled)
heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)

# ── Colorbar：場景內直立色階條 ──
def make_colorbar(x, y, z_low, z_high, width=0.0015, n=64):
    """從黃(底, err=0)到深藍紫(頂, err=max)的 viridis_r 色階條"""
    vs, cs = [], []
    for i in range(n + 1):
        t = i / n
        z = z_low + (z_high - z_low) * t
        rgb = list(CMAP_ERR(t)[:3])
        vs.append([x - width/2, y, z]); cs.append(rgb)
        vs.append([x + width/2, y, z]); cs.append(rgb)
    ts = []
    for i in range(n):
        v0, v1, v2, v3 = 2*i, 2*i+1, 2*(i+1), 2*(i+1)+1
        ts += [[v0, v1, v3], [v0, v3, v2]]
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(vs)
    m.triangles = o3d.utility.Vector3iVector(ts)
    m.vertex_colors = o3d.utility.Vector3dVector(cs)
    m.compute_vertex_normals()
    return m

def make_tick_spheres(x, y, z_low, z_high, max_mm, n_ticks=5, radius=0.0008):
    """在 color bar 旁邊放小球當刻度標記（viridis_r 配色）"""
    spheres = []
    for i in range(n_ticks + 1):
        t = i / n_ticks
        z = z_low + (z_high - z_low) * t
        rgb = list(CMAP_ERR(t)[:3])
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate([x + 0.003, y, z])
        s.paint_uniform_color(rgb)
        s.compute_vertex_normals()
        spheres.append(s)
        print(f'  tick {i}: z={z*1000:+.2f}mm  →  err {t*max_mm:.2f}mm  '
              f'(RGB={rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})')
    return spheres

# 把 colorbar 放在物體右邊
bx_max = orig_verts[:, 0].max()
bz_min, bz_max = orig_verts[:, 2].min(), orig_verts[:, 2].max()
bar_x = bx_max + 0.010   # +10mm
bar_y = 0.0
print('\n=== Colorbar ticks (0 = 綠 = 0mm, 1 = 紅 = HEATMAP_MAX) ===')
bar = make_colorbar(bar_x, bar_y, bz_min, bz_max)
ticks = make_tick_spheres(bar_x, bar_y, bz_min, bz_max, HEATMAP_MAX_MM, n_ticks=4)

o3d.visualization.draw_geometries([orig_o3d, heatmap_pcd, bar, *ticks],
                                  window_name=f'Error heatmap (0={HEATMAP_MAX_MM:.1f}mm), '
                                              f'mean={distances.mean()*1000:.2f}mm',
                                  mesh_show_back_face=True)

# ── 用 matplotlib 畫一張帶刻度的 colorbar PNG ──
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(2, 6))
norm = mpl.colors.Normalize(vmin=0, vmax=HEATMAP_MAX_MM)
gradient = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(gradient, aspect='auto', cmap='viridis_r', origin='lower',
          extent=[0, 1, 0, HEATMAP_MAX_MM])
ax.set_xticks([])
ax.set_ylabel('Reconstruction error (mm)', fontsize=12)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
ax.set_yticks(np.linspace(0, HEATMAP_MAX_MM, 5))

# 在 colorbar 上標出統計位置
stat_labels = {
    'mean':   distances.mean()*1000,
    'median': np.median(distances)*1000,
    'p90':    np.percentile(distances, 90)*1000,
    'p99':    np.percentile(distances, 99)*1000,
    'max':    distances.max()*1000,
}
for label, val in stat_labels.items():
    if val <= HEATMAP_MAX_MM:
        ax.axhline(val, color='white', lw=0.7, alpha=0.6)
        ax.text(-0.05, val, f'{label}={val:.2f}', ha='right', va='center',
                fontsize=9, color='black')

ax.set_title(f'Error Colorbar\nviridis_r', fontsize=10)
plt.tight_layout()

out_path = os.path.join(SCRIPT_DIR, 'renders', 'error_colorbar.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\n✓ Colorbar saved: {out_path}')
plt.show()
