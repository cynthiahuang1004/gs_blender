"""3D 視覺化某個 session 的 sample region。

跑法：
    python visualize_sample_3d.py             # 預設看 session_000
    python visualize_sample_3d.py 5           # 看 session_005

顏色說明：
    灰色 mesh   = 物體（已套 base_rotation 跟 scale）
    🔴 紅色點    = AUTO_CLIP 抓到的「特徵 vertex」（z < z_min + AUTO_CLIP_BAND_MM）
    🟢 綠色小球  = 每個 sample 的中心位置
    🟡 黃色方框  = 每個 sample 的 sensor FOV (sensor_width × sensor_width)
    🔵 藍色軸    = 座標系：X 紅、Y 綠、Z 藍 (Open3D 預設)
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
session_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
session_dir = os.path.join(SCRIPT_DIR, 'renders', f'session_{session_idx:03d}')
session_path = os.path.join(session_dir, 'session.json')

# ── Load ──
with open(session_path) as f:
    sess = json.load(f)

obj_name      = sess['obj']
fixed_scale   = sess['fixed_scale']
base_rotation = sess.get('base_rotation', [0, 0, 0])
z_anchor      = sess.get('z_anchor', None)
cells         = sess.get('valid_cells', [])
sensor_width  = sess.get('_sensor_length_mm', 10.0) / 1000.0 * 2

print(f'Session {session_idx:03d}')
print(f'  obj           : {obj_name}')
print(f'  fixed_scale   : {fixed_scale}')
print(f'  base_rotation : {[round(np.degrees(r), 1) for r in base_rotation]}°')
print(f'  z_anchor      : {z_anchor*1000:.2f} mm' if z_anchor else '  z_anchor: N/A')
print(f'  cells         : {len(cells)}')
print(f'  sensor FOV    : {sensor_width*1000:.1f} × {sensor_width*1000:.1f} mm')

mesh_path = os.path.join(SCRIPT_DIR, 'meshes', f'{obj_name}.obj')
mesh = trimesh.load(mesh_path, force='mesh')
R = Rotation.from_euler('xyz', base_rotation)
verts = R.apply(mesh.vertices) / fixed_scale

z_min = verts[:, 2].min()
z_max = verts[:, 2].max()
print(f'  mesh z range  : [{z_min*1000:.2f}, {z_max*1000:.2f}] mm')

# ── (1) Mesh ──
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
o3d_mesh.paint_uniform_color([0.75, 0.75, 0.75])
o3d_mesh.compute_vertex_normals()

# ── (2) Ridge / feature vertices (距 z_min 最近 2mm 內) ──
AUTO_CLIP_BAND_MM = 2.0
ridge_mask = verts[:, 2] < z_min + AUTO_CLIP_BAND_MM / 1000.0
ridge_verts = verts[ridge_mask]
ridge_pcd = o3d.geometry.PointCloud()
ridge_pcd.points = o3d.utility.Vector3dVector(ridge_verts)
ridge_pcd.paint_uniform_color([1.0, 0.1, 0.1])
print(f'  ridge verts   : {len(ridge_verts)} (z < {(z_min + AUTO_CLIP_BAND_MM/1000)*1000:.2f}mm)')

# ── (3) Sample centers (綠色小球) ──
geoms = [o3d_mesh, ridge_pcd]
for cell in cells:
    cx, cy = cell['cx'], cell['cy']
    cz = z_min  # 放在最深處給易見
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.0006)
    s.translate([cx, cy, cz])
    s.paint_uniform_color([0.0, 0.9, 0.0])
    s.compute_vertex_normals()
    geoms.append(s)

# ── (4) Sample FOV (黃色框,LineSet) ──
half = sensor_width / 2
fov_lines = []
fov_points = []
fov_colors = []
idx_offset = 0
for cell in cells:
    cx, cy = cell['cx'], cell['cy']
    cz = z_min - 0.0001  # 稍微低一點避免跟 mesh 重疊閃爍
    # 四個角
    fov_points += [
        [cx-half, cy-half, cz],
        [cx+half, cy-half, cz],
        [cx+half, cy+half, cz],
        [cx-half, cy+half, cz],
    ]
    # 四條邊
    o = idx_offset
    fov_lines += [[o, o+1], [o+1, o+2], [o+2, o+3], [o+3, o]]
    fov_colors += [[1.0, 0.85, 0.0]] * 4
    idx_offset += 4

if fov_points:
    fov_set = o3d.geometry.LineSet()
    fov_set.points = o3d.utility.Vector3dVector(fov_points)
    fov_set.lines  = o3d.utility.Vector2iVector(fov_lines)
    fov_set.colors = o3d.utility.Vector3dVector(fov_colors)
    geoms.append(fov_set)

# ── (5) AUTO_CLIP bounding box (青色框) ──
if len(ridge_verts) > 0:
    fxy = ridge_verts[:, :2]
    fx_min, fx_max = fxy[:, 0].min(), fxy[:, 0].max()
    fy_min, fy_max = fxy[:, 1].min(), fxy[:, 1].max()
    margin = 0.001
    bx_min, bx_max = fx_min - margin, fx_max + margin
    by_min, by_max = fy_min - margin, fy_max + margin
    bz = z_min - 0.0005
    clip_pts = [
        [bx_min, by_min, bz], [bx_max, by_min, bz],
        [bx_max, by_max, bz], [bx_min, by_max, bz],
    ]
    clip_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    clip_set = o3d.geometry.LineSet()
    clip_set.points = o3d.utility.Vector3dVector(clip_pts)
    clip_set.lines  = o3d.utility.Vector2iVector(clip_lines)
    clip_set.colors = o3d.utility.Vector3dVector([[0.0, 0.8, 0.9]] * 4)
    geoms.append(clip_set)
    print(f'  clip bbox     : {(bx_max-bx_min)*1000:.1f} × {(by_max-by_min)*1000:.1f} mm')

# ── 座標軸 ──
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
geoms.append(axes)

print('\n打開 Open3D 視窗...（拖曳旋轉,滾輪縮放）')
o3d.visualization.draw_geometries(
    geoms,
    window_name=f'Session {session_idx:03d} — gray=mesh, red=ridge, green=samples, yellow=FOV, cyan=clip box',
    mesh_show_back_face=True,
)
