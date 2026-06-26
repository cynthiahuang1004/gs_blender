import numpy as np
import open3d as o3d
import os, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
render_dir = os.path.join(SCRIPT_DIR, 'renders')

# Collect all session dirs
session_dirs = sorted([
    os.path.join(render_dir, d)
    for d in os.listdir(render_dir)
    if d.startswith('session_') and os.path.isdir(os.path.join(render_dir, d))
])

if not session_dirs:
    print('找不到 session 資料夾')
    exit()

print(f'找到 {len(session_dirs)} 個 session')

all_pts = []
total_files = 0


def euler_xyz_to_matrix(rx, ry, rz):
    """Blender XYZ Euler → 3x3 rotation matrix (apply Rx first, Rz last)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


for session_path in session_dirs:
    # Read base_rotation for this session (different sessions may have different rotations)
    session_json_path = os.path.join(session_path, 'session.json')
    base_rotation = [0.0, 0.0, 0.0]
    if os.path.exists(session_json_path):
        with open(session_json_path) as f:
            sess = json.load(f)
        base_rotation = sess.get('base_rotation', [0.0, 0.0, 0.0])

    has_rotation = any(abs(r) > 1e-6 for r in base_rotation)
    R_inv = euler_xyz_to_matrix(*base_rotation).T  # inverse = transpose for rotation matrix

    sensor_dirs = sorted([
        os.path.join(session_path, d)
        for d in os.listdir(session_path)
        if d.startswith('sensor_') and os.path.isdir(os.path.join(session_path, d))
    ])

    session_pts = []

    for sensor_path in sensor_dirs:
        raw_dir = os.path.join(sensor_path, 'raw_data')
        if not os.path.exists(raw_dir):
            continue

        npy_files = sorted([f for f in os.listdir(raw_dir)
                            if f.endswith('_gt.npy') and len(f) == 11])
        total_files += len(npy_files)

        for fname in npy_files:
            pose_path = os.path.join(raw_dir, fname.replace('_gt.npy', '_pose.json'))
            if not os.path.exists(pose_path):
                continue
            dmap = np.load(os.path.join(raw_dir, fname))
            with open(pose_path) as f:
                pose = json.load(f)

            mask = dmap > 1e-6
            if mask.sum() == 0:
                continue

            H, W = dmap.shape
            length = pose['camera_length']

            # The camera is fixed at world (0, 0, -cam_z). Each sample moves the
            # object so its pressed cell's lowest vertex lands at world (sample_x, sample_y).
            # After the move, pose['location'] = object-origin world position.
            # To reconstruct in object-local frame:
            #   object_local_xy = world_xy - object_origin_world_xy
            #                   = camera_xy  - pose['location'][0:2]
            obj_loc_x = pose['location'][0]
            obj_loc_y = pose['location'][1]
            obj_loc_z = pose['location'][2]
            cam_h = abs(pose['camera_location'][2])

            # Camera 旋轉 (0, 180°, 0)：image 左邊 = 世界 +X，上面 = 世界 +Y。
            # 用 linspace(length, -length, ...) 反向，讓 col 0 = 世界 +length，row 0 = 世界 +length。
            xs = np.linspace(length, -length, W)
            ys = np.linspace(length, -length, H)
            xv, yv = np.meshgrid(xs, ys)

            # Perspective unprojection: 對 z=0 平面，pixel 直接 = 世界 X/Y。
            # 對 world_z != 0 的表面，scale by (world_z + h) / h。
            world_z = -dmap[mask]
            persp = (world_z + cam_h) / cam_h
            world_x = xv[mask] * persp
            world_y = yv[mask] * persp

            X = world_x - obj_loc_x
            Y = world_y - obj_loc_y
            Z = world_z - obj_loc_z

            session_pts.append(np.column_stack([X, Y, Z]))

    if not session_pts:
        print(f'  {os.path.basename(session_path)}: 無有效資料，跳過')
        continue

    pts_sess = np.vstack(session_pts)

    # Un-rotate sessions captured with a non-zero base_rotation
    if has_rotation:
        pts_sess = (R_inv @ pts_sess.T).T

    all_pts.append(pts_sess)
    print(f'  {os.path.basename(session_path)}: 處理完成，{len(pts_sess)} 點')

print(f'總共讀取 {total_files} 個 GT depth 檔')

if not all_pts:
    print('沒有有效資料')
    exit()

pts = np.vstack(all_pts)
print(f'總點數: {len(pts)}')
print(f'X=[{pts[:,0].min()*1000:.2f},{pts[:,0].max()*1000:.2f}]mm')
print(f'Y=[{pts[:,1].min()*1000:.2f},{pts[:,1].max()*1000:.2f}]mm')
print(f'Z=[{pts[:,2].min()*1000:.2f},{pts[:,2].max()*1000:.2f}]mm')

# 視覺化點雲
z = pts[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
colors = np.zeros((len(pts), 3))
colors[:, 0] = np.clip(z_norm * 2 - 1, 0, 1)
colors[:, 1] = np.clip(1 - np.abs(z_norm * 2 - 1), 0, 1)
colors[:, 2] = np.clip(1 - z_norm * 2, 0, 1)

pcd_vis = o3d.geometry.PointCloud()
pcd_vis.points = o3d.utility.Vector3dVector(pts)
pcd_vis.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd_vis], window_name='point cloud')

# Poisson 重建
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd = pcd.voxel_down_sample(0.0001)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
print(f'voxel 後點數: {len(pcd.points)}')

if len(pcd.points) == 0:
    print('點雲太少，無法重建')
    exit()

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=50)
)
normals = np.asarray(pcd.normals)
normals[normals[:, 2] < 0] *= -1
pcd.normals = o3d.utility.Vector3dVector(normals)

mesh_recon, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
densities = np.asarray(densities)
mesh_recon.remove_vertices_by_mask(densities < np.quantile(densities, 0.1))

tri_clusters, cluster_n_tri, _ = mesh_recon.cluster_connected_triangles()
tri_clusters = np.asarray(tri_clusters)
cluster_n_tri = np.asarray(cluster_n_tri)
mesh_recon.remove_triangles_by_mask(tri_clusters != cluster_n_tri.argmax())
mesh_recon.remove_unreferenced_vertices()
mesh_recon = mesh_recon.filter_smooth_laplacian(number_of_iterations=2)
mesh_recon.compute_vertex_normals()

mesh_recon.paint_uniform_color([0.7, 0.7, 0.9])
o3d.visualization.draw_geometries([mesh_recon], window_name='GT reconstruction',
                                   mesh_show_back_face=True)
o3d.io.write_triangle_mesh(os.path.join(render_dir, 'gt_reconstruction.ply'), mesh_recon)
print('完成')
