"""
gs_rgb_render.py — Blender render script for RGB BO (bo_rgb.py).

Fixed scene: session_000/sensor_0000/raw_data/0000_pose.json
             meshes/pattern_01_2_lines_angle_1_2.obj
             meshes/202000 6152_200.obj  (platform)

Env vars
--------
RGB_PARAMS : path to params JSON  (world_strength, dof_fstop,
             obj_r/g/b, obj_roughness, plat_r/g/b, plat_roughness, plat_metallic)
RGB_OUT    : output path without .png extension
"""

import bpy, os, math, json
import numpy as np
from mathutils import Matrix

PARAMS_PATH = os.environ.get('RGB_PARAMS', '')
OUT_PATH    = os.environ.get('RGB_OUT',    '/tmp/gs_rgb_out')
# RGB_SCRIPT_DIR is the project root — passed by bo_rgb.py so this script
# can find meshes/pose even when bpy.data.filepath is in a temp directory.
SCRIPT_DIR    = os.environ.get('RGB_SCRIPT_DIR',
                    os.path.dirname(os.path.abspath(bpy.data.filepath)))
PLATFORM_PATH = os.path.join(SCRIPT_DIR, 'meshes', '202000 6152_200.obj')
MESH_PATH     = os.path.join(SCRIPT_DIR, 'meshes', 'pattern_01_2_lines_angle_1_2.obj')
POSE_PATH     = os.path.join(SCRIPT_DIR, 'renders_line_angle',
                              'session_000', 'sensor_0000', 'raw_data', '0000_pose.json')

CAM_Z     = -0.085
DOF_FOCUS =  0.085

HIDE_NAMES = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
              'LightSurfaceBL', 'LightSurfaceTR',
              'LightSurfaceTL', 'LightSurfaceBR',
              'LightSurfaceRGreen', 'LightSurfaceLGreen']

with open(PARAMS_PATH) as f:
    P = json.load(f)
with open(POSE_PATH) as f:
    pose = json.load(f)

# ── Hide GelSight surfaces ─────────────────────────────────────
for name in HIDE_NAMES:
    if name in bpy.data.objects:
        bpy.data.objects[name].hide_render = True

# ── World background ───────────────────────────────────────────
world = bpy.data.worlds.get('World')
if world and world.node_tree:
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value    = (0.25, 0.25, 0.25, 1.0)
        bg.inputs['Strength'].default_value = float(P.get('world_strength', 2.0))

# ── Platform ───────────────────────────────────────────────────
before = set(bpy.data.objects.keys())
bpy.ops.wm.obj_import(filepath=PLATFORM_PATH,
                      directory=os.path.dirname(PLATFORM_PATH),
                      files=[{'name': os.path.basename(PLATFORM_PATH)}])
plat_objs = [o for o in bpy.data.objects if o.name not in before]

plat_mat = bpy.data.materials.new('plat_bo')
plat_mat.use_nodes = True
p_bsdf = plat_mat.node_tree.nodes.get('Principled BSDF')
if p_bsdf:
    p_bsdf.inputs['Base Color'].default_value        = (
        float(P.get('plat_r', 26/255)), float(P.get('plat_g', 115/255)),
        float(P.get('plat_b', 106/255)), 1.0)
    p_bsdf.inputs['Roughness'].default_value          = float(P.get('plat_roughness', 0.25))
    p_bsdf.inputs['Metallic'].default_value           = float(P.get('plat_metallic',  0.85))
    p_bsdf.inputs['Specular IOR Level'].default_value = 0.6

for po in plat_objs:
    po.data.materials.clear()
    po.data.materials.append(plat_mat)
    po.scale          = (0.001, -0.001, 0.001)
    po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
    po.location       = (-0.08, 0.055, 0.04)
    po.hide_render    = False

bpy.context.view_layer.update()

# ── Object (blue plastic) at fixed pose ───────────────────────
before2 = set(bpy.data.objects.keys())
bpy.ops.wm.obj_import(filepath=MESH_PATH,
                      directory=os.path.dirname(MESH_PATH),
                      files=[{'name': os.path.basename(MESH_PATH)}])
new_objs = [o for o in bpy.data.objects if o.name not in before2]

if new_objs:
    obj = new_objs[0]

    obj_mat = bpy.data.materials.new('obj_bo')
    obj_mat.use_nodes = True
    o_bsdf = obj_mat.node_tree.nodes.get('Principled BSDF')
    if o_bsdf:
        o_bsdf.inputs['Base Color'].default_value        = (
            float(P.get('obj_r', 0.03)), float(P.get('obj_g', 0.18)),
            float(P.get('obj_b', 0.75)), 1.0)
        o_bsdf.inputs['Roughness'].default_value          = float(P.get('obj_roughness', 0.35))
        o_bsdf.inputs['Metallic'].default_value           = 0.0
        o_bsdf.inputs['Specular IOR Level'].default_value = 0.5

    obj.data.materials.clear()
    obj.data.materials.append(obj_mat)

    wm = pose['world_matrix']
    obj.matrix_world = Matrix([wm[0], wm[1], wm[2], wm[3]])
    obj.hide_render  = False
    bpy.context.view_layer.update()
else:
    print('[gs_rgb_render] WARNING: mesh import produced no objects')

# ── Camera ─────────────────────────────────────────────────────
cam_obj  = bpy.data.objects['Camera']
cam_data = cam_obj.data
cam_obj.location[2]         = CAM_Z
cam_data.angle              = math.radians(float(pose.get('camera_fov', 54.0)))
cam_data.dof.use_dof        = True
cam_data.dof.focus_distance = DOF_FOCUS
cam_data.dof.aperture_fstop = float(P.get('dof_fstop', 1.2))

# ── Scene settings ─────────────────────────────────────────────
scene = bpy.context.scene
scene.use_nodes = False
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode  = 'RGB'

# ── Render ─────────────────────────────────────────────────────
scene.render.filepath = OUT_PATH
scene.frame_set(0)
bpy.ops.render.render(write_still=True)

# ── Post-FX (barrel, radial blur, vignette) ────────────────────
def _barrel(img, k1):
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx) / cx; yn = (yy - cy) / cy
    r2 = xn**2 + yn**2; fac = 1.0 + k1 * r2
    xs = np.clip(xn * fac * cx + cx, 0, W - 1)
    ys = np.clip(yn * fac * cy + cy, 0, H - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None]; wy = (ys - y0)[:, :, None]
    return (img[y0, x0] * (1-wy) * (1-wx) + img[y0, x1] * (1-wy) * wx +
            img[y1, x0] * wy * (1-wx) + img[y1, x1] * wy * wx)

def _gauss_blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-x**2 / (2 * sigma**2)); k /= k.sum()
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, img[:,:,c])
        out[:,:,c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    return np.clip(out, 0.0, 1.0)

def _boost_sat_center(rgb, factor, dist):
    if abs(factor - 1.0) < 0.01:
        return rgb
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    gray = gray[:,:,None]
    center_w = np.clip(1.0 - dist, 0, 1)[:,:,None] ** 2
    local_fac = 1.0 + (factor - 1.0) * center_w
    return np.clip(gray + (rgb - gray) * local_fac, 0, 1)

png_path = OUT_PATH + '.png'
try:
    if os.path.exists(png_path):
        img_bpy = bpy.data.images.load(png_path)
        W2, H2 = img_bpy.size[0], img_bpy.size[1]
        px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H2, W2, 4)
        bpy.data.images.remove(img_bpy)

        k1       = float(P.get('barrel_k1',    0.07))
        sigma    = float(P.get('blur_sigma',   1.25))
        vign     = float(P.get('vignette',     0.25))
        tint_r   = float(P.get('tint_r',       0.85))
        tint_g   = float(P.get('tint_g',       0.70))
        tint_b   = float(P.get('tint_b',       0.25))
        tint_str = float(P.get('tint_strength',0.25))
        tint_cx  = float(P.get('tint_cx',      0.0))
        tint_cy  = float(P.get('tint_cy',      0.0))
        sat      = float(P.get('sat_boost',    1.2))
        haze     = float(P.get('haze_opacity', 0.10))
        falloff  = 1.5

        rgb = _barrel(px[:,:,:3], k1)
        blurred = _gauss_blur(rgb, sigma)
        cy3, cx3 = H2 / 2.0, W2 / 2.0
        yy3, xx3 = np.mgrid[0:H2, 0:W2]
        tcx = cx3 + tint_cx * cx3
        tcy = cy3 + tint_cy * cy3
        dist = np.sqrt(((yy3 - tcy) / cy3)**2 + ((xx3 - tcx) / cx3)**2)
        weight = np.clip(dist ** falloff, 0, 1)[:,:,None]
        rgb = rgb * (1 - weight) + blurred * weight
        tint_color = np.array([[[tint_r, tint_g, tint_b]]], dtype=np.float32)
        tint_w = np.clip(dist ** 2 * tint_str, 0, 1)[:,:,None]
        rgb = rgb * (1 - tint_w) + tint_color * tint_w
        haze_w = np.clip(dist ** 1.5 * haze, 0, 1)[:,:,None]
        rgb = rgb * (1 - haze_w) + tint_color * 0.7 * haze_w
        rgb = _boost_sat_center(rgb, sat, dist)
        mask = np.clip(1.0 - dist**2 * vign, 0, 1)[:,:,None]
        rgb = np.clip(rgb * mask, 0, 1)

        out_bpy = bpy.data.images.new('_fx_tmp', W2, H2, alpha=False)
        out_px = np.ones((H2, W2, 4), dtype=np.float32)
        out_px[:,:,:3] = rgb
        out_bpy.pixels = out_px.flatten().tolist()
        out_bpy.filepath_raw = png_path
        out_bpy.file_format = 'PNG'
        out_bpy.save()
        bpy.data.images.remove(out_bpy)
        print(f'[gs_rgb_render] post-FX applied → {png_path}', flush=True)
except Exception as _e:
    import traceback; traceback.print_exc()
    print(f'[gs_rgb_render] post-FX ERROR: {_e}', flush=True)

print(f'[gs_rgb_render] saved → {OUT_PATH}')
import os as _os; _os._exit(0)
