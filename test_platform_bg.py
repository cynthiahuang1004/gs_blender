"""
test_platform_bg.py
--------------------
Test: import the robot arm platform OBJ as background geometry,
render one RGB sample, see how it looks.

Run:
    blender --background gelsight_sampler.blend --python test_platform_bg.py
"""

import bpy, os, sys, json, math
import numpy as np
from mathutils import Matrix

SCRIPT_DIR    = os.path.dirname(os.path.abspath(bpy.data.filepath))
POSE_PATH     = os.path.join(SCRIPT_DIR,
    'renders_line_angle', 'session_000', 'sensor_0000', 'raw_data', '0000_pose.json')
MESH_PATH     = os.path.join(SCRIPT_DIR, 'meshes', 'pattern_01_2_lines_angle_1_2.obj')
PLATFORM_PATH = os.path.join(SCRIPT_DIR, 'meshes', '202000 6152_200.obj')
OUT_DIR       = os.path.join(SCRIPT_DIR, 'bo_results')

CAM_Z     = -0.085
DOF_FOCUS = 0.085
DOF_FSTOP = 1.2

HIDE_NAMES = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
              'LightSurfaceBL', 'LightSurfaceTR',
              'LightSurfaceTL', 'LightSurfaceBR',
              'LightSurfaceRGreen', 'LightSurfaceLGreen']

# ── Load pose ──────────────────────────────────────────────────
with open(POSE_PATH) as f:
    pose = json.load(f)

# ── Import tactile object ──────────────────────────────────────
before = set(bpy.data.objects.keys())
bpy.ops.wm.obj_import(filepath=MESH_PATH, directory=os.path.dirname(MESH_PATH),
                       files=[{'name': os.path.basename(MESH_PATH)}])
new_objs = [o.name for o in bpy.data.objects if o.name not in before]
obj = bpy.data.objects[new_objs[0]]

blue_mat = bpy.data.materials.new('plastic_blue')
blue_mat.use_nodes = True
bsdf = blue_mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value          = (0.03, 0.18, 0.75, 1.0)
    bsdf.inputs['Roughness'].default_value           = 0.35
    bsdf.inputs['Specular IOR Level'].default_value  = 0.5
    bsdf.inputs['Metallic'].default_value            = 0.0
obj.data.materials.clear()
obj.data.materials.append(blue_mat)

wm = pose['world_matrix']
obj.matrix_world = Matrix([wm[0], wm[1], wm[2], wm[3]])
bpy.context.view_layer.update()
print('Object location:', list(obj.matrix_world.to_translation()))

# ── Import platform ────────────────────────────────────────────
before2 = set(bpy.data.objects.keys())
bpy.ops.wm.obj_import(filepath=PLATFORM_PATH,
                       directory=os.path.dirname(PLATFORM_PATH),
                       files=[{'name': os.path.basename(PLATFORM_PATH)}])
platform_objs = [o for o in bpy.data.objects if o.name not in before2]
print(f'Platform objects imported: {[o.name for o in platform_objs]}')

# Gray plastic material for the platform
gray_mat = bpy.data.materials.new('platform_teal')
gray_mat.use_nodes = True
p_bsdf = gray_mat.node_tree.nodes.get('Principled BSDF')
if p_bsdf:
    p_bsdf.inputs['Base Color'].default_value          = (26/255, 115/255, 106/255, 1.0)
    p_bsdf.inputs['Roughness'].default_value           = 0.25
    p_bsdf.inputs['Specular IOR Level'].default_value  = 0.6
    p_bsdf.inputs['Metallic'].default_value            = 0.85

for po in platform_objs:
    po.data.materials.clear()
    po.data.materials.append(gray_mat)
    po.hide_render = False
    # negative Y flips face; rotate Z 90° to make ribs horizontal
    po.scale = (0.001, -0.001, 0.001)
    po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
    po.location = (-0.08, 0.055, 0.04)
    bpy.context.view_layer.update()
    print(f'  {po.name}: location={list(po.location)}  rotation_deg={[round(math.degrees(r),1) for r in po.rotation_euler]}')
    print(f'  dimensions after rotate: {[round(d,4) for d in po.dimensions]}')

# ── Camera ─────────────────────────────────────────────────────
cam_obj  = bpy.data.objects['Camera']
cam_data = cam_obj.data
cam_obj.location[2]         = CAM_Z
cam_data.angle              = math.radians(pose['camera_fov'])
cam_data.dof.use_dof        = False

# ── World background ───────────────────────────────────────────
world = bpy.data.worlds.get('World')
if world and world.node_tree:
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value    = (0.25, 0.25, 0.25, 1.0)
        bg_node.inputs['Strength'].default_value = 2.0

# ── Hide gel / light surfaces ──────────────────────────────────
for name in HIDE_NAMES:
    if name in bpy.data.objects:
        bpy.data.objects[name].hide_render = True
obj.hide_render = False

# ── Render ─────────────────────────────────────────────────────
scene = bpy.context.scene
scene.use_nodes = False
scene.render.film_transparent = False   # platform fills background, no need for transparency
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode  = 'RGB'

out = os.path.join(OUT_DIR, 'debug_platform_bg')
scene.render.filepath = out
scene.frame_set(0)
bpy.ops.render.render(write_still=True)
print(f'\nSaved: {out}.png')

cam_data.dof.use_dof = False

# ── Post-process: gel simulation (barrel distortion + vignette + radial blur) ──
GEL_BARREL_K1    = 0.07  # barrel distortion strength (positive=outward)
GEL_VIGNETTE     = 0.25  # edge darkening strength (0=none, 1=strong)
GEL_BLUR_SIGMA   = 1.25   # max blur at edges
GEL_BLUR_FALLOFF = 1.5   # sharpness of center: higher = sharper centre

def _barrel(img, k1):
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx) / cx;  yn = (yy - cy) / cy
    r2 = xn**2 + yn**2
    fac = 1.0 + k1 * r2
    xs = np.clip(xn * fac * cx + cx, 0, W - 1)
    ys = np.clip(yn * fac * cy + cy, 0, H - 1)
    x0 = np.floor(xs).astype(int);  x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int);  y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None];     wy = (ys - y0)[:, :, None]
    return np.clip(
        img[y0, x0] * (1-wx) * (1-wy) + img[y0, x1] * wx * (1-wy) +
        img[y1, x0] * (1-wx) * wy     + img[y1, x1] * wx * wy, 0, 1)

def _vignette(img, strength):
    H, W = img.shape[:2]
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((yy - cy) / cy)**2 + ((xx - cx) / cx)**2)
    mask = np.clip(1.0 - dist**2 * strength, 0, 1)[:, :, None]
    return img * mask

def _gaussian_blur(img, sigma):
    size = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(size) - size // 2
    k = np.exp(-x**2 / (2 * sigma**2));  k /= k.sum()
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, img[:,:,c])
        out[:,:,c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    return np.clip(out, 0.0, 1.0)

png_path = out + '.png'
img_bpy = bpy.data.images.load(png_path)
W, H = img_bpy.size[0], img_bpy.size[1]
px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H, W, 4)
bpy.data.images.remove(img_bpy)

rgb = px[:, :, :3]

# 1. barrel distortion
rgb = _barrel(rgb, GEL_BARREL_K1)

# 2. radial blur (centre sharp → edge blurry)
blurred = _gaussian_blur(rgb, GEL_BLUR_SIGMA)
cy2, cx2 = H / 2.0, W / 2.0
yy2, xx2 = np.mgrid[0:H, 0:W]
dist2 = np.sqrt(((yy2 - cy2) / cy2)**2 + ((xx2 - cx2) / cx2)**2)
weight = np.clip(dist2 ** GEL_BLUR_FALLOFF, 0.0, 1.0)[:, :, None]
rgb = rgb * (1.0 - weight) + blurred * weight

# 3. vignette
rgb = _vignette(rgb, GEL_VIGNETTE)

out_img = bpy.data.images.new('_gel_tmp', W, H, alpha=False)
out_px = np.ones((H, W, 4), dtype=np.float32)
out_px[:, :, :3] = rgb
out_img.pixels = out_px.flatten().tolist()
out_img.filepath_raw = png_path
out_img.file_format = 'PNG'
out_img.save()
bpy.data.images.remove(out_img)
print(f'Gel FX applied: barrel={GEL_BARREL_K1} vignette={GEL_VIGNETTE} blur_sigma={GEL_BLUR_SIGMA}')

try:
    bpy.ops.wm.quit_blender()
except Exception:
    sys.exit(0)
