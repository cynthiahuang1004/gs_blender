"""
render_rgb_batch.py
-------------------
Blender script: renders RGB images for all sessions/samples in renders_line_angle/.
Each rendered object (transparent background) is composited over a randomly chosen
real background image from real_data/rgb_images/.

Run:
    blender --background gelsight_sampler.blend --python render_rgb_batch.py

Force re-render existing images:
    set GELSIGHT_FORCE=1
    blender --background gelsight_sampler.blend --python render_rgb_batch.py
"""

import bpy, os, sys, json, math
import numpy as np
from mathutils import Matrix

SCRIPT_DIR    = os.path.dirname(os.path.abspath(bpy.data.filepath))
RENDERS_DIR   = os.environ.get('GELSIGHT_RENDER_DIR',
                                os.path.join(SCRIPT_DIR, 'renders_line_angle'))
MESH_DIR      = os.path.join(SCRIPT_DIR, 'meshes')
PLATFORM_PATH = os.path.join(SCRIPT_DIR, 'meshes', '202000 6152_200.obj')
FORCE         = os.environ.get('GELSIGHT_FORCE') == '1'

# Camera / DOF params
CAM_Z     = -0.085
DOF_FOCUS = 0.085
DOF_FSTOP = 1.2

# Gel/light surfaces to hide so only the object is visible
HIDE_NAMES = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
              'LightSurfaceBL', 'LightSurfaceTR',
              'LightSurfaceTL', 'LightSurfaceBR',
              'LightSurfaceRGreen', 'LightSurfaceLGreen']

# ── Import platform once and keep it hidden until render ──────
def _setup_platform():
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=PLATFORM_PATH,
                          directory=os.path.dirname(PLATFORM_PATH),
                          files=[{'name': os.path.basename(PLATFORM_PATH)}])
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    platform_objs = new_objs

    teal_mat = bpy.data.materials.new('platform_teal')
    teal_mat.use_nodes = True
    p_bsdf = teal_mat.node_tree.nodes.get('Principled BSDF')
    if p_bsdf:
        p_bsdf.inputs['Base Color'].default_value         = (26/255, 115/255, 106/255, 1.0)
        p_bsdf.inputs['Roughness'].default_value          = 0.25
        p_bsdf.inputs['Specular IOR Level'].default_value = 0.6
        p_bsdf.inputs['Metallic'].default_value           = 0.85

    for po in platform_objs:
        po.data.materials.clear()
        po.data.materials.append(teal_mat)
        po.scale          = (0.001, -0.001, 0.001)
        po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
        po.location       = (-0.08, 0.055, 0.04)
        po.hide_render    = False
    bpy.context.view_layer.update()
    print(f'Platform imported: {[o.name for o in platform_objs]}')
    return platform_objs

_platform_objs = _setup_platform()

# ── Gel post-processing parameters ───────────────────────────
GEL_BARREL_K1    = 0.07
GEL_VIGNETTE     = 0.25
GEL_BLUR_SIGMA   = 1.25
GEL_BLUR_FALLOFF = 1.5

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

def _gaussian_blur(img, sigma):
    size = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(size) - size // 2
    k = np.exp(-x**2 / (2 * sigma**2));  k /= k.sum()
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, img[:,:,c])
        out[:,:,c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, h)
    return np.clip(out, 0.0, 1.0)

def _gel_fx(png_path):
    img_bpy = bpy.data.images.load(png_path)
    W, H = img_bpy.size[0], img_bpy.size[1]
    px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img_bpy)
    rgb = px[:, :, :3]
    # barrel distortion
    rgb = _barrel(rgb, GEL_BARREL_K1)
    # radial blur
    blurred = _gaussian_blur(rgb, GEL_BLUR_SIGMA)
    cy2, cx2 = H / 2.0, W / 2.0
    yy2, xx2 = np.mgrid[0:H, 0:W]
    dist2 = np.sqrt(((yy2-cy2)/cy2)**2 + ((xx2-cx2)/cx2)**2)
    weight = np.clip(dist2 ** GEL_BLUR_FALLOFF, 0.0, 1.0)[:, :, None]
    rgb = rgb * (1.0 - weight) + blurred * weight
    # vignette
    yy3, xx3 = np.mgrid[0:H, 0:W]
    dist3 = np.sqrt(((yy3-cy2)/cy2)**2 + ((xx3-cx2)/cx2)**2)
    mask = np.clip(1.0 - dist3**2 * GEL_VIGNETTE, 0, 1)[:, :, None]
    rgb = np.clip(rgb * mask, 0, 1)
    out_img = bpy.data.images.new('_gel_tmp', W, H, alpha=False)
    out_px = np.ones((H, W, 4), dtype=np.float32)
    out_px[:, :, :3] = rgb
    out_img.pixels = out_px.flatten().tolist()
    out_img.filepath_raw = png_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)

# ── Blue plastic material ─────────────────────────────────────
_blue_mat = bpy.data.materials.new('plastic_blue_rgb')
_blue_mat.use_nodes = True
_bsdf = _blue_mat.node_tree.nodes.get('Principled BSDF')
if _bsdf:
    _bsdf.inputs['Base Color'].default_value         = (0.03, 0.18, 0.75, 1.0)
    _bsdf.inputs['Roughness'].default_value          = 0.35
    _bsdf.inputs['Specular IOR Level'].default_value = 0.5
    _bsdf.inputs['Metallic'].default_value           = 0.0

# ── Mesh cache (import each unique mesh once) ─────────────────
_mesh_cache = {}   # stem → Blender object name

def get_mesh(stem):
    if stem in _mesh_cache:
        return bpy.data.objects[_mesh_cache[stem]]
    path = os.path.join(MESH_DIR, stem + '.obj')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Mesh not found: {path}')
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=path, directory=MESH_DIR,
                          files=[{'name': stem + '.obj'}])
    new_objs = [o.name for o in bpy.data.objects if o.name not in before]
    if not new_objs:
        raise RuntimeError(f'Import produced no objects: {path}')
    name = new_objs[0]
    obj  = bpy.data.objects[name]
    obj.data.materials.clear()
    obj.data.materials.append(_blue_mat)
    obj.hide_render = True   # hidden until this sample's render
    _mesh_cache[stem] = name
    print(f'  Imported mesh: {stem} → {name}')
    return obj

# ── Render a single sample ────────────────────────────────────
def render_rgb(mesh_obj, pose, out_path):
    # Place object at recorded world position
    wm = pose['world_matrix']
    mesh_obj.matrix_world = Matrix([wm[0], wm[1], wm[2], wm[3]])
    mesh_obj.hide_render = False
    bpy.context.view_layer.update()

    # Camera
    cam_obj  = bpy.data.objects['Camera']
    cam_data = cam_obj.data
    cam_obj.location[2]         = CAM_Z
    cam_data.angle              = math.radians(pose['camera_fov'])
    cam_data.dof.use_dof        = True
    cam_data.dof.focus_distance = DOF_FOCUS
    cam_data.dof.aperture_fstop = DOF_FSTOP

    # World background brightness
    world = bpy.data.worlds.get('World')
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value    = (0.25, 0.25, 0.25, 1.0)
            bg_node.inputs['Strength'].default_value = 2.0

    # Hide gel / light surfaces; platform stays visible
    for name in HIDE_NAMES:
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = True

    # Render directly — platform provides the background
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGB'
    scene.render.filepath = out_path   # Blender appends .png
    scene.frame_set(0)
    bpy.ops.render.render(write_still=True)

    # gel post-processing (barrel + radial blur + vignette)
    _gel_fx(out_path + '.png')

    mesh_obj.hide_render = True
    cam_data.dof.use_dof = False

# ── Main loop ─────────────────────────────────────────────────
rendered = 0
skipped  = 0

sessions = sorted(d for d in os.listdir(RENDERS_DIR) if d.startswith('session_'))
for session_name in sessions:
    session_dir  = os.path.join(RENDERS_DIR, session_name)
    session_json = os.path.join(session_dir, 'session.json')
    if not os.path.exists(session_json):
        print(f'[skip] {session_name}: no session.json')
        continue
    with open(session_json) as f:
        session = json.load(f)

    try:
        mesh_obj = get_mesh(session['obj'])
    except Exception as e:
        print(f'[skip] {session_name}: {e}')
        continue

    for sensor_name in sorted(d for d in os.listdir(session_dir) if d.startswith('sensor_')):
        sensor_dir = os.path.join(session_dir, sensor_name)
        raw_dir    = os.path.join(sensor_dir, 'raw_data')
        rgb_dir    = os.path.join(sensor_dir, 'rgb')
        if not os.path.isdir(raw_dir):
            continue
        os.makedirs(rgb_dir, exist_ok=True)

        pose_files = sorted(f for f in os.listdir(raw_dir) if f.endswith('_pose.json'))
        for pose_file in pose_files:
            idx      = pose_file[:-len('_pose.json')]   # '0000', '0001', …
            out_path = os.path.join(rgb_dir, idx)       # Blender appends .png

            if not FORCE and os.path.exists(out_path + '.png'):
                skipped += 1
                continue

            with open(os.path.join(raw_dir, pose_file)) as f:
                pose = json.load(f)

            print(f'  [{rendered+1:3d}] {session_name}/{sensor_name}/{idx}')
            render_rgb(mesh_obj, pose, out_path)
            rendered += 1

print(f'\nDone.  Rendered={rendered}  Skipped={skipped}')

try:
    bpy.ops.wm.quit_blender()
except Exception:
    sys.exit(0)
