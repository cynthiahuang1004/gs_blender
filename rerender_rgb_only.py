"""
rerender_rgb_only.py
====================
Re-render ONLY the RGB images for a given session, reading existing pose.json
files for exact object positioning. Does NOT re-render tactile, depth, or pose.

Use case: object color was wrong (e.g. pattern_33 was blue, should be black).

The script must be run inside Blender:
    blender --background gelsight_sampler.blend --python rerender_rgb_only.py

Environment variables:
    GELSIGHT_RENDER_DIR   — session directory (e.g. renders_v3/pattern_33/session_002)
    GELSIGHT_RGB_PARAMS   — path to best_rgb_params.json
    GELSIGHT_OBJ_COLOR    — "black" or "blue" (default: auto from BLACK_OBJS)
"""
import os, sys, json, math, time
import numpy as np

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir)

import bpy
from mathutils import Matrix, Vector, Euler

# ── Load RGB params (same as scripting.py) ──
_rgb_params_path = os.environ.get('GELSIGHT_RGB_PARAMS',
                                   os.path.join(dir, 'bo_results', 'rgb', 'best_rgb_params.json'))
with open(_rgb_params_path) as _f:
    _rgb_bo = json.load(_f)

# ── Constants from scripting.py ──
BLACK_OBJS = {
    'pattern_04_3_lines_angle_2', 'pattern_06_5_lines_angle_1',
    'pattern_31_rod', 'pattern_32', 'pattern_33',
    'pattern_35', 'pattern_36', 'pattern_37',
}
OBJ_BLACK_COLOR = (0.02, 0.02, 0.02, 1.0)
OBJ_BLUE_COLOR = (_rgb_bo.get('obj_r', 0.03), _rgb_bo.get('obj_g', 0.08),
                  _rgb_bo.get('obj_b', 0.40), 1.0)

RGB_CAM_Z        = -0.085
RGB_BASE_SIZE_REF = 0.082
RGB_FOV          = _rgb_bo.get('rgb_fov',              55.0)
RGB_DOF_FOCUS    = 0.085
RGB_DOF_FSTOP    = _rgb_bo.get('dof_fstop',      1.2)
RGB_WORLD_STR    = _rgb_bo.get('world_strength',  2.0)
RGB_BARREL_K1    = _rgb_bo.get('barrel_k1',       0.07)
RGB_VIGNETTE     = _rgb_bo.get('vignette',        0.25)
RGB_BLUR_SIGMA   = _rgb_bo.get('blur_sigma',      1.25)
RGB_BLUR_FALLOFF = 1.5
RGB_TINT_R       = _rgb_bo.get('tint_r',          0.35)
RGB_TINT_G       = _rgb_bo.get('tint_g',          0.33)
RGB_TINT_B       = _rgb_bo.get('tint_b',          0.30)
RGB_TINT_STR     = _rgb_bo.get('tint_strength',   0.6)
RGB_TINT_CX      = _rgb_bo.get('tint_cx',         0.0)
RGB_TINT_CY      = _rgb_bo.get('tint_cy',         0.0)
RGB_HAZE_OPACITY = _rgb_bo.get('haze_opacity',    0.3)
RGB_BLUE_SHIFT   = _rgb_bo.get('blue_shift',      0.0)
RGB_CONTRAST     = _rgb_bo.get('contrast',        1.0)
RGB_GAMMA        = _rgb_bo.get('gamma',           1.0)
RGB_CLARITY      = _rgb_bo.get('clarity',         0.0)
RGB_SAT_BOOST    = _rgb_bo.get('sat_boost',       0.0)
RGB_BRIGHTNESS   = _rgb_bo.get('brightness',      1.0)
RGB_SPEC_STR     = _rgb_bo.get('spec_str',        0.0)
RGB_SPEC_CX      = _rgb_bo.get('spec_cx',         0.0)
RGB_SPEC_CY      = _rgb_bo.get('spec_cy',        -0.2)
RGB_SPEC_SIZE    = _rgb_bo.get('spec_size',        0.3)
RGB_EDGE_DARK    = _rgb_bo.get('edge_dark',        0.0)
RGB_GEL_GRAD_STR = _rgb_bo.get('gel_grad_str',    0.0)
RGB_GEL_GRAD_ANG = _rgb_bo.get('gel_grad_ang',    0.0)
RGB_WB_R         = _rgb_bo.get('wb_r',            1.0)
RGB_WB_G         = _rgb_bo.get('wb_g',            1.0)
RGB_WB_B         = _rgb_bo.get('wb_b',            1.0)
RGB_GLOBAL_BLUR  = _rgb_bo.get('global_blur',     0.0)
RGB_REFRACT_K2   = _rgb_bo.get('refract_k2',      0.0)

RGB_HIDE_NAMES = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
                  'LightSurfaceBL', 'LightSurfaceTR',
                  'LightSurfaceTL', 'LightSurfaceBR',
                  'LightSurfaceRGreen', 'LightSurfaceLGreen']


# ── Post-FX helpers (pure numpy, no cv2 — same as scripting.py) ──
def _barrel(img, k1):
    H, W = img.shape[:2]
    cx2, cy2 = W / 2.0, H / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx - cx2) / cx2; yn = (yy - cy2) / cy2
    r2 = xn**2 + yn**2; fac = 1.0 + k1 * r2
    xs = np.clip(xn * fac * cx2 + cx2, 0, W - 1)
    ys = np.clip(yn * fac * cy2 + cy2, 0, H - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    wx = (xs - x0)[:, :, None]; wy = (ys - y0)[:, :, None]
    return np.clip(img[y0,x0]*(1-wx)*(1-wy)+img[y0,x1]*wx*(1-wy)+
                   img[y1,x0]*(1-wx)*wy+img[y1,x1]*wx*wy, 0, 1)


def _gaussian_blur_ch(img, sigma):
    size = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(size) - size // 2
    k = np.exp(-x**2 / (2 * sigma**2)); k /= k.sum()
    out = np.empty_like(img)
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


def _gel_fx(png_path):
    img_bpy = bpy.data.images.load(png_path)
    W, H = img_bpy.size[0], img_bpy.size[1]
    px = np.empty(W * H * 4, dtype=np.float32)
    img_bpy.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)
    bpy.data.images.remove(img_bpy)

    rgb = _barrel(px[:, :, :3], RGB_BARREL_K1)
    if abs(RGB_REFRACT_K2) > 1e-4:
        rgb = _barrel(rgb, RGB_REFRACT_K2)
    if RGB_GLOBAL_BLUR > 0.1:
        rgb = _gaussian_blur_ch(rgb, RGB_GLOBAL_BLUR)

    blurred = _gaussian_blur_ch(rgb, RGB_BLUR_SIGMA)
    cy3, cx3 = H / 2.0, W / 2.0
    yy3, xx3 = np.mgrid[0:H, 0:W]
    tcx = cx3 + RGB_TINT_CX * cx3
    tcy = cy3 + RGB_TINT_CY * cy3
    dist3 = np.sqrt(((yy3 - tcy) / cy3) ** 2 + ((xx3 - tcx) / cx3) ** 2)
    weight = np.clip(dist3 ** RGB_BLUR_FALLOFF, 0, 1)[:, :, None]
    rgb = rgb * (1 - weight) + blurred * weight

    tint_color = np.array([[[RGB_TINT_R, RGB_TINT_G, RGB_TINT_B]]], dtype=np.float32)
    tint_w = np.clip(dist3 ** 2 * RGB_TINT_STR, 0, 1)[:, :, None]
    rgb = rgb * (1 - tint_w) + tint_color * tint_w

    haze_w = np.clip(dist3 ** 1.5 * RGB_HAZE_OPACITY, 0, 1)[:, :, None]
    rgb = rgb * (1 - haze_w) + tint_color * 0.7 * haze_w

    if abs(RGB_BLUE_SHIFT) > 1e-4:
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] - RGB_BLUE_SHIFT * 0.3, 0, 1)
        rgb[:, :, 1] = np.clip(rgb[:, :, 1] - RGB_BLUE_SHIFT * 0.1, 0, 1)
        rgb[:, :, 2] = np.clip(rgb[:, :, 2] + RGB_BLUE_SHIFT * 0.2, 0, 1)

    if abs(RGB_CONTRAST - 1.0) > 1e-4:
        rgb = np.clip((rgb - 0.5) * RGB_CONTRAST + 0.5, 0, 1)
    if abs(RGB_GAMMA - 1.0) > 1e-4:
        rgb = np.clip(np.power(np.maximum(rgb, 0), 1.0 / RGB_GAMMA), 0, 1)
    if abs(RGB_CLARITY) > 1e-4:
        soft = _gaussian_blur_ch(rgb, 3.0)
        detail = rgb - soft
        rgb = np.clip(rgb + detail * RGB_CLARITY, 0, 1)

    rgb = _boost_sat_center(rgb, RGB_SAT_BOOST, dist3)

    if abs(RGB_BRIGHTNESS - 1.0) > 1e-4:
        rgb = np.clip(rgb * RGB_BRIGHTNESS, 0, 1)
    if RGB_SPEC_STR > 1e-4:
        scx = cx3 + RGB_SPEC_CX * cx3
        scy = cy3 + RGB_SPEC_CY * cy3
        spec_dist = np.sqrt(((yy3 - scy) / cy3) ** 2 + ((xx3 - scx) / cx3) ** 2)
        spec = np.exp(-spec_dist ** 2 / (2 * max(RGB_SPEC_SIZE, 1e-3) ** 2))
        rgb = np.clip(rgb + spec[:, :, None] * RGB_SPEC_STR, 0, 1)
    if abs(RGB_EDGE_DARK) > 1e-4:
        edge_mask = np.clip(dist3 ** 3 * RGB_EDGE_DARK, 0, 0.5)[:, :, None]
        rgb = rgb * (1 - edge_mask)
    if abs(RGB_GEL_GRAD_STR) > 1e-4:
        angle_rad = RGB_GEL_GRAD_ANG * np.pi / 180.0
        grad = ((xx3 - cx3) / cx3 * np.cos(angle_rad) +
                (yy3 - cy3) / cy3 * np.sin(angle_rad))
        grad_mask = (1.0 + grad * RGB_GEL_GRAD_STR * 0.3)[:, :, None]
        rgb = np.clip(rgb * grad_mask, 0, 1)

    mask = np.clip(1.0 - dist3 ** 2 * RGB_VIGNETTE, 0, 1)[:, :, None]
    rgb = np.clip(rgb * mask, 0, 1)

    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * RGB_WB_R, 0, 1)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] * RGB_WB_G, 0, 1)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * RGB_WB_B, 0, 1)

    out_img = bpy.data.images.new('_gel_tmp', W, H, alpha=False)
    out_px = np.ones((H, W, 4), dtype=np.float32)
    out_px[:, :, :3] = rgb
    out_img.pixels.foreach_set(np.ascontiguousarray(out_px.reshape(-1)))
    out_img.filepath_raw = png_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)


# ── Platform setup ──
_platform_objs_global = []
_platform_initial_mats = []


def _setup_platform():
    global _platform_objs_global, _platform_initial_mats
    plat_path = os.path.join(dir, 'meshes', '202000 6152_200.obj')
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=plat_path, directory=os.path.dirname(plat_path),
                          files=[{'name': os.path.basename(plat_path)}])
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    teal_mat = bpy.data.materials.new('platform_teal_rgb')
    teal_mat.use_nodes = True
    p_bsdf = teal_mat.node_tree.nodes.get('Principled BSDF')
    if p_bsdf:
        p_bsdf.inputs['Base Color'].default_value = (
            _rgb_bo.get('plat_r', 26 / 255),
            _rgb_bo.get('plat_g', 115 / 255),
            _rgb_bo.get('plat_b', 106 / 255), 1.0)
        p_bsdf.inputs['Roughness'].default_value = _rgb_bo.get('plat_roughness', 0.25)
        p_bsdf.inputs['Specular IOR Level'].default_value = 0.6
        p_bsdf.inputs['Metallic'].default_value = _rgb_bo.get('plat_metallic', 0.85)
    for po in new_objs:
        po.data.materials.clear()
        po.data.materials.append(teal_mat)
        po.scale = (0.001, -0.001, 0.001)
        po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
        po.location = (-0.08, 0.055, 0.04)
        po.hide_render = True
    bpy.context.view_layer.update()
    _platform_objs_global = new_objs
    _platform_initial_mats = [po.matrix_world.copy() for po in new_objs]
    print(f'Platform ready: {[o.name for o in new_objs]}')


def _set_platform_rotation(rz):
    rot = Euler((0.0, 0.0, rz)).to_matrix().to_4x4()
    for po, m0 in zip(_platform_objs_global, _platform_initial_mats):
        po.matrix_world = rot @ m0
    bpy.context.view_layer.update()


def _rgb_cam_z_for(obj_name, base_rotation, fixed_scale):
    o = bpy.data.objects[obj_name]
    saved_rot = tuple(o.rotation_euler)
    saved_loc = tuple(o.location)
    o.scale = (1 / fixed_scale, 1 / fixed_scale, 1 / fixed_scale)
    o.rotation_euler = (base_rotation[0], base_rotation[1], 0.0)
    o.location = (0, 0, 0)
    bpy.context.view_layer.update()
    xs = [(o.matrix_world @ v.co).x for v in o.data.vertices]
    ys = [(o.matrix_world @ v.co).y for v in o.data.vertices]
    base_size = max(max(xs) - min(xs), max(ys) - min(ys))
    o.rotation_euler = saved_rot
    o.location = saved_loc
    bpy.context.view_layer.update()
    return RGB_CAM_Z * (base_size / RGB_BASE_SIZE_REF), base_size


def render_rgb_sample(obj_name, fov_deg, out_path, cam_z=None):
    cam = bpy.data.objects['Camera']
    cam_data = cam.data
    orig_cam_z = cam.location[2]
    orig_fov = cam_data.angle
    orig_dof = cam_data.dof.use_dof

    cam.location[2] = cam_z if cam_z is not None else RGB_CAM_Z
    cam_data.angle = math.radians(fov_deg)
    cam_data.dof.use_dof = False

    world = bpy.data.worlds.get('World')
    orig_bg = None
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            orig_bg = (tuple(bg_node.inputs['Color'].default_value),
                       bg_node.inputs['Strength'].default_value)
            bg_node.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1.0)
            bg_node.inputs['Strength'].default_value = RGB_WORLD_STR

    vis_save = {}
    for name in RGB_HIDE_NAMES:
        if name in bpy.data.objects:
            vis_save[name] = bpy.data.objects[name].hide_render
            bpy.data.objects[name].hide_render = True
    bpy.data.objects[obj_name].hide_render = False
    for _po in _platform_objs_global:
        _po.hide_render = False

    scene = bpy.context.scene
    orig_nodes = scene.use_nodes
    orig_trans = scene.render.film_transparent
    orig_mode = scene.render.image_settings.color_mode
    orig_fp = scene.render.filepath

    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = out_path
    scene.frame_set(0)
    bpy.ops.render.render(write_still=True)

    _gel_fx(out_path + '.png')

    scene.use_nodes = orig_nodes
    scene.render.film_transparent = orig_trans
    scene.render.image_settings.color_mode = orig_mode
    scene.render.filepath = orig_fp
    bpy.data.objects[obj_name].hide_render = True
    for _po in _platform_objs_global:
        _po.hide_render = True
    for name, vis in vis_save.items():
        bpy.data.objects[name].hide_render = vis

    cam.location[2] = orig_cam_z
    cam_data.angle = orig_fov
    cam_data.dof.use_dof = orig_dof
    if orig_bg is not None and world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = orig_bg[0]
            bg_node.inputs['Strength'].default_value = orig_bg[1]


# ── Main ──
def main():
    render_dir = os.environ.get('GELSIGHT_RENDER_DIR')
    if not render_dir:
        print('ERROR: GELSIGHT_RENDER_DIR not set')
        sys.exit(1)

    session_path = os.path.join(render_dir, 'session.json')
    with open(session_path) as f:
        session = json.load(f)

    obj_stem = session['obj']
    fixed_rotation = tuple(session.get('base_rotation', session.get('fixed_rotation')))
    fixed_scale = session['fixed_scale']
    n_samples = len(session.get('valid_cells', []))
    if n_samples == 0:
        n_samples = session.get('NUM_OBJ_SAMPLES', 0)

    print(f'Re-render RGB only: {obj_stem}')
    print(f'  session: {render_dir}')
    print(f'  samples: {n_samples}')
    print(f'  scale: {fixed_scale}')

    # Determine object color
    color_override = os.environ.get('GELSIGHT_OBJ_COLOR', '').lower()
    if color_override == 'black':
        obj_color = OBJ_BLACK_COLOR
    elif color_override == 'blue':
        obj_color = OBJ_BLUE_COLOR
    elif obj_stem in BLACK_OBJS:
        obj_color = OBJ_BLACK_COLOR
    else:
        obj_color = OBJ_BLUE_COLOR
    print(f'  color: {"BLACK" if obj_color == OBJ_BLACK_COLOR else "BLUE"} '
          f'{obj_color[:3]}')

    # Import mesh
    mesh_path = os.path.join(dir, 'meshes', f'{obj_stem}.obj')
    _before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=mesh_path, directory=os.path.dirname(mesh_path),
                          files=[{'name': os.path.basename(mesh_path)}])
    _new = [o.name for o in bpy.data.objects if o.name not in _before]
    obj_name = _new[0] if _new else obj_stem

    # Apply material
    solid_mat = bpy.data.materials.new(f'solid_{obj_name}')
    solid_mat.use_nodes = True
    bsdf = solid_mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = obj_color
    bsdf.inputs['Roughness'].default_value = _rgb_bo.get('obj_roughness', 0.25)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0
    obj_bl = bpy.data.objects[obj_name]
    obj_bl.data.materials.clear()
    obj_bl.data.materials.append(solid_mat)
    obj_bl.hide_render = True

    # Setup platform
    _setup_platform()

    # Compute high camera z
    rgb_cam_z_high, _rgb_bsz = _rgb_cam_z_for(obj_name, fixed_rotation, fixed_scale)
    print(f'  RGB cam_z={rgb_cam_z_high * 1000:.1f}mm')

    sensor_dir = os.path.join(render_dir, 'sensor_0000')
    rgb_dir = os.path.join(sensor_dir, 'rgb')
    os.makedirs(rgb_dir, exist_ok=True)

    t0 = time.time()
    rendered = 0

    for idx in range(n_samples):
        idx_str = f'{idx:04d}'
        pose_path = os.path.join(sensor_dir, 'raw_data', f'{idx_str}_pose.json')
        if not os.path.exists(pose_path):
            print(f'  SKIP {idx_str}: no pose.json')
            continue

        with open(pose_path) as f:
            pose = json.load(f)

        # Restore exact object transform from saved pose
        obj_bl.location = pose['location']
        obj_bl.rotation_euler = pose['rotation_euler']
        obj_bl.scale = pose['scale']
        bpy.context.view_layer.update()

        # Platform rotation follows object rz
        _set_platform_rotation(pose['rotation_euler'][2])

        # Render RGB
        render_rgb_sample(obj_name, RGB_FOV,
                          os.path.join(rgb_dir, idx_str),
                          cam_z=rgb_cam_z_high)
        rendered += 1

        if rendered % 50 == 0 or rendered == n_samples:
            elapsed = time.time() - t0
            rate = rendered / elapsed if elapsed > 0 else 0
            print(f'  [{rendered}/{n_samples}]  {rate:.1f} samples/s  '
                  f'{elapsed:.0f}s', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone: {rendered} RGB images re-rendered in {elapsed:.0f}s')

    # Cleanup
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[obj_name].select_set(True)
    bpy.ops.object.delete()

    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        sys.exit(0)


if __name__ == '__main__':
    main()
