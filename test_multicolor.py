"""
test_multicolor.py
-------------------
Test: render one RGB sample per object with quadrant-based multi-color material.
Only the top 1mm of raised features is colored (4 quadrant colors).
Everything else (base plate, lower parts) is gray.
Each object is placed at center using its session.json parameters.

Run:
    blender --background gelsight_sampler.blend --python test_multicolor.py
"""

import bpy, os, sys, json, math, random, mathutils
import numpy as np
from mathutils import Matrix

SCRIPT_DIR    = os.path.dirname(os.path.abspath(bpy.data.filepath))
MESHES_DIR    = os.path.join(SCRIPT_DIR, 'meshes')
OUT_DIR       = os.path.join(SCRIPT_DIR, 'test_multicolor_out')
PLATFORM_PATH = os.path.join(MESHES_DIR, '202000 6152_200.obj')
RENDERS_ROOT  = os.path.join(SCRIPT_DIR, 'renders')

RGB_PARAMS_PATH = os.path.join(SCRIPT_DIR, 'bo_results', 'best_rgb_params.json')
with open(RGB_PARAMS_PATH) as f:
    _rgb = json.load(f)

# ── RGB post-FX parameters from best_rgb_params.json ──────────
RGB_BARREL_K1    = _rgb.get('barrel_k1',      0.07)
RGB_VIGNETTE     = _rgb.get('vignette',       0.25)
RGB_BLUR_SIGMA   = _rgb.get('blur_sigma',     1.25)
RGB_BLUR_FALLOFF = 1.5
RGB_TINT_R       = _rgb.get('tint_r',         0.85)
RGB_TINT_G       = _rgb.get('tint_g',         0.70)
RGB_TINT_B       = _rgb.get('tint_b',         0.25)
RGB_TINT_STR     = _rgb.get('tint_strength',  0.25)
RGB_TINT_CX      = _rgb.get('tint_cx',        0.0)
RGB_TINT_CY      = _rgb.get('tint_cy',        0.0)
RGB_SAT_BOOST    = _rgb.get('sat_boost',       1.2)
RGB_HAZE_OPACITY = _rgb.get('haze_opacity',   0.10)
RGB_WORLD_STR    = _rgb.get('world_strength',  2.0)


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
    px = np.array(img_bpy.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img_bpy)
    rgb = _barrel(px[:,:,:3], RGB_BARREL_K1)
    blurred = _gaussian_blur_ch(rgb, RGB_BLUR_SIGMA)
    cy3, cx3 = H / 2.0, W / 2.0
    yy3, xx3 = np.mgrid[0:H, 0:W]
    tcx = cx3 + RGB_TINT_CX * cx3
    tcy = cy3 + RGB_TINT_CY * cy3
    dist3 = np.sqrt(((yy3-tcy)/cy3)**2 + ((xx3-tcx)/cx3)**2)
    weight = np.clip(dist3 ** RGB_BLUR_FALLOFF, 0, 1)[:,:,None]
    rgb = rgb * (1 - weight) + blurred * weight
    tint_color = np.array([[[RGB_TINT_R, RGB_TINT_G, RGB_TINT_B]]], dtype=np.float32)
    tint_w = np.clip(dist3 ** 2 * RGB_TINT_STR, 0, 1)[:,:,None]
    rgb = rgb * (1 - tint_w) + tint_color * tint_w
    haze_w = np.clip(dist3 ** 1.5 * RGB_HAZE_OPACITY, 0, 1)[:,:,None]
    rgb = rgb * (1 - haze_w) + tint_color * 0.7 * haze_w
    rgb = _boost_sat_center(rgb, RGB_SAT_BOOST, dist3)
    mask = np.clip(1.0 - dist3**2 * RGB_VIGNETTE, 0, 1)[:,:,None]
    rgb = np.clip(rgb * mask, 0, 1)
    out_img = bpy.data.images.new('_gel_tmp', W, H, alpha=False)
    out_px = np.ones((H, W, 4), dtype=np.float32); out_px[:,:,:3] = rgb
    out_img.pixels = out_px.flatten().tolist()
    out_img.filepath_raw = png_path
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)


QUAD_BASE_COLOR = (0.45, 0.45, 0.45, 1.0)  # gray for base
QUAD_FEATURE_LOW = True
COLOR_DEPTH_MM = 1.0  # only top 1mm colored

QUAD_COLORS = [
    (0.72, 0.66, 0.28, 1.0),  # yellow
    (0.30, 0.40, 0.66, 1.0),  # blue
    (0.62, 0.26, 0.26, 1.0),  # red
    (0.10, 0.10, 0.10, 1.0),  # black
]

CAM_Z = -0.085
HIDE_NAMES = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
              'LightSurfaceBL', 'LightSurfaceTR',
              'LightSurfaceTL', 'LightSurfaceBR',
              'LightSurfaceRGreen', 'LightSurfaceLGreen']

MESH_NAMES = [f[:-4] for f in sorted(os.listdir(MESHES_DIR))
              if f.endswith('.obj') and '6152' not in f]

os.makedirs(OUT_DIR, exist_ok=True)


def _get_z_range(mesh_path):
    zs = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith('v '):
                zs.append(float(line.split()[3]))
    return (max(zs) - min(zs)) if zs else 1.0


def _load_session(mesh_name):
    path = os.path.join(RENDERS_ROOT, mesh_name, 'session_000', 'session.json')
    with open(path) as f:
        return json.load(f)


def create_multicolor_material(name, seed):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = _rgb.get('obj_roughness', 0.35)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0

    # Use world-space position for both quadrant and height splits
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-1000, 0)

    sep_pos = nodes.new('ShaderNodeSeparateXYZ')
    sep_pos.location = (-800, 0)
    links.new(geom.outputs['Position'], sep_pos.inputs['Vector'])

    # Quadrant split: world X > 0 = right, world Y > 0 = top
    # (object is centered at origin)
    gtx = nodes.new('ShaderNodeMath')
    gtx.operation = 'GREATER_THAN'
    gtx.location = (-600, 150)
    gtx.inputs[1].default_value = 0.0
    links.new(sep_pos.outputs['X'], gtx.inputs[0])

    gty = nodes.new('ShaderNodeMath')
    gty.operation = 'GREATER_THAN'
    gty.location = (-600, -50)
    gty.inputs[1].default_value = 0.0
    links.new(sep_pos.outputs['Y'], gty.inputs[0])

    rng = random.Random(seed)
    colors = list(QUAD_COLORS)
    rng.shuffle(colors)
    tl, tr, bl, br = colors

    def mix_rgba(fac_out, col_a, col_b, x=0, y=0):
        m = nodes.new('ShaderNodeMix')
        m.data_type = 'RGBA'
        m.location = (x, y)
        links.new(fac_out, m.inputs[0])
        if hasattr(col_a, 'node'):
            links.new(col_a, m.inputs[6])
        else:
            m.inputs[6].default_value = col_a
        if hasattr(col_b, 'node'):
            links.new(col_b, m.inputs[7])
        else:
            m.inputs[7].default_value = col_b
        return m

    top_mix = mix_rgba(gtx.outputs['Value'], tl, tr, x=-400, y=150)
    bot_mix = mix_rgba(gtx.outputs['Value'], bl, br, x=-400, y=-50)
    quad_mix = mix_rgba(gty.outputs['Value'],
                        bot_mix.outputs[2], top_mix.outputs[2], x=-200, y=50)

    # Height split using world-space Z:
    # Object tip is at Z = -press_depth (-0.001)
    # Z < 0 = within 1mm of sensor contact → colored
    # Z > 0 = base plate / upper parts → gray
    hcmp = nodes.new('ShaderNodeMath')
    hcmp.operation = 'GREATER_THAN'
    hcmp.location = (-400, 300)
    hcmp.inputs[1].default_value = 0.0  # Z > 0 = base → gray
    links.new(sep_pos.outputs['Z'], hcmp.inputs[0])

    # fac=1 (Z>0, base) → gray, fac=0 (Z<0, feature tip) → color
    final = mix_rgba(hcmp.outputs['Value'],
                     quad_mix.outputs[2], QUAD_BASE_COLOR, x=0, y=100)

    links.new(final.outputs[2], bsdf.inputs['Base Color'])
    return mat


def setup_platform():
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=PLATFORM_PATH,
                           directory=os.path.dirname(PLATFORM_PATH),
                           files=[{'name': os.path.basename(PLATFORM_PATH)}])
    platform_objs = [o for o in bpy.data.objects if o.name not in before]

    plat_mat = bpy.data.materials.new('platform_teal_test')
    plat_mat.use_nodes = True
    p_bsdf = plat_mat.node_tree.nodes.get('Principled BSDF')
    if p_bsdf:
        p_bsdf.inputs['Base Color'].default_value = (
            _rgb.get('plat_r', 0.226), _rgb.get('plat_g', 0.35),
            _rgb.get('plat_b', 0.40), 1.0)
        p_bsdf.inputs['Roughness'].default_value = _rgb.get('plat_roughness', 0.35)
        p_bsdf.inputs['Metallic'].default_value = _rgb.get('plat_metallic', 0.582)

    for po in platform_objs:
        po.data.materials.clear()
        po.data.materials.append(plat_mat)
        po.hide_render = False
        po.scale = (0.001, -0.001, 0.001)
        po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
        po.location = (-0.08, 0.055, 0.04)

    return platform_objs


def find_lowest(obj_name):
    """Find lowest Z vertex in world space."""
    obj = bpy.data.objects[obj_name]
    bpy.context.view_layer.update()
    mw = obj.matrix_world
    lowest_z = None
    lowest_co = None
    for v in obj.data.vertices:
        wc = mw @ v.co
        if lowest_z is None or wc.z < lowest_z:
            lowest_z = wc.z
            lowest_co = wc
    return lowest_co


def render_object(mesh_name, seed):
    mesh_path = os.path.join(MESHES_DIR, mesh_name + '.obj')
    session = _load_session(mesh_name)

    base_rotation = session['base_rotation']
    press_depth = 0.001  # 1mm

    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=mesh_path,
                           directory=os.path.dirname(mesh_path),
                           files=[{'name': os.path.basename(mesh_path)}])
    new_objs = [o.name for o in bpy.data.objects if o.name not in before]
    if not new_objs:
        print(f'  SKIP {mesh_name}: import failed')
        return
    imported_name = new_objs[0]
    obj = bpy.data.objects[imported_name]

    # Apply rotation first, then measure to compute scale for 82mm base
    obj.rotation_euler = base_rotation
    obj.location = (0, 0, 0)
    obj.scale = (1, 1, 1)
    bpy.context.view_layer.update()

    dims = obj.dimensions  # world-space dimensions after rotation
    max_base = max(dims[0], dims[1])  # XY plane = base
    target_size = 0.082  # 82mm in meters
    uniform_scale = target_size / max_base if max_base > 0 else 1.0
    obj.scale = (uniform_scale, uniform_scale, uniform_scale)
    bpy.context.view_layer.update()

    print(f'  dims_after_rot={[round(d*1000,1) for d in dims]}mm  scale={uniform_scale:.6f}')

    # Center the object using bounding box, then set tip at z = -press_depth
    bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    bb_min = mathutils.Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    bb_max = mathutils.Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    bb_center = (bb_min + bb_max) / 2

    # Center XY, position Z so lowest point is at -press_depth
    obj.location = (obj.location.x - bb_center.x,
                    obj.location.y - bb_center.y,
                    obj.location.z - bb_min.z - press_depth)
    bpy.context.view_layer.update()

    # Apply multicolor material
    mat = create_multicolor_material(f'multicolor_{mesh_name}', seed)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.hide_render = False

    # Render
    out_path = os.path.join(OUT_DIR, mesh_name)
    scene.render.filepath = out_path
    scene.frame_set(0)
    bpy.ops.render.render(write_still=True)

    # Apply gel post-FX (barrel, blur, tint, haze, sat_boost, vignette)
    _gel_fx(out_path + '.png')
    print(f'  Saved: {out_path}.png (with post-FX)')

    # Cleanup
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()


# ── Setup scene ───────────────────────────────────────────────
cam_obj = bpy.data.objects['Camera']
cam_data = cam_obj.data
cam_obj.location[2] = CAM_Z
cam_data.dof.use_dof = False

world = bpy.data.worlds.get('World')
if world and world.node_tree:
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1.0)
        bg_node.inputs['Strength'].default_value = RGB_WORLD_STR

for name in HIDE_NAMES:
    if name in bpy.data.objects:
        bpy.data.objects[name].hide_render = True

scene = bpy.context.scene
scene.use_nodes = False
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'

platform_objs = setup_platform()

# ── Render each object ────────────────────────────────────────
print(f'\nRendering {len(MESH_NAMES)} objects to {OUT_DIR}/')
for i, mesh_name in enumerate(MESH_NAMES):
    print(f'[{i+1}/{len(MESH_NAMES)}] {mesh_name}')
    render_object(mesh_name, seed=i * 42 + 7)

print(f'\nDone! {len(MESH_NAMES)} images saved to {OUT_DIR}/')

try:
    bpy.ops.wm.quit_blender()
except Exception:
    sys.exit(0)
