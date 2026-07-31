"""
scripting_tune_tactile.py
=========================
Blender script for tuning tactile contact appearance.
Loads an object, presses it into the gel, and renders a tactile image.

Env vars:
    GELSIGHT_FIXED_PARAMS  - JSON with all sensor params
    GELSIGHT_BG_RENDER     - output path (without .png)
    TUNE_OBJ_FILE          - mesh .obj file path
    TUNE_PRESS_DEPTH       - press depth in meters (default 0.001)
    TUNE_OBJ_ROTATION      - JSON list [rx, ry, rz] in radians
    TUNE_OBJ_SCALE         - uniform scale factor
"""

import bpy, os, sys, json, math, functools
from math import pi, tan
from mathutils import Euler

# os._exit() at the end skips stdout flushing — force flush on every print
print = functools.partial(print, flush=True)

PARAMS_PATH = os.environ.get('GELSIGHT_FIXED_PARAMS', '')
RENDER_PATH = os.environ.get('GELSIGHT_BG_RENDER', '')
OBJ_FILE = os.environ.get('TUNE_OBJ_FILE', '')
PRESS_DEPTH = float(os.environ.get('TUNE_PRESS_DEPTH', '0.001'))
OBJ_ROTATION = json.loads(os.environ.get('TUNE_OBJ_ROTATION', '[0,0,0]'))
OBJ_SCALE = float(os.environ.get('TUNE_OBJ_SCALE', '1.0'))
OBJ_OFFSET = json.loads(os.environ.get('TUNE_OBJ_OFFSET', '[0,0]'))
Z_ANCHOR = float(os.environ.get('TUNE_Z_ANCHOR', '0'))
# Exact world location [x,y,z] from a pose.json — overrides offset/z_anchor placement
OBJ_LOCATION = json.loads(os.environ.get('TUNE_OBJ_LOCATION', 'null'))

if not PARAMS_PATH or not RENDER_PATH:
    print('ERROR: GELSIGHT_FIXED_PARAMS and GELSIGHT_BG_RENDER must be set.')
    os._exit(1)

with open(PARAMS_PATH) as f:
    P = json.load(f)


def set_emittor(name, strength, color):
    node = bpy.data.materials[name].node_tree.nodes['Emission']
    node.inputs['Color'].default_value = color
    node.inputs['Strength'].default_value = strength


def set_smoothness(val):
    gs = bpy.data.objects['GelSurface']
    gs.modifiers['CorrectiveSmooth'].iterations = val
    sw = gs.modifiers['Shrinkwrap']
    sw.offset = 1e-3 * (val * val) / 50000
    sw.wrap_method = 'PROJECT'
    sw.use_project_z = True
    sw.use_negative_direction = True
    sw.use_positive_direction = False


def set_lights_transform(scale_y, light_z, rot_z, init_mats):
    rot_mat = Euler((0, 0, rot_z)).to_matrix().to_4x4()
    for name in ['LightSurfaceBL', 'LightSurfaceTR',
                 'LightSurfaceTL', 'LightSurfaceBR']:
        bpy.data.objects[name].matrix_world = rot_mat @ init_mats[name]
        bpy.data.objects[name].location[2] = light_z
        bpy.data.objects[name].scale[1] = scale_y


def set_cam(fov, length):
    height = length / tan((fov / 360) * pi)
    bpy.data.objects['Camera'].location[2] = -height
    bpy.data.objects['Camera'].data.angle = (fov / 180) * pi
    tree = bpy.data.scenes['Scene'].node_tree
    tree.nodes['Map Range'].inputs[2].default_value = height
    tree.nodes['Map Range'].inputs[1].default_value = height - 0.002


def set_green_light(obj_name, mat_name, strength, r, g, b):
    if obj_name in bpy.data.objects:
        bpy.data.objects[obj_name].hide_render = False
    if mat_name in bpy.data.materials:
        node = bpy.data.materials[mat_name].node_tree.nodes['Emission']
        node.inputs['Strength'].default_value = strength
        node.inputs['Color'].default_value = (r, g, b, 1.0)


def set_gel_material(roughness, fac, dark_base_gain, contact_color,
                     contact_z_thresh, bg_z_thresh):
    mat = bpy.data.materials['aluminum-specular-mat']
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    glossy = nodes.get('Glossy BSDF')
    if glossy:
        glossy.inputs['Roughness'].default_value = roughness
    mix_shader = nodes.get('Mix Shader')
    if mix_shader:
        mix_shader.inputs['Fac'].default_value = fac

    # Dark contact: mix dark color into glossy Color by world-Z (same as scripting.py)
    if glossy is not None and not glossy.inputs['Color'].is_linked:
        geom = nodes.new('ShaderNodeNewGeometry')
        sep = nodes.new('ShaderNodeSeparateXYZ')
        links.new(geom.outputs['Position'], sep.inputs[0])

        mr = nodes.new('ShaderNodeMapRange')
        mr.clamp = True
        mr.interpolation_type = 'SMOOTHSTEP'
        mr.inputs['From Min'].default_value = contact_z_thresh
        mr.inputs['From Max'].default_value = bg_z_thresh
        mr.inputs['To Min'].default_value = 0.0
        mr.inputs['To Max'].default_value = 1.0
        links.new(sep.outputs['Z'], mr.inputs['Value'])

        mix = nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        mix.inputs[6].default_value = contact_color
        mix.inputs[7].default_value = (dark_base_gain, dark_base_gain, dark_base_gain, 1.0)
        links.new(mr.outputs['Result'], mix.inputs[0])
        links.new(mix.outputs[2], glossy.inputs['Color'])


def find_lowest(obj_name):
    obj = bpy.data.objects[obj_name]
    bpy.context.view_layer.update()
    mw = obj.matrix_world
    lowest_z = None
    for v in obj.data.vertices:
        wc = mw @ v.co
        if lowest_z is None or wc.z < lowest_z:
            lowest_z = wc.z
    return lowest_z if lowest_z is not None else 0.0


# ── Save light initial matrices ──
light_names = ['LightSurfaceBL', 'LightSurfaceTR',
               'LightSurfaceTL', 'LightSurfaceBR']
_init_matrices = {n: bpy.data.objects[n].matrix_world.copy() for n in light_names}

# ── Apply sensor parameters ──
set_smoothness(int(P.get('smoothness', 30)))
set_lights_transform(
    scale_y=float(P.get('scale_y', 0.4918)),
    light_z=float(P.get('light_z', -0.004139)),
    rot_z=float(P.get('rot_z', -3.14159)),
    init_mats=_init_matrices)
set_cam(float(P.get('fov', 60.0)), float(P.get('length', 0.008751)))

# Contact appearance params
cr = float(P.get('contact_r', 60.0 / 255.0))
cg = float(P.get('contact_g', 60.0 / 255.0))
cb = float(P.get('contact_b', 60.0 / 255.0))
set_gel_material(
    roughness=float(P.get('gel_roughness', 0.4455)),
    fac=float(P.get('gel_fac', 0.2971)),
    dark_base_gain=float(P.get('dark_base_gain', 0.9)),
    contact_color=(cr, cg, cb, 1.0),
    contact_z_thresh=float(P.get('contact_z_thresh', -0.001)),
    bg_z_thresh=float(P.get('bg_z_thresh', 0.0005)),
)

# All 6 emittors
set_emittor('BLEmittor',
    float(P.get('top_str', 80.0)),
    (float(P.get('top_r', 0.3)), float(P.get('top_g', 0.65)), float(P.get('top_b', 0.3)), 1.0))
set_emittor('TREmittor',
    float(P.get('bot_str', 40.0)),
    (float(P.get('bot_r', 0.1)), float(P.get('bot_g', 0.5)), float(P.get('bot_b', 0.9)), 1.0))
set_emittor('TLEmittor',
    float(P.get('left_str', 30.0)),
    (float(P.get('left_r', 0.9)), float(P.get('left_g', 0.05)), float(P.get('left_b', 0.05)), 1.0))
set_emittor('BREmittor',
    float(P.get('right_str', 120.0)),
    (float(P.get('right_r', 1.0)), float(P.get('right_g', 0.0)), float(P.get('right_b', 0.0)), 1.0))
set_green_light('LightSurfaceRGreen', 'RGreenEmittor',
    float(P.get('lg_str', 60.0)),
    float(P.get('lg_r', 0.3)), float(P.get('lg_g', 0.65)), float(P.get('lg_b', 0.3)))
set_green_light('LightSurfaceLGreen', 'LGreenEmittor',
    float(P.get('rg_str', 120.0)),
    float(P.get('rg_r', 0.3)), float(P.get('rg_g', 0.7)), float(P.get('rg_b', 0.3)))

# ── Import and place object ──
if OBJ_FILE and os.path.exists(OBJ_FILE):
    before = set(bpy.data.objects.keys())
    print(f'[tune] Importing {OBJ_FILE}...')
    bpy.ops.wm.obj_import(filepath=OBJ_FILE,
                          directory=os.path.dirname(OBJ_FILE),
                          files=[{'name': os.path.basename(OBJ_FILE)}])
    new_objs = [o.name for o in bpy.data.objects if o.name not in before]
    # Find the mesh object (skip empties)
    mesh_objs = [n for n in new_objs if bpy.data.objects[n].type == 'MESH']
    if not mesh_objs:
        # Check children of empty
        for n in new_objs:
            for child in bpy.data.objects[n].children:
                if child.type == 'MESH':
                    mesh_objs.append(child.name)
    print(f'[tune] new_objs={new_objs}, mesh_objs={mesh_objs}, OBJ_SCALE={OBJ_SCALE}')

    if mesh_objs:
        imported_name = mesh_objs[0]
        obj = bpy.data.objects[imported_name]
        obj.rotation_euler = OBJ_ROTATION
        s = 1.0 / OBJ_SCALE  # dataset convention: obj.scale = 1 / fixed_scale (always)
        obj.scale = (s, s, s)
        bpy.context.view_layer.update()

        if OBJ_LOCATION is not None:
            # Exact placement from a pose.json (fully reproduces a dataset sample)
            obj.location = tuple(OBJ_LOCATION)
        else:
            z_anchor = Z_ANCHOR if Z_ANCHOR != 0 else find_lowest(imported_name)
            cx, cy = OBJ_OFFSET[0], OBJ_OFFSET[1]
            obj.location = (-cx, -cy, -PRESS_DEPTH - z_anchor)
        print(f'[tune] loc=({obj.location[0]*1000:.2f}, {obj.location[1]*1000:.2f}, {obj.location[2]*1000:.3f})mm, scale={s}')
        bpy.context.view_layer.update()
        lowest = find_lowest(imported_name)
        print(f'[tune] lowest vertex z = {lowest*1000:.3f}mm (should be ~-1mm when pressed)')

        obj.hide_render = True
        bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].target = obj
        bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].show_render = True
        print(f'[tune] Loaded {imported_name}, press_depth={PRESS_DEPTH*1000:.1f}mm')
else:
    print('[tune] No object file, rendering flat gel')
    indenter = bpy.data.objects['IndenterSurface']
    indenter.location = (0, 0, -1)
    indenter.hide_render = True
    bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].show_render = False

# ── Render ──
bpy.context.scene.render.filepath = RENDER_PATH
bpy.context.scene.frame_set(0)
bpy.ops.render.render(write_still=True)
print(f'[tune] saved: {RENDER_PATH}.png')

# ── GT depth (object only, gel hidden) — same logic as scripting.py get_gt_depth ──
if os.environ.get('TUNE_GT_DEPTH', '') and OBJ_FILE and mesh_objs:
    import glob
    import numpy as np

    # GT depth can use a DEEPER press than the tactile render (visual: real tactile
    # looks thin but real GT depth is thick). Shift the object down by the difference.
    GT_PRESS = float(os.environ.get('TUNE_GT_PRESS_DEPTH', str(PRESS_DEPTH)))
    if abs(GT_PRESS - PRESS_DEPTH) > 1e-9:
        obj.location = (obj.location[0], obj.location[1],
                        obj.location[2] - (GT_PRESS - PRESS_DEPTH))
        bpy.context.view_layer.update()
        print(f'[tune] GT depth press = {GT_PRESS*1000:.1f}mm (tactile was {PRESS_DEPTH*1000:.1f}mm)')

    for name in ['GelSurface', 'InterfaceSurface', 'EpoxySurface']:
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = True
    obj.hide_render = False

    scene = bpy.context.scene
    tree = scene.node_tree
    rl_node = next(n for n in tree.nodes if n.type == 'R_LAYERS')
    fo_node = tree.nodes.new('CompositorNodeOutputFile')
    fo_node.format.file_format = 'OPEN_EXR'
    fo_node.format.color_depth = '32'
    # 每個 run 用獨立子目錄放 EXR，避免平行執行時 glob 撿到別的 process 的檔案
    _gt_tmp_dir = RENDER_PATH + '_gttmp'
    os.makedirs(_gt_tmp_dir, exist_ok=True)
    fo_node.base_path = _gt_tmp_dir
    fo_node.file_slots[0].path = 'gt_depth_tmp_'
    tree.links.new(rl_node.outputs['Depth'], fo_node.inputs[0])
    scene.use_nodes = True
    bpy.ops.render.render(write_still=False)

    exr_files = sorted(glob.glob(os.path.join(_gt_tmp_dir, 'gt_depth_tmp_*.exr')))
    exr_img = bpy.data.images.load(exr_files[-1])
    w, h = exr_img.size
    dmap = np.array(exr_img.pixels[:], dtype=np.float32).reshape(h, w, exr_img.channels)[:, :, 0]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    cam_z = abs(bpy.data.objects['Camera'].location[2])
    bg = dmap > (cam_z * 0.99)
    dmap[bg] = 0.0
    dmap = cam_z - dmap
    dmap[bg] = 0.0
    np.save(RENDER_PATH + '_gt.npy', dmap)
    for f in exr_files:
        os.remove(f)
    try:
        os.rmdir(_gt_tmp_dir)
    except OSError:
        pass
    print(f'[tune] saved GT depth: {RENDER_PATH}_gt.npy')

sys.stdout.flush()
os._exit(0)
