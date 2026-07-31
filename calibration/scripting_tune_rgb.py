"""
scripting_tune_rgb.py
=====================
Blender script for tuning RGB renders. Renders RAW rgb (no post FX) from
both cameras; post-processing is applied by tune_rgb.py outside Blender.

Env vars:
    GELSIGHT_FIXED_PARAMS  - JSON with scene params (world_strength, obj_roughness,
                             plat_r/g/b/roughness/metallic, rgb_fov)
    GELSIGHT_BG_RENDER     - output base path (writes <base>_high.png, <base>_contact.png)
    TUNE_OBJ_FILE          - mesh .obj path
    TUNE_OBJ_ROTATION      - JSON [rx, ry, rz]
    TUNE_OBJ_SCALE         - fixed_scale (>1 means use 1/scale)
    TUNE_OBJ_LOCATION      - JSON [x, y, z] world placement
"""

import bpy, os, sys, json, math, functools
from math import pi, tan

print = functools.partial(print, flush=True)

PARAMS_PATH = os.environ.get('GELSIGHT_FIXED_PARAMS', '')
RENDER_PATH = os.environ.get('GELSIGHT_BG_RENDER', '')
OBJ_FILE = os.environ.get('TUNE_OBJ_FILE', '')
OBJ_ROTATION = json.loads(os.environ.get('TUNE_OBJ_ROTATION', '[0,0,0]'))
OBJ_SCALE = float(os.environ.get('TUNE_OBJ_SCALE', '1.0'))
OBJ_LOCATION = json.loads(os.environ.get('TUNE_OBJ_LOCATION', '[0,0,0]'))

with open(PARAMS_PATH) as f:
    P = json.load(f)

RGB_CAM_Z = -0.085
# contact camera = tactile camera height (fov 60, length 0.008751)
TACTILE_CAM_Z = -(0.008751 / tan((60.0 / 360) * pi))
RGB_FOV = float(P.get('rgb_fov', 40.0))

HIDE_NAMES = ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
              'LightSurfaceBL', 'LightSurfaceTR',
              'LightSurfaceTL', 'LightSurfaceBR',
              'LightSurfaceRGreen', 'LightSurfaceLGreen',
              'IndenterSurface']

QUAD_BASE_COLOR = (0.45, 0.45, 0.45, 1.0)
QUAD_COLORS = [
    (0.72, 0.66, 0.28, 1.0),
    (0.30, 0.40, 0.66, 1.0),
    (0.62, 0.26, 0.26, 1.0),
    (0.10, 0.10, 0.10, 1.0),
]


def create_multicolor_material(name, seed, cross_center, obj_rotation, split_axes):
    """Same shader as scripting.py _create_multicolor_material."""
    import random as _rnd
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = float(P.get('obj_roughness', 0.25))
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0
    geom = nodes.new('ShaderNodeNewGeometry')
    sub_cc = nodes.new('ShaderNodeVectorMath')
    sub_cc.operation = 'SUBTRACT'
    sub_cc.inputs[1].default_value = cross_center
    links.new(geom.outputs['Position'], sub_cc.inputs[0])
    inv_rot = nodes.new('ShaderNodeVectorRotate')
    inv_rot.rotation_type = 'EULER_XYZ'
    inv_rot.invert = True
    inv_rot.inputs['Rotation'].default_value = obj_rotation
    links.new(sub_cc.outputs['Vector'], inv_rot.inputs['Vector'])
    sep_obj = nodes.new('ShaderNodeSeparateXYZ')
    links.new(inv_rot.outputs['Vector'], sep_obj.inputs['Vector'])
    sep_pos = nodes.new('ShaderNodeSeparateXYZ')
    links.new(geom.outputs['Position'], sep_pos.inputs['Vector'])
    axis_names = ['X', 'Y', 'Z']
    gtx = nodes.new('ShaderNodeMath')
    gtx.operation = 'GREATER_THAN'
    gtx.inputs[1].default_value = 0.0
    links.new(sep_obj.outputs[axis_names[split_axes[0]]], gtx.inputs[0])
    gty = nodes.new('ShaderNodeMath')
    gty.operation = 'GREATER_THAN'
    gty.inputs[1].default_value = 0.0
    links.new(sep_obj.outputs[axis_names[split_axes[1]]], gty.inputs[0])
    rng = _rnd.Random(seed)
    colors = list(QUAD_COLORS)
    rng.shuffle(colors)
    tl, tr, bl, br = colors

    def _mix(fac_out, col_a, col_b):
        m = nodes.new('ShaderNodeMix')
        m.data_type = 'RGBA'
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

    top_mix = _mix(gtx.outputs['Value'], tl, tr)
    bot_mix = _mix(gtx.outputs['Value'], bl, br)
    quad_mix = _mix(gty.outputs['Value'], bot_mix.outputs[2], top_mix.outputs[2])
    hcmp = nodes.new('ShaderNodeMath')
    hcmp.operation = 'GREATER_THAN'
    hcmp.inputs[1].default_value = 0.0
    links.new(sep_pos.outputs['Z'], hcmp.inputs[0])
    final = _mix(hcmp.outputs['Value'], quad_mix.outputs[2], QUAD_BASE_COLOR)
    links.new(final.outputs[2], bsdf.inputs['Base Color'])
    return mat


# ── Hide gel/lights, set world background ──
for name in HIDE_NAMES:
    if name in bpy.data.objects:
        bpy.data.objects[name].hide_render = True

world = bpy.data.worlds.get('World')
if world and world.node_tree:
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1.0)
        bg_node.inputs['Strength'].default_value = float(P.get('world_strength', 2.37))

# ── Platform ──
plat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'meshes', '202000 6152_200.obj')
before = set(bpy.data.objects.keys())
bpy.ops.wm.obj_import(filepath=plat_path, directory=os.path.dirname(plat_path),
                      files=[{'name': os.path.basename(plat_path)}])
plat_objs = [o for o in bpy.data.objects if o.name not in before]
teal_mat = bpy.data.materials.new('platform_teal_tune')
teal_mat.use_nodes = True
p_bsdf = teal_mat.node_tree.nodes.get('Principled BSDF')
if p_bsdf:
    p_bsdf.inputs['Base Color'].default_value = (
        float(P.get('plat_r', 0.226)), float(P.get('plat_g', 0.35)),
        float(P.get('plat_b', 0.40)), 1.0)
    p_bsdf.inputs['Roughness'].default_value = float(P.get('plat_roughness', 0.35))
    p_bsdf.inputs['Specular IOR Level'].default_value = 0.6
    p_bsdf.inputs['Metallic'].default_value = float(P.get('plat_metallic', 0.582))
for po in plat_objs:
    po.data.materials.clear()
    po.data.materials.append(teal_mat)
    po.scale = (0.001, -0.001, 0.001)
    po.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
    po.location = (-0.08, 0.055, 0.04)
    po.hide_render = False
bpy.context.view_layer.update()
# Platform follows the object's z-rotation (about world origin = press point)
from mathutils import Euler
_rz = OBJ_ROTATION[2]
_rot = Euler((0.0, 0.0, _rz)).to_matrix().to_4x4()
for po in plat_objs:
    po.matrix_world = _rot @ po.matrix_world
bpy.context.view_layer.update()
print(f'[tune] platform ready (rz={_rz:.3f})')

# ── Import + place object ──
before = set(bpy.data.objects.keys())
bpy.ops.wm.obj_import(filepath=OBJ_FILE, directory=os.path.dirname(OBJ_FILE),
                      files=[{'name': os.path.basename(OBJ_FILE)}])
new_objs = [o.name for o in bpy.data.objects if o.name not in before]
mesh_objs = [n for n in new_objs if bpy.data.objects[n].type == 'MESH']
obj = bpy.data.objects[mesh_objs[0]]
obj.rotation_euler = OBJ_ROTATION
s = 1.0 / OBJ_SCALE  # dataset convention: obj.scale = 1 / fixed_scale (always)
obj.scale = (s, s, s)
obj.location = tuple(OBJ_LOCATION)
bpy.context.view_layer.update()
print(f'[tune] object at ({OBJ_LOCATION[0]*1000:.2f}, {OBJ_LOCATION[1]*1000:.2f}, {OBJ_LOCATION[2]*1000:.3f})mm')

# ── Solid color material (same rule as dataset: blue default, black list) ──
BLACK_OBJS = {
    'pattern_04_3_lines_angle_2', 'pattern_06_5_lines_angle_1',
    'pattern_31_rod', 'pattern_32', 'pattern_35', 'pattern_36', 'pattern_37',
}
stem = os.path.basename(OBJ_FILE)[:-4]
if stem in BLACK_OBJS:
    obj_color = (0.02, 0.02, 0.02, 1.0)
else:
    obj_color = (float(P.get('obj_r', 0.03)), float(P.get('obj_g', 0.08)),
                 float(P.get('obj_b', 0.40)), 1.0)
mat = bpy.data.materials.new('solid_tune')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = obj_color
bsdf.inputs['Roughness'].default_value = float(P.get('obj_roughness', 0.25))
bsdf.inputs['Specular IOR Level'].default_value = 0.5
bsdf.inputs['Metallic'].default_value = 0.0
obj.data.materials.clear()
obj.data.materials.append(mat)
obj.hide_render = False
print(f'[tune] solid color {obj_color[:3]} ({"black" if stem in BLACK_OBJS else "blue"})')

# ── Normalize apparent base size: camera distance ∝ base plate size ──
# Measure XY extent with rz=0 (rotation-invariant base size)
_saved_rot = tuple(obj.rotation_euler)
obj.rotation_euler = (OBJ_ROTATION[0], OBJ_ROTATION[1], 0.0)
bpy.context.view_layer.update()
_xs = [ (obj.matrix_world @ v.co).x for v in obj.data.vertices ]
_ys = [ (obj.matrix_world @ v.co).y for v in obj.data.vertices ]
base_size = max(max(_xs) - min(_xs), max(_ys) - min(_ys))
obj.rotation_euler = _saved_rot
bpy.context.view_layer.update()
cam_z_high = RGB_CAM_Z * (base_size / 0.082)
print(f'[tune] base_size={base_size*1000:.1f}mm -> high cam_z={cam_z_high*1000:.1f}mm')

# ── Render (raw, no post FX) ──
cam = bpy.data.objects['Camera']
cam_data = cam.data
cam_data.dof.use_dof = False
cam_data.angle = math.radians(RGB_FOV)

scene = bpy.context.scene
scene.use_nodes = False
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'

for tag, cz in [('high', cam_z_high), ('contact', TACTILE_CAM_Z)]:
    cam.location[2] = cz
    scene.render.filepath = RENDER_PATH + '_' + tag
    scene.frame_set(0)
    bpy.ops.render.render(write_still=True)
    print(f'[tune] saved: {RENDER_PATH}_{tag}.png (cam_z={cz*1000:.2f}mm)')

sys.stdout.flush()
os._exit(0)
