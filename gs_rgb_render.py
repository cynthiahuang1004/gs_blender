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
print(f'[gs_rgb_render] saved → {OUT_PATH}')
import os as _os; _os._exit(0)
