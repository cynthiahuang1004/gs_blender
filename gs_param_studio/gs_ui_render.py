"""
gs_ui_render.py — Blender render script for GelSight Studio UI.

Env vars
--------
GS_UI_MODE    : "bg" | "tactile"
GS_UI_PARAMS  : path to params JSON
GS_UI_OBJ     : path to OBJ file  (tactile mode, optional)
GS_UI_DEPTH   : press depth in mm (default 1.0)
GS_UI_OUT     : output path without .png extension
"""

import bpy, os, math, json
from math import pi, tan
from mathutils import Euler

MODE        = os.environ.get('GS_UI_MODE',   'bg')
PARAMS_PATH = os.environ.get('GS_UI_PARAMS', '')
OBJ_PATH    = os.environ.get('GS_UI_OBJ',    '')
DEPTH_MM    = float(os.environ.get('GS_UI_DEPTH', '1.0'))
OUT_PATH    = os.environ.get('GS_UI_OUT',    '/tmp/gs_ui_out')

with open(PARAMS_PATH) as f:
    P = json.load(f)

# ── Helpers ───────────────────────────────────────────────────

def set_emittor(name, strength, color):
    mat = bpy.data.materials.get(name)
    if mat:
        nd = mat.node_tree.nodes.get('Emission')
        if nd:
            nd.inputs['Color'].default_value    = color
            nd.inputs['Strength'].default_value = strength

def set_smoothness(val):
    gs = bpy.data.objects['GelSurface']
    gs.modifiers['CorrectiveSmooth'].iterations = int(val)
    sw = gs.modifiers['Shrinkwrap']
    sw.offset               = 1e-3 * (val * val) / 50000
    sw.wrap_method          = 'PROJECT'
    sw.use_project_z        = True
    sw.use_negative_direction = True
    sw.use_positive_direction = False

def set_cam(fov, length):
    height = length / tan((fov / 360) * pi)
    bpy.data.objects['Camera'].location[2] = -height
    bpy.data.objects['Camera'].data.angle  = (fov / 180) * pi
    tree = bpy.data.scenes['Scene'].node_tree
    tree.nodes['Map Range'].inputs[2].default_value = height
    tree.nodes['Map Range'].inputs[1].default_value = height - 0.002

def set_gel_material(roughness, fac):
    mat   = bpy.data.materials['aluminum-specular-mat']
    nodes = mat.node_tree.nodes
    glossy = nodes.get('Glossy BSDF')
    if glossy: glossy.inputs['Roughness'].default_value = roughness
    mix = nodes.get('Mix Shader')
    if mix: mix.inputs['Fac'].default_value = fac

# ── Apply sensor parameters ───────────────────────────────────

set_smoothness(int(float(P.get('smoothness', 30))))
set_cam(float(P.get('fov', 45.0)), float(P.get('length', 0.009)))
set_gel_material(float(P.get('gel_roughness', 0.4455)), float(P.get('gel_fac', 0.2971)))

# Lights: rotation + scale (same as scripting.py _apply_fixed + apply())
light_names = ['LightSurfaceBL', 'LightSurfaceTR', 'LightSurfaceTL', 'LightSurfaceBR']
init_mats   = {n: bpy.data.objects[n].matrix_world.copy() for n in light_names}
scale       = float(P.get('scale', 0.4918))
rot_z       = float(P.get('light_rot_z', -math.pi))
rot_mat     = Euler((0, 0, rot_z)).to_matrix().to_4x4()
co_z        = -0.0065 + 0.0011 / scale
for n in light_names:
    bpy.data.objects[n].matrix_world = rot_mat @ init_mats[n]
    bpy.data.objects[n].location[2]  = co_z
    bpy.data.objects[n].scale[1]     = scale

# Green lights
for obj_name, mat_name, sk, rk, gk, bk in [
    ('LightSurfaceRGreen', 'RGreenEmittor', 'lg_str', 'lg_r', 'lg_g', 'lg_b'),
    ('LightSurfaceLGreen', 'LGreenEmittor', 'rg_str', 'rg_r', 'rg_g', 'rg_b'),
]:
    o = bpy.data.objects.get(obj_name)
    if o: o.hide_render = False
    m = bpy.data.materials.get(mat_name)
    if m:
        nd = m.node_tree.nodes.get('Emission')
        if nd:
            nd.inputs['Strength'].default_value = float(P.get(sk, 60.))
            nd.inputs['Color'].default_value = (
                float(P.get(rk, .3)), float(P.get(gk, .65)), float(P.get(bk, .3)), 1.)

# 4 main emittors
for bl, sk, rk, gk, bk in [
    ('BLEmittor', 'top_str',   'top_r',   'top_g',   'top_b'),
    ('TREmittor', 'bot_str',   'bot_r',   'bot_g',   'bot_b'),
    ('TLEmittor', 'left_str',  'left_r',  'left_g',  'left_b'),
    ('BREmittor', 'right_str', 'right_r', 'right_g', 'right_b'),
]:
    set_emittor(bl, float(P.get(sk, 80.)),
                (float(P.get(rk, .5)), float(P.get(gk, .5)), float(P.get(bk, .5)), 1.))

bpy.context.scene.frame_set(0)

# ── Mode: bg ─────────────────────────────────────────────────
if MODE == 'bg':
    ind = bpy.data.objects['IndenterSurface']
    ind.location    = (0, 0, -1)
    ind.hide_render = True
    sw = bpy.data.objects['GelSurface'].modifiers['Shrinkwrap']
    sw.target       = ind
    sw.show_render  = False

# ── Mode: tactile ─────────────────────────────────────────────
elif MODE == 'tactile' and OBJ_PATH and os.path.exists(OBJ_PATH):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=OBJ_PATH,
                          directory=os.path.dirname(OBJ_PATH),
                          files=[{'name': os.path.basename(OBJ_PATH)}])
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    if not new_objs:
        print('ERROR: no objects imported'); import sys; sys.exit(1)

    obj = new_objs[0]
    obj.location       = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    bpy.context.scene.frame_set(0)

    # Auto-scale: if dimensions > 0.1 m (likely mm units), scale by 0.001
    dims    = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0.1:
        s = 0.001
    elif max_dim > 0 and max_dim < 0.001:
        s = 0.010 / max_dim   # very small — scale up to ~10mm
    else:
        s = 1.0
    obj.scale = (s, s, s)
    bpy.context.scene.frame_set(0)

    # Find lowest vertex, place so bottom is at -depth
    mw     = obj.matrix_world
    verts_z = [(mw @ v.co).z for v in obj.data.vertices]
    min_z   = min(verts_z) if verts_z else 0.0
    depth   = DEPTH_MM * 0.001
    obj.location = (0, 0, -depth - min_z)
    bpy.context.scene.frame_set(0)

    obj.hide_render = True
    sw = bpy.data.objects['GelSurface'].modifiers['Shrinkwrap']
    sw.target      = obj
    sw.show_render = True

else:
    # Flat gel fallback
    ind = bpy.data.objects['IndenterSurface']
    ind.location    = (0, 0, -1)
    ind.hide_render = True
    sw = bpy.data.objects['GelSurface'].modifiers['Shrinkwrap']
    sw.target      = ind
    sw.show_render = False

# ── Render ────────────────────────────────────────────────────
bpy.context.scene.render.filepath = OUT_PATH
bpy.context.scene.frame_set(0)
bpy.ops.render.render(write_still=True)
print(f'[gs_ui_render] saved → {OUT_PATH}')
import os as _os; _os._exit(0)
