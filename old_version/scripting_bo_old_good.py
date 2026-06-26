"""
scripting_bo.py
===============
Minimal Blender script for Bayesian optimization background renders.
Reads GELSIGHT_FIXED_PARAMS and GELSIGHT_BG_RENDER env vars,
applies sensor parameters, renders one flat-gel background image, then exits.

Called by bo_optimize.py via:
    blender --background gelsight_sampler.blend --python scripting_bo.py
"""

import bpy, os, sys, json
from math import pi, tan
from mathutils import Euler

# ── Read env vars ─────────────────────────────────────────────────────────────
PARAMS_PATH = os.environ.get('GELSIGHT_FIXED_PARAMS', '')
RENDER_PATH = os.environ.get('GELSIGHT_BG_RENDER', '')

if not PARAMS_PATH or not RENDER_PATH:
    print('ERROR: GELSIGHT_FIXED_PARAMS and GELSIGHT_BG_RENDER must both be set.')
    os._exit(1)

with open(PARAMS_PATH) as f:
    P = json.load(f)

# ── Helpers (same logic as scripting.py, no randomization) ───────────────────

def set_emittor(name, strength, color):
    node = bpy.data.materials[name].node_tree.nodes['Emission']
    node.inputs['Color'].default_value = color
    node.inputs['Strength'].default_value = strength


def set_smoothness(val):
    gs = bpy.data.objects['GelSurface']
    gs.modifiers['CorrectiveSmooth'].iterations = val
    sw = gs.modifiers['Shrinkwrap']
    sw.offset            = 1e-3 * (val * val) / 50000
    sw.wrap_method       = 'PROJECT'
    sw.use_project_z     = True
    sw.use_negative_direction = True
    sw.use_positive_direction = False


def set_lights_transform(scale_y, light_z, rot_z, init_mats):
    """
    Apply light array rotation + Z height + emitter Y-scale.
    Order: matrix_world first (rotation), then override location[2] and scale[1],
    so the overrides are not clobbered by the matrix decomposition.
    """
    rot_mat = Euler((0, 0, rot_z)).to_matrix().to_4x4()
    for name in ['LightSurfaceTop', 'LightSurfaceBottom',
                 'LightSurfaceLeft', 'LightSurfaceRight']:
        bpy.data.objects[name].matrix_world = rot_mat @ init_mats[name]
        bpy.data.objects[name].location[2]  = light_z
        bpy.data.objects[name].scale[1]     = scale_y


def set_cam(fov, length):
    height = length / tan((fov / 360) * pi)
    bpy.data.objects['Camera'].location[2]  = -height
    bpy.data.objects['Camera'].data.angle   = (fov / 180) * pi
    tree = bpy.data.scenes['Scene'].node_tree
    tree.nodes['Map Range'].inputs[2].default_value = height
    tree.nodes['Map Range'].inputs[1].default_value = height - 0.002


def set_green_lights(strength):
    for obj_name in ['LightSurfaceLeftGreen', 'LightSurfaceRightGreen']:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].hide_render = False
    for mat_name in ['LeftGreenEmittor.001', 'RightGreenEmittor.001']:
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
            mat.node_tree.nodes['Emission'].inputs['Strength'].default_value = strength


def set_gel_material(roughness, fac):
    mat   = bpy.data.materials['aluminum-specular-mat']
    nodes = mat.node_tree.nodes
    glossy = nodes.get('Glossy BSDF')
    if glossy:
        glossy.inputs['Roughness'].default_value = roughness
    mix = nodes.get('Mix Shader')
    if mix:
        mix.inputs['Fac'].default_value = fac


def place_indenter_away():
    """Give a flat gel: hide IndenterSurface and disable shrinkwrap in render."""
    obj = bpy.data.objects['IndenterSurface']
    obj.location       = (0.0, 0.0, -1.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.hide_render    = True
    bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].target    = obj
    bpy.data.objects['GelSurface'].modifiers['Shrinkwrap'].show_render = False


# ── Apply parameters ──────────────────────────────────────────────────────────

# Save light initial matrices before any transform (used by set_lights_transform)
light_names = ['LightSurfaceTop', 'LightSurfaceBottom',
               'LightSurfaceLeft', 'LightSurfaceRight']
_init_matrices = {n: bpy.data.objects[n].matrix_world.copy() for n in light_names}

set_smoothness(int(P.get('smoothness', 50)))
set_lights_transform(
    scale_y = float(P.get('scale',        0.5)),
    light_z = float(P.get('light_z',     -0.005)),
    rot_z   = float(P.get('light_rot_z',  0.0)),
    init_mats = _init_matrices,
)
set_cam(float(P.get('fov', 40.0)), float(P.get('length', 0.01)))
set_green_lights(float(P.get('green_str', 50.0)))
set_gel_material(float(P.get('gel_roughness', 0.5)),
                 float(P.get('gel_fac', 0.25)))

# Set light colours / strengths
top_str  = float(P.get('top_str',  75.0))
top_r    = float(P.get('top_r',    0.95))
top_b    = float(P.get('top_b',    0.95))
left_str = float(P.get('left_str', 75.0))
left_r   = float(P.get('left_r',   0.95))
left_g   = float(P.get('left_g',   0.02))
left_b   = float(P.get('left_b',   0.15))

set_emittor('TopEmittor',    top_str,  (top_r, 0.0,   top_b,  1.0))
set_emittor('BottomEmittor', top_str,  (top_r, 0.0,   top_b,  1.0))
set_emittor('LeftEmittor',   left_str, (left_r, left_g, left_b, 1.0))
set_emittor('RightEmittor',  left_str, (left_r, left_g, left_b, 1.0))

# Flat gel (no object contact)
place_indenter_away()

# ── Render ────────────────────────────────────────────────────────────────────

bpy.context.scene.render.filepath = RENDER_PATH
bpy.context.scene.frame_set(0)
bpy.ops.render.render(write_still=True)
print(f'[scripting_bo] saved: {RENDER_PATH}.png')

# Force exit — os._exit bypasses Python exception handlers
os._exit(0)
