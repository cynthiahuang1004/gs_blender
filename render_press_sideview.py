"""
render_press_sideview.py  (run inside Blender with gelsight_sampler.blend)
=========================================================================
Side-view press trajectory: observer camera at ~50mm height watches
the object approach, press into gel, and retract. Uses the RGB render
pipeline (same as render_rgb_height_test) which correctly shows the scene.

Usage:
    GELSIGHT_RGB_PARAMS=bo_results/rgb/best_rgb_params.json \
    /home/shared/blender-4.2.0-linux-x64/blender -t 8 --background \
      gelsight_sampler.blend --python render_press_sideview.py
"""
import os, sys, json, math, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import bpy
from mathutils import Vector

OBJ_NAME = 'pattern_04_3_lines_angle_1'
SESSION = 'session_000'
SAMPLE_IDX = 100
OUT_DIR = os.path.join(ROOT, 'render_press_sideview')

CAM_X = 0.015
CAM_Y = 0.015
CAM_Z = -0.050
CAM_FOV = 65

APPROACH_START = -0.008
PRESS_MAX = 0.0012


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    import rerender_rgb_only as lib

    # ── Import mesh ──
    mesh_path = os.path.join(ROOT, 'meshes', f'{OBJ_NAME}.obj')
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=mesh_path, directory=os.path.dirname(mesh_path),
                          files=[{'name': os.path.basename(mesh_path)}])
    blender_name = [o.name for o in bpy.data.objects if o.name not in before][0]

    color = lib.OBJ_BLACK_COLOR if OBJ_NAME in lib.BLACK_OBJS else lib.OBJ_BLUE_COLOR
    mat = bpy.data.materials.new(f'solid_{blender_name}')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = lib._rgb_bo.get('obj_roughness', 0.25)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    obj = bpy.data.objects[blender_name]
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # ── Setup platform ──
    lib._setup_platform()

    # ── Load pose ──
    pose_path = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION,
                             'sensor_0000', 'raw_data', f'{SAMPLE_IDX:04d}_pose.json')
    with open(pose_path) as f:
        pose = json.load(f)

    session_path = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION, 'session.json')
    with open(session_path) as f:
        sess = json.load(f)
    z_anchor = sess.get('z_anchor', 0.0)

    obj.rotation_euler = pose['rotation_euler']
    obj.scale = pose['scale']
    base_x, base_y = pose['location'][0], pose['location'][1]
    lib._set_platform_rotation(pose['rotation_euler'][2])

    # ── Make sensor parts visible ──
    for name in ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
                 'LightSurfaceBL', 'LightSurfaceTR', 'LightSurfaceTL', 'LightSurfaceBR',
                 'LightSurfaceRGreen', 'LightSurfaceLGreen']:
        o = bpy.data.objects.get(name)
        if o:
            o.hide_render = False
            o.hide_viewport = False

    # ── Camera setup ──
    cam = bpy.data.objects['Camera']
    cam_data = cam.data
    cam_data.dof.use_dof = False
    cam_data.angle = math.radians(CAM_FOV)

    cam.location = (CAM_X, CAM_Y, CAM_Z)
    target = Vector((0, 0, 0))
    direction = target - Vector((CAM_X, CAM_Y, CAM_Z))
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # ── World background ──
    world = bpy.data.worlds.get('World')
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1.0)
            bg_node.inputs['Strength'].default_value = lib.RGB_WORLD_STR

    # ── Scene settings ──
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    # Hide tactile-only objects
    for name in lib.RGB_HIDE_NAMES:
        o = bpy.data.objects.get(name)
        if o:
            o.hide_render = True

    # Show sensor structure parts
    for name in ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
                 'LightSurfaceBL', 'LightSurfaceTR', 'LightSurfaceTL', 'LightSurfaceBR',
                 'LightSurfaceRGreen', 'LightSurfaceLGreen']:
        o = bpy.data.objects.get(name)
        if o:
            o.hide_render = False

    # Show object and platform
    obj.hide_render = False
    for po in lib._platform_objs_global:
        po.hide_render = False

    # ── Build trajectory ──
    depths = np.concatenate([
        np.linspace(APPROACH_START, 0.0, 10),
        np.linspace(0.0, PRESS_MAX, 15),
        np.full(5, PRESS_MAX),
        np.linspace(PRESS_MAX, 0.0, 10),
        np.linspace(0.0, APPROACH_START, 5),
    ])
    N = len(depths)

    print(f'Rendering {N} side-view press frames...', flush=True)
    paths = []

    for i, dz in enumerate(depths):
        if dz >= 0:
            obj.location = (base_x, base_y, -dz - z_anchor)
        else:
            obj.location = (base_x, base_y, dz - z_anchor)

        # Update shrinkwrap if gel exists
        gel = bpy.data.objects.get('GelSurface')
        if gel and gel.modifiers.get("Shrinkwrap"):
            gel.modifiers["Shrinkwrap"].target = obj

        bpy.context.view_layer.update()
        scene.frame_set(0)

        fp = os.path.join(OUT_DIR, f'frame_{i:03d}')
        scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        paths.append(fp + '.png')

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f'  [{i+1}/{N}] depth={dz*1000:.2f}mm  {time.time()-t0:.0f}s', flush=True)

    print(f'\nAll {N} frames rendered in {time.time()-t0:.0f}s -> {OUT_DIR}', flush=True)

    # ── Assemble GIF (system python, not Blender) ──
    json.dump({'paths': paths}, open(os.path.join(OUT_DIR, 'frame_paths.json'), 'w'))
    print('Assemble GIF with system python after this.')

    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        sys.exit(0)


main()
