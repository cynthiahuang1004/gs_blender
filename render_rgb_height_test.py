"""
render_rgb_height_test.py
=========================
Render 5 test frames at different camera heights + angles to find
the best view that shows the sensor setup (gel, lights, object).

Camera is offset from center and tilted to show the environment.

Usage:
    GELSIGHT_RGB_PARAMS=bo_results/rgb/best_rgb_params.json \
    /home/shared/blender-4.2.0-linux-x64/blender -t 8 --background \
      gelsight_sampler.blend --python render_rgb_height_test.py
"""
import os, sys, json, math, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import bpy
from mathutils import Vector, Euler

OBJ_NAME = 'pattern_04_3_lines_angle_1'
SESSION = 'session_000'
SAMPLE_IDX = 100
OUT_DIR = os.path.join(ROOT, 'render_rgb_height_test')

# 5 test configurations: (cam_z, cam_x_offset, cam_y_offset, description)
# cam_z is negative (above gel), offsets shift camera to the side
CONFIGS = [
    (-0.04,  0.015, 0.015, "40mm_slight_offset"),
    (-0.06,  0.020, 0.020, "60mm_medium_offset"),
    (-0.08,  0.025, 0.025, "80mm_more_offset"),
    (-0.12,  0.035, 0.035, "120mm_wide"),
    (-0.18,  0.050, 0.050, "180mm_very_wide"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    import rerender_rgb_only as lib

    # Import mesh
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
    obj_bl = bpy.data.objects[blender_name]
    obj_bl.data.materials.clear()
    obj_bl.data.materials.append(mat)
    obj_bl.hide_render = False

    # Setup platform
    lib._setup_platform()

    # Place object
    sensor_dir = os.path.join(ROOT, 'renders_v3', OBJ_NAME, SESSION, 'sensor_0000')
    pose_path = os.path.join(sensor_dir, 'raw_data', f'{SAMPLE_IDX:04d}_pose.json')
    with open(pose_path) as f:
        pose = json.load(f)

    obj_bl.location = pose['location']
    obj_bl.rotation_euler = pose['rotation_euler']
    obj_bl.scale = pose['scale']
    bpy.context.view_layer.update()
    lib._set_platform_rotation(pose['rotation_euler'][2])

    # Make sensor parts visible
    for name in ['GelSurface', 'InterfaceSurface', 'EpoxySurface',
                 'LightSurfaceBL', 'LightSurfaceTR', 'LightSurfaceTL', 'LightSurfaceBR',
                 'LightSurfaceRGreen', 'LightSurfaceLGreen']:
        o = bpy.data.objects.get(name)
        if o:
            o.hide_render = False
            o.hide_viewport = False

    # Camera setup
    cam = bpy.data.objects['Camera']
    cam_data = cam.data
    cam_data.dof.use_dof = False

    # World
    world = bpy.data.worlds.get('World')
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1.0)
            bg_node.inputs['Strength'].default_value = lib.RGB_WORLD_STR

    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'

    # Widen FOV for setup view
    cam_data.angle = math.radians(65)

    target = Vector((0, 0, 0))  # gel center

    for i, (cz, cx, cy, desc) in enumerate(CONFIGS):
        cam_pos = Vector((cx, cy, cz))
        cam.location = cam_pos

        # Point camera at gel center
        direction = target - cam_pos
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

        bpy.context.view_layer.update()
        scene.frame_set(0)

        fp = os.path.join(OUT_DIR, f'{desc}')
        scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)

        print(f'  [{i+1}/5] {desc}: cam=({cx*1000:.0f}, {cy*1000:.0f}, {cz*1000:.0f})mm  '
              f'{time.time()-t0:.0f}s', flush=True)

    print(f'\nDone: 5 test frames in {time.time()-t0:.0f}s -> {OUT_DIR}', flush=True)

    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        sys.exit(0)


main()
